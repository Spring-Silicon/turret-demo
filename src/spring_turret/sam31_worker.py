#!/usr/bin/env python3
"""SAM 3.1 text-grounding worker for Intel XPU.

The HTTP and hardware process deliberately keeps the multi-gigabyte model in a
separate process. Requests use a JSON header plus raw JPEG bytes (legacy base64
JSON is also accepted); responses are JSON lines and diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import gc
import io
import json
import os
import signal
import sys
import time
import types
from pathlib import Path
from typing import Any

# This module is also executed as a standalone script in the inference venv.
if __package__:
    from .prompts import COLORS, normalize_prompts
    from .sam31_graph import CompiledStage
    from .sam31_native import NativeImageStage
    from .sam31_w8a8 import W8A8ImageStage, PackedHeadStage, packed_profile, configure_source as configure_w8a8_source
    from .sam31_preprocess import ExactImagePreprocessor
    from .worker_protocol import JPEG_BYTES, iter_requests
    from .prefetch import LatestPreparation, RequestInbox
else:
    from prompts import COLORS, normalize_prompts
    from sam31_graph import CompiledStage
    from sam31_native import NativeImageStage
    from sam31_w8a8 import W8A8ImageStage, PackedHeadStage, packed_profile, configure_source as configure_w8a8_source
    from sam31_preprocess import ExactImagePreprocessor
    from worker_protocol import JPEG_BYTES, iter_requests
    from prefetch import LatestPreparation, RequestInbox

_PROTOCOL_OUTPUT: Any = None


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":")), file=_PROTOCOL_OUTPUT, flush=True)


def _text_only_geometry_encoder(torch: Any):
    class TextOnlyGeometryEncoder(torch.nn.Module):
        """Exact empty-geometry path without zero-sized ROI tensors."""

        def __init__(self, base: Any) -> None:
            super().__init__()
            self.base = base

        def forward(
            self,
            geo_prompt: Any,
            img_feats: list[Any],
            img_sizes: Any,
            img_pos_embeds: list[Any] | None = None,
        ) -> tuple[Any, Any]:
            del geo_prompt, img_sizes
            image = img_feats[-1]
            position = (
                img_pos_embeds[-1]
                if img_pos_embeds is not None
                else torch.zeros_like(image)
            )
            batch = image.shape[1]
            embeddings = self.base.cls_embed.weight.view(
                1, 1, self.base.d_model
            ).repeat(1, batch, 1)
            mask = torch.zeros(batch, 1, dtype=torch.bool, device=embeddings.device)
            if self.base.final_proj is not None:
                embeddings = self.base.norm(self.base.final_proj(embeddings))
            if self.base.encode is not None:
                for layer in self.base.encode:
                    embeddings = layer(
                        tgt=embeddings,
                        memory=image,
                        tgt_key_padding_mask=mask,
                        pos=position,
                    )
                embeddings = self.base.encode_norm(embeddings)
            return embeddings, mask

    return TextOnlyGeometryEncoder


def _grounding_wrapper(torch: Any, FindStage: Any, Prompt: Any):
    class GroundingWrapper(torch.nn.Module):
        """Static tensor boundary for text-prompt box detection."""

        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model
            self.register_buffer("img_ids", torch.tensor([0], dtype=torch.long))
            self.register_buffer("text_ids", torch.tensor([0], dtype=torch.long))

        def forward(self, pixels: Any, token_ids: Any) -> tuple[Any, Any, Any]:
            backbone = self.model.backbone.forward_image(
                pixels, need_interactive_out=False, need_propagation_out=False
            )
            # Image grounding consumes tensors; the 3.1 tri-neck wraps them.
            backbone["backbone_fpn"] = [
                feature.tensors for feature in backbone["backbone_fpn"]
            ]
            language = self.model.backbone.language_backbone
            text_mask = (token_ids != 0).bool().ne(1)
            _, text_memory = language.encoder(token_ids)
            text_memory = language.resizer(text_memory.transpose(0, 1))
            backbone["language_features"] = text_memory
            backbone["language_mask"] = text_mask

            find = FindStage(
                img_ids=self.img_ids,
                text_ids=self.text_ids,
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            geometry = Prompt(
                box_embeddings=torch.zeros(
                    (0, 1, 4), device=pixels.device, dtype=pixels.dtype
                ),
                box_mask=torch.zeros((1, 0), device=pixels.device, dtype=torch.bool),
            )
            output = self.model.forward_grounding(
                backbone_out=backbone,
                find_input=find,
                geometric_prompt=geometry,
                find_target=None,
            )
            return (
                output["pred_logits"],
                output["pred_boxes"],
                output["presence_logit_dec"],
            )

    return GroundingWrapper


def _shared_wrappers(torch: Any, FindStage: Any, Prompt: Any):
    class ImageEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.backbone = model.backbone

        def forward(self, pixels):
            backbone = self.backbone.forward_image(
                pixels, need_interactive_out=False, need_propagation_out=False
            )
            # This detector consumes exactly the final feature level; mask and
            # tracker branches are disabled. Keep the unfused image features.
            return (
                backbone["backbone_fpn"][-1].tensors,
                backbone["vision_pos_enc"][-1],
            )

    class TextEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.language = model.backbone.language_backbone

        def forward(self, token_ids):
            _, memory = self.language.encoder(token_ids)
            return (
                self.language.resizer(memory.transpose(0, 1)),
                token_ids.eq(0),
            )

    class GroundingHead(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, features, position, text_memory, text_mask):
            batch = text_mask.shape[0]
            backbone = {
                "vision_features": features,
                "backbone_fpn": [features],
                "vision_pos_enc": [position],
                "language_features": text_memory,
                "language_mask": text_mask,
            }
            # Every independent text prompt refers to the same encoded image.
            find = FindStage(
                img_ids=torch.zeros(batch, dtype=torch.long, device=features.device),
                text_ids=torch.arange(batch, device=features.device),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            geometry = Prompt(
                box_embeddings=features.new_zeros((0, batch, 4)),
                box_mask=torch.zeros(
                    (batch, 0), device=features.device, dtype=torch.bool
                ),
            )
            output = self.model.forward_grounding(
                backbone_out=backbone,
                find_input=find,
                geometric_prompt=geometry,
                find_target=None,
            )
            return (
                output["pred_logits"],
                output["pred_boxes"],
                output["presence_logit_dec"],
            )

    return ImageEncoder, TextEncoder, GroundingHead


def _enable_real_rope(torch: Any, model: Any) -> None:
    for module in model.modules():
        frequencies = getattr(module, "freqs_cis", None)
        if not torch.is_tensor(frequencies) or not torch.is_complex(frequencies):
            continue
        module.use_rope_real = True
        real = frequencies.real.contiguous()
        imaginary = frequencies.imag.contiguous()
        if "freqs_cis_real" in module._buffers:
            module.freqs_cis_real = real
            module.freqs_cis_imag = imaginary
        else:
            module.register_buffer("freqs_cis_real", real, persistent=False)
            module.register_buffer("freqs_cis_imag", imaginary, persistent=False)


@contextlib.contextmanager
def _redirect_cuda_construction(torch: Any):
    """Build Meta's CUDA-default modules on CPU before moving them to XPU."""
    original_factories: dict[str, Any] = {}
    for name in (
        "arange",
        "as_tensor",
        "empty",
        "eye",
        "full",
        "linspace",
        "ones",
        "rand",
        "randn",
        "tensor",
        "zeros",
    ):
        original = getattr(torch, name)
        original_factories[name] = original

        def factory(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            requested = kwargs.get("device")
            if requested is not None and str(requested).split(":", 1)[0] == "cuda":
                kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        setattr(torch, name, factory)
    original_properties = torch.cuda.get_device_properties
    torch.cuda.get_device_properties = lambda index=0: types.SimpleNamespace(major=0)
    try:
        yield
    finally:
        for name, original in original_factories.items():
            setattr(torch, name, original)
        torch.cuda.get_device_properties = original_properties


def _move_plain_tensors(torch: Any, model: Any, device: Any) -> None:
    """Move upstream tensor caches which are not parameters or buffers."""
    moved_values: dict[int, tuple[Any, Any]] = {}

    def move(value: Any) -> Any:
        if id(value) in moved_values:
            return moved_values[id(value)][1]
        if torch.is_tensor(value):
            # Keep the source alive too so Python cannot reuse its id.
            moved_values[id(value)] = (value, value.to(device=device))
            return moved_values[id(value)][1]
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for module in model.modules():
        for name, value in list(vars(module).items()):
            if name.startswith("_"):
                continue
            moved = move(value)
            if moved is not value:
                setattr(module, name, moved)


def _build_detector(torch: Any, checkpoint: Path) -> Any:
    """Build the exact 3.1 detector, strictly loading every detector weight.

    SAM 3's image builder has a different neck. Only the unused tracker and
    pixel-mask head are omitted; text grounding and box scoring are unchanged.
    """
    import sam3
    from sam3 import model_builder as builder
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    bpe = str(Path(sam3.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz")
    model = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=SAM3VLBackboneTri(
            scalp=0,
            visual=builder._create_multiplex_tri_backbone(
                compile_mode=None, use_fa3=False, use_rope_real=False
            ),
            text=builder._create_text_encoder(bpe),
        ),
        transformer=builder._create_sam3_transformer(use_fa3=False),
        segmentation_head=builder._create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None,
        input_geometry_encoder=builder._create_geometry_encoder(),
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=builder._create_dot_product_scoring(),
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    weights = weights.get("model", weights)
    detector_weights = {
        key.removeprefix("detector."): value
        for key, value in weights.items()
        if key.startswith("detector.")
    }
    model.load_state_dict(detector_weights, strict=True, assign=True)
    model.segmentation_head = None
    model.eval().requires_grad_(False)
    return model


def validate_detections(
    torch: Any, reference: Any, actual: Any, confidence: float
) -> dict[str, Any]:
    """Bound inference drift on meaningful detector outputs, not rejected boxes.

    SAM's 200 queries include unconstrained no-object boxes/logits. Comparing
    those boxes is not a detection accuracy test. Bound confidence near the
    decision threshold and XYXY for every box retained by either path. Large
    changes that add/remove a confident detection therefore always fail.
    """

    def unpack(output: Any) -> tuple[Any, Any]:
        logits, boxes, presence = output
        scores = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).squeeze(-1)
        center, size = boxes[..., :2], boxes[..., 2:]
        return scores.float(), torch.cat(
            (center - size / 2, center + size / 2), -1
        ).clamp(0, 1).float()

    expected_scores, expected_boxes = unpack(reference)
    actual_scores, actual_boxes = unpack(actual)
    for value in (*reference, *actual):
        if not torch.isfinite(value).all().item():
            raise RuntimeError("SAM returned non-finite values")
    candidates = (expected_scores > confidence - 0.03) | (
        actual_scores > confidence - 0.03
    )
    torch.testing.assert_close(
        actual_scores[candidates], expected_scores[candidates], rtol=0, atol=0.03
    )
    selected = (expected_scores > confidence) | (actual_scores > confidence)
    torch.testing.assert_close(
        actual_boxes[selected], expected_boxes[selected], rtol=0, atol=0.01
    )
    return {
        "max_confidence_error": round(
            (actual_scores[candidates] - expected_scores[candidates])
            .abs()
            .max()
            .item(),
            6,
        )
        if candidates.any().item()
        else 0,
        "max_all_query_score_error": round(
            (actual_scores - expected_scores).abs().max().item(), 6
        ),
        "compared_boxes": selected.sum().item(),
        "max_box_coordinate_error": round(
            (actual_boxes[selected] - expected_boxes[selected]).abs().max().item(), 6
        )
        if selected.any().item()
        else 0,
    }


def check_detection_candidate(torch, reference, actual, confidence, *, report_only=False):
    """Report numerical failures only with explicit W8A8 development opt-in.

    The reference and tolerances are unchanged. Shape/dtype errors, NaNs,
    native faults and graph qualification failures are never waived.
    """
    if report_only and (len(reference) != len(actual) or any(
            a.shape != b.shape or a.dtype != b.dtype for a, b in zip(reference, actual))):
        raise RuntimeError("SAM detection output shape/dtype changed")
    try:
        return {"passed": True, **validate_detections(torch, reference, actual, confidence)}
    except AssertionError as error:
        if not report_only:
            raise
        return {"passed": False, "policy": "report-only-development", "error": str(error)}


class Sam31Engine:
    def __init__(
        self,
        checkpoint: Path,
        device_index: int,
        precision: str,
        confidence: float,
        use_sycl_graph: bool,
        grounding_batch_size: int = 1,
        native_bundle: Path | None = None,
        w8a8_development_bundle: Path | None = None,
        allow_unqualified_w8a8: bool = False,
    ) -> None:
        if type(allow_unqualified_w8a8) is not bool or (allow_unqualified_w8a8 and w8a8_development_bundle is None):
            raise ValueError("Unqualified execution requires an explicit W8A8 development bundle")
        self.allow_unqualified_w8a8 = allow_unqualified_w8a8
        self.packed_bundle = (w8a8_development_bundle if w8a8_development_bundle is not None
                              and packed_profile(w8a8_development_bundle) else None)
        if self.packed_bundle is not None and grounding_batch_size != 1:
            raise ValueError("Retained packed heads require batch size 1; multiple prompts still share one image")
        if native_bundle is not None and w8a8_development_bundle is not None:
            raise ValueError("Select only one SAM image bundle")
        if w8a8_development_bundle is not None:
            if precision != "float16":
                raise ValueError("W8A8 requires FP16 autocast with FP32 masters")
            configure_w8a8_source(w8a8_development_bundle, checkpoint)
        import torch
        from PIL import Image
        from sam3.model.data_misc import FindStage
        from sam3.model.geometry_encoders import Prompt
        from torchvision.transforms import v2

        if not torch.xpu.is_available():
            raise RuntimeError("PyTorch XPU is unavailable")
        torch.xpu.set_device(device_index)
        torch.set_grad_enabled(False)
        self.torch = torch
        self.Image = Image
        self.device = torch.device(f"xpu:{device_index}")
        self.dtype = torch.float16 if precision == "float16" else torch.bfloat16
        if native_bundle is not None and precision != "float16":
            raise ValueError("native SAM requires float16 autocast with FP32 master weights")
        self.confidence = confidence
        if not use_sycl_graph:
            raise ValueError("shared-feature inference requires SYCL graphs")
        if grounding_batch_size not in (1, 2, 4, 8):
            raise ValueError("grounding batch size must be 1, 2, 4, or 8")
        self.grounding_batch_size = grounding_batch_size
        self.graph_active = False
        self.graph_error: str | None = None
        self.text_cache: dict[str, Any] = {}
        self.grounding_stages: dict[int, Any] = {}
        self.validated_batches: set[int] = set()
        self.validation: dict[str, Any] = {}

        with (
            contextlib.redirect_stdout(sys.stderr),
            _redirect_cuda_construction(torch),
        ):
            model = _build_detector(torch, checkpoint)
        model.geometry_encoder = _text_only_geometry_encoder(torch)(
            model.geometry_encoder
        )
        _enable_real_rope(torch, model)
        # Keep master weights and normalization/residual paths in FP32; AMP
        # selects half precision for matrix multiplies. model.half() changes
        # the detector numerics significantly across Inductor fusion boundaries.
        model._apply(lambda tensor: tensor.contiguous())
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        buffer_bytes = sum(
            buffer.numel() * buffer.element_size() for buffer in model.buffers()
        )
        print(
            f"SAM 3.1 model state: parameters={parameter_bytes} "
            f"buffers={buffer_bytes} dtype={self.dtype}",
            file=sys.stderr,
            flush=True,
        )
        try:
            gc.collect()
            model.to(device=self.device)
            model.eval()
        except Exception:
            print(
                "XPU move failed: "
                f"allocated={torch.xpu.memory_allocated(device_index)} "
                f"reserved={torch.xpu.memory_reserved(device_index)} "
                f"free_total={torch.xpu.mem_get_info(device_index)}",
                file=sys.stderr,
                flush=True,
            )
            raise
        _move_plain_tensors(torch, model, self.device)
        self.tokenizer = model.backbone.language_backbone.tokenizer
        wrapper_type = _grounding_wrapper(torch, FindStage, Prompt)
        # Independent full-pass reference used only during correctness checks.
        self.wrapper = wrapper_type(model).to(self.device).eval()
        image_type, text_type, head_type = _shared_wrappers(torch, FindStage, Prompt)
        self.image_stage = (
            W8A8ImageStage(torch, image_type(model).eval(), w8a8_development_bundle, self.device, self._progress)
            if w8a8_development_bundle is not None else
            NativeImageStage(torch, native_bundle, checkpoint, self.device, self._progress)
            if native_bundle is not None else
            CompiledStage(torch, image_type(model).eval(), "image", self._progress)
        )
        self.text_stage = CompiledStage(
            torch, text_type(model).eval(), "text", self._progress
        )
        self.head = head_type(model).eval()
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(1008, 1008)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.preprocessor = (None if getattr(self.image_stage, "accepts_cpu", False)
                             else ExactImagePreprocessor(torch, self.device, self._progress))

    def _inputs(self, jpeg: bytes, prompt: str) -> tuple[Any, Any]:
        return self._pixels(jpeg), self._tokens(prompt)

    def _pixels(self, jpeg: bytes) -> Any:
        if self.preprocessor is not None:
            return self.preprocessor(jpeg)
        image = self.Image.open(io.BytesIO(jpeg)).convert("RGB")
        pixels = self.transform(image).unsqueeze(0)
        return pixels if getattr(self.image_stage, "accepts_cpu", False) else pixels.to(self.device)

    def _tokens(self, prompt: str) -> Any:
        return self.tokenizer([prompt], context_length=32).to(self.device)

    @staticmethod
    def _progress(stage: str, component: str) -> None:
        print(f"{stage}: {component}", file=sys.stderr, flush=True)
        _emit({"type": "progress", "stage": stage, "component": component})

    def _encode_prompts(self, prompts: list[str]) -> list[Any]:
        # Keep only active categories. Clone graph outputs so encoding another
        # prompt cannot overwrite a cached embedding through static aliases.
        self.text_cache = {
            prompt: self.text_cache[prompt]
            for prompt in prompts
            if prompt in self.text_cache
        }
        for prompt in prompts:
            if prompt not in self.text_cache:
                self.text_cache[prompt] = tuple(
                    value.clone() for value in self.text_stage(self._tokens(prompt))
                )
        return [self.text_cache[prompt] for prompt in prompts]

    def _batch_sizes(self, count: int) -> set[int]:
        size = self.grounding_batch_size
        return {
            1 if min(size, count - start) == 1 else size
            for start in range(0, count, size)
        }

    def _run_heads(self, features: Any, embeddings: list[Any]) -> tuple[Any, ...]:
        torch = self.torch
        parts: list[Any] = []
        size = self.grounding_batch_size
        for start in range(0, len(embeddings), size):
            group = embeddings[start : start + size]
            count = len(group)
            batch = 1 if count == 1 else size
            group = group + [group[-1]] * (batch - count)
            memory = torch.cat([value[0] for value in group], dim=1)
            mask = torch.cat([value[1] for value in group], dim=0)
            if batch not in self.grounding_stages:
                packed = getattr(self, "packed_bundle", None)
                self.grounding_stages[batch] = (
                    PackedHeadStage(torch, self.head, f"packed-grounding-{batch}", self._progress, packed)
                    if packed is not None else
                    CompiledStage(torch, self.head, f"grounding-{batch}", self._progress)
                )
            outputs = self.grounding_stages[batch](*features, memory, mask)
            # Later chunks reuse the same static graph outputs. Own the useful
            # slices before replaying, and never publish padded categories.
            parts.append(tuple(value[:count].clone() for value in outputs))
        return tuple(torch.cat(values, dim=0) for values in zip(*parts, strict=True))

    def _prepare(self, pixels: Any, prompts: list[str]) -> list[Any]:
        embeddings = self._encode_prompts(prompts)
        batches = self._batch_sizes(len(prompts))
        if not batches <= self.validated_batches:
            actual = self._run_heads(self.image_stage(pixels), embeddings)
            self._progress("validating", "shared features versus full passes")
            for index, prompt in enumerate(prompts):
                expected = self.wrapper(pixels.to(self.device), self._tokens(prompt))
                self.validation[prompt] = check_detection_candidate(
                    self.torch,
                    expected,
                    tuple(value[index : index + 1] for value in actual),
                    self.confidence,
                    report_only=getattr(self, "allow_unqualified_w8a8", False),
                )
            self.validated_batches.update(batches)
        self.graph_active = True
        return embeddings

    def detect(self, jpeg: bytes, prompt: str) -> dict[str, Any]:
        return self.detect_many(jpeg, [prompt])

    def detect_many(self, jpeg: bytes, prompts: list[str], *, client_overlay=False,
                    prepared_pixels=None) -> dict[str, Any]:
        prompts = normalize_prompts(prompts)
        if not prompts:
            raise ValueError("at least one nonempty prompt is required")
        torch = self.torch
        worker_started = time.perf_counter()
        pixels = (self._pixels(jpeg) if prepared_pixels is None
                  else self.preprocessor(jpeg, prepared=prepared_pixels))
        torch.xpu.synchronize()
        preprocess_ms = (time.perf_counter() - worker_started) * 1000
        detections = []
        categories = []
        with torch.inference_mode(), torch.autocast("xpu", dtype=self.dtype):
            cached = all(prompt in self.text_cache for prompt in prompts)
            text_started = time.perf_counter()
            embeddings = self._prepare(pixels, prompts)
            torch.xpu.synchronize()
            prompt_setup_ms = (time.perf_counter() - text_started) * 1000
            started = time.perf_counter()
            features = self.image_stage(pixels)
            torch.xpu.synchronize()
            image_done = time.perf_counter()
            outputs = self._run_heads(features, embeddings)
            torch.xpu.synchronize()
            grounding_done = time.perf_counter()
            boxes_by_prompt = self._decode_outputs(outputs)
            for index, (prompt, boxes) in enumerate(
                zip(prompts, boxes_by_prompt, strict=True)
            ):
                color = COLORS[index]
                categories.append(
                    {"prompt": prompt, "color": color, "count": len(boxes)}
                )
                detections.extend(
                    {**box, "prompt": prompt, "prompt_index": index, "color": color}
                    for box in boxes
                )
        torch.xpu.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        result = {
            "boxes": detections,
            "categories": categories,
            "latency_ms": round(latency_ms),
            "torch": torch.__version__,
            "device": torch.xpu.get_device_name(self.device),
            "torch_compile": True,
            "image_backend": getattr(self.image_stage, "backend", "torch.compile"),
            "native_image_validation": getattr(self.image_stage, "proof", None),
            "accuracy_policy": "report-only-development" if getattr(self, "allow_unqualified_w8a8", False) else "enforced",
            "preprocess_validation": getattr(getattr(self, "preprocessor", None), "validation", None),
            "preprocess_prefetched": prepared_pixels is not None,
            "sycl_graph": self.graph_active,
            "sycl_graph_error": self.graph_error,
            "validation": self.validation,
            "shared_image_features": True,
            "text_cache_hit": cached,
            "grounding_batch_size": self.grounding_batch_size,
            "timing": {
                "preprocess_ms": round(preprocess_ms, 2),
                "cpu_prepare_ms": round(getattr(getattr(self, "preprocessor", None), "last_cpu_ms", 0), 2),
                "prompt_setup_ms": round(prompt_setup_ms, 2),
                "image_encoder_ms": round((image_done - started) * 1000, 2),
                "grounding_ms": round((grounding_done - image_done) * 1000, 2),
                "postprocess_ms": round(
                    latency_ms - (grounding_done - started) * 1000, 2
                ),
            },
        }
        annotation_started = time.perf_counter()
        native_ms = getattr(self.image_stage, "last_execution_ms", None)
        if native_ms is not None:
            result["timing"]["native_image_replay_ms"] = round(native_ms, 2)
        result["client_overlay"] = client_overlay
        if not client_overlay:
            result["jpeg"] = base64.b64encode(annotate(jpeg, result["boxes"])).decode()
        result["timing"]["annotation_ms"] = round(
            (time.perf_counter() - annotation_started) * 1000, 2
        )
        result["timing"]["worker_total_ms"] = round(
            (time.perf_counter() - worker_started) * 1000, 2
        )
        return result

    def _decode_outputs(self, outputs: Any) -> list[list[dict[str, Any]]]:
        torch = self.torch
        logits, boxes, presence = outputs
        probability = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).squeeze(-1)
        center, size = boxes[..., :2], boxes[..., 2:]
        xyxy = torch.cat((center - size / 2, center + size / 2), -1).clamp(0, 1)
        # Tiny fixed-size outputs: transfer once, avoiding variable-size GPU
        # nonzero/filter synchronizations for every category separately.
        coordinates = xyxy.float().cpu().tolist()
        scores = probability.float().cpu().tolist()
        return [
            [
                {"xyxy": box, "score": round(float(score), 4)}
                for box, score in zip(boxes, probabilities, strict=True)
                if score > self.confidence
            ]
            for boxes, probabilities in zip(coordinates, scores, strict=True)
        ]


def annotate(jpeg: bytes, boxes: list[dict[str, Any]]) -> bytes:
    """Draw on the exact input frame, never on a newer camera frame."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(14, image.width // 64))
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        rect = (
            round(x1 * image.width),
            round(y1 * image.height),
            round(x2 * image.width),
            round(y2 * image.height),
        )
        color = box["color"]
        draw.rectangle(rect, outline=color, width=3)
        label = f"{box['prompt'][:48]} {box['score']:.0%}"
        x, y = rect[0], max(0, rect[1] - 24)
        bounds = draw.textbbox((x + 4, y + 2), label, font=font)
        draw.rectangle((x, y, bounds[2] + 4, bounds[3] + 2), fill="#0b0d0d")
        draw.text((x + 4, y + 2), label, fill=color, font=font)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def main() -> None:
    # Native libraries can write directly to fd 1; keep those diagnostics out
    # of the JSON protocol as well as ordinary Python prints.
    global _PROTOCOL_OUTPUT
    _PROTOCOL_OUTPUT = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--precision", choices=("float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--no-sycl-graph", action="store_true")
    parser.add_argument("--native-bundle", type=Path)
    parser.add_argument("--w8a8-development-bundle", type=Path)
    parser.add_argument("--allow-unqualified-w8a8", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not 0 < args.confidence < 1:
        raise SystemExit("confidence must be between zero and one")

    engine = preparation = None
    # A normal model switch/stop must reap the resident native runner as well.
    def terminate(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    try:
        engine = Sam31Engine(
            checkpoint=args.checkpoint,
            device_index=args.device,
            precision=args.precision,
            confidence=args.confidence,
            use_sycl_graph=not args.no_sycl_graph,
            native_bundle=args.native_bundle,
            w8a8_development_bundle=args.w8a8_development_bundle,
            allow_unqualified_w8a8=args.allow_unqualified_w8a8,
        )
        _emit(
            {
                "type": "ready",
                "engine": "sam3.1/israel-w8a8-development" if args.w8a8_development_bundle else
                          "sam3.1/native-image+inductor-xpu" if args.native_bundle else "sam3.1/torch.compile/inductor-xpu",
                "torch_compile": False,
                "sycl_graph_requested": not args.no_sycl_graph,
                "request_transport": JPEG_BYTES,
                "cpu_prefetch": engine.preprocessor is not None,
            }
        )
        if engine.preprocessor is not None:
            preparation = LatestPreparation(engine.preprocessor.prepare_cpu)
            requests = RequestInbox(sys.stdin.buffer, preparation)
        else:
            requests = iter_requests(sys.stdin.buffer)
        for request in requests:
            try:
                request_id = int(request["id"])
                prompts = normalize_prompts(request["prompts"])
                jpeg = request["jpeg"]
                prepared = preparation.take(request.get("prepared_token"), jpeg) if preparation else None
                result = engine.detect_many(jpeg, prompts, client_overlay=request.get("client_overlay") is True,
                                            prepared_pixels=prepared)
                _emit({"type": "result", "id": request_id, **result})
            except Exception as error:
                _emit(
                    {
                        "type": "error",
                        "id": request.get("id"),
                        "error": str(error),
                    }
                )
    except Exception as error:
        _emit({"type": "fatal", "error": str(error)})
        raise
    finally:
        if preparation is not None:
            preparation.close()
        if engine is not None and isinstance(engine.image_stage, (NativeImageStage, W8A8ImageStage)):
            engine.image_stage.close()


if __name__ == "__main__":
    main()
