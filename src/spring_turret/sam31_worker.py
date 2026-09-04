#!/usr/bin/env python3
"""SAM 3.1 text-grounding worker for Intel XPU.

The HTTP and hardware process deliberately keeps the multi-gigabyte model in a
separate process.  Requests and responses are newline-delimited JSON on stdin
and stdout; model diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import gc
import io
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

# This module is also executed as a standalone script in the inference venv.
if __package__:
    from .prompts import COLORS, normalize_prompts
else:
    from prompts import COLORS, normalize_prompts

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


class Sam31Engine:
    def __init__(
        self,
        checkpoint: Path,
        device_index: int,
        precision: str,
        confidence: float,
        use_sycl_graph: bool,
    ) -> None:
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
        self.confidence = confidence
        self.use_sycl_graph = use_sycl_graph
        self.graph_active = False
        self.graph_error: str | None = None
        self.compiled: Any | None = None
        self.inference_callable: Any | None = None

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
        self.wrapper = wrapper_type(model).to(self.device).eval()
        self.compiled = torch.compile(
            self.wrapper,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True},
        )
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(1008, 1008)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _inputs(self, jpeg: bytes, prompt: str) -> tuple[Any, Any]:
        image = self.Image.open(io.BytesIO(jpeg)).convert("RGB")
        pixels = self.transform(image).unsqueeze(0).to(self.device)
        tokens = self.tokenizer([prompt], context_length=32).to(self.device)
        return pixels, tokens

    def _activate_sycl_graph(self, pixels: Any, tokens: Any) -> None:
        if self.inference_callable is not None:
            return
        torch = self.torch
        print("Validating eager detector", file=sys.stderr, flush=True)
        _emit({"type": "progress", "stage": "validating"})
        reference = tuple(value.clone() for value in self.wrapper(pixels, tokens))
        print("Compiling detector", file=sys.stderr, flush=True)
        _emit({"type": "progress", "stage": "compiling"})
        # Warm the exact inference tensors and stream used during capture.
        # Cloning only after warmup changes Dynamo's inference-tensor guards
        # and would trigger a forbidden compilation inside graph capture.
        self.static_pixels = pixels.clone()
        self.static_tokens = tokens.clone()
        self.capture_stream = torch.xpu.Stream()
        torch.xpu.synchronize()
        with torch.xpu.stream(self.capture_stream):
            for _ in range(2):
                compiled_output = self.compiled(self.static_pixels, self.static_tokens)
        torch.xpu.synchronize()
        self.validation = validate_detections(
            torch, reference, compiled_output, self.confidence
        )
        print(
            f"Eager/compiled detection parity: {self.validation}",
            file=sys.stderr,
            flush=True,
        )
        del reference
        if not self.use_sycl_graph:
            self.inference_callable = self.compiled
            return
        print("Capturing SYCL graph", file=sys.stderr, flush=True)
        _emit({"type": "progress", "stage": "capturing"})
        self.graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(self.graph, stream=self.capture_stream):
            self.graph_output = self.compiled(self.static_pixels, self.static_tokens)
        self.graph.replay()
        torch.xpu.synchronize()
        for expected, actual in zip(compiled_output, self.graph_output, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0.001, atol=0.001)
        self.graph_active = True

        def replay(new_pixels: Any, new_tokens: Any) -> Any:
            self.static_pixels.copy_(new_pixels)
            self.static_tokens.copy_(new_tokens)
            self.graph.replay()
            return self.graph_output

        self.inference_callable = replay

    def detect(self, jpeg: bytes, prompt: str) -> dict[str, Any]:
        return self.detect_many(jpeg, [prompt])

    def detect_many(self, jpeg: bytes, prompts: list[str]) -> dict[str, Any]:
        prompts = normalize_prompts(prompts)
        if not prompts:
            raise ValueError("at least one nonempty prompt is required")
        torch = self.torch
        # One image, one compiled shape, sequential independent text queries.
        # Copy each result to CPU before graph replay overwrites its outputs.
        pixels, tokens = self._inputs(jpeg, prompts[0])
        detections = []
        categories = []
        with torch.inference_mode(), torch.autocast("xpu", dtype=self.dtype):
            self._activate_sycl_graph(pixels, tokens)
            started = time.perf_counter()
            for index, prompt in enumerate(prompts):
                if index:
                    tokens = self.tokenizer([prompt], context_length=32).to(self.device)
                boxes = self._detect_boxes(pixels, tokens)
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
            "sycl_graph": self.graph_active,
            "sycl_graph_error": self.graph_error,
            "validation": self.validation,
        }
        result["jpeg"] = base64.b64encode(annotate(jpeg, result["boxes"])).decode()
        return result

    def _detect_boxes(self, pixels: Any, tokens: Any) -> list[dict[str, Any]]:
        torch = self.torch
        logits, boxes, presence = self.inference_callable(pixels, tokens)
        probability = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).squeeze(-1)[
            0
        ]
        keep = probability > self.confidence
        probability, boxes = probability[keep], boxes[0][keep]
        center, size = boxes[..., :2], boxes[..., 2:]
        xyxy = torch.cat((center - size / 2, center + size / 2), -1).clamp(0, 1)
        return [
            {"xyxy": coordinates, "score": round(float(score), 4)}
            for coordinates, score in zip(
                xyxy.float().cpu().tolist(),
                probability.float().cpu().tolist(),
                strict=True,
            )
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
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not 0 < args.confidence < 1:
        raise SystemExit("confidence must be between zero and one")

    try:
        engine = Sam31Engine(
            checkpoint=args.checkpoint,
            device_index=args.device,
            precision=args.precision,
            confidence=args.confidence,
            use_sycl_graph=not args.no_sycl_graph,
        )
        _emit(
            {
                "type": "ready",
                "engine": "sam3.1/torch.compile/inductor-xpu",
                "torch_compile": False,
                "sycl_graph_requested": not args.no_sycl_graph,
            }
        )
        for line in sys.stdin:
            request: dict[str, Any] = {}
            try:
                request = json.loads(line)
                request_id = int(request["id"])
                prompts = normalize_prompts(request["prompts"])
                jpeg = base64.b64decode(request["jpeg"], validate=True)
                result = engine.detect_many(jpeg, prompts)
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


if __name__ == "__main__":
    main()
