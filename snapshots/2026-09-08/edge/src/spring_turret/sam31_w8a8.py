"""Pinned Israel W8A8 development kernels; dense validation stays independent.

This candidate is explicitly NOT accuracy-qualified. Native QKV + queued MLP
run inside the same Torch XPU graph, without the dense runner's host IPC copies.
"""
from __future__ import annotations

import ast
import copy
import importlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import MethodType, ModuleType

if __package__:
    from .sam31_graph import CompiledStage
    from .sam31_native import CHECKPOINT, digest
else:
    from sam31_graph import CompiledStage
    from sam31_native import CHECKPOINT, digest

SOURCE_DIR = "results/turret_megakernel/native_qkv_sources"
RESULTS = "results/turret_megakernel"
PINNED_FILES = json.loads(Path(__file__).with_name("w8a8_manifest.json").read_text())
PACKED_FILES = json.loads(Path(__file__).with_name("w8a8_packed_manifest.json").read_text())
GRAPHICS_FILES = json.loads(Path(__file__).with_name("w8a8_graphics_manifest.json").read_text())
PACKED_CONFIG = f"{RESULTS}/engine_only_retained_packed_config.json"
CAST_CACHE_CONFIG = f"{RESULTS}/engine_only_retained_cast_cached_config.json"
CAST_CACHE_FILES = json.loads(Path(__file__).with_name("w8a8_cast_cache_manifest.json").read_text())
PACKED_SOURCE = f"{RESULTS}/packed_head_rounded_sources"
MLP_SOURCE = f"{RESULTS}/mlp_cached_recip_vec16_sources"
REPORT = "results/turret_megakernel/demo_native_qkv_grouped_gpu1.json"
HELPERS = {"persistent_forward", "mega_forward", "attention_projection_residual",
           "attention_residual_block_forward"}


def packed_profile(bundle: Path) -> bool:
    return (bundle / PACKED_CONFIG).exists()


def cast_cache_profile(bundle: Path) -> bool:
    return (bundle / CAST_CACHE_CONFIG).exists()


def verify_graphics(bundle: Path) -> Path:
    directory = bundle / "runtime/graphics"
    for name, expected in GRAPHICS_FILES.items():
        if digest(directory / name) != expected:
            raise ValueError(f"W8A8 graphics runtime checksum mismatch: {name}")
    return directory


def verify_bundle(bundle: Path, checkpoint: Path) -> None:
    if digest(checkpoint) != CHECKPOINT:
        raise ValueError("W8A8 requires the pinned SAM 3.1 checkpoint")
    files = {**PINNED_FILES, **(PACKED_FILES if packed_profile(bundle) else {})}
    if cast_cache_profile(bundle):
        if not packed_profile(bundle):
            raise ValueError("Cast-cached heads require the retained packed bundle")
        files.update(CAST_CACHE_FILES)
    for name, expected in files.items():
        if digest(bundle / name) != expected:
            raise ValueError(f"W8A8 bundle checksum mismatch: {name}")
    if packed_profile(bundle):
        verify_graphics(bundle)


def configure_source(bundle: Path, checkpoint: Path) -> None:
    verify_bundle(bundle, checkpoint)
    existing = sys.modules.get("sam3")
    if existing and not Path(existing.__file__).resolve().is_relative_to(bundle.resolve()):
        raise ValueError("W8A8 must load its pinned SAM source before importing sam3")
    sys.path.insert(0, str(bundle))
    sys.path.insert(0, str(bundle / SOURCE_DIR))
    if packed_profile(bundle):
        # The retained MLP adapter imports unchanged base helpers from SOURCE_DIR.
        # Load corrected packed-mask sources, never the rejected first prototype.
        for directory in ("scripts/turret_megakernel", PACKED_SOURCE, MLP_SOURCE):
            sys.path.insert(0, str(bundle / directory))


class PackedHeadStage(CompiledStage):
    """Patch only during optimized tracing; eager dense references stay original."""

    def __init__(self, torch, module, name, progress, bundle, proof=None):
        self.library = bundle / "runtime/joint_head_attention_packed512.so"
        self.direct_replay_passed = False
        self.cast_cache_enabled = cast_cache_profile(bundle)
        self.cast_cache_proof = None
        self.image_proof = proof
        super().__init__(torch, module, name, progress)

    def _capture_cached(self, inputs, implementation):
        """Cache exact casts once; preserve the compiled control and FP32 masters.

        Model weights are immutable for a worker's entire lifetime. A model or
        device change creates a new worker/cache/graph, never an in-place update.
        The source helper fails closed on an unrecognized generated program.
        """
        from megakernel_design_head_cast_cache_v3 import prepare_cached_head
        torch = self.torch
        if any(value.device.type != "xpu" for value in inputs):
            raise ValueError("Retained cast cache supports only XPU")
        if self.compiled is None:
            self._compile(inputs)
        self.inputs = tuple(value.clone() for value in inputs)
        self.stream = torch.xpu.Stream()
        torch.xpu.synchronize()
        self.progress("compiling", self.name)
        with torch.xpu.stream(self.stream):
            for _ in range(2):
                control = self.compiled(*self.inputs)
            expected = tuple(value.clone() for value in control)
            cached = prepare_cached_head(self.compiled, self.inputs)
            direct = tuple(value.clone() for value in cached(*self.inputs))
        torch.xpu.synchronize()
        self._require_exact(expected, direct, "cached/retained")
        cached.validate_cache()
        if len(implementation["invocations"]) < 18:
            raise RuntimeError("Packed SAM heads did not capture all six decoder layers")
        self.progress("capturing", self.name)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=self.stream):
            outputs = cached(*self.inputs)
        graph.replay()
        torch.xpu.synchronize()
        self._require_exact(expected, outputs, "cached replay/retained")
        cached.validate_cache()
        # Publish only a fully validated graph; keep its caches and masters alive.
        self.cached_head, self.outputs, self.graph = cached, outputs, graph
        self.direct_replay_passed = True
        self.cast_cache_proof = {
            key: cached.report[key] for key in (
                "unique_casts", "cache_bytes", "removed_cast_launches",
                "rewritten_gemm_read_operands", "exact_structural_diff_passed",
                "original_source_sha256", "transformed_source_sha256")}
        self.cast_cache_proof.update(retained_direct_replay_exact=True,
                                     cache_signatures_validated=True)
        if self.image_proof is not None:
            self.image_proof["head_cast_cache"] = self.cast_cache_proof
        self.calls += 1
        return self.outputs

    def _require_exact(self, expected, actual, label):
        if not all(self.torch.equal(a, b) and bool(self.torch.isfinite(b).all())
                   for a, b in zip(expected, actual, strict=True)):
            raise RuntimeError(f"Packed SAM heads {label} outputs differ")

    def __call__(self, *inputs):
        if self.graph is not None:
            return super().__call__(*inputs)
        from packed_head_attention import packed_heads
        with packed_heads(self.library) as implementation:
            if self.cast_cache_enabled:
                return self._capture_cached(inputs, implementation)
            result = super().__call__(*inputs)
            replay = tuple(value.clone() for value in result)
            direct = self.compiled(*self.inputs)
            if not all(self.torch.equal(a, b) and bool(self.torch.isfinite(b).all())
                       for a, b in zip(replay, direct, strict=True)):
                raise RuntimeError("Packed SAM heads direct/replay outputs differ")
            if len(implementation["invocations"]) < 18:
                raise RuntimeError("Packed SAM heads did not capture all six decoder layers")
            self.direct_replay_passed = True
            return result


def helper_definitions(source: str) -> ast.Module:
    """Reuse exact audited forwards without importing the benchmark CLI/setup.

    The complete source is hash-pinned before this runs. Selecting only these
    function definitions excludes its main(), sys.path edits and eager imports
    of the obsolete /home/spring/turret-demo checkout.
    """
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    if {node.name for node in nodes} != HELPERS:
        raise ValueError("W8A8 snapshot is missing required image forwards")
    return ast.Module(body=nodes, type_ignores=[])


def _helpers(bundle: Path, torch):
    path = bundle / SOURCE_DIR / "demo_candidate.py"
    module = ModuleType("spring_turret_w8a8_forwards")
    module.__file__ = str(path)
    persistent = importlib.import_module("persistent_int8")
    module.__dict__.update(
        torch=torch,
        quantize=importlib.import_module("sam3.optimization.smoothquant_kernels").quantize,
        quantized_norm=persistent.quantized_norm, matmul=persistent.matmul,
        qkv_rope=importlib.import_module("qkv_rope").qkv_rope,
        native_qkv_rope=importlib.import_module("native_qkv_rope").qkv_rope,
    )
    sys.modules[module.__name__] = module
    exec(compile(helper_definitions(path.read_text()), str(path), "exec"), module.__dict__)
    return module


class W8A8ImageStage(CompiledStage):
    backend = "israel-w8a8-development"

    def __init__(self, torch, image, bundle: Path, device, progress):
        if torch.__version__ != "2.14.0+xpu" or "B580" not in torch.xpu.get_device_name(device):
            raise ValueError("This W8A8 candidate requires Torch 2.14.0+xpu on B580")
        self.bundle = bundle
        self.packed = packed_profile(bundle)
        if self.packed:
            self.backend = "israel-w8a8-packed-development"
            # Parent must set LD_LIBRARY_PATH BEFORE importing Torch/loading
            # Level Zero. Changing it inside this process is too late.
            expected = (bundle / "runtime/graphics/libze_intel_gpu.so.1").resolve()
            mappings = Path("/proc/self/maps").read_text().splitlines()
            paths = {Path(line.split()[-1]).resolve() for line in mappings if "libze_intel_gpu.so" in line}
            if paths != {expected}:
                raise ValueError("Packed W8A8 requires its pinned process-local Intel graphics runtime")
        self.context = ExitStack()
        self.proof = {"source": "israel", "accuracy_qualified": False,
                      "source_confidence_failures": 5, "source_box_failures": 4,
                      "report_sha256": PINNED_FILES[REPORT], "direct_replay_passed": False}
        if self.packed:
            self.proof.update(retained_config_sha256=PACKED_FILES[PACKED_CONFIG],
                              mlp="cached-recip-vec16", heads="packed512-rounded",
                              graphics_runtime="israel-26.27-igc-2.38.5")
            if cast_cache_profile(bundle):
                self.backend = "israel-w8a8-packed-cast-cached-development"
                self.proof.update(retained_config_sha256=CAST_CACHE_FILES[CAST_CACHE_CONFIG],
                                  heads="packed512-rounded-cast-cached")
        progress("loading", "Israel W8A8 development image")
        # Separate module topology, shared immutable FP32 masters. Quantization
        # replaces modules/allocates packed buffers only on the image copy;
        # engine.wrapper remains the ORIGINAL dense eager reference.
        masters = list(image.parameters()) + list(image.buffers())
        image = copy.deepcopy(image, {id(value): value for value in masters})
        from sam3.optimization.detector_smoothquant import install_smoothquant, final_detector_level
        from sam3.model.model_misc import AttentionType
        from native_mlp_owned import Workspace
        from native_qkv_rope import prepare
        from persistent_int8 import make_gelu_lut
        from native_norm import native_norm_forward
        functions = _helpers(bundle, torch)
        report = json.loads((bundle / REPORT).read_text())
        maxima = torch.load(bundle / "results/detector_smoothquant/final_direct/activation_maxima.pt",
                            map_location="cpu", weights_only=True)
        correction = torch.load(bundle / RESULTS / "bias_alpha065.pt", map_location=device, weights_only=True)
        if correction["alpha"] != .65:
            raise ValueError("W8A8 calibration alpha changed")
        self.context.enter_context(final_detector_level(image))
        replacements = self.context.enter_context(install_smoothquant(image, maxima, .65, backend="triton"))
        if len(replacements) != 128 or set(correction["corrections"]) != set(replacements):
            raise ValueError("W8A8 calibration scope changed")
        lut = make_gelu_lut(device)
        for name, module in replacements.items():
            module.effective_bias.add_(correction["corrections"][name])
            module.kernel_config = report["projection_config"] if name.endswith("attn.proj") else report["kernel_config"]
            if name.endswith("attn.qkv"):
                packed, library = prepare(module.packed_weight_t, bundle / "runtime/joint_qkv_rope_grouped.so")
                module.register_buffer("native_qkv_weight", packed)
                module.native_qkv_library = library
                module.kernel_config = report["qkv_rope_config"]
            if module.bf16_gelu:
                module.register_buffer("gelu_lut", lut)
            module.forward = MethodType(functions.persistent_forward, module)
        library = "joint_mlp_cached_recip_vec16.so" if self.packed else "joint_mlp_blockstore.so"
        config = {**report["mlp_config"], "native_owned_library": str(bundle / "runtime" / library)}
        self.workspace = Workspace(5184, 1024, 4736, 1024, device, **config)
        for block in image.backbone.vision_backbone.trunk.blocks:
            if (block.training or not isinstance(block.ls1, torch.nn.Identity) or
                    not isinstance(block.ls2, torch.nn.Identity) or
                    not isinstance(block.mlp.norm, torch.nn.Identity) or block.attn.cls_token or
                    block.attn.use_rel_pos or block.attn.use_fa3 or
                    block.attn.attn_type != AttentionType.Vanilla or
                    not block.attn.use_rope_real or block.attn.num_heads != 16 or
                    block.window_size not in (0, 24)):
                raise ValueError("Unsupported W8A8 image block")
            block.attn.qkv.input_norm, block.mlp.fc1.input_norm = block.norm1, block.norm2
            block.norm1 = torch.nn.Identity()
            block.norm2 = torch.nn.Identity()
            if block.window_size:
                block.attn.qkv.window_input, block.attn.proj.raster_output = True, True
            block.mlp.workspace = Workspace(5184, 1024, 4736, 1024, device,
                                            shared=self.workspace, **self.workspace.config)
            block.mlp.workspace.prepare(block.mlp.fc1.packed_weight_t, block.mlp.fc2.packed_weight_t)
            if self.packed:
                block.mlp.workspace.prepare_smooth(block.mlp.fc2.smooth_scale)
            block.mlp.forward = MethodType(functions.mega_forward, block.mlp)
            block.attn.fuse_qkv_rope = True
            block.forward = MethodType(functions.attention_residual_block_forward, block)
        for name, module in image.backbone.vision_backbone.named_modules():
            if name in ("trunk.ln_pre", "trunk.ln_post") and isinstance(module, torch.nn.LayerNorm):
                module.forward = MethodType(native_norm_forward, module)
        super().__init__(torch, image, "W8A8 development image", progress)

    def __call__(self, pixels):
        result = super().__call__(pixels)
        if not self.proof["direct_replay_passed"]:
            replay = tuple(value.clone() for value in result)
            direct = self.compiled(pixels)
            if not all(self.torch.equal(a, b) and bool(self.torch.isfinite(b).all())
                       for a, b in zip(replay, direct, strict=True)):
                raise RuntimeError("W8A8 direct/replay outputs differ")
            self.proof["direct_replay_passed"] = True
        if self.workspace.state[-1].item() != 0:
            raise RuntimeError("W8A8 native MLP scheduler error")
        return result

    def close(self):
        self.context.close()
