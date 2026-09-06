"""Pinned Israel W8A8 development image; dense text/heads/reference stay intact.

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
REPORT = "results/turret_megakernel/demo_native_qkv_grouped_gpu1.json"
HELPERS = {"persistent_forward", "mega_forward", "attention_projection_residual",
           "attention_residual_block_forward"}


def verify_bundle(bundle: Path, checkpoint: Path) -> None:
    if digest(checkpoint) != CHECKPOINT:
        raise ValueError("W8A8 requires the pinned SAM 3.1 checkpoint")
    for name, expected in PINNED_FILES.items():
        if digest(bundle / name) != expected:
            raise ValueError(f"W8A8 bundle checksum mismatch: {name}")


def configure_source(bundle: Path, checkpoint: Path) -> None:
    verify_bundle(bundle, checkpoint)
    existing = sys.modules.get("sam3")
    if existing and not Path(existing.__file__).resolve().is_relative_to(bundle.resolve()):
        raise ValueError("W8A8 must load its pinned SAM source before importing sam3")
    sys.path.insert(0, str(bundle))
    sys.path.insert(0, str(bundle / SOURCE_DIR))


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
        self.context = ExitStack()
        self.proof = {"source": "israel", "accuracy_qualified": False,
                      "source_confidence_failures": 5, "source_box_failures": 4,
                      "report_sha256": PINNED_FILES[REPORT], "direct_replay_passed": False}
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
        config = {**report["mlp_config"], "native_owned_library": str(bundle / "runtime/joint_mlp_blockstore.so")}
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
