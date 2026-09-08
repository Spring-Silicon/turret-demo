"""SmoothQuant W8A8 for SAM 3.1's shared detector image graph only.

FP32 master parameters are retained. Packed INT8 weights and activations feed
INT32 GEMMs; per-output weight and per-token activation scales reconstruct the
FP16 result. MLP fc1 instead consumes/returns BF16 and rounds before GELU.
Attention, RoPE, normalization, patch convolution and the final neck stay dense.
"""
from contextlib import contextmanager
from collections.abc import Mapping
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class ImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, pixels):
        if pixels.shape != (1, 3, 1008, 1008) or pixels.dtype != torch.float32:
            raise ValueError("Expected FP32 [1, 3, 1008, 1008]")
        backbone = self.backbone.forward_image(
            pixels, need_interactive_out=False, need_propagation_out=False
        )
        return backbone["backbone_fpn"][-1].tensors, backbone["vision_pos_enc"][-1]


def _final_neck(self, tensor_list, *, need_sam3_out=True,
                need_interactive_out=True, need_propagation_out=True):
    if not need_sam3_out or need_interactive_out or need_propagation_out:
        raise ValueError("Final-only neck supports the detector image boundary only")
    from sam3.model.data_misc import NestedTensor
    x = self.trunk(tensor_list)[-1]
    data = getattr(x, "tensors", x)
    value = self.convs[-1](data)
    position = self.position_encoding(value).to(value.dtype)
    return [NestedTensor(value, getattr(x, "mask", None))], [position], [], [], [], []


@contextmanager
def final_detector_level(encoder):
    neck = encoder.backbone.vision_backbone
    if encoder.backbone.scalp != 0 or len(neck.convs) != 3:
        raise ValueError("Unexpected detector neck configuration")
    original = neck.forward
    neck.forward = MethodType(_final_neck, neck)
    try:
        yield
    finally:
        neck.forward = original


def image_linears(encoder):
    blocks = encoder.backbone.vision_backbone.trunk.blocks
    if len(blocks) != 32:
        raise ValueError("Expected the 32-block SAM 3.1 image ViT")
    result = {}
    for i, block in enumerate(blocks):
        for name in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            result[f"blocks.{i}.{name}"] = block.get_submodule(name)
    return result


class Calibration:
    def __init__(self, encoder):
        self.encoder = encoder
        self.maxima = {}
        self.calls = {}
        self.handles = []

    def observe(self, name, value):
        dtype = torch.bfloat16 if name.endswith("mlp.fc1") else torch.float16
        flat = value.detach().to(dtype).float().reshape(-1, value.shape[-1])
        maximum = flat.abs().amax(dim=0)
        self.maxima[name] = torch.maximum(self.maxima.get(name, maximum), maximum)
        self.calls[name] = self.calls.get(name, 0) + 1

    def __enter__(self):
        for name, linear in image_linears(self.encoder).items():
            if name.endswith("mlp.fc1"):
                block_idx = int(name.split(".")[1])
                module = self.encoder.backbone.vision_backbone.trunk.blocks[block_idx].mlp
            else:
                module = linear
            self.handles.append(module.register_forward_pre_hook(
                lambda mod, args, name=name: self.observe(name, args[0])))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def state(self):
        expected = set(image_linears(self.encoder))
        if set(self.maxima) != expected:
            raise ValueError("Calibration did not execute every image Linear route")
        if not all(torch.isfinite(v).all().item() for v in self.maxima.values()):
            raise ValueError("Nonfinite calibration")
        return {k: v.cpu() for k, v in self.maxima.items()}


class SmoothQuantLinear(nn.Module):
    def __init__(self, source, act_max, alpha=0.5, *, bf16_gelu=False,
                 backend="aten"):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between zero and one")
        if source.weight.dtype != torch.float32:
            raise ValueError("SmoothQuant requires FP32 master weights")
        self.weight = source.weight
        self.bias = source.bias
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.bf16_gelu = bf16_gelu
        self.backend = backend
        dtype = torch.bfloat16 if bf16_gelu else torch.float16
        # Start from the same rounded operands that the dense route consumes.
        effective = self.weight.detach().to(dtype).float()
        act_max = act_max.to(device=effective.device, dtype=torch.float32)
        if act_max.shape != (self.in_features,) or not torch.isfinite(act_max).all():
            raise ValueError("Invalid activation maxima")
        scale = (act_max.clamp_min(1e-5).pow(alpha) /
                 effective.abs().amax(0).clamp_min(1e-5).pow(1-alpha)).clamp_min(1e-5)
        smoothed = effective * scale
        weight_scale = smoothed.abs().amax(1).clamp_min(1e-5) / 127
        quantized = (smoothed / weight_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        self.register_buffer("smooth_scale", scale)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("packed_weight_t", quantized.t().contiguous())
        bias = (self.bias.detach().to(dtype).float() if self.bias is not None
                else torch.zeros(self.out_features, device=effective.device))
        self.register_buffer("effective_bias", bias)

    def forward(self, x):
        dtype = torch.bfloat16 if self.bf16_gelu else torch.float16
        shape = x.shape[:-1] + (self.out_features,)
        if self.backend == "triton":
            from .smoothquant_kernels import quantize, dequantize
            a, scale = quantize(x, self.smooth_scale, dtype)
        elif self.backend == "aten":
            flat = x.to(dtype).float().reshape(-1, self.in_features) / self.smooth_scale
            scale = flat.abs().amax(-1, keepdim=True).clamp_min(1e-5) / 127
            a = (flat / scale).round().clamp(-127, 127).to(torch.int8)
        else:
            raise ValueError(f"Unknown backend {self.backend}")
        with torch.autocast(x.device.type, enabled=False):
            accum = torch._int_mm(a, self.packed_weight_t)
            if self.backend == "triton":
                out = dequantize(accum, scale, self.weight_scale, self.effective_bias,
                                 dtype, self.bf16_gelu)
            else:
                out = (accum.float() * scale * self.weight_scale + self.effective_bias).to(dtype)
                if self.bf16_gelu:
                    # Rounding before GELU is part of the upstream BF16 contract.
                    out = F.gelu(out)
        return out.reshape(shape)


def _quant_mlp(self, x):
    x = self.fc1(x)
    x = self.drop1(x)
    x = self.norm(x)
    return self.drop2(self.fc2(x))


@contextmanager
def install_smoothquant(encoder, maxima, alpha=0.5, backend="aten"):
    originals = image_linears(encoder)
    if set(maxima) != set(originals):
        raise ValueError("Calibration scope differs from image Linear scope")
    replacements = {}
    # Construct all candidates before modifying any live route.
    for name, source in originals.items():
        replacements[name] = SmoothQuantLinear(source, maxima[name], alpha[name] if isinstance(alpha,Mapping) else alpha,
                            bf16_gelu=name.endswith("mlp.fc1"), backend=backend)
    trunk = encoder.backbone.vision_backbone.trunk
    mlp_forwards = [(block.mlp, block.mlp.forward) for block in trunk.blocks]
    try:
        for name, replacement in replacements.items():
            parent, leaf = name.rsplit(".", 1)
            setattr(trunk.get_submodule(parent), leaf, replacement)
        for mlp, _ in mlp_forwards:
            if not isinstance(mlp.act, nn.GELU):
                raise ValueError("Expected exact GELU")
            mlp.forward = MethodType(_quant_mlp, mlp)
        yield replacements
    finally:
        for name, source in originals.items():
            parent, leaf = name.rsplit(".", 1)
            setattr(trunk.get_submodule(parent), leaf, source)
        for mlp, original in mlp_forwards:
            mlp.forward = original
