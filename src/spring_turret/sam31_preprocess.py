"""Exact uint8 resize plus compiled GPU normalization, without FP32 host copies."""

import io
import time

if __package__:
    from .sam31_graph import CompiledStage
else:
    from sam31_graph import CompiledStage


def normalization_module(torch):
    class Normalize(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Evaluate the original CPU float32 operation order for every
            # possible uint8 value. A lookup prevents compiler FMA/reassociation
            # from changing any normalized pixel's bits.
            values = torch.arange(256, dtype=torch.float32)
            self.register_buffer("values", values.mul_(1.0 / 255).sub_(.5).div_(.5))

        def forward(self, hwc):
            pixels = self.values[hwc.long()]
            return (pixels.permute(2, 0, 1).unsqueeze(0),)

    return Normalize()


class ExactImagePreprocessor:
    def __init__(self, torch, device, progress):
        from PIL import Image
        from torchvision.transforms import v2
        from torchvision.io import decode_jpeg, ImageReadMode

        self.torch, self.device, self.Image = torch, device, Image
        self.decode_jpeg, self.rgb_mode = decode_jpeg, ImageReadMode.RGB
        self.tensor_resize = v2.Resize(size=(1008, 1008))
        self.resize = v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(1008, 1008)),
        ])
        self.reference = v2.Compose([
            self.resize, v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[.5]*3, std=[.5]*3),
        ])
        self.stage = CompiledStage(torch, normalization_module(torch).to(device),
                                   "image-normalization", progress)
        self.validation = None
        self.last_cpu_ms = 0.0

    def prepare_cpu(self, jpeg):
        """Owned uint8 CPU buffer only; safe on the preparation thread."""
        started = time.perf_counter()
        torch = self.torch
        image = self.Image.open(io.BytesIO(jpeg))  # Header only on the fast path.
        if image.format == "JPEG" and image.mode in ("RGB", "L"):
            encoded = torch.frombuffer(bytearray(jpeg), dtype=torch.uint8)
            decoded = self.decode_jpeg(encoded, mode=self.rgb_mode)
            decoded = decoded.permute(1, 2, 0).contiguous().permute(2, 0, 1)
            resized = self.tensor_resize(decoded)
        else:
            # Preserve existing semantics for CMYK/non-JPEG diagnostic inputs.
            resized = self.resize(image.convert("RGB"))
        resized = resized.permute(1, 2, 0).contiguous()
        return resized, (time.perf_counter() - started) * 1000

    def __call__(self, jpeg, *, prepared=None):
        torch = self.torch
        resized, self.last_cpu_ms = self.prepare_cpu(jpeg) if prepared is None else prepared
        # Once captured, copy uint8 straight to the graph's owned input buffer;
        # avoid allocating an intermediate GPU image and copying it again.
        if self.stage.graph is None:
            resized = resized.to(self.device)
        with torch.inference_mode():
            pixels, = self.stage(resized)
            if self.validation is None:
                image = self.Image.open(io.BytesIO(jpeg))
                reference = self.reference(image.convert("RGB")).unsqueeze(0)
                if not torch.equal(pixels.cpu(), reference):
                    raise RuntimeError("GPU preprocessing differs from original CPU pixels")
                self.validation = {"bitwise_equal": True, "backend": "compiled-uint8-lut"}
        return pixels
