"""Exact uint8 resize plus compiled GPU normalization, without FP32 host copies."""

import io

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

        self.torch, self.device, self.Image = torch, device, Image
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

    def __call__(self, jpeg):
        torch = self.torch
        image = self.Image.open(io.BytesIO(jpeg)).convert("RGB")
        # Keep the existing PIL decode and torchvision uint8 antialiased resize.
        # Upload 3 MiB uint8 rather than materializing/uploading 12 MiB float32.
        # HWC lookup retains the original channels-last CHW input strides.
        resized = self.resize(image).permute(1, 2, 0).contiguous().to(self.device)
        with torch.inference_mode():
            pixels, = self.stage(resized)
            if self.validation is None:
                reference = self.reference(image).unsqueeze(0)
                if not torch.equal(pixels.cpu(), reference):
                    raise RuntimeError("GPU preprocessing differs from original CPU pixels")
                self.validation = {"bitwise_equal": True, "backend": "compiled-uint8-lut"}
        return pixels
