"""Video preprocessing with exact PIL pixels and no FP32 host image transfer.

The video and image predictors have different resize/normalization contracts.
Keep the video path explicit: PIL bilinear, float32 division by 255, then .5/.5.
"""
import io
import time


def video_normalization(torch):
    class Normalize(torch.nn.Module):
        def __init__(self):
            super().__init__()
            values = torch.arange(256, dtype=torch.float32)
            self.register_buffer("values", values.div_(255).sub_(.5).div_(.5))

        def forward(self, hwc):
            # A LUT preserves the CPU operation order exactly, including values
            # where compiler reassociation would change the last float32 bits.
            return (self.values[hwc.long()].permute(2, 0, 1).contiguous(),)
    return Normalize()


class VideoPreprocessor:
    def __init__(self, torch, device, progress=None, *, image_size=1008):
        if __package__:
            from .sam31_graph import CompiledStage
        else:
            from sam31_graph import CompiledStage
        self.torch, self.device = torch, device
        self.image_size = image_size
        self.last_cpu_ms = 0.
        self.stage = CompiledStage(torch, video_normalization(torch).to(device),
                                  "video-normalization", lambda *_: progress and progress("compiling_video_normalization"))
        self.validated = False

    def prepare_cpu(self, jpeg):
        """Exact PIL bilinear pixels; owned CPU-only buffer for optional overlap."""
        import numpy as np
        from PIL import Image
        started = time.perf_counter()
        with Image.open(io.BytesIO(jpeg)) as encoded:
            image = encoded.convert("RGB")
        size = image.size
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        # Owned, contiguous uint8: 3 MB instead of a 12 MB normalized host image.
        host = self.torch.from_numpy(np.array(resized, dtype=np.uint8, copy=True))
        return host, size, (time.perf_counter()-started)*1000

    def __call__(self, jpeg, *, prepared=None):
        from PIL import Image
        from torchvision.transforms import functional as TF
        torch = self.torch
        host, size, self.last_cpu_ms = self.prepare_cpu(jpeg) if prepared is None else prepared
        source = host.to(self.device) if self.stage.graph is None else host
        with torch.inference_mode():
            pixels, = self.stage(source)
            if not self.validated:
                # Check against the unmodified video contract, not a reference
                # derived from the candidate's resized pixels.
                with Image.open(io.BytesIO(jpeg)) as encoded:
                    resized = TF.resize(encoded.convert('RGB'), [self.image_size, self.image_size])
                reference = (TF.to_tensor(resized) - .5) / .5
                if not torch.equal(pixels.cpu(), reference):
                    raise RuntimeError("Video preprocessing changed model input pixels")
                self.validated = True
        # The tracking image-region cache intentionally shares features across
        # prompt sessions by tensor OBJECT identity. A replay buffer is mutable
        # and reused each frame: handing it through would reuse stale image
        # features forever. Each frame must own a fresh immutable tensor, which
        # all of that frame's prompt sessions can safely share.
        return pixels.clone(), size
