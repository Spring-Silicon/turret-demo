"""Real XPU pixel-parity and preprocessing timing check (no model/servo writes)."""
import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.sam31_preprocess import ExactImagePreprocessor

p = argparse.ArgumentParser()
p.add_argument("--jpeg", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.xpu.set_device(0)
pre = ExactImagePreprocessor(torch, torch.device('xpu:0'), lambda *args: print(args, file=sys.stderr))
camera = a.jpeg.read_bytes()
frames = [camera]
generator = torch.Generator().manual_seed(18)
for width, height, mode in [(1280, 720, 'RGB'), (720, 1280, 'RGB'), (321, 157, 'RGB'), (333, 222, 'L')]:
    shape = (height, width, 3) if mode == 'RGB' else (height, width)
    data = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator).numpy()
    image = Image.fromarray(data)
    buf = io.BytesIO(); image.save(buf, format='JPEG'); frames.append(buf.getvalue())

with torch.inference_mode():
    for jpeg in frames + frames[::-1]:
        expected = pre.reference(Image.open(io.BytesIO(jpeg)).convert('RGB')).unsqueeze(0)
        actual = pre(jpeg)
        torch.xpu.synchronize()
        assert torch.equal(actual.cpu(), expected), 'Changed-frame pixel mismatch'
        assert actual.stride() == expected.stride(), (actual.stride(), expected.stride())
    # Cover ALL 256 byte values in the exact captured input shape, on changed
    # replays, without JPEG quantization obscuring the exhaustive table check.
    values = torch.arange(1008*1008*3).remainder(256).to(torch.uint8).reshape(1008, 1008, 3)
    for u8 in (values, 255-values, values):
        actual, = pre.stage(u8.to(pre.device))
        expected = u8.float().mul_(1.0/255).sub_(.5).div_(.5).permute(2, 0, 1).unsqueeze(0)
        assert torch.equal(actual.cpu(), expected), 'Exhaustive LUT pixel mismatch'

    def original():
        image = Image.open(io.BytesIO(camera)).convert('RGB')
        return pre.reference(image).unsqueeze(0).to(pre.device)
    timings = {'original_ms': [], 'optimized_ms': []}
    for i in range(30):
        items = [('original_ms', original), ('optimized_ms', lambda: pre(camera))]
        if i % 2: items.reverse()
        for name, fn in items:
            start = time.perf_counter(); output = fn(); torch.xpu.synchronize()
            if i >= 5: timings[name].append((time.perf_counter()-start)*1000)
print(json.dumps({'bitwise_pixel_parity': True, 'changed_image_cases': 10,
                  'all_256_values_replayed': True,
                  **{k: round(statistics.median(v), 3) for k,v in timings.items()}}, indent=2))
