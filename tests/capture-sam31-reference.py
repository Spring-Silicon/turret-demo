"""Capture exact current-release outputs before changing model/runtime.

Run in the configured inference Python with PYTHONPATH pointing at the installed
release, the service stopped, and its normal user-space GPU libraries. This does
not alter servo or service state. Checkpoint-derived output tensors stay local.
"""
import argparse
import hashlib
import json
from pathlib import Path

from spring_turret.sam31_worker import Sam31Engine

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
p.add_argument('--images', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
c = json.loads(a.config.read_text())['inference']
engine = Sam31Engine(Path(c['checkpoint']), int(c.get('device', 0)), c.get('precision', 'float16'),
                     float(c.get('confidence', .5)), True,
                     w8a8_development_bundle=Path(c['sam31_w8a8_development_bundle']))
torch = engine.torch
rows = []
try:
    with torch.inference_mode(), torch.autocast('xpu', dtype=engine.dtype):
        for path in sorted(a.images.glob('*.jpg')):
            jpeg = path.read_bytes()
            pixels = engine._pixels(jpeg)
            features = engine.image_stage(pixels)
            prompts = ['face', 'person', 'chair', 'bottle']
            raw = engine._run_heads(features, engine._encode_prompts(prompts))
            for index, prompt in enumerate(prompts):
                observed = tuple(t[index:index+1].cpu().clone() for t in raw)
                dense = tuple(t.cpu().clone() for t in engine.wrapper(pixels, engine._tokens(prompt)))
                rows.append({'image_sha256':hashlib.sha256(jpeg).hexdigest(), 'image':path.name,
                             'prompt':prompt, 'actual':observed, 'dense':dense})
            print(json.dumps({'baseline_image':path.name, 'prompts':len(prompts)}), flush=True)
    torch.save({'bundle':c['sam31_w8a8_development_bundle'], 'torch':str(torch.__version__),
                'rows':rows}, a.output)
finally:
    engine.image_stage.close()
