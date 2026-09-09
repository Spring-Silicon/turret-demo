"""Strict full-output parity against the previous preprocessing implementation.

Run with the demo stopped and an idle XPU. This compares the SAME configured
model, not W8A8 against dense SAM, and never relaxes equality on development mode.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
p.add_argument('--reference-preprocessor', type=Path, required=True)
p.add_argument('--images', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
c = json.loads(a.config.read_text())['inference']
cache = Path(c['cache_dir']) / 'sam31-israel-w8a8'
os.environ.update(OMP_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='2',
    TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'), TRITON_CACHE_DIR=str(cache/'triton'),
    XDG_CACHE_HOME=str(cache))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from spring_turret.sam31_worker import Sam31Engine
from spring_turret.prefetch import LatestPreparation

spec = importlib.util.spec_from_file_location('spring_turret.reference_preprocessor', a.reference_preprocessor)
reference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_module)
engine = Sam31Engine(Path(c['checkpoint']), int(c.get('device', 0)), c.get('precision', 'float16'),
    float(c.get('confidence', .5)), True,
    w8a8_development_bundle=Path(c['sam31_w8a8_development_bundle']),
    allow_unqualified_w8a8=c.get('sam31_allow_unqualified_w8a8', False))
torch = engine.torch
optimized = engine.preprocessor
reference = reference_module.ExactImagePreprocessor(torch, engine.device, engine._progress)
images = [path.read_bytes() for path in sorted(a.images.glob('*.jpg'))]
assert len(images) >= 3, 'Need multiple actual camera images'
original_decode = engine._decode_outputs
raw = None
def capture(outputs):
    global raw
    raw = tuple(value.detach().cpu().clone() for value in outputs)
    return original_decode(outputs)
engine._decode_outputs = capture
reports = []
try:
    for prompts in (['person'], ['person', 'chair', 'laptop', 'bottle']):
        engine.preprocessor = reference
        engine.detect_many(images[0], prompts, client_overlay=True)  # Warm the prompt batch.
        for index, jpeg in enumerate(images):
            engine.preprocessor = reference
            expected = engine.detect_many(jpeg, prompts, client_overlay=True)
            expected_raw = raw
            engine.preprocessor = optimized
            preparation = LatestPreparation(optimized.prepare_cpu)
            preparation.submit('input', jpeg)
            prepared = None
            deadline = time.monotonic() + 5
            while prepared is None and time.monotonic() < deadline:
                prepared = preparation.take('input', jpeg)
                if prepared is None:
                    time.sleep(.001)
            assert prepared is not None, 'Background preparation did not complete'
            halt = threading.Event()
            def feed_changed_frames():
                i = 0
                while not halt.is_set():
                    preparation.submit(f'background-{i}', images[i % len(images)])
                    i += 1
                    halt.wait(.02)
            feeder = threading.Thread(target=feed_changed_frames)
            feeder.start()
            try:
                actual = engine.detect_many(jpeg, prompts, client_overlay=True, prepared_pixels=prepared)
                assert actual['preprocess_prefetched']
                assert len(raw) == len(expected_raw)
                for old, new in zip(expected_raw, raw, strict=True):
                    assert old.dtype == new.dtype and old.shape == new.shape
                    assert torch.isfinite(new).all()
                    assert torch.equal(old, new), 'Raw logits/boxes/presence changed'
                assert actual['boxes'] == expected['boxes'], 'Published boxes/scores changed'
                assert actual['categories'] == expected['categories'], 'Class counts changed'
                report = {'image':index, 'prompts':len(prompts), 'bitwise_raw_equal':True,
                    'raw_values':sum(t.numel() for t in raw), 'detected_boxes':len(actual['boxes'])}
                reports.append(report)
                print(json.dumps(report), flush=True)
            finally:
                halt.set(); feeder.join(); preparation.close()
    assert sum(r['detected_boxes'] for r in reports) > 0, 'Only empty detections tested'
    result = {'passed':True, 'comparison':'same W8A8 model, 0.15.2 vs overlapped preprocessing',
        'raw_logits_boxes_presence_bitwise_equal':True, 'published_boxes_scores_counts_equal':True,
        'pairs':reports, 'model_accuracy_qualification_changed':False}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('STRICT FULL-OUTPUT PARITY PASSED', flush=True)
finally:
    engine.image_stage.close()
