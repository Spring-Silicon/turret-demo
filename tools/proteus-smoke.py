#!/usr/bin/env python3
"""On-device worker smoke test. Never opens or commands the servo device."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spring_turret.detection import WorkerClient, validate_tracking_result
from spring_turret.hardware import inference_runtime
from spring_turret.models import is_tracking_model


def api(path, body=None):
    request = Request('http://127.0.0.1:8080' + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    with urlopen(request, timeout=5) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--bundle')
    parser.add_argument('--memory-bundle')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--resize', help='Exercise a camera input size, e.g. 1280x720')
    parser.add_argument('--prompt', default='person')
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--pause-live', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())['inference']
    cfg['model'] = args.profile
    if args.bundle:
        cfg['proteus_bundle'] = args.bundle
    if args.memory_bundle:
        cfg['proteus_memory_bundle'] = args.memory_bundle
    if args.probe:
        spec = inference_runtime(cfg).prepare()
        command = spec.command[:spec.command.index('-u')] + ['-c',
            'import torch; print(torch.__version__); print(torch.xpu.is_available()); '
            'print(torch.xpu.get_device_name(0)); import scipy, PIL, timm; print("imports ok")']
        return subprocess.call(command, env=spec.environment)
    original = json.loads(api('/api/status'))
    jpeg = args.image.read_bytes() if args.image else api(original['detection']['frame_url'])
    if args.resize:
        from PIL import Image
        size = tuple(map(int, args.resize.split('x')))
        if len(size) != 2 or min(size) <= 0:
            parser.error('--resize must be WIDTHxHEIGHT')
        with Image.open(io.BytesIO(jpeg)) as image:
            output = io.BytesIO()
            image.convert('RGB').resize(size, Image.Resampling.BILINEAR).save(output, 'JPEG', quality=95)
            jpeg = output.getvalue()
    paused = False
    client = WorkerClient(cfg)
    progress = lambda stage: print(json.dumps({'progress':stage}), flush=True)
    try:
        if args.pause_live:
            api('/api/detection/prompts', {'prompts':[]})
            paused = True
            time.sleep(4)
        started = time.monotonic()
        client.launch()
        ready = client.receive(900, progress)
        print(json.dumps({'ready':ready, 'load_seconds':time.monotonic()-started}), flush=True)
        for index in range(args.frames):
            # Exercise session invalidation separately from the steady frames.
            result = client.detect(index, jpeg, [args.prompt], progress,
                session_revision=1 if index < args.frames-1 else 2,
                captured_at=time.monotonic(), camera_identity='smoke-fixed-image')
            if args.profile == 'sam3.1-nomem':
                assert result.get('mask_detection') is True
                assert result.get('temporal_tracking') is not True
                assert result.get('torch_compile') is True and result.get('cuda_graph') is True
            else:
                validate_tracking_result(result, [args.prompt], temporal=is_tracking_model(args.profile),
                    backend='proteus-' + args.profile)
            print(json.dumps({key:result.get(key) for key in ('id','tracking_frame','tracking_reset',
                'temporal_tracking','memory_frames','active_instance_ids','seed_score','torch_compile',
                'sycl_graph','cuda_graph','compilation_scope','timing','warning')} |
                {'boxes':len(result['boxes']), 'overlay':result.get('mask_overlay') is not None}), flush=True)
    finally:
        client.stop()
        if paused:
            current = json.loads(api('/api/status'))['detection']
            # Do not overwrite a new user selection made during the test.
            if not current['prompts'] and current['model'] == original['detection']['model']:
                api('/api/detection/prompts', {'prompts':original['detection']['prompts']})
                print(json.dumps({'restored_prompts':original['detection']['prompts']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
