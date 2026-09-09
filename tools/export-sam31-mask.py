#!/usr/bin/env python3
"""Compile fixed SAM mask stages offline using the inference runtime's Python.

Use the same graphics environment as the mask worker, and stop live inference
first. This tool reads an existing JPEG and never opens camera or motor devices.
A newly exported package still needs fresh-process qualification before activation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-directory', type=Path,
                        default=Path(__file__).resolve().parents[1]/'src/spring_turret')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mask-bundle', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=.5)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--prompt', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.device_type, args.precision, args.compiled_bundle = 'xpu', 'float16', None
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; export into a new directory')
    if not args.worker_directory.is_dir() or not args.frame.is_file():
        parser.error('Worker directory and existing JPEG are required')
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.worker_directory.resolve()))
    import torch
    from sam31_mask_worker import MaskEngine
    from sam31_mask_artifacts import MaskArtifactWriter
    begin = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=output.name+'.staging-', dir=output.parent) as temporary:
        staging = Path(temporary)/'package'
        staging.mkdir()
        writer = MaskArtifactWriter(torch, args, staging)
        engine = MaskEngine(args, stage_factory=writer.stage)
        result = engine.detect_many(args.frame.read_bytes(), args.prompt or ['person'])
        writer.finish()
        torch.xpu.synchronize()
        os.replace(staging, output)
    print(json.dumps({'package': str(output), 'seconds': time.perf_counter()-begin,
                      'model_gpu_ms': result['timing']['model_gpu_ms'],
                      'qualification': 'fresh-process comparison required before activation'}))


if __name__ == '__main__':
    main()
