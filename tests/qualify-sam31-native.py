#!/usr/bin/env python3
"""Opt-in destination qualification of the copied native image. No servo access."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.sam31_worker import Sam31Engine, validate_detections
from spring_turret.sam31_native import ARTIFACT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    fixture = args.bundle / "resident-attention-prefetch1-gpu0/native-attention-prefetch1"
    report = {"artifact": ARTIFACT, "checks": [], "timing": {}, "passed": False}
    engine = None
    try:
        engine = Sam31Engine(args.checkpoint, 0, "float16", .5, True,
                             native_bundle=args.bundle)
        with torch.inference_mode(), torch.autocast("xpu", dtype=torch.float16):
            prompts = ["person", "car", "truck", "bottle"]
            for index in [0, 1, 2, 3, 4, 0]:
                pixels = next(iter(load_file(fixture / f"input-{index}.safetensors").values()))
                expected_features = load_file(fixture / f"output-{index}.safetensors")
                features = engine.image_stage(pixels)
                for expected, actual in zip(sorted(expected_features.items()), features, strict=True):
                    assert torch.equal(expected[1].view(torch.int16), actual.cpu().view(torch.int16)), "native feature bits changed"
                embeddings = engine._encode_prompts(prompts)
                outputs = engine._run_heads(features, embeddings)
                for j, prompt in enumerate(prompts):
                    expected = engine.wrapper(pixels.to(engine.device), engine._tokens(prompt))
                    result = validate_detections(torch, expected,
                        tuple(value[j:j + 1] for value in outputs), .5)
                    report["checks"].append({"image": index, "prompt": prompt, **result})
                    print(json.dumps(report["checks"][-1]), flush=True)
            for count in (1, 2, 4):
                selected = prompts[:count]
                embeddings = engine._encode_prompts(selected)
                image_ms, head_ms = [], []
                for _ in range(args.iterations):
                    before_image, before_text = engine.image_stage.calls, engine.text_stage.calls
                    start = time.perf_counter()
                    features = engine.image_stage(pixels)
                    torch.xpu.synchronize()
                    mid = time.perf_counter()
                    engine._run_heads(features, embeddings)
                    torch.xpu.synchronize()
                    end = time.perf_counter()
                    assert engine.image_stage.calls == before_image + 1
                    assert engine.text_stage.calls == before_text
                    image_ms.append((mid - start) * 1000)
                    head_ms.append((end - mid) * 1000)
                report["timing"][str(count)] = {
                    "image_with_ipc_ms": statistics.median(image_ms),
                    "grounding_ms": statistics.median(head_ms),
                    "model_with_ipc_ms": statistics.median([a + b for a, b in zip(image_ms, head_ms)]),
                }
            report["native_replay"] = engine.image_stage.proof
            report["passed"] = True
    finally:
        if engine is not None:
            engine.image_stage.close()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["timing"], indent=2))


if __name__ == "__main__":
    main()
