#!/usr/bin/env python3
"""Opt-in real XPU test; never opens or moves the servo.

Run with the inference Python and --checkpoint; uses a live camera frame unless
--image points at a local JPEG. Compiling on first use can take several minutes.
"""

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.sam31_worker import Sam31Engine, validate_detections


def test_parity_gate(torch):
    logits = torch.full((1, 200, 1), -6.0)
    logits[0, 0, 0] = 3.0
    reference = (logits, torch.full((1, 200, 4), 0.5), torch.tensor([[10.0]]))
    rejected_changed = tuple(x.clone() for x in reference)
    rejected_changed[0][0, 1, 0] = -3
    rejected_changed[1][0, 1] = 0.1
    validate_detections(torch, reference, rejected_changed, 0.5)
    for kind in ("missing", "extra", "moved"):
        bad = tuple(x.clone() for x in reference)
        if kind == "missing":
            bad[0][0, 0, 0] = -6
        elif kind == "extra":
            bad[0][0, 1, 0] = 3
        else:
            bad[1][0, 0, 0] += 0.1
        try:
            validate_detections(torch, reference, bad, 0.5)
        except AssertionError:
            continue
        raise AssertionError(f"parity gate accepted a {kind} detection")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--camera", default="http://127.0.0.1:8080/stream.mjpg")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch

    test_parity_gate(torch)
    if args.image:
        jpeg = args.image.read_bytes()
    else:
        with urllib.request.urlopen(args.camera, timeout=10) as stream:
            buffer = b""
            while b"\xff\xd9" not in buffer:
                buffer += stream.read(4096)
                if len(buffer) > 8 * 1024 * 1024:
                    raise RuntimeError("camera did not supply a JPEG")
            jpeg = buffer[buffer.index(b"\xff\xd8") : buffer.index(b"\xff\xd9") + 2]
    engine = Sam31Engine(args.checkpoint, 0, "float16", 0.5, True)
    singles = {}
    for prompt in ("keyboard", "person", "chair", "keyboard"):
        result = engine.detect(jpeg, prompt)
        # Verify graph inputs are updated when the prompt changes, not captured
        # as constants. Compare raw outputs before confidence filtering.
        torch = engine.torch
        with torch.inference_mode(), torch.autocast("xpu", dtype=engine.dtype):
            pixels, tokens = engine._inputs(jpeg, prompt)
            expected = engine.compiled(pixels, tokens)
            validation = validate_detections(
                torch, engine.wrapper(pixels, tokens), expected, 0.5
            )
            actual = engine.inference_callable(pixels, tokens)
            for reference, replayed in zip(expected, actual, strict=True):
                torch.testing.assert_close(replayed, reference, rtol=0.001, atol=0.001)
        annotated = base64.b64decode(result.pop("jpeg"))
        assert result["torch_compile"] and result["sycl_graph"]
        singles[prompt] = [
            {"xyxy": box["xyxy"], "score": box["score"]} for box in result["boxes"]
        ]
        if args.output:
            args.output.write_bytes(annotated)
        print(
            json.dumps({"prompt": prompt, **result, "validation": validation}),
            flush=True,
        )
    prompts = ["person", "keyboard", "chair"]
    combined = engine.detect_many(jpeg, prompts)
    assert [category["prompt"] for category in combined["categories"]] == prompts
    for index, prompt in enumerate(prompts):
        boxes = [box for box in combined["boxes"] if box["prompt"] == prompt]
        assert [
            {"xyxy": box["xyxy"], "score": box["score"]} for box in boxes
        ] == singles[prompt]
        assert all(box["prompt_index"] == index for box in boxes)
        assert combined["categories"][index]["count"] == len(boxes)
    assert len({category["color"] for category in combined["categories"]}) == len(
        prompts
    )
    annotated = base64.b64decode(combined.pop("jpeg"))
    if args.output:
        args.output.write_bytes(annotated)
    print(json.dumps({"multi_prompt_verified": True, **combined}), flush=True)


if __name__ == "__main__":
    main()
