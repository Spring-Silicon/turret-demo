#!/usr/bin/env python3
"""Opt-in CUDA/XPU parity, cache/alias checks and benchmarks. No servo access."""

import argparse
import base64
import io
import json
import statistics
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


def parity(engine, jpeg, prompts):
    torch = engine.torch
    with torch.inference_mode(), torch.autocast(engine.device_type, dtype=engine.dtype):
        pixels = engine._pixels(jpeg)
        embeddings = engine._encode_prompts(prompts)
        actual = engine._run_heads(engine.image_stage(pixels), embeddings)
        checks = {}
        for index, prompt in enumerate(prompts):
            reference = engine.wrapper(pixels, engine._tokens(prompt))
            checks[prompt] = validate_detections(
                torch, reference, tuple(x[index : index + 1] for x in actual), 0.5
            )
        return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device-type", choices=("xpu", "cuda"), default="xpu")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--camera", default="http://127.0.0.1:8080/stream.mjpg")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--prompt-counts", type=int, nargs="+", default=[1, 2, 3, 4, 8])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-detections", action="store_true")
    args = parser.parse_args()
    import torch
    from PIL import Image

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
    engine = Sam31Engine(args.checkpoint, 0, "float16", 0.5, True, 1, device_type=args.device_type)
    reports = []
    prompts = ["person", "computer monitor", "keyboard", "chair"]
    for batch_size in args.batch_sizes:
        engine.grounding_batch_size = batch_size
        warm = engine.detect_many(jpeg, prompts)
        checks = parity(engine, jpeg, prompts)
        if args.require_detections:
            assert len([b for b in warm["boxes"] if b["prompt"] == "person"]) >= 2
        for count in args.prompt_counts:
            selected = (prompts + ["cup", "bottle", "hand", "table"])[:count]
            engine.detect_many(jpeg, selected)
            samples = []
            for _ in range(args.iterations):
                before_image, before_text = (
                    engine.image_stage.calls,
                    engine.text_stage.calls,
                )
                result = engine.detect_many(jpeg, selected)
                assert engine.image_stage.calls == before_image + 1
                assert engine.text_stage.calls == before_text
                graph_key = "cuda_graph" if args.device_type == "cuda" else "sycl_graph"
                assert result["text_cache_hit"] and result[graph_key]
                assert len(result["categories"]) == count
                assert [c["prompt"] for c in result["categories"]] == selected
                assert all(
                    b["prompt"] == selected[b["prompt_index"]] for b in result["boxes"]
                )
                samples.append(result)
            report = {
                "device": engine.runtime.get_device_name(),
                "device_type": args.device_type,
                "torch": torch.__version__,
                "torch_compile": result["torch_compile"],
                graph_key: result[graph_key],
                "accuracy_policy": result["accuracy_policy"],
                "preprocess_validation": result["preprocess_validation"],
                "batch_size": batch_size,
                "prompts": count,
                "latency_ms": statistics.median(r["latency_ms"] for r in samples),
                "timing": {
                    k: round(statistics.median(r["timing"][k] for r in samples), 2)
                    for k in result["timing"]
                },
                "allocated_mb": round(engine.runtime.memory_allocated() / 1e6),
                "reserved_mb": round(engine.runtime.memory_reserved() / 1e6),
                "counts": [c["count"] for c in result["categories"]],
                "validation": checks,
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
            if args.output:
                args.output.write_bytes(base64.b64decode(result["jpeg"]))
    # A different image and changed/reordered prompts must update graph inputs.
    buffer = io.BytesIO()
    Image.open(io.BytesIO(jpeg)).transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(
        buffer, format="JPEG"
    )
    changed = ["chair", "person", "bottle"]
    checks = parity(engine, buffer.getvalue(), changed)
    before = engine.text_stage.calls
    engine.detect_many(buffer.getvalue(), ["person", "chair", "bottle"])
    assert engine.text_stage.calls == before
    engine.detect_many(buffer.getvalue(), ["person", "chair", "cup"])
    assert engine.text_stage.calls == before + 1
    print(
        json.dumps(
            {"shared_features_verified": True, "changed_image_and_prompts": checks}
        ),
        flush=True,
    )
    if args.report:
        args.report.write_text(json.dumps({"benchmarks": reports,
            "changed_image_and_prompts": checks, "shared_features_verified": True}, indent=2)+"\n")


if __name__ == "__main__":
    main()
