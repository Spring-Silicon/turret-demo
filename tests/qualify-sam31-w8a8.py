#!/usr/bin/env python3
"""Verify copied W8A8 feature parity and REPORT (not waive) detector failures."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.sam31_worker import Sam31Engine, validate_detections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image", type=Path)
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    report = {"copy_parity_passed": False, "accuracy_qualified": False, "frames": [], "checks": []}
    engine = None
    try:
        engine = Sam31Engine(args.checkpoint, 0, "float16", .5, True, w8a8_development_bundle=args.bundle)
        root = args.bundle / "results/turret_megakernel"
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            frozen = torch.load(root / "frozen_refs.pt", weights_only=True, map_location="cpu")
            saved = torch.load(root / "features_native_qkv_grouped_gpu1.pt", weights_only=True, map_location="cpu")
        cases = saved["configuration"]["validation"]
        with torch.inference_mode(), torch.autocast("xpu", dtype=torch.float16):
            for i in list(range(len(cases))) + [0]:
                pixels = frozen["pixels"][i].to(engine.device)
                features = tuple(x.clone() for x in engine.image_stage(pixels))
                direct = engine.image_stage.compiled(pixels)
                exact = [torch.equal(a.cpu(), b) and torch.equal(a, c)
                         for a, b, c in zip(features, saved["features"][i], direct, strict=True)]
                assert all(exact), f"Copied feature/direct/replay mismatch on frame {i}"
                assert all(bool(torch.isfinite(x).all()) for x in features)
                report["frames"].append({"frame": i, "source_and_replay_exact": exact})
                if len(report["frames"]) > len(cases):
                    continue
                prompts = cases[i]["prompts"]
                outputs = engine._run_heads(features, engine._encode_prompts(prompts))
                for j, prompt in enumerate(prompts):
                    expected = tuple(v.to(engine.device) for v in frozen["refs"][i, j])
                    actual = tuple(v[j:j+1] for v in outputs)
                    row = {"frame": i, "prompt": prompt}
                    try:
                        row.update(validate_detections(torch, expected, actual, .5))
                        row["passed"] = True
                    except AssertionError as error:
                        row.update(passed=False, error=str(error))
                    # Independent gates: the unchanged validator raises on
                    # confidence first, so a failure doesn't establish box parity.
                    def unpack(values):
                        logits, boxes, presence = values
                        scores = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).squeeze(-1).float()
                        center, size = boxes[..., :2], boxes[..., 2:]
                        return scores, torch.cat((center-size/2, center+size/2), -1).clamp(0, 1).float()
                    es, eb = unpack(expected)
                    ac, ab = unpack(actual)
                    near, retained = (es > .47) | (ac > .47), (es > .5) | (ac > .5)
                    row["confidence_passed"] = bool(((es-ac).abs()[near] <= .03).all())
                    row["boxes_passed"] = bool(((eb-ab).abs()[retained] <= .01).all())
                    report["checks"].append(row)
                print(json.dumps(report["frames"][-1]), flush=True)
            report["failures"] = {k: sum(not row[f"{k}_passed"] for row in report["checks"])
                                  for k in ("confidence", "boxes")}
            # Do not activate a transfer that differs from the reported candidate.
            assert report["failures"] == {"confidence": 5, "boxes": 4}, report["failures"]
            times = []
            for _ in range(20):
                started = time.perf_counter()
                engine.image_stage(pixels)
                torch.xpu.synchronize()
                times.append((time.perf_counter()-started)*1000)
            report["image_replay_ms"] = statistics.median(times)
            report["copy_parity_passed"] = True
            report["image_proof"] = engine.image_stage.proof
        if args.image:
            # Exercises unchanged LIVE dense startup gates; no validated_batches
            # injection or quantized reference substitution is allowed here.
            rows = []
            for _ in range(7):
                result = engine.detect_many(args.image.read_bytes(), ["person"], client_overlay=True)
                rows.append(result["timing"])
            report["worker"] = {"boxes": result["boxes"], "validation": result["validation"],
                                "timing": {k: statistics.median(r[k] for r in rows[2:]) for k in rows[0]}}
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if engine is not None:
            engine.image_stage.close()
    print(json.dumps({k:v for k,v in report.items() if k not in ("frames", "checks")}), flush=True)


if __name__ == "__main__":
    main()
