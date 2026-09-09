#!/usr/bin/env python3
"""Verify copied W8A8 feature parity and REPORT (not waive) detector failures."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.sam31_worker import Sam31Engine, validate_detections
from spring_turret.sam31_w8a8 import packed_profile
from spring_turret.sam31_graph import CompiledStage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image", type=Path)
    p.add_argument("--images", type=Path, help="Additional camera frames for old-head versus packed-head checks")
    p.add_argument("--baseline", type=Path, help="Current-release outputs captured before any runtime changes")
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    report = {"copy_parity_passed": False, "accuracy_qualified": False, "frames": [], "checks": []}
    engine = None
    try:
        engine = Sam31Engine(args.checkpoint, 0, "float16", .5, True, w8a8_development_bundle=args.bundle)
        root = args.bundle / "results/turret_megakernel"
        packed = packed_profile(args.bundle)
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            frozen = torch.load(root / "frozen_refs.pt", weights_only=True, map_location="cpu")
            saved = torch.load(root / "features_native_qkv_grouped_gpu1.pt", weights_only=True, map_location="cpu")
            source = (torch.load(root / "packed_heads_rounded_split512_gpu0.pt", weights_only=True,
                                 map_location="cpu")["outputs"] if packed else None)
            baseline = (torch.load(args.baseline, weights_only=True, map_location="cpu")
                        if args.baseline else None)
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
                    if packed:
                        exact_heads = [torch.equal(a.cpu(), b) for a, b in
                                       zip(actual, source["candidate"][i, j], strict=True)]
                        if not all(exact_heads):
                            report["copy_failure"] = {
                                "frame": i, "prompt": prompt, "exact_heads": exact_heads,
                                "max_abs_error": [(a.cpu().float()-b.float()).abs().max().item() for a,b in
                                                  zip(actual, source["candidate"][i,j], strict=True)],
                                "input_strides": [list(x.stride()) for x in engine.grounding_stages[1].inputs],
                            }
                            torch.save({"actual": tuple(x.cpu() for x in actual),
                                        "inputs": tuple(x.cpu() for x in engine.grounding_stages[1].inputs)},
                                       args.output.with_suffix(".failure.pt"))
                            print(json.dumps(report["copy_failure"]), flush=True)
                        assert all(exact_heads), f"Copied packed head differs from Israel: {i}, {j}"
                        original = tuple(v.to(engine.device) for v in source["original_retained"][i, j])
                        row["original_head_drift"] = validate_detections(torch, original, actual, .5)
                        row["source_heads_exact"] = exact_heads
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
                    if packed:
                        old_scores, _ = unpack(original)
                        assert torch.equal(old_scores > .5, ac > .5), f"Detection set changed: {i}, {j}"
                    near, retained = (es > .47) | (ac > .47), (es > .5) | (ac > .5)
                    row["confidence_passed"] = bool(((es-ac).abs()[near] <= .03).all())
                    row["boxes_passed"] = bool(((eb-ab).abs()[retained] <= .01).all())
                    report["checks"].append(row)
                print(json.dumps(report["frames"][-1]), flush=True)
            report["failures"] = {k: sum(not row[f"{k}_passed"] for row in report["checks"])
                                  for k in ("confidence", "boxes")}
            # Do not activate a transfer that differs from the reported candidate.
            assert report["failures"] == {"confidence": 5, "boxes": 4}, report["failures"]
            if packed:
                source_report = json.loads((root / "packed_heads_rounded_split512_gpu0.json").read_text())
                for gate in ("confidence", "boxes"):
                    old = {(row["frame_index"], row["prompt"]) for row in source_report["original_retained_checks"]
                           if not row[gate + "_pass"]}
                    new = {(row["frame"], row["prompt"]) for row in report["checks"] if not row[gate + "_passed"]}
                    assert new == old, f"Failure identities changed: {gate} {old} -> {new}"
                report["no_incremental_development_failures"] = True
                report["packed_head_proof"] = engine.grounding_stages[1].direct_replay_passed
            times = []
            for _ in range(20):
                started = time.perf_counter()
                engine.image_stage(pixels)
                torch.xpu.synchronize()
                times.append((time.perf_counter()-started)*1000)
            report["image_replay_ms"] = statistics.median(times)
            report["copy_parity_passed"] = True
            report["image_proof"] = engine.image_stage.proof
            if packed and args.images:
                original_stage = CompiledStage(torch, engine.head, "original-head-control", engine._progress)
                report["camera_checks"] = []
                prompts = ["face", "person", "chair", "bottle"]
                image_paths = sorted(args.images.glob("*.jpg"))
                assert len(image_paths) >= 3
                for path in image_paths:
                    pixels = engine._pixels(path.read_bytes())
                    features = engine.image_stage(pixels)
                    embeddings = engine._encode_prompts(prompts)
                    actual = engine._run_heads(features, embeddings)
                    for j, prompt in enumerate(prompts):
                        expected = tuple(x.clone() for x in original_stage(*features, *embeddings[j]))
                        observed = tuple(x[j:j+1] for x in actual)
                        row = {"image": path.name, "prompt": prompt,
                               "old_head_drift": validate_detections(torch, expected, observed, .5)}
                        es, _ = unpack(expected)
                        ac, _ = unpack(observed)
                        assert torch.equal(es > .5, ac > .5), f"Live detection set changed: {row}"
                        dense = engine.wrapper(pixels, engine._tokens(prompt))
                        ds, db = unpack(dense)
                        def gates(values):
                            scores, boxes = unpack(values)
                            near, keep = (ds > .47) | (scores > .47), (ds > .5) | (scores > .5)
                            return [bool(((ds-scores).abs()[near] <= .03).all()),
                                    bool(((db-boxes).abs()[keep] <= .01).all())]
                        old_gates, new_gates = gates(expected), gates(observed)
                        assert all(not old or new for old, new in zip(old_gates, new_gates, strict=True)), row
                        row.update(original_dense_gates=old_gates, packed_dense_gates=new_gates,
                                   detection_count=int((ac > .5).sum()))
                        if baseline is not None:
                            match = [r for r in baseline["rows"] if r["prompt"] == prompt and
                                     r["image_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()]
                            assert len(match) == 1
                            prior = tuple(x.to(engine.device) for x in match[0]["actual"])
                            row["deployed_baseline_drift"] = validate_detections(torch, prior, observed, .5)
                            prior_scores, _ = unpack(prior)
                            assert torch.equal(prior_scores > .5, ac > .5), row
                            # Compare both models against the SAME original dense
                            # outputs captured with the prior runtime/driver.
                            ds, db = unpack(tuple(x.to(engine.device) for x in match[0]["dense"]))
                            prior_gates, observed_gates = gates(prior), gates(observed)
                            assert all(not old or new for old, new in zip(prior_gates, observed_gates, strict=True)), row
                            row.update(deployed_baseline_dense_gates=prior_gates,
                                       candidate_vs_baseline_dense_gates=observed_gates)
                        report["camera_checks"].append(row)
                    print(json.dumps({"camera": path.name, "checked_prompts": len(prompts)}), flush=True)
                report["no_incremental_camera_failures"] = True
                report["deployed_baseline_checked"] = baseline is not None
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
