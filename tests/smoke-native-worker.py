#!/usr/bin/env python3
"""Full installed JSON worker smoke/latency test with a saved JPEG; no motors."""

import argparse
import json
import statistics
import time
from pathlib import Path

from spring_turret.detection import WorkerClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())["inference"]
    config["model"] = "sam3.1"
    worker = WorkerClient(config)
    report = {}
    try:
        worker.launch()
        ipc_dir = Path(worker.native_temp.name)
        assert worker.receive(120)["type"] == "ready"
        jpeg = args.image.read_bytes()
        request_id = 0
        for prompts in (["person"], ["person", "cup"], ["cup", "person"]):
            samples = []
            for iteration in range(7):
                request_id += 1
                start = time.perf_counter()
                result = worker.detect(request_id, jpeg, prompts, lambda stage: print(stage, flush=True))
                elapsed = (time.perf_counter() - start) * 1000
                assert result["image_backend"] == "graphs-native-sycl"
                assert result["native_image_validation"]["command_graph_count"] == 1
                assert result["torch_compile"] and result["sycl_graph"]
                assert [c["prompt"] for c in result["categories"]] == prompts
                assert sum(c["count"] for c in result["categories"]) == len(result["boxes"])
                if iteration:
                    assert result["text_cache_hit"]
                    samples.append({**result["timing"], "json_roundtrip_ms": elapsed})
            report[" + ".join(prompts)] = {
                "timing": {key: round(statistics.median(sample[key] for sample in samples), 2)
                           for key in samples[0]},
                "categories": result["categories"],
                "validation": result["validation"],
            }
    finally:
        worker.stop()
    assert not ipc_dir.exists(), "worker left shared-memory buffers behind"
    report["ipc_cleaned"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
