#!/usr/bin/env python3
"""Real XPU integration check. Run ONLY after unloading other GPU model workers."""
import argparse
import io
import json
from pathlib import Path
import statistics
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.detection import WorkerClient
from PIL import Image, ImageOps

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--cache-dir", required=True)
parser.add_argument("--image", type=Path, required=True, help="Official Ultralytics bus.jpg")
parser.add_argument("--camera-url", help="Optional live /stream.mjpg URL")
args = parser.parse_args()


def camera_frame():
    with urllib.request.urlopen(args.camera_url, timeout=5) as stream:
        data = b""
        while len(data) < 2**22:
            data += stream.read(4096)
            start, end = data.find(b"\xff\xd8"), data.find(b"\xff\xd9")
            if 0 <= start < end:
                return data[start:end+2]
        raise RuntimeError("No complete camera JPEG")


worker = WorkerClient({"python": sys.executable, "checkpoint": args.checkpoint,
                       "cache_dir": args.cache_dir, "model": "yolo26x"})
try:
    worker.launch()
    assert worker.receive(120)["type"] == "ready"
    original = args.image.read_bytes()
    flipped = io.BytesIO()
    ImageOps.mirror(Image.open(io.BytesIO(original))).save(flipped, "JPEG")
    metrics = []
    for index in range(15):
        jpeg = original if index == 0 else flipped.getvalue() if index == 1 else camera_frame() if args.camera_url else original
        result = worker.detect(index, jpeg, ["person", "bus", "chair", "laptop", "cup", "bottle"],
                               lambda stage: print(stage, flush=True))
        assert result["torch_compile"] and result["sycl_graph"]
        assert result["graph_replays"] == index+1
        assert all(0 <= v <= 1 for b in result["boxes"] for v in b["xyxy"])
        if index < 2:
            counts = {c["prompt"]: c["count"] for c in result["categories"]}
            assert counts["person"] >= 2 and counts["bus"] >= 1, counts
        result.pop("jpeg")
        print(json.dumps({"frame": index, **result}), flush=True)
        if index >= 5:
            metrics.append(result["timing"])
    print(json.dumps({"warm_medians_ms": {key: statistics.median(m[key] for m in metrics)
                                         for key in metrics[0]}}), flush=True)
finally:
    worker.stop()
