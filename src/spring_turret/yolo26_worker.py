#!/usr/bin/env python3
"""YOLO26x COCO detection: full-graph Inductor XPU + explicit SYCL replay."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

if __package__:
    from .models import COCO_CLASSES, model_prompts
    from .prompts import COLORS
    from .sam31_graph import CompiledStage
    from .sam31_worker import annotate
else:
    from models import COCO_CLASSES, model_prompts
    from prompts import COLORS
    from sam31_graph import CompiledStage
    from sam31_worker import annotate

CHECKPOINT_SHA256 = "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
_OUTPUT = None


def emit(payload):
    print(json.dumps(payload, separators=(",", ":")), file=_OUTPUT, flush=True)


def progress(stage, component="yolo26x"):
    print(f"{component}: {stage}", file=sys.stderr, flush=True)
    emit({"type": "progress", "stage": stage})


def decode(rows, prompts, width, height, resized, padding, confidence):
    """Undo centered 640px letterbox; preserve every retained requested instance."""
    boxes = []
    indices = {prompt: index for index, prompt in enumerate(prompts)}
    for x1, y1, x2, y2, score, class_id in rows:
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score, class_id)):
            raise RuntimeError("YOLO26x returned non-finite detections")
        if class_id != int(class_id) or not 0 <= class_id < len(COCO_CLASSES):
            raise RuntimeError("YOLO26x returned an invalid class ID")
        name = COCO_CLASSES[int(class_id)]
        if score <= confidence or name not in indices:
            continue
        coords = [(x1-padding[0])/resized[0], (y1-padding[1])/resized[1],
                  (x2-padding[0])/resized[0], (y2-padding[1])/resized[1]]
        coords = [max(0.0, min(1.0, value)) for value in coords]
        if coords[0] >= coords[2] or coords[1] >= coords[3]:
            continue
        index = indices[name]
        boxes.append({"xyxy": coords, "score": round(score, 4), "prompt": name,
                      "prompt_index": index, "color": COLORS[index]})
    return boxes


def validate_predictions(reference, actual, confidence):
    """Match retained detections, not top-k row order (ties may reorder)."""
    for rows in (reference, actual):
        if any(not all(math.isfinite(v) for v in row) for row in rows):
            raise RuntimeError("YOLO26x validation found non-finite outputs")
    max_box, max_score, matched = 0.0, 0.0, 0
    for expected, candidates in ((reference, actual), (actual, reference)):
        available = list(candidates)
        for row in expected:
            if row[4] < confidence - .03:
                continue
            matches = [(max(abs(a-b) for a, b in zip(row[:4], other[:4])), i, other)
                       for i, other in enumerate(available) if other[5] == row[5]]
            if not matches:
                raise RuntimeError("YOLO26x eager/compiled class mismatch")
            box_error, index, other = min(matches, key=lambda item: item[0])
            score_error = abs(row[4]-other[4])
            if box_error > 6.4 or score_error > .03:
                raise RuntimeError(f"YOLO26x eager/compiled mismatch: box={box_error:.3f}px score={score_error:.4f}")
            available.pop(index)
            max_box, max_score = max(max_box, box_error), max(max_score, score_error)
            matched += 1
    return {"matched": matched, "max_box_error_px": round(max_box, 4),
            "max_score_error": round(max_score, 6)}


class Yolo26Engine:
    def __init__(self, checkpoint, device_index=0, precision="float16", confidence=.5):
        # These checkpoints contain Python objects. Verify the official release
        # digest BEFORE Ultralytics unpickles anything; never fetch at runtime.
        with Path(checkpoint).open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError("Expected the official v8.4.0 yolo26x.pt checkpoint (SHA256 mismatch)")
        import torch
        import ultralytics
        from ultralytics.nn.tasks import load_checkpoint

        if ultralytics.__version__ != "8.4.140":
            raise RuntimeError("YOLO26x requires ultralytics==8.4.140")
        if not torch.xpu.is_available():
            raise RuntimeError("An Intel XPU is required; there is no CPU fallback")
        torch.xpu.set_device(device_index)
        self.torch, self.device = torch, device_index
        self.dtype = getattr(torch, precision)
        self.confidence = confidence
        model, checkpoint_data = load_checkpoint(str(checkpoint), device="cpu", fuse=True)
        del checkpoint_data
        if (model.task != "detect" or not model.end2end or
                tuple(model.names[i] for i in range(len(model.names))) != COCO_CLASSES):
            raise RuntimeError("Expected YOLO26x end-to-end COCO detection")
        model.requires_grad_(False).eval().to(f"xpu:{device_index}")
        head = model.model[-1]
        head.export, head.dynamic, head.format, head.max_det = True, False, "torch", 300

        class Wrapper(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, pixels):
                return (self.net(pixels),)

        self.module = Wrapper(model).eval()
        self.stage = None
        self.validation = []

    def pixels(self, jpeg):
        import cv2
        import numpy as np
        from PIL import Image

        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        width, height = image.size
        ratio = min(640 / width, 640 / height)
        resized = (round(width * ratio), round(height * ratio))
        padding = (round((640-resized[0])/2-.1), round((640-resized[1])/2-.1))
        rgb = cv2.resize(np.asarray(image), resized, interpolation=cv2.INTER_LINEAR)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[padding[1]:padding[1]+resized[1], padding[0]:padding[0]+resized[0]] = rgb
        pixels = self.torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0)
        pixels = pixels.to(f"xpu:{self.device}", dtype=self.torch.float32).div_(255).contiguous()
        return pixels, width, height, resized, padding

    def detect_many(self, jpeg, prompts):
        prompts = model_prompts("yolo26x", prompts)
        torch = self.torch
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("xpu", dtype=self.dtype):
            pixels, width, height, resized, padding = self.pixels(jpeg)
            torch.xpu.synchronize()
            prepared = time.perf_counter()
            if self.stage is None:
                # Initialize fixed-shape anchor caches outside Dynamo/capture.
                self.module(pixels)
                torch.xpu.synchronize()
                self.stage = CompiledStage(torch, self.module, "yolo26x", progress)
            infer_started = time.perf_counter()
            output = self.stage(pixels)[0]
            torch.xpu.synchronize()
            inferred = time.perf_counter()
            rows = output.float().cpu()[0].tolist()
            if len(rows) != 300 or any(len(row) != 6 for row in rows):
                raise RuntimeError("YOLO26x output must have shape [1,300,6]")
            decoded_at = time.perf_counter()
            if len(self.validation) < 3:
                progress("validating")
                reference = self.module(pixels)[0].float().cpu()[0].tolist()
                self.validation.append(validate_predictions(reference, rows, self.confidence))
            validated_at = time.perf_counter()
        boxes = decode(rows, prompts, width, height, resized, padding, self.confidence)
        decoded = time.perf_counter()
        annotated = annotate(jpeg, boxes)
        done = time.perf_counter()
        return {
            "boxes": boxes,
            "categories": [{"prompt": prompt, "color": COLORS[i],
                            "count": sum(box["prompt"] == prompt for box in boxes)}
                           for i, prompt in enumerate(prompts)],
            "latency_ms": round((inferred-infer_started)*1000, 2),
            "torch": torch.__version__, "device": torch.xpu.get_device_name(self.device),
            "torch_compile": True, "sycl_graph": self.stage.graph is not None,
            "sycl_graph_error": None, "validation": self.validation,
            "graph_replays": self.stage.calls, "input_size": [640, 640],
            "checkpoint_sha256": CHECKPOINT_SHA256, "max_detections": 300,
            "shared_image_features": True, "forward_passes": 1,
            "timing": {"preprocess_ms": round((prepared-started)*1000, 2),
                       "model_ms": round((inferred-infer_started)*1000, 2),
                       "validation_ms": round((validated_at-decoded_at)*1000, 2),
                       "postprocess_ms": round((decoded-validated_at+decoded_at-inferred)*1000, 2),
                       "annotation_ms": round((done-decoded)*1000, 2),
                       "worker_total_ms": round((done-started)*1000, 2)},
            "jpeg": base64.b64encode(annotated).decode(),
        }


def main():
    global _OUTPUT
    _OUTPUT = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--confidence", type=float, default=.5)
    args = parser.parse_args()
    try:
        engine = Yolo26Engine(args.checkpoint, args.device, args.precision, args.confidence)
        emit({"type": "ready", "engine": "yolo26x/torch.compile/inductor-xpu"})
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                result = engine.detect_many(base64.b64decode(request["jpeg"], validate=True), request["prompts"])
                emit({"type": "result", "id": request["id"], **result})
            except Exception as error:
                emit({"type": "error", "id": request.get("id"), "error": str(error)})
                raise
    except Exception as error:
        emit({"type": "fatal", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
