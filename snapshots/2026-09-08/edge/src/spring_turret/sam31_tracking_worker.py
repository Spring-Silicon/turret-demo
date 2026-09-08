#!/usr/bin/env python3
"""Dense BF16 SAM3.1 temporal tracking worker. No servo access."""
from __future__ import annotations

import argparse
import io
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

if __package__:
    from .prompts import COLORS, normalize_prompts
    from .sam31_tracking import OnlineSession, build_model, configure_source
    from .sam31_tracking_graph import install_tracking_graphs, tracking_graph_status
    from .tracking_masks import encode_mask_overlay, instance_color
    from .worker_protocol import JPEG_BYTES, iter_requests
else:
    from prompts import COLORS, normalize_prompts
    from sam31_tracking import OnlineSession, build_model, configure_source
    from sam31_tracking_graph import install_tracking_graphs, tracking_graph_status
    from tracking_masks import encode_mask_overlay, instance_color
    from worker_protocol import JPEG_BYTES, iter_requests


class TrackingEngine:
    def __init__(self, args, *, compile_stages=True, progress=None):
        self.sources = configure_source(Path(args.source_bundle))
        import torch
        self.torch = torch
        torch.set_num_threads(4)
        torch.xpu.set_device(args.device)
        self.device = torch.device("xpu", args.device)
        self.model = build_model(torch, Path(args.checkpoint))
        self.graph_stages = (install_tracking_graphs(torch, self.model, "xpu", progress=progress, layout="regions")
                             if compile_stages else {})
        self.confidence = args.confidence
        self.sessions, self.ids = [], {}
        self.next_id, self.session_key = 1, None
        self.last_captured_at = None
        self.previous_duration = 0.0

    def detect(self, request):
        from PIL import Image
        from torchvision.transforms import functional as TF
        torch = self.torch
        start = time.perf_counter()
        prompts = normalize_prompts(request["prompts"])
        image = Image.open(io.BytesIO(request["jpeg"])).convert("RGB")
        width, height = image.size
        # Exactly the official video JPEG preprocessing: whole frame, bilinear
        # PIL resize, uint8->float32 /255, then mean=.5 / std=.5.
        pixels = TF.to_tensor(TF.resize(image, [1008, 1008]))
        pixels = ((pixels - .5) / .5).to(self.device)
        key = (tuple(prompts), width, height, request.get("session_revision"), request.get("camera_identity"))
        timestamp = request.get("captured_at")
        gap = (timestamp - self.last_captured_at if timestamp is not None and self.last_captured_at is not None else 0)
        reset = key != self.session_key or gap < 0 or gap > max(2, self.previous_duration + 2)
        self.last_captured_at = timestamp
        with torch.inference_mode(), torch.autocast("xpu", dtype=torch.bfloat16):
            if reset:
                self.sessions, self.ids = [], {}
                self.sessions = [OnlineSession(self.model, pixels, width, height, p) for p in prompts]
                self.session_key = key
            prepared = time.perf_counter()
            boxes, active_ids, propagated_ids, categories = [], [], [], []
            display_masks = []
            memory_frames = 0
            for prompt_index, (prompt, session) in enumerate(zip(prompts, self.sessions, strict=True)):
                result = session.step(pixels)
                active = set(result["active_ids"])
                for native_id in active:
                    pair = (prompt_index, native_id)
                    if pair not in self.ids:
                        self.ids[pair] = self.next_id
                        self.next_id += 1
                    active_ids.append(self.ids[pair])
                propagated_ids.extend(self.ids[(prompt_index, i)] for i in result["propagated_ids"] if i in active)
                count = 0
                for native_id, score, xywh, mask in zip(result["out_obj_ids"], result["out_probs"], result["out_boxes_xywh"], result["out_binary_masks"], strict=True):
                    if float(score) < self.confidence:
                        continue
                    x, y, w, h = map(float, xywh)
                    if w <= 0 or h <= 0:
                        continue
                    color = instance_color(COLORS[prompt_index], int(native_id))
                    instance_id = self.ids[(prompt_index, int(native_id))]
                    boxes.append({"xyxy":[x, y, min(1., x+w), min(1., y+h)],
                                  "score":float(score), "prompt":prompt, "prompt_index":prompt_index,
                                  "color":color, "instance_id":instance_id})
                    display_masks.append((mask, color, instance_id))
                    count += 1
                categories.append({"prompt":prompt, "color":COLORS[prompt_index], "count":count})
                memory_frames += result["memory_frames"]
                for pair in list(self.ids):
                    if pair[0] == prompt_index and pair[1] not in active:
                        del self.ids[pair]
            torch.xpu.synchronize()
        tracked = time.perf_counter()
        mask_overlay = encode_mask_overlay(display_masks, width, height)
        done = time.perf_counter()
        self.previous_duration = done - start
        return {
            "boxes":boxes, "categories":categories, "client_overlay":True,
            "mask_overlay":mask_overlay,
            "latency_ms":round((tracked-prepared)*1000),
            "torch":torch.__version__, "device":torch.xpu.get_device_name(self.device),
            "image_backend":"sam31-dense-bf16-temporal", "precision":"bfloat16",
            **tracking_graph_status(self.graph_stages, "xpu"),
            "tracking_source_sha256":hashlib.sha256(json.dumps(self.sources, sort_keys=True).encode()).hexdigest(),
            "temporal_tracking":True, "tracking_backend":"sam31-object-multiplex",
            "tracking_frame":self.sessions[0].index if self.sessions else 0,
            "tracking_reset":reset, "active_instance_ids":active_ids,
            "propagated_instance_ids":propagated_ids, "memory_frames":memory_frames,
            "max_objects_per_prompt":16,
            "timing":{"preprocess_ms":round((prepared-start)*1000, 2),
                      "tracking_ms":round((tracked-prepared)*1000, 2),
                      "mask_overlay_ms":round((done-tracked)*1000, 2),
                      "worker_total_ms":round((done-start)*1000, 2)},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--confidence", type=float, default=.5)
    args = parser.parse_args()
    class NmsWarningOnce(logging.Filter):
        seen = False
        def filter(self, record):
            if record.getMessage().startswith("Falling back to CPU mask NMS implementation"):
                if self.seen:
                    return False
                self.seen = True
            return True
    logging.getLogger().addFilter(NmsWarningOnce())
    # Upstream diagnostics and native libraries must not corrupt JSON framing.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    def emit(body):
        print(json.dumps(body, separators=(",", ":"), allow_nan=False), file=protocol, flush=True)
    try:
        emit({"type":"progress", "stage":"loading_tracker"})
        engine = TrackingEngine(args, progress=lambda stage: emit({"type":"progress", "stage":stage}))
        emit({"type":"ready", "request_transport":JPEG_BYTES, "cpu_prefetch":False})
        for request in iter_requests(sys.stdin.buffer):
            result = engine.detect(request)
            emit({"type":"result", "id":request["id"], **result})
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit({"type":"fatal", "error":str(error)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
