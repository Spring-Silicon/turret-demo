#!/usr/bin/env python3
"""Dense BF16 SAM3.1 temporal tracking worker. No servo access."""
from __future__ import annotations

import argparse
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
    from .session_policy import ManagedSession
    from .sam31_tracking_graph import install_tracking_graphs, tracking_graph_status
    from .tracking_masks import encode_mask_overlay, instance_color, mask_centroid
    from .tracking_preprocess import VideoPreprocessor
    from .worker_protocol import JPEG_BYTES, iter_requests
    from .prefetch import LatestPreparation, RequestInbox
else:
    from prompts import COLORS, normalize_prompts
    from sam31_tracking import OnlineSession, build_model, configure_source
    from session_policy import ManagedSession
    from sam31_tracking_graph import install_tracking_graphs, tracking_graph_status
    from tracking_masks import encode_mask_overlay, instance_color, mask_centroid
    from tracking_preprocess import VideoPreprocessor
    from worker_protocol import JPEG_BYTES, iter_requests
    from prefetch import LatestPreparation, RequestInbox


class TrackingEngine:
    def __init__(self, args, *, compile_stages=True, progress=None):
        self.sources = configure_source(Path(args.source_bundle), args.device_type)
        import torch
        self.torch = torch
        torch.set_num_threads(4)
        self.accelerator = getattr(torch, args.device_type)
        self.accelerator.set_device(args.device)
        self.device = torch.device(args.device_type, args.device)
        self.model = build_model(torch, Path(args.checkpoint), self.device)
        self.graph_stages = (install_tracking_graphs(torch, self.model, self.device.type, progress=progress, layout="regions")
                             if compile_stages else {})
        self.session_factory = OnlineSession
        self.confidence = args.confidence
        self.sessions, self.ids = [], {}
        self.next_id, self.session_key = 1, None
        self.last_captured_at = None
        self.previous_duration = 0.0
        self.preprocess = VideoPreprocessor(torch, self.device, progress)
        self.device_name = self.accelerator.get_device_name(self.device)
        self.source_digest = None
        self.source_count = -1

    def before_detect(self):
        return None

    def after_detect(self, result, token):
        return result

    def detect(self, request):
        torch = self.torch
        start = time.perf_counter()
        token = self.before_detect()
        prompts = normalize_prompts(request["prompts"])
        pixels, (width, height) = self.preprocess(request["jpeg"], prepared=request.get('_prepared_pixels'))
        key = (tuple(prompts), width, height, request.get("session_revision"), request.get("camera_identity"))
        timestamp = request.get("captured_at")
        gap = (timestamp - self.last_captured_at if timestamp is not None and self.last_captured_at is not None else 0)
        reset = key != self.session_key or gap < 0 or gap > max(2, self.previous_duration + 2)
        self.last_captured_at = timestamp
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            if reset:
                self.sessions, self.ids = [], {}
                self.sessions = [ManagedSession(self.session_factory(self.model, pixels, width, height, p),
                    confidence=self.confidence) for p in prompts]
                self.session_key = key
            prepared = time.perf_counter()
            boxes, active_ids, propagated_ids, categories, retired_ids = [], [], [], [], []
            display_masks = []
            memory_frames = dropped_objects = 0
            for prompt_index, (prompt, session) in enumerate(zip(prompts, self.sessions, strict=True)):
                result = session.step(pixels, timestamp=timestamp)
                retired_ids.extend(self.ids[(prompt_index, i)] for i in result["retired_ids"] if (prompt_index, i) in self.ids)
                dropped_objects += result["dropped_objects"]
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
                                  "color":color, "instance_id":instance_id,
                                  "mask_centroid":mask_centroid(mask)})
                    display_masks.append((mask, color, instance_id))
                    count += 1
                categories.append({"prompt":prompt, "color":COLORS[prompt_index], "count":count})
                memory_frames += result["memory_frames"]
                for pair in list(self.ids):
                    if pair[0] == prompt_index and pair[1] not in active:
                        del self.ids[pair]
            self.accelerator.synchronize(self.device)
        tracked = time.perf_counter()
        mask_overlay = encode_mask_overlay(display_masks, width, height)
        done = time.perf_counter()
        self.previous_duration = done - start
        if self.source_count != len(self.sources):
            self.source_count = len(self.sources)
            self.source_digest = hashlib.sha256(json.dumps(self.sources, sort_keys=True).encode()).hexdigest()
        result = {
            "boxes":boxes, "categories":categories, "client_overlay":True,
            "mask_overlay":mask_overlay,
            "latency_ms":round((tracked-prepared)*1000),
            "torch":torch.__version__, "device":self.device_name,
            "device_type":self.device.type,
            "image_backend":"sam31-dense-bf16-temporal", "precision":"bfloat16",
            **tracking_graph_status(self.graph_stages, self.device.type),
            "tracking_source_sha256":self.source_digest,
            "temporal_tracking":True, "tracking_backend":"sam31-object-multiplex",
            "tracking_frame":self.sessions[0].index if self.sessions else 0,
            "tracking_reset":reset, "active_instance_ids":active_ids,
            "propagated_instance_ids":propagated_ids, "memory_frames":memory_frames,
            "retired_instance_ids":retired_ids, "dropped_objects":dropped_objects,
            "track_retention":{"seconds":5, "minimum_missing_frames":16},
            "preprocess_bitwise_equal":self.preprocess.validated,
            "preprocess_cache_hit":request.get('_prepared_pixels') is not None,
            "max_objects_per_prompt":16,
            "timing":{"preprocess_ms":round((prepared-start)*1000, 2),
                      "tracking_ms":round((tracked-prepared)*1000, 2),
                      "mask_overlay_ms":round((done-tracked)*1000, 2),
                      "worker_total_ms":round((done-start)*1000, 2)},
        }
        return self.after_detect(result, token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--device-type", choices=("xpu", "cuda"), default="xpu")
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
    preparation = None
    try:
        emit({"type":"progress", "stage":"loading_tracker"})
        engine = TrackingEngine(args, progress=lambda stage: emit({"type":"progress", "stage":stage}))
        preparation = LatestPreparation(engine.preprocess.prepare_cpu)
        emit({"type":"ready", "request_transport":JPEG_BYTES, "cpu_prefetch":True})
        for request in RequestInbox(sys.stdin.buffer, preparation):
            request['_prepared_pixels'] = preparation.take(request.get('prepared_token'), request['jpeg'])
            result = engine.detect(request)
            emit({"type":"result", "id":request["id"], **result})
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit({"type":"fatal", "error":str(error)})
        return 1
    finally:
        if preparation is not None:
            preparation.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
