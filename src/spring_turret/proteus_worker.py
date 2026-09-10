#!/usr/bin/env python3
"""Live adapters for the Proteus experiments. No camera/servo device access."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spring_turret.models import model_prompts
from spring_turret.prompts import COLORS
from spring_turret.proteus_models import configure, build_seeded_tracker, build_oob, build_seed, require
from spring_turret.track_lifetime import TrackLifetime
from spring_turret.tracking_masks import encode_mask_overlay, instance_color, mask_centroid
from spring_turret.worker_protocol import JPEG_BYTES, iter_requests, encode_request

WARNINGS = {
    'efficient-nomem': 'Experimental: memory off; Proteus basketball test lost the mask after initialization',
    'hybrid-nomem': 'Experimental: memory off; source uses zero-threshold initialization and produced incorrect masks',
    'efficient-tracking': 'Experimental: memory on; Proteus basketball test produced no detections',
    'efficient-memory': 'Experimental: seven memory slots; promising basketball result, broader accuracy unqualified',
}


def memory_frames(model, session, enabled):
    """Check actual encoded state, not merely a configuration flag."""
    require(model.num_maskmem == (7 if enabled else 0), 'Experimental memory recipe changed')
    outputs = [out for store in session['output_dict'].values() for out in store.values()]
    encoded = [out for out in outputs if out.get('maskmem_features') is not None]
    if enabled:
        require(bool(encoded) and all(out.get('maskmem_pos_enc') is not None for out in encoded),
                'Efficient Memory failed to produce temporal memory')
    else:
        require(not encoded, 'NoMem worker produced temporal memory')
    return len(encoded)


class CurrentRawFrame:
    """Only the current camera frame is accessible, never offline/future frames."""
    def __init__(self, image, index):
        self.image, self.index, self.references = image, index, {}

    def __len__(self):
        return self.index + 1

    def __getitem__(self, index):
        if index != self.index:
            raise IndexError('Experimental tracker requested an unavailable camera frame')
        return self.image


class SeedClient:
    """Different upstream SAM packages must not share a Python interpreter."""
    def __init__(self, args, progress):
        from spring_turret.detection import WorkerClient
        runtime = args.bundle / 'proteus-composed-b580-20260907/private-host/usr'
        command = [str(runtime / 'lib/x86_64-linux-gnu/ld-linux-x86-64.so.2'),
            '--library-path', os.environ['LD_LIBRARY_PATH'], str(runtime / 'bin/python3.12'),
            '-u', __file__, '--bundle', str(args.bundle), '--checkpoint', args.checkpoint,
            '--profile', args.profile, '--seed-only']
        if args.memory_bundle is not None:
            command.extend(('--memory-bundle', str(args.memory_bundle)))
        self.client = WorkerClient({'model': args.profile})
        # Same process group as the outer worker: a forced worker replacement
        # must also reap its initializer instead of leaving a GPU process behind.
        self.client.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.client.receive(900, progress)
        self.sequence = 0

    def detect(self, jpeg, prompt, progress):
        self.sequence += 1
        return self.client.detect(self.sequence, jpeg, [prompt], progress).get('seed')

    def close(self):
        proc = self.client.process
        if proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.stdout:
            proc.stdout.close()


def seed_main(args, emit):
    torch, device = configure(args.bundle, args.profile, seed=True, memory_bundle=args.memory_bundle)
    processor = build_seed(args.bundle, args.profile, args.checkpoint, torch, device)
    emit({'type':'ready', 'request_transport':JPEG_BYTES})
    from PIL import Image
    for request in iter_requests(sys.stdin.buffer):
        prompt, = model_prompts(args.profile, request['prompts'])
        with Image.open(io.BytesIO(request['jpeg'])) as encoded:
            image = encoded.convert('RGB')
        with torch.inference_mode(), torch.autocast('xpu', dtype=torch.bfloat16):
            state = processor.set_image(image)
            state = processor.set_text_prompt(prompt, state)
        torch.xpu.synchronize()
        seed = None
        if state['scores'].numel():
            best = int(state['scores'].argmax())
            score = float(state['scores'][best].float().cpu())
            mask = state['masks'][best, 0].bool().cpu().numpy()
            if mask.any():
                stream = io.BytesIO()
                Image.fromarray(mask).save(stream, format='PNG')
                seed = {'score':score, 'png':base64.b64encode(stream.getvalue()).decode()}
        emit({'type':'result', 'id':request['id'], 'seed':seed})


class SeededTrackerEngine:
    def __init__(self, args, progress):
        self.args, self.progress = args, progress
        self.memory = args.profile == 'efficient-memory'
        self.torch, self.device = configure(args.bundle, args.profile, memory_bundle=args.memory_bundle)
        self.seed = SeedClient(args, progress)
        self.model, self.audit = None, {}
        self.session, self.key = None, None
        self.index, self.next_id, self.identity = 0, 1, None
        self.lifetime = TrackLifetime()
        self.last_timestamp = None
        self.last_duration = 0.

    def detect(self, request):
        import numpy as np
        from PIL import Image
        from spring_turret.sam31_tracking import prune_tracker_state
        torch = self.torch
        started = time.perf_counter()
        prompts = model_prompts(self.args.profile, request['prompts'])
        require(len(prompts) == 1, 'Enter one object prompt for this experimental profile')
        with Image.open(io.BytesIO(request['jpeg'])) as encoded:
            image = encoded.convert('RGB')
            width, height = image.size
            raw = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).contiguous()
        key = (tuple(prompts), width, height, request.get('session_revision'), request.get('camera_identity'))
        timestamp = request.get('captured_at')
        now = time.monotonic() if timestamp is None else timestamp
        gap = 0 if self.last_timestamp is None else now - self.last_timestamp
        reset = key != self.key or gap < 0 or gap > max(2., self.last_duration + 2.)
        retired = []
        if reset:
            if self.identity is not None:
                retired.append(self.identity)
            self.session, self.identity, self.index = None, None, 0
            self.lifetime = TrackLifetime()
            # Shape-specific resize kernels must not survive a resolution change.
            if self.key and self.key[1:3] != key[1:3]:
                self.model, self.audit = None, {}
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
            self.key = key
        self.last_timestamp = now
        prepared = time.perf_counter()
        mask, score, seeded, retained_memory = None, 0., False, 0
        with torch.inference_mode(), torch.autocast('xpu', dtype=torch.bfloat16):
            if self.session is None:
                seed = self.seed.detect(request['jpeg'], prompts[0], self.progress)
                if seed:
                    with Image.open(io.BytesIO(base64.b64decode(seed['png'], validate=True))) as png:
                        mask = np.array(png, dtype=np.bool_)
                    require(mask.shape == (height, width), 'Initializer mask does not match the current camera frame')
                    if self.model is None:
                        self.progress('loading_experimental_tracker')
                        self.model, self.audit = build_seeded_tracker(self.args.bundle, self.args.profile,
                            self.args.checkpoint, raw, torch, self.device)
                    self.index = 0
                    # Stable pointer-time normalization from the first live frame;
                    # the source clip also had more than max_obj_ptrs frames.
                    horizon = self.model.max_obj_ptrs_in_encoder if self.memory else 1
                    self.session = self.model.init_state(height, width, horizon)
                    self.session['images'] = CurrentRawFrame(raw, 0)
                    if self.memory:
                        self.model.add_new_mask(self.session, 0, 1, torch.from_numpy(mask))
                    else:
                        self.model.add_new_masks(self.session, 0, [1], torch.from_numpy(mask[None]))
                    self.model.propagate_in_video_preflight(self.session, run_mem_encoder=True)
                    self.identity, self.next_id = self.next_id, self.next_id + 1
                    score, seeded = seed['score'], True
            else:
                self.session['images'] = CurrentRawFrame(raw, self.index)
                self.session['num_frames'] = max(self.index + 1,
                    self.model.max_obj_ptrs_in_encoder if self.memory else 1)
                result = next(self.model.propagate_in_video(self.session, start_frame_idx=self.index,
                    max_frame_num_to_track=0, reverse=False, tqdm_disable=True, run_mem_encoder=True))
                frame, ids, _, masks, scores = result
                require(frame == self.index and list(map(int, ids)) == [1], 'Experimental object identity changed')
                require(bool(torch.isfinite(masks).all()) and bool(torch.isfinite(scores).all()), 'Nonfinite experimental masks')
                mask = (masks[0, 0] > 0).cpu().numpy()
                score = float(scores[0].float().sigmoid().cpu())
                prune_tracker_state(self.session, self.index, keep_first=True)
                audit = self.audit.get('feature_request_audit', {})
                for values in audit.values():
                    del values[:-32]
            if self.session is not None:
                retained_memory = memory_frames(self.model, self.session, self.memory)
            torch.xpu.synchronize()
        self.index += 1
        tracked = time.perf_counter()
        boxes, overlays = [], []
        # Preserve the source's bad initializer result as a diagnostic, but never
        # assign a fake high confidence to it or publish empty/stale masks.
        visible = mask is not None and bool(mask.any()) and score >= self.args.confidence
        if visible:
            ys, xs = np.nonzero(mask)
            color = instance_color(COLORS[0], self.identity)
            boxes.append({'xyxy':[float(xs.min()/width), float(ys.min()/height),
                float((xs.max()+1)/width), float((ys.max()+1)/height)], 'score':score,
                'prompt':prompts[0], 'prompt_index':0, 'color':color,
                'instance_id':self.identity, 'mask_centroid':mask_centroid(mask)})
            overlays.append((mask, color, self.identity))
        active = [] if self.identity is None else [self.identity]
        expired = self.lifetime.update(active, active if visible else [], self.index, now)
        if expired:
            retired.extend(expired)
            self.session, self.identity = None, None
            active = []
            retained_memory = 0
        mask_overlay = encode_mask_overlay(overlays, width, height)
        done = time.perf_counter()
        self.last_duration = done - started
        return {'boxes':boxes, 'categories':[{'prompt':prompts[0], 'color':COLORS[0], 'count':len(boxes)}],
            'client_overlay':True, 'mask_detection':True, 'mask_overlay':mask_overlay,
            'temporal_tracking':self.memory, 'memory_enabled':self.memory, 'memory_frames':retained_memory,
            'memory_slots':7 if self.memory else 0,
            'active_instance_ids':active, 'retired_instance_ids':retired,
            'propagated_instance_ids':active if not seeded and self.session is not None else [],
            'tracking_backend':'proteus-' + self.args.profile, 'tracking_frame':self.index,
            'tracking_reset':reset, 'tracking_phase':'acquiring' if self.session is None else 'propagating',
            'experimental_profile':self.args.profile, 'warning':WARNINGS[self.args.profile],
            'seed_score':score if seeded else None, 'max_objects_per_prompt':1,
            'torch':torch.__version__, 'device':torch.xpu.get_device_name(0), 'precision':'W8A8 + BF16',
            'torch_compile':False, 'sycl_graph':self.model is not None, 'cuda_graph':False,
            'compilation_scope':'TinyViT backbone graph; native W8A8 kernels; eager decoder',
            'latency_ms':round((tracked-prepared)*1000),
            'timing':{'preprocess_ms':round((prepared-started)*1000, 2),
                'model_ms':round((tracked-prepared)*1000, 2), 'mask_overlay_ms':round((done-tracked)*1000, 2),
                'worker_total_ms':round((done-started)*1000, 2)}}

    def close(self):
        self.seed.close()


def make_oob_engine(args, progress):
    from spring_turret.sam31_tracking_worker import TrackingEngine
    from spring_turret.sam31_tracking import OnlineSession
    from spring_turret.tracking_preprocess import VideoPreprocessor

    class EfficientEngine(TrackingEngine):
        def __init__(self):
            self.torch, self.device = configure(args.bundle, args.profile)
            self.accelerator = self.torch.xpu
            self.model = build_oob(args.bundle, self.torch, self.device)
            # Keep the original neural execution eager. Only the shared input
            # normalizer is compiled; status must not claim a compiled model.
            self.graph_stages, self.sources = {}, {}
            self.session_factory = OnlineSession
            self.confidence = args.confidence
            self.sessions, self.ids = [], {}
            self.next_id, self.session_key = 1, None
            self.last_captured_at, self.previous_duration = None, 0.
            self.preprocess = VideoPreprocessor(self.torch, self.device, progress)
            self.device_name = self.accelerator.get_device_name(0)
            self.source_digest, self.source_count = None, -1

        def after_detect(self, result, token):
            result.update(experimental_profile=args.profile, warning=WARNINGS[args.profile],
                tracking_backend='proteus-' + args.profile, image_backend='efficientsam31-tinyvit-mobileclip',
                memory_enabled=True, torch_compile=False, sycl_graph=False,
                compilation_scope='eager model; compiled input normalization only', mask_detection=True)
            return result

        def close(self):
            pass

    return EfficientEngine()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--memory-bundle', type=Path)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--profile', choices=tuple(WARNINGS), required=True)
    parser.add_argument('--confidence', type=float, default=.5)
    parser.add_argument('--seed-only', action='store_true')
    args = parser.parse_args()
    if args.profile == 'efficient-memory' and args.memory_bundle is None:
        parser.error('Efficient Memory requires --memory-bundle')
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    def emit(value):
        print(json.dumps(value, separators=(',', ':'), allow_nan=False), file=protocol, flush=True)
    progress = lambda stage: emit({'type':'progress', 'stage':stage})
    engine = None
    try:
        progress('loading_experimental_model')
        if args.seed_only:
            seed_main(args, emit)
            return 0
        engine = (make_oob_engine(args, progress) if args.profile == 'efficient-tracking'
                  else SeededTrackerEngine(args, progress))
        emit({'type':'ready', 'request_transport':JPEG_BYTES})
        for request in iter_requests(sys.stdin.buffer):
            emit({'type':'result', 'id':request['id'], **engine.detect(request)})
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit({'type':'fatal', 'error':str(error)})
        return 1
    finally:
        if engine is not None:
            engine.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
