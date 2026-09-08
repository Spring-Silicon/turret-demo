#!/usr/bin/env python3
"""Pinned sleepy-joe W4A4 detector; one image pass and one head per text prompt."""
from __future__ import annotations

import argparse
import base64
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time

if __package__:
    from . import sam31_worker as common
    from .sam31_w4a4 import BACKEND, RECIPE, MANIFEST, build_stages
    from .sam31_graph import CompiledStage
    from .sam31_preprocess import ExactImagePreprocessor
    from .prefetch import LatestPreparation, RequestInbox
    from .worker_protocol import JPEG_BYTES
    from .prompts import COLORS, normalize_prompts
else:
    import sam31_worker as common
    from sam31_w4a4 import BACKEND, RECIPE, MANIFEST, build_stages
    from sam31_graph import CompiledStage
    from sam31_preprocess import ExactImagePreprocessor
    from prefetch import LatestPreparation, RequestInbox
    from worker_protocol import JPEG_BYTES
    from prompts import COLORS, normalize_prompts


class ModelGraph:
    """Compile components once, capture image and all requested heads together."""
    def __init__(self, torch, image, head, progress, stream=None):
        self.torch, self.progress = torch, progress
        self.stream = stream
        self.graph, self.calls = None, 0
        self.image, self.head = image, head
        self._compile(())

    def _compile(self, inputs):
        self.device_type, self.runtime = 'xpu', self.torch.xpu
        def direct(pixels, *text):
            features = self.image(pixels)
            if len(text) == 2:
                return self.head(*features, *text)
            parts = []
            for index in range(0, len(text), 2):
                parts.append(tuple(value.clone() for value in self.head(*features, *text[index:index + 2])))
            return tuple(self.torch.cat(values, dim=0) for values in zip(*parts, strict=True))
        self.compiled = direct

    def __call__(self, *inputs):
        torch, runtime = self.torch, self.runtime
        if self.graph is None:
            self.progress('compiling', 'w4a4-model')
            self.inputs = tuple(value.clone() for value in inputs)
            if self.stream is None:
                self.stream = runtime.Stream()
            runtime.synchronize()
            with runtime.stream(self.stream):
                for _ in range(2):
                    outputs = self.compiled(*self.inputs)
            runtime.synchronize()
            expected = tuple(value.clone() for value in outputs)
            self.progress('capturing', 'w4a4-model')
            graph = runtime.XPUGraph()
            with runtime.graph(graph, stream=self.stream):
                self.outputs = self.compiled(*self.inputs)
            graph.replay()
            runtime.synchronize()
            for reference, actual in zip(expected, self.outputs, strict=True):
                torch.testing.assert_close(actual, reference, rtol=.001, atol=.001)
                if not torch.isfinite(actual).all().item():
                    raise RuntimeError('Nonfinite W4A4 output')
            self.graph = graph
        else:
            if len(inputs) != len(self.inputs):
                raise ValueError('W4A4 graph input count changed')
            for source, target in zip(inputs, self.inputs, strict=True):
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise ValueError('W4A4 graph input shape/dtype changed')
                target.copy_(source)
            self.graph.replay()
        self.calls += 1
        return self.outputs


class W4A4Engine(common.Sam31Engine):
    def __init__(self, checkpoint, bundle, device_index=0, precision='float16', confidence=.5,
                 allow_accuracy_tradeoff=False):
        if not allow_accuracy_tradeoff:
            raise ValueError('This W4A4 candidate requires explicit acceptance of its measured accuracy tradeoff')
        if precision != 'float16':
            raise ValueError('W4A4 requires FP16 autocast with FP32 masters')
        import torch
        if not torch.xpu.is_available():
            raise RuntimeError('XPU is unavailable')
        torch.xpu.set_device(device_index)
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        self.torch, self.runtime = torch, torch.xpu
        self.device, self.device_type, self.dtype = torch.device(f'xpu:{device_index}'), 'xpu', torch.float16
        self.confidence, self.bundle = confidence, bundle
        self._progress('loading', RECIPE)
        image, text, head, self.tokenizer = build_stages(bundle, checkpoint, torch, self.device)
        options = {'emulate_precision_casts': True, 'triton.cudagraphs': False}
        self.compiled_image = torch.compile(image, fullgraph=True, dynamic=False, options=options)
        self.compiled_head = torch.compile(head, fullgraph=True, dynamic=False, options=options)
        # The retained AP/timing fixture caches eager text outputs. Compiling
        # this once-per-prompt step changes embeddings and downstream scores.
        # Image and grounding remain compiled and captured per frame.
        self.text_stage = text
        self.text_cache, self.model_stage, self.prompt_count = {}, None, None
        # Native contexts are keyed by queue. Reuse one queue across prompt-count
        # recaptures so changing categories cannot accumulate native workspaces.
        self.model_stream = self.runtime.Stream()
        self.preprocessor = ExactImagePreprocessor(torch, self.device, self._progress)
        self.validation = {'accuracy_policy': 'approved-measured-tradeoff',
                           'person_ap': 65.3444541679307, 'dense_person_ap': 66.08056295834682,
                           'images': 512, 'general_prompt_accuracy_qualified': False}
        self.start_event, self.end_event = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)

    def run_model(self, pixels, prompts):
        embeddings = self._encode_prompts(prompts)
        if self.prompt_count != len(prompts):
            # Only retain the active prompt-count graph, not an unbounded cache.
            self.model_stage = None
            gc.collect()
            self.model_stage = ModelGraph(self.torch, self.compiled_image, self.compiled_head,
                                          self._progress, self.model_stream)
            self.prompt_count = len(prompts)
        return self.model_stage(pixels, *(value for pair in embeddings for value in pair))

    def detect_many(self, jpeg, prompts, *, client_overlay=False, prepared_pixels=None):
        prompts = normalize_prompts(prompts)
        if not prompts:
            raise ValueError('At least one text prompt is required')
        torch = self.torch
        started = time.perf_counter()
        pixels = self.preprocessor(jpeg, prepared=prepared_pixels)
        self.runtime.synchronize()
        prepared = time.perf_counter()
        with torch.inference_mode(), torch.autocast('xpu', dtype=self.dtype):
            cached = all(p in self.text_cache for p in prompts)
            cold = self.model_stage is None or self.prompt_count != len(prompts) or not cached
            if cold:
                self.run_model(pixels, prompts)
                self.runtime.synchronize()
            model_started = time.perf_counter()
            self.start_event.record()
            outputs = self.run_model(pixels, prompts)
            self.end_event.record()
            self.runtime.synchronize()
            model_done = time.perf_counter()
            boxes = self._decode_outputs(outputs)
        detections, categories = [], []
        for index, (prompt, group) in enumerate(zip(prompts, boxes, strict=True)):
            color = COLORS[index]
            categories.append({'prompt': prompt, 'color': color, 'count': len(group)})
            detections.extend({**box, 'prompt': prompt, 'prompt_index': index, 'color': color} for box in group)
        result = {
            'boxes': detections, 'categories': categories,
            'latency_ms': round((time.perf_counter() - model_started) * 1000),
            'torch': torch.__version__, 'device': self.runtime.get_device_name(self.device), 'device_type': 'xpu',
            'image_backend': BACKEND, 'precision': 'W4A4-MLP/FP16-attention',
            'torch_compile': True, 'sycl_graph': self.model_stage.graph is not None, 'cuda_graph': False,
            'sycl_graph_error': None, 'accuracy_policy': 'approved-measured-tradeoff',
            'validation': self.validation, 'native_image_validation': {'source': 'sleepy-joe', 'recipe': RECIPE,
                'manifest_sha256': MANIFEST, 'direct_replay_passed': True, 'accuracy_parity': False},
            'shared_image_features': True, 'text_cache_hit': cached, 'grounding_batch_size': 1,
            'model_graph_count': 1, 'image_passes_per_frame': 1, 'head_passes_per_frame': len(prompts),
            'preprocess_validation': self.preprocessor.validation, 'preprocess_prefetched': prepared_pixels is not None,
            'client_overlay': client_overlay,
            'timing': {'preprocess_ms': round((prepared - started) * 1000, 2),
                       'cpu_prepare_ms': round(self.preprocessor.last_cpu_ms, 2),
                       'prompt_setup_ms': round((model_started - prepared) * 1000, 2),
                       'model_gpu_ms': round(self.start_event.elapsed_time(self.end_event), 2),
                       'model_wall_ms': round((model_done - model_started) * 1000, 2),
                       'postprocess_ms': round((time.perf_counter() - model_done) * 1000, 2)},
        }
        annotation_started = time.perf_counter()
        if not client_overlay:
            result['jpeg'] = base64.b64encode(common.annotate(jpeg, detections)).decode()
        result['timing']['annotation_ms'] = round((time.perf_counter() - annotation_started) * 1000, 2)
        result['timing']['worker_total_ms'] = round((time.perf_counter() - started) * 1000, 2)
        return result


def main():
    common._PROTOCOL_OUTPUT = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--w4a4-bundle', type=Path, required=True)
    parser.add_argument('--allow-w4a4-accuracy-tradeoff', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--device-type', choices=('xpu',), default='xpu')
    parser.add_argument('--precision', choices=('float16',), default='float16')
    parser.add_argument('--confidence', type=float, default=.5)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error('Confidence must be between zero and one')
    preparation, terminating = None, False
    def terminate(signum, frame):
        nonlocal terminating
        terminating = True
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    try:
        engine = W4A4Engine(args.checkpoint, args.w4a4_bundle, args.device, args.precision, args.confidence,
                           args.allow_w4a4_accuracy_tradeoff)
        common._emit({'type': 'ready', 'engine': BACKEND, 'torch_compile': False,
                      'sycl_graph_requested': True, 'request_transport': JPEG_BYTES, 'cpu_prefetch': True})
        preparation = LatestPreparation(engine.preprocessor.prepare_cpu)
        for request in RequestInbox(sys.stdin.buffer, preparation):
            try:
                jpeg = request['jpeg']
                result = engine.detect_many(jpeg, request['prompts'], client_overlay=request.get('client_overlay') is True,
                    prepared_pixels=preparation.take(request.get('prepared_token'), jpeg))
                common._emit({'type': 'result', 'id': int(request['id']), **result})
            except Exception as error:
                common._emit({'type': 'error', 'id': request.get('id'), 'error': str(error)})
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        common._emit({'type': 'fatal', 'error': str(error)})
        raise
    finally:
        if preparation:
            preparation.close()
        if terminating:
            os._exit(0)  # Inbox daemon can still be blocked on the stdin lock.


if __name__ == '__main__':
    main()
