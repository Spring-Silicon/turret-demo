#!/usr/bin/env python3
"""Per-frame SAM masks: native B580 or standard Torch/CUDA; no servo access."""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
from pathlib import Path
import sys
import time

if __package__:
    from . import sam31_worker as common
    from .sam31_preprocess import ExactImagePreprocessor
    from .prefetch import LatestPreparation, RequestInbox
    from .prompts import COLORS, normalize_prompts
    from .tracking_masks import encode_mask_overlay, instance_color, mask_centroid
    from .worker_protocol import JPEG_BYTES
    from .output_transfer import OutputTransfer
    from .mask_postprocess import packed_mask_output, centroid_from_moments, encode_labels, validate_packed
    from .sam31_graph import CompiledStage
else:
    import sam31_worker as common
    from sam31_preprocess import ExactImagePreprocessor
    from prefetch import LatestPreparation, RequestInbox
    from prompts import COLORS, normalize_prompts
    from tracking_masks import encode_mask_overlay, instance_color, mask_centroid
    from worker_protocol import JPEG_BYTES
    from output_transfer import OutputTransfer
    from mask_postprocess import packed_mask_output, centroid_from_moments, encode_labels, validate_packed
    from sam31_graph import CompiledStage


class MaskGraph:
    """One owned replay for image, all prompt heads, masks and fixed-slot output."""
    def __init__(self, engine, inputs):
        self.engine = engine
        torch, runtime = engine.torch, engine.runtime
        self.inputs = tuple(value.clone() for value in inputs)
        self.stream = engine.stream
        engine._progress('loading' if engine.artifacts else 'compiling', 'mask-model')
        runtime.synchronize()
        with runtime.stream(self.stream):
            for _ in range(2): outputs = self.direct(*self.inputs)
        runtime.synchronize()
        reference = tuple(value.clone() for value in outputs)
        engine._progress('capturing', 'mask-model')
        self.graph = runtime.XPUGraph() if engine.device_type == 'xpu' else runtime.CUDAGraph()
        with runtime.graph(self.graph, stream=self.stream):
            self.outputs = self.direct(*self.inputs)
        self.graph.replay()
        runtime.synchronize()
        self.validate(reference)

    def direct(self, pixels, *text):
        e = self.engine
        low, position, trunk = e.compiled['image'](pixels)
        high, middle = e.compiled['fpn'](trunk)
        outputs = []
        for i in range(0, len(text), 2):
            if e.device_type == 'xpu':
                detection = e.compiled['grounding'](low, position, *text[i:i+2])
                masks = e.compiled['masks'](high, middle, low, *detection[3:])
            else:
                detection = e.compiled['grounding'](low, position, high, middle, *text[i:i+2])
                masks = detection[3]
            selected = e.compiled['output'](detection[0], detection[2], masks)
            boxes = detection[1][0, selected[2].clamp_min(0)]
            # Own each prompt's outputs before invoking reused head workspaces.
            outputs.extend(value.clone() for value in (*selected, boxes))
        return tuple(outputs)

    def validate(self, reference):
        torch = self.engine.torch
        for expected, actual in zip(reference, self.outputs, strict=True):
            if actual.dtype.is_floating_point:
                if not torch.isfinite(actual).all().item(): raise RuntimeError('Nonfinite mask result')
                torch.testing.assert_close(actual, expected, rtol=.001, atol=.001)
            elif not torch.equal(actual, expected):
                raise RuntimeError('Mask replay changed discrete outputs')

    def __call__(self, *inputs):
        if len(inputs) != len(self.inputs): raise ValueError('Mask prompt count changed')
        for source, destination in zip(inputs, self.inputs, strict=True):
            if (source.shape,source.dtype,source.device) != (destination.shape,destination.dtype,destination.device):
                raise ValueError('Mask graph input contract changed')
            destination.copy_(source)
        self.graph.replay()
        return self.outputs


class MaskEngine(common.Sam31Engine):
    def __init__(self, args, *, stage_factory=None):
        import torch
        if __package__:
            from . import sam31_mask_layers as layers
        else:
            import sam31_mask_layers as layers
        # Thor's CPU resize benefits from eight threads; keep the Arc setting.
        # Thread-count qualification requires identical resized uint8 pixels.
        torch.set_num_threads(8 if args.device_type == 'cuda' else 4)
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        self.torch, self.device_type = torch, args.device_type
        self.runtime = getattr(torch, self.device_type)
        if not self.runtime.is_available(): raise RuntimeError(f'{self.device_type} unavailable')
        self.runtime.set_device(args.device)
        self.device = torch.device(self.device_type, args.device)
        self.dtype, self.confidence, self.capacity = torch.float16, args.confidence, 10
        self.graph, self.text_cache = None, {}
        self.image_optimization = None
        self.stream = self.runtime.Stream()
        self._progress('loading', 'mask-model')
        self.artifacts = None
        if getattr(args, 'compiled_bundle', None):
            if stage_factory is not None:
                raise ValueError('Cannot export and load compiled mask artifacts together')
            if __package__:
                from .sam31_mask_artifacts import MaskArtifactReader
            else:
                from sam31_mask_artifacts import MaskArtifactReader
            self.artifacts = MaskArtifactReader(torch, args)
            stage_factory = self.artifacts.stage
        if self.device_type == 'xpu':
            if __package__:
                from .sam31_mask_native import build_stages, BACKEND, RECIPE
            else:
                from sam31_mask_native import build_stages, BACKEND, RECIPE
            image, self.text_stage, head, self.tokenizer = build_stages(Path(args.mask_bundle), Path(args.checkpoint), torch, self.device)
            self.backbone = image.backbone
            self.native_owner = image
            segmentation = layers.InstanceMasks(args.checkpoint, self.device, tiled_norm=True)
            modules = {'grounding':layers.grounding_with_mask_inputs(head), 'masks':segmentation}
            self.backend, self.recipe = BACKEND, RECIPE
            self.image_optimization = getattr(image, '_mask_optimization', None)
            if self.image_optimization:
                self.backend = self.image_optimization['image_backend']
                self.recipe = self.image_optimization['recipe']
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            model = common._build_detector(torch, Path(args.checkpoint))
            model.geometry_encoder = common._text_only_geometry_encoder(torch)(model.geometry_encoder)
            common._enable_real_rope(torch, model)
            model._apply(lambda value: value.contiguous())
            model.to(self.device).eval()
            common._move_plain_tensors(torch, model, self.device)
            from sam3.model.data_misc import FindStage
            from sam3.model.geometry_encoders import Prompt
            _, text_type, _ = common._shared_wrappers(torch, FindStage, Prompt)
            self.text_stage = text_type(model).eval()
            self.tokenizer = model.backbone.language_backbone.tokenizer
            segmentation = layers.InstanceMasks(args.checkpoint, self.device, tiled_norm=False)
            modules = {'grounding':layers.ReferenceMaskHead(model, segmentation.head).eval()}
            self.backbone = model.backbone
            self.backend, self.recipe = 'sam31-mask-torch-cuda', 'dense-same-checkpoint'
        modules.update(image=layers.ImageFeatures(self.backbone).eval(),
                       fpn=layers.ExtraFeatures(self.backbone).eval(),
                       output=layers.BinaryMaskOutput(self.capacity, self.confidence).eval())
        options = {'emulate_precision_casts':True, 'triton.cudagraphs':False}
        self.compiled = {name:(stage_factory(name, module) if stage_factory else
                              torch.compile(module, fullgraph=True, dynamic=False, options=options))
                         for name,module in modules.items()}
        self.preprocessor = ExactImagePreprocessor(torch, self.device, self._progress)
        self.start_event = self.runtime.Event(enable_timing=True)
        self.end_event = self.runtime.Event(enable_timing=True)
        self.device_name = self.runtime.get_device_name(self.device)
        self.output_transfer = OutputTransfer(torch, self.runtime)
        self.postprocessor_cache = {}

    def detect_many(self, jpeg, prompts, *, prepared_pixels=None, preparation_wait_ms=0., **unused):
        import numpy as np
        from PIL import Image
        prompts = normalize_prompts(prompts)
        if not prompts: raise ValueError('At least one prompt is required')
        start = time.perf_counter()-preparation_wait_ms/1000
        width, height = Image.open(io.BytesIO(jpeg)).size
        pixels = self.preprocessor(jpeg, prepared=prepared_pixels)
        self.runtime.synchronize()
        prepared = time.perf_counter()
        with self.torch.inference_mode(), self.torch.autocast(self.device_type,dtype=self.dtype):
            cached = all(p in self.text_cache for p in prompts)
            embeddings = self._encode_prompts(prompts)
            inputs = (pixels, *(v for pair in embeddings for v in pair))
            if self.graph is None or len(self.graph.inputs) != len(inputs):
                self.graph = None
                gc.collect()
                self.graph = MaskGraph(self, inputs)
            self.runtime.synchronize()
            model_start = time.perf_counter()
            self.start_event.record()
            outputs = self.graph(*inputs)
            self.end_event.record()
            self.runtime.synchronize()
            model_end = time.perf_counter()
            post_shape = (width,height,len(prompts))
            cached_post = self.postprocessor_cache.pop(post_shape, None)
            if cached_post is None:
                # Retain the eight supported prompt-count shapes. Recreating a
                # class/graph on every size or prompt switch can exhaust Dynamo's
                # recompile limit even when returning to an already-seen shape.
                if len(self.postprocessor_cache) >= 8:
                    self.postprocessor_cache.pop(next(iter(self.postprocessor_cache)))
                cached_post = [CompiledStage(self.torch,
                    packed_mask_output(self.torch,width,height).to(self.device),
                    'mask-postprocess',self._progress), False]
            self.postprocessor_cache[post_shape] = cached_post
            packed = cached_post[0](*outputs)
            cpu = self.output_transfer(packed)
            if not cached_post[1]:
                validate_packed(cpu,outputs,width,height)
                cached_post[1] = True
        transferred = time.perf_counter()
        labels,areas,x_moments,y_moments,ranks,*metadata = cpu
        boxes, categories, overflow = [], [], {}
        colors = {}
        for p, prompt in enumerate(prompts):
            scores, queries, count, omitted, xywh = metadata[p*5:p*5+5]
            count, omitted = int(count), int(omitted)
            if not 0 <= count <= self.capacity or omitted < 0: raise RuntimeError('Invalid mask count')
            overflow[prompt] = omitted
            categories.append({'prompt':prompt,'color':COLORS[p],'count':count})
            for i in range(count):
                center, size = xywh[i,:2], xywh[i,2:]
                coords = np.clip(np.concatenate((center-size/2,center+size/2)),0,1).tolist()
                color = instance_color(COLORS[p],int(queries[i]))
                slot=p*self.capacity+i
                centroid = centroid_from_moments(areas[slot],x_moments[slot],y_moments[slot])
                boxes.append({'xyxy':coords,'score':float(scores[i]),'prompt':prompt,
                              'prompt_index':p,'color':color,'mask_centroid':centroid})
                colors[int(ranks[slot])] = color
        decoded = time.perf_counter()
        overlay = encode_labels(labels,[colors[i] for i in range(1,len(colors)+1)])
        end = time.perf_counter()
        return {'boxes':boxes,'categories':categories,'mask_overlay':overlay,'mask_detection':True,
                'temporal_tracking':False,'client_overlay':True,'mask_capacity_per_prompt':self.capacity,
                'mask_overflow':overflow,'torch':self.torch.__version__,'device':self.device_name,
                'device_type':self.device_type,'image_backend':self.backend,
                'precision':('W4A4-MLP/W8A8-projections/FP16-attention' if self.image_optimization
                             else 'W4A4-MLP/FP16-attention' if self.device_type=='xpu' else 'float16'),
                'image_optimization':self.image_optimization,
                'compiled_mask_artifacts': self.artifacts is not None,
                'torch_compile':True,'cuda_graph':self.device_type=='cuda','sycl_graph':self.device_type=='xpu',
                'compilation_scope':'image+fpn+all-prompt-grounding+masks+selection',
                'model_graph_count':1,'image_passes_per_frame':1,'head_passes_per_frame':len(prompts),
                'text_cache_hit':cached,'recipe':self.recipe,'preprocess_validation':self.preprocessor.validation,
                'preprocess_cache_hit':prepared_pixels is not None,
                'mask_postprocess':'gpu-integer-moments-and-labels',
                'mask_postprocess_bitwise_equal':cached_post[1],
                'mask_postprocess_graph_cache_entries':len(self.postprocessor_cache),
                'latency_ms':round((model_end-model_start)*1000),
                'timing':{'preprocess_ms':round((prepared-start)*1000,2),
                          'preparation_wait_ms':round(preparation_wait_ms,2),
                          'prompt_setup_ms':round((model_start-prepared)*1000,2),
                          'model_gpu_ms':round(self.start_event.elapsed_time(self.end_event),2),
                          'model_ms':round((model_end-model_start)*1000,2),
                          'postprocess_ms':round((decoded-model_end)*1000,2),
                          'postprocess_gpu_readback_ms':round((transferred-model_end)*1000,2),
                          'metadata_ms':round((decoded-transferred)*1000,2),
                          'mask_overlay_ms':round((end-decoded)*1000,2),
                          'worker_total_ms':round((end-start)*1000,2)}}


def main():
    common._PROTOCOL_OUTPUT = os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--mask-bundle')
    parser.add_argument('--compiled-bundle')
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--device-type',choices=('xpu','cuda'),default='xpu')
    parser.add_argument('--precision',choices=('float16',),default='float16')
    parser.add_argument('--confidence',type=float,default=.5)
    args = parser.parse_args()
    if args.device_type=='xpu' and not args.mask_bundle: parser.error('XPU masks require a frozen mask bundle')
    preparation = None
    try:
        engine = MaskEngine(args)
        common._emit({'type':'ready','request_transport':JPEG_BYTES,'cpu_prefetch':True})
        preparation = LatestPreparation(engine.preprocessor.prepare_cpu)
        for request in RequestInbox(sys.stdin.buffer,preparation):
            started=time.perf_counter()
            prepared=preparation.take(request.get('prepared_token'),request['jpeg'],wait_seconds=.025)
            waited=(time.perf_counter()-started)*1000
            result = engine.detect_many(request['jpeg'],request['prompts'],
                prepared_pixels=prepared,preparation_wait_ms=waited)
            common._emit({'type':'result','id':request['id'],**result})
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        common._emit({'type':'fatal','error':str(error)})
        return 1
    finally:
        if preparation is not None: preparation.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
