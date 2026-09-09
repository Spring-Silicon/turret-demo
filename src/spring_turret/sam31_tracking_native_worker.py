"""Pinned Israel native Object Multiplex workers; no camera or servo access."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

# Run the regular worker protocol under package-qualified module names. The
# frozen model owns its original unqualified graph-module namespace separately.
if not __package__:
    spec = importlib.util.spec_from_file_location('spring_turret',
        Path(__file__).with_name('__init__.py'), submodule_search_locations=[str(Path(__file__).parent)])
    package = importlib.util.module_from_spec(spec)
    sys.modules['spring_turret'] = package
    spec.loader.exec_module(package)
from spring_turret import sam31_tracking_worker as host
from spring_turret.tracking_graph_cache import install_nonblocking_cache, finish_cache_warmup
from spring_turret.tracking_preprocess import VideoPreprocessor
from spring_turret.tracking_suppression import install_tracking_suppression

MANIFEST_SHA = '34958e69ecb21d124669edb2e8518fe73e256eca9fdad01fc890a6c6afd8df48'
DNNL_SHA = '0c38542cc9fdba8d6bf4c7837289a7260cc9fbaad1c587fea6074f67a247958a'


class NativeTrackingEngine(host.TrackingEngine):
    def __init__(self, args, *, compile_stages=True, progress=None):
        if getattr(args, 'device_type', 'xpu') != 'xpu':
            raise ValueError('Israel native tracking requires Intel XPU')
        if not compile_stages:
            raise ValueError('The pinned native recipe requires its qualified graph hook')
        bundle = Path(os.environ['SPRING_SAM31_TRACKING_NATIVE_BUNDLE']).resolve()
        manifest_path = bundle / 'manifest.json'
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        from spring_turret import sam31_tracking_hillclimb
        if manifest_sha == sam31_tracking_hillclimb.MANIFEST_SHA:
            sam31_tracking_hillclimb.initialize(self, args, bundle, progress)
            return
        if manifest_sha != MANIFEST_SHA:
            raise ValueError('Unknown Israel tracking bundle')
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256((bundle/'lib/libdnnl.so.3').read_bytes()).hexdigest() != DNNL_SHA:
            raise ValueError('Pinned oneDNN runtime changed')
        root = (bundle / 'sam3_1').resolve()
        helper = root/'scripts/turret_megakernel/engine_graph.py'
        if hashlib.sha256(helper.read_bytes()).hexdigest() != '07d5dc426740bbb0aa15fe16c42bb99a466ec1c043f088e3bbd6cc5acf45f938':
            raise ValueError('Pinned native helper changed')
        if Path('/home/spring/sam3_1').resolve() != root:
            raise ValueError('Pinned source-path binding is missing')
        for name, expected in manifest['files'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected['sha256']:
                raise ValueError('Tracking source/artifact drift: ' + name)
        if Path(args.checkpoint).resolve() != (root/'checkpoints/sam3.1_multiplex.pt').resolve():
            raise ValueError('Tracking checkpoint does not match the selected device config')
        sys.path.insert(0, str(root/'scripts/tracking_native/validation'))
        # Frozen calibration metadata contains source-root-relative filenames.
        # This process owns inference only, so no caller filesystem state changes.
        os.chdir(root)
        from common import build_reference
        results = root/'results/tracking_native_20260907'
        torch, adapter, model, metadata = build_reference('selected', args.device,
            native_hook='resolution_672_native.bundle:install',
            native_options={'calibration_report':str(results/'direct_conv/train32_gpu1_v1/report.json'),
                'calibration_sha256':'23607525cae3efb29e2753d36c3dfe67c374c74232e2b88b1bc175ad616d2c7c',
                'backend':'onednn','alpha':.5},
            graph_hook='resolution_672_native.graph:install')
        install_tracking_suppression(model)
        self.torch, self.model = torch, model
        self.accelerator = torch.xpu
        self.source_digest, self.source_count = None, -1
        self.device = torch.device('xpu', args.device)
        self.device_name = torch.xpu.get_device_name(self.device)
        self.image_size, self.session_factory = 672, adapter.OnlineSession
        self.graph_stages = model._native_tracking_regions
        install_nonblocking_cache(self.graph_stages)
        self.cache_warmup_frames = 0
        for stage in self.graph_stages.values(): stage.progress = progress
        self.sources = metadata['sam_source_sha256']
        self.confidence = args.confidence
        self.sessions, self.ids = [], {}
        self.next_id, self.session_key = 1, None
        self.last_captured_at, self.previous_duration = None, 0.
        self.receipt = {
            'image_backend':'israel-native672-v24-temporal',
            'precision':'w4a4-image-w8a8-heads-bf16-attention',
            'model_input_size':[672,672], 'temporal_spatial_kv_keep':.5,
            'native_bundle_sha256':MANIFEST_SHA,
            'source_qualification':'208 overlapping validation frames; person prompt; mask J87.792%, zero mask ID switches',
            'accuracy_tradeoff':'Lower input resolution and quantization; -1.872 mask-J points versus optimized1008, not dense equivalence.'}
        self.metadata = metadata
        self.preprocess = VideoPreprocessor(torch, self.device, progress, image_size=672)

    def before_detect(self):
        return self.graph_stages['image_and_detection'].calls

    def after_detect(self, result, image_calls_before):
        image_passes = self.graph_stages['image_and_detection'].calls - image_calls_before
        if image_passes < 1:
            raise RuntimeError('Tracking reused stale image features across frames')
        self.cache_warmup_frames += 1
        if self.cache_warmup_frames == 64:
            finish_cache_warmup(self.graph_stages)
        result.update({
            'graph_cache_policy':'replay-or-direct-no-recompile',
            'graph_cache_warmup_complete':self.cache_warmup_frames >= 64,
            'graph_cache_execution':{name:{'direct_calls':stage.direct_calls,
                'replay_calls':stage.replay_calls} for name,stage in self.graph_stages.items()},
            'image_passes_per_frame':image_passes, **self.receipt})
        # Diagnostic records, not SAM attention/session memory.
        for values in getattr(self.model, '_native_work_observations', {}).values():
            if isinstance(values, list):
                del values[:-32]
        events = getattr(self.model, '_native_preflight_stats', {}).get('events')
        if isinstance(events, list):
            del events[:-32]
        return result


if __name__ == '__main__':
    host.TrackingEngine = NativeTrackingEngine
    raise SystemExit(host.main())
