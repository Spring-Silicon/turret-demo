"""Israel full1008 v18 neural backend with the shared live tracking lifecycle."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

if not __package__:
    spec = importlib.util.spec_from_file_location('spring_turret',
        Path(__file__).with_name('__init__.py'), submodule_search_locations=[str(Path(__file__).parent)])
    package = importlib.util.module_from_spec(spec)
    sys.modules['spring_turret'] = package
    spec.loader.exec_module(package)

from spring_turret import sam31_tracking_worker as host
from spring_turret.sam31_tracking_v18 import initialize
from spring_turret.tracking_graph_cache import finish_cache_warmup


class V18TrackingEngine(host.TrackingEngine):
    def __init__(self, args, *, compile_stages=True, progress=None):
        if getattr(args, 'device_type', 'xpu') != 'xpu':
            raise ValueError('Israel v18 tracking requires Intel XPU')
        if not compile_stages:
            raise ValueError('The pinned v18 recipe requires its graph hook')
        initialize(self, args, Path(os.environ['SPRING_SAM31_TRACKING_NATIVE_BUNDLE']), progress)

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
        # Bound diagnostic logs only, never model temporal memory.
        for values in getattr(self.model, '_native_work_observations', {}).values():
            if isinstance(values, list): del values[:-32]
        events = getattr(self.model, '_native_preflight_stats', {}).get('events')
        if isinstance(events, list): del events[:-32]
        return result


if __name__ == '__main__':
    host.TrackingEngine = V18TrackingEngine
    raise SystemExit(host.main())
