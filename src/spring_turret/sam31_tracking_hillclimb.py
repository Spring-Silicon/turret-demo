"""Pinned full1008 native-v15 build from Israel's tracking hillclimb.

Only neural execution changes here. The worker retains the demo's shared
session lifetime, suppression publication, target policy and mask output.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys

MANIFEST_SHA = '368a39b0abe03f0cf288f3162de40bdb1f5fdf4fa13555e66a5ed683a2c8fd39'


def verify_bundle(bundle, checkpoint):
    bundle = Path(bundle).resolve()
    path = bundle / 'manifest.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise ValueError('Unknown Israel full1008 tracking bundle')
    manifest = json.loads(path.read_text())
    for relative, expected in manifest['files'].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle):
            raise ValueError('Tracking bundle path escapes its root')
        with path.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                raise ValueError('Tracking source/artifact drift: ' + relative)
    with Path(checkpoint).open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != manifest['checkpoint_sha256']:
            raise ValueError('Tracking checkpoint mismatch')
    return manifest


def initialize(engine, args, bundle, progress):
    manifest = verify_bundle(bundle, args.checkpoint)
    root = Path(bundle).resolve() / 'source'
    sys.path[:0] = [str(root / directory) for directory in (
        'scripts/tracking_search_20260909',
        'scripts/tracking_implementation_20260909',
        'scripts/tracking_native', 'scripts/tracking_native/benchmark')]
    from common import imports, load_adapter, load_callable
    imports()
    os.environ['USE_PERFLIB'] = '0'
    os.environ['ENABLE_PROFILING'] = '0'
    adapter, sources = load_adapter()
    import numpy as np
    import torch
    if torch.__version__ != manifest['torch']:
        raise ValueError('Tracking bundle Torch build mismatch')
    # Resolve native attention before the legacy quantization helper adds its
    # own attention.py directory, exactly as in the audited source runner.
    from attention.install_token_major import install_token_major_attention  # noqa: F401
    from torch_candidates import apply_candidate
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.xpu.set_device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model = adapter.build_model(torch, Path(args.checkpoint)).eval()
    config = json.loads((root / 'results/tracking_implementation_20260909/native_config_v15.json').read_text())
    calibration = torch.load(root / 'results/tracking_compression_20260907/calibration.pt',
                             map_location='cpu', weights_only=True)
    maxima = {name: value['amax'] if isinstance(value, dict) else value
              for name, value in calibration.items()}
    legacy = apply_candidate(model, config['legacy'], maxima)
    installed = []
    for item in config['installers']:
        installed.append(load_callable(item['hook'])(model, item.get('config', {}), {}))
    model.eval()
    from graph_fused import install as install_graphs
    from inspectable_graph import graph_factory
    torch.xpu.XPUGraph = graph_factory(root / 'results/tracking_native_20260907/benchmark/inspectable_build_v2/inspectable_graph.so')
    graph_metadata = install_graphs(model)
    from .tracking_suppression import install_tracking_suppression
    from .tracking_graph_cache import install_nonblocking_cache
    from .tracking_preprocess import VideoPreprocessor
    install_tracking_suppression(model)
    engine.torch, engine.model = torch, model
    engine.accelerator, engine.device = torch.xpu, torch.device('xpu', args.device)
    engine.device_name = torch.xpu.get_device_name(engine.device)
    engine.image_size, engine.session_factory = 1008, adapter.OnlineSession
    engine.graph_stages = model._native_tracking_regions
    install_nonblocking_cache(engine.graph_stages)
    for stage in engine.graph_stages.values():
        stage.progress = progress
    engine.sources, engine.source_digest, engine.source_count = sources, None, -1
    engine.cache_warmup_frames = 0
    engine.confidence = args.confidence
    engine.sessions, engine.ids = [], {}
    engine.next_id, engine.session_key = 1, None
    engine.last_captured_at, engine.previous_duration = None, 0.
    engine.preprocess = VideoPreprocessor(torch, engine.device, progress, image_size=1008)
    engine.metadata = {'legacy': legacy, 'installers': installed, 'graphs': graph_metadata}
    engine.receipt = {
        'image_backend': 'israel-full1008-native-v15-temporal',
        'precision': 'w4a4-image-w8a8-heads-bf16-attention',
        'model_input_size': [1008, 1008], 'temporal_spatial_kv_keep': .5,
        'mlp_token_keep': .75, 'native_bundle_sha256': MANIFEST_SHA,
        'source_qualification': manifest['qualification'],
        'accuracy_tradeoff': 'Quantized projections, KV50 and MLP token merging; not dense BF16 equivalence.',
    }
