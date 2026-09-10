"""Pinned Proteus neural builders, without the offline benchmark/supervisor.

The hybrid experiment did not publish a builder function. Extract only its
hash-pinned construction statements, retaining its exact kernels and weights.
Never execute its CLI, admission, video loop, rendering, or filesystem writes.
The resulting model is consumed by the same bounded live worker protocol.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
import types

HYBRID_SHA = '478dffa63ae4aa7fcf0661761fb9e3615790699dd0a2bb99f0a2b2c408e2621e'
SEED_SHA = 'ae29cf801f8fbe546c488b2e23c6c17420a7d13fb0b7c8fe1569f5bc4cad0ad5'
MEMORY_BUILDER_SHA = 'b7955e0ffda2ed061a4a0e5345666944c820750f48fdb16e9d4b9d5e87cffc56'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def construction(path, digest, start, end, namespace, replacements):
    """Extract a uniquely delimited builder from the reviewed, immutable source."""
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == digest, f'Proteus source changed: {path.name}')
    tree = ast.parse(source, str(path))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    rendered = [ast.unparse(node) for node in main.body]
    first = [i for i, text in enumerate(rendered) if text.startswith(start)]
    last = [i for i, text in enumerate(rendered) if text.startswith(end)]
    require(len(first) == len(last) == 1 and first[0] < last[0], 'Pinned builder boundaries changed')

    class Relocate(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in replacements:
                node.value = replacements[node.value]
            return node

    tree = ast.Module(body=main.body[first[0]:last[0]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(Relocate().visit(tree)), str(path), 'exec'), namespace)
    return namespace


def source_path(root, profile, memory_bundle=None):
    if profile == 'efficient-memory':
        require(memory_bundle is not None, 'Efficient Memory source bundle is not configured')
        path = Path(memory_bundle) / 'efficient-sam3-nomux-25ms-20260910/sam3'
        require(hashlib.sha256((path / 'sam3/model_builder.py').read_bytes()).hexdigest()
                == MEMORY_BUILDER_SHA, 'Efficient Memory builder changed; review the new snapshot')
        return path
    if profile == 'efficient-nomem':
        return root / 'efficient-sam3-nomux-25ms-20260910/sam3'
    if profile == 'hybrid-nomem':
        return root / 'sam31-speed-memoff-20260910'
    return root / 'basketball-demo-6070826/efficient-sam31-oob-20260910/sam3'


def configure(root, profile, *, seed=False, memory_bundle=None):
    # Hybrid's image initializer and tracker use different upstream packages.
    # They therefore always live in separate OS processes, never mixed imports.
    if seed and profile == 'hybrid-nomem':
        profile = 'efficient-tracking'
    path = source_path(root, profile, memory_bundle)
    if profile == 'hybrid-nomem':
        # The original experiment's loader resolves the deliberately linked sam3.
        from spring_turret.sam31_tracking import configure_source
        configure_source(path)
    else:
        sys.path.insert(0, str(path))
    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.modules['cv2'] = types.ModuleType('cv2')
    import torch
    require(torch.xpu.is_available(), 'Proteus profiles require Intel XPU')
    torch.set_num_threads(4)
    torch.xpu.set_device(0)  # ONEAPI_DEVICE_SELECTOR exposes the configured GPU.
    device = torch.device('xpu', 0)
    if profile != 'hybrid-nomem':
        import sam3.device as device_utils
        device_utils.get_device = lambda: device
        device_utils.get_autocast_device_type = lambda value: str(value).split(':')[0]
        device_utils.get_autocast_dtype = lambda value: torch.bfloat16
    # Upstream's platform-specific connected components is unsupported on XPU.
    # Same CPU SciPy implementation as the three Proteus benchmark runners.
    from sam3.perflib import connected_components as cc
    from sam3.model import sam3_tracker_utils

    def connected_components(value):
        import numpy as np
        from scipy.ndimage import label
        shape = tuple(value.shape)
        values = value.squeeze(1) if value.ndim == 4 else value
        labels, counts = [], []
        for row in values:
            ids, count = label(row.cpu().numpy())
            ids = ids.astype(np.int64, copy=False)
            sizes = np.bincount(ids.ravel(), minlength=count + 1)
            area = np.where(ids > 0, sizes[ids], 0).astype(np.int64)
            labels.append(torch.from_numpy(ids))
            counts.append(torch.from_numpy(area))
        if not labels:
            empty = torch.empty(shape, dtype=torch.int64, device=value.device)
            return empty, empty.clone()
        return (torch.stack(labels).to(value.device).view(shape),
                torch.stack(counts).to(value.device).view(shape))
    cc.connected_components = connected_components
    sam3_tracker_utils.connected_components = connected_components
    if profile != 'efficient-tracking' and not seed:
        install_exact_resize_fallback(torch)
    return torch, device


def install_exact_resize_fallback(torch):
    """Keep the source GPU upsampler; use real Pillow for unsupported downsampling."""
    from sam3.optimization import pillow_resize_xpu as resize_module
    original = resize_module.PillowBilinearResizeXPU

    class ExactResize:
        def __init__(self, device, in_size=(480, 640), out_size=(1008, 1008)):
            self.device, self.in_size, self.out_size = device, tuple(in_size), tuple(out_size)
            self.gpu = original(device, in_size, out_size) if all(a <= b for a, b in zip(in_size, out_size)) else None

        def __call__(self, image):
            if self.gpu is not None:
                return self.gpu(image)
            import numpy as np
            from PIL import Image
            require(image.dtype == torch.uint8 and tuple(image.shape) == (3, *self.in_size),
                    'Expected current RGB uint8 camera input')
            rgb = image.permute(1, 2, 0).cpu().numpy()
            resized = Image.fromarray(rgb).resize(self.out_size[::-1], Image.Resampling.BILINEAR)
            pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1).contiguous().to(self.device)
            normalized = pixels.float().div_(255.).sub_(.5).div_(.5).unsqueeze(0)
            return normalized, pixels

    resize_module.PillowBilinearResizeXPU = ExactResize


def build_seed(root, profile, checkpoint, torch, device):
    demo = root / 'basketball-demo-6070826'
    from sam3 import model_builder as builder
    from sam3.model.sam3_image_processor import Sam3Processor
    if profile in ('efficient-nomem', 'efficient-memory'):
        detector = builder.build_efficientsam3_image_model(device=device, checkpoint_path=None,
            backbone_type='tinyvit', model_name='11m', text_encoder_type='MobileCLIP-S0',
            text_encoder_context_length=16, enable_inst_interactivity=False, compile=False).eval()
        builder._load_checkpoint(detector, str(demo / 'agent-efficient-release/efficientsam3_tinyvit.pt'))
        detector.to(device).eval().requires_grad_(False)
        return Sam3Processor(detector, resolution=1008, device=device, confidence_threshold=.5)
    repo = demo / 'efficient-sam31-oob-20260910'
    namespace = construction(demo / 'es31vision_meta_text_image_seed.py', SEED_SHA,
        'from sam3 import model_builder', 'image = Image.open',
        dict(torch=torch, device=device, REPO=repo, require=require,
             CHECKPOINT=repo / 'efficient_sam3p1_tinyvit_m_mobileclip_s0_ctx16.pt'),
        {'/home/spring/sam31-speed-memoff-20260910/assets/sam3.1_multiplex.pt': str(checkpoint)})
    # The source's zero-threshold/top-score initialization is intentionally
    # retained and prominently disclosed; it is not a validated detector.
    return namespace['processor']


def build_seeded_tracker(root, profile, checkpoint, raw, torch, device):
    from sam3.optimization.pillow_resize_xpu import PillowBilinearResizeXPU
    height, width = raw.shape[-2:]
    resizer = PillowBilinearResizeXPU(device, in_size=(height, width), out_size=(1008, 1008))
    lab = root / 'sam31-speed-memoff-20260910'
    if profile in ('efficient-nomem', 'efficient-memory'):
        import numpy as np
        from PIL import Image
        # Retain the source experiment's first-frame quantization calibration,
        # independent of the object or camera resolution selected in the UI.
        with Image.open(root / 'efficient-nomem-calibration.jpg') as image:
            cal = torch.from_numpy(np.array(image.convert('RGB'))).permute(2, 0, 1).contiguous()
        calibration, _ = PillowBilinearResizeXPU(device, in_size=tuple(cal.shape[-2:]),
            out_size=(1008, 1008))(cal.to(device))
        from sam3 import model_builder
        builder = (model_builder.build_efficientsam3_memory_video_model if profile == 'efficient-memory'
                   else model_builder.build_efficientsam3_nomux_video_model)
        model = builder(
            str(root / 'basketball-demo-6070826/agent-efficient-release/efficientsam3_tinyvit.pt'),
            str(lab / 'assets/efficient_sam3_tinyvit_m.pt'), device=device,
            enable_xpu_optimizations=True, xpu_calibration_image=calibration,
            xpu_input_size=(height, width), capture_xpu_graph=True)
        manifest = model.efficientsam3_nomux_manifest
        require(manifest['w8a8_linear_count'] == 40 and
                len(manifest['attention_prelude_modules']) == 6 and
                len(manifest['deconvolution_modules']) == 5, 'Efficient kernel recipe changed')
        if profile == 'efficient-memory':
            require(model.num_maskmem == 7 and manifest['memory_enabled']
                    and manifest['tracker_neck_tensor_count'] == 22
                    and not manifest['detector_in_execution_graph'] and not manifest['multiplex_state'],
                    'Efficient Memory recipe changed')
        return model, {}
    script = root / 'basketball-demo-6070826/basketball_tracker_es31vision_meta_text.py'
    # Importing defines helpers only; the guarded CLI is never called.
    require(hashlib.sha256(script.read_bytes()).hexdigest() == HYBRID_SHA, 'Hybrid source changed')
    benchmark = load_module('proteus_hybrid_snapshot', script)
    benchmark.ROOT = lab
    # Its configure_source call has already happened in configure().
    adapter = load_module('proteus_hybrid_adapter', lab / 'adapters/tracking_adapter.py')
    adapter.configure_source = lambda _: {}
    original_loader = benchmark.load_module
    benchmark.load_module = lambda name, path: adapter if name == 'tracking_adapter_experiment' else original_loader(name, path)
    # Keep the source recipe's quantization calibration image, not the live view.
    resizers = {(height, width): resizer}
    def upload_image(image):
        size = tuple(image.shape[-2:])
        if size not in resizers:
            resizers[size] = PillowBilinearResizeXPU(device, in_size=size, out_size=(1008, 1008))
        return resizers[size](image.to(device))
    namespace = dict(vars(benchmark), torch=torch, device=device, ROOT=lab,
        args=types.SimpleNamespace(variant='hybrid_w8a8_nomem'), build_start=time.perf_counter(),
        upload_image=upload_image,
        resize_audit={'compared_calls':0, 'max_abs_byte_diff':0,
                      'max_abs_normalized_diff':0., 'exact':True})
    namespace = construction(script, HYBRID_SHA, 'adapter = load_module', 'frames = JpegFrames', namespace,
        {'/home/spring/basketball-demo-6070826/efficient-sam31-oob-20260910/efficient_sam3p1_tinyvit_m_mobileclip_s0_ctx16.pt':
            str(root / 'basketball-demo-6070826/efficient-sam31-oob-20260910/efficient_sam3p1_tinyvit_m_mobileclip_s0_ctx16.pt')})
    # load_retained_weights uses the original relative checkpoint path.
    model = namespace['model']
    require(model.num_maskmem == 0 and namespace['trunk_xpu_graph'], 'Hybrid NoMem recipe changed')
    return model, namespace


def build_oob(root, torch, device):
    from sam3 import model_builder as builder
    repo = root / 'basketball-demo-6070826/efficient-sam31-oob-20260910'
    wrapper = builder.build_efficientsam3_multiplex_video_model(
        checkpoint_path=str(repo / 'efficient_sam3p1_tinyvit_m_mobileclip_s0_ctx16.pt'),
        load_from_HF=False, bpe_path=str(repo / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
        device='xpu', compile=False, backbone_type='tinyvit', model_name='m',
        text_encoder_type='mobileclip-s0', text_encoder_context_length=16,
        text_encoder_pos_embed_table_size=16, interpolate_pos_embed=False, use_fa3=False)
    model = wrapper._model
    model.use_batched_grounding = False
    model.postprocess_batch_size = 1
    require(model.tracker.model.num_maskmem == 7, 'Efficient Tracking memory policy changed')
    return model.eval().requires_grad_(False)
