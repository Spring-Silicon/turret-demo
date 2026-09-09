"""Pinned sleepy-joe mask qualification image/grounding setup, 2026-09-08."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

MANIFEST = 'fddeac69e45690387725eea8faae241d28514317cab6e958280b9ad6b07d1fcf'
RUNTIME_MANIFEST = '6dea19e32668dcf3c907e80414c6b4eca4652dd0e33f30e946917228fa271cdc'
CHECKPOINT = '0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6'
TORCH_COMMIT = '08187d9e0fba026dc8217405802ab5381dc88d90'
BACKEND = 'sleepy-w4a4-mask-tiled'
RECIPE = 'masks/tiled-qualification-gpu1'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_bundle(bundle, checkpoint):
    if digest(checkpoint) != CHECKPOINT:
        raise ValueError('W4A4 checkpoint changed')
    for name, expected in (('manifest.json', MANIFEST), ('runtime-manifest.json', RUNTIME_MANIFEST)):
        if digest(bundle / name) != expected:
            raise ValueError(f'W4A4 {name} changed')
        for relative, checksum in json.loads((bundle / name).read_text()).items():
            path = bundle / relative
            if not path.resolve().is_relative_to(bundle.resolve()) or digest(path) != checksum:
                raise ValueError(f'W4A4 artifact changed: {relative}')


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_stages(bundle, checkpoint, torch, device):
    """Same installation order/flags as the frozen benchmark's selected recipe."""
    verify_bundle(bundle, checkpoint)
    if torch.__version__ != '2.14.0+xpu' or torch.version.git_version != TORCH_COMMIT:
        raise ValueError('W4A4 requires its measured PyTorch build')
    if device.type != 'xpu' or 'B580' not in torch.xpu.get_device_name(device):
        raise ValueError('This native W4A4 build requires Intel Arc B580')
    if any(name == 'sam3' or name.startswith('sam3.') for name in sys.modules):
        raise RuntimeError('W4A4 must start in a fresh worker before importing SAM')
    root = bundle / 'runtime/repository'
    selected = root / 'outputs/sam31-native-w4a4'
    report = json.loads((bundle / 'receipt/report.json').read_text())
    def library(folder):
        matches = [selected / Path(p).relative_to('/home/spring/springsilicon/graphs/outputs/sam31-native-w4a4')
                   for p in report['libraries'] if f'/sam31-native-w4a4/{folder}/' in p]
        result, = matches
        return result
    sys.path[:0] = [str(bundle / 'runtime/sam'), str(selected / 'integration'),
                   str(root / 'ci/benchmarks/sam31_tradeoffs')]
    # Match the benchmark's import order: initialize SAM/Dynamo/custom operators
    # before temporarily redirecting CUDA-default construction factories.
    native_model = _load(selected / 'integration/model.py', 'sam31_w4a4_native_model')
    from sam3.model.data_misc import FindStage
    from sam3.model.geometry_encoders import Prompt
    from safetensors.torch import load_file
    # Only original, pinned detector-construction functions; not its old worker.
    source = root / 'outputs/turret-demo-scope-audit/src/spring_turret/sam31_worker.py'
    names = {'_build_detector', '_text_only_geometry_encoder', '_enable_real_rope',
             '_move_plain_tensors', '_redirect_cuda_construction', '_shared_wrappers'}
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in tree.body} != names:
        raise ValueError('Incomplete frozen detector constructor')
    namespace = dict(contextlib=contextlib, types=types, Path=Path, Any=Any)
    exec(compile(tree, str(source), 'exec', dont_inherit=True), namespace)
    demo = types.SimpleNamespace(**namespace)
    with demo._redirect_cuda_construction(torch):
        model = demo._build_detector(torch, checkpoint)
    model.geometry_encoder = demo._text_only_geometry_encoder(torch)(model.geometry_encoder)
    demo._enable_real_rope(torch, model)
    model._apply(lambda value: value.contiguous())
    model.to(device).eval()
    demo._move_plain_tensors(torch, model, device)
    image_type, text_type, _ = demo._shared_wrappers(torch, FindStage, Prompt)
    image, text = image_type(model).eval(), text_type(model).eval()
    tokenizer = model.backbone.language_backbone.tokenizer
    sys.path.insert(0, str(selected / 'grounding'))
    from mkldnn_epilogue import load_candidate
    head, counts = load_candidate(str(device), packed_mask=True, query_tile=128, fuse_relu=True,
                                  tla_library=library('grounding/head-no-workgroup-barrier'))
    expected = {'split_decoder': 6, 'tla_self': 6, 'fusion_ffn': 6, 'axis_rpb': 12, 'mkldnn_ffn': 6}
    if counts != expected:
        raise ValueError(f'Wrong W4A4 head replacements: {counts}')
    native_model.install(model, load_file(root / 'outputs/sam31-theoretical-frontier/calibration.safetensors'),
                         variant=4, fused=True, vnni=True, pipeline_fc2=True, pipeline_fc1=True, lut32=True)
    sys.path.insert(0, str(selected / 'int4/fixed_shape_unroll'))
    import native_fixed_shape_unroll
    import native_lut32
    native_lut32.s4fc1_pack = native_fixed_shape_unroll.s4fc1_pack
    # Final FC1 selection from the qualified runner; earlier overwritten
    # specializations do not alter weights or quantization configuration.
    fastinv_fc1 = _load(selected / 'int4/reciprocal/fc1_fastinv/native_schedule.py', 'sam31_fastinv_fc1')
    native_lut32.s4fc1_pack = fastinv_fc1.s4fc1_pack
    import native_pipeline
    sys.path.insert(0, str(selected / 'int4/pointer_induction'))
    import native_pointer_induction
    native_pipeline.s4linear_residual = native_pointer_induction.s4linear_residual
    import routing
    routing.install(model, routing.RoutingConfig(scope='block', score='gradient', keep=.875, start_layer=16, alignment=8))
    sys.path.insert(0, str(selected / 'attention'))
    import image as attention
    attention.install(model, routing)
    attention.LIBRARY = library('attention/local-no-workgroup-barrier')
    attention.GLOBAL_LIBRARY = library('attention/no-workgroup-barrier')
    import native_rope
    native_rope.install(model, routing, attention)
    sys.path.insert(0, str(selected / 'routing'))
    import native_router
    native_router.install(model, routing, variant='rank', local_only=True)
    sys.path[:0] = [str(selected / 'normalization'), str(selected / 'normalization/compensated-pack')]
    import fused_blocks
    sys.path.insert(0, str(selected / 'normalization/fast-inv'))
    import native_fastinv_norm_pack
    fused_blocks.residual_pack = native_fastinv_norm_pack.residual_pack
    fused_blocks.LIBRARY_PATH = native_fastinv_norm_pack.LIBRARY_PATH
    fused_blocks.install(model, routing, native_router)
    import qkv_install
    sys.path.insert(0, str(selected / 'qkv-vnni/persistent-payload'))
    import native_qkv_payload
    qkv_install.run, qkv_install.LIBRARY_PATH = native_qkv_payload.run, native_qkv_payload.LIBRARY_PATH
    qkv_install.install(model, routing)
    for folder, name in (('gather-norm', 'w4a4_gather_install'), ('dense-window-exact', 'w4a4_window_install')):
        directory = selected / 'normalization' / folder
        sys.path.insert(0, str(directory))
        _load(directory / 'install.py', name).install(model)
    helpers = root / 'ci/benchmarks/sam31_native/hillclimb75'
    sys.path.insert(0, str(helpers))
    mapping = _load(helpers / 'native_mapping.py', 'sam31_mask_native_mapping')
    layout = _load(helpers / 'fc2_weight_layout.py', 'sam31_mask_weight_layout')
    selection = json.loads((bundle / 'receipt/selection.json').read_text())
    panel = selection['fc2_weight_panel_tiles']
    backend, _, _ = mapping.load_backend(root / selection['fc2_candidate'], selection['fc2_mapping'],
                                         weight_panel_tiles=panel)
    native_pointer_induction.LIBRARY = backend
    owners = [backend]
    count = 0
    for module in model.modules():
        if isinstance(module, native_model.NativeMlp):
            original = module.fc2.packed_weight
            packed = layout.repack(original, panel)
            if not torch.equal(layout.repack(packed, panel, inverse=True), original):
                raise ValueError('Mask FC2 physical weight repacking changed values')
            module.fc2.packed_weight = packed
            owners.extend((original, packed)); count += 1
    if count != 32:
        raise ValueError('Unexpected mask image MLP count')
    for kind, item in report.get('hillclimb_attention', {}).items():
        path = root / Path(item['build']['library']).relative_to('/home/spring/springsilicon/graphs')
        setattr(attention, 'GLOBAL_LIBRARY' if kind == 'global' else 'LIBRARY', path)
    qkv_backend = _load(helpers / 'qkv_backend.py', 'sam31_mask_qkv_backend')
    native_qkv_payload.LIBRARY, _, _ = qkv_backend.load_backend(root / selection['qkv_candidate'])
    fc1_backend = _load(helpers / 'fc1_backend.py', 'sam31_mask_fc1_backend')
    fastinv_fc1.LIBRARY, _, _ = fc1_backend.load_backend(root / selection['fc1_candidate'])
    owners.extend((native_qkv_payload.LIBRARY, fastinv_fc1.LIBRARY))
    image._native_owners = owners
    image.eval()
    if any(m.training for m in image.modules()):
        raise RuntimeError('W4A4 image contains training modules')
    extension = bundle / 'attention8'
    if extension.exists():
        if __package__:
            from .sam31_mask_attention8 import install
        else:
            from sam31_mask_attention8 import install
        image._mask_optimization = install(extension, model, CHECKPOINT, device)
    return image, text, head, tokenizer
