"""Pinned sleepy-joe skip4_attention8 mask-only extension to the native image."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

MANIFEST = '2ec94c43535ea3cfd4c384b0d2adcfbb26907d18bbc0734655c66f3cf6958640'
RECIPE = 'masks/w8a8/skip4_attention8-v0'
BACKEND = 'sleepy-w4a4-w8a8-skip4-mask-tiled'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_extension(root):
    root = Path(root).resolve()
    if digest(root / 'manifest.json') != MANIFEST:
        raise ValueError('Unknown mask attention8 bundle')
    for relative, expected in json.loads((root / 'manifest.json').read_text()).items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise ValueError(f'Mask attention8 artifact drift: {relative}')
    selection = json.loads((root / 'selection.json').read_text())
    if (selection['recipe'], selection['skipped_blocks'], selection['projection_variant']) != (
        'skip4_attention8', [24, 26, 28, 30], 0
    ):
        raise ValueError('Unexpected mask attention8 recipe')
    return selection


def install(root, model, checkpoint_sha256, device):
    root = Path(root).resolve()
    selection = verify_extension(root)
    name = 'spring_turret_mask_attention8_runtime'
    if name in sys.modules:
        raise RuntimeError('Mask attention8 requires a fresh inference worker')
    package = types.ModuleType(name)
    package.__path__ = [str(root / 'runtime')]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + '.install', root / 'runtime/install.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    manifest_path = root / 'dataset-manifest.json'
    state = {'model': model, 'device': device, 'report': {'checkpoint_sha256': checkpoint_sha256},
             'manifest_path': manifest_path, 'manifest': json.loads(manifest_path.read_text())}
    with tempfile.TemporaryDirectory(prefix='turret-mask-attention8-') as output:
        result = module.install(state, output, calibration=root / 'calibration.safetensors', variant=0)
    if (result['retained_blocks'] != 28 or len(result['projections']) != 56
        or result['library_sha256'] != selection['library_sha256']
        or result['calibration_sha256'] != selection['calibration_sha256']):
        raise ValueError('Mask attention8 installation did not match the qualified recipe')
    return {'recipe': RECIPE, 'image_backend': BACKEND, 'retained_blocks': 28,
            'projection_count': 56, 'skipped_blocks': selection['skipped_blocks'],
            'bundle_sha256': MANIFEST, 'projection_library_sha256': result['library_sha256'],
            'source_model_gpu_ms': selection['median_gpu_ms']['candidate'],
            'source_mask_ap_loss_pp': selection['additional_mask_ap_loss_pp'],
            'accuracy_scope': '512 COCO person confirmation images reused in selection; other prompts unqualified'}
