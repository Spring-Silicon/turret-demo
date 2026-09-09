"""Versioned, offline-compiled SAM mask stages for the pinned Arc runtime.

The package contains executable PyTorch artifacts; only load packages produced
locally by the offline exporter. Integrity checks detect corruption and drift,
not an untrusted publisher. GPU streams and command graphs remain process-local.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

STAGES = frozenset(('image', 'fpn', 'grounding', 'masks', 'output'))
SCHEMA = 1
SOURCE_FILES = ('sam31_mask_artifacts.py', 'sam31_mask_worker.py',
                'sam31_mask_layers.py', 'sam31_mask_native.py',
                'sam31_mask_attention8.py', 'sam31_worker.py', 'sam31_graph.py',
                'sam31_preprocess.py', 'mask_postprocess.py')


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def runtime_contract(torch, args):
    if args.device_type != 'xpu':
        raise ValueError('Compiled mask artifacts currently require XPU')
    bundle = Path(args.mask_bundle)
    manifests = {name: digest(bundle/name) for name in ('manifest.json', 'runtime-manifest.json')}
    if (bundle/'attention8').exists():
        manifests['attention8/manifest.json'] = digest(bundle/'attention8/manifest.json')
    return {
        'torch': str(torch.__version__), 'torch_commit': torch.version.git_version,
        'python': list(sys.version_info[:3]), 'device': args.device,
        'device_name': torch.xpu.get_device_name(args.device),
        'confidence': args.confidence, 'capacity': 10, 'precision': 'float16',
        'freezing': True, 'native_manifests': manifests,
        'sources': {name: digest(Path(__file__).with_name(name)) for name in SOURCE_FILES},
    }


def tensor_contract(inputs):
    return [{'shape': list(v.shape), 'stride': list(v.stride()), 'dtype': str(v.dtype),
             'device': str(v.device), 'requires_grad': v.requires_grad} for v in inputs]


def package_files():
    return {name+suffix for name in STAGES for suffix in ('.py', '.json', '.pt')} | {'kernels.cache'}


def read_manifest(directory, expected):
    root = Path(directory).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('runtime') != expected:
        raise ValueError('Compiled SAM mask package does not match this runtime; rebuild it offline')
    stages = manifest.get('stages', {})
    if set(stages) != STAGES or set(manifest.get('files', {})) != package_files():
        raise ValueError('Compiled SAM mask package is missing required stages/files')
    for name, item in manifest['files'].items():
        path = root/name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f'Invalid compiled SAM stage path: {name}')
        if path.stat().st_size != item.get('bytes') or digest(path) != item.get('sha256'):
            raise ValueError(f'Compiled SAM stage failed integrity check: {name}')
    for name, item in stages.items():
        if not isinstance(item.get('inputs'), list) or not item['inputs']:
            raise ValueError(f'Compiled SAM stage has no input contract: {name}')
    return manifest


def pack_weight(value):
    # Copy the physical span, preserving transposes, gaps and zero strides.
    # A plain .cpu() can materialize a view and change its compiled layout.
    shape, stride = list(value.shape), list(value.stride())
    if any(s < 0 for s in stride):
        raise ValueError('Negative compiled weight stride')
    size = sum((n-1)*s for n, s in zip(shape, stride)) + 1 if value.numel() else 0
    return {'data': value.as_strided((size,), (1,)).detach().cpu(),
            'shape': shape, 'stride': stride}


def same_view(left, right):
    return (left.data_ptr() == right.data_ptr() and left.shape == right.shape
            and left.stride() == right.stride() and left.dtype == right.dtype)


class LoadedStage:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self.fn = None

    def __call__(self, *inputs):
        owner, name = self.owner, self.name
        if tensor_contract(inputs) != owner.manifest['stages'][name]['inputs']:
            raise ValueError(f'Compiled SAM input contract changed: {name}')
        if self.fn is None:
            from torch._inductor.codecache import PyCodeCache
            from torch._inductor.utils import align_inputs_from_check_idxs
            self.plan = json.loads((owner.root/(name+'.json')).read_text())
            state = owner.torch.load(owner.root/(name+'.pt'), weights_only=True, map_location='cpu')
            self.weights = {k: v['data'].to(owner.device).as_strided(v['shape'], v['stride'])
                            for k, v in state.items()}
            constants = {k.removeprefix('const_'): v for k, v in self.weights.items()
                         if k.startswith('const_')}
            key, path = PyCodeCache.write((owner.root/(name+'.py')).read_text())
            module = PyCodeCache.load_by_key_path(key, path, attrs=constants)
            self.fn = align_inputs_from_check_idxs(module.call, self.plan['alignment'],
                                                   set(self.plan['mutated']))
        values = [inputs[item['input']] if 'input' in item else self.weights[item['weight']]
                  for item in self.plan['inputs']]
        raw = self.fn(values)
        result = tuple(raw[i] for i in self.plan['outputs'])
        return result[0] if name == 'masks' else result


class MaskArtifactReader:
    def __init__(self, torch, args):
        import torch._inductor.config
        self.torch, self.device = torch, torch.device('xpu', args.device)
        self.root = Path(args.compiled_bundle).resolve()
        self.manifest = read_manifest(self.root, runtime_contract(torch, args))
        torch._inductor.config.freezing = True
        torch.compiler.load_cache_artifacts((self.root/'kernels.cache').read_bytes())

    def stage(self, name, module):
        return LoadedStage(self, name)


class ExportedStage:
    def __init__(self, owner, name, module):
        self.owner, self.name = owner, name
        self.fn = owner.torch.compile(module, fullgraph=True, dynamic=False,
            options={'emulate_precision_casts': True, 'triton.cudagraphs': False})
        self.ready = False

    def __call__(self, *inputs):
        if self.ready:
            return self.fn(*inputs)
        from unittest.mock import patch
        from torch._inductor.output_code import CompiledFxGraph
        captures = []
        original_post = CompiledFxGraph.post_compile

        def capture(graph, examples, constants, kwargs):
            original_post(graph, examples, constants, kwargs)
            data = {'graph': graph, 'constants': constants.unwrap(graph)}
            raw = graph.current_callable

            def record(args):
                if 'inputs' not in data:
                    data['inputs'] = list(args)
                    data['outputs'] = raw(args)
                    return data['outputs']
                return raw(args)

            graph.current_callable = record
            captures.append(data)

        # Offline only, in this isolated worker. Preserve the normal compiler,
        # freezing, kernels and wrappers; record the resulting execution plan.
        with patch.object(CompiledFxGraph, 'post_compile', capture):
            result = self.fn(*inputs)
        if len(captures) != 1:
            raise ValueError(f'{self.name}: expected exactly one compiled region')
        data = captures[0]
        graph = data['graph']
        if graph.torchbind_constants or graph.opaque_value_type_classes:
            raise ValueError('Unsupported opaque constants in compiled SAM stage')
        weights, plan = {}, []
        for j, value in enumerate(data['inputs']):
            if not isinstance(value, self.owner.torch.Tensor):
                raise ValueError('Compiled SAM schedule requires tensor inputs')
            matches = [i for i, v in enumerate(inputs) if same_view(v, value)]
            if matches:
                plan.append({'input': matches[0]})
            else:
                if j in graph.mutated_input_idxs or any(
                    value.untyped_storage().data_ptr() == v.untyped_storage().data_ptr() for v in inputs
                ):
                    raise ValueError('Cannot freeze a mutable argument or an unmapped input view')
                key = 'arg_'+str(j)
                weights[key] = value
                plan.append({'weight': key})
        for key, value in data['constants'].items():
            if not isinstance(value, self.owner.torch.Tensor):
                raise ValueError('Compiled SAM constants must be tensors')
            weights['const_'+key] = value
        visible = (result,) if isinstance(result, self.owner.torch.Tensor) else result
        outputs = []
        for value in visible:
            matches = [i for i, v in enumerate(data['outputs']) if same_view(v, value)]
            if not matches:
                raise ValueError('Output outside compiled SAM region cannot be packaged')
            outputs.append(matches[0])
        root, name = self.owner.root, self.name
        (root/(name+'.py')).write_text(graph.source_code)
        (root/(name+'.json')).write_text(json.dumps({'inputs': plan, 'outputs': outputs,
            'alignment': list(graph.inputs_to_check), 'mutated': list(graph.mutated_input_idxs)}))
        self.owner.torch.save({k: pack_weight(v) for k, v in weights.items()}, root/(name+'.pt'))
        self.owner.entries[name] = {'inputs': tensor_contract(inputs)}
        self.ready = True
        return result


class MaskArtifactWriter:
    """Writes into a caller-owned staging directory; publishing is explicit."""
    def __init__(self, torch, args, directory):
        import torch._inductor.config
        self.torch, self.root = torch, Path(directory)
        self.runtime = runtime_contract(torch, args)
        self.entries = {}
        torch._inductor.config.freezing = True

    def stage(self, name, module):
        if name not in STAGES:
            raise ValueError(f'Unknown mask stage: {name}')
        return ExportedStage(self, name, module)

    def finish(self):
        if set(self.entries) != STAGES:
            raise ValueError('Cannot publish an incomplete compiled SAM package')
        cache = self.torch.compiler.save_cache_artifacts()
        if cache is None:
            raise ValueError('Compiled SAM kernel cache was not captured')
        (self.root/'kernels.cache').write_bytes(cache[0])
        manifest = {'schema': SCHEMA, 'runtime': self.runtime, 'stages': self.entries,
            'files': {name: {'bytes': (self.root/name).stat().st_size,
                             'sha256': digest(self.root/name)} for name in sorted(package_files())}}
        (self.root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        read_manifest(self.root, self.runtime)
        return manifest
