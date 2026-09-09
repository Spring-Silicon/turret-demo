"""Relocate a frozen bundle, replacing only build steps with pinned DSO loads."""
import ast
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

BUILDERS = {
    'int4/native_int4.py': 'int4/int4-',
    'integration/native_pack.py': 'integration/pack-',
    'int4/native_fused.py': 'int4/fused-',
    'int4/native_fused_vnni.py': 'int4/fused-vnni-',
    'int4/native_int4_vnni.py': 'int4/int4-vnni-',
    'int4/native_int4_residual.py': 'int4/int4-residual-',
    'int4/new/native_pipeline.py': 'int4/new/int4-pipeline-residual-',
    'int4/new/native_fused_pipeline.py': 'int4/new/fused-pipeline-',
    'int4/lut32/native_lut32.py': 'components/fused_lut32-',
    'int4/fixed_shape_unroll/native_fixed_shape_unroll.py': 'components/fused_fixed_shape_unroll-',
    'attention/native_rope.py': 'attention/rope-',
    'routing/native_router.py': 'routing/router-',
    'normalization/native_residual_norm_pack.py': 'normalization/residual-norm-pack-',
    'normalization/native_norm.py': 'normalization/norm-',
    'normalization/compensated-pack/native_compensated_norm_pack.py': 'normalization/compensated-pack/residual-norm-pack-',
    'qkv-vnni/native_qkv_vnni.py': 'components/qkv_vnni-',
    'qkv-vnni/paired-k32/native_qkv_k32.py': 'components/qkv_k32-',
}

def freeze_builder(source, library_relative):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build']
    if functions:
        function, = functions
        returns = [n for n in function.body if isinstance(n, ast.Return)]
        ret, = returns
        handle, path = [n.id for n in ret.value.elts]
        indices = [i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and n.value.func.attr in ('CDLL', 'load')]
        index, = indices
        prefix = ast.parse(f'import ctypes\n{path} = Path(__file__).parent / {library_relative!r}\n{handle} = ctypes.CDLL(str({path}))').body
        function.body = prefix + function.body[index + 1:]
    else:
        matches = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'builder'
                   and n.value.func.attr == 'load']
        node, = matches
        node.value = ast.parse(f'(ctypes.CDLL(str(Path(__file__).parent / {library_relative!r})), Path(__file__).parent / {library_relative!r})', mode='eval').body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + '\n'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(bundle):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for relative, expected in manifest.items():
        if digest(bundle / relative) != expected:
            raise ValueError(f'Corrupt bundle: {relative}')
    runtime = bundle / 'runtime'
    runtime.mkdir(exist_ok=False)
    shutil.copytree(bundle / 'repository', runtime / 'repository')
    shutil.copytree(bundle / 'sam', runtime / 'sam')
    root = runtime / 'repository/outputs/sam31-native-w4a4'
    report = json.loads((bundle / 'receipt/report.json').read_text())
    report['libraries'].update(json.loads((bundle / 'extra-libraries.json').read_text()))
    libraries = [Path(p).relative_to('/home/spring/springsilicon/graphs/outputs/sam31-native-w4a4')
                 for p in report['libraries'] if '/outputs/sam31-native-w4a4/' in p]
    changed = {}
    selected_root = Path('/home/spring/springsilicon/graphs/outputs/sam31-native-w4a4')
    explicit = {
        'int4/pointer_induction/native_pointer_induction.py': Path(report['pointer_fc2']['build']['command'][-1]).relative_to(selected_root),
        'int4/reciprocal/fc1_fastinv/native_schedule.py': Path(report['fast_reciprocal']['fc1']['library']).relative_to(selected_root),
        'normalization/fast-inv/native_fastinv_norm_pack.py': Path(report['fast_reciprocal']['norm']['library']).relative_to(selected_root),
        'qkv-vnni/persistent-payload/native_qkv_payload.py': Path(report['qkv_payload']['library']).relative_to(selected_root),
    }
    BUILDERS.update({name: None for name in explicit})
    for relative, prefix in BUILDERS.items():
        matches = [explicit[relative]] if relative in explicit else [p for p in libraries if str(p).startswith(prefix)
                    and len(str(p)[len(prefix):]) == 19
                    and all(c in '0123456789abcdef' for c in str(p)[len(prefix):-3])]
        library, = matches
        source = root / relative
        before = digest(source)
        source.write_text(freeze_builder(source.read_text(), os.path.relpath(root / library, source.parent)))
        changed[relative] = {'original_sha256': before, 'runtime_sha256': digest(source), 'library': str(library)}
    # Resolve source receipt paths against the relocated repository, not cwd.
    for name in ('native_mapping.py', 'qkv_backend.py', 'fc1_backend.py'):
        source = runtime / 'repository/ci/benchmarks/sam31_native/hillclimb75' / name
        before = digest(source)
        text = source.read_text()
        assert text.count('Path(path).read_bytes()') == 1
        prefix = '/home/spring/springsilicon/graphs/'
        text = text.replace('Path(path).read_bytes()',
            f'(Path(__file__).resolve().parents[6] / "repository" / str(path).removeprefix({prefix!r})).read_bytes()')
        # Library membership is verified against the original path, loading
        # the same hashed DSO from the relocated directory afterward.
        if name == 'qkv_backend.py':
            text = text.replace('path = Path(build["library"]).resolve(strict=True)', 'path = Path(build["library"])')
            text = text.replace('library = ctypes.CDLL(str(path))',
                f'path = Path(__file__).resolve().parents[4] / str(path).removeprefix({prefix!r})\n    library = ctypes.CDLL(str(path))')
        elif name == 'fc1_backend.py':
            text = text.replace('if str(path) not in build["files_sha256"]:',
                f'if {prefix!r} + str(path.relative_to(Path(__file__).resolve().parents[4])) not in build["files_sha256"]:')
        source.write_text(text)
        changed[name] = {'original_sha256': before, 'runtime_sha256': digest(source)}
    (bundle / 'relocation.json').write_text(json.dumps(changed, indent=2) + '\n')
    files = {str(p.relative_to(bundle)): digest(p) for p in sorted(runtime.rglob('*')) if p.is_file()}
    files['relocation.json'] = digest(bundle / 'relocation.json')
    (bundle / 'runtime-manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    print(json.dumps({'runtime_manifest_sha256': digest(bundle / 'runtime-manifest.json'), 'replaced_builders': len(changed)}))

if __name__ == '__main__':
    main(Path(sys.argv[1]))
