"""Freeze sleepy-joe's qualified skip4_attention8 add-on; never alter its campaign."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path('/home/spring/springsilicon/graphs')
RUN = ROOT / 'outputs/sam31-native-w4a4/masks/w8a8/default-gpu0'
SOURCE = ROOT / 'ci/benchmarks/sam31_native/masks/w8a8'
OUT = Path(tempfile.mkdtemp(prefix='sam31-mask-attention8-', dir='/var/tmp'))

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def copy(source, relative, expected=None):
    source, relative = Path(source), Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts
    if expected is not None and digest(source) != expected:
        raise ValueError(f'Source drift: {source}')
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if expected is not None:
        assert digest(target) == expected

selection = json.loads((SOURCE / 'selection.json').read_text())
session = json.loads((RUN / 'session.json').read_text())
report = json.loads((RUN / 'report.json').read_text())
optimization = session['image_optimization']
assert selection['recipe'] == session['image_recipe'] == 'skip4_attention8'
assert selection['projection_variant'] == optimization['tile_variant'] == 0
assert selection['skipped_blocks'] == optimization['skipped_blocks'] == [24, 26, 28, 30]
assert selection['library_sha256'] == optimization['library_sha256']
assert selection['calibration_sha256'] == optimization['calibration_sha256']
copy(SOURCE / 'selection.json', 'selection.json')
copy(ROOT / selection['paired_report'], 'paired-result.json', selection['paired_report_sha256'])
for name in ('session.json', 'result.json', 'report.json'):
    copy(RUN / name, name)
for name, checksum in optimization['sources'].items():
    copy(RUN / 'executed/w8a8' / name, Path('source') / name, checksum)
# The upstream receipt did not snapshot this weight-preparation dependency.
# Freeze the current file explicitly instead of claiming recorded provenance.
quantization = ROOT / 'ci/benchmarks/sam31_tradeoffs/quantization.py'
copy(quantization, 'source/quantization.py', digest(quantization))
(OUT / 'supplementary-source.json').write_text(json.dumps({
    'path': str(quantization), 'sha256': digest(quantization),
    'provenance': 'current source-host checkout; absent from upstream run snapshot'
}, indent=2) + '\n')
for name in ('README.md', 'commands.md'):
    copy(SOURCE / name, name)
library = Path(optimization['library'])
copy(library, 'lib/projections.so', optimization['library_sha256'])
copy(library.with_suffix('.build.json'), 'lib/projections.build.json')
calibration = Path(optimization['calibration'])
copy(calibration, 'calibration.safetensors', optimization['calibration_sha256'])
copy(calibration.with_suffix('.json'), 'calibration.json')
copy(ROOT / 'outputs/sam31-accuracy-roofline/evaluation/manifest.json', 'dataset-manifest.json',
     optimization['calibration_receipt']['dataset_manifest_sha256'])

# Relocation only: keep original source alongside runtime copies. Replace the
# compiler invocation with the exact qualified library and localize imports.
runtime = OUT / 'runtime'
runtime.mkdir()
for name in ('install.py', 'native.py', 'quantization.py'):
    shutil.copy2(OUT / 'source' / name, runtime / name)
changes = {
    'install.py': [
        ('from bootstrap import ROOT, digest',
         'import hashlib\nROOT = Path(__file__).resolve().parent.parent\n'
         'def digest(path):\n    with Path(path).open("rb") as stream:\n'
         '        return hashlib.file_digest(stream, "sha256").hexdigest()'),
        ('from sam31_tradeoffs.quantization import Int8Linear', 'from .quantization import Int8Linear'),
        ('ROOT / "outputs/sam31-attention-selective/calibration.safetensors"', 'ROOT / "calibration.safetensors"'),
    ],
    'native.py': [
        ('from sam31_native.build import load', ''),
        ('LIBRARY, LIBRARY_PATH = load("projections", FLAGS, source_dir=Path(__file__).parent)',
         'LIBRARY_PATH = Path(__file__).resolve().parent.parent / "lib/projections.so"\n'
         'LIBRARY = ctypes.CDLL(str(LIBRARY_PATH))'),
    ],
}
for name, replacements in changes.items():
    path = runtime / name
    text = path.read_text()
    for before, after in replacements:
        assert text.count(before) == 1, (name, before)
        text = text.replace(before, after)
    path.write_text(text)
    compile(text, str(path), 'exec')
(OUT / 'relocation.json').write_text(json.dumps(changes, indent=2) + '\n')
files = {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT / 'manifest.json').write_text(json.dumps(files, indent=2) + '\n')
print(json.dumps({'bundle': str(OUT), 'files': len(files), 'manifest_sha256': digest(OUT / 'manifest.json')}))
