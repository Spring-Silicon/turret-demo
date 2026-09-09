"""Freeze the approved sleepy-joe recipe; no GPU execution or source mutation."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path('/home/spring/springsilicon/graphs')
RUN = ROOT / 'outputs/sam31-native-w4a4/model-fixed-fc1-barrier-gpu1'
OUT = Path(tempfile.mkdtemp(prefix='sam31-w4a4-', dir='/var/tmp'))
report = json.loads((RUN / 'report.json').read_text())

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def copy(source, relative, expected=None):
    if expected and digest(source) != expected:
        raise ValueError(f'Changed source: {source}')
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if expected and digest(target) != expected:
        raise ValueError(f'Copy mismatch: {target}')

for relative, expected in report['source_snapshot_sha256'].items():
    copy(RUN / 'source-snapshot' / relative, relative, expected)
for absolute, expected in report['libraries'].items():
    source = Path(absolute)
    relative = Path('repository') / source.relative_to(ROOT)
    copy(source, relative, expected)
    receipt = source.with_suffix('.build.json')
    if receipt.exists():
        copy(receipt, relative.with_suffix('.build.json'))

# Imported for normalization constants, although not selected for per-frame work.
extra = ROOT / 'outputs/sam31-native-w4a4/normalization/norm-dcc46837207284fc.so'
copy(extra, Path('repository') / extra.relative_to(ROOT))
copy(extra.with_suffix('.build.json'), Path('repository') / extra.with_suffix('.build.json').relative_to(ROOT))
(OUT / 'extra-libraries.json').write_text(json.dumps({str(extra): digest(extra)}, indent=2))

for folder in ('attention', 'attention/row-denominator',
               'attention/local-no-workgroup-barrier', 'attention/no-workgroup-barrier'):
    relative = Path('outputs/sam31-native-w4a4') / folder / 'build.json'
    copy(ROOT / relative, Path('repository') / relative)
    build = json.loads((ROOT / relative).read_text())
    assert build['library'] in report['libraries']
    assert build['sha256'] == report['libraries'][build['library']]

for relative, expected in (
    ('outputs/sam31-theoretical-frontier/calibration.safetensors', report['calibration_sha256']),
    ('outputs/sam31-turret-detector/export-grounding/program.pt2', None),
    ('outputs/sam31-turret-detector/model-comparison-20260907/fixtures/cache-0.safetensors', report['prompt_cache_sha256']),
):
    copy(ROOT / relative, Path('repository') / relative, expected)
copy(Path('/home/spring/sam3_1/sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
     'sam/sam3/assets/bpe_simple_vocab_16e6.txt.gz')
copy(Path('/home/spring/sam3_1/LICENSE'), 'sam/LICENSE')
for name in ('report.json', 'accuracy-vs-float.json', 'prediction-equality.json'):
    copy(RUN / name, Path('receipt') / name)
manifest = json.loads((ROOT / 'outputs/sam31-accuracy-roofline/evaluation/manifest.json').read_text())
rows = manifest['splits']['confirmation'][:8]
predictions = json.loads((RUN / 'predictions-confirmation.json').read_text())
for row in rows:
    copy(Path(row['path']), f"fixtures/{row['image_id']}.jpg", row['sha256'])
(OUT / 'fixtures/expected.json').write_text(json.dumps({str(r['image_id']): predictions[str(r['image_id'])] for r in rows}))
(OUT / 'fixtures/images.json').write_text(json.dumps(rows, indent=2))
files = {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT / 'manifest.json').write_text(json.dumps(files, indent=2) + '\n')
print(json.dumps({'bundle': str(OUT), 'files': len(files), 'bytes': sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()), 'manifest_sha256': digest(OUT / 'manifest.json')}))
