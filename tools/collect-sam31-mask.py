"""Freeze the completed mask qualification without changing campaign files."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path('/home/spring/springsilicon/graphs')
RUN = ROOT / 'outputs/sam31-native-w4a4/masks/tiled-qualification-gpu1'
OUT = Path(tempfile.mkdtemp(prefix='sam31-mask-', dir='/var/tmp'))
report = json.loads((RUN/'report.json').read_text())
session = json.loads((RUN/'session.json').read_text())
selection = session['selected_detector']

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def copy(source, relative, expected=None):
    source, relative = Path(source), Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts
    if expected and digest(source) != expected: raise ValueError(f'Changed source: {source}')
    target = OUT/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if expected: assert digest(target) == expected

def repository(path, expected=None):
    path = Path(path)
    source = path if path.is_absolute() else ROOT/path
    copy(source, Path('repository')/source.relative_to(ROOT), expected)

for relative, expected in report['source_snapshot_sha256'].items():
    copy(RUN/'source-snapshot'/relative, relative, expected)
for path, expected in report['libraries'].items():
    repository(path, expected)
    receipt = Path(path).with_suffix('.build.json')
    if receipt.exists(): repository(receipt)

def dependencies(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('files_sha256', 'sources', 'source_sha256') and isinstance(item, dict):
                for path, expected in item.items():
                    if isinstance(expected, str) and len(expected)==64 and (ROOT/path).is_file():
                        repository(path, expected)
            elif key == 'candidate_directory' and isinstance(item,str):
                directory = Path(item)
                if (directory/'build.json').is_file(): repository(directory/'build.json')
            dependencies(item)
    elif isinstance(value,list):
        for item in value: dependencies(item)
dependencies(report)
for name in ('native_mapping.py','qkv_backend.py','fc1_backend.py','fc2_weight_layout.py'):
    repository(ROOT/'ci/benchmarks/sam31_native/hillclimb75'/name)
for path in ('outputs/sam31-theoretical-frontier/calibration.safetensors',
             'outputs/sam31-turret-detector/export-grounding/program.pt2',
             'outputs/sam31-turret-detector/model-comparison-20260907/fixtures/cache-0.safetensors'):
    repository(path)
for folder in ('attention','attention/row-denominator','attention/local-no-workgroup-barrier','attention/no-workgroup-barrier'):
    repository(f'outputs/sam31-native-w4a4/{folder}/build.json')
extra=ROOT/'outputs/sam31-native-w4a4/normalization/norm-dcc46837207284fc.so'
repository(extra); repository(extra.with_suffix('.build.json'))
(OUT/'extra-libraries.json').write_text(json.dumps({str(extra):digest(extra)}))
copy('/home/spring/sam3_1/sam3/assets/bpe_simple_vocab_16e6.txt.gz','sam/sam3/assets/bpe_simple_vocab_16e6.txt.gz')
copy('/home/spring/sam3_1/LICENSE','sam/LICENSE')
for original,expected in session['sources'].items():
    copy(RUN/'executed'/Path(original).name, Path('masks')/Path(original).name,expected)
for name in ('result.json','report.json','session.json','mask-predictions.json','example.safetensors'):
    copy(RUN/name,Path('receipt')/name)
copy(ROOT/'ci/benchmarks/sam31_native/masks/README.md','receipt/README.md')
(OUT/'receipt/selection.json').write_text(json.dumps(selection,indent=2)+'\n')
manifest=json.loads((ROOT/'outputs/sam31-accuracy-roofline/evaluation/manifest.json').read_text())
rows=manifest['splits']['confirmation'][:3]
for row in rows: copy(row['path'],f"fixtures/{row['image_id']}.jpg",row['sha256'])
(OUT/'fixtures/images.json').write_text(json.dumps(rows))
files={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps({'bundle':str(OUT),'files':len(files),'manifest_sha256':digest(OUT/'manifest.json')}))
