#!/usr/bin/env python3
"""Build a dependency-free application release from a clean, pinned Git checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ARC_OVERRIDES = {
    'sam31_graph.py': 'a242a33cc1bfd28cc1df69f4d2565f70c6fe6677dfdffc07bbaa54037da20776',
    'sam31_w4a4.py': '315907150997b890b69525fc79f1125fa27e6d8a561f2782e573e632fd849eb4',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(repo, output, role, commit):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    if role not in ('arc', 'thor'):
        raise ValueError('role must be arc or thor')
    if not re.fullmatch('[0-9a-f]{40}', commit) or git('rev-parse', 'HEAD') != commit:
        raise ValueError('Check out the exact full source commit first')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Source checkout is dirty; review and commit changes first')
    if output.exists() or output.resolve().is_relative_to(repo.resolve()):
        raise ValueError('Choose a new output directory outside the checkout')
    output.mkdir(parents=True)
    app = output / 'app'
    app.mkdir()
    for name in ('pyproject.toml', 'README.md', 'MANIFEST.in'):
        shutil.copy2(repo / name, app / name)
    for name in ('src', 'scripts'):
        shutil.copytree(repo / name, app / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
    overrides = {}
    if role == 'arc':
        for name, expected in ARC_OVERRIDES.items():
            source = repo / 'deploy/overlays/arc-20260910' / name
            if digest(source) != expected:
                raise ValueError(f'Arc runtime override changed: {name}')
            shutil.copy2(source, app / 'src/spring_turret' / name)
            overrides[name] = expected
    metadata = app / 'pyproject.toml'
    content = metadata.read_text()
    version = re.search(r'^version = "([^"]+)"$', content, re.M).group(1)
    version += f'+g{commit[:12]}.{role}'
    metadata.write_text(re.sub(r'^version = "[^"]+"$', f'version = "{version}"', content, flags=re.M))
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                    '--no-index', '--wheel-dir', str(output / 'dist'), str(app)], check=True)
    wheel, = (output / 'dist').glob('*.whl')
    files = {}
    for directory in ('src/spring_turret', 'scripts'):
        for path in sorted((app / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                files[str(path.relative_to(app))] = digest(path)
    receipt = {'schema': 1, 'source_commit': commit, 'role': role, 'version': version,
               'wheel': wheel.name, 'wheel_sha256': digest(wheel),
               'runtime_overrides': overrides, 'files': files}
    (output / 'release.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('source_commit', 'role', 'wheel', 'wheel_sha256')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('arc', 'thor'), required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.output.resolve(), args.role, args.commit)
