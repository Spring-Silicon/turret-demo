#!/usr/bin/env python3
"""A source sync must not silently invalidate pinned Arc native artifacts."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release', ROOT / 'tools/build_app_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_arc_overrides_match_qualified_bytes(self):
        for name, digest in release.ARC_OVERRIDES.items():
            self.assertEqual(release.digest(ROOT / 'deploy/overlays/arc-20260910' / name), digest)

    def test_requires_checked_out_full_commit_and_clean_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            root.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
            git('init', '-q')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                'commit', '--allow-empty', '-qm', 'test')
            commit = git('rev-parse', 'HEAD')
            output = Path(directory) / 'release'
            with self.assertRaisesRegex(ValueError, 'exact full'):
                release.build(root, output, 'arc', commit[:12])
            (root / 'unreviewed').write_text('not committed')
            with self.assertRaisesRegex(ValueError, 'dirty'):
                release.build(root, output, 'arc', commit)
            self.assertFalse(output.exists())

    def test_thor_build_preserves_runtime_dependencies(self):
        docker = (ROOT / 'deploy/repro/Dockerfile.app-overlay').read_text()
        self.assertIn('FROM ${BASE_IMAGE}', docker)
        self.assertIn('--no-index --no-deps', docker)
        self.assertNotIn('apt-get', docker)


if __name__ == '__main__':
    unittest.main()
