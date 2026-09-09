"""Recovery tooling tests: no remote machines, GPUs or motor commands."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('deployment_bundle', REPO/'tools/deployment_bundle.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class SymlinkTests(unittest.TestCase):
    def test_configuration_cannot_export_login_keys(self):
        config={'servo':{'calibration_file':'/home/spring/.ssh/id_ed25519'},
                'tracking':{'geometry_file':'/home/spring/.local/share/turret-demo/geometry.json'}}
        with self.assertRaisesRegex(ValueError, 'outside the expected state root'):
            bundle.paths_for('arc', config)

    def test_external_target_requires_explicit_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            runtime = base/'runtime'; runtime.mkdir()
            dependency = base/'dependency'; dependency.mkdir()
            (dependency/'kernel.py').write_text('pass\n')
            (runtime/'native').symlink_to(dependency)
            with self.assertRaisesRegex(ValueError, 'Unbundled'):
                bundle.symlink_inventory([runtime])
            result = bundle.symlink_inventory([runtime, dependency])
            self.assertEqual(result[str(runtime/'native')]['provided_by'], 'archive')

    def test_nested_internal_links_and_broken_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'lib').mkdir()
            (root/'lib64').symlink_to('lib')
            self.assertEqual(len(bundle.symlink_inventory([root])), 1)
            (root/'missing').symlink_to('does-not-exist')
            with self.assertRaises(FileNotFoundError):
                bundle.symlink_inventory([root])


@unittest.skipUnless(shutil.which('zstd'), 'zstd required for archive tests')
class VerifyTests(unittest.TestCase):
    def make_bundle(self, root, *, missing=False, wrong_content=False, unsafe=False):
        content = b'known configuration\n'
        with tarfile.open(root/'runtime.tar', 'w') as archive:
            if not missing:
                info = tarfile.TarInfo('../config' if unsafe else 'etc/demo.json')
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        subprocess.run(['zstd', '-q', str(root/'runtime.tar')], check=True)
        compressed = root/'runtime.tar.zst'
        manifest = {'schema':1, 'roots':['/etc/demo.json'],
            'files':{'/etc/demo.json':hashlib.sha256(b'wrong' if wrong_content else content).hexdigest()},
            'archives':{'runtime.tar.zst':{'sha256':bundle.digest(compressed),'bytes':compressed.stat().st_size}}}
        (root/'inventory.json').write_text(json.dumps(manifest))
        (root/'SHA256SUMS').write_text(''.join(f'{bundle.digest(root/name)}  {name}\n'
            for name in ('inventory.json','runtime.tar.zst')))

    def test_verifies_actual_archived_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.make_bundle(root)
            bundle.verify(root)

    def test_rejects_missing_roots_and_wrong_contents_and_path_traversal(self):
        for option in ('missing','wrong_content','unsafe'):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); self.make_bundle(root, **{option:True})
                with self.assertRaises(ValueError):
                    bundle.verify(root)

    def test_rejects_modified_inventory_and_archive(self):
        for name in ('inventory.json','runtime.tar.zst'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); self.make_bundle(root)
                with (root/name).open('ab') as target: target.write(b' ')
                with self.assertRaises(ValueError): bundle.verify(root)


class DeploymentLockTests(unittest.TestCase):
    def test_device_specific_locks_and_shared_hardware_contract(self):
        for profile, serial in [('arc','5B3D045331'),('thor','5B3D044488')]:
            lock = json.loads((REPO/f'deploy/repro/{profile}-20260909.json').read_text())
            self.assertEqual(lock['profile'], profile)
            self.assertEqual(len(lock['inventory_sha256']), 64)
            self.assertIn(serial, lock['config']['servo']['calibration_file'])
            self.assertEqual(lock['config']['servo']['axes']['x']['id'], 2)
            self.assertEqual(lock['config']['servo']['axes']['y']['id'], 1)
            self.assertIn('runtime.tar.zst', lock['archives'])
            self.assertEqual(lock['config']['camera']['width'], 1280)
        dockerfile = (REPO/'deploy/repro/Dockerfile.thor').read_text()
        self.assertIn('--no-deps --no-build-isolation', dockerfile)
        builder = (REPO/'scripts/build-thor-from-recovery.sh').read_text()
        self.assertIn('--network=none', builder)
        self.assertNotIn('docker run', builder)


if __name__ == '__main__': unittest.main()
