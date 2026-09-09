"""Reject incomplete, corrupt and incompatible offline mask packages."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from spring_turret.sam31_mask_artifacts import STAGES, digest, read_manifest, package_files
from spring_turret.detection import validate_config
from spring_turret.hardware import inference_runtime


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = {'torch': 'pinned', 'confidence': .5, 'sources': {'worker': 'abc'}}
        self.manifest = {'schema': 1, 'runtime': self.runtime,
            'stages': {name: {'inputs': [{'shape': [1], 'dtype': 'torch.float16'}]} for name in STAGES},
            'files': {}}
        for name in package_files():
            path = self.root/name
            path.write_bytes(('compiled '+name).encode())
            self.manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        self.write()

    def write(self):
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))

    def test_complete_package_and_changed_runtime(self):
        self.assertEqual(read_manifest(self.root, self.runtime), self.manifest)
        for key, value in [('torch','another build'), ('confidence',.6), ('sources',{'worker':'new'})]:
            changed = {**self.runtime, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'runtime'):
                read_manifest(self.root, changed)

    def test_corruption_of_same_length_is_rejected(self):
        path = self.root/'image.py'
        path.write_bytes(b'X'*path.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'integrity'):
            read_manifest(self.root, self.runtime)

    def test_partial_package_is_rejected(self):
        del self.manifest['stages']['masks']; self.write()
        with self.assertRaisesRegex(ValueError, 'required stages'):
            read_manifest(self.root, self.runtime)

    def test_external_symlink_is_rejected(self):
        with TemporaryDirectory() as other:
            target = Path(other)/'image.py'
            target.write_bytes((self.root/'image.py').read_bytes())
            (self.root/'image.py').unlink()
            (self.root/'image.py').symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'stage path'):
                read_manifest(self.root, self.runtime)

    def test_configuration_is_explicit_and_scoped_to_arc_mask(self):
        config = {'enabled':True, 'python':'/bin/python', 'checkpoint':'/model.pt',
            'cache_dir':self.temp.name, 'model':'sam3.1-mask', 'sam31_mask_bundle':'/native',
            'sam31_mask_compiled_bundle':'/compiled'}
        validate_config(config)
        spec = inference_runtime(config).prepare()
        self.assertEqual(spec.command[spec.command.index('--compiled-bundle')+1], '/compiled')
        for changed in ({'sam31_mask_compiled_bundle':'relative'}, {'device_type':'cuda'},
                        {'sam31_mask_bundle':None}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_config({**config, **changed})
        spec = inference_runtime({**config, 'model':'sam3.1-tracking', 'sam31_tracking_bundle':'/tracking'}).prepare()
        self.assertNotIn('--compiled-bundle', spec.command)


class WeightLayoutTests(unittest.TestCase):
    def test_physical_layout_survives_cpu_package(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch is checked separately in the inference environment')
        from spring_turret.sam31_mask_artifacts import pack_weight
        base = torch.arange(24).reshape(4, 6)
        for value in (base, base.t(), base[:, ::2], base[:1].expand(4, 6), base[1:, 1:], base[:0]):
            with self.subTest(shape=value.shape, stride=value.stride()):
                packed = pack_weight(value)
                restored = packed['data'].as_strided(packed['shape'], packed['stride'])
                self.assertEqual(restored.stride(), value.stride())
                self.assertTrue(torch.equal(restored, value))


if __name__ == '__main__':
    unittest.main()
