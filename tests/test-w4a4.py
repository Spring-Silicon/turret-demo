#!/usr/bin/env python3
"""CPU-only opt-in, isolation and shared-image graph contracts."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spring_turret.detection import validate_config, WorkerClient
from spring_turret.sam31_w4a4 import verify_bundle, CHECKPOINT, MANIFEST, RUNTIME_MANIFEST
from spring_turret.sam31_w4a4_worker import ModelGraph, W4A4Engine


class W4A4Tests(unittest.TestCase):
    def config(self, cache='/cache'):
        return dict(enabled=True, python='/venv/bin/python', checkpoint='/weights', cache_dir=cache,
                    sam31_w4a4_bundle='/bundle', sam31_allow_w4a4_accuracy_tradeoff=True,
                    sam31_tracking_bundle='/tracking')

    def test_explicit_tradeoff_and_exclusive_xpu_bundle(self):
        good = self.config()
        validate_config(good)
        for change in ({'sam31_allow_w4a4_accuracy_tradeoff': False},
                       {'sam31_allow_w4a4_accuracy_tradeoff': 'true'},
                       {'sam31_w4a4_bundle': 'relative'}, {'device_type': 'cuda'},
                       {'precision': 'bfloat16'}, {'sam31_native_bundle': '/old'},
                       {'sam31_w8a8_development_bundle': '/old'}):
            with self.assertRaises(ValueError):
                validate_config({**good, **change})
        del good['sam31_w4a4_bundle']
        with self.assertRaises(ValueError):
            validate_config(good)
        with self.assertRaises(ValueError):
            W4A4Engine(Path('/weights'), Path('/bundle'))

    def test_only_nontracking_sam_gets_new_worker(self):
        with tempfile.TemporaryDirectory() as cache, patch('spring_turret.detection.subprocess.Popen') as popen:
            config = self.config(cache)
            WorkerClient(config).launch()
            args, env = popen.call_args.args[0], popen.call_args.kwargs['env']
            self.assertTrue(args[2].endswith('/sam31_w4a4_worker.py'))
            self.assertIn('--w4a4-bundle', args)
            self.assertIn('--allow-w4a4-accuracy-tradeoff', args)
            self.assertEqual(env['TORCHINDUCTOR_FREEZING'], '1')
            self.assertIn('sam31-sleepy-w4a4', env['TORCHINDUCTOR_CACHE_DIR'])
            for model, worker in (('sam3.1-mask', 'sam31_mask_worker.py'), ('sam3.1-tracking', 'sam31_tracking_worker.py')):
                WorkerClient({**config, 'model': model}).launch()
                args = popen.call_args.args[0]
                self.assertNotIn('--w4a4-bundle', args)
                self.assertNotIn('--allow-w4a4-accuracy-tradeoff', args)
                self.assertFalse(args[2].endswith('/sam31_w4a4_worker.py'))
                if model != 'sam3.1-mask': self.assertNotIn('TORCHINDUCTOR_FREEZING', popen.call_args.kwargs['env'])

    def test_one_image_multiple_independent_heads(self):
        torch, image, head = MagicMock(), MagicMock(), MagicMock()
        image.return_value = ('features', 'position')
        head.return_value = (MagicMock(), MagicMock(), MagicMock())
        stage = ModelGraph(torch, image, head, lambda *args: None)
        stage._compile(())
        stage.compiled('pixels', 'memory1', 'mask1', 'memory2', 'mask2')
        image.assert_called_once_with('pixels')
        self.assertEqual(head.call_count, 2)
        self.assertEqual(head.call_args_list[0].args, ('features', 'position', 'memory1', 'mask1'))
        self.assertEqual(head.call_args_list[1].args, ('features', 'position', 'memory2', 'mask2'))

    def test_manifest_and_checkpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            for name in ('manifest.json', 'runtime-manifest.json'):
                (bundle / name).write_text(json.dumps({'file': 'hash'}))
            pins = {'weights': CHECKPOINT, 'manifest.json': MANIFEST, 'runtime-manifest.json': RUNTIME_MANIFEST, 'file': 'hash'}
            for corrupt in (None, *pins):
                def digest(path):
                    return 'bad' if path.name == corrupt else pins[path.name]
                with patch('spring_turret.sam31_w4a4.digest', side_effect=digest):
                    if corrupt:
                        with self.assertRaises(ValueError):
                            verify_bundle(bundle, Path('/weights'))
                    else:
                        verify_bundle(bundle, Path('/weights'))

    def test_failed_graph_validation_is_not_activated(self):
        torch, image, head, tensor = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        image.return_value = ('features', 'position')
        head.return_value = (tensor, tensor, tensor)
        torch.testing.assert_close = MagicMock(side_effect=AssertionError('replay changed'))
        stage = ModelGraph(torch, image, head, lambda *args: None)
        with self.assertRaises(AssertionError):
            stage(tensor, tensor, tensor)
        self.assertIsNone(stage.graph)

    def test_prompt_recapture_can_reuse_the_native_queue(self):
        torch, stream = MagicMock(), MagicMock()
        torch.testing.assert_close = MagicMock()
        stage = ModelGraph(torch, MagicMock(return_value=('features','position')),
                           MagicMock(return_value=(MagicMock(),)), lambda *args:None, stream)
        stage(MagicMock(), MagicMock(), MagicMock())
        torch.xpu.Stream.assert_not_called()
        self.assertIs(stage.stream, stream)


if __name__ == '__main__':
    unittest.main()
