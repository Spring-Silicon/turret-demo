"""CPU routing and admission checks; temporal GPU qualification is separate."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spring_turret.detection import validate_config, WorkerClient
from spring_turret.sam31_tracking_native_worker import NativeTrackingEngine
from spring_turret import sam31_tracking_hillclimb as hillclimb

class NativeTrackingTests(unittest.TestCase):
    def config(self, cache):
        return dict(enabled=True, model='sam3.1-tracking',python='/venv/bin/python',
            checkpoint='/models/checkpoint.pt',cache_dir=cache,
            sam31_tracking_bundle='/sam',sam31_tracking_native_bundle='/native')

    def test_explicit_xpu_only(self):
        config=self.config('/cache')
        validate_config(config)
        for change in ({'device_type':'cuda'},{'sam31_tracking_native_bundle':'relative'},
                       {'sam31_tracking_native_bundle':None},{'sam31_tracking_bundle':None}):
            with self.assertRaises(ValueError):validate_config({**config,**change})
        with self.assertRaisesRegex(ValueError,'Intel XPU'):
            NativeTrackingEngine(SimpleNamespace(device_type='cuda'))

    def test_distinct_worker_cache_and_runtime(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch('spring_turret.detection.subprocess.Popen') as popen:
                WorkerClient(self.config(cache)).launch()
                args=popen.call_args.args[0];env=popen.call_args.kwargs['env']
                self.assertTrue(args[2].endswith('/sam31_tracking_native_worker.py'))
                self.assertEqual(env['SPRING_SAM31_TRACKING_NATIVE_BUNDLE'],'/native')
                self.assertTrue(env['LD_LIBRARY_PATH'].startswith('/native/lib:/venv/lib:'))
                self.assertIn('sam31-tracking-native-',env['TORCHINDUCTOR_CACHE_DIR'])
                self.assertEqual(args[args.index('--precision')+1],'bfloat16')

    def test_nontracking_does_not_use_native_tracking(self):
        with tempfile.TemporaryDirectory() as cache:
            for model,worker in [('sam3.1','sam31_worker.py'),('sam3.1-mask','sam31_mask_worker.py')]:
                cfg={**self.config(cache),'model':model}
                with patch('spring_turret.detection.subprocess.Popen') as popen:
                    WorkerClient(cfg).launch()
                    args=popen.call_args.args[0];env=popen.call_args.kwargs['env']
                    self.assertNotIn('sam31_tracking_native_worker.py',args[2])
                    self.assertNotIn('SPRING_SAM31_TRACKING_NATIVE_BUNDLE',env)
                    self.assertNotIn('/native/lib',env.get('LD_LIBRARY_PATH',''))

    def test_unknown_bundle_rejected_before_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'manifest.json').write_text('{}')
            with patch.dict('os.environ',{'SPRING_SAM31_TRACKING_NATIVE_BUNDLE':directory}):
                with self.assertRaisesRegex(ValueError,'Unknown Israel tracking bundle'):
                    NativeTrackingEngine(SimpleNamespace())

    def test_hillclimb_dispatch_requires_pinned_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory, 'manifest.json')
            manifest.write_text('{"recipe":"test"}')
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with patch.dict('os.environ', {'SPRING_SAM31_TRACKING_NATIVE_BUNDLE': directory}), \
                    patch.object(hillclimb, 'MANIFEST_SHA', digest), \
                    patch.object(hillclimb, 'initialize') as initialize:
                args = SimpleNamespace()
                engine = NativeTrackingEngine(args)
                initialize.assert_called_once_with(engine, args, Path(directory).resolve(), None)

    def test_hillclimb_verifies_sources_checkpoint_and_containment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'kernel.so').write_bytes(b'kernel')
            (root/'checkpoint.pt').write_bytes(b'weights')
            manifest = {'files': {'kernel.so': hashlib.sha256(b'kernel').hexdigest()},
                        'checkpoint_sha256': hashlib.sha256(b'weights').hexdigest()}
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            with patch.object(hillclimb, 'MANIFEST_SHA', hashlib.sha256(path.read_bytes()).hexdigest()):
                hillclimb.verify_bundle(root, root/'checkpoint.pt')
                (root/'kernel.so').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'artifact drift'):
                    hillclimb.verify_bundle(root, root/'checkpoint.pt')
                (root/'kernel.so').write_bytes(b'kernel')
                (root/'checkpoint.pt').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                    hillclimb.verify_bundle(root, root/'checkpoint.pt')
            manifest['files'] = {'../outside': 'unused'}
            path.write_text(json.dumps(manifest))
            with patch.object(hillclimb, 'MANIFEST_SHA', hashlib.sha256(path.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, 'escapes its root'):
                    hillclimb.verify_bundle(root, root/'checkpoint.pt')

    def test_native_bundles_have_separate_caches(self):
        from spring_turret.hardware import inference_runtime
        with tempfile.TemporaryDirectory() as directory:
            first = inference_runtime(self.config(directory)).prepare()
            second = inference_runtime({**self.config(directory), 'sam31_tracking_native_bundle':'/new-native'}).prepare()
            self.assertNotEqual(first.environment['TORCHINDUCTOR_CACHE_DIR'],
                                second.environment['TORCHINDUCTOR_CACHE_DIR'])

if __name__=='__main__':unittest.main()
