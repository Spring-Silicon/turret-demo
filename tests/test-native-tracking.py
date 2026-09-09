"""CPU routing and admission checks; temporal GPU qualification is separate."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spring_turret.detection import validate_config, WorkerClient
from spring_turret.sam31_tracking_native_worker import NativeTrackingEngine

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
                self.assertIn('sam31-tracking-israel-native672-v24',env['TORCHINDUCTOR_CACHE_DIR'])
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

if __name__=='__main__':unittest.main()
