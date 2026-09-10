"""CPU-only tests for the additional temporal profile and hardware mapping."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spring_turret.detection import validate_config
from spring_turret.hardware import inference_runtime
from spring_turret.models import model_available, is_tracking_model
from spring_turret.policy import continuity
from spring_turret.sam31_tracking_v18 import verify_bundle
from spring_turret.sam31_tracking_v18_worker import V18TrackingEngine


class V18Tests(unittest.TestCase):
    def config(self, cache='/cache'):
        return dict(enabled=True, model='sam3.1-v18', python='/venv/bin/python',
            checkpoint='/weights', cache_dir=cache, sam31_tracking_bundle='/sam',
            sam31_tracking_v18_bundle='/v18', sam31_tracking_native_bundle='/old-native')

    def test_admission(self):
        config=self.config()
        validate_config(config)
        self.assertTrue(model_available('sam3.1-v18',config))
        for change in ({'device_type':'cuda'}, {'sam31_tracking_v18_bundle':'relative'},
                       {'sam31_tracking_v18_bundle':None}, {'sam31_tracking_bundle':None}):
            with self.assertRaises(ValueError): validate_config({**config,**change})
        self.assertFalse(model_available('sam3.1-v18',{**config,'device_type':'cuda'}))
        self.assertFalse(model_available('sam3.1-v18',{**config,'sam31_tracking_v18_bundle':None}))
        with self.assertRaisesRegex(ValueError,'Intel XPU'):
            V18TrackingEngine(SimpleNamespace(device_type='cuda'))

    def test_distinct_worker_and_cache(self):
        with tempfile.TemporaryDirectory() as cache:
            config=self.config(cache)
            spec=inference_runtime(config).prepare()
            self.assertTrue(spec.command[2].endswith('/sam31_tracking_v18_worker.py'))
            self.assertIn('sam31-tracking-israel-full1008-v18',spec.environment['TORCHINDUCTOR_CACHE_DIR'])
            self.assertEqual(spec.environment['SPRING_SAM31_TRACKING_NATIVE_BUNDLE'],'/v18')
            self.assertTrue(spec.environment['LD_LIBRARY_PATH'].startswith('/v18/lib:/venv/lib:'))
            self.assertEqual(spec.command[spec.command.index('--precision')+1],'bfloat16')
            self.assertEqual(spec.command[spec.command.index('--source-bundle')+1],'/sam')
            old=inference_runtime({**config,'model':'sam3.1-tracking'}).prepare()
            self.assertTrue(old.command[2].endswith('/sam31_tracking_native_worker.py'))
            self.assertEqual(old.environment['SPRING_SAM31_TRACKING_NATIVE_BUNDLE'],'/old-native')

    def test_temporal_policy_and_bundle_pin(self):
        self.assertTrue(is_tracking_model('sam3.1-v18'))
        self.assertEqual(continuity('sam3.1-v18'),'temporal-id')
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Unknown Israel'):
                verify_bundle(directory,'/weights')

    def test_no_stale_frame_or_unbounded_diagnostics(self):
        engine=V18TrackingEngine.__new__(V18TrackingEngine)
        stage=SimpleNamespace(calls=2,direct_calls=1,replay_calls=1)
        engine.graph_stages={'image_and_detection':stage}
        engine.cache_warmup_frames=0
        engine.receipt={'image_backend':'israel-full1008-native-v18-temporal'}
        engine.model=SimpleNamespace()
        result=engine.after_detect({},1)
        self.assertEqual(result['image_passes_per_frame'],1)
        with self.assertRaisesRegex(RuntimeError,'stale image'):
            engine.after_detect({},2)


if __name__ == '__main__': unittest.main()
