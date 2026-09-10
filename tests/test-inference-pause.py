"""Pause is backend-owned, drains in-flight work, and retains the worker."""
import importlib.util
from pathlib import Path
import threading
import time
import unittest

spec = importlib.util.spec_from_file_location('fixtures', Path(__file__).with_name('test-detection.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
from spring_turret.detection import DetectionController


class PauseTests(unittest.TestCase):
    def controller(self):
        self.worker = fixtures.Worker({})
        controller = DetectionController({'enabled': True, 'max_fps': 30}, fixtures.Camera(), lambda _: self.worker)
        controller.start()
        self.addCleanup(controller.stop)
        return controller

    def test_pause_holds_last_result_and_retains_worker_prompts_revision(self):
        c = self.controller()
        self.worker.release.set()
        c.set_prompts(['hand'])
        fixtures.eventually(lambda: c.status()['state'] == 'running')
        c.set_paused(True)
        before = c.status()
        requests = len(self.worker.requests)
        time.sleep(.15)
        after = c.status()
        self.assertEqual(after['state'], 'paused')
        self.assertEqual(len(self.worker.requests), requests)
        for key in ('frame_url', 'frame_sequence', 'prompts', 'revision', 'model'):
            self.assertEqual(after[key], before[key])
        self.assertIs(c.worker, self.worker)
        self.assertFalse(self.worker.stopped)
        c.set_paused(False)
        fixtures.eventually(lambda: c.status()['state'] == 'running' and c.status()['frame_sequence'] > before['frame_sequence'])
        self.assertGreaterEqual(c.status()['captured_at'], c.resume_after)
        self.assertIs(c.worker, self.worker)

    def test_rapid_pause_resume_discards_pre_resume_inflight_result(self):
        c = self.controller()
        c.set_prompts(['hand'])
        self.assertTrue(self.worker.entered.wait(1))
        first = self.worker.requests[0][0]
        c.set_paused(True)
        c.set_paused(False)
        self.worker.release.set()
        fixtures.eventually(lambda: c.status()['state'] == 'running')
        self.assertGreater(c.status()['frame_sequence'], first)
        self.assertGreaterEqual(c.status()['captured_at'], c.resume_after)

    def test_pause_before_loading_and_prompt_model_changes_do_not_start_work(self):
        c = self.controller()
        c.config['sam31_mask_bundle'] = '/mask'
        c.set_paused(True)
        c.set_prompts(['hand'])
        c.set_model('sam3.1-mask')
        c.set_prompts(['cup'])
        time.sleep(.15)
        self.assertIsNone(c.worker)
        self.assertEqual(c.status()['state'], 'paused')
        self.assertEqual(c.status()['prompts'], ['cup'])
        self.worker.release.set()
        c.set_paused(False)
        fixtures.eventually(lambda: c.status()['state'] == 'running')
        self.assertEqual(self.worker.requests[-1][2], ['cup'])

    def test_pause_during_load_prevents_first_inference(self):
        c = self.controller()
        loading, ready = threading.Event(), threading.Event()
        def receive(timeout):
            loading.set()
            ready.wait(2)
            return {'type': 'ready'}
        self.worker.receive = receive
        c.set_prompts(['hand'])
        self.assertTrue(loading.wait(1))
        c.set_paused(True)
        ready.set()
        time.sleep(.15)
        self.assertFalse(self.worker.requests)
        self.assertIs(c.worker, self.worker)
        self.worker.release.set()
        c.set_paused(False)
        fixtures.eventually(lambda: c.status()['state'] == 'running')

    def test_pause_validation_and_idempotence(self):
        c = self.controller()
        for value in (1, 0, None, 'true', [], {}):
            with self.assertRaises(ValueError): c.set_paused(value)
        c.set_paused(True)
        revision = c.revision
        c.set_paused(True)
        self.assertEqual(c.revision, revision)
        c.set_paused(False)
        cutoff = c.resume_after
        c.set_paused(False)
        self.assertEqual(c.resume_after, cutoff)
        with self.assertRaises(ValueError):
            DetectionController({}, fixtures.Camera()).set_paused(True)


if __name__ == '__main__': unittest.main()
