"""Disconnect/reconnect and run-intent regression tests; no hardware writes."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from spring_turret.server import CameraStream
from spring_turret.servo import DeviceUnavailable, ServoDisarmed
from spring_turret.detection import DetectionController
from spring_turret.tracking import TrackingController

def fixture(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tests'/f'test-{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

servos, detectors, trackers = map(fixture, ('server', 'detection', 'tracking'))


class ServoRecoveryTests(unittest.TestCase):
    setUp = servos.ControllerTests.setUp
    tearDown = servos.ControllerTests.tearDown

    def test_run_intent_survives_disconnect_but_old_goal_does_not(self):
        self.controller.arm()
        self.controller.move('x', 20)
        old_goal = self.packet.registers[2][116]
        self.packet.missing.add(2)
        with self.assertRaises(DeviceUnavailable): self.controller.move('y', 10)
        self.assertFalse(self.controller.armed)
        self.assertTrue(self.controller.status()['recovering'])
        self.packet.missing.clear()
        self.packet.registers[2][132] = 2050
        self.controller._tick_locked()
        self.assertTrue(self.controller.armed)
        self.assertTrue(self.controller.run_requested)
        self.assertEqual(self.packet.registers[2][116], 2050)
        self.assertNotEqual(self.packet.registers[2][116], old_goal)
        self.assertFalse(self.zero_path.exists())

    def test_stop_while_unplugged_prevents_rearm_and_start_can_be_queued(self):
        self.controller.arm()
        self.packet.missing.add(2)
        with self.assertRaises(DeviceUnavailable): self.controller.disable()
        self.assertFalse(self.controller.run_requested)
        self.packet.missing.clear()
        self.controller._tick_locked()
        self.assertFalse(self.controller.armed)
        self.packet.missing.add(2)
        with self.assertRaises(DeviceUnavailable): self.controller.arm()
        self.assertTrue(self.controller.run_requested)
        with self.assertRaises(ServoDisarmed): self.controller.recalibrate()
        self.packet.missing.clear()
        self.controller._tick_locked()
        self.assertTrue(self.controller.armed)

    def test_hardware_fault_is_not_bypassed_by_automatic_retry(self):
        self.controller.arm()
        self.packet.registers[1][70] = 4
        with self.assertRaises(DeviceUnavailable): self.controller.move('x', 0)
        self.packet.writes.clear()
        for _ in range(3):
            with self.assertRaises(DeviceUnavailable): self.controller._tick_locked()
        self.assertTrue(self.controller.run_requested)
        self.assertFalse(self.controller.armed)
        self.assertFalse(any(a == 64 and v == 1 for _, a, v in self.packet.writes))
        self.packet.registers[1][70] = 0
        self.controller._tick_locked()
        self.assertTrue(self.controller.armed)


class LoopRecoveryTests(unittest.TestCase):
    def test_camera_stall_is_reaped_and_capture_restarts(self):
        with tempfile.NamedTemporaryFile() as device:
            camera = CameraStream({'device':device.name,'width':1280,'height':720,
                                   'framerate':30,'restart_delay_seconds':.02})
            command = [sys.executable, '-c',
                       "import sys,time; sys.stdout.buffer.write(b'\\xff\\xd8jpeg\\xff\\xd9'); sys.stdout.flush(); time.sleep(30)"]
            camera._pipeline = lambda:command
            camera.start()
            try:
                detectors.eventually(lambda:camera.latest_sequence >= 1)
                detectors.eventually(lambda:camera.latest_sequence >= 2, timeout=7)
                self.assertGreaterEqual(camera.connection_generation, 2)
                self.assertTrue(camera.status()['online'])
            finally:
                camera.stop()
            self.assertFalse(camera.thread.is_alive())

    def test_worker_failure_retries_without_resubmitting_prompts(self):
        workers = []
        def factory(config):
            worker = detectors.Worker(config)
            worker.fail = not workers
            worker.release.set()
            workers.append(worker)
            return worker
        controller = DetectionController({'enabled':True,'max_fps':30}, detectors.Camera(), factory)
        controller.start()
        try:
            controller.set_prompt('hand')
            revision = controller.revision
            detectors.eventually(lambda:controller.status()['state'] == 'error')
            detectors.eventually(lambda:controller.status()['state'] == 'running', timeout=5)
            self.assertEqual(controller.revision, revision)
            self.assertEqual(controller.prompts, ['hand'])
            self.assertEqual(len(workers), 2)
        finally:
            controller.stop()

    def test_tracking_exception_preserves_target_and_start_intent(self):
        detection = trackers.Detector(time.monotonic)
        servo = trackers.Servo()
        servo.armed = True
        tracker = TrackingController({'calibrated':True}, detection, servo, trackers.Camera())
        tracker.target = 'cup'
        detection.frame_version = lambda:(1,1)
        detection.wait_for_update = lambda *a, **kw:tracker.stop_event.set()
        for error in (DeviceUnavailable('unplugged'), ValueError('bad frame')):
            tracker.stop_event.clear()
            with patch.object(tracker, '_tick', side_effect=error):
                tracker._run()
            self.assertEqual(tracker.target, 'cup')
            self.assertTrue(servo.armed)


if __name__ == '__main__': unittest.main()
