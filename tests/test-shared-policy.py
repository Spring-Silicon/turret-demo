"""Same completed model results must produce identical hardware commands."""
import copy
import importlib.util
from pathlib import Path
import sys
import subprocess
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.policy import select_candidates, POLICY_VERSION
from spring_turret.session_policy import ManagedSession
from spring_turret.hardware import inference_runtime, XpuRuntime, CudaRuntime
from spring_turret.detection import WorkerClient

spec = importlib.util.spec_from_file_location("tracking_tests", Path(__file__).with_name("test-tracking.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class SharedPolicyTests(unittest.TestCase):
    def test_idle_worker_exits_by_eof_without_term(self):
        worker = WorkerClient({})
        worker.process = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True)
        with patch("spring_turret.detection.os.killpg") as signal_group:
            worker.stop()
            self.assertEqual(worker.process.returncode, 0)
            # The only remaining signal is cleanup of any surviving descendants.
            self.assertTrue(all(call.args[1] == 9 for call in signal_group.call_args_list))

    def test_cancel_wakes_a_stalled_native_receive_before_900_second_deadline(self):
        worker = WorkerClient({})
        worker.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True)
        try:
            worker.cancel()
            worker.cancel_requested_at = time.monotonic() - 3
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                worker.receive(900)
            self.assertLess(time.monotonic() - started, .5)
        finally:
            worker.process.kill()
            worker.process.wait()
            worker.stop()

    def test_different_runtime_implementations_same_interface(self):
        self.assertIsInstance(inference_runtime({"device_type":"xpu"}), XpuRuntime)
        self.assertIsInstance(inference_runtime({"device_type":"cuda"}), CudaRuntime)
        for device in ("xpu", "cuda"):
            with self.assertRaises(ValueError):
                inference_runtime({"device_type":device, "model":"yolo26x"})

    def replay(self, device, model):
        now = [10.]
        detector, servo, camera = fixtures.Detector(lambda:now[0]), fixtures.Servo(), fixtures.Camera()
        detector.data.update(model=model, device_type=device, active_instance_ids=[1,2],
                             temporal_tracking=model == "sam3.1-tracking")
        control = fixtures.TrackingController({"calibrated":True}, detector, servo, camera)
        records = []
        with patch("spring_turret.tracking.time.monotonic", lambda:now[0]):
            control.set_target("cup")
            control.arm()
            def frame(boxes, active=(1,2)):
                now[0] += .5
                detector.captured = now[0] - .02
                detector.data.update(boxes=boxes, active_instance_ids=list(active),
                    frame_sequence=detector.data["frame_sequence"]+1, captured_at=detector.captured,
                    frame_pose={"sampled_at":detector.captured, "axes":{
                        axis:{"degrees":0., "goal_degrees":0.} for axis in ("x","y")}})
                control._tick()
                records.append((copy.deepcopy(servo.calls), control.status()))
            one = {**fixtures.box(cx=.5), "instance_id":1}
            two = {**fixtures.box(cx=.8), "instance_id":2}
            if model != "sam3.1":
                one["mask_centroid"], two["mask_centroid"] = [.52,.53], [.75,.6]
            frame([one,two])
            control.set_instance(1, detector.data["frame_sequence"], 2)
            frame([one,two])
            frame([one])
            frame([one], active=(1,))
            detector.data["revision"] += 1
            frame([one])
            control.disable()
            before = len(servo.calls)
            frame([two])
            self.assertEqual(len(servo.calls), before)
            self.assertEqual(records[-1][1]["policy_version"], POLICY_VERSION)
        return records

    def test_same_inputs_same_target_hold_reacquire_centroid_and_motor_commands(self):
        for model in ("sam3.1", "sam3.1-mask", "sam3.1-tracking"):
            with self.subTest(model=model):
                self.assertEqual(self.replay("xpu",model), self.replay("cuda",model))

    def test_class_isolation_even_if_id_matches_wrong_class(self):
        detection = {"boxes":[{"prompt":"bottle","instance_id":1}], "temporal_tracking":False}
        self.assertEqual(select_candidates(detection,"cup",1,True),([],None,False,False))

    def test_shared_session_retirement_grace_and_dense_native_output_equivalence(self):
        class RawSession:
            def __init__(self):
                self.index, self.visible = 0, True
                self.model, self.trackers, self.metadata = object(), [], {}
            def step(self, pixels):
                self.index += 1
                return {"active_ids":[7], "propagated_ids":[7], "memory_frames":1,
                    "out_obj_ids":np.array([7]), "out_probs":np.array([.9 if self.visible else .1]),
                    "out_boxes_xywh":np.array([[.1,.1,.2,.2]]),
                    "out_binary_masks":np.ones((1,2,2),dtype=bool)}
        for adapter in ("native","dense"):
            raw = RawSession()
            session = ManagedSession(raw, confidence=.5)
            with patch("spring_turret.session_policy.retire_tracks") as retire:
                self.assertEqual(session.step(None,timestamp=0)["active_ids"], [7])
                raw.visible = False
                for i in range(1,17):
                    result = session.step(None,timestamp=i*.4)
                    self.assertEqual(result["active_ids"],[7])
                result = session.step(None,timestamp=7)
                self.assertEqual(result["active_ids"],[])
                self.assertEqual(result["retired_ids"],[7])
                self.assertEqual(result["propagated_ids"],[])
                self.assertEqual(len(result["out_obj_ids"]),0)
                retire.assert_called_once_with(raw.model,raw.trackers,raw.metadata,{7})

    def test_orphaned_tracker_retires_before_propagation_without_resetting_healthy_memory(self):
        healthy_memory = object()
        broken = {"obj_ids": [7, 8], "output_dict": {"cond_frame_outputs": {}}}
        healthy = {"obj_ids": [9], "output_dict": {"cond_frame_outputs": {0: healthy_memory}}}
        raw = SimpleNamespace(index=40, model=object(), trackers=[broken, healthy], metadata={})
        def retire(model, trackers, metadata, ids):
            self.assertEqual(ids, {7, 8})
            trackers.remove(broken)
        def step(pixels):
            # This is the upstream failure condition behind 'No points ...'.
            if any(not t["output_dict"]["cond_frame_outputs"] for t in raw.trackers):
                raise RuntimeError("No points are provided; please add points first")
            raw.index += 1
            return {"active_ids": [9], "propagated_ids": [9], "memory_frames": 1,
                    "out_obj_ids": np.array([9]), "out_probs": np.array([.9]),
                    "out_boxes_xywh": np.array([[.1,.1,.2,.2]]),
                    "out_binary_masks": np.ones((1,2,2), dtype=bool)}
        raw.step = step
        session = ManagedSession(raw, confidence=.5)
        session.previous_active = {7, 8, 9}
        with patch("spring_turret.session_policy.retire_tracks", side_effect=retire) as call:
            result = session.step(None, timestamp=10)
            self.assertEqual(result["retired_ids"], [7, 8])
            self.assertEqual(result["active_ids"], [9])
            self.assertIs(raw.trackers[0]["output_dict"]["cond_frame_outputs"][0], healthy_memory)
            self.assertEqual(raw.index, 41)
            call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
