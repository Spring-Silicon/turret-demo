#!/usr/bin/env python3
"""Camera-framing policy tests with simulated detections and no motor hardware."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.tracking import TrackingController, nearest_box, validate_config
from spring_turret.servo import ServoDisarmed


def box(prompt="cup", cx=0.7, cy=0.5, score=0.9):
    return {"prompt": prompt, "score": score, "xyxy": [cx - .04, cy - .04, cx + .04, cy + .04]}


class Camera:
    online = True
    def status(self): return {"online": self.online, "width": 1280, "height": 720}


class Detector:
    def __init__(self, clock):
        self.clock, self.captured = clock, 0
        self.data = {"enabled": True, "model": "sam3.1", "prompts": ["cup", "bottle"], "revision": 1,
                     "frame_sequence": 0, "state": "running", "boxes": []}
    def status(self): return {**copy.deepcopy(self.data), "frame_age_ms": (self.clock() - self.captured) * 1000}
    @property
    def model(self): return self.data["model"]
    def set_prompts(self, prompts):
        self.data.update(prompts=prompts, revision=self.data["revision"] + 1, state="loading", boxes=[])
    def set_model(self, model):
        if model not in ("sam3.1", "sam3.1-mask"): raise ValueError("unknown model")
        self.data["model"] = model
        self.set_prompts(["person"])
    def selection(self, revision, sequence, instance_id):
        if (revision, sequence) != (self.data["revision"], self.data["frame_sequence"]):
            raise ValueError("stale frame")
        return next(b for b in self.data["boxes"] if b["instance_id"] == instance_id)


class Servo:
    def __init__(self):
        self.armed, self.online, self.limited = False, True, False
        self.calls, self.manual = [], []
        self.goals = {"x": 0.0, "y": 0.0}
    def status(self): return {"armed": self.armed, "online": self.online,
                             "axes": {a: {"goal_degrees": v} for a, v in self.goals.items()}}
    def arm(self): self.armed = True
    def disable(self): self.armed = False
    def recalibrate(self):
        if self.armed: raise ServoDisarmed("Stop first")
        self.goals = {"x": 0, "y": 0}
    def track(self, offsets):
        if not self.armed: raise ServoDisarmed("stopped")
        self.calls.append(dict(offsets))
        return {"limited": self.limited, "goal_degrees": dict(offsets)}
    point = track
    def move(self, axis, degrees): self.manual.append((axis, degrees))


class TrackingTests(unittest.TestCase):
    def test_mask_dropout_reacquires_moved_target_without_old_bearing_gate(self):
        self.persistent_setup()
        self.detection.data.update(model="sam3.1-mask", temporal_tracking=False)
        self.start()
        first = {**box(cx=.5), "instance_id":10, "mask_centroid":[.5,.5]}
        self.frame([first])
        self.assertEqual(self.tracker.state, "centered")
        self.assertIsNone(self.tracker.instance_id)
        self.frame([])
        calls = len(self.servo.calls)
        for _ in range(5): self.frame([])
        self.assertEqual(len(self.servo.calls), calls)  # One hold, no scanning.
        self.assertEqual(self.tracker.state, "lost")
        moved = {**box(cx=.7), "instance_id":99, "mask_centroid":[.7,.5]}
        wrong_class = {**box(prompt="bottle", cx=.5), "instance_id":20}
        self.frame([wrong_class, moved])
        self.assertEqual(self.tracker.box["instance_id"], 99)
        self.assertEqual(self.tracker.state, "tracking")
        self.assertEqual(self.tracker.status()["continuity"], "nearest-of-class")
        self.assertIsNone(self.tracker.status()["hold_reason"])

    def test_mask_clicked_id_preferred_until_lost_then_nearest_class_resumes(self):
        self.persistent_setup()
        self.detection.data.update(model="sam3.1-mask", temporal_tracking=False)
        self.start()
        near = {**box(cx=.51), "instance_id":10}
        far = {**box(cx=.7), "instance_id":20}
        self.frame([near, far])
        self.tracker.set_instance(1, self.detection.data["frame_sequence"], 20)
        self.frame([near, far])
        self.assertEqual(self.tracker.box["instance_id"], 20)
        self.frame([near])
        self.assertEqual(self.tracker.box["instance_id"], 10)
        self.assertIsNone(self.tracker.instance_id)
        self.assertEqual(self.tracker.status()["selection"], "class")
        self.tracker.disable()
        calls = len(self.servo.calls)
        self.frame([far])
        self.assertFalse(self.servo.armed)
        self.assertEqual(len(self.servo.calls), calls)

    def test_shared_temporal_policy_holds_active_identity(self):
        self.persistent_setup()
        self.detection.data.update(model="sam3.1-tracking", temporal_tracking=True, active_instance_ids=[10,20])
        self.start()
        first = {**box(cx=.5), "instance_id":10}
        self.frame([first])
        self.frame([{**box(cx=.7), "instance_id":20}])
        self.assertEqual(self.tracker.state, "lost")
        self.assertEqual(self.tracker.instance_id, 10)
        self.assertEqual(self.tracker.status()["continuity"], "temporal-id")

    def persistent_setup(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("geometry_test", Path(__file__).with_name("test-geometry.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.tracker.geometry = module.Geometry(module.fixture())
        self.tracker.config["hold_reacquire"] = True
        original = self.servo.status
        def status():
            result = original()
            for axis, binding in module.fixture()["axes"].items():
                result["axes"][axis].update(binding)
            return result
        self.servo.status = status
        self.camera.status = lambda: {"online": True, **module.fixture()["camera"]}
        return module

    def test_persistent_camera_rotation_is_not_target_motion(self):
        module = self.persistent_setup()
        self.start()
        reference = module.pose()
        geometry = self.tracker.geometry
        expected = geometry.goals(740, 380, reference, reference)
        for x, y in [(0,0), (12,0), (-12,0), (0,12), (0,-12), (8,8)]:
            coords = []
            for u, v in [(700,350), (780,410)]:
                pu, pv = geometry.reproject(u, v, reference, module.pose(x,y))
                coords.append((pu/1280, pv/720))
            u, v = geometry.reproject(740, 380, reference, module.pose(x,y))
            current = {"prompt":"cup", "score":.9, "instance_id":10,
                       "xyxy":[min(p[0] for p in coords),min(p[1] for p in coords),
                               max(p[0] for p in coords),max(p[1] for p in coords)],
                       "mask_centroid":[u/1280,v/720]}
            self.frame([current], x=x, y=y)
            self.assertEqual(self.tracker.state, "tracking")
            for axis in expected:
                self.assertAlmostEqual(self.servo.calls[-1][axis], expected[axis], places=5)

    def test_persistent_manual_model_prompt_recalibration_update_intent(self):
        self.persistent_setup()
        self.start()
        self.frame([{**box(cx=.5), "instance_id":10}])
        self.tracker.manual_move("x", 5)
        self.assertIsNone(self.tracker.target)
        self.start()
        self.assertIsNone(self.tracker.instance_id)
        self.frame([{**box(cx=.5), "instance_id":20}])
        self.tracker.set_prompts(["cup"])
        self.assertIsNone(self.tracker.instance_id)
        self.start()
        self.frame([{**box(cx=.5), "instance_id":30}])
        self.tracker.set_model("sam3.1-mask")
        self.assertIsNone(self.tracker.instance_id)
        self.assertEqual(self.tracker.target, "cup")

    def test_temporal_centering_keeps_identity_and_expiry_allows_reacquisition(self):
        self.detection.data.update(temporal_tracking=True, active_instance_ids=[10, 20])
        self.start()
        self.frame([{**box(cx=.5), "instance_id":10}, {**box(cx=.8), "instance_id":20}])
        self.assertEqual(self.tracker.instance_id, 10)
        self.assertEqual(self.tracker.status()["selection"], "class")
        self.frame([{**box(cx=.51), "instance_id":20}])
        self.assertEqual(self.tracker.state, "lost")
        self.assertEqual(self.tracker.instance_id, 10)
        # An absent box alone cannot retarget. Only the worker retiring that
        # identity (after the occlusion grace) makes it eligible to reacquire.
        self.detection.data.update(active_instance_ids=[20], retired_instance_ids=[10])
        self.frame([{**box(cx=.6), "instance_id":20}])
        self.assertEqual(self.tracker.instance_id, 20)
        self.assertEqual(self.tracker.state, "tracking")
        self.assertEqual(self.tracker.status()["selection"], "class")
        self.tracker.disable()
        calls = len(self.servo.calls)
        self.detection.data.update(active_instance_ids=[30])
        self.frame([{**box(cx=.6), "instance_id":30}])
        self.assertEqual(len(self.servo.calls), calls)
        self.assertFalse(self.servo.armed)

    def test_temporal_tracking_locks_nearest_then_holds_through_occlusion(self):
        self.detection.data["temporal_tracking"] = True
        self.start()
        selected = {**box(cx=.6), "instance_id": 10}
        neighbor = {**box(cx=.8), "instance_id": 20}
        self.frame([selected, neighbor])
        self.assertEqual(self.tracker.instance_id, 10)
        self.frame([neighbor])
        self.assertEqual(self.tracker.state, "lost")
        self.assertEqual(self.tracker.instance_id, 10)
        self.assertEqual(self.servo.calls[-1], {"x":0, "y":0})
        self.frame([selected, {**neighbor, "xyxy":[.49,.49,.51,.51]}])
        self.assertEqual(self.tracker.box["instance_id"], 10)
        self.tracker.set_instance(1, self.detection.data["frame_sequence"], 20)
        self.frame([selected, neighbor])
        self.assertEqual(self.tracker.box["instance_id"], 20)
        self.tracker.set_target("cup")
        self.assertIsNone(self.tracker.instance_id)
        self.frame([selected, neighbor])
        self.assertEqual(self.tracker.instance_id, 10)

    def test_geometry_is_used_and_bad_camera_binding_holds(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location("geometry_test",Path(__file__).with_name("test-geometry.py"))
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.tracker.geometry=module.Geometry(module.fixture())
        old_status=self.servo.status
        def servo_status():
            s=old_status()
            for k,v in module.fixture()["axes"].items(): s["axes"][k].update(v)
            return s
        self.servo.status=servo_status
        self.camera.status=lambda:{"online":True,**module.fixture()["camera"]}
        self.start()
        self.frame([box(cx=.7,cy=.5)],x=10,y=35)
        self.assertEqual(self.tracker.status()["mapping"],"fisheye-kinematics")
        expected=self.tracker.geometry.goals(.7*1280,.5*720,self.detection.data["frame_pose"],self.servo.status())
        for name in expected: self.assertAlmostEqual(self.servo.calls[-1][name],expected[name])
        unconstrained_status=self.servo.status
        def bounded_status():
            s=unconstrained_status()
            for axis in s['axes'].values():
                axis.update(min_degrees=-5,max_degrees=5)
            return s
        self.servo.status=bounded_status
        self.frame([box(cx=.9,cy=.7)],x=0,y=0)
        self.assertEqual(self.tracker.state,'limited')
        # The geometric boundary is reported even though the motor layer did
        # not need to clamp the already-feasible result.
        self.assertFalse(self.servo.limited)
        self.assertTrue(all(-5<=v<=5 for v in self.servo.calls[-1].values()))
        self.camera.status=lambda:{"online":True,**module.fixture()["camera"],"identity":"different"}
        self.frame([box(cx=.8)])
        self.assertEqual(self.tracker.state,"uncalibrated")
        self.assertIn("identity",self.tracker.error)
        self.assertEqual(self.servo.calls[-1],{"x":0,"y":0})

    def test_bearing_filter_removes_camera_motion_before_filtering_and_resets(self):
        import importlib.util
        from spring_turret.bearing_filter import BearingFilter
        spec=importlib.util.spec_from_file_location('geometry_test',Path(__file__).with_name('test-geometry.py'))
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        g=module.Geometry(module.fixture())
        self.tracker.geometry=g
        self.tracker.bearing_filter=BearingFilter(.5,40)
        old_status=self.servo.status
        def servo_status():
            s=old_status()
            for k,v in module.fixture()['axes'].items():s['axes'][k].update(v)
            return s
        self.servo.status=servo_status
        self.camera.status=lambda:{'online':True,**module.fixture()['camera']}
        self.start()
        reference=module.pose()
        expected=g.goals(800,420,reference,reference)
        for x,y in [(0,0),(5,0),(-5,0),(0,5),(0,-5)]:
            u,v=g.reproject(800,420,reference,module.pose(x,y))
            self.frame([{**box(cx=u/1280,cy=v/720),'instance_id':8}],x=x,y=y)
            for axis in ('x','y'):
                self.assertAlmostEqual(self.servo.calls[-1][axis],expected[axis],places=5)
        self.assertIsNotNone(self.tracker.bearing_filter.direction)
        self.tracker.disable()
        self.assertIsNone(self.tracker.bearing_filter.direction)
        self.tracker.arm()
        self.frame([{**box(cx=.6),'instance_id':8}])
        self.tracker.set_target('bottle')
        self.assertIsNone(self.tracker.bearing_filter.direction)

    def test_recalibration_clears_old_coordinate_tracking_without_moving(self):
        self.start()
        self.frame([box()])
        self.tracker.instance_id = 42
        with self.assertRaises(ServoDisarmed): self.tracker.recalibrate()
        self.assertEqual(self.tracker.target, "cup")
        self.servo.armed = False
        before = self.servo.calls.copy()
        self.tracker.recalibrate()
        self.assertEqual(self.servo.calls, before)
        self.assertIsNone(self.tracker.target)
        self.assertIsNone(self.tracker.instance_id)
        self.assertIsNone(self.tracker.previous_pose)
        self.assertIsNone(self.tracker.last_frame)
        self.assertEqual(self.tracker.state, "off")
        self.assertFalse(self.servo.armed)

    def test_model_change_keeps_class_clears_instance_and_holds_without_arming(self):
        self.start()
        self.frame([box()])
        self.assertTrue(self.tracker.moving)
        self.tracker.instance_id = 42
        self.tracker.set_model("sam3.1-mask")
        self.assertEqual(self.tracker.target, "cup")
        self.assertIsNone(self.tracker.instance_id)
        self.assertEqual(self.servo.calls[-1], {"x": 0, "y": 0})
        self.servo.armed = False
        self.tracker.set_model("sam3.1")
        self.assertFalse(self.servo.armed)
        self.assertEqual(self.tracker.state, "waiting")

    def setUp(self):
        self.now = 10.0
        self.timer = patch("spring_turret.tracking.time.monotonic", lambda: self.now)
        self.timer.start()
        self.addCleanup(self.timer.stop)
        self.camera, self.servo = Camera(), Servo()
        self.detection = Detector(lambda: self.now)
        self.tracker = TrackingController({"calibrated": True}, self.detection, self.servo, self.camera)

    def frame(self, boxes, age=100, x=0.0, y=0.0, goal_x=0.0, goal_y=0.0):
        self.now += .4
        self.detection.captured = self.now - age / 1000
        self.detection.data.update(frame_sequence=self.detection.data["frame_sequence"] + 1,
                                   boxes=boxes, state="running", captured_at=self.detection.captured,
                                   frame_pose={"sampled_at": self.detection.captured - .01,
                                               "axes": {"x": {"degrees": x, "goal_degrees": goal_x},
                                                        "y": {"degrees": y, "goal_degrees": goal_y}}})
        self.tracker._tick()

    def start(self):
        self.tracker.set_target("cup")
        self.tracker.arm()

    def test_nearest_matching_class_uses_pixel_distance_not_normalized_distance(self):
        horizontal, vertical = box(cx=.6), box(cx=.5, cy=.65)
        closest_other_class = box("bottle", cx=.5)
        self.assertIs(nearest_box([horizontal, closest_other_class, vertical], "cup", 1280, 720), vertical)
        self.assertIsNone(nearest_box([closest_other_class], "cup", 1280, 720))
        self.assertIs(nearest_box([box(score=.6), horizontal], "cup", 1280, 720), horizontal)

    def test_malformed_boxes_are_ignored(self):
        invalid = [None, {}, {**box(), "xyxy": [0, 1]}, {**box(), "score": float("nan")}]
        for coords in ([0, 0, 0, 1], [0, 1, 1, 0], [False, 0, 1, 1], [-.1, 0, 1, 1], [0, 0, float("inf"), 1]):
            invalid.append({**box(), "xyxy": coords})
        self.assertIsNone(nearest_box(invalid, "cup", 1280, 720))

    def test_selection_never_arms_and_stop_never_rearms(self):
        self.tracker.set_target("cup")
        self.frame([box()])
        self.assertFalse(self.servo.armed)
        self.assertEqual(self.servo.calls, [])
        self.tracker.arm()
        self.frame([box()])
        self.assertAlmostEqual(self.servo.calls[-1]["x"], 32)
        self.assertEqual(self.servo.calls[-1]["y"], 0)
        self.tracker.disable()
        before = list(self.servo.calls)
        self.frame([box()])
        self.assertEqual(self.servo.calls, before)
        self.assertFalse(self.servo.armed)
        self.assertEqual(self.tracker.state, "stopped")

    def test_direction_deadband_and_uncapped_corrections(self):
        self.start()
        self.frame([box(cx=.55, cy=.4)])
        self.assertAlmostEqual(self.servo.calls[-1]["x"], 8)
        self.assertAlmostEqual(self.servo.calls[-1]["y"], 9)
        self.tracker.config.update(x_direction=-1, y_direction=1)
        self.frame([box(cx=.9, cy=.1)])
        self.assertAlmostEqual(self.servo.calls[-1]["x"], -64)
        self.assertAlmostEqual(self.servo.calls[-1]["y"], -36)
        calls = len(self.servo.calls)
        self.frame([box(cx=.502, cy=.498)])
        self.assertEqual(self.servo.calls[-1], {"x": 0, "y": 0})
        self.assertEqual(self.tracker.state, "centered")
        calls = len(self.servo.calls)
        self.frame([box(cx=.5, cy=.5)])
        self.assertEqual(self.servo.calls[-1], {"x": 0, "y": 0})

    def test_follows_nearest_instance_on_each_fresh_frame(self):
        self.start()
        left, right = box(cx=.3), box(cx=.6)
        self.frame([left, right, box("bottle", cx=.5)])
        self.assertEqual(self.tracker.box, right)
        self.assertGreater(self.servo.calls[-1]["x"], 0)
        left, right = box(cx=.45), box(cx=.8)
        self.frame([left, right])
        self.assertEqual(self.tracker.box, left)
        self.assertLess(self.servo.calls[-1]["x"], 0)

    def test_same_frame_is_not_reused_but_every_new_motion_frame_is_processed(self):
        self.start()
        self.frame([box()])
        calls = len(self.servo.calls)
        for _ in range(4): self.tracker._tick()
        self.assertEqual(len(self.servo.calls), calls)
        # This new frame was captured while the preceding correction was in
        # flight. There is no settling delay or post-move capture-time barrier.
        self.now += .2
        self.detection.captured = self.now - .2
        self.detection.data["frame_sequence"] += 1
        self.tracker._tick()
        self.assertEqual(len(self.servo.calls), calls + 1)
        self.frame([box()])
        self.assertEqual(len(self.servo.calls), calls + 2)

    def test_frames_from_before_start_or_class_selection_never_move(self):
        self.start()
        self.frame([box()], age=500)
        self.assertEqual(self.servo.calls, [])
        self.frame([box()])
        self.assertTrue(self.servo.calls)
        self.tracker.set_target("bottle")
        calls = len(self.servo.calls)
        self.frame([box("bottle")], age=500)
        self.assertEqual(len(self.servo.calls), calls)
        self.frame([box("bottle")])
        self.assertEqual(len(self.servo.calls), calls + 1)

    def test_slow_detections_move_once_without_age_cutoff_or_timeout_hold(self):
        self.start()
        for age in (1500, 10000, 60000):
            with self.subTest(age=age):
                self.now += age / 1000 + 1  # Frame was captured after Start.
                before = len(self.servo.calls)
                self.frame([box()], age=age)
                self.assertEqual(len(self.servo.calls), before + 1)
                self.assertEqual(self.tracker.state, "tracking")
                self.now += 60
                self.tracker._tick()
                self.assertEqual(len(self.servo.calls), before + 1)  # Never replay old correction.
        self.tracker.disable()
        self.now += 2
        self.frame([box()], age=1500)
        self.assertFalse(self.servo.armed)
        self.assertEqual(self.tracker.state, "stopped")

    def test_target_loss_or_faults_hold_once_without_searching(self):
        for condition in ("lost", "camera", "detector", "future"):
            with self.subTest(condition=condition):
                self.start()
                self.camera.online = True
                self.frame([box()])
                if condition == "lost": self.frame([])
                if condition == "future": self.detection.captured = self.now + 1
                if condition == "camera": self.camera.online = False
                if condition == "detector": self.detection.data["state"] = "error"
                self.tracker._tick()
                self.assertEqual(self.servo.calls[-1], {"x": 0, "y": 0})
                calls = len(self.servo.calls)
                self.tracker._tick()
                self.assertEqual(len(self.servo.calls), calls)
                self.assertTrue(self.servo.armed)
                self.tracker.set_target(None)

    def test_select_clear_switch_and_prompt_removal(self):
        for value in (False, 1, [], "", "unapplied"):
            with self.assertRaises(ValueError): self.tracker.set_target(value)
        self.start()
        self.frame([box()])
        self.tracker.set_target("bottle")
        self.assertEqual(self.tracker.target, "bottle")
        self.assertEqual(self.servo.calls[-1], {"x": 0, "y": 0})
        self.frame([box("bottle")])
        self.tracker.set_prompts(["cup"])
        self.assertIsNone(self.tracker.target)
        self.assertEqual(self.tracker.state, "off")
        self.tracker.set_target("cup")
        self.tracker.set_prompts(["cup", "bottle"])
        self.assertEqual(self.tracker.target, "cup")
        self.assertEqual(self.tracker.state, "waiting")

    def test_manual_controls_take_over_without_overwriting_their_goal(self):
        self.start()
        self.frame([box()])
        calls = len(self.servo.calls)
        self.tracker.manual_move("y", 45)
        self.assertEqual(self.servo.manual, [("y", 45)])
        self.assertIsNone(self.tracker.target)
        self.assertEqual(len(self.servo.calls), calls)
        self.frame([box()])
        self.assertEqual(len(self.servo.calls), calls)

    def test_angle_limit_is_reported(self):
        self.start()
        self.servo.limited = True
        self.frame([box()])
        self.assertEqual(self.tracker.state, "limited")

    def test_clicked_instance_overrides_nearest_and_can_switch_within_same_class(self):
        left, right = {**box(cx=.3), "instance_id": 10}, {**box(cx=.6), "instance_id": 20}
        self.frame([left, right])
        self.tracker.set_instance(1, self.detection.data["frame_sequence"], 10)
        self.assertFalse(self.servo.armed)
        self.assertEqual(self.servo.calls, [])
        self.tracker.arm()
        self.frame([right, left])
        self.assertEqual(self.tracker.box, left)
        self.assertLess(self.servo.calls[-1]["x"], 0)
        self.tracker.set_instance(1, self.detection.data["frame_sequence"], 20)
        self.frame([left, right])
        self.assertEqual(self.tracker.box, right)
        self.assertGreater(self.servo.calls[-1]["x"], 0)
        self.frame([left])
        self.assertEqual(self.tracker.state, "tracking")
        self.assertEqual(self.tracker.box, left)
        self.assertLess(self.servo.calls[-1]["x"], 0)
        self.assertIsNone(self.tracker.instance_id)
        self.frame([right, left])
        self.assertEqual(self.tracker.box, right)

    def test_click_retargets_until_centered_then_resumes_original_nearest_tracking(self):
        selected = {**box(cx=.3), "instance_id": 10}
        self.frame([selected])
        self.tracker.set_instance(1, self.detection.data["frame_sequence"], 10)
        self.tracker.arm()
        self.frame([selected, {**box(cx=.55), "instance_id": 20}])
        self.assertEqual(self.tracker.box, selected)
        self.assertEqual(self.tracker.status()["selection"], "retarget")
        centered = {**box(cx=.5), "instance_id": 10}
        self.frame([centered])
        self.assertIsNone(self.tracker.instance_id)
        self.assertEqual(self.tracker.target, "cup")
        self.assertEqual(self.tracker.status()["selection"], "class")
        other = {**box(cx=.52), "instance_id": 20}
        self.frame([{**selected, "xyxy": [.75, .45, .85, .55]}, other])
        self.assertEqual(self.tracker.box, other)

    def test_lost_clicked_id_does_not_block_reacquiring_a_new_instance(self):
        selected = {**box(cx=.3), "instance_id": 10}
        self.frame([selected])
        self.tracker.set_instance(1, self.detection.data["frame_sequence"], 10)
        self.tracker.arm()
        self.frame([selected])
        self.frame([])
        self.assertEqual(self.tracker.state, "lost")
        self.assertIsNone(self.tracker.instance_id)
        self.assertEqual(self.servo.calls[-1], {"x": 0, "y": 0})
        self.frame([{**box(cx=.6), "instance_id": 999}])
        self.assertEqual(self.tracker.state, "tracking")
        self.assertEqual(self.tracker.box["instance_id"], 999)
        self.assertTrue(self.servo.armed)

    def test_class_selection_manual_and_prompt_change_clear_instance_mode(self):
        selected = {**box(cx=.3), "instance_id": 10}
        def pick():
            self.frame([selected])
            self.tracker.set_instance(self.detection.data["revision"], self.detection.data["frame_sequence"], 10)
        pick()
        self.tracker.set_target("cup")
        self.assertIsNone(self.tracker.instance_id)
        self.assertEqual(self.tracker.target, "cup")
        pick()
        self.tracker.manual_move("x", 4)
        self.assertIsNone(self.tracker.instance_id)
        self.assertIsNone(self.tracker.target)
        pick()
        self.tracker.set_prompts(["cup"])
        self.assertIsNone(self.tracker.instance_id)
        self.assertEqual(self.tracker.target, "cup")

    def test_full_angle_is_based_on_capture_pose_not_accumulated_goals(self):
        self.start()
        self.frame([box(cx=.75)], x=10, goal_x=80)
        self.assertAlmostEqual(self.servo.calls[-1]["x"], 50)
        # Camera has moved 20 degrees since capture; the next frame's error
        # falls by 20 degrees. The estimated absolute destination stays at 50.
        self.frame([box(cx=.625)], x=30, goal_x=50)
        self.assertAlmostEqual(self.servo.calls[-1]["x"], 50)

    def test_centered_frame_brakes_an_overshooting_goal(self):
        self.start()
        self.frame([box(cx=.5)], x=20, goal_x=45)
        self.assertAlmostEqual(self.servo.calls[-1]["x"], 20)

    def test_centered_frame_keeps_current_hold_not_an_older_capture_goal(self):
        self.start()
        self.servo.goals["x"] = 20.2
        self.frame([box(cx=.5)], x=20, goal_x=19.8)
        self.assertAlmostEqual(self.servo.calls[-1]["x"], 20.2)

    def test_settled_pose_learns_bias_but_motion_does_not(self):
        self.start()
        self.frame([box(cx=.5, cy=.48)], y=8, goal_y=10)
        self.frame([box(cx=.5, cy=.48)], y=8, goal_y=10)
        self.assertAlmostEqual(self.servo.calls[-1]["y"], 11.8)
        self.assertEqual(self.tracker.hold_bias["y"], 2)
        self.tracker.set_target(None)
        self.start()
        self.frame([box(cx=.5, cy=.48)], y=4, goal_y=10)
        self.frame([box(cx=.5, cy=.48)], y=8, goal_y=10)
        self.assertEqual(self.tracker.hold_bias["y"], 0)
        self.assertAlmostEqual(self.servo.calls[-1]["y"], 9.8)

    def test_missing_malformed_old_or_unpaired_pose_holds_instead_of_guessing(self):
        for mutate in (
            lambda d: d.update(frame_pose=None),
            lambda d: d["frame_pose"].update(sampled_at=0),
            lambda d: d["frame_pose"].update(sampled_at=d["captured_at"] - .2),
            lambda d: d["frame_pose"]["axes"]["x"].update(degrees=float("nan")),
            lambda d: d["frame_pose"]["axes"].pop("y"),
        ):
            self.start()
            self.frame([box()])
            self.detection.data["frame_sequence"] += 1
            mutate(self.detection.data)
            self.tracker._tick()
            self.assertEqual(self.tracker.state, "waiting")
            self.assertEqual(self.servo.calls[-1], {"x": 0, "y": 0})
            self.tracker.set_target(None)

    def test_tracking_config_is_bounded(self):
        validate_config({})
        for config in ({"x_direction": 0}, {"y_direction": True}, {"max_step_degrees": 3}, {"settle_seconds": .1},
                       {"max_frame_age_seconds": 10}, {"x_degrees_per_frame": float("nan")}, {"deadband": 0},
                       {"x_gain": 24}, {"y_degrees_per_frame": 181},
                       {"hold_reacquire": 1}, {"hold_reacquire": True},
                       {"unknown": 1}, {"calibrated": 1}, []):
            with self.assertRaises(ValueError): validate_config(config)

    def test_uncalibrated_directions_never_move(self):
        self.tracker = TrackingController({}, self.detection, self.servo, self.camera)
        self.start()
        self.frame([box()])
        self.assertEqual(self.servo.calls, [])
        self.assertEqual(self.tracker.state, "uncalibrated")


if __name__ == "__main__":
    unittest.main()
