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
        self.data = {"enabled": True, "prompts": ["cup", "bottle"], "revision": 1,
                     "frame_sequence": 0, "state": "running", "boxes": []}
    def status(self): return {**copy.deepcopy(self.data), "frame_age_ms": (self.clock() - self.captured) * 1000}
    def set_prompts(self, prompts):
        self.data.update(prompts=prompts, revision=self.data["revision"] + 1, state="loading", boxes=[])
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
    def track(self, offsets):
        if not self.armed: raise ServoDisarmed("stopped")
        self.calls.append(dict(offsets))
        return {"limited": self.limited, "goal_degrees": dict(offsets)}
    point = track
    def move(self, axis, degrees): self.manual.append((axis, degrees))


class TrackingTests(unittest.TestCase):
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
        self.frame([box(cx=.505, cy=.495)])
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

    def test_target_loss_or_stale_frames_hold_once_without_searching(self):
        for condition in ("lost", "stale", "camera", "detector"):
            with self.subTest(condition=condition):
                self.start()
                self.camera.online = True
                self.frame([box()])
                if condition == "lost": self.frame([])
                if condition == "stale": self.now += 1
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
        self.assertEqual(self.tracker.state, "lost")
        self.assertEqual(self.servo.calls[-1], {"x": 0, "y": 0})
        self.assertEqual(self.tracker.instance_id, 20)
        self.frame([right, left])
        self.assertEqual(self.tracker.box, right)

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
        self.assertIsNone(self.tracker.target)

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
