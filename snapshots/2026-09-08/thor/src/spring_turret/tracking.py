"""Opt-in framing from completed detections; never arms or renews the lease."""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

from spring_turret.servo import DeviceUnavailable, ServoDisarmed
from spring_turret.geometry import Geometry

LOGGER = logging.getLogger("spring-turret.tracking")
DEFAULTS = {
    "calibrated": False,
    "x_direction": 1, "y_direction": -1,
    "x_degrees_per_frame": 160.0, "y_degrees_per_frame": 90.0,
    "deadband": 0.012,
    "max_frame_age_seconds": 0.75,  # Legacy config accepted but no longer expires detections.
    "geometry_file": None,
}


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("tracking contains unsupported settings")
    merged = {**DEFAULTS, **config}
    if merged["geometry_file"] is not None and (type(merged["geometry_file"]) is not str or not Path(merged["geometry_file"]).is_absolute()):
        raise ValueError("tracking.geometry_file must be an absolute path or null")
    if type(merged["calibrated"]) is not bool:
        raise ValueError("tracking.calibrated must be a boolean")
    for axis in ("x", "y"):
        value = merged[f"{axis}_direction"]
        if type(value) is not int or value not in (-1, 1):
            raise ValueError(f"tracking.{axis}_direction must be -1 or 1")
    for key, (low, high) in {
        "x_degrees_per_frame": (1, 360), "y_degrees_per_frame": (1, 180),
        "deadband": (0.001, 0.1), "max_frame_age_seconds": (0.1, 1),
    }.items():
        value = merged[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"tracking.{key} must be between {low} and {high}")


def nearest_box(boxes: list[dict], prompt: str, width: int, height: int) -> dict | None:
    """Choose box-center distance in pixels, not normalized square-image space."""
    candidates = []
    for box in boxes:
        if not isinstance(box, dict) or box.get("prompt") != prompt:
            continue
        coords, score = box.get("xyxy"), box.get("score")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            continue
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in coords):
            continue
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            continue
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        distance = ((cx - 0.5) * width) ** 2 + ((cy - 0.5) * height) ** 2
        candidates.append((distance, -score, box))
    return min(candidates, key=lambda item: item[:2])[2] if candidates else None


class TrackingController:
    def __init__(self, config: dict, detection: Any, servo: Any, camera: Any):
        validate_config(config)
        self.config = {**DEFAULTS, **config}
        self.geometry = Geometry.load(self.config["geometry_file"]) if self.config["geometry_file"] else None
        self.detection, self.servo, self.camera = detection, servo, camera
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera-tracking", daemon=True)
        self.target: str | None = None
        self.instance_id: int | None = None
        self.state = "off"
        self.error: str | None = None
        self.box: dict | None = None
        self.frame: tuple[int, int] | None = None
        self.last_frame: tuple[int, int] | None = None
        self.ignore_before = 0.0
        self.moving = False
        self.error_pixels: list[float] | None = None
        self.previous_pose: dict | None = None
        self.hold_bias = {"x": 0.0, "y": 0.0}
        self.goal_degrees: dict | None = None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3)

    def _pause(self, state: str) -> None:
        if self.moving:
            try:
                self.servo.track({"x": 0.0, "y": 0.0})
            except ServoDisarmed:
                pass
        self.moving = False
        self.previous_pose = None
        self.hold_bias = {"x": 0.0, "y": 0.0}
        self.goal_degrees = None
        self.state, self.box, self.frame, self.error_pixels = state, None, None, None

    def set_target(self, prompt: Any) -> None:
        with self.lock:
            detection = self.detection.status()
            if prompt is not None and (
                type(prompt) is not str or not prompt or
                prompt not in detection.get("prompts", []) or not detection.get("enabled")
            ):
                raise ValueError("target must be one of the applied object classes, or null")
            self._select(prompt, None)

    def set_instance(self, revision: int, sequence: int, instance_id: int) -> None:
        with self.lock:
            box = self.detection.selection(revision, sequence, instance_id)
            self._select(box["prompt"], instance_id)

    def _select(self, prompt: str | None, instance_id: int | None) -> None:
        if (prompt, instance_id) == (self.target, self.instance_id):
            return
        # Clear first even if holding fails: never leave stale automatic intent.
        self.target = self.instance_id = None
        was_moving = self.moving
        self._pause("off")
        if prompt and not was_moving and self.servo.status()["armed"]:
            self.servo.track({"x": 0.0, "y": 0.0})
        self.target, self.instance_id = prompt, instance_id
        self.last_frame = None
        self.ignore_before = time.monotonic()
        self.error = None
        self.state = "waiting" if prompt else "off"

    def set_prompts(self, prompts: Any) -> None:
        with self.lock:
            self.detection.set_prompts(prompts)
            if self.instance_id is not None or self.target not in self.detection.status()["prompts"]:
                self.target = self.instance_id = None
            self._pause("waiting" if self.target else "off")
            self.last_frame = None
            self.ignore_before = time.monotonic()

    def set_model(self, model: Any) -> None:
        with self.lock:
            previous = self.detection.status()["model"]
            self.detection.set_model(model)
            if previous != model:
                self.target = self.instance_id = None
                self._pause("off")
                self.last_frame = None
                self.ignore_before = time.monotonic()
                self.error = None

    def arm(self) -> None:
        with self.lock:
            self.servo.arm()
            # Never move using a frame captured before the user pressed Start.
            self.ignore_before = time.monotonic()
            self.last_frame = None
            self.moving = False
            self.previous_pose = None
            self.hold_bias = {"x": 0.0, "y": 0.0}
            self.goal_degrees = None

    def disable(self) -> None:
        with self.lock:
            self.moving = False
            self.state = "stopped" if self.target else "off"
            self.box = self.frame = self.error_pixels = None
            self.goal_degrees = None
            self.servo.disable()

    def recalibrate(self) -> None:
        with self.lock:
            self.servo.recalibrate()
            self.target = self.instance_id = None
            self.moving = False
            self._pause("off")
            self.last_frame = None
            self.ignore_before = time.monotonic()
            self.error = None

    def manual_move(self, axis: str, degrees: float) -> None:
        with self.lock:
            self.servo.move(axis, degrees)
            self.target, self.moving = None, False
            self.instance_id = None
            self._pause("off")

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {"target": self.target, "instance_id": self.instance_id,
                    "selection": "retarget" if self.instance_id is not None else "class",
                    "state": self.state, "error": self.error,
                    "box": self.box, "frame": list(self.frame) if self.frame else None,
                    "error_pixels": self.error_pixels, "mode": "absolute-angle",
                    "mapping": "fisheye-kinematics" if self.geometry else "linear",
                    "goal_degrees": self.goal_degrees}

    def _tick(self) -> None:
        with self.lock:
            if not self.target:
                return
            if not self.config["calibrated"]:
                self._pause("uncalibrated")
                return
            servo = self.servo.status()
            if not servo["armed"] or not servo["online"]:
                self.moving = False
                self._pause("stopped")
                return
            camera, detection = self.camera.status(), self.detection.status()
            now = time.monotonic()
            age = detection.get("frame_age_ms")
            if self.target not in detection.get("prompts", []):
                self.target = self.instance_id = None
                self._pause("off")
                return
            if (not camera["online"] or detection.get("state") != "running" or
                type(age) not in (int, float) or not math.isfinite(age) or
                age < 0):
                self._pause("waiting")
                return
            frame = (detection["revision"], detection["frame_sequence"])
            if frame == self.last_frame:
                return
            self.last_frame = frame
            # Only reject images from before Start or a class/prompt change.
            # New results during motion are used immediately, with no settling
            # delay or per-move frame barrier.
            if now - age / 1000 < self.ignore_before:
                return
            boxes = detection.get("boxes", [])
            if self.instance_id is not None:
                selected = [box for box in boxes if box.get("instance_id") == self.instance_id]
                if selected:
                    boxes = selected
                elif detection.get("temporal_tracking") is True:
                    # A SAM identity is a persistent track, not a nearest-box
                    # guess. Hold through occlusion; never jump to its neighbor.
                    self._pause("lost")
                    return
                else:
                    # Clicking is a temporary preference, not a persistent ID
                    # lock. If the association disappears, resume the original
                    # nearest-of-class behavior on this same fresh frame.
                    self.instance_id = None
            box = nearest_box(boxes, self.target, camera["width"], camera["height"])
            if box is None:
                self._pause("lost")
                return
            if detection.get("temporal_tracking") is True and self.instance_id is None:
                # Class selection acquires the nearest instance once. A click
                # can override it; selecting the class again reacquires.
                self.instance_id = box["instance_id"]
            x1, y1, x2, y2 = box["xyxy"]
            errors = {"x": (x1 + x2) / 2 - 0.5, "y": (y1 + y2) / 2 - 0.5}
            pose = detection.get("frame_pose")
            captured_at = detection.get("captured_at")
            if not self._valid_pose(pose, captured_at):
                self._pause("waiting")
                return
            geometric = None
            if self.geometry:
                try:
                    self.geometry.check_binding(camera, servo, self.config)
                    geometric = self.geometry.goals((x1+x2)/2*camera["width"], (y1+y2)/2*camera["height"], pose, servo)
                except ValueError as error:
                    self._pause("uncalibrated")
                    self.error = str(error)
                    return  # Never silently switch a commissioned device to guessed geometry.
            goals = {}
            for axis, error in errors.items():
                sample = pose["axes"][axis]
                # Learn load/stiction bias only from two stationary samples of
                # the SAME goal, never mistake in-flight motion for static error.
                if self.previous_pose:
                    previous = self.previous_pose["axes"][axis]
                    elapsed = pose["sampled_at"] - self.previous_pose["sampled_at"]
                    # This interval only qualifies stationary load-bias samples;
                    # it does not reject a delayed detection or stop tracking.
                    if (0 < elapsed <= 0.75 and
                        abs(sample["goal_degrees"] - previous["goal_degrees"]) < 0.1 and
                        abs(sample["degrees"] - previous["degrees"]) / elapsed < 1.0):
                        self.hold_bias[axis] = sample["goal_degrees"] - sample["degrees"]
                scale = self.config[f"{axis}_degrees_per_frame"]
                correction = error * scale * self.config[f"{axis}_direction"]
                centered = abs(error) <= self.config["deadband"]
                if geometric is not None:
                    correction = geometric[axis] - sample["degrees"]
                    # Coupled geometry can require BOTH motors even if only
                    # one image coordinate is outside the centering deadband.
                    centered = all(abs(e) <= self.config["deadband"] for e in errors.values())
                goals[axis] = sample["degrees"] + (0.0 if centered else correction) + self.hold_bias[axis]
                # Keep a settled holding goal inside the image deadband, but
                # brake at the observed center if an older goal would overshoot.
                current_goal = servo["axes"][axis]["goal_degrees"]
                if centered and abs(current_goal - goals[axis]) <= self.config["deadband"] * scale:
                    goals[axis] = current_goal
            self.previous_pose = pose
            result = self.servo.point(goals)
            self.goal_degrees = result["goal_degrees"]
            self.moving = True
            self.state = ("limited" if result["limited"] else
                          "centered" if all(abs(e) <= self.config["deadband"] for e in errors.values()) else "tracking")
            if self.state == "centered":
                # Once the clicked object is centered, the ordinary tracker
                # takes over; no ID continuity is required to keep following.
                self.instance_id = None
            self.box, self.frame = box, frame
            self.error_pixels = [round(errors["x"] * camera["width"], 1), round(errors["y"] * camera["height"], 1)]
            self.error = None

    def _valid_pose(self, pose: Any, captured_at: Any) -> bool:
        if not isinstance(pose, dict):
            return False
        sampled_at = pose.get("sampled_at")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (sampled_at, captured_at)):
            return False
        if sampled_at < self.ignore_before or not 0 <= captured_at - sampled_at <= 0.1:
            return False
        if not isinstance(pose.get("axes"), dict) or set(pose["axes"]) != {"x", "y"}:
            return False
        return all(isinstance(axis, dict) and all(
            type(axis.get(key)) in (int, float) and math.isfinite(axis[key])
            for key in ("degrees", "goal_degrees")) for axis in pose["axes"].values())

    def _run(self) -> None:
        while not self.stop_event.is_set():
            version = self.detection.frame_version()
            try:
                self._tick()
            except ServoDisarmed:
                with self.lock:
                    self.moving = False
                    self._pause("stopped")
            except Exception as error:
                LOGGER.exception("camera tracking stopped")
                with self.lock:
                    self.target = self.instance_id = None
                    self.moving = False
                    self.state, self.error = "error", str(error)
                    self.box = self.frame = self.error_pixels = None
                    try:
                        self.servo.disable()
                    except DeviceUnavailable:
                        LOGGER.exception("tracking fault torque-off unconfirmed")
            # Wake on a new inference result, not a fixed-rate control timer.
            # The timeout only provides checks for Stop and hardware/model faults
            # when inference has stopped publishing results.
            self.detection.wait_for_update(version, timeout=0.1)
