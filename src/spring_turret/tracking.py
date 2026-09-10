"""Framing from completed detections; Start intent survives transient faults."""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

from spring_turret.servo import DeviceUnavailable, ServoDisarmed
from spring_turret.geometry import GeometryResource
from spring_turret.bearing_filter import BearingFilter
from spring_turret.tracking_masks import valid_mask_centroid
from spring_turret.interfaces import CameraDevice, MotorDevice, DetectionSource
from spring_turret.policy import POLICY_VERSION, DEAD_BAND, continuity, select_candidates

LOGGER = logging.getLogger("spring-turret.tracking")
DEFAULTS = {
    "calibrated": False,  # Legacy mapping metadata, not an operator-approval gate.
    "x_direction": 1, "y_direction": -1,
    "x_degrees_per_frame": 160.0, "y_degrees_per_frame": 90.0,
    "deadband": DEAD_BAND,
    "max_frame_age_seconds": 0.75,  # Legacy config accepted but no longer expires detections.
    "geometry_file": None,
    "bearing_filter": None,  # Optional assembly-qualified target-noise filter.
    "hold_reacquire": False,  # Legacy accepted; shared model policy now owns continuity.
}


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("tracking contains unsupported settings")
    merged = {**DEFAULTS, **config}
    if merged["geometry_file"] is not None and (type(merged["geometry_file"]) is not str or not Path(merged["geometry_file"]).is_absolute()):
        raise ValueError("tracking.geometry_file must be an absolute path or null")
    if type(merged["calibrated"]) is not bool:
        raise ValueError("tracking.calibrated must be a boolean")
    if type(merged["hold_reacquire"]) is not bool:
        raise ValueError("tracking.hold_reacquire must be a boolean")
    if merged["hold_reacquire"] and not merged["geometry_file"]:
        raise ValueError("tracking.hold_reacquire requires calibrated geometry")
    for axis in ("x", "y"):
        value = merged[f"{axis}_direction"]
        if type(value) is not int or value not in (-1, 1):
            raise ValueError(f"tracking.{axis}_direction must be -1 or 1")
    filtering = merged["bearing_filter"]
    if filtering is not None:
        if (not isinstance(filtering, dict) or set(filtering) != {"min_cutoff_hz", "speed_gain"}
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in filtering.values())
                or not .1 <= filtering["min_cutoff_hz"] <= 30
                or not 0 <= filtering["speed_gain"] <= 100):
            raise ValueError("tracking.bearing_filter requires min_cutoff_hz and speed_gain")
        if not merged["geometry_file"]:
            raise ValueError("tracking.bearing_filter requires calibrated geometry")
    for key, (low, high) in {
        "x_degrees_per_frame": (1, 360), "y_degrees_per_frame": (1, 180),
        "deadband": (0.001, 0.1), "max_frame_age_seconds": (0.1, 1),
    }.items():
        value = merged[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"tracking.{key} must be between {low} and {high}")


def tracking_point(box: dict) -> tuple[float, float] | None:
    """Use an original-mask centroid; box-only detectors retain their center.

    An explicitly empty/invalid mask has no point: do not silently aim at its
    box instead. Centroids of concave/disconnected masks need not lie inside it.
    """
    if "mask_centroid" in box:
        point = box["mask_centroid"]
        return tuple(point) if valid_mask_centroid(point) else None
    x1, y1, x2, y2 = box["xyxy"]
    return (x1 + x2) / 2, (y1 + y2) / 2


def nearest_box(boxes: list[dict], prompt: str, width: int, height: int) -> dict | None:
    """Choose aiming-point distance in pixels, not normalized square space."""
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
        point = tracking_point(box)
        if point is None:
            continue
        cx, cy = point
        distance = ((cx - 0.5) * width) ** 2 + ((cy - 0.5) * height) ** 2
        candidates.append((distance, -score, box))
    return min(candidates, key=lambda item: item[:2])[2] if candidates else None


class TrackingController:
    def __init__(self, config: dict, detection: DetectionSource, servo: MotorDevice, camera: CameraDevice):
        validate_config(config)
        self.config = {**DEFAULTS, **config}
        self.geometry_resource = GeometryResource(self.config["geometry_file"])
        self.geometry = self.geometry_resource.value
        filtering = self.config["bearing_filter"]
        self.bearing_filter = BearingFilter(**filtering) if filtering else None
        self.detection, self.servo, self.camera = detection, servo, camera
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera-tracking", daemon=True)
        self.target: str | None = None
        self.instance_id: int | None = None
        self.clicked = False
        self.lock_revision = None
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
        self.last_armed_at = None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3)

    def _pause(self, state: str) -> None:
        if self.bearing_filter:
            self.bearing_filter.reset()
        if self.moving:
            try:
                self.servo.track({"x": 0.0, "y": 0.0})
            except (ServoDisarmed, DeviceUnavailable):
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
            try:
                self.servo.track({"x": 0.0, "y": 0.0})
            except (ServoDisarmed, DeviceUnavailable):
                pass  # Retain the requested class across a disappearing bus.
        self.target, self.instance_id = prompt, instance_id
        self.lock_revision = None
        self.clicked = instance_id is not None
        self.last_frame = None
        self.ignore_before = time.monotonic()
        self.error = None
        self.state = "waiting" if prompt else "off"

    def set_prompts(self, prompts: Any) -> None:
        with self.lock:
            self.detection.set_prompts(prompts)
            if self.target not in self.detection.status()["prompts"]:
                self.target = None
            self.instance_id, self.clicked = None, False
            self._pause("waiting" if self.target else "off")
            self.lock_revision = None
            self.last_frame = None
            self.ignore_before = time.monotonic()

    def set_model(self, model: Any) -> None:
        with self.lock:
            before = self.detection.status()
            previous = before.get("implementation_model", before["model"])
            self.detection.set_model(model)
            current = self.detection.status()
            if previous != current.get("implementation_model", current["model"]):
                self.instance_id, self.clicked = None, False
                self._pause("waiting" if self.target else "off")
                self.lock_revision = None
                self.last_frame = None
                self.ignore_before = time.monotonic()
                self.error = None

    def arm(self) -> None:
        with self.lock:
            self.servo.arm()
            if self.bearing_filter:
                self.bearing_filter.reset()
            # Never move using a frame captured before the user pressed Start.
            self.ignore_before = time.monotonic()
            self.last_frame = None
            self.moving = False
            self.previous_pose = None
            self.hold_bias = {"x": 0.0, "y": 0.0}
            self.goal_degrees = None

    def disable(self) -> None:
        with self.lock:
            if self.bearing_filter:
                self.bearing_filter.reset()
            self.moving = False
            self.state = "stopped" if self.target else "off"
            self.box = self.frame = self.error_pixels = None
            self.goal_degrees = None
            self.servo.disable()

    def recalibrate(self) -> None:
        with self.lock:
            self.servo.recalibrate()
            # Keep the selected class so Set zeros -> Start resumes tracking;
            # discard the instance and all coordinates from the old zero frame.
            self.instance_id = None
            self.clicked = False
            self.lock_revision = None
            self.moving = False
            self._pause("stopped" if self.target else "off")
            self.last_frame = None
            self.ignore_before = time.monotonic()
            self.error = None

    def manual_move(self, axis: str, degrees: float) -> None:
        with self.lock:
            self.servo.move(axis, degrees)
            self.target, self.moving = None, False
            self.instance_id = None
            self._pause("off")

    def _setup_reason(self):
        if self.geometry_resource.error:
            return self.geometry_resource.error
        return None

    def status(self) -> dict[str, Any]:
        with self.lock:
            model = getattr(self.detection, "model", None)
            return {"target": self.target, "instance_id": self.instance_id,
                    "selection": "retarget" if self.clicked and self.instance_id is not None else "class",
                    "state": self.state, "error": self.error or self._setup_reason(),
                    "calibrated": self.config["calibrated"],
                    "configured_directions": {a: self.config[a + "_direction"] for a in ("x", "y")},
                    "box": self.box, "frame": list(self.frame) if self.frame else None,
                    "error_pixels": self.error_pixels, "mode": "absolute-angle",
                    "mapping": "fisheye-kinematics" if self.geometry else "linear",
                    "policy_version": POLICY_VERSION, "continuity": continuity(model),
                    "hold_reason": "temporary-occlusion" if self.state == "lost" and self.instance_id is not None else None,
                    "goal_degrees": self.goal_degrees}

    def _tick(self) -> None:
        with self.lock:
            if self.geometry_resource.error:
                self.geometry = self.geometry_resource.refresh()
            if not self.target:
                return
            # Physical zero readiness is enforced by servo.arm(). Do not add a
            # second approval gate for the configured camera-to-angle mapping.
            if self.geometry_resource.error:
                self._pause("uncalibrated")
                return
            servo = self.servo.status()
            if not servo["armed"] or not servo["online"]:
                self.moving = False
                self._pause("recovering" if servo.get("run_requested", False) else "stopped")
                return
            if servo.get("armed_at") != self.last_armed_at:
                self.last_armed_at = servo.get("armed_at")
                self.ignore_before = max(self.ignore_before, self.last_armed_at or 0)
                self.last_frame = None
                self.moving = False
                self._pause("waiting")
            camera, detection = self.camera.status(), self.detection.status()
            now = time.monotonic()
            age = detection.get("frame_age_ms")
            if self.target not in detection.get("prompts", []):
                self._pause("waiting")
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
            if self.lock_revision is not None and detection["revision"] != self.lock_revision:
                self.instance_id, self.clicked = None, False
            self.lock_revision = detection["revision"]
            boxes, self.instance_id, self.clicked, holding = select_candidates(
                detection, self.target, self.instance_id, self.clicked)
            if holding:
                self._pause("lost")
                return
            box = nearest_box(boxes, self.target, camera["width"], camera["height"])
            if box is None:
                self._pause("lost")
                return
            if detection.get("temporal_tracking") is True and self.instance_id is None:
                # Class selection acquires the nearest instance once. A click
                # can override it; selecting the class again reacquires.
                self.instance_id = box["instance_id"]
            cx, cy = tracking_point(box)
            errors = {"x": cx - 0.5, "y": cy - 0.5}
            control_errors = errors
            pose = detection.get("frame_pose")
            captured_at = detection.get("captured_at")
            if not self._valid_pose(pose, captured_at):
                self._pause("waiting")
                return
            geometric = None
            geometry_limited = False
            if self.geometry:
                try:
                    self.geometry.check_binding(camera, servo, self.config)
                    world = self.geometry.world_direction(cx*camera["width"], cy*camera["height"], pose, servo)
                    if self.bearing_filter:
                        filtered = self.bearing_filter.update(world,captured_at,(self.target,box.get("instance_id")))
                        try:
                            u,v = self.geometry.image_point(filtered,pose,servo)
                        except ValueError:
                            # A lagging estimate outside the calibrated view is
                            # reacquired from this visible measurement, not a
                            # reason to fall back to uncalibrated geometry.
                            self.bearing_filter.reset()
                            filtered = world
                            u,v = cx*camera["width"],cy*camera["height"]
                        world = filtered
                        control_errors = {"x":u/camera["width"]-.5,"y":v/camera["height"]-.5}
                    solution = self.geometry.solve_direction(world,pose,servo)
                    geometric = solution["goals"]
                    geometry_limited = solution["limited"]
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
                    centered = all(abs(e) <= self.config["deadband"] for e in control_errors.values())
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
            self.state = ("limited" if result["limited"] or geometry_limited else
                          "centered" if all(abs(e) <= self.config["deadband"] for e in control_errors.values()) else "tracking")
            if (self.state == "centered"
                    and detection.get("temporal_tracking") is not True):
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
            except (ServoDisarmed, DeviceUnavailable) as error:
                with self.lock:
                    self.moving = False
                    self._pause("recovering" if self.servo.status().get("run_requested", False) else "stopped")
                    self.error = str(error)
            except Exception as error:
                LOGGER.exception("camera tracking frame failed; waiting for next result")
                with self.lock:
                    try:
                        self._pause("waiting")
                    except Exception:
                        self.moving = False
                        LOGGER.exception("tracking hold unavailable")
                    self.error = str(error)
            # Wake on a new inference result, not a fixed-rate control timer.
            # The timeout only provides checks for Stop and hardware/model faults
            # when inference has stopped publishing results.
            self.detection.wait_for_update(version, timeout=0.1)
