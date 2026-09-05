"""Opt-in camera framing from fresh detections; never arms or renews the lease."""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

from spring_turret.servo import DeviceUnavailable, ServoDisarmed

LOGGER = logging.getLogger("spring-turret.tracking")
DEFAULTS = {
    "calibrated": False,
    "x_direction": 1, "y_direction": -1,
    "x_gain": 24.0, "y_gain": 18.0,
    "deadband": 0.012,
    "max_frame_age_seconds": 0.75,
}


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("tracking contains unsupported settings")
    merged = {**DEFAULTS, **config}
    if type(merged["calibrated"]) is not bool:
        raise ValueError("tracking.calibrated must be a boolean")
    for axis in ("x", "y"):
        value = merged[f"{axis}_direction"]
        if type(value) is not int or value not in (-1, 1):
            raise ValueError(f"tracking.{axis}_direction must be -1 or 1")
    for key, (low, high) in {
        "x_gain": (0.1, 90), "y_gain": (0.1, 90),
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
        self.detection, self.servo, self.camera = detection, servo, camera
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera-tracking", daemon=True)
        self.target: str | None = None
        self.state = "off"
        self.error: str | None = None
        self.box: dict | None = None
        self.frame: tuple[int, int] | None = None
        self.last_frame: tuple[int, int] | None = None
        self.ignore_before = 0.0
        self.moving = False
        self.error_pixels: list[float] | None = None

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
        self.state, self.box, self.frame, self.error_pixels = state, None, None, None

    def set_target(self, prompt: Any) -> None:
        with self.lock:
            detection = self.detection.status()
            if prompt is not None and (
                type(prompt) is not str or not prompt or
                prompt not in detection.get("prompts", []) or not detection.get("enabled")
            ):
                raise ValueError("target must be one of the applied object classes, or null")
            if prompt == self.target:
                return
            # Clear first even if holding fails: never leave stale automatic intent.
            self.target = None
            was_moving = self.moving
            self._pause("off")
            if prompt and not was_moving and self.servo.status()["armed"]:
                self.servo.track({"x": 0.0, "y": 0.0})
            self.target = prompt
            self.last_frame = None
            self.ignore_before = time.monotonic()
            self.error = None
            self.state = "waiting" if prompt else "off"

    def set_prompts(self, prompts: Any) -> None:
        with self.lock:
            self.detection.set_prompts(prompts)
            if self.target not in self.detection.status()["prompts"]:
                self.target = None
            self._pause("waiting" if self.target else "off")
            self.last_frame = None
            self.ignore_before = time.monotonic()

    def arm(self) -> None:
        with self.lock:
            self.servo.arm()
            # Never move using a frame captured before the user pressed Start.
            self.ignore_before = time.monotonic()
            self.last_frame = None
            self.moving = False

    def disable(self) -> None:
        with self.lock:
            self.moving = False
            self.state = "stopped" if self.target else "off"
            self.box = self.frame = self.error_pixels = None
            self.servo.disable()

    def manual_move(self, axis: str, degrees: float) -> None:
        with self.lock:
            self.servo.move(axis, degrees)
            self.target, self.moving = None, False
            self._pause("off")

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {"target": self.target, "state": self.state, "error": self.error,
                    "box": self.box, "frame": list(self.frame) if self.frame else None,
                    "error_pixels": self.error_pixels}

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
                self.target = None
                self._pause("off")
                return
            if (not camera["online"] or detection.get("state") != "running" or
                type(age) not in (int, float) or not math.isfinite(age) or
                not 0 <= age <= self.config["max_frame_age_seconds"] * 1000):
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
            box = nearest_box(detection.get("boxes", []), self.target, camera["width"], camera["height"])
            if box is None:
                self._pause("lost")
                return
            x1, y1, x2, y2 = box["xyxy"]
            errors = {"x": (x1 + x2) / 2 - 0.5, "y": (y1 + y2) / 2 - 0.5}
            offsets = {}
            for axis, error in errors.items():
                step = error * self.config[f"{axis}_gain"] * self.config[f"{axis}_direction"]
                offsets[axis] = 0.0 if abs(error) <= self.config["deadband"] else step
            if any(offsets.values()):
                result = self.servo.track(offsets)
                self.moving = True
                self.state = "limited" if result["limited"] else "tracking"
            else:
                # Keep the servo's holding bias once the image is centered;
                # replacing its goal with the measured position can reintroduce
                # gravity/stiction error and cause a centering oscillation.
                self.state = "centered"
            self.box, self.frame = box, frame
            self.error_pixels = [round(errors["x"] * camera["width"], 1), round(errors["y"] * camera["height"], 1)]
            self.error = None

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
                    self.target = None
                    self.moving = False
                    self.state, self.error = "error", str(error)
                    self.box = self.frame = self.error_pixels = None
                    try:
                        self.servo.disable()
                    except DeviceUnavailable:
                        LOGGER.exception("tracking fault torque-off unconfirmed")
            # Wake on a new inference result, not a fixed-rate control timer.
            # The timeout only provides checks for Stop, stale data and faults
            # when inference has stopped publishing results.
            self.detection.wait_for_update(version, timeout=0.1)
