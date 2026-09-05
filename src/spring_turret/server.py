#!/usr/bin/env python3
"""Camera and fail-closed servo service for the turret demo."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from spring_turret.detection import (
    DetectionController,
    validate_config as validate_inference,
)
from spring_turret.servo import (
    DeviceUnavailable, ServoController, ServoDisarmed, TurretError,
    validate_config as validate_servo,
)
from spring_turret.tracking import TrackingController, validate_config as validate_tracking


LOGGER = logging.getLogger("spring-turret")
JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
MAX_JSON_BYTES = 4096


def extract_jpeg_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Extract complete JPEGs from a raw byte stream and retain the tail."""
    frames: list[bytes] = []
    while True:
        start = buffer.find(JPEG_START)
        if start < 0:
            return frames, buffer[-1:]
        end = buffer.find(JPEG_END, start + len(JPEG_START))
        if end < 0:
            return frames, buffer[start:]
        end += len(JPEG_END)
        frames.append(buffer[start:end])
        buffer = buffer[end:]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"listen", "camera", "servo"}
    if not required <= set(config) or set(config) - required - {"inference", "tracking"}:
        raise ValueError(
            f"config must contain {sorted(required)} and optional inference/tracking"
        )
    validate_inference(config.get("inference", {}))
    validate_tracking(config.get("tracking", {}))

    listen = config["listen"]
    camera = config["camera"]
    servo = config["servo"]
    if not isinstance(listen.get("host"), str):
        raise ValueError("listen.host must be a string")
    if not 1 <= int(listen.get("port", 0)) <= 65535:
        raise ValueError("listen.port must be between 1 and 65535")
    if not Path(camera.get("device", "")).is_absolute():
        raise ValueError("camera.device must be an absolute path")
    for key in ("width", "height", "framerate"):
        if int(camera.get(key, 0)) <= 0:
            raise ValueError(f"camera.{key} must be positive")
    validate_servo(servo)
    return config


class CameraStream:
    """Maintain one GStreamer capture process and broadcast its latest JPEG."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.condition = threading.Condition()
        self.latest_frame: bytes | None = None
        self.latest_sequence = 0
        self.latest_monotonic = 0.0
        self.error = "camera has not started"
        self.process: subprocess.Popen[bytes] | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            process = self.process
            self.condition.notify_all()
        if process is not None and process.poll() is None:
            process.terminate()
        self.thread.join(timeout=5)
        if process is not None and process.poll() is None:
            process.kill()

    def _set_error(self, message: str) -> None:
        with self.condition:
            self.error = message
            self.condition.notify_all()

    def _pipeline(self) -> list[str]:
        caps = (
            f"image/jpeg,width={int(self.config['width'])},"
            f"height={int(self.config['height'])},"
            f"framerate={int(self.config['framerate'])}/1"
        )
        return [
            "/usr/bin/gst-launch-1.0",
            "-q",
            "v4l2src",
            f"device={self.config['device']}",
            "!",
            caps,
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        ]

    def _run(self) -> None:
        restart_delay = float(self.config.get("restart_delay_seconds", 2))
        while not self.stop_event.is_set():
            device = Path(self.config["device"])
            if not device.exists():
                self._set_error(f"camera device is missing: {device}")
                self.stop_event.wait(restart_delay)
                continue

            process: subprocess.Popen[bytes] | None = None
            try:
                process = subprocess.Popen(
                    self._pipeline(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                with self.condition:
                    self.process = process
                if process.stdout is None:
                    raise DeviceUnavailable("camera process has no output stream")
                buffer = b""
                while not self.stop_event.is_set():
                    chunk = process.stdout.read1(65536)
                    if not chunk:
                        raise DeviceUnavailable(
                            f"camera pipeline exited with status {process.poll()}"
                        )
                    buffer += chunk
                    frames, buffer = extract_jpeg_frames(buffer)
                    if len(buffer) > 8 * 1024 * 1024:
                        buffer = buffer[-2:]
                    for frame in frames:
                        with self.condition:
                            self.latest_frame = frame
                            self.latest_sequence += 1
                            self.latest_monotonic = time.monotonic()
                            self.error = ""
                            self.condition.notify_all()
            except (OSError, TurretError) as error:
                self._set_error(str(error))
            finally:
                with self.condition:
                    self.process = None
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                self.stop_event.wait(restart_delay)

    def wait_for_frame(
        self, previous_sequence: int, timeout: float
    ) -> tuple[int, bytes | None]:
        sequence, jpeg, _ = self.wait_for_sample(previous_sequence, timeout)
        return sequence, jpeg

    def wait_for_sample(
        self, previous_sequence: int, timeout: float, after: float = 0.0
    ) -> tuple[int, bytes | None, float]:
        """Return a JPEG and its receipt timestamp atomically, optionally after a pose read."""
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    (self.latest_sequence != previous_sequence and self.latest_monotonic >= after)
                    or self.stop_event.is_set()
                ),
                timeout=timeout,
            )
            return self.latest_sequence, self.latest_frame, self.latest_monotonic

    def status(self) -> dict[str, Any]:
        with self.condition:
            age = (
                time.monotonic() - self.latest_monotonic
                if self.latest_monotonic
                else None
            )
            return {
                "online": age is not None and age < 3,
                "device": self.config["device"],
                "width": int(self.config["width"]),
                "height": int(self.config["height"]),
                "framerate": int(self.config["framerate"]),
                "frame_age_ms": round(age * 1000) if age is not None else None,
                "error": self.error or None,
            }


class TurretApplication:
    def __init__(
        self,
        config: dict[str, Any],
        camera: Any | None = None,
        servo: Any | None = None,
        static_dir: Path | None = None,
        detection: Any | None = None,
    ):
        self.config = config
        self.camera = camera or CameraStream(config["camera"])
        self.servo = servo or ServoController(config["servo"])
        self.static_dir = static_dir or Path(__file__).with_name("static")
        self.detection = detection or DetectionController(
            config.get("inference", {}), self.camera, pose_provider=self.servo.sample_pose
        )
        self.tracking = TrackingController(config.get("tracking", {}), self.detection, self.servo, self.camera)

    def start(self) -> None:
        self.camera.start()
        self.servo.start()
        self.detection.start()
        self.tracking.start()

    def stop(self) -> None:
        self.tracking.stop()
        self.servo.stop()
        self.detection.stop()
        self.camera.stop()

    def status(self) -> dict[str, Any]:
        return {
            "profile": "turret-demo",
            "camera": self.camera.status(),
            "servo": self.servo.status(),
            "detection": self.detection.status(),
            "tracking": self.tracking.status(),
        }


def make_handler(application: TurretApplication) -> type[BaseHTTPRequestHandler]:
    class TurretHandler(BaseHTTPRequestHandler):
        server_version = "SpringTurret/1"

        def log_message(self, format_string: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), format_string % args)

        def _security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self'; script-src 'self'; "
                "style-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._security_headers()
            self.end_headers()
            self.wfile.write(body)

        def _static(self, filename: str, content_type: str) -> None:
            path = application.static_dir / filename
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._security_headers()
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._static("index.html", "text/html; charset=utf-8")
            elif path == "/app.css":
                self._static("app.css", "text/css; charset=utf-8")
            elif path == "/app.js":
                self._static("app.js", "application/javascript; charset=utf-8")
            elif path == "/api/status":
                self._json(HTTPStatus.OK, application.status())
            elif path == "/stream.mjpg":
                self._stream()
            elif path.startswith("/api/detection/frame/") and path.endswith(".jpg"):
                key = path.removeprefix("/api/detection/frame/").removesuffix(".jpg")
                frame = application.detection.frame(key)
                if frame is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self._security_headers()
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _stream(self) -> None:
            sequence, frame = application.camera.wait_for_frame(0, 10)
            if frame is None:
                self._json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "camera has not produced a frame"},
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self._security_headers()
            self.end_headers()
            try:
                while frame is not None:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    sequence, frame = application.camera.wait_for_frame(sequence, 15)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _request_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("invalid Content-Length") from error
            if not 1 <= length <= MAX_JSON_BYTES:
                raise ValueError("JSON body must contain between 1 and 4096 bytes")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
            return body

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            try:
                if path == "/api/detection/prompts":
                    body = self._request_json()
                    if set(body) != {"prompts"}:
                        raise ValueError("body must contain only prompts")
                    application.tracking.set_prompts(body["prompts"])
                elif path == "/api/detection/prompt":
                    body = self._request_json()
                    if set(body) != {"prompt"}:
                        raise ValueError("body must contain only prompt")
                    application.tracking.set_prompts([body["prompt"]])
                elif path == "/api/tracking/target":
                    body = self._request_json()
                    if set(body) != {"target"}:
                        raise ValueError("body must contain only target")
                    application.tracking.set_target(body["target"])
                elif path == "/api/servo/arm":
                    application.tracking.arm()
                elif path == "/api/servo/disable":
                    application.tracking.disable()
                elif path == "/api/servo/keepalive":
                    application.servo.keepalive()
                elif path == "/api/servo/position":
                    body = self._request_json()
                    if set(body) != {"axis", "degrees"}:
                        raise ValueError("body must contain only axis and degrees")
                    application.tracking.manual_move(body["axis"], body["degrees"])
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
            except ServoDisarmed as error:
                self._json(HTTPStatus.CONFLICT, {"error": str(error)})
                return
            except DeviceUnavailable as error:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
                return
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._json(HTTPStatus.OK, application.status())

    return TurretHandler


def serve(config_path: Path) -> None:
    config = load_config(config_path)
    application = TurretApplication(config)
    server = ThreadingHTTPServer(
        (config["listen"]["host"], int(config["listen"]["port"])),
        make_handler(application),
    )
    server.daemon_threads = True

    def request_shutdown(_signal: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    application.start()
    try:
        LOGGER.info(
            "turret demo listening on %s:%s",
            config["listen"]["host"],
            config["listen"]["port"],
        )
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        application.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    serve(args.config)


if __name__ == "__main__":
    main()
