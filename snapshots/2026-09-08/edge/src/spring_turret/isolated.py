"""Backend-owned hardware with a disposable, process-isolated HTTP viewer.

The data channel is one atomically replaced snapshot in private tmpfs. It has no
reader lock, queue, acknowledgments or dependency on the number/speed of viewers.
Only explicit commands cross the private Unix socket in the reverse direction.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

from spring_turret.servo import DeviceUnavailable, ServoDisarmed

LOGGER = logging.getLogger("spring-turret.backend")
HEADER = struct.Struct("!III")
MAX_METADATA = 1024 * 1024
MAX_JPEG = 8 * 1024 * 1024
MAX_COMMAND = 8192


def write_snapshot(directory: Path, metadata: dict, camera: bytes, detection: bytes) -> None:
    body = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
    if len(body) > MAX_METADATA or max(len(camera), len(detection)) > MAX_JPEG:
        raise ValueError("Viewer snapshot exceeds bounded IPC size")
    # Only this publisher writes. Replacement, not an advisory lock, gives each
    # reader a complete immutable generation, including the exact-frame JPEGs.
    with (directory / "snapshot.next").open("wb") as out:
        out.writelines((HEADER.pack(len(body), len(camera), len(detection)), body, camera, detection))
    os.replace(directory / "snapshot.next", directory / "snapshot")


def read_snapshot(directory: Path) -> tuple[dict, bytes, bytes]:
    with (directory / "snapshot").open("rb") as source:
        header = source.read(HEADER.size)
        if len(header) != HEADER.size: raise ValueError("Incomplete snapshot header")
        sizes = HEADER.unpack(header)
        if sizes[0] > MAX_METADATA or max(sizes[1:]) > MAX_JPEG:
            raise ValueError("Invalid snapshot size")
        data = source.read(sum(sizes) + 1)
    if len(data) != sum(sizes): raise ValueError("Incomplete snapshot generation")
    metadata = json.loads(data[:sizes[0]])
    if not isinstance(metadata, dict): raise ValueError("Invalid snapshot metadata")
    return metadata, data[sizes[0]:sizes[0]+sizes[1]], data[sizes[0]+sizes[1]:]


def receive_json(connection: socket.socket, limit: int) -> dict:
    def exact(size):
        parts = bytearray()
        while len(parts) < size:
            part = connection.recv(size - len(parts))
            if not part: raise ConnectionError("Incomplete IPC message")
            parts.extend(part)
        return bytes(parts)
    size = struct.unpack("!I", exact(4))[0]
    if size > limit: raise ValueError("IPC message too large")
    value = json.loads(exact(size))
    if not isinstance(value, dict): raise ValueError("IPC message must be an object")
    return value


def send_json(connection: socket.socket, value: dict) -> None:
    body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(body) > MAX_METADATA: raise ValueError("IPC reply too large")
    connection.sendall(struct.pack("!I", len(body)) + body)


def command_request(directory: Path, path: str, body: dict) -> dict:
    request = {"path": path, "body": body, "issued_at": time.monotonic()}
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(3)
            connection.connect(str(directory / "commands.sock"))
            send_json(connection, request)
            reply = receive_json(connection, MAX_METADATA)
    except (OSError, ValueError) as error:
        # Never retry a potentially executed Start/move after a lost reply.
        raise DeviceUnavailable("Backend command reply unavailable; check motor state") from error
    error_type = {"conflict": ServoDisarmed, "unavailable": DeviceUnavailable, "invalid": ValueError}
    if reply.get("error_type"):
        raise error_type.get(reply["error_type"], DeviceUnavailable)(reply.get("error", "Backend command failed"))
    return reply


class CommandServer:
    """Bounded private command ingress, never used by status/video subscribers."""
    def __init__(self, directory, application, stop_event):
        self.application, self.stop_event = application, stop_event
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(directory / "commands.sock"))
        self.socket.listen(8)
        self.socket.settimeout(.2)
        self.thread = threading.Thread(target=self._run, name="commands", daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try: connection, _ = self.socket.accept()
            except TimeoutError: continue
            except OSError: break
            with connection:
                try:
                    connection.settimeout(.25)
                    request = receive_json(connection, MAX_COMMAND)
                    issued = request.get("issued_at")
                    if (type(issued) not in (int, float) or not math.isfinite(issued)
                            or not 0 <= time.monotonic() - issued <= 2):
                        raise ValueError("Expired command; submit again")
                    if self.stop_event.is_set(): break
                    state = self.application.command(request.get("path"), request.get("body"))
                    reply = {"state": state, "control_at": time.monotonic()}
                except ServoDisarmed as error: reply = {"error_type": "conflict", "error": str(error)}
                except DeviceUnavailable as error: reply = {"error_type": "unavailable", "error": str(error)}
                except (TypeError, ValueError) as error: reply = {"error_type": "invalid", "error": str(error)}
                except (OSError, ConnectionError): continue
                except Exception:
                    LOGGER.exception("private command failed")
                    reply = {"error_type": "unavailable", "error": "Backend command failed; check motor state"}
                try:
                    connection.settimeout(.25)
                    send_json(connection, reply)
                except (OSError, ValueError): pass  # A lost viewer cannot hold this ingress open.

    def close(self):
        self.socket.close()
        if self.thread.is_alive(): self.thread.join(timeout=1)


class BackendRuntime:
    def __init__(self, application, *, directory: Path | None = None, viewer_command=None):
        self.application = application
        root = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
        self.temporary = tempfile.TemporaryDirectory(prefix="spring-turret-", dir=root) if directory is None else None
        self.directory = Path(self.temporary.name) if self.temporary else directory
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.viewer = None
        self.viewer_command = viewer_command
        self.commands = CommandServer(self.directory, application, self.stop_event)
        self.publisher = threading.Thread(target=self._publish, name="viewer-snapshot", daemon=True)

    def _publish(self):
        controls, control_at, previous = {}, 0.0, None
        detector = self.application.detection
        while not self.stop_event.is_set():
            try:
                now = time.monotonic()
                refreshed = now - control_at >= .2
                if refreshed:
                    control_at = now
                    # Fixed publication cost, independent of connected viewers.
                    controls = {"servo": self.application.servo.status(),
                                "tracking": self.application.tracking.status()}
                camera_sequence, camera_jpeg, camera_at = self.application.camera.wait_for_sample(-1, 0)
                camera = self.application.camera.status()
                with detector.condition:
                    detection = detector.status()
                    key = f"{detection['revision']}-{detection.get('frame_sequence')}"
                    jpeg = detector.frames.get(key, b"")
                version = (camera_sequence, detection["revision"], detection.get("frame_sequence"),
                           detection.get("state"))
                if version != previous or refreshed:
                    worker = getattr(detector, "worker", None)
                    model_process = getattr(worker, "process", None)
                    state = {"profile": "turret-demo", "camera": camera, "detection": detection, **controls,
                             "runtime": {"architecture": "process-isolated", "backend_pid": os.getpid(),
                                         "viewer_pid": self.viewer.pid if self.viewer else None,
                                         "model_pid": getattr(model_process, "pid", None)}}
                    metadata = {"state": state, "published_at": time.monotonic(), "control_at": control_at,
                                "camera_sequence": camera_sequence, "camera_at": camera_at}
                    # Serialization and file I/O are outside EVERY device/model lock.
                    write_snapshot(self.directory, metadata, camera_jpeg or b"", jpeg)
                    previous = version
                    self.ready.set()
            except Exception:
                LOGGER.exception("viewer snapshot publication failed; control continues")
                self.stop_event.wait(.2)
            # This publisher never drives, acknowledges or paces inference/control.
            self.stop_event.wait(.005)

    def run(self):
        self.commands.thread.start()
        self.publisher.start()
        while not self.stop_event.is_set() and not self.ready.wait(.1): pass
        while not self.stop_event.is_set():
            if self.viewer is None or self.viewer.poll() is not None:
                if self.viewer is not None:
                    LOGGER.warning("viewer exited (%s); restarting without resetting control", self.viewer.returncode)
                    if self.stop_event.wait(1): break
                listen = self.application.config["listen"]
                command = self.viewer_command or [sys.executable, "-m", "spring_turret.viewer",
                    "--ipc-dir", str(self.directory), "--parent-pid", str(os.getpid()),
                    "--host", listen["host"], "--port", str(listen["port"])]
                self.viewer = subprocess.Popen(command, close_fds=True)
                LOGGER.info("backend pid=%s viewer pid=%s IPC=%s", os.getpid(), self.viewer.pid, self.directory)
            self.stop_event.wait(.2)

    def close(self):
        self.stop_event.set()
        if self.viewer is not None and self.viewer.poll() is None:
            self.viewer.terminate()
            try: self.viewer.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.viewer.kill()
                self.viewer.wait(timeout=3)
        self.commands.close()
        if self.publisher.is_alive(): self.publisher.join(timeout=2)
        if self.temporary: self.temporary.cleanup()
