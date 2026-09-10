"""Disposable web frontend: no camera, GPU worker, or servo controller instances."""
from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
import copy
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
from http.server import ThreadingHTTPServer

from spring_turret.isolated import command_request, read_snapshot
from spring_turret.api_contract import public_status
from spring_turret.models import MODELS
from spring_turret.server import make_handler

LOGGER = logging.getLogger("spring-turret.viewer")


class SnapshotStore:
    """All locks, JPEG history, copies and viewer backpressure are frontend-local."""
    def __init__(self, directory: Path):
        self.directory = directory
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.frames = OrderedDict()
        self.event_cache = None
        self.metadata, self.camera_jpeg, detection = read_snapshot(directory)
        self._install(self.metadata, self.camera_jpeg, detection)
        self.thread = threading.Thread(target=self._read, name="snapshot-reader", daemon=True)

    def _install(self, metadata, camera_jpeg, detection_jpeg):
        with self.condition:
            incoming = metadata["state"]
            if hasattr(self, "state"):
                if metadata["control_at"] < self.metadata["control_at"]:
                    incoming.update(servo=self.state["servo"], tracking=self.state["tracking"])
                    metadata["control_at"] = self.metadata["control_at"]
                if incoming["detection"]["revision"] < self.state["detection"]["revision"]:
                    incoming["detection"] = self.state["detection"]
                    detection_jpeg = b""  # Never associate an old JPEG with new metadata.
                elif incoming["detection"]["revision"] != self.state["detection"]["revision"]:
                    self.frames.clear()
            self.state, self.metadata, self.camera_jpeg = incoming, metadata, camera_jpeg
            detection = incoming["detection"]
            if detection_jpeg and detection.get("frame_sequence") is not None:
                self.frames[f"{detection['revision']}-{detection['frame_sequence']}"] = detection_jpeg
                while len(self.frames) > 8: self.frames.popitem(last=False)
            self.condition.notify_all()

    def _read(self):
        previous = None
        while not self.stop_event.is_set():
            try:
                stat = (self.directory / "snapshot").stat()
                version = (stat.st_ino, stat.st_mtime_ns)
                if version != previous:
                    self._install(*read_snapshot(self.directory))
                    previous = version
            except (OSError, ValueError, KeyError):
                # A dead backend is marked unavailable below, never portrayed
                # as a successful Stop. Existing unconfirmed motor state remains.
                pass
            self.stop_event.wait(.005)

    def status(self, section=None):
        with self.condition:
            # Video subscribers need detection metadata, not a deep copy of
            # servo configuration, geometry, camera state and model registry.
            state = copy.deepcopy(self.state if section is None else {section:self.state[section]})
            now = time.monotonic()
            elapsed = max(0, now - self.metadata["published_at"])
            for name in ("camera", "detection"):
                if name in state and state[name].get("frame_age_ms") is not None:
                    state[name]["frame_age_ms"] += round(elapsed * 1000)
            if section is None:
                state.setdefault("runtime", {}).update(viewer_pid=os.getpid(), snapshot_age_ms=round(elapsed * 1000))
            if elapsed > 2:
                if "camera" in state: state["camera"].update(online=False, error="Backend snapshot unavailable")
                if "servo" in state: state["servo"].update(online=False, ready=False, can_recalibrate=False,
                                      can_recover_gains=False, recalibrate_reason="Backend unavailable; waiting for reconnection",
                                      error="Backend unavailable; motor state unconfirmed")
                if "detection" in state:
                    state["detection"].update(state="error", error="Backend snapshot unavailable", boxes=[], mask_overlay=None)
                if "tracking" in state: state["tracking"].update(state="error", error="Backend snapshot unavailable")
            state = public_status(state)
            return state if section is None else state[section]

    def apply_command(self, reply):
        with self.condition:
            # Make Start/Stop/prompt replies immediately visible. Do not let an
            # already-published, pre-command control snapshot undo that state.
            self.state.update(reply["state"])
            self.event_cache = None
            self.metadata["state"] = self.state
            self.metadata["control_at"] = reply["control_at"]
            self.condition.notify_all()
        return self.status()

    def close(self):
        self.stop_event.set()
        with self.condition: self.condition.notify_all()
        if self.thread.is_alive(): self.thread.join(timeout=1)


class DetectionView:
    def __init__(self, store):
        self.store = store
        self.condition, self.frames, self.stop_event = store.condition, store.frames, store.stop_event

    def status(self): return self.store.status("detection")

    def event(self, previous):
        """Encode one immutable JPEG+metadata event per frame, not per viewer.

        Only the disposable viewer owns this cache/lock. Slow network writes
        occur after returning; no viewer can backpressure inference or motors.
        Heartbeats deliberately refresh age/errors and never replay old JPEGs.
        """
        with self.condition:
            metadata = self.status()
            current = (metadata["revision"], metadata.get("frame_sequence"))
            key = (*current, metadata.get("state"), metadata.get("error"))
            frame = self.frames.get(f"{current[0]}-{current[1]}") if current != previous and metadata.get("state") == "running" else None
            if frame is not None:
                if self.store.event_cache is None or self.store.event_cache[0] != key:
                    metadata["jpeg"] = base64.b64encode(frame).decode()
                    payload = b"data: " + json.dumps(metadata, separators=(",", ":")).encode() + b"\n\n"
                    self.store.event_cache = (key, payload)
                return current, self.store.event_cache[1]
            return current, b"data: " + json.dumps(metadata, separators=(",", ":")).encode() + b"\n\n"

    def frame(self, key):
        with self.condition: return self.frames.get(key)

    def frame_version(self):
        with self.condition:
            state = self.store.state["detection"]
            return state["revision"], state.get("frame_sequence")

    def wait_for_update(self, previous, timeout):
        with self.condition:
            self.condition.wait_for(lambda: self.stop_event.is_set() or self.frame_version() != previous, timeout)


class CameraView:
    def __init__(self, store): self.store = store

    def status(self): return self.store.status("camera")

    def wait_for_frame(self, previous_sequence, timeout):
        with self.store.condition:
            self.store.condition.wait_for(lambda: self.store.stop_event.is_set() or
                self.store.metadata["camera_sequence"] != previous_sequence, timeout)
            return self.store.metadata["camera_sequence"], self.store.camera_jpeg or None


class ViewerApplication:
    def __init__(self, directory):
        self.store = SnapshotStore(directory)
        self.static_dir = Path(__file__).with_name("static")
        self.detection, self.camera = DetectionView(self.store), CameraView(self.store)

    def status(self): return self.store.status()

    def command(self, path, body):
        if path == "/api/detection/model" and (not isinstance(body, dict) or
                not isinstance(body.get("model"), str) or body["model"] not in MODELS):
            raise ValueError("model must be one of " + ", ".join(MODELS))
        return self.store.apply_command(command_request(self.store.directory, path, body))


class ViewerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        self.capacity = threading.BoundedSemaphore(32)
        super().__init__(*args, **kwargs)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5)
        return connection, address

    def process_request(self, request, address):
        if not self.capacity.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try: super().process_request(request, address)
        except BaseException:
            self.capacity.release()
            raise

    def process_request_thread(self, request, address):
        try: super().process_request_thread(request, address)
        finally: self.capacity.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    application = ViewerApplication(args.ipc_dir)
    server = ViewerHTTPServer((args.host, args.port), make_handler(application))
    server.timeout = .2
    def shutdown(_signal, _frame): application.store.stop_event.set()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    application.store.thread.start()
    try:
        LOGGER.info("viewer pid=%s backend pid=%s listening on %s:%s", os.getpid(), args.parent_pid, args.host, args.port)
        while not application.store.stop_event.is_set() and os.getppid() == args.parent_pid:
            server.handle_request()
    finally:
        server.server_close()
        application.store.close()


if __name__ == "__main__": main()
