#!/usr/bin/env python3
"""Real process/IPC tests using simulated frames and motors only."""
import copy
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from collections import OrderedDict
from http.client import HTTPConnection, HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spring_turret.isolated import (BackendRuntime, CommandServer, command_request,
    read_snapshot, write_snapshot, send_json, receive_json, MAX_METADATA)
from spring_turret.viewer import SnapshotStore, DetectionView


def metadata(sequence=1):
    now = time.monotonic()
    return {"published_at": now, "control_at": now, "camera_at": now, "camera_sequence": sequence,
        "state": {"profile": "turret-demo", "camera": {"online": True, "frame_age_ms": 0},
            "servo": {"online": True, "ready": True, "armed": False, "axes": {}},
            "tracking": {"state": "off", "target": None},
            "detection": {"enabled": True, "state": "running", "model": "sam3.1", "revision": 1,
                "frame_sequence": sequence, "frame_age_ms": 0, "boxes": [], "prompts": ["dog"]}}}


class FakeFrames:
    def __init__(self):
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.sequence = 0
        self.state = "running"
        self.frames = OrderedDict()
        self.thread = threading.Thread(target=self.run, daemon=True)
    def run(self):
        while not self.stop_event.wait(.02):
            with self.condition:
                self.sequence += 1
                self.frames[f"1-{self.sequence}"] = str(self.sequence).encode()
                while len(self.frames) > 8: self.frames.popitem(last=False)
                self.condition.notify_all()
    def status(self):
        with self.condition: return metadata(self.sequence)["state"]["detection"]
    def frame_version(self): return 1, self.sequence
    def wait_for_sample(self, *_): return self.sequence, b"raw-jpeg", time.monotonic()


class FakeApplication:
    def __init__(self, port):
        self.config = {"listen": {"host": "127.0.0.1", "port": port}}
        self.detection = FakeFrames()
        from types import SimpleNamespace
        self.camera = SimpleNamespace(wait_for_sample=self.detection.wait_for_sample,
            status=lambda: {"online": True, "width": 1280, "height": 720, "frame_age_ms": 0})
        self.armed = False
        self.commands = 0
        self.servo = SimpleNamespace(status=lambda: {"online": True, "ready": True,
            "armed": self.armed, "axes": {}, "commands": self.commands})
        self.tracking = SimpleNamespace(status=lambda: {"target": "dog", "state": "tracking"})
    def command(self, path, body):
        if path == "/api/servo/arm": self.armed = True
        elif path == "/api/servo/disable": self.armed = False
        else: raise ValueError("unsupported fake command")
        self.commands += 1
        return {"profile": "turret-demo", "camera": self.camera.status(), "servo": self.servo.status(),
                "tracking": self.tracking.status(), "detection": self.detection.status()}


def fake_backend(directory, port):
    app = FakeApplication(port)
    runtime = BackendRuntime(app, directory=Path(directory))
    signal.signal(signal.SIGTERM, lambda *_: runtime.stop_event.set())
    app.detection.thread.start()
    try: runtime.run()
    finally:
        runtime.close()
        app.detection.stop_event.set()
        app.detection.thread.join(timeout=1)


class IsolationTests(unittest.TestCase):
    def test_video_serialization_is_shared_and_does_not_replay_on_stale_backend(self):
        write_snapshot(self.directory, metadata(), b"camera", b"1")
        store = SnapshotStore(self.directory)
        self.addCleanup(store.close)
        view = DetectionView(store)
        with patch("spring_turret.viewer.base64.b64encode", wraps=__import__("base64").b64encode) as encode:
            first = view.event((-1,None))
            for _ in range(30):
                current, payload = view.event((-1,None))
                self.assertIs(payload, first[1])
            self.assertEqual(encode.call_count, 1)
            heartbeat = json.loads(view.event(current)[1][6:])
            self.assertNotIn("jpeg", heartbeat)
            new = metadata(2)
            store._install(new, b"camera", b"2")
            frame = json.loads(view.event(current)[1][6:])
            self.assertEqual(frame["frame_sequence"], 2)
            self.assertEqual(frame["jpeg"], "Mg==")
            self.assertEqual(encode.call_count, 2)
            store.metadata["published_at"] -= 3
            dead = json.loads(view.event((-1,None))[1][6:])
            self.assertEqual(dead["state"], "error")
            self.assertNotIn("jpeg", dead)
            self.assertIsNone(dead["mask_overlay"])

    def test_section_status_is_owned_without_copying_unrelated_sections(self):
        write_snapshot(self.directory, metadata(), b"camera", b"1")
        store = SnapshotStore(self.directory)
        self.addCleanup(store.close)
        class DoNotCopy:
            def __deepcopy__(self, memo): raise AssertionError("copied servo for video")
        store.state["servo"]["sentinel"] = DoNotCopy()
        detection = store.status("detection")
        detection["prompts"].append("unwanted")
        self.assertEqual(store.status("detection")["prompts"], ["dog"])

    def test_status_event_and_command_reply_share_progress_contract(self):
        initial = metadata()
        initial["state"]["detection"].update(progress_stage="compiling_tracker_memory_update")
        write_snapshot(self.directory, initial, b"camera", b"1")
        store = SnapshotStore(self.directory)
        self.addCleanup(store.close)
        full = store.status()["detection"]
        section = DetectionView(store).status()
        _, payload = DetectionView(store).event(None)
        event = json.loads(payload.removeprefix(b"data: "))
        reply = store.apply_command({"state":initial["state"], "control_at":initial["control_at"]})["detection"]
        for result in (full, section, event, reply):
            self.assertEqual(result["progress"]["phase"], "preparing")
            self.assertEqual(result["api_version"], 1)
            self.assertEqual(result["state"], "running")
            self.assertEqual(result["frame_sequence"], 1)
            self.assertEqual(result["timing"], full["timing"])
        self.assertEqual(event["jpeg"], "MQ==")
        self.assertNotIn("progress", store.state["detection"])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_atomic_snapshot_keeps_metadata_and_jpeg_paired(self):
        write_snapshot(self.directory, metadata(), b"camera", b"1")
        failures = []
        def writer():
            for sequence in range(2, 150):
                write_snapshot(self.directory, metadata(sequence), b"camera", str(sequence).encode())
        thread = threading.Thread(target=writer)
        thread.start()
        for _ in range(200):
            state, raw, detected = read_snapshot(self.directory)
            if int(detected) != state["state"]["detection"]["frame_sequence"]: failures.append(state)
            self.assertEqual(raw, b"camera")
        thread.join()
        self.assertEqual(failures, [])
        self.assertEqual([p.name for p in self.directory.iterdir()], ["snapshot"])

    def test_reader_preserves_stop_reply_over_older_control_snapshot(self):
        before = metadata()
        before["state"]["servo"]["armed"] = True
        write_snapshot(self.directory, before, b"camera", b"1")
        store = SnapshotStore(self.directory)
        self.addCleanup(store.close)
        stopped = copy.deepcopy(before["state"])
        stopped["servo"]["armed"] = False
        store.apply_command({"state": stopped, "control_at": before["control_at"] + .1})
        older = copy.deepcopy(before)
        older["published_at"] += .2
        store._install(older, b"camera", b"1")
        self.assertFalse(store.status()["servo"]["armed"])

    def test_stale_backend_does_not_claim_confirmed_stop(self):
        before = metadata()
        before["state"]["servo"]["armed"] = True
        before["published_at"] -= 3
        write_snapshot(self.directory, before, b"camera", b"1")
        store = SnapshotStore(self.directory)
        self.addCleanup(store.close)
        state = store.status()
        self.assertTrue(state["servo"]["armed"])
        self.assertFalse(state["servo"]["online"])
        self.assertIn("unconfirmed", state["servo"]["error"])
        self.assertEqual(state["detection"]["state"], "error")
        self.assertEqual(state["detection"]["progress"]["phase"], "error")
        self.assertIsNone(state["detection"]["mask_overlay"])

    def test_expired_command_is_not_executed_and_stop_is_forwarded(self):
        application = FakeApplication(0)
        stop = threading.Event()
        server = CommandServer(self.directory, application, stop)
        server.thread.start()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.connect(str(self.directory / "commands.sock"))
                send_json(conn, {"path": "/api/servo/arm", "body": {}, "issued_at": time.monotonic() - 3})
                self.assertEqual(receive_json(conn, MAX_METADATA)["error_type"], "invalid")
            self.assertEqual(application.commands, 0)
            self.assertTrue(command_request(self.directory, "/api/servo/arm", {})["state"]["servo"]["armed"])
            self.assertFalse(command_request(self.directory, "/api/servo/disable", {})["state"]["servo"]["armed"])
        finally:
            stop.set()
            server.close()

    def test_viewer_freeze_and_restart_do_not_pause_backend(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        old_path = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + old_path if old_path else "")
        process = multiprocessing.get_context("spawn").Process(target=fake_backend, args=(str(self.directory), port))
        process.start()
        if old_path is None: os.environ.pop("PYTHONPATH", None)
        else: os.environ["PYTHONPATH"] = old_path
        frozen = None
        def http(path="/api/status", body=None):
            conn = HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET" if body is None else "POST", path, None if body is None else json.dumps(body),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            data = json.loads(response.read())
            conn.close()
            self.assertEqual(response.status, 200)
            return data
        def wait(predicate, timeout=8):
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                try:
                    result = predicate()
                    if result: return result
                except (OSError, KeyError, HTTPException): pass
                time.sleep(.03)
            self.fail("timed out waiting for isolated processes")
        try:
            state = wait(http)
            self.assertEqual(state["runtime"]["backend_pid"], process.pid)
            viewer = state["runtime"]["viewer_pid"]
            self.assertNotEqual(viewer, process.pid)
            http("/api/servo/arm", {})  # Simulated motors only.
            for _ in range(20): http()
            self.assertEqual(http()["servo"]["commands"], 1)  # GET never invokes command IPC.
            os.kill(viewer, signal.SIGSTOP)
            frozen = viewer
            first = read_snapshot(self.directory)[0]["state"]["detection"]["frame_sequence"]
            time.sleep(.3)
            later = read_snapshot(self.directory)[0]["state"]
            self.assertGreater(later["detection"]["frame_sequence"], first + 5)
            self.assertTrue(later["servo"]["armed"])
            os.kill(viewer, signal.SIGCONT)
            frozen = None
            os.kill(viewer, signal.SIGTERM)
            state = wait(lambda: (s if (s := http())["runtime"]["viewer_pid"] != viewer else None))
            self.assertEqual(state["runtime"]["backend_pid"], process.pid)
            self.assertTrue(state["servo"]["armed"])
            self.assertGreater(state["detection"]["frame_sequence"], first)
            self.assertFalse(http("/api/servo/disable", {})["servo"]["armed"])
        finally:
            if frozen:
                try: os.kill(frozen, signal.SIGCONT)
                except ProcessLookupError: pass
            process.terminate()
            process.join(timeout=8)
            if process.is_alive(): process.kill(); process.join(timeout=2)


if __name__ == "__main__": unittest.main()
