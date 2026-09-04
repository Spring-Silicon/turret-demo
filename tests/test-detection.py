#!/usr/bin/env python3
"""No GPU needed: prompt cancellation, latest-frame routing, errors, and API."""

import base64
import json
import sys
import threading
import time
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spring_turret.detection import DetectionController, validate_config
from spring_turret.server import TurretApplication, make_handler


def eventually(check, timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if check():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


class Camera:
    sequence = 0

    def wait_for_frame(self, previous, timeout):
        self.sequence += 1
        return self.sequence, b"camera-jpeg"

    def status(self):
        return {"online": True, "frame_age_ms": 1}


class Worker:
    def __init__(self, config):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.requests = []
        self.stopped = False
        self.fail = False

    def launch(self):
        pass

    def receive(self, timeout):
        return {"type": "ready"}

    def detect(self, request_id, jpeg, prompt, progress):
        self.requests.append((request_id, jpeg, prompt))
        self.entered.set()
        self.release.wait(2)
        if self.fail:
            raise RuntimeError("test inference error")
        return {
            "type": "result",
            "id": request_id,
            "boxes": [],
            "latency_ms": 1,
            "torch_compile": True,
            "sycl_graph": True,
            "jpeg": base64.b64encode(prompt.encode()).decode(),
        }

    def stop(self):
        self.stopped = True
        self.release.set()


class DetectionTests(unittest.TestCase):
    def controller(self):
        worker = Worker({})
        controller = DetectionController(
            {"enabled": True, "max_fps": 30}, Camera(), lambda _: worker
        )
        controller.start()
        self.addCleanup(controller.stop)
        return controller, worker

    def test_prompt_replacement_discards_inflight_result(self):
        c, worker = self.controller()
        c.set_prompt("person")
        self.assertTrue(worker.entered.wait(1))
        c.set_prompt("chair")
        self.assertEqual(c.status()["revision"], 2)
        worker.release.set()
        eventually(lambda: c.status()["state"] == "running")
        status = c.status()
        self.assertEqual(status["prompt"], "chair")
        self.assertEqual(c.frame(status["frame_url"].split("/")[-1][:-4]), b"chair")
        self.assertTrue(all(key.startswith("2-") for key in c.frames))

    def test_clear_discards_result_and_stops_sampling(self):
        c, worker = self.controller()
        c.set_prompt("person")
        self.assertTrue(worker.entered.wait(1))
        c.set_prompt("")
        worker.release.set()
        time.sleep(0.1)
        self.assertEqual(c.status()["state"], "idle")
        self.assertNotIn("frame_url", c.status())
        self.assertEqual(len(worker.requests), 1)
        self.assertFalse(c.frames)

    def test_failure_is_reported_without_retry_loop(self):
        c, worker = self.controller()
        worker.fail = True
        worker.release.set()
        c.set_prompt("person")
        eventually(lambda: c.status()["state"] == "error")
        self.assertEqual(c.status()["error"], "test inference error")
        time.sleep(0.1)
        self.assertEqual(len(worker.requests), 1)
        self.assertTrue(worker.stopped)

    def test_bad_prompts(self):
        c, _ = self.controller()
        for prompt in (True, None, ["person"], "x" * 257, "a\nb"):
            with self.assertRaises(ValueError):
                c.set_prompt(prompt)
        c.set_prompt("  chair  ")
        self.assertEqual(c.status()["prompt"], "chair")

    def test_disabled_is_optional(self):
        c = DetectionController({}, Camera())
        c.start()
        c.stop()
        self.assertEqual(c.status()["state"], "disabled")
        with self.assertRaises(ValueError):
            c.set_prompt("person")

    def test_config(self):
        validate_config({})
        base = {
            "enabled": True,
            "python": "/venv/bin/python",
            "checkpoint": "/models/sam.pt",
            "cache_dir": "/cache",
        }
        validate_config(base)
        for key, value in (
            ("python", "relative"),
            ("confidence", 0),
            ("max_fps", 99),
            ("precision", "int8"),
        ):
            with self.assertRaises(ValueError):
                validate_config({**base, key: value})

    def test_http_prompt_and_exact_frame(self):
        c, worker = self.controller()
        worker.release.set()

        class NoServo:
            def status(self):
                return {"armed": False}

        app = TurretApplication(
            {"camera": {}, "servo": {}}, camera=Camera(), servo=NoServo(), detection=c
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        connection = HTTPConnection(*server.server_address)
        self.addCleanup(connection.close)
        connection.request(
            "POST",
            "/api/detection/prompt",
            json.dumps({"prompt": "chair"}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertFalse(json.loads(response.read())["servo"]["armed"])
        eventually(lambda: c.status()["state"] == "running")
        frame_url = c.status()["frame_url"]
        connection.request("GET", frame_url)
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), b"chair")
        c.set_prompt("")
        connection.request("GET", frame_url)
        response = connection.getresponse()
        self.assertEqual(response.status, 404)
        response.read()
        connection.request("POST", "/api/detection/prompt", '{"prompt":true}')
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        response.read()


if __name__ == "__main__":
    unittest.main()
