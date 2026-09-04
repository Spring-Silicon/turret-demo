#!/usr/bin/env python3
"""Exercise turret API safety gating and bounded motion."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import tempfile
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO_ROOT / "src/spring_turret/server.py"


def load_server():
    loader = importlib.machinery.SourceFileLoader("spring_turret_server", str(SERVER_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class FakeCamera:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def status(self) -> dict:
        return {
            "online": True,
            "device": "/dev/spring-turret-camera",
            "width": 1280,
            "height": 720,
            "framerate": 30,
            "frame_age_ms": 4,
            "error": None,
        }


class FakeServo:
    def __init__(self, module):
        self.module = module
        self.armed = False
        self.position = 2048

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.armed = False

    def arm(self) -> None:
        self.armed = True

    def disable(self) -> None:
        self.armed = False

    def center(self) -> None:
        self.move(2048)

    def move(self, position: int) -> None:
        if not self.armed:
            raise self.module.ServoDisarmed("servo is disarmed; use Arm before moving")
        if not 1536 <= position <= 2560:
            raise ValueError("position must be between 1536 and 2560")
        self.position = position

    def status(self) -> dict:
        return {
            "online": True,
            "device": "/dev/spring-turret-servo",
            "protocol": "dynamixel-2.0",
            "baudrate": 57_600,
            "id": 1,
            "model": 1200,
            "model_name": "XL330-M288-T",
            "position": self.position,
            "min_position": 1536,
            "center_position": 2048,
            "max_position": 2560,
            "armed": self.armed,
            "error": None,
        }


class FakePacket:
    def __init__(self):
        self.torque_writes = []
        self.four_byte_writes = []
        self.position = 2048

    def write1ByteTxRx(self, port, servo_id, address, value):
        self.torque_writes.append((servo_id, address, value))
        return 0, 0

    def write4ByteTxRx(self, port, servo_id, address, value):
        self.four_byte_writes.append((servo_id, address, value))
        return 0, 0

    def read4ByteTxRx(self, port, servo_id, address):
        return self.position, 0, 0


class FakePort:
    def closePort(self):
        pass


def request(port: int, method: str, path: str, body=None):
    headers = {}
    payload = None
    if body is not None:
        payload = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, dict(response.headers), data


def main() -> None:
    module = load_server()
    assert module.extract_jpeg_frames(b"noise\xff\xd8one\xff\xd9tail") == (
        [b"\xff\xd8one\xff\xd9"],
        b"l",
    )

    controller = module.ServoController(
        {
            "device": "/dev/spring-turret-servo",
            "protocol": "dynamixel-2.0",
            "model_number": 1200,
            "baudrate": 57_600,
            "id": 1,
            "min_position": 1536,
            "center_position": 2048,
            "max_position": 2560,
            "profile_velocity": 40,
            "profile_acceleration": 5,
        }
    )
    packet = FakePacket()
    controller.packet = packet
    controller.port = FakePort()
    controller.model = 1200
    controller.comm_success = 0
    try:
        controller.move(2200)
        raise AssertionError("disarmed position command was accepted")
    except module.ServoDisarmed:
        pass
    assert packet.four_byte_writes == []
    controller.arm()
    controller.move(2200)
    controller.disable()
    assert packet.torque_writes == [(1, 64, 1), (1, 64, 0)]
    assert packet.four_byte_writes == [
        (1, 108, 5),
        (1, 112, 40),
        (1, 116, 2048),
        (1, 116, 2200),
    ]

    packet.position = 100
    try:
        controller.arm()
        raise AssertionError("out-of-range servo position was armed")
    except module.DeviceUnavailable:
        pass
    assert packet.torque_writes == [(1, 64, 1), (1, 64, 0)]

    with tempfile.TemporaryDirectory() as directory:
        static = Path(directory)
        (static / "index.html").write_text("turret demo", encoding="utf-8")
        config = {
            "listen": {"host": "127.0.0.1", "port": 8080},
            "camera": {},
            "servo": {},
        }
        servo = FakeServo(module)
        application = module.TurretApplication(
            config, camera=FakeCamera(), servo=servo, static_dir=static
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), module.make_handler(application))
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            status, _, body = request(port, "GET", "/api/status")
            assert status == 200
            assert json.loads(body)["servo"]["armed"] is False

            status, _, _ = request(port, "POST", "/api/servo/position", {"position": 2200})
            assert status == 409
            status, _, body = request(port, "POST", "/api/servo/arm")
            assert status == 200 and json.loads(body)["servo"]["armed"] is True
            status, _, body = request(port, "POST", "/api/servo/position", {"position": 2200})
            assert status == 200 and json.loads(body)["servo"]["position"] == 2200
            status, _, _ = request(port, "POST", "/api/servo/position", {"position": 3000})
            assert status == 400
            status, _, body = request(port, "POST", "/api/servo/disable")
            assert status == 200 and json.loads(body)["servo"]["armed"] is False
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    print("validated turret demo HTTP safety controls")


if __name__ == "__main__":
    main()
