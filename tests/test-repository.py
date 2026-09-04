#!/usr/bin/env python3
"""Validate the pinned hardware, deployment, and minimal UI contracts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    hardware = json.loads((ROOT / "hardware.json").read_text())
    assert hardware["qualification_status"] == "camera_verified_servo_not_responding"
    assert hardware["camera"]["usb_id"] == "0c45:0261"
    assert hardware["camera"]["serial"] == "UC684"
    assert hardware["camera"]["capture_verified"] is True
    assert hardware["servo_adapter"]["usb_id"] == "1a86:55d3"
    assert hardware["servo_adapter"]["serial"] == "5B61036033"
    assert hardware["servo"]["response_verified"] is False
    assert hardware["servo"]["motion_verified"] is False

    config = json.loads((ROOT / "config/spring-turret-demo.json").read_text())
    servo = config["servo"]
    assert servo["min_position"] < servo["center_position"] < servo["max_position"]
    assert config["camera"]["device"] == "/dev/spring-turret-camera"
    assert servo["device"] == "/dev/spring-turret-servo"

    rules = (ROOT / "deploy/99-spring-turret.rules").read_text()
    for value in ("0c45", "0261", "UC684", "1a86", "55d3", "5B61036033"):
        assert value in rules
    assert 'SYMLINK+="spring-turret-camera"' in rules
    assert 'SYMLINK+="spring-turret-servo"' in rules

    service = (ROOT / "deploy/spring-turret-demo.service").read_text()
    assert "User=spring-turret" in service
    assert "SupplementaryGroups=video dialout" in service
    assert "NoNewPrivileges=true" in service

    html = (ROOT / "src/spring_turret/static/index.html").read_text()
    server = (ROOT / "src/spring_turret/server.py").read_text()
    assert 'id="arm-button" type="button" disabled' in html
    assert 'id="stop-button" type="button" disabled' in html
    assert "password" not in html.lower()
    assert "authentication" not in server.lower()
    assert "self.armed = False" in server
    assert 'write1ByteTxRx(int(self.config["id"]), 40, 0)' in server
    assert "raise ServoDisarmed" in server

    print("validated repository and hardware contracts")


if __name__ == "__main__":
    main()
