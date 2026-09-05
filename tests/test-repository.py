#!/usr/bin/env python3
"""Validate the pinned hardware, deployment, and minimal UI contracts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    hardware = json.loads((ROOT / "hardware.json").read_text())
    assert hardware["qualification_status"] == (
        "camera_and_servo_communication_verified_motion_not_verified"
    )
    assert hardware["camera"]["usb_id"] == "0c45:0261"
    assert hardware["camera"]["serial"] == "UC684"
    assert hardware["camera"]["capture_verified"] is True
    assert hardware["servo_adapter"]["usb_id"] == "1a86:55d3"
    assert hardware["servo_adapter"]["serial"] == "5B61036033"
    assert hardware["servo"]["name"] == "ROBOTIS DYNAMIXEL XL330-M288-T"
    assert hardware["servo"]["model_number"] == 1200
    assert hardware["servo"]["configured_baudrate"] == 57_600
    assert hardware["servo"]["response_verified"] is True
    assert hardware["servo"]["position_read_verified"] is True
    assert hardware["servo"]["motion_verified"] is False

    config = json.loads((ROOT / "config/spring-turret-demo.json").read_text())
    assert set(config) == {"listen", "camera", "servo"}
    servo = config["servo"]
    assert servo["calibrated"] is False
    assert servo["axes"]["x"]["id"] == 2
    assert servo["axes"]["y"]["id"] == 1
    for axis in servo["axes"].values():
        assert axis["center_position"] == 2048
        assert axis["profile_velocity"] == 0
        assert axis["profile_acceleration"] == 0
    assert [servo["axes"]["x"][key] for key in ("min_degrees", "max_degrees")] == [-90, 90]
    assert [servo["axes"]["y"][key] for key in ("min_degrees", "max_degrees")] == [-90, 90]
    assert config["camera"]["device"] == "/dev/spring-turret-camera"
    assert servo["device"] == "/dev/spring-turret-servo"
    assert servo["protocol"] == "dynamixel-2.0"
    assert servo["model_number"] == 1200
    assert servo["baudrate"] == 57_600

    assert not (ROOT / "scripts/install.sh").exists()

    html = (ROOT / "src/spring_turret/static/index.html").read_text()
    server = (ROOT / "src/spring_turret/server.py").read_text()
    controller = (ROOT / "src/spring_turret/servo.py").read_text()
    assert html.count('type="range"') == 2
    assert html.index('id="x-slider"') < html.index('id="motor-toggle"') < html.index('id="y-slider"')
    assert 'id="start-icon"' in html and 'id="stop-icon"' in html
    assert 'id="position-slider"' not in html
    assert 'class="jog"' not in html
    assert html.count('id="add-prompt"') == 1
    assert html.index('id="prompt-rows"') < html.index('id="add-prompt"')
    template = html.split('<template id="prompt-row-template">')[1].split(
        "</template>"
    )[0]
    assert 'class="prompt-count"' in template
    assert 'class="remove-prompt"' in template
    assert 'class="target-prompt"' in template
    assert 'aria-pressed="false"' in template
    assert 'id="frame-center"' in html and 'id="tracking-overlay"' in html
    assert "Add object" not in template
    assert "password" not in html.lower()
    assert "authentication" not in server.lower()
    assert "self.armed = False" in controller
    assert "XL330_TORQUE_ENABLE = 64" in controller
    assert "XL330_GOAL_POSITION = 116" in controller
    assert "XL330_PRESENT_POSITION = 132" in controller
    assert "PacketHandler(XL330_PROTOCOL_VERSION)" in controller
    assert '"dynamixel-2.0"' in controller
    assert "scservo_sdk" not in server
    assert "raise ServoDisarmed" in controller
    assert 'Path(__file__).with_name("static")' in server
    assert 'default=Path("/etc/' not in server

    print("validated repository and hardware contracts")


if __name__ == "__main__":
    main()
