"""Atomic, device-bound servo zero persistence; no hardware register changes."""

import json
import os
import tempfile
import hashlib
import shutil
import time
from pathlib import Path


def controller_identity(config):
    # The stable USB basename is identical inside /host/dev and on the host.
    return {"device": Path(config["device"]).name,
            "axes": {name: {key: axis[key] for key in ("id", "direction")}
                     for name, axis in config["axes"].items()}}


def ensure_storage(config):
    if not config.get("calibration_file"):
        identity = json.dumps(controller_identity(config), sort_keys=True).encode()
        root = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        config["calibration_file"] = str(root / "spring-turret" /
            ("servo-zeros-" + hashlib.sha256(identity).hexdigest()[:16] + ".json"))


def preserve_invalid(path):
    """Keep an operator-replaced invalid file for diagnosis/recovery."""
    path = Path(path)
    if path.exists():
        backup = path.with_name(path.name + f".invalid-{time.time_ns()}")
        shutil.copy2(path, backup)
        return backup


def atomic_json(path, data, prefix=".servo-zeros-"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=prefix,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_zeros(config: dict) -> None:
    if not config.get("calibration_file"):
        return
    path = Path(config["calibration_file"])
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return
    if (not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] not in (1, 2)
            or not isinstance(data.get("axes"), dict) or set(data["axes"]) != {"x", "y"}):
        raise ValueError("Invalid saved servo zeros")
    if data["version"] == 2 and data.get("controller") != controller_identity(config):
        raise ValueError("Saved servo zeros do not match the configured controller")
    for name, axis in config["axes"].items():
        saved = data["axes"][name]
        if (not isinstance(saved, dict) or set(saved) != {"id", "direction", "center_position"}
                or any(type(saved[key]) is not int for key in saved)
                or saved["id"] != axis["id"] or saved["direction"] != axis["direction"]
                or not 0 <= saved["center_position"] <= 4095):
            raise ValueError("Saved servo zeros do not match configured axes")
    for name, saved in data["axes"].items():
        config["axes"][name]["center_position"] = saved["center_position"]
    # v2 records are written only after live commissioning/torque/stability
    # checks and an explicit Set zeros action. Legacy files don't certify
    # an installation that is otherwise marked uncalibrated.
    if data["version"] == 2:
        config["calibrated"] = True


def save_zeros(config: dict, positions: dict[str, int]) -> None:
    path = Path(config["calibration_file"])
    data = {"version": 2, "controller": controller_identity(config), "axes": {
        name: {"id": axis["id"], "direction": axis["direction"],
               "center_position": positions[name] % 4096}
        for name, axis in config["axes"].items()
    }}
    atomic_json(path, data)
