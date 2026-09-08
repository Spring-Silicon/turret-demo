"""Atomic, device-bound servo zero persistence; no hardware register changes."""

import json
import os
import tempfile
from pathlib import Path


def load_zeros(config: dict) -> None:
    if not config.get("calibration_file"):
        return
    path = Path(config["calibration_file"])
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return
    if (not isinstance(data, dict) or data.get("version") != 1
            or not isinstance(data.get("axes"), dict) or set(data["axes"]) != {"x", "y"}):
        raise ValueError("Invalid saved servo zeros")
    for name, axis in config["axes"].items():
        saved = data["axes"][name]
        if (not isinstance(saved, dict) or set(saved) != {"id", "direction", "center_position"}
                or any(type(saved[key]) is not int for key in saved)
                or saved["id"] != axis["id"] or saved["direction"] != axis["direction"]
                or not 0 <= saved["center_position"] <= 4095):
            raise ValueError("Saved servo zeros do not match configured axes")
    for name, saved in data["axes"].items():
        config["axes"][name]["center_position"] = saved["center_position"]


def save_zeros(config: dict, positions: dict[str, int]) -> None:
    path = Path(config["calibration_file"])
    data = {"version": 1, "axes": {
        name: {"id": axis["id"], "direction": axis["direction"],
               "center_position": positions[name] % 4096}
        for name, axis in config["axes"].items()
    }}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".servo-zeros-",
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
