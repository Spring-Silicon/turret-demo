"""Device-bound P/D overrides, separate from mechanical zeros and I gains."""
import json
import os
from pathlib import Path
import tempfile


def validate_pd(value):
    if (not isinstance(value, dict) or set(value) != {"p", "d"}
            or any(type(v) is not int for v in value.values())
            or not 1 <= value["p"] <= 16383 or not 0 <= value["d"] <= 16383):
        raise ValueError("P must be an integer 1–16383; D must be an integer 0–16383")


class GainSettings:
    def __init__(self, config):
        self.path = (Path(config["calibration_file"]).with_suffix(".gains.json")
                     if config.get("calibration_file") else None)
        self.identity = {name: {k: axis[k] for k in ("id", "direction")}
                         for name, axis in config["axes"].items()}
        self.values = {}
        self.baseline = {name: {k: axis["position_gains"][k] for k in ("p", "d")}
                         for name, axis in config["axes"].items() if axis.get("position_gains")}
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            if (not isinstance(data, dict) or data.get("version") != 1
                    or data.get("identity") != self.identity):
                raise ValueError("Saved servo gains do not match configured axes")
            for field in ("values", "baseline"):
                entries = data.get(field)
                if not isinstance(entries, dict) or not set(entries) <= {"x", "y"}:
                    raise ValueError("Invalid saved servo gains")
                for entry in entries.values():
                    validate_pd(entry)
            if not set(data["values"]) <= set(data["baseline"]):
                raise ValueError("Saved servo gains lack a reset baseline")
            self.values = data["values"]
            # Explicit installation baselines take precedence over older files.
            self.baseline = {**data["baseline"], **self.baseline}

    def save(self, name, value):
        values = {**self.values, name: dict(value)}
        if self.path:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent,
                                                 prefix=".servo-gains-", delete=False) as out:
                    temporary = Path(out.name)
                    json.dump({"version": 1, "identity": self.identity,
                               "baseline": self.baseline, "values": values}, out)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        self.values = values
