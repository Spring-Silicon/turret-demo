"""Operator-authorized, bounded static-scene calibration. No direct bus access."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--base", default="http://localhost:8080")
p.add_argument("--allow-motion", action="store_true", required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
def api(path="/api/status", body=None):
    request = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=4) as r:
        return json.load(r)
initial = api()
assert initial["camera"]["online"] and initial["servo"]["ready"]
assert not initial["servo"]["armed"], "Start with motors stopped; operator state must be explicit"
origin = {k: v["origin"] for k,v in initial["servo"]["axes"].items()}
start = {k: v["degrees"] for k,v in initial["servo"]["axes"].items()}
for k, envelope in (("x", 8), ("y", 6)):
    axis = initial["servo"]["axes"][k]
    assert axis["min_degrees"] <= start[k]-envelope and start[k]+envelope <= axis["max_degrees"]
props = dict(line.split("=", 1) for line in subprocess.check_output(
    ["udevadm", "info", "--query=property", "--name=" + initial["camera"]["device"]], text=True).splitlines() if "=" in line)
identity = ":".join(props[k] for k in ("ID_VENDOR_ID", "ID_MODEL_ID", "ID_SERIAL_SHORT"))
receipt = {"initial": initial, "identity": identity, "samples": []}
(a.output / "capture.json").write_text(json.dumps(receipt, indent=2))
expected = dict(start)
def checked():
    s = api()
    assert s["servo"]["armed"] and s["servo"]["online"], "Stopped/offline; never re-arm automatically"
    assert s["tracking"]["target"] is None, "Operator changed tracking target"
    for k,v in s["servo"]["axes"].items():
        assert v["origin"] == origin[k] and abs(v["goal_degrees"]-expected[k]) < .15, "Operator changed calibration/goal"
        assert abs(v["degrees"]-start[k]) <= (11 if k == "x" else 9), "Motion left bounded envelope"
    api("/api/servo/keepalive", {})
    return s
def settle():
    history = []
    deadline = time.monotonic()+4
    while time.monotonic() < deadline:
        s = checked()
        q = {k:v["degrees"] for k,v in s["servo"]["axes"].items()}
        history.append(q)
        if len(history) >= 5 and all(max(h[k] for h in history[-5:])-min(h[k] for h in history[-5:]) <= .18
                and abs(q[k]-expected[k]) < 3 for k in q):
            return s
        time.sleep(.12)
    raise RuntimeError("Encoders did not settle")
def move(goal):
    for k in ("x", "y"):
        while abs(goal[k]-expected[k]) > .001:
            checked()
            value = expected[k] + max(-3, min(3, goal[k]-expected[k]))
            api("/api/servo/position", {"axis": k, "degrees": value})
            expected[k] = value
            time.sleep(.14)
    return settle()
def capture(index, split):
    before = checked()
    with urllib.request.urlopen(a.base + "/stream.mjpg", timeout=3) as r:
        buffer = b""
        while b"\xff\xd9" not in buffer:
            buffer += r.read(4096)
            assert len(buffer) < 4_000_000
    jpeg = buffer[buffer.index(b"\xff\xd8"):buffer.index(b"\xff\xd9")+2]
    after = checked()
    q = {k: (before["servo"]["axes"][k]["degrees"] + after["servo"]["axes"][k]["degrees"])/2 for k in expected}
    assert all(abs(before["servo"]["axes"][k]["degrees"]-after["servo"]["axes"][k]["degrees"]) <= .18 for k in expected)
    name = f"{index:02d}.jpg"
    (a.output / name).write_bytes(jpeg)
    receipt["samples"].append({"image": name, "split": split, "pose": q})
    (a.output / "capture.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt["samples"][-1]), flush=True)
armed = False
try:
    api("/api/tracking/target", {"target": None})
    armed = True  # Also attempt Stop if the arm response itself is lost.
    s = api("/api/servo/arm", {})
    expected = {k:v["goal_degrees"] for k,v in s["servo"]["axes"].items()}
    settle()
    offsets = [(0,0,"train"),(8,0,"train"),(-8,0,"train"),(0,6,"train"),(0,-6,"train"),
               (8,6,"train"),(-8,6,"train"),(8,-6,"train"),(-8,-6,"train"),
               (5,3,"test"),(-5,3,"test"),(5,-3,"test"),(-5,-3,"test"),(2,0,"test"),(0,-2,"test")]
    for index, (dx,dy,split) in enumerate(offsets):
        move(start)  # Never jump directly between opposite envelope corners.
        move({"x": start["x"]+dx, "y": start["y"]+dy})
        capture(index, split)
    move(start)
    receipt["restored"] = checked()["servo"]
finally:
    # An operator Stop/goal change aborts without trying to restore motion.
    if armed:
        api("/api/servo/disable", {})
    receipt["final"] = api()["servo"]
    (a.output / "capture.json").write_text(json.dumps(receipt, indent=2))
