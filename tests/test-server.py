#!/usr/bin/env python3
"""Two-axis controller and HTTP tests, with no physical motor writes."""
import copy
import json
import sys
import threading
import time
import types
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spring_turret import server, servo


class FakePort:
    def __init__(self, *_): pass
    def openPort(self): return True
    def setBaudRate(self, _): return True
    def closePort(self): pass


class FakePacket:
    def __init__(self):
        defaults = {10: 0, 11: 4, 12: 255, 20: 0, 48: 4095, 52: 0,
                    64: 0, 70: 0, 98: 0, 116: 100, 132: 2048}
        self.registers = {1: defaults.copy(), 2: defaults.copy()}
        self.writes = []
        self.fail = None
        self.missing = set()
        self.model = {1: 1200, 2: 1200}

    def ping(self, port, sid):
        return (0, -1, 0) if sid in self.missing else (self.model[sid], 0, 0)

    def read1ByteTxRx(self, port, sid, addr):
        if sid in self.missing or self.fail == ("read", sid, addr):
            return 0, -1, 0
        return self.registers[sid][addr], 0, 0

    read4ByteTxRx = read1ByteTxRx

    def write1ByteTxRx(self, port, sid, addr, value):
        self.writes.append((sid, addr, value))
        if sid in self.missing or self.fail == ("write", sid, addr, value):
            return -1, 0
        self.registers[sid][addr] = value
        return 0, 0

    write4ByteTxRx = write1ByteTxRx


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.config = server.load_config(ROOT / "config/spring-turret-demo.json")
        self.config["servo"]["calibrated"] = True
        for axis in self.config["servo"]["axes"].values():
            axis.update(min_degrees=-45, max_degrees=45)
        self.packet = FakePacket()
        sdk = types.SimpleNamespace(COMM_SUCCESS=0, PortHandler=FakePort,
                                    PacketHandler=lambda _: self.packet)
        self.sdk = patch.dict(sys.modules, {"dynamixel_sdk": sdk})
        self.sdk.start()
        self.controller = servo.ServoController(self.config["servo"])

    def tearDown(self):
        self.controller.stop()
        self.sdk.stop()

    def test_arm_holds_both_then_moves_only_selected_axis(self):
        with self.assertRaises(servo.ServoDisarmed): self.controller.move("x", 10)
        self.assertEqual(self.packet.writes, [])
        self.controller.arm()
        writes = self.packet.writes
        first_enable = next(i for i, (_, a, v) in enumerate(writes) if a == 64 and v == 1)
        self.assertIn((2, 116, 2048), writes[:first_enable])
        self.assertIn((1, 116, 2048), writes[:first_enable])
        self.controller.move("x", 10)
        self.assertEqual(self.packet.registers[2][116], 2162)
        self.assertEqual(self.packet.registers[1][116], 2048)
        self.controller.move("y", -20)
        self.assertEqual(self.packet.registers[1][116], 1820)
        state = self.controller.status()
        self.assertEqual(state["axes"]["x"]["position"], 2048)
        self.assertAlmostEqual(state["axes"]["x"]["goal_degrees"], 10, delta=.05)
        self.controller.disable()
        self.assertFalse(self.controller.armed)
        self.assertEqual([self.packet.registers[i][64] for i in (1, 2)], [0, 0])

    def test_partial_arm_failure_turns_first_motor_back_off(self):
        self.packet.fail = ("write", 1, 64, 1)
        with self.assertRaises(servo.DeviceUnavailable): self.controller.arm()
        self.assertIn((2, 64, 1), self.packet.writes)
        self.assertEqual([self.packet.registers[i][64] for i in (1, 2)], [0, 0])
        self.assertFalse(self.controller.armed)

    def test_stop_attempts_both_even_if_x_fails(self):
        self.controller.arm()
        self.packet.fail = ("write", 2, 64, 0)
        with self.assertRaises(servo.DeviceUnavailable): self.controller.disable()
        self.assertEqual(self.packet.registers[1][64], 0)
        self.assertIsNone(self.controller.status()["axes"]["x"]["torque"])
        self.assertIn("unconfirmed", self.controller.error)

    def test_missing_motor_never_arms_other(self):
        self.packet.missing.add(1)
        with self.assertRaises(servo.DeviceUnavailable): self.controller.arm()
        self.assertFalse(any(a == 64 and v == 1 for _, a, v in self.packet.writes))

    def test_wrong_model_never_written(self):
        self.packet.model[1] = 42
        with self.assertRaises(servo.DeviceUnavailable): self.controller.arm()
        self.assertFalse(any(sid == 1 for sid, _, _ in self.packet.writes))

    def test_disconnect_stops_other_and_reconnect_is_disarmed(self):
        self.controller.arm()
        self.packet.missing.add(2)
        with self.assertRaises(servo.DeviceUnavailable): self.controller.move("y", 10)
        self.assertEqual(self.packet.registers[1][64], 0)
        self.packet.missing.clear()
        self.controller._poll_locked()
        self.assertFalse(self.controller.armed)
        self.assertEqual([self.packet.registers[i][64] for i in (1, 2)], [0, 0])

    def test_modes_and_ranges_fail_before_torque_on(self):
        for address, value in ((11, 1), (11, 3), (10, 8), (10, 4), (10, 1), (12, 2), (20, 10),
                               (132, 100), (132, 4294967295), (70, 4)):
            with self.subTest(address=address, value=value):
                original = self.packet.registers[1][address]
                self.packet.registers[1][address] = value
                self.packet.writes.clear()
                with self.assertRaises(servo.DeviceUnavailable): self.controller.arm()
                self.assertFalse(any(a == 64 and v == 1 for _, a, v in self.packet.writes))
                self.packet.registers[1][address] = original
        self.config["servo"]["calibrated"] = False
        with self.assertRaises(servo.DeviceUnavailable): self.controller.arm()

    def test_lease_timeout_rejects_late_move(self):
        self.controller.arm()
        self.controller.last_keepalive = time.monotonic() - 4
        with self.assertRaises(servo.DeviceUnavailable): self.controller.move("x", 10)
        self.assertEqual([self.packet.registers[i][64] for i in (1, 2)], [0, 0])
        self.controller.keepalive()
        self.assertFalse(self.controller.armed)

    def test_invalid_degrees_or_axis_never_write(self):
        for name, value in [("z", 0), ([], 0), ("x", True), ("x", "10"), ("y", None),
                            ("x", float("nan")), ("x", float("inf")), ("y", 46), ("x", -46)]:
            with self.assertRaises(ValueError): self.controller.move(name, value)
        self.assertEqual(self.packet.writes, [])

    def test_direction_conversion(self):
        axis = copy.deepcopy(self.config["servo"]["axes"]["y"])
        axis["direction"] = -1
        self.assertEqual(servo.to_position(axis, 45), 1536)
        self.assertEqual(servo.to_degrees(axis, 1536), 45)

    def test_outside_feedback_returns_both_axes_without_stopping(self):
        self.controller.arm()
        lease = self.controller.last_keepalive
        for positions, targets in (((2600, 1500), (2560, 1536)),
                                   ((1500, 2600), (1536, 2560))):
            with self.subTest(positions=positions):
                self.packet.writes.clear()
                for sid, position in zip((2, 1), positions):
                    self.packet.registers[sid][132] = position
                self.controller._tick_locked()
                self.assertEqual(self.packet.writes, [(2, 116, targets[0]), (1, 116, targets[1])])
                state = self.controller.status()
                self.assertTrue(state["armed"])
                self.assertTrue(state["ready"])
                self.assertIsNone(state["error"])
                self.assertEqual([self.packet.registers[i][64] for i in (1, 2)], [1, 1])
                # Recovery is not a browser heartbeat and cannot renew the lease.
                self.assertEqual(self.controller.last_keepalive, lease)
                self.packet.writes.clear()
                self.controller._tick_locked()
                self.assertEqual(self.packet.writes, [])

    def test_recovery_preserves_new_inward_commands_until_next_excursion(self):
        self.controller.arm()
        self.packet.registers[2][132] = 2600
        self.controller._tick_locked()
        self.assertEqual(self.packet.registers[2][116], 2560)
        self.controller.move("x", 0)
        self.packet.writes.clear()
        self.controller._tick_locked()
        self.assertEqual(self.packet.writes, [])
        self.assertEqual(self.packet.registers[2][116], 2048)
        # Once feedback has returned inside, a new excursion gets corrected.
        self.packet.registers[2][132] = 2560
        self.controller._tick_locked()
        self.packet.registers[2][132] = 2561
        self.controller._tick_locked()
        self.assertEqual(self.packet.writes, [(2, 116, 2560)])
        # Already heading to the limit: do not rewrite/reset that goal.
        self.packet.registers[2][132] = 2560
        self.controller._tick_locked()
        self.packet.registers[2][132] = 2600
        self.packet.writes.clear()
        self.controller._tick_locked()
        self.assertEqual(self.packet.writes, [])

    def test_recovery_respects_asymmetric_reversed_and_rollover_coordinates(self):
        axes = self.config["servo"]["axes"]
        axes["x"].update(center_position=2031, min_degrees=-90, max_degrees=90)
        axes["y"].update(center_position=4065, min_degrees=30, max_degrees=90)
        self.packet.registers[2][132] = 2031
        for direction, turn_offset in ((1, 0), (1, -4096), (-1, 0), (-1, -4096)):
            with self.subTest(direction=direction, turn_offset=turn_offset):
                axes["y"]["direction"] = direction
                axis = {**axes["y"], "center_position": 4065 + turn_offset}
                self.packet.registers[1][132] = servo.to_position(axis, 45) & 0xFFFFFFFF
                self.controller.arm()
                for measured, target in ((29, 30), (91, 90)):
                    self.packet.registers[1][132] = servo.to_position(axis, measured) & 0xFFFFFFFF
                    self.packet.writes.clear()
                    self.controller._tick_locked()
                    goal = servo.to_position(axis, target)
                    self.assertEqual(self.packet.writes, [(1, 116, goal)])
                    self.assertEqual(self.controller.axes["y"]["origin"], axis["center_position"])
                    self.assertAlmostEqual(self.controller.status()["axes"]["y"]["goal_degrees"], target, delta=.05)
                    self.assertTrue(self.controller.armed)
                self.controller.disable()

    def test_stopped_feedback_never_causes_recovery_or_auto_arm(self):
        self.controller.arm()
        self.packet.registers[2][132] = 2600
        self.controller._tick_locked()
        self.controller.disable()
        self.packet.writes.clear()
        self.controller._tick_locked()
        self.assertEqual(self.packet.writes, [])
        self.assertFalse(self.controller.armed)
        self.assertFalse(self.controller.status()["ready"])
        with self.assertRaises(servo.DeviceUnavailable): self.controller.arm()
        self.assertFalse(any(addr == 116 or (addr == 64 and value == 1)
                             for _, addr, value in self.packet.writes))
        # Stop/re-arm resets the excursion tracker, even at the same boundary.
        self.packet.registers[2][132] = 2048
        self.controller.arm()
        self.packet.registers[2][132] = 2600
        self.packet.writes.clear()
        self.controller._tick_locked()
        self.assertEqual(self.packet.writes, [(2, 116, 2560)])

    def test_all_fault_checks_precede_recovery_writes(self):
        for fault in ("hardware", "torque", "communication", "lease", "position_high", "position_low"):
            with self.subTest(fault=fault):
                self.packet.registers[2][132] = 2048
                self.packet.registers[1][132] = 2048
                self.controller.arm()
                self.packet.registers[2][132] = 2600
                if fault == "hardware": self.packet.registers[1][70] = 4
                if fault == "torque": self.packet.registers[1][64] = 0
                if fault == "communication": self.packet.fail = ("read", 1, 132)
                if fault == "lease": self.controller.last_keepalive = time.monotonic() - 4
                if fault == "position_high": self.packet.registers[1][132] = 1048576
                if fault == "position_low": self.packet.registers[1][132] = -1048576 & 0xFFFFFFFF
                self.packet.writes.clear()
                with self.assertRaises(servo.DeviceUnavailable): self.controller.move("x", 0)
                self.assertFalse(any(addr == 116 for _, addr, _ in self.packet.writes))
                self.assertFalse(self.controller.armed)
                self.assertEqual([self.packet.registers[i][64] for i in (1, 2)], [0, 0])
                self.packet.fail = None
                self.packet.registers[1][70] = 0

    def test_recovery_write_failure_stops_both(self):
        self.controller.arm()
        self.packet.registers[2][132] = 2600
        self.packet.fail = ("write", 2, 116, 2560)
        self.packet.writes.clear()
        with self.assertRaises(servo.DeviceUnavailable): self.controller.move("y", 10)
        self.assertEqual(self.packet.writes, [(2, 116, 2560), (2, 64, 0), (1, 64, 0)])
        self.assertFalse(self.controller.armed)
        self.assertEqual([self.packet.registers[i][64] for i in (1, 2)], [0, 0])

    def test_uncapped_profiles_are_written_before_torque_on(self):
        for axis in self.config["servo"]["axes"].values():
            axis.update(profile_velocity=0, profile_acceleration=0)
        servo.validate_config(self.config["servo"])
        self.controller.arm()
        writes = self.packet.writes
        first_enable = next(i for i, (_, a, v) in enumerate(writes) if a == 64 and v == 1)
        for sid in (1, 2):
            self.assertIn((sid, 108, 0), writes[:first_enable])
            self.assertIn((sid, 112, 0), writes[:first_enable])
            self.assertIn((sid, 116, 2048), writes[:first_enable])
            self.assertEqual(self.packet.registers[sid][98], 50)
        self.controller.disable()
        self.assertEqual([self.packet.registers[i][64] for i in (1, 2)], [0, 0])

    def test_profile_register_ranges(self):
        for field in ("profile_velocity", "profile_acceleration"):
            for value in (0, 1, 100, 32767):
                config = copy.deepcopy(self.config["servo"])
                config["axes"]["x"][field] = value
                servo.validate_config(config)
            for value in (-1, 32768, True, 0.5, "0"):
                config = copy.deepcopy(self.config["servo"])
                config["axes"]["x"][field] = value
                with self.assertRaises(ValueError): servo.validate_config(config)

    def test_rollover_short_moves_and_reboot_coordinate(self):
        self.config["servo"]["axes"]["y"]["center_position"] = 4065
        self.packet.registers[1][132] = 4065
        self.controller.arm()
        self.controller.move("y", 5)
        self.assertEqual(self.packet.registers[1][116], 4122)
        self.controller.disable()
        # After reboot the encoder reports 26 instead of 4122; keep the same angle.
        self.packet.registers[1][132] = 26
        self.controller.arm()
        self.assertAlmostEqual(self.controller.status()["axes"]["y"]["degrees"], 5, delta=.05)
        self.controller.move("y", 0)
        self.assertEqual(self.packet.registers[1][116], -31)
        self.assertLess(abs(self.packet.registers[1][116] - 26), 60)

    def test_config_validation(self):
        for key, value in (("id", 2), ("direction", 0), ("center_position", -1),
                           ("min_degrees", -180), ("min_degrees", 45),
                           ("max_degrees", 180), ("max_degrees", float("nan")), ("profile_velocity", True)):
            config = copy.deepcopy(self.config["servo"])
            config["axes"]["y"][key] = value
            with self.assertRaises(ValueError): servo.validate_config(config)

    def test_requested_asymmetric_limits_and_rearm_gate(self):
        axes = self.config["servo"]["axes"]
        axes["x"].update(min_degrees=-90, max_degrees=90)
        axes["y"].update(min_degrees=30, max_degrees=90)
        servo.validate_config(self.config["servo"])
        self.controller._poll_locked()
        self.controller.error = ""  # A successful monitor iteration clears startup status.
        self.assertFalse(self.controller.status()["ready"])
        self.assertIn("Y outside", self.controller.status()["error"])
        with self.assertRaises(servo.DeviceUnavailable): self.controller.arm()
        self.assertFalse(any(a == 64 and v == 1 for _, a, v in self.packet.writes))
        self.packet.registers[1][132] = servo.to_position(axes["y"], 45)
        self.controller._poll_locked()
        self.assertTrue(self.controller.status()["ready"])
        self.controller.arm()
        for name, degrees in (("x", -90), ("x", 90), ("y", 30), ("y", 90)):
            self.controller.move(name, degrees)
            self.assertEqual(self.packet.registers[axes[name]["id"]][116], servo.to_position(axes[name], degrees))
        previous = self.packet.writes.copy()
        for name, degrees in (("x", -90.01), ("x", 90.01), ("y", 29.99), ("y", 90.01), ("y", 0)):
            with self.assertRaises(ValueError): self.controller.move(name, degrees)
        self.assertEqual(previous, self.packet.writes)

    def test_http(self):
        class Camera:
            def status(self): return {"online": True}
        app = server.TurretApplication(self.config, camera=Camera(), servo=self.controller)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
        httpd.daemon_threads = True
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        def request(path, body=None):
            conn = HTTPConnection("127.0.0.1", httpd.server_port, timeout=3)
            conn.request("POST", path, json.dumps(body) if body is not None else None,
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            data = response.read()
            conn.close()
            return response.status, json.loads(data)
        try:
            self.assertEqual(request("/api/servo/position", {"axis": "x", "degrees": 10})[0], 409)
            self.assertEqual(request("/api/servo/arm")[0], 200)
            code, body = request("/api/servo/position", {"axis": "x", "degrees": 10})
            self.assertEqual(code, 200)
            self.assertEqual(body["servo"]["axes"]["x"]["goal"], 2162)
            for payload in ({"position": 2200}, {"axis": "x", "degrees": "10"},
                            {"axis": "y", "degrees": 100}, {"axis": "z", "degrees": 10},
                            {"axis": "x", "degrees": True}, {"axis": "x", "degrees": float("nan")}):
                self.assertEqual(request("/api/servo/position", payload)[0], 400)
            self.assertEqual(request("/api/servo/keepalive")[0], 200)
            self.assertFalse(request("/api/servo/disable")[1]["servo"]["armed"])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
