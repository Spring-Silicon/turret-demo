#!/usr/bin/env python3
"""Packed feedback fault/parity tests. No serial hardware is used."""
import importlib.util
import sys
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("servo_tests", ROOT / "tests/test-server.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
servo = support.servo


class Packet(support.FakePacket):
    def __init__(self):
        super().__init__()
        self.reads = []
        self.responses = deque()
        for sid in (1, 2):
            self.registers[sid].update({6: 53, 68: 2})
            self.reset_aliases(sid)

    def reset_aliases(self, sid):
        base = 224 if self.registers[sid][6] >= 53 else 208
        self.registers[sid].update({168+2*i: base+i for i in range(8)})

    def byte(self, sid, addr):
        if 132 <= addr <= 135:
            return (self.registers[sid][132] >> (8*(addr-132))) & 255
        return self.registers[sid].get(addr, 0)

    def readTxRx(self, port, sid, addr, size):
        self.reads.append((sid, addr, size))
        if self.responses:
            response = self.responses.popleft()
            if response is not None:
                return response
        if sid in self.missing:
            return [], -3001, 0
        base = 224 if self.registers[sid][6] >= 53 else 208
        if addr != base or size != 8:
            return [], 0, 2
        return [self.byte(sid, self.registers[sid][168+2*i]) for i in range(size)], 0, 0


class PackedFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.config = support.server.load_config(ROOT / "config/spring-turret-demo.json")["servo"]
        self.config.update(calibrated=True, packed_feedback=True)
        self.config.pop("calibration_file", None)
        for axis in self.config["axes"].values():
            axis.update(center_position=2048, min_degrees=-90, max_degrees=90)
        self.packet = Packet()
        sdk = types.SimpleNamespace(COMM_SUCCESS=0, PortHandler=support.FakePort,
                                    PacketHandler=lambda _: self.packet)
        self.sdk = patch.dict(sys.modules, {"dynamixel_sdk": sdk})
        self.sdk.start()
        self.controller = servo.ServoController(self.config)
        self.addCleanup(self.sdk.stop)
        self.addCleanup(self.controller.stop)

    def assert_no_goals(self):
        self.assertFalse(any(addr == 116 for _, addr, _ in self.packet.writes))

    def test_config_is_opt_in_and_boolean(self):
        for bad in (0, 1, "true", None, {}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                servo.validate_config({**self.config, "packed_feedback": bad})
        self.config["packed_feedback"] = False
        self.controller.arm()
        self.controller._poll_locked()
        self.assertEqual(self.packet.reads, [])
        self.assertFalse(any(168 <= addr <= 182 for _, addr, _ in self.packet.writes))

    def test_firmware_bases_and_mapping_use_only_ram(self):
        self.packet.registers[1][6] = 52
        self.packet.reset_aliases(1)
        self.controller._poll_locked()
        self.assertEqual(self.packet.reads, [(2, 224, 8), (1, 208, 8)])
        for sid in (1, 2):
            self.assertEqual([self.packet.registers[sid][168+2*i] for i in range(8)],
                             list(servo.FEEDBACK_REGISTERS))
        self.assertFalse(any(addr < 64 for _, addr, _ in self.packet.writes))
        self.assert_no_goals()

    def test_models_and_torque_checked_before_mapping_either_axis(self):
        self.controller._open_locked()
        self.packet.model[1] = 42
        with self.assertRaises(servo.DeviceUnavailable):
            self.controller._prepare_feedback_locked()
        self.assertEqual(self.packet.writes, [])
        self.packet.model[1] = 1200
        self.packet.registers[1][64] = 1
        with self.assertRaises(servo.DeviceUnavailable):
            self.controller._prepare_feedback_locked()
        self.assertEqual(self.packet.writes, [])

    def test_bad_return_level_or_firmware_prevents_setup(self):
        self.controller._open_locked()
        for addr, value in ((68, 1), (6, 0)):
            before = self.packet.registers[1][addr]
            self.packet.registers[1][addr] = value
            with self.assertRaises(servo.DeviceUnavailable):
                self.controller._prepare_feedback_locked()
            self.assertEqual(self.packet.writes, [])
            self.packet.registers[1][addr] = before

    def test_every_point_gets_two_new_checked_packets(self):
        self.controller.arm()
        self.controller._poll_locked()
        self.packet.reads.clear()
        self.controller.point({"x": 5, "y": -5})
        self.assertEqual(self.packet.reads, [(2, 224, 8), (1, 224, 8)])
        self.assertEqual(self.controller.status()["feedback_mode"], "packed")

    def test_new_fault_after_healthy_poll_prevents_command(self):
        self.controller.arm()
        self.controller._poll_locked()
        self.packet.writes.clear()
        self.packet.responses.append(([], -1, 0))
        with self.assertRaises(servo.DeviceUnavailable):
            self.controller.point({"x": 5, "y": -5})
        self.assert_no_goals()
        self.assertFalse(self.controller.armed)
        self.assertIsNone(self.controller.sample_pose())

    def test_both_axes_checked_before_limit_recovery_or_goals(self):
        self.controller.arm()
        self.packet.registers[2][132] = 3200  # X outside its 90-degree range.
        self.packet.registers[1][70] = 4
        self.packet.writes.clear()
        with self.assertRaisesRegex(servo.DeviceUnavailable, "hardware fault"):
            self.controller.point({"x": 0, "y": 0})
        self.assert_no_goals()

    def test_invalid_packet_fields_all_fail_before_command(self):
        for label, field, value in (("torque", 4, 0), ("hardware", 5, 4),
                                     ("sentinel", 6, 0), ("watchdog", 7, 255),
                                     ("disabled watchdog", 7, 0), ("invalid torque", 4, 2)):
            with self.subTest(label=label):
                self.controller.arm()
                data = [0, 8, 0, 0, 1, 0, 2, 50]
                data[field] = value
                self.packet.responses.extend([None, (data, 0, 0)])
                self.packet.writes.clear()
                with self.assertRaises(servo.DeviceUnavailable):
                    self.controller.point({"x": 1, "y": 1})
                self.assert_no_goals()

    def test_device_errors_and_malformed_packets_not_retried(self):
        for response in (([], 0, 128), ([0]*7, 0, 0), ([0]*9, 0, 0),
                         ([0, 8, 0, 0, 1, 0, 2, 256], 0, 0)):
            with self.subTest(response=response):
                self.controller.arm()
                self.packet.reads.clear()
                self.packet.writes.clear()
                self.packet.responses.append(response)
                with self.assertRaises(servo.DeviceUnavailable):
                    self.controller.point({"x": 1, "y": 1})
                self.assertEqual(len(self.packet.reads), 1)
                self.assert_no_goals()

    def test_only_lost_or_corrupt_reads_retried_once_without_pose_history(self):
        self.controller.arm()
        self.controller._poll_locked()
        history = list(self.controller.pose_history)
        self.packet.responses.append(([255]*8, -3001, 0))
        self.controller._poll_locked()
        self.assertEqual(self.controller.read_retries, 1)
        self.assertEqual(list(self.controller.pose_history), history)
        self.assertTrue(self.controller.axes["x"]["position_retried"])
        self.assertEqual(self.controller.axes["x"]["position"], 2048)
        self.packet.responses.extend([([], -3002, 0), ([], -3002, 0)])
        self.packet.writes.clear()
        with self.assertRaisesRegex(servo.DeviceUnavailable, "communication=-3002"):
            self.controller.point({"x": 1, "y": 1})
        self.assert_no_goals()
        self.assertIsNone(self.controller.sample_pose())

    def test_signed_positions_and_per_axis_timestamps(self):
        self.packet.registers[2][132] = -2048 & 0xffffffff
        self.controller._poll_locked()
        self.assertEqual(self.controller.axes["x"]["position"], -2048)
        self.controller.arm()
        self.controller._poll_locked()
        x, y = [self.controller.axes[name] for name in ("x", "y")]
        self.assertLess(x["observed_at"], y["observed_at"])
        self.assertGreater(x["read_duration_ms"], 0)
        self.assertEqual(len(self.controller.pose_history), 1)
        self.packet.registers[2][132] = 1048576
        with self.assertRaisesRegex(servo.DeviceUnavailable, "invalid extended-position"):
            self.controller.point({"x": 0, "y": 0})

    def test_reset_while_unarmed_cannot_supply_fake_pose_to_arm(self):
        self.controller._poll_locked()
        self.packet.reset_aliases(1)
        self.packet.writes.clear()
        with self.assertRaisesRegex(servo.DeviceUnavailable, "mapping reset"):
            self.controller.arm()
        self.assert_no_goals()
        self.assertFalse(any(addr == 64 and value == 1 for _, addr, value in self.packet.writes))
        # A new connection reconfigures RAM but never automatically arms.
        self.controller._poll_locked()
        self.assertFalse(self.controller.armed)
        self.assertEqual(self.controller.axes["y"]["position"], 2048)
        self.controller.arm()
        self.assertTrue(self.controller.armed)

    def test_reset_while_armed_disarms_and_clears_history(self):
        self.controller.arm()
        self.controller._poll_locked()
        self.packet.reset_aliases(2)
        self.packet.writes.clear()
        with self.assertRaisesRegex(servo.DeviceUnavailable, "mapping reset"):
            self.controller.point({"x": 1, "y": 1})
        self.assert_no_goals()
        self.assertEqual(len(self.controller.pose_history), 0)
        self.assertEqual(self.packet.registers[1][64], 0)
        self.assertEqual(self.packet.registers[2][64], 0)

    def test_arm_checks_full_mapping_not_just_sentinel(self):
        self.controller._poll_locked()
        self.packet.registers[2][168] = 65
        self.packet.writes.clear()
        with self.assertRaisesRegex(servo.DeviceUnavailable, "mapping readback mismatch"):
            self.controller.arm()
        self.assert_no_goals()

    def test_partial_setup_failure_restores_prior_aliases_both_axes(self):
        self.packet.fail = ("write", 1, 174, 135)
        with self.assertRaises(servo.DeviceUnavailable):
            self.controller.arm()
        self.assertFalse(self.controller.armed)
        self.assert_no_goals()
        for sid in (1, 2):
            self.assertEqual([self.packet.registers[sid][168+2*i] for i in range(8)], list(range(224, 232)))
        self.assertEqual(self.controller._saved_aliases, {})

    def test_acknowledged_but_ineffective_mapping_write_fails_readback(self):
        original = self.packet.write2ByteTxRx
        def ignored(port, sid, addr, value):
            if (sid, addr, value) == (1, 174, 135):
                return 0, 0
            return original(port, sid, addr, value)
        with patch.object(self.packet, "write2ByteTxRx", side_effect=ignored):
            with self.assertRaisesRegex(servo.DeviceUnavailable, "mapping readback mismatch"):
                self.controller.arm()
        self.assert_no_goals()
        for sid in (1, 2):
            self.assertEqual([self.packet.registers[sid][168+2*i] for i in range(8)], list(range(224, 232)))

    def test_packet_device_error_with_timeout_is_not_retried(self):
        self.controller.arm()
        self.packet.responses.append(([], -3001, 128))
        self.packet.reads.clear()
        self.packet.writes.clear()
        with self.assertRaisesRegex(servo.DeviceUnavailable, "device_error=128"):
            self.controller.point({"x": 1, "y": 1})
        self.assertEqual(len(self.packet.reads), 1)
        self.assert_no_goals()

    def test_stop_restores_nondefault_prior_mapping(self):
        for sid in (1, 2):
            self.packet.registers[sid].update({168+2*i: 132+i for i in range(8)})
        self.controller.arm()
        self.controller.stop()
        for sid in (1, 2):
            self.assertEqual([self.packet.registers[sid][168+2*i] for i in range(8)], list(range(132, 140)))
            self.assertEqual(self.packet.registers[sid][64], 0)

    def test_stop_never_restores_mapping_on_wrong_model(self):
        self.controller.arm()
        self.packet.model[2] = 42
        self.packet.writes.clear()
        self.controller.stop()
        self.assertFalse(any(sid == 2 for sid, _, _ in self.packet.writes))
        self.assertEqual([self.packet.registers[1][168+2*i] for i in range(8)], list(range(224, 232)))
        self.assertIn("feedback restoration failed", self.controller.error)

    def test_stop_reports_restore_failure_and_attempts_other_axis(self):
        self.controller.arm()
        self.packet.fail = ("write", 2, 168, 224)
        self.controller.stop()
        self.assertIn("feedback restoration failed", self.controller.error)
        self.assertEqual([self.packet.registers[1][168+2*i] for i in range(8)], list(range(224, 232)))


if __name__ == "__main__":
    unittest.main()
