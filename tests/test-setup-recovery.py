"""First-install and damaged-state recovery through the real shared API."""
import copy
import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spring_turret.server import TurretApplication
from spring_turret.servo import ServoController, DeviceUnavailable, ServoDisarmed
from spring_turret.tracking import TrackingController
from spring_turret.calibration import load_zeros

fixtures = runpy.run_path(str(ROOT / "tests/test-server.py"))
geometry_fixture = runpy.run_path(str(ROOT / "tests/test-geometry.py"))["fixture"]


class Camera:
    def __init__(self, config): self.config = config
    def status(self): return {**self.config, "online": True, "identity": "test:usb:123"}


class SetupRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = json.loads((ROOT / "config/spring-turret-demo.json").read_text())
        self.config["servo"]["calibration_file"] = str(Path(self.tmp.name) / "state/zeros.json")
        self.config["tracking"] = {"calibrated": False}
        self.packet = fixtures["FakePacket"]()
        sdk = patch.dict(sys.modules, dynamixel_sdk=types.SimpleNamespace(COMM_SUCCESS=0,
            PortHandler=fixtures["FakePort"], PacketHandler=lambda _: self.packet))
        sdk.start()
        self.addCleanup(sdk.stop)
        self.app = self.application()

    def application(self):
        return TurretApplication(copy.deepcopy(self.config), camera=Camera(self.config["camera"]))

    def test_complete_first_install_via_commands_and_restart(self):
        self.app.servo._poll_locked()
        self.assertFalse(self.app.status()["servo"]["can_start"])
        self.assertTrue(self.app.status()["servo"]["can_recalibrate"])
        self.assertNotIn("can_confirm_setup", self.app.status()["tracking"])
        with self.assertRaises(DeviceUnavailable): self.app.command("/api/servo/arm", {})
        self.packet.writes.clear()
        state = self.app.command("/api/servo/recalibrate", {})
        self.assertTrue(state["servo"]["calibrated"])
        self.assertFalse(state["tracking"]["calibrated"])  # Zeroing is not optical/mapping calibration.
        self.assertEqual(self.packet.writes, [])
        self.assertIsNone(state["tracking"]["error"])
        self.assertEqual(state["tracking"]["mapping"], "linear")
        self.assertIsNone(state["tracking"]["target"])
        self.assertFalse(state["servo"]["run_requested"])
        restarted = self.application()
        self.assertIsNone(restarted.status()["tracking"]["error"])
        self.assertTrue(restarted.status()["servo"]["can_start"])
        self.assertFalse(restarted.status()["servo"]["run_requested"])
        restarted.command("/api/servo/arm", {})
        self.assertTrue(restarted.status()["servo"]["armed"])
        restarted.command("/api/servo/disable", {})
        self.config["tracking"]["x_direction"] = -1
        self.assertIsNone(self.application().status()["tracking"]["error"])

    def test_obsolete_tracking_confirmation_cannot_block_start(self):
        self.app.command("/api/servo/recalibrate", {})
        path = Path(self.config["servo"]["calibration_file"]).with_suffix(".tracking.json")
        path.write_text("{truncated")
        app = self.application()
        app.servo._poll_locked()
        self.assertIsNone(app.status()["tracking"]["error"])
        self.assertFalse(app.status()["tracking"]["calibrated"])
        app.command("/api/servo/arm", {})
        self.assertTrue(app.status()["servo"]["armed"])
        self.assertEqual(path.read_text(), "{truncated")  # Obsolete metadata is ignored.

    def test_missing_or_invalid_geometry_keeps_api_alive_and_retries_restored_file(self):
        path = Path(self.tmp.name) / "geometry.json"
        self.config["tracking"].update(calibrated=True, geometry_file=str(path))
        app = self.application()
        self.assertIn("Restore", app.status()["tracking"]["error"])
        self.assertIsNone(app.tracking.geometry)
        self.assertIsNone(app.detection.instances.geometry)
        path.write_text("null")
        app.tracking.geometry_resource.retry_at = 0
        app.tracking._tick()
        self.assertIn("geometry unavailable", app.status()["tracking"]["error"])
        path.write_text(json.dumps(geometry_fixture()))
        app.tracking.geometry_resource.retry_at = 0
        app.detection.instances.geometry_resource.retry_at = 0
        app.tracking._tick()
        app.detection.instances.update([], None, 1)
        self.assertIsNotNone(app.tracking.geometry)
        self.assertIsNotNone(app.detection.instances.geometry)
        self.assertIsNone(app.status()["tracking"]["error"])
        self.assertEqual(app.status()["tracking"]["mapping"], "fisheye-kinematics")
        self.assertFalse(app.status()["servo"]["armed"])

    def test_corrupt_zeros_and_gains_do_not_take_down_application(self):
        path = Path(self.config["servo"]["calibration_file"])
        path.parent.mkdir()
        path.write_text("bad")
        path.with_suffix(".gains.json").write_text("null")
        app = self.application()
        app.servo._poll_locked()
        state = app.status()
        self.assertTrue(state["servo"]["can_recalibrate"])
        self.assertTrue(state["servo"]["can_recover_gains"])
        self.assertFalse(state["servo"]["can_start"])
        app.command("/api/servo/recover-gains", {})
        self.assertFalse(app.status()["servo"]["can_start"])  # Still needs real zero confirmation.
        app.command("/api/servo/recalibrate", {})
        self.assertTrue(app.status()["servo"]["can_start"])
        self.assertFalse(app.status()["servo"]["armed"])
        self.assertTrue(list(path.parent.glob("*.invalid-*")))

    def test_legacy_zero_file_does_not_certify_uncommissioned_setup(self):
        self.app.command("/api/servo/recalibrate", {})
        path = Path(self.config["servo"]["calibration_file"])
        data = json.loads(path.read_text())
        data["version"] = 1
        data.pop("controller")
        path.write_text(json.dumps(data))
        app = self.application()
        self.assertFalse(app.status()["servo"]["calibrated"])
        app.servo._poll_locked()
        self.assertTrue(app.status()["servo"]["can_recalibrate"])

    def test_setup_commands_reject_extra_fields(self):
        for command in ("/api/servo/recalibrate", "/api/servo/recover-gains"):
            with self.subTest(command=command):
                with self.assertRaises(ValueError): self.app.command(command, {"armed": True})
        self.assertFalse(self.app.servo.armed)


if __name__ == "__main__": unittest.main()
