"""Bounded two-axis XL330 controller; one serial port and one bus lock."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from spring_turret.calibration import load_zeros, save_zeros
from spring_turret.pose_history import interpolate_pose
from spring_turret.servo_gains import GainSettings, validate_pd

LOGGER = logging.getLogger("spring-turret.servo")
XL330_PROTOCOL_VERSION = 2.0
XL330_TORQUE_ENABLE = 64
XL330_GOAL_POSITION = 116
XL330_PRESENT_POSITION = 132
POSITION_GAIN_REGISTERS = {"p": 84, "i": 82, "d": 80}
# RAM aliases: signed position, torque, fault, return-level sentinel, watchdog.
# The sentinel detects a reset to the default (zero-valued) indirect-data table.
FEEDBACK_REGISTERS = (132, 133, 134, 135, 64, 70, 68, 98)
INDIRECT_ADDRESS = 168
COUNTS_PER_DEGREE = 4096 / 360


class TurretError(RuntimeError):
    """Error safe to show to the operator."""


class DeviceUnavailable(TurretError):
    pass


class ServoDisarmed(TurretError):
    pass


def validate_config(config: dict[str, Any]) -> None:
    if not Path(config.get("device", "")).is_absolute():
        raise ValueError("servo.device must be an absolute path")
    if config.get("protocol") != "dynamixel-2.0":
        raise ValueError("servo.protocol must be dynamixel-2.0")
    if config.get("model_number") != 1200:
        raise ValueError("servo.model_number must be 1200 (XL330-M288-T)")
    if config.get("operating_mode") != 4:
        raise ValueError("servo.operating_mode must be 4 (extended position)")
    if config.get("baudrate") not in (9600, 57600, 115200, 1000000, 2000000, 3000000, 4000000):
        raise ValueError("unsupported XL330 baudrate")
    if type(config.get("calibrated")) is not bool:
        raise ValueError("servo.calibrated must be a boolean")
    if type(config.get("packed_feedback", False)) is not bool:
        raise ValueError("servo.packed_feedback must be a boolean")
    if "calibration_file" in config and (
        not isinstance(config["calibration_file"], str)
        or not Path(config["calibration_file"]).is_absolute()
    ):
        raise ValueError("servo.calibration_file must be an absolute path")
    axes = config.get("axes", {})
    if set(axes) != {"x", "y"}:
        raise ValueError("servo.axes must contain x and y")
    for name, axis in axes.items():
        for field in ("id", "center_position", "direction", "profile_velocity", "profile_acceleration"):
            if type(axis.get(field)) is not int:
                raise ValueError(f"servo.axes.{name}.{field} must be an integer")
        if not 0 <= axis["id"] <= 252 or axis["direction"] not in (-1, 1):
            raise ValueError(f"invalid {name} ID or direction")
        for field in ("min_degrees", "max_degrees"):
            value = axis.get(field)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name}.{field} must be a finite number")
        if not -180 < axis["min_degrees"] < axis["max_degrees"] < 180:
            raise ValueError(f"{name} limits must increase and stay strictly within ±180 degrees")
        if not 0 <= axis["center_position"] <= 4095:
            raise ValueError(f"{name} center_position must be a single-turn encoder reading")
        # In velocity-profile mode, zero explicitly disables that profile cap.
        for field in ("profile_velocity", "profile_acceleration"):
            if not 0 <= axis[field] <= 32767:
                raise ValueError(f"{name} {field} must be between 0 and 32767 (0 = uncapped)")
        gains = axis.get("position_gains")
        if gains is not None:
            if (not isinstance(gains, dict) or set(gains) != set(POSITION_GAIN_REGISTERS)
                    or any(type(v) is not int or not 0 <= v <= 16383 for v in gains.values())
                    or gains["p"] == 0):
                raise ValueError(f"{name} position_gains must contain integer p/i/d register values (p > 0)")
    if axes["x"]["id"] == axes["y"]["id"]:
        raise ValueError("X and Y must have different servo IDs")


def to_position(axis: dict, degrees: float) -> int:
    return round(axis["center_position"] + axis["direction"] * degrees * COUNTS_PER_DEGREE)


def to_degrees(axis: dict, position: int | None) -> float | None:
    if position is None:
        return None
    return round((position - axis["center_position"]) / COUNTS_PER_DEGREE * axis["direction"], 2)


def nearest_center(axis: dict, position: int) -> int:
    """Re-establish the local turn after reboot without commanding a full turn."""
    zero = axis["center_position"]
    return zero + ((position - zero + 2048) // 4096) * 4096


class ServoController:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        load_zeros(config)
        self.gain_settings = GainSettings(config)
        self.lock = threading.RLock()
        self.pose_lock = threading.Lock()
        self.cached_pose: dict | None = None
        self.pose_history: deque[dict] = deque(maxlen=256)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor, name="servo", daemon=True)
        self.port: Any = None
        self.packet: Any = None
        self.comm_success = 0
        self.read_retries = 0
        self._feedback_bases: dict[str, int] = {}
        self._saved_aliases: dict[str, list[int]] = {}
        self.axes = {name: {"online": False, "model": None, "position": None,
                            "goal": None, "torque": None, "origin": None,
                            "position_gains": None} for name in ("x", "y")}
        self._recovery_boundary: dict[str, int | None] = {name: None for name in self.axes}
        self.armed = False
        # Operator intent is independent of whether the hardware is connected.
        # Faults drop torque; only an explicit Stop clears this latch.
        self.run_requested = False
        self.armed_at = None
        self.error = "Checking X/Y servos…"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3)
        with self.lock:
            self.run_requested = False
            self._disable_all_locked()
            self._disconnect_locked()

    def _open_locked(self) -> None:
        if self.packet is not None:
            return
        from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

        port = PortHandler(self.config["device"])
        if not port.openPort() or not port.setBaudRate(self.config["baudrate"]):
            port.closePort()
            raise DeviceUnavailable("Cannot open servo controller")
        self.port = port
        self.packet = PacketHandler(XL330_PROTOCOL_VERSION)
        self.comm_success = COMM_SUCCESS

    def _check(self, result: int, error: int, name: str, operation: str) -> None:
        if result != self.comm_success or error:
            raise DeviceUnavailable(f"{name.upper()} (ID {self.config['axes'][name]['id']}): {operation} failed "
                                    f"(communication={result}, device_error={error})")

    def _ping(self, name: str) -> None:
        model, result, error = self.packet.ping(self.port, self.config["axes"][name]["id"])
        self._check(result, error, name, "PING; check wiring and unique IDs")
        if model != self.config["model_number"]:
            raise DeviceUnavailable(f"{name.upper()}: expected XL330-M288-T, got model {model}")
        self.axes[name]["model"] = model

    def _read(self, name: str, address: int, size: int) -> int:
        value, result, error = getattr(self.packet, f"read{size}ByteTxRx")(
            self.port, self.config["axes"][name]["id"], address)
        # A read is idempotent. Reissue the SAME register once after a lost or
        # corrupt response; the SDK clears RX before transmitting the request.
        # Never retry device faults or writes, and never reuse a failed value.
        if result in (-3001, -3002) and not error:
            self.read_retries += 1
            LOGGER.warning("%s register %d read communication=%d; retrying once", name, address, result)
            value, result, error = getattr(self.packet, f"read{size}ByteTxRx")(
                self.port, self.config["axes"][name]["id"], address)
        self._check(result, error, name, f"read register {address}")
        return value

    def _write(self, name: str, address: int, size: int, value: int) -> None:
        result, error = getattr(self.packet, f"write{size}ByteTxRx")(
            self.port, self.config["axes"][name]["id"], address, value)
        self._check(result, error, name, f"write register {address}")

    def _prepare_feedback_locked(self) -> None:
        """Configure RAM only, after verifying BOTH models and torque-off states."""
        if not self.config.get("packed_feedback", False) or self._feedback_bases:
            return
        if self.armed:
            raise DeviceUnavailable("Cannot configure feedback while armed")
        bases, saved = {}, {}
        for name in self.axes:
            self._ping(name)
            if self._read(name, XL330_TORQUE_ENABLE, 1) != 0:
                raise DeviceUnavailable(f"{name.upper()}: feedback setup requires torque off")
            firmware = self._read(name, 6, 1)
            if not 1 <= firmware <= 255 or self._read(name, 68, 1) != 2:
                raise DeviceUnavailable(f"{name.upper()}: unsupported feedback configuration")
            # XL330 firmware before V53 has 20 slots at 208, V53+ has 28 at 224.
            bases[name] = 224 if firmware >= 53 else 208
            saved[name] = [self._read(name, INDIRECT_ADDRESS + 2*i, 2)
                           for i in range(len(FEEDBACK_REGISTERS))]
        self._saved_aliases = saved  # Retain rollback even if a write fails midway.
        for name in self.axes:
            for i, address in enumerate(FEEDBACK_REGISTERS):
                self._write(name, INDIRECT_ADDRESS + 2*i, 2, address)
        self._verify_feedback_mapping_locked()
        self._feedback_bases = bases

    def _verify_feedback_mapping_locked(self) -> None:
        for name in self._saved_aliases:
            for i, address in enumerate(FEEDBACK_REGISTERS):
                if self._read(name, INDIRECT_ADDRESS + 2*i, 2) != address:
                    raise DeviceUnavailable(f"{name.upper()}: feedback mapping readback mismatch")

    def _restore_feedback_locked(self) -> list[str]:
        errors = []
        for name, values in self._saved_aliases.items():
            try:
                self._ping(name)
                if self._read(name, XL330_TORQUE_ENABLE, 1) != 0:
                    raise DeviceUnavailable(f"{name.upper()}: feedback restore requires torque off")
                for i, value in enumerate(values):
                    self._write(name, INDIRECT_ADDRESS + 2*i, 2, value)
                    if self._read(name, INDIRECT_ADDRESS + 2*i, 2) != value:
                        raise DeviceUnavailable(f"{name.upper()}: feedback restore readback mismatch")
            except Exception as error:
                errors.append(str(error))
        self._feedback_bases.clear()
        self._saved_aliases.clear()
        return errors

    def _read_feedback_locked(self, name: str) -> tuple[int, int, int]:
        """A new acknowledged packet per axis, never a cached validation.

        Unlike the SDK GroupSyncRead helper, readTxRx retains the device-error
        byte. A failed or short response never supplies position/history data.
        """
        address = self._feedback_bases[name]
        args = (self.port, self.config["axes"][name]["id"], address, len(FEEDBACK_REGISTERS))
        data, result, error = self.packet.readTxRx(*args)
        if result in (-3001, -3002) and not error:
            self.read_retries += 1
            LOGGER.warning("%s packed feedback communication=%d; retrying once", name, result)
            data, result, error = self.packet.readTxRx(*args)
        self._check(result, error, name, "read packed feedback")
        if (len(data) != len(FEEDBACK_REGISTERS)
                or any(type(v) is not int or not 0 <= v <= 255 for v in data)):
            raise DeviceUnavailable(f"{name.upper()}: malformed packed feedback")
        if data[6] != 2:
            raise DeviceUnavailable(f"{name.upper()}: feedback mapping reset or invalid")
        if data[4] not in (0, 1):
            raise DeviceUnavailable(f"{name.upper()}: invalid torque reading")
        if self.armed and data[7] != 50:
            raise DeviceUnavailable(f"{name.upper()}: unexpected bus watchdog state")
        return int.from_bytes(bytes(data[:4]), "little"), data[4], data[5]

    def _disable_all_locked(self) -> list[str]:
        """Never let a failed first motor prevent attempting to stop the second."""
        self.armed = False
        with self.pose_lock:
            self.cached_pose = None
            self.pose_history.clear()
        errors = []
        for name, state in self.axes.items():
            self._recovery_boundary[name] = None
            try:
                self._open_locked()
                self._ping(name)  # Never write this table to a different model.
                self._write(name, XL330_TORQUE_ENABLE, 1, 0)
                if self._read(name, XL330_TORQUE_ENABLE, 1) != 0:
                    raise DeviceUnavailable(f"{name.upper()}: torque-off not confirmed")
                state["torque"] = False
            except Exception as error:
                state["torque"] = None
                errors.append(str(error))
        return errors

    def _disconnect_locked(self) -> None:
        if self.packet is not None:
            errors = self._restore_feedback_locked()
            if errors:
                detail = "feedback restoration failed: " + "; ".join(errors)
                LOGGER.error(detail)
                self.error = f"{self.error}; {detail}" if self.error else detail
        if self.port is not None:
            try:
                self.port.closePort()
            except Exception:
                LOGGER.exception("closing servo bus")
        self.port = self.packet = None
        self.armed = False
        for state in self.axes.values():
            state["online"] = False
            state["position_gains"] = None
            state["position"] = state["goal"] = None

    def _fault_locked(self, error: Exception) -> None:
        errors = self._disable_all_locked()
        self.error = str(error)
        if errors:
            self.error += "; torque-off unconfirmed—disconnect motor power if necessary"
        self._disconnect_locked()

    def _poll_locked(self) -> None:
        sampled_at = time.monotonic()
        if not self.armed:
            with self.pose_lock:
                self.cached_pose = None
                self.pose_history.clear()
        if self.packet is None:
            errors = self._disable_all_locked()
            if errors:
                raise DeviceUnavailable("; ".join(errors))
        self._prepare_feedback_locked()
        for name, state in self.axes.items():
            retries_before = self.read_retries
            read_started = time.monotonic()
            packed = bool(self._feedback_bases)
            if packed:
                position, torque, hardware_error = self._read_feedback_locked(name)
            else:
                position = self._read(name, XL330_PRESENT_POSITION, 4)
            read_finished = time.monotonic()
            # Separate axes are read sequentially. The response midpoint is an
            # estimate, with the transaction duration retained as uncertainty.
            state["observed_at"] = (read_started + read_finished) / 2
            state["read_duration_ms"] = (read_finished - read_started) * 1000
            state["position_retried"] = self.read_retries != retries_before
            state["position"] = position if position < 2**31 else position - 2**32
            if not -1048575 <= state["position"] <= 1048575:
                raise DeviceUnavailable(f"{name.upper()}: invalid extended-position reading")
            if not self.armed:
                state["origin"] = nearest_center(self.config["axes"][name], state["position"])
            if not packed:
                torque = self._read(name, XL330_TORQUE_ENABLE, 1)
                hardware_error = self._read(name, 70, 1)
            state["torque"] = bool(torque)
            if hardware_error:
                raise DeviceUnavailable(f"{name.upper()}: servo hardware fault")
            if state["torque"] != self.armed:
                raise DeviceUnavailable(f"{name.upper()}: unexpected torque state")
            state["online"] = True
            if state["position_gains"] is None:
                self._read_gains_locked(name)
        completed_at = time.monotonic()
        if (self.armed and completed_at - sampled_at <= 0.1
                and not any(state["position_retried"] for state in self.axes.values())):
            with self.pose_lock:
                self.cached_pose = {"sampled_at": sampled_at, "read_completed_at": completed_at,
                    "axes": {name: {
                        "origin": state["origin"], "id": self.config["axes"][name]["id"],
                        "direction": self.config["axes"][name]["direction"],
                        "degrees": to_degrees(self._axis_config(name), state["position"]),
                        "goal_degrees": to_degrees(self._axis_config(name), state["goal"]),
                        "observed_at": state["observed_at"],
                        "read_duration_ms": state["read_duration_ms"],
                    } for name, state in self.axes.items()}}
                self.pose_history.append(self.cached_pose)

    def _axis_config(self, name: str) -> dict:
        return {**self.config["axes"][name], "center_position": self.axes[name]["origin"]}

    def _read_gains_locked(self, name: str) -> dict:
        gains = {key: self._read(name, address, 2) for key, address in POSITION_GAIN_REGISTERS.items()}
        if any(not 0 <= value <= 16383 for value in gains.values()):
            raise DeviceUnavailable(f"{name.upper()}: invalid position gain readback")
        self.axes[name]["position_gains"] = gains
        if gains["p"] > 0:
            self.gain_settings.baseline.setdefault(name, {key: gains[key] for key in ("p", "d")})
        return gains

    def set_gains(self, name: str, p: int, d: int) -> None:
        if not isinstance(name, str) or name not in self.axes:
            raise ValueError("axis must be x or y")
        requested = {"p": p, "d": d}
        validate_pd(requested)
        with self.lock:
            if not self.packet or not self.axes[name]["online"]:
                raise DeviceUnavailable("Servo offline; reconnect before changing gains")
            previous = None
            try:
                self._ping(name)
                previous = self._read_gains_locked(name)
                for key, value in requested.items():
                    if previous[key] != value:
                        self._write(name, POSITION_GAIN_REGISTERS[key], 2, value)
                actual = self._read_gains_locked(name)
                if actual != {**previous, **requested}:
                    raise DeviceUnavailable(f"{name.upper()}: gain readback mismatch")
                self.gain_settings.save(name, requested)
            except Exception as error:
                # Roll back a partial write or failed persistence. Never report
                # an unacknowledged value as applied, or overwrite the I gain.
                try:
                    if previous is not None:
                        for key in ("p", "d"):
                            self._write(name, POSITION_GAIN_REGISTERS[key], 2, previous[key])
                        if self._read_gains_locked(name) != previous:
                            raise DeviceUnavailable("Gain rollback readback mismatch")
                except Exception as rollback:
                    # Recovery must reapply the last observed P/D before torque
                    # returns, even on installations without configured gains.
                    if previous is not None:
                        self.gain_settings.values[name] = {key: previous[key] for key in ("p", "d")}
                    self._fault_locked(rollback)
                    raise DeviceUnavailable(f"Gain update failed; rollback unconfirmed: {rollback}") from error
                raise DeviceUnavailable(f"Gain update failed: {error}") from error

    def reset_gains(self, name: str) -> None:
        if not isinstance(name, str) or name not in self.axes:
            raise ValueError("axis must be x or y")
        with self.lock:
            baseline = self.gain_settings.baseline.get(name)
            if baseline is None:
                raise DeviceUnavailable("Servo gain baseline is not available yet")
            self.set_gains(name, **baseline)

    def _position_limits(self, name: str) -> tuple[int, int]:
        axis = self._axis_config(name)
        low, high = sorted(to_position(axis, deg) for deg in (axis["min_degrees"], axis["max_degrees"]))
        return low, high

    def _within_limits(self, name: str, position: int) -> bool:
        low, high = self._position_limits(name)
        return low <= position <= high and -1048575 <= position <= 1048575

    def _recover_limits_locked(self) -> None:
        if not self.armed:
            return
        for name, state in self.axes.items():
            low, high = self._position_limits(name)
            boundary = min(high, max(low, state["position"]))
            if boundary == state["position"]:
                self._recovery_boundary[name] = None
                continue
            # Correct once per excursion, not once per poll. A subsequent valid
            # slider command can move farther inward without being overwritten.
            if self._recovery_boundary[name] != boundary:
                if state["goal"] != boundary:
                    self._write(name, XL330_GOAL_POSITION, 4, boundary)
                    state["goal"] = boundary
                self._recovery_boundary[name] = boundary

    def _tick_locked(self) -> None:
        self._poll_locked()
        if self.run_requested and not self.armed and not self.stop_event.is_set():
            self.arm()  # Revalidate both devices and hold their current pose.
        self._recover_limits_locked()

    def _monitor(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                try:
                    self._tick_locked()
                    self.error = ""
                except Exception as error:
                    self._fault_locked(error)
            self.stop_event.wait((0.01 if self.armed else 0.2) if self.packet is not None else 2)

    def arm(self) -> None:
        with self.lock:
            if not self.config["calibrated"]:
                raise DeviceUnavailable("X/Y IDs and mechanical zero must be calibrated first")
            self.run_requested = True
            if self.armed:
                return
            try:
                self._poll_locked()
                self._verify_feedback_mapping_locked()
                # Extended mode avoids the long-way-around move at encoder rollover.
                # EEPROM is deliberately never changed by the running web service.
                for name, axis in self.config["axes"].items():
                    for address, size, expected in ((11, 1, 4), (10, 1, 0), (12, 1, 255), (20, 4, 0)):
                        if self._read(name, address, size) != expected:
                            raise DeviceUnavailable(f"{name.upper()}: register {address} needs commissioning")
                    if not self._within_limits(name, self.axes[name]["position"]):
                        raise DeviceUnavailable(f"{name.upper()}: outside calibrated range; reposition with torque off")
                # Prepare BOTH axes before enabling either. No move to zero on Start.
                for name, axis in self.config["axes"].items():
                    self._write(name, 98, 1, 0)  # Clear a previous bus watchdog timeout.
                    # Optional assembly-qualified RAM gains. Never alter the
                    # operating mode/EEPROM or assume both assemblies share gains.
                    gains = axis.get("position_gains")
                    if name in self.gain_settings.values:
                        gains = {**(gains or self._read_gains_locked(name)),
                                 **self.gain_settings.values[name]}
                    if gains is not None:
                        for key, address in POSITION_GAIN_REGISTERS.items():
                            self._write(name, address, 2, gains[key])
                            if self._read(name, address, 2) != gains[key]:
                                raise DeviceUnavailable(f"{name.upper()}: {key.upper()} gain readback mismatch")
                        self.axes[name]["position_gains"] = dict(gains)
                    self._write(name, 108, 4, axis["profile_acceleration"])
                    self._write(name, 112, 4, axis["profile_velocity"])
                    position = self.axes[name]["position"]
                    self._write(name, XL330_GOAL_POSITION, 4, position)
                    self.axes[name]["goal"] = position
                    self._write(name, 98, 1, 50)  # 1 s without bus traffic stops motion (holds torque).
                for name in self.axes:
                    self._write(name, XL330_TORQUE_ENABLE, 1, 1)
                    if self._read(name, XL330_TORQUE_ENABLE, 1) != 1:
                        raise DeviceUnavailable(f"{name.upper()}: Start not confirmed")
                    self.axes[name]["torque"] = True
                self.armed = True
                self.armed_at = time.monotonic()
                self.error = ""
            except Exception as error:
                self._fault_locked(error)
                raise DeviceUnavailable(self.error) from error

    def disable(self) -> None:
        with self.lock:
            # Stop must win even when the cable is absent and torque-off cannot
            # be acknowledged. Reconnection must not undo the operator's Stop.
            self.run_requested = False
            errors = self._disable_all_locked()
            if errors:
                self.error = "; ".join(errors) + "; torque-off unconfirmed"
                self._disconnect_locked()
                raise DeviceUnavailable(self.error)
            self.error = ""
            for state in self.axes.values():
                state["goal"] = None

    def recalibrate(self) -> None:
        """Save both current encoders as zero, torque-off and without motion."""
        with self.lock:
            if self.armed or self.run_requested:
                raise ServoDisarmed("Stop motors before recalibrating servo zeros")
            if not self.config["calibrated"]:
                raise DeviceUnavailable("Commission the servo IDs and modes before setting zeros")
            if not self.config.get("calibration_file"):
                raise DeviceUnavailable("Persistent servo calibration storage is not configured")
            try:
                # Reconnect read-only: do not let the monitor's normal initial
                # torque-off sequence conceal an unexpectedly enabled motor.
                self._open_locked()
                for name in self.axes:
                    self._ping(name)
                    if self._read(name, XL330_TORQUE_ENABLE, 1) != 0:
                        raise DeviceUnavailable(f"{name.upper()}: stop motors before setting zeros")
                self._poll_locked()  # Both readings must be fresh and torque-off.
                positions = {name: state["position"] for name, state in self.axes.items()}
                for name in self.axes:
                    self._ping(name)
                    for address, size, expected in ((11, 1, 4), (10, 1, 0), (12, 1, 255), (20, 4, 0)):
                        if self._read(name, address, size) != expected:
                            raise DeviceUnavailable(f"{name.upper()}: register {address} needs commissioning")
                time.sleep(0.05)
                self._poll_locked()
                if any(abs(self.axes[name]["position"] - value) > 2 for name, value in positions.items()):
                    raise DeviceUnavailable("Hold both axes still while setting servo zeros")
                positions = {name: state["position"] for name, state in self.axes.items()}
                # Commit both zeros together before changing either in memory.
                save_zeros(self.config, positions)
                for name, position in positions.items():
                    self.config["axes"][name]["center_position"] = position % 4096
                    self.axes[name].update(origin=position, goal=None)
                    self._recovery_boundary[name] = None
                self.error = ""
            except Exception as error:
                self._fault_locked(error)
                raise DeviceUnavailable(self.error) from error

    def keepalive(self) -> None:
        """Compatibility no-op for already-open clients; never arms or moves."""

    def move(self, name: str, degrees: float) -> None:
        if not isinstance(name, str) or name not in self.axes:
            raise ValueError("axis must be x or y")
        axis = self.config["axes"][name]
        if type(degrees) not in (int, float) or not math.isfinite(degrees):
            raise ValueError("degrees must be a finite number")
        if not axis["min_degrees"] <= degrees <= axis["max_degrees"]:
            raise ValueError(f"{name.upper()} must be between {axis['min_degrees']}° and {axis['max_degrees']}°")
        with self.lock:
            if not self.armed:
                raise ServoDisarmed("Motors stopped; press Start before moving")
            try:
                self._tick_locked()
                position = to_position(self._axis_config(name), degrees)
                self._write(name, XL330_GOAL_POSITION, 4, position)
                self.axes[name]["goal"] = position
            except Exception as error:
                self._fault_locked(error)
                raise DeviceUnavailable(self.error) from error

    def sample_pose(self, captured_at: float | None = None) -> dict[str, Any] | None:
        """Nonblocking checked encoder snapshot; never performs serial I/O.

        Called after inference, history normally brackets the frame receipt time
        on each axis. Interpolate that historical pose, never substitute the
        current angle. A fast worker without both brackets uses the older checked
        snapshot. History is invalidated on Stop, faults and zero changes.
        """
        with self.pose_lock:
            now = time.monotonic()
            at = now if captured_at is None else captured_at
            if (not self.armed or type(at) not in (int, float) or not math.isfinite(at)
                    or at > now or not self.pose_history
                    or not 0 <= now - self.pose_history[-1]["read_completed_at"] <= 0.1):
                return None
            if captured_at is not None:
                interpolated = interpolate_pose(self.pose_history, at, now)
                if interpolated is not None:
                    return interpolated
            for pose in reversed(self.pose_history):
                if pose["read_completed_at"] <= at and 0 <= at - pose["sampled_at"] <= 0.1:
                    return {**pose, "axes": {name: dict(axis) for name, axis in pose["axes"].items()}}
            return None

    def point(self, degrees: dict[str, float]) -> dict[str, Any]:
        """Command an absolute camera pointing pose, with only angle clamping."""
        return self.track(degrees, absolute=True)

    def track(self, offsets: dict[str, float], *, absolute: bool = False) -> dict[str, Any]:
        """Camera-framing goal correction while explicitly armed.

        With absolute=True, values are absolute degree goals (zero means zero).
        Otherwise, all-zero offsets hold the measured pose (clamped to limits).
        Otherwise integrate fresh-frame corrections into goals to overcome static
        position error. Only configured angular bounds clamp the requested goal;
        there is no step-size or encoder-to-goal lead cap.
        A zero correction on one axis preserves that axis's existing hold goal.
        Both axes are validated under the same bus lock before either is moved.
        """
        if set(offsets) != {"x", "y"} or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in offsets.values()
        ):
            raise ValueError("tracking offsets must contain finite x/y degree corrections")
        with self.lock:
            if not self.armed:
                raise ServoDisarmed("Motors stopped; press Start before tracking")
            try:
                self._tick_locked()
                goals, limited = {}, False
                hold = not absolute and not any(offsets.values())
                for name, offset in offsets.items():
                    axis, state = self._axis_config(name), self.axes[name]
                    if absolute:
                        requested = axis["center_position"] + axis["direction"] * offset * COUNTS_PER_DEGREE
                    elif hold:
                        requested = state["position"]
                    else:
                        requested = state["goal"] + axis["direction"] * offset * COUNTS_PER_DEGREE
                    low, high = self._position_limits(name)
                    # Clamp before rounding so even a huge finite correction
                    # cannot overflow conversion or produce an invalid goal.
                    goal = round(min(high, max(low, requested)))
                    if not -1048575 <= goal <= 1048575:
                        raise DeviceUnavailable(f"{name.upper()}: invalid tracking position")
                    limited |= requested < low or requested > high
                    goals[name] = goal
                for name, goal in goals.items():
                    if goal != self.axes[name]["goal"]:
                        self._write(name, XL330_GOAL_POSITION, 4, goal)
                        self.axes[name]["goal"] = goal
                return {"limited": limited, "goals": goals, "goal_degrees": {
                    name: to_degrees(self._axis_config(name), goal) for name, goal in goals.items()}}
            except Exception as error:
                self._fault_locked(error)
                raise DeviceUnavailable(self.error) from error

    def status(self) -> dict[str, Any]:
        with self.lock:
            online = all(state["online"] for state in self.axes.values())
            outside = [name.upper() for name, state in self.axes.items()
                       if state["position"] is not None and not self._within_limits(name, state["position"])]
            range_error = (f"{'/'.join(outside)} outside configured limits; reposition with torque off"
                           if outside and not self.armed else None)
            return {
                "online": online, "ready": online and self.config["calibrated"] and (self.armed or not outside),
                "can_recalibrate": bool(self.config.get("calibration_file")) and self.config["calibrated"]
                    and online and not self.armed and not self.run_requested
                    and all(state["torque"] is False for state in self.axes.values()),
                "device": self.config["device"], "protocol": "dynamixel-2.0",
                "baudrate": self.config["baudrate"], "armed": self.armed,
                "run_requested": self.run_requested, "armed_at": self.armed_at,
                "recovering": self.run_requested and not self.armed,
                "can_start": self.config["calibrated"],
                "read_retries": self.read_retries,
                "feedback_mode": "packed" if self._feedback_bases else "registers",
                "error": self.error or range_error or (None if self.config["calibrated"] else "X/Y calibration required"),
                "axes": {name: {**state, "id": self.config["axes"][name]["id"],
                    "position_gains": dict(state["position_gains"]) if state["position_gains"] else None,
                    "gain_baseline": dict(self.gain_settings.baseline[name])
                        if name in self.gain_settings.baseline else None,
                    "direction": self.config["axes"][name]["direction"],
                    "degrees": to_degrees(self._axis_config(name), state["position"]),
                    "goal_degrees": to_degrees(self._axis_config(name), state["goal"]),
                    "min_degrees": self.config["axes"][name]["min_degrees"],
                    "max_degrees": self.config["axes"][name]["max_degrees"]}
                    for name, state in self.axes.items()},
            }
