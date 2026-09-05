"""Bounded two-axis XL330 controller; one serial port and one bus lock."""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("spring-turret.servo")
XL330_PROTOCOL_VERSION = 2.0
XL330_TORQUE_ENABLE = 64
XL330_GOAL_POSITION = 116
XL330_PRESENT_POSITION = 132
COUNTS_PER_DEGREE = 4096 / 360
CONTROL_TIMEOUT = 3.0


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
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor, name="servo", daemon=True)
        self.port: Any = None
        self.packet: Any = None
        self.comm_success = 0
        self.axes = {name: {"online": False, "model": None, "position": None,
                            "goal": None, "torque": None, "origin": None} for name in ("x", "y")}
        self._recovery_boundary: dict[str, int | None] = {name: None for name in self.axes}
        self.armed = False
        self.last_keepalive = 0.0
        self.error = "Checking X/Y servos…"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3)
        with self.lock:
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
            raise DeviceUnavailable(f"{name.upper()} (ID {self.config['axes'][name]['id']}): {operation} failed")

    def _ping(self, name: str) -> None:
        model, result, error = self.packet.ping(self.port, self.config["axes"][name]["id"])
        self._check(result, error, name, "PING; check wiring and unique IDs")
        if model != self.config["model_number"]:
            raise DeviceUnavailable(f"{name.upper()}: expected XL330-M288-T, got model {model}")
        self.axes[name]["model"] = model

    def _read(self, name: str, address: int, size: int) -> int:
        value, result, error = getattr(self.packet, f"read{size}ByteTxRx")(
            self.port, self.config["axes"][name]["id"], address)
        self._check(result, error, name, f"read register {address}")
        return value

    def _write(self, name: str, address: int, size: int, value: int) -> None:
        result, error = getattr(self.packet, f"write{size}ByteTxRx")(
            self.port, self.config["axes"][name]["id"], address, value)
        self._check(result, error, name, f"write register {address}")

    def _disable_all_locked(self) -> list[str]:
        """Never let a failed first motor prevent attempting to stop the second."""
        self.armed = False
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
        if self.port is not None:
            try:
                self.port.closePort()
            except Exception:
                LOGGER.exception("closing servo bus")
        self.port = self.packet = None
        self.armed = False
        for state in self.axes.values():
            state["online"] = False
            state["position"] = state["goal"] = None

    def _fault_locked(self, error: Exception) -> None:
        errors = self._disable_all_locked()
        self.error = str(error)
        if errors:
            self.error += "; torque-off unconfirmed—disconnect motor power if necessary"
        self._disconnect_locked()

    def _poll_locked(self) -> None:
        if self.packet is None:
            errors = self._disable_all_locked()
            if errors:
                raise DeviceUnavailable("; ".join(errors))
        for name, state in self.axes.items():
            position = self._read(name, XL330_PRESENT_POSITION, 4)
            state["position"] = position if position < 2**31 else position - 2**32
            if not -1048575 <= state["position"] <= 1048575:
                raise DeviceUnavailable(f"{name.upper()}: invalid extended-position reading")
            if not self.armed:
                state["origin"] = nearest_center(self.config["axes"][name], state["position"])
            state["torque"] = bool(self._read(name, XL330_TORQUE_ENABLE, 1))
            if self._read(name, 70, 1):
                raise DeviceUnavailable(f"{name.upper()}: servo hardware fault")
            if state["torque"] != self.armed:
                raise DeviceUnavailable(f"{name.upper()}: unexpected torque state")
            state["online"] = True

    def _axis_config(self, name: str) -> dict:
        return {**self.config["axes"][name], "center_position": self.axes[name]["origin"]}

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
        if self.armed and time.monotonic() - self.last_keepalive > CONTROL_TIMEOUT:
            raise DeviceUnavailable("Controls disconnected; motors stopped")
        # Validate BOTH axes and the control lease before any corrective motion.
        self._recover_limits_locked()

    def _monitor(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                try:
                    self._tick_locked()
                    self.error = ""
                except Exception as error:
                    self._fault_locked(error)
            self.stop_event.wait(0.2 if self.packet is not None else 2)

    def arm(self) -> None:
        with self.lock:
            if self.armed:
                return
            try:
                if not self.config["calibrated"]:
                    raise DeviceUnavailable("X/Y IDs and mechanical zero must be calibrated first")
                self._poll_locked()
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
                self.last_keepalive = time.monotonic()
                self.armed = True
                self.error = ""
            except Exception as error:
                self._fault_locked(error)
                raise DeviceUnavailable(self.error) from error

    def disable(self) -> None:
        with self.lock:
            errors = self._disable_all_locked()
            if errors:
                self.error = "; ".join(errors) + "; torque-off unconfirmed"
                self._disconnect_locked()
                raise DeviceUnavailable(self.error)
            self.error = ""
            for state in self.axes.values():
                state["goal"] = None

    def keepalive(self) -> None:
        with self.lock:
            if self.armed:
                self.last_keepalive = time.monotonic()

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
                self.last_keepalive = time.monotonic()
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
                "device": self.config["device"], "protocol": "dynamixel-2.0",
                "baudrate": self.config["baudrate"], "armed": self.armed,
                "error": self.error or range_error or (None if self.config["calibrated"] else "X/Y calibration required"),
                "axes": {name: {**state, "id": self.config["axes"][name]["id"],
                    "degrees": to_degrees(self._axis_config(name), state["position"]),
                    "goal_degrees": to_degrees(self._axis_config(name), state["goal"]),
                    "min_degrees": self.config["axes"][name]["min_degrees"],
                    "max_degrees": self.config["axes"][name]["max_degrees"]}
                    for name, state in self.axes.items()},
            }
