"""Latest-frame SAM worker orchestration. This module never controls the servo."""

from __future__ import annotations

import base64
import json
import math
import os
import selectors
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from spring_turret.prompts import COLORS, MAX_PROMPTS, normalize_prompts


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or not isinstance(
        config.get("enabled", False), bool
    ):
        raise ValueError("inference.enabled must be a boolean")
    if not config.get("enabled", False):
        return
    for key in ("python", "checkpoint", "cache_dir"):
        if not Path(config.get(key, "")).is_absolute():
            raise ValueError(f"inference.{key} must be an absolute path")
    if not 0 < float(config.get("confidence", 0.5)) < 1:
        raise ValueError("inference.confidence must be between zero and one")
    fps = config.get("max_fps", 0)
    if type(fps) not in (int, float) or not math.isfinite(fps) or not (fps == 0 or 0.1 <= fps <= 30):
        raise ValueError("inference.max_fps must be 0 (uncapped) or between 0.1 and 30")
    if config.get("precision", "float16") not in ("float16", "bfloat16"):
        raise ValueError("inference.precision must be float16 or bfloat16")
    if isinstance(config.get("device", 0), bool) or int(config.get("device", 0)) < 0:
        raise ValueError("inference.device must be a nonnegative integer")


class WorkerClient:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.process: subprocess.Popen | None = None
        self.pending = b""

    def launch(self) -> None:
        cache = Path(self.config["cache_dir"])
        cache.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "4",
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "XDG_CACHE_HOME": str(cache),
        }
        self.process = subprocess.Popen(
            [
                self.config["python"],
                "-u",
                str(Path(__file__).with_name("sam31_worker.py")),
                "--checkpoint",
                self.config["checkpoint"],
                "--device",
                str(self.config.get("device", 0)),
                "--precision",
                self.config.get("precision", "float16"),
                "--confidence",
                str(self.config.get("confidence", 0.5)),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )

    def receive(self, timeout: float, progress: Any = None) -> dict[str, Any]:
        assert self.process and self.process.stdout
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                if b"\n" in self.pending:
                    line, self.pending = self.pending.split(b"\n", 1)
                    message = json.loads(line)
                    if message.get("type") == "progress":
                        if progress:
                            progress(message["stage"])
                        continue
                    if message.get("type") in ("fatal", "error"):
                        raise RuntimeError(message.get("error", "SAM worker failed"))
                    return message
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError(
                        "SAM worker timed out; submit the prompt to retry"
                    )
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("SAM worker exited; check the service log")
                self.pending += chunk
                if len(self.pending) > 4 * 1024 * 1024:
                    raise RuntimeError("SAM worker response exceeded 4 MiB")

    def detect(
        self, request_id: int, jpeg: bytes, prompts: list[str], progress: Any
    ) -> dict[str, Any]:
        assert self.process and self.process.stdin
        body = {
            "id": request_id,
            "jpeg": base64.b64encode(jpeg).decode(),
            "prompts": prompts,
        }
        self.process.stdin.write(json.dumps(body).encode() + b"\n")
        self.process.stdin.flush()
        result = self.receive(900, progress)
        if result.get("type") != "result" or result.get("id") != request_id:
            raise RuntimeError("SAM worker returned an unexpected response")
        return result

    def stop(self) -> None:
        process = self.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()


class DetectionController:
    def __init__(
        self, config: dict[str, Any], camera: Any, worker_factory: Any = WorkerClient
    ):
        self.config, self.camera, self.worker_factory = config, camera, worker_factory
        self.enabled = config.get("enabled", False)
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="detection", daemon=True)
        self.worker: Any = None
        self.prompts: list[str] = []
        self.revision = 0
        self.sequence = 0
        self.state = "idle" if self.enabled else "disabled"
        self.error: str | None = None
        self.result: dict[str, Any] = {}
        self.completed_at = 0.0
        self.frames: OrderedDict[str, bytes] = OrderedDict()

    def start(self) -> None:
        if self.enabled:
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
            worker = self.worker
        if worker:
            worker.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=5)

    def set_prompt(self, prompt: Any) -> None:
        """Compatibility for the original single-category endpoint."""
        self.set_prompts([prompt])

    def set_prompts(self, prompts: Any) -> None:
        prompts = normalize_prompts(prompts)
        if not self.enabled:
            raise ValueError("SAM inference is not configured on this device")
        with self.condition:
            self.prompts = prompts
            self.revision += 1
            self.result = {}
            self.frames.clear()
            self.completed_at = 0
            self.error = None
            self.state = "loading" if self.prompts else "idle"
            self.condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self.condition:
            age = (
                round((time.monotonic() - self.completed_at) * 1000)
                if self.completed_at
                else None
            )
            return {
                "enabled": self.enabled,
                "prompt": self.prompts[0] if len(self.prompts) == 1 else "",
                "prompts": list(self.prompts),
                "max_prompts": MAX_PROMPTS,
                "colors": COLORS,
                "revision": self.revision,
                "state": self.state,
                "error": self.error,
                "frame_age_ms": age,
                **self.result,
            }

    def frame(self, key: str) -> bytes | None:
        with self.condition:
            return self.frames.get(key)

    def frame_version(self) -> tuple[int, int | None]:
        with self.condition:
            return self.revision, self.result.get("frame_sequence")

    def wait_for_update(self, previous: tuple[int, int | None], timeout: float) -> None:
        with self.condition:
            self.condition.wait_for(
                lambda: self.stop_event.is_set() or self.frame_version() != previous,
                timeout=timeout,
            )

    def _sampling_delay(self, elapsed: float) -> float:
        fps = float(self.config.get("max_fps", 0))
        return max(0, 1 / fps - elapsed) if fps else 0.0

    def _progress(self, revision: int, stage: str) -> None:
        with self.condition:
            if revision == self.revision and self.prompts:
                self.state = stage

    def _run(self) -> None:
        failed_revision = -1
        last_camera_sequence = 0
        try:
            while not self.stop_event.is_set():
                with self.condition:
                    self.condition.wait_for(
                        lambda: (
                            self.stop_event.is_set()
                            or (self.prompts and self.revision != failed_revision)
                        )
                    )
                    if self.stop_event.is_set():
                        break
                    prompts, revision = list(self.prompts), self.revision
                sequence, jpeg = self.camera.wait_for_frame(last_camera_sequence, 1)
                if (
                    jpeg is None
                    or sequence == last_camera_sequence
                    or not self.camera.status()["online"]
                ):
                    self._progress(revision, "waiting_for_camera")
                    self.stop_event.wait(0.1)
                    continue
                last_camera_sequence = sequence
                started = time.monotonic()
                captured_at = (
                    started - self.camera.status().get("frame_age_ms", 0) / 1000
                )
                try:
                    if self.worker is None:
                        worker = self.worker_factory(self.config)
                        with self.condition:
                            self.worker = worker
                        worker.launch()
                        if self.stop_event.is_set():
                            break
                        if worker.receive(120).get("type") != "ready":
                            raise RuntimeError("SAM worker did not become ready")
                    self.sequence += 1
                    result = self.worker.detect(
                        self.sequence,
                        jpeg,
                        prompts,
                        lambda stage: self._progress(revision, stage),
                    )
                    if not result.get("torch_compile") or not result.get("sycl_graph"):
                        raise RuntimeError(
                            "SAM worker did not verify compiled SYCL graph execution"
                        )
                    annotated = base64.b64decode(result.pop("jpeg"), validate=True)
                    result.pop("type", None)
                    result.pop("id", None)
                    key = f"{revision}-{self.sequence}"
                    with self.condition:
                        if revision != self.revision:
                            continue  # Never display boxes from an obsolete prompt.
                        self.frames[key] = annotated
                        while len(self.frames) > 3:
                            self.frames.popitem(last=False)
                        self.result = {
                            **result,
                            "frame_sequence": self.sequence,
                            "frame_url": f"/api/detection/frame/{key}.jpg",
                        }
                        self.completed_at = captured_at
                        self.state = "running"
                        self.error = None
                        self.condition.notify_all()
                except Exception as error:
                    with self.condition:
                        if revision == self.revision:
                            self.state, self.error = "error", str(error)
                            self.result = {}
                            self.frames.clear()
                        worker, self.worker = self.worker, None
                    if worker:
                        worker.stop()
                    failed_revision = revision
                delay = self._sampling_delay(time.monotonic() - started)
                if delay:
                    self.stop_event.wait(delay)
        finally:
            if self.worker:
                self.worker.stop()
