"""Latest-frame model worker orchestration. This module never controls the servo."""

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

from spring_turret.prompts import COLORS, MAX_PROMPTS
from spring_turret.instances import InstanceAssociator
from spring_turret.models import MODELS, model_prompts


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
    if config.get("model", "sam3.1") not in MODELS:
        raise ValueError("inference.model must be sam3.1 or yolo26x")
    if "yolo26x_checkpoint" in config and not Path(config["yolo26x_checkpoint"]).is_absolute():
        raise ValueError("inference.yolo26x_checkpoint must be an absolute path")
    if config.get("model") == "yolo26x" and not config.get("yolo26x_checkpoint"):
        raise ValueError("inference.yolo26x_checkpoint is required for YOLO26x")
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
        self.cancelled = threading.Event()

    def launch(self) -> None:
        cache = Path(self.config["cache_dir"])
        if self.config.get("model") == "yolo26x":
            cache /= "yolo26x"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "ultralytics").mkdir(exist_ok=True)
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "4",
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "XDG_CACHE_HOME": str(cache),
            "YOLO_CONFIG_DIR": str(cache / "ultralytics"),
            "YOLO_AUTOINSTALL": "false",
            "YOLO_OFFLINE": "true",
        }
        self.process = subprocess.Popen(
            [
                self.config["python"],
                "-u",
                str(Path(__file__).with_name(MODELS[self.config.get("model", "sam3.1")]["worker"])),
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
        if self.cancelled.is_set():
            self.cancel()

    def cancel(self) -> None:
        """Interrupt cold compilation without blocking the HTTP/tracking locks."""
        self.cancelled.set()
        process = self.process
        if process and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass

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
                        raise RuntimeError(message.get("error", "Model worker failed"))
                    return message
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError(
                        "Model worker timed out; submit the prompt to retry"
                    )
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("Model worker exited; check the service log")
                self.pending += chunk
                if len(self.pending) > 4 * 1024 * 1024:
                    raise RuntimeError("Model worker response exceeded 4 MiB")

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
            raise RuntimeError("Model worker returned an unexpected response")
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
        self, config: dict[str, Any], camera: Any, worker_factory: Any = WorkerClient,
        pose_provider: Any = None, tracking_config: dict | None = None,
    ):
        self.config, self.camera, self.worker_factory = config, camera, worker_factory
        self.pose_provider = pose_provider
        self.instances = InstanceAssociator(tracking_config or {})
        self.frame_selections: OrderedDict[str, dict] = OrderedDict()
        self.enabled = config.get("enabled", False)
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="detection", daemon=True)
        self.worker: Any = None
        self.prompts: list[str] = []
        self.model = config.get("model", "sam3.1")
        self.saved_prompts = {"sam3.1": [], "yolo26x": ["person"]}
        self.worker_model: str | None = None
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
        if not self.enabled:
            raise ValueError("Inference is not configured on this device")
        with self.condition:
            prompts = model_prompts(self.model, prompts)
            self.prompts = prompts
            self.saved_prompts[self.model] = list(prompts)
            self.revision += 1
            self.result = {}
            self.frames.clear()
            self.frame_selections.clear()
            self.instances.clear()
            self.completed_at = 0
            self.error = None
            self.state = "loading" if self.prompts else "idle"
            self.condition.notify_all()

    def set_model(self, model: Any) -> None:
        if type(model) is not str or model not in MODELS:
            raise ValueError("model must be sam3.1 or yolo26x")
        if not self.enabled or (model == "yolo26x" and not self.config.get("yolo26x_checkpoint")):
            raise ValueError(f"{MODELS[model]['label']} is not configured on this device")
        with self.condition:
            if model == self.model:
                return
            self.model = model
            self.set_prompts(self.saved_prompts[model])
            # Interrupt old cold compilation too. Only the worker thread reaps
            # and replaces the process; no overlapping GPU model lifetimes.
            if self.worker is not None and self.worker_model != model:
                self.worker.cancel()
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
                "model": self.model,
                "models": [{"id": key, "label": value["label"], "available": self.enabled and
                            (key == "sam3.1" or bool(self.config.get("yolo26x_checkpoint")))}
                           for key, value in MODELS.items()],
                "classes": MODELS[self.model]["classes"],
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

    def selection(self, revision: int, sequence: int, instance_id: int) -> dict:
        if any(type(v) is not int or v < 0 for v in (revision, sequence, instance_id)):
            raise ValueError("selection requires integer revision, frame_sequence and instance_id")
        with self.condition:
            frame = self.frame_selections.get(f"{revision}-{sequence}")
            if (revision != self.revision or self.state != "running" or not frame or
                    not 0 <= time.monotonic() - frame["captured_at"] <= .75):
                raise ValueError("That camera frame is stale; click a box in a fresh frame")
            for box in frame["boxes"]:
                if box["instance_id"] == instance_id:
                    return dict(box)
            raise ValueError("That object is not in the displayed frame")

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
                            or (self.worker is not None and self.worker_model != self.model)
                            or (self.prompts and self.revision != failed_revision)
                        )
                    )
                    if self.stop_event.is_set():
                        break
                    prompts, revision, model = list(self.prompts), self.revision, self.model
                if self.worker is not None and self.worker_model != model:
                    self.worker.stop()  # Exit/reap the old process before allocating the new model.
                    with self.condition:
                        self.worker, self.worker_model = None, None
                if not prompts:
                    continue
                pose = self.pose_provider() if self.pose_provider else None
                sequence, jpeg, captured_at = self.camera.wait_for_sample(
                    last_camera_sequence, 1, after=pose["sampled_at"] if pose else 0.0
                )
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
                # This is a receipt-time pose estimate, not a hardware exposure
                # timestamp. Never associate a delayed camera frame with an old pose.
                if pose and not 0 <= captured_at - pose["sampled_at"] <= 0.1:
                    pose = None
                try:
                    if self.worker is None:
                        worker_config = {**self.config, "model": model}
                        if model == "yolo26x":
                            worker_config["checkpoint"] = self.config["yolo26x_checkpoint"]
                        worker = self.worker_factory(worker_config)
                        with self.condition:
                            self.worker = worker
                            self.worker_model = model
                        worker.launch()
                        if self.stop_event.is_set():
                            break
                        if worker.receive(120).get("type") != "ready":
                            raise RuntimeError("Model worker did not become ready")
                    with self.condition:
                        if revision != self.revision:
                            continue
                    self.sequence += 1
                    result = self.worker.detect(
                        self.sequence,
                        jpeg,
                        prompts,
                        lambda stage: self._progress(revision, stage),
                    )
                    if not result.get("torch_compile") or not result.get("sycl_graph"):
                        raise RuntimeError(
                            "Model worker did not verify compiled SYCL graph execution"
                        )
                    annotated = base64.b64decode(result.pop("jpeg"), validate=True)
                    result.pop("type", None)
                    result.pop("id", None)
                    key = f"{revision}-{self.sequence}"
                    with self.condition:
                        if revision != self.revision:
                            continue  # Never display boxes from an obsolete prompt.
                        self.frames[key] = annotated
                        result["boxes"] = self.instances.update(result.get("boxes", []), pose, captured_at)
                        self.frame_selections[key] = {"boxes": result["boxes"], "captured_at": captured_at}
                        while len(self.frames) > 8:
                            removed, _ = self.frames.popitem(last=False)
                            self.frame_selections.pop(removed, None)
                        self.result = {
                            **result,
                            "frame_sequence": self.sequence,
                            "frame_url": f"/api/detection/frame/{key}.jpg",
                            "captured_at": captured_at,
                            "frame_pose": pose,
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
                            self.frame_selections.clear()
                            self.instances.clear()
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
