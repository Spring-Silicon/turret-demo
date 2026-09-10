"""Latest-frame model worker orchestration. This module never controls the servo."""

from __future__ import annotations

import base64
import json
import math
import os
import selectors
import signal
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from spring_turret.prompts import COLORS, MAX_PROMPTS
from spring_turret.instances import InstanceAssociator
from spring_turret.models import (MODELS, PROTEUS_MODELS, model_available, model_prompts,
                                 is_tracking_model, is_session_model, max_prompts)
from spring_turret.worker_protocol import JPEG_BYTES, encode_request
from spring_turret.tracking_masks import valid_mask_centroid
from spring_turret.api_contract import API_VERSION, detection_result
from spring_turret.hardware import inference_runtime


def validate_tracking_result(result, prompts, *, temporal=True, backend="sam31-object-multiplex"):
    """The temporal mode has a distinct, honest contract from compiled detectors."""
    def valid_id(value):
        return type(value) is int and 0 <= value < 2**53
    if (result.get("temporal_tracking") is not temporal
            or result.get("tracking_backend") != backend
            or type(result.get("tracking_frame")) is not int or result["tracking_frame"] < 1
            or type(result.get("memory_frames")) is not int or result["memory_frames"] < 0):
        raise RuntimeError("Worker did not verify SAM temporal tracking")
    if not temporal and result["memory_frames"] != 0:
        raise RuntimeError("No-memory worker reported temporal memories")
    active = result.get("active_instance_ids")
    if not isinstance(active, list) or not all(map(valid_id, active)) or len(set(active)) != len(active):
        raise RuntimeError("Tracker returned invalid active object IDs")
    retired = result.get("retired_instance_ids", [])
    if (not isinstance(retired, list) or not all(map(valid_id, retired))
            or len(set(retired)) != len(retired) or set(retired).intersection(active)):
        raise RuntimeError("Tracker returned invalid retired object IDs")
    dropped = result.get("dropped_objects", 0)
    if type(dropped) is not int or dropped < 0:
        raise RuntimeError("Tracker returned invalid capacity diagnostics")
    boxes, seen = result.get("boxes"), set()
    if not isinstance(boxes, list):
        raise RuntimeError("Tracker returned invalid boxes")
    for box in boxes:
        if not isinstance(box, dict):
            raise RuntimeError("Tracker returned invalid box")
        instance_id, coords, score = box.get("instance_id"), box.get("xyxy"), box.get("score")
        if (not valid_id(instance_id) or instance_id not in active or instance_id in seen
                or box.get("prompt") not in prompts
                or not isinstance(coords, list) or len(coords) != 4
                or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in coords)
                or not coords[0] < coords[2] or not coords[1] < coords[3]
                or type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1):
            raise RuntimeError("Tracker returned invalid object geometry or identity")
        seen.add(instance_id)
        if box.get("mask_centroid") is not None and not valid_mask_centroid(box["mask_centroid"]):
            raise RuntimeError("Tracker returned an invalid mask centroid")


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
        raise ValueError("inference.model must be one of " + ", ".join(MODELS))
    for key in ("proteus_bundle", "proteus_memory_bundle"):
        if key not in config:
            continue
        if (not isinstance(config[key], str)
                or not Path(config[key]).is_absolute()
                or config.get("device_type", "xpu") != "xpu"):
            raise ValueError(f"inference.{key} requires an absolute path and XPU")
    if config.get("model") in (*PROTEUS_MODELS, "sam3.1-nomem") and not model_available(config["model"], config):
        raise ValueError("The selected experimental model is not configured for this device")
    if "sam31_mask_bundle" in config:
        path = config["sam31_mask_bundle"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("inference.sam31_mask_bundle must be an absolute path")
        if config.get("device_type", "xpu") != "xpu":
            raise ValueError("Native mask bundle requires XPU")
    if "sam31_mask_compiled_bundle" in config:
        path = config["sam31_mask_compiled_bundle"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("inference.sam31_mask_compiled_bundle must be an absolute path")
        if config.get("device_type", "xpu") != "xpu" or not config.get("sam31_mask_bundle"):
            raise ValueError("Compiled mask artifacts require XPU and its native mask bundle")
    if config.get("model") == "sam3.1-mask" and not model_available("sam3.1-mask", config):
        raise ValueError("SAM3.1 mask requires a mask bundle on XPU")
    if "sam31_tracking_bundle" in config and (not isinstance(config["sam31_tracking_bundle"], str)
            or not Path(config["sam31_tracking_bundle"]).is_absolute()):
        raise ValueError("inference.sam31_tracking_bundle must be an absolute path")
    if "sam31_tracking_native_bundle" in config:
        path = config["sam31_tracking_native_bundle"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("inference.sam31_tracking_native_bundle must be an absolute path")
        if config.get("device_type", "xpu") != "xpu" or not config.get("sam31_tracking_bundle"):
            raise ValueError("Native tracking requires XPU and a tracking source bundle")
    if config.get("model") == "sam3.1-tracking" and not model_available("sam3.1-tracking", config):
        raise ValueError("SAM 3.1 Tracking requires a tracking source bundle and XPU or CUDA")
    if "sam31_tracking_v18_bundle" in config:
        path = config["sam31_tracking_v18_bundle"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("inference.sam31_tracking_v18_bundle must be an absolute path")
        if config.get("device_type", "xpu") != "xpu" or not config.get("sam31_tracking_bundle"):
            raise ValueError("SAM 3.1 v18 requires XPU and a tracking source bundle")
    if config.get("model") == "sam3.1-v18" and not model_available("sam3.1-v18", config):
        raise ValueError("SAM 3.1 v18 requires its pinned native tracking bundle")
    if "yolo26x_checkpoint" in config:
        raise ValueError("YOLO support was removed; remove inference.yolo26x_checkpoint")
    if not 0 < float(config.get("confidence", 0.5)) < 1:
        raise ValueError("inference.confidence must be between zero and one")
    fps = config.get("max_fps", 0)
    if type(fps) not in (int, float) or not math.isfinite(fps) or not (fps == 0 or 0.1 <= fps <= 30):
        raise ValueError("inference.max_fps must be 0 (uncapped) or between 0.1 and 30")
    if config.get("precision", "float16") not in ("float16", "bfloat16"):
        raise ValueError("inference.precision must be float16 or bfloat16")
    if config.get("device_type", "xpu") not in ("xpu", "cuda"):
        raise ValueError("inference.device_type must be xpu or cuda")
    if config.get("device_type") == "cuda":
        if config.get("sam31_native_bundle") or config.get("sam31_w8a8_development_bundle"):
            raise ValueError("Intel native/W8A8 bundles require XPU")
    if type(config.get("sam31_cpu_prefetch", True)) is not bool:
        raise ValueError("inference.sam31_cpu_prefetch must be a boolean")
    if sum(key in config for key in ("sam31_native_bundle", "sam31_w8a8_development_bundle", "sam31_w4a4_bundle")) > 1:
        raise ValueError("Select only one SAM image bundle")
    tradeoff = config.get("sam31_allow_w4a4_accuracy_tradeoff", False)
    if type(tradeoff) is not bool or bool(config.get("sam31_w4a4_bundle")) != tradeoff:
        raise ValueError("W4A4 requires an explicit bundle and accuracy-tradeoff opt-in")
    if config.get("sam31_w4a4_bundle") and config.get("device_type", "xpu") != "xpu":
        raise ValueError("W4A4 requires Intel XPU")
    if type(config.get("sam31_allow_unqualified_w8a8", False)) is not bool:
        raise ValueError("inference.sam31_allow_unqualified_w8a8 must be a boolean")
    if config.get("sam31_allow_unqualified_w8a8") and not config.get("sam31_w8a8_development_bundle"):
        raise ValueError("Unqualified execution requires a W8A8 development bundle")
    for key in ("sam31_native_bundle", "sam31_w8a8_development_bundle", "sam31_w4a4_bundle"):
        if key in config:
            if not isinstance(config[key], str) or not Path(config[key]).is_absolute():
                raise ValueError(f"inference.{key} must be an absolute path")
            if config.get("precision", "float16") != "float16":
                raise ValueError("native SAM requires inference.precision=float16")
    if isinstance(config.get("device", 0), bool) or int(config.get("device", 0)) < 0:
        raise ValueError("inference.device must be a nonnegative integer")


class WorkerClient:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.process: subprocess.Popen | None = None
        self.pending = b""
        self.cancelled = threading.Event()
        self.cancel_requested_at: float | None = None
        self.native_temp: Any = None
        self.request_transport = "json-base64"
        self.prefetch_supported = False
        self.previous_prompts = None

    def launch(self) -> None:
        if self.config.get("model", "sam3.1") not in MODELS:
            raise ValueError("Unsupported inference model")
        spec = inference_runtime(self.config).prepare()
        self.native_temp = spec.temporary
        self.process = subprocess.Popen(
            spec.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env=spec.environment, start_new_session=True)
        if self.cancelled.is_set():
            self.cancel()

    def cancel(self) -> None:
        """Request a bounded drain without signalling an in-flight GPU kernel."""
        if self.cancel_requested_at is None:
            self.cancel_requested_at = time.monotonic()
        self.cancelled.set()

    def receive(self, timeout: float, progress: Any = None, tick: Any = None) -> dict[str, Any]:
        assert self.process and self.process.stdout
        deadline = time.monotonic() + timeout
        # Mask CPU preparation takes about 6 ms on Arc. A 5 ms camera check can
        # delay most of that work until inference has already finished. This
        # bounded check only submits new frames; it never substitutes an older
        # prepared frame or changes the camera/inference scheduling policy.
        poll_seconds = .001 if self.config.get("model") == "sam3.1-mask" else .005
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                if (self.cancel_requested_at is not None
                        and time.monotonic() - self.cancel_requested_at >= 2):
                    raise RuntimeError("Model worker cancelled after drain timeout")
                if b"\n" in self.pending:
                    line, self.pending = self.pending.split(b"\n", 1)
                    message = json.loads(line)
                    if message.get("type") == "progress":
                        if progress:
                            progress(message["stage"])
                        continue
                    if message.get("type") in ("fatal", "error"):
                        raise RuntimeError(message.get("error", "Model worker failed"))
                    if message.get("type") == "ready":
                        self.request_transport = (JPEG_BYTES if message.get("request_transport") == JPEG_BYTES
                                                  else "json-base64")
                        self.prefetch_supported = (self.request_transport == JPEG_BYTES
                            and message.get("cpu_prefetch") is True and self.config.get("sam31_cpu_prefetch", True))
                    return message
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Model worker timed out; submit the prompt to retry"
                    )
                if not selector.select(min(remaining, poll_seconds if tick else .1)):
                    if tick:
                        tick()
                    continue
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("Model worker exited; check the service log")
                self.pending += chunk
                if len(self.pending) > 4 * 1024 * 1024:
                    raise RuntimeError("Model worker response exceeded 4 MiB")

    def detect(
        self, request_id: int, jpeg: bytes, prompts: list[str], progress: Any,
        *, prepared_token=None, prepare_next=None, session_revision=None, captured_at=None, camera_identity=None,
    ) -> dict[str, Any]:
        assert self.process and self.process.stdin
        body = {
            "id": request_id,
            "prompts": prompts,
            "client_overlay": True,
            "prepared_token": prepared_token,
        }
        if is_session_model(self.config.get("model")):
            body.update(session_revision=session_revision, captured_at=captured_at, camera_identity=camera_identity)
        self.process.stdin.write(encode_request(body, jpeg, self.request_transport))
        self.process.stdin.flush()
        sent = 0
        last_token = None
        prepared_sent_at = None
        def prefetch():
            nonlocal sent, prepared_sent_at, last_token
            if self.cancelled.is_set():
                return
            candidate = prepare_next()
            if candidate is not None and candidate["token"] != last_token:
                self.process.stdin.write(encode_request({"kind":"prepare", "token":candidate["token"]},
                                                         candidate["jpeg"], self.request_transport))
                self.process.stdin.flush()
                prepared_sent_at = time.monotonic()
                last_token = candidate["token"]
                sent += 1
        # Only steady, validated prompt batches: no background Torch operations
        # during initial compilation/capture or a changed prompt batch's setup.
        tick = (prefetch if self.prefetch_supported and prepare_next is not None
                and self.previous_prompts == tuple(prompts) else None)
        result = self.receive(900, progress, tick)
        if result.get("type") != "result" or result.get("id") != request_id:
            raise RuntimeError("Model worker returned an unexpected response")
        self.previous_prompts = tuple(prompts)
        result["request_transport"] = self.request_transport
        result["prefetch_candidate_matched"] = prepared_token is not None
        result["prefetch_sent"] = bool(sent)
        result["prefetch_frames_sent"] = sent
        result["prefetch_window_ms"] = (round((time.monotonic() - prepared_sent_at) * 1000, 2)
                                         if prepared_sent_at is not None else None)
        return result

    def stop(self) -> None:
        process = self.process
        if process is not None:
            if process.poll() is None:
                # EOF lets a completed worker synchronize and exit normally.
                # Forced termination is only the fallback for stalled work.
                if process.stdin and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=3)
            # Reap surviving descendants even if the worker itself exited early.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        if self.native_temp is not None:
            self.native_temp.cleanup()
            self.native_temp = None


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
        self.saved_prompts = {model: [] for model in MODELS}
        self.worker_model: str | None = None
        self.revision = 0
        self.sequence = 0
        self.state = "idle" if self.enabled else "disabled"
        self.error: str | None = None
        self.progress_stage: str | None = None
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
        # The inference thread owns its worker. Let the current completed frame
        # drain before its finally block closes/reaps the GPU process.
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        if self.thread.is_alive() and worker:
            worker.cancel()
            self.thread.join(timeout=10)
        if self.thread.is_alive() and worker:
            worker.stop()
            self.thread.join(timeout=3)

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
            self.progress_stage = None
            self.state = "loading" if self.prompts else "idle"
            self.condition.notify_all()

    def set_model(self, model: Any) -> None:
        if type(model) is not str or model not in MODELS:
            raise ValueError("model must be one of " + ", ".join(MODELS))
        if not model_available(model, self.config):
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
                "api_version": API_VERSION,
                "enabled": self.enabled,
                "device_type": self.config.get("device_type", "xpu"),
                "model": self.model,
                "models": [{"id": key, "label": value["label"], "available": model_available(key, self.config)}
                           for key, value in MODELS.items()],
                "classes": MODELS[self.model]["classes"],
                "prompt": self.prompts[0] if len(self.prompts) == 1 else "",
                "prompts": list(self.prompts),
                "max_prompts": max_prompts(self.model),
                "colors": COLORS,
                "revision": self.revision,
                "state": self.state,
                "progress_stage": self.progress_stage,
                "error": self.error,
                "frame_age_ms": age,
                **detection_result(self.result, self.config.get("device_type", "xpu")),
            }

    def frame(self, key: str) -> bytes | None:
        with self.condition:
            return self.frames.get(key)

    def _cache_frame(self, key: str, jpeg: bytes, boxes: list, captured_at: float) -> None:
        """Called under condition; retain bounded click metadata independent of age."""
        self.frames[key] = jpeg
        self.frame_selections[key] = {"boxes": boxes, "captured_at": captured_at}
        while len(self.frames) > 8:
            self.frames.popitem(last=False)
        # JPEG eviction must not invalidate a displayed frame. Keep the latest
        # 128 results even when inference itself takes longer than 750 ms.
        while len(self.frame_selections) > 128:
            self.frame_selections.popitem(last=False)

    def selection(self, revision: int, sequence: int, instance_id: int) -> dict:
        if any(type(v) is not int or v < 0 for v in (revision, sequence, instance_id)):
            raise ValueError("selection requires integer revision, frame_sequence and instance_id")
        with self.condition:
            frame = self.frame_selections.get(f"{revision}-{sequence}")
            if revision != self.revision or self.state != "running" or not frame:
                raise ValueError("That camera frame is no longer available; click a displayed box")
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
                self.progress_stage = stage
                # A new temporal graph shape is work on the next frame, not a
                # loss of the last completed result. Keep its JPEG/masks/IDs
                # together; controllers still consume each frame only once.
                if not (stage.startswith("compiling_tracker_") and self.result.get("frame_url")):
                    self.state = stage
                self.condition.notify_all()

    def _run(self) -> None:
        failed_revision = -1
        last_camera_sequence = 0
        prefetched = None
        retry_at, retry_delay = 0.0, 2.0
        try:
            while not self.stop_event.is_set():
                with self.condition:
                    self.condition.wait_for(
                        lambda: (
                            self.stop_event.is_set()
                            or (self.worker is not None and self.worker_model != self.model)
                            or (self.prompts and (self.revision != failed_revision or time.monotonic() >= retry_at))
                        ), timeout=1
                    )
                    if self.stop_event.is_set():
                        break
                    prompts, revision, model = list(self.prompts), self.revision, self.model
                    if revision == failed_revision and time.monotonic() < retry_at:
                        continue
                if self.worker is not None and self.worker_model != model:
                    self.worker.stop()  # Exit/reap the old process before allocating the new model.
                    with self.condition:
                        self.worker, self.worker_model = None, None
                        prefetched = None
                if not prompts:
                    continue
                cycle_started = time.monotonic()
                sequence, jpeg, captured_at = self.camera.wait_for_sample(
                    last_camera_sequence, 1
                )
                camera_ready = time.monotonic()
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
                try:
                    if self.worker is None:
                        worker_config = {**self.config, "model": model}
                        worker = self.worker_factory(worker_config)
                        with self.condition:
                            self.worker = worker
                            self.worker_model = model
                        worker.launch()
                        if self.stop_event.is_set():
                            break
                        # Cold checkpoint packing/compiler startup can exceed
                        # two minutes on the edge CPU. This is not a per-frame
                        # inference timeout; normal detection keeps its deadline.
                        if worker.receive(600).get("type") != "ready":
                            raise RuntimeError("Model worker did not become ready")
                        if time.monotonic() - captured_at > .1:
                            continue  # Do not process a pre-load/reconnect image.
                    with self.condition:
                        if revision != self.revision:
                            continue
                    self.sequence += 1
                    prepared_token = None
                    if (prefetched is not None and prefetched["revision"] == revision
                            and prefetched["id"] == self.sequence and prefetched["camera_sequence"] == sequence
                            and prefetched["jpeg"] == jpeg and 0 <= time.monotonic() - captured_at <= .1):
                        prepared_token = prefetched["token"]
                    prefetched = None
                    def prepare_next():
                        nonlocal prefetched
                        with self.condition:
                            if self.stop_event.is_set() or revision != self.revision:
                                return None
                            next_sequence, next_jpeg, next_at = self.camera.wait_for_sample(sequence, 0)
                            if next_sequence == sequence or next_jpeg is None or not 0 <= time.monotonic() - next_at <= .1:
                                return None
                            if prefetched is not None and prefetched["camera_sequence"] == next_sequence:
                                return None
                            prefetched = {"revision":revision, "id":self.sequence+1,
                                "camera_sequence":next_sequence, "jpeg":next_jpeg,
                                "token":f"{revision}:{self.sequence+1}:{next_sequence}"}
                            return prefetched
                    extra = ({"prepared_token":prepared_token, "prepare_next":prepare_next}
                             if getattr(self.worker, "prefetch_supported", False) else {})
                    if is_session_model(model):
                        camera_state = self.camera.status()
                        identity = camera_state.get("identity")
                        if "connection_generation" in camera_state:
                            identity = f"{identity}:{camera_state['connection_generation']}"
                        extra.update(session_revision=revision, captured_at=captured_at,
                                     camera_identity=identity)
                    result = self.worker.detect(
                        self.sequence,
                        jpeg,
                        prompts,
                        lambda stage: self._progress(revision, stage),
                        **extra,
                    )
                    graph_key = "cuda_graph" if self.config.get("device_type") == "cuda" else "sycl_graph"
                    if model in PROTEUS_MODELS:
                        if result.get("experimental_profile") != model:
                            raise RuntimeError("Experimental worker returned the wrong model profile")
                        validate_tracking_result(result, prompts, temporal=is_tracking_model(model),
                                                 backend="proteus-" + model)
                    elif is_tracking_model(model):
                        validate_tracking_result(result, prompts)
                    elif result.get("torch_compile") is not True or result.get(graph_key) is not True:
                        raise RuntimeError(
                            f"Model worker did not verify compiled {graph_key} execution"
                        )
                    worker_done = time.monotonic()
                    # Resolve the historical frame pose AFTER inference, when
                    # checked encoder reads can bracket each axis at capture.
                    # The provider must not return the current motor angles.
                    pose = self.pose_provider(captured_at) if self.pose_provider else None
                    if pose and (not 0 <= captured_at - pose["sampled_at"] <= 0.1
                                 or (pose.get("read_completed_at", pose["sampled_at"]) > captured_at
                                     and not (pose.get("interpolated") is True
                                              and pose["sampled_at"] == captured_at))):
                        pose = None
                    pose_done = time.monotonic()
                    annotated = jpeg if result.get("client_overlay") is True else base64.b64decode(result.pop("jpeg"), validate=True)
                    result.pop("type", None)
                    result.pop("id", None)
                    key = f"{revision}-{self.sequence}"
                    with self.condition:
                        if revision != self.revision:
                            continue  # Never display boxes from an obsolete prompt.
                        if not is_session_model(model):
                            result["boxes"] = self.instances.update(result.get("boxes", []), pose, captured_at)
                        self._cache_frame(key, annotated, result["boxes"], captured_at)
                        self.result = {
                            **result,
                            "frame_sequence": self.sequence,
                            "frame_url": f"/api/detection/frame/{key}.jpg",
                            "captured_at": captured_at,
                            "frame_pose": pose,
                            "pipeline_timing": {
                                "pose_ms": round((pose_done-worker_done)*1000, 2),
                                "capture_wait_ms": round((camera_ready-cycle_started)*1000, 2),
                                "worker_roundtrip_ms": round((worker_done-started)*1000, 2),
                                "cycle_ms": round((time.monotonic()-cycle_started)*1000, 2),
                            },
                        }
                        self.completed_at = captured_at
                        self.state = "running"
                        self.progress_stage = None
                        self.error = None
                        retry_delay = 2.0
                        self.condition.notify_all()
                except Exception as error:
                    with self.condition:
                        if revision == self.revision:
                            self.state, self.error = "error", str(error)
                            self.progress_stage = None
                            self.result = {}
                            self.frames.clear()
                            self.frame_selections.clear()
                            self.instances.clear()
                        worker, self.worker = self.worker, None
                        prefetched = None
                    if worker:
                        worker.stop()
                    failed_revision = revision
                    retry_at = time.monotonic() + retry_delay
                    retry_delay = min(30.0, retry_delay * 2)
                delay = self._sampling_delay(time.monotonic() - started)
                if delay:
                    self.stop_event.wait(delay)
        finally:
            if self.worker:
                self.worker.stop()
