"""Hardware-specific inference launch adapters behind one worker interface.

Only device runtime, model artifact selection and launch arguments belong here.
No target selection, camera scheduling, result publication or motor policy.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol
from .models import MODELS, is_tracking_model


@dataclass
class WorkerSpec:
    command: list[str]
    environment: dict[str, str]
    temporary: Any = None


class InferenceRuntime(Protocol):
    def prepare(self) -> WorkerSpec: ...


class XpuRuntime:
    def __init__(self, config):
        self.config, self.native_temp = config, None

    def prepare(self):
        from .sam31_w8a8 import packed_profile, verify_graphics
        w4a4 = self.config.get("model", "sam3.1") == "sam3.1" and self.config.get("sam31_w4a4_bundle")
        sam_bundle = self.config.get("sam31_w8a8_development_bundle")
        packed = (self.config.get("model", "sam3.1") == "sam3.1" and sam_bundle is not None
                  and packed_profile(Path(sam_bundle)))
        cache = Path(self.config["cache_dir"])
        tracking = is_tracking_model(self.config.get("model"))
        v18 = self.config.get("model") == "sam3.1-v18"
        native_tracking_bundle = self.config.get("sam31_tracking_v18_bundle" if v18 else "sam31_tracking_native_bundle")
        if v18 and not native_tracking_bundle:
            raise ValueError("SAM 3.1 v18 requires its pinned native tracking bundle")
        if self.config.get("model") == "sam3.1-mask":
            cache /= "sam31-mask-20260908"
        elif v18:
            cache /= "sam31-tracking-israel-full1008-v18"
        elif self.config.get("model") == "sam3.1-tracking":
            native = self.config.get("sam31_tracking_native_bundle")
            cache /= ("sam31-tracking-native-" + hashlib.sha256(native.encode()).hexdigest()[:12]
                      if native else "sam31-tracking")
        elif w4a4:
            cache /= "sam31-sleepy-w4a4-fixed-fc1-barrier"
        elif self.config.get("sam31_w8a8_development_bundle"):
            cache /= "sam31-israel-w8a8-packed" if packed else "sam31-israel-w8a8"
        elif self.config.get("sam31_native_bundle"):
            cache /= "sam31-native"
        elif self.config.get("device_type") == "cuda":
            cache /= "sam31-cuda"
        cache.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "4",
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "XDG_CACHE_HOME": str(cache),
        }
        if w4a4:
            env["TORCHINDUCTOR_FREEZING"] = "1"
            env["LD_LIBRARY_PATH"] = str(Path(self.config["python"]).parent.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
            graphics_bundle = Path(self.config.get("sam31_tracking_bundle", "/nonexistent"))
            if (graphics_bundle / "runtime/graphics").is_dir():
                env["LD_LIBRARY_PATH"] = str(verify_graphics(graphics_bundle)) + ":" + env["LD_LIBRARY_PATH"]
        if self.config.get("model") == "sam3.1-mask":
            env["TORCHINDUCTOR_FREEZING"] = "1"
            if self.config.get("device_type", "xpu") == "xpu":
                env["LD_LIBRARY_PATH"] = str(Path(self.config["python"]).parent.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
                graphics_bundle = Path(self.config.get("sam31_tracking_bundle", "/nonexistent"))
                if (graphics_bundle / "runtime/graphics").is_dir():
                    env["LD_LIBRARY_PATH"] = str(verify_graphics(graphics_bundle)) + ":" + env["LD_LIBRARY_PATH"]
        if self.config.get("model", "sam3.1") == "sam3.1" and self.config.get("sam31_w8a8_development_bundle"):
            # Custom ops use Torch's SYCL ABI, not the dense graphs runner's
            # isolated oneAPI runtime. No system-wide library changes.
            env["LD_LIBRARY_PATH"] = str(Path(self.config["python"]).parent.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
            if packed:
                env["LD_LIBRARY_PATH"] = str(verify_graphics(Path(sam_bundle))) + ":" + env["LD_LIBRARY_PATH"]
        if self.config.get("model", "sam3.1") == "sam3.1" and self.config.get("sam31_native_bundle"):
            # Parent-owned so an interrupted/killed compile cannot leak shared
            # buffers in /dev/shm. Each worker gets an isolated directory.
            self.native_temp = tempfile.TemporaryDirectory(prefix="turret-native-", dir="/dev/shm")
            env["SPRING_NATIVE_IPC_DIR"] = self.native_temp.name
        if tracking and self.config.get("device_type", "xpu") == "xpu":
            if native_tracking_bundle:
                env["SPRING_SAM31_TRACKING_NATIVE_BUNDLE"] = native_tracking_bundle
            tracking_bundle = Path(self.config["sam31_tracking_bundle"])
            env["LD_LIBRARY_PATH"] = str(Path(self.config["python"]).parent.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
            if (tracking_bundle / "runtime/graphics").is_dir():
                env["LD_LIBRARY_PATH"] = str(verify_graphics(tracking_bundle)) + ":" + env["LD_LIBRARY_PATH"]
            if native_tracking_bundle:
                env["LD_LIBRARY_PATH"] = str(Path(native_tracking_bundle) / "lib") + ":" + env["LD_LIBRARY_PATH"]
        return WorkerSpec(
            [
                self.config["python"],
                "-u",
                str(Path(__file__).with_name("sam31_tracking_native_worker.py"
                    if self.config.get("model") == "sam3.1-tracking" and self.config.get("sam31_tracking_native_bundle")
                    else "sam31_w4a4_worker.py" if w4a4 else MODELS[self.config.get("model", "sam3.1")]["worker"])),
                "--checkpoint",
                self.config["checkpoint"],
                "--device",
                str(self.config.get("device", 0)),
                *(["--device-type", self.config.get("device_type", "xpu")]
                  if self.config.get("model", "sam3.1") in MODELS else []),
                "--precision",
                "bfloat16" if tracking else self.config.get("precision", "float16"),
                "--confidence",
                str(self.config.get("confidence", 0.5)),
                *(["--source-bundle", self.config["sam31_tracking_bundle"]]
                  if tracking else []),
                *(["--mask-bundle", self.config["sam31_mask_bundle"]]
                  if self.config.get("model") == "sam3.1-mask" and self.config.get("sam31_mask_bundle") else []),
                *(["--compiled-bundle", self.config["sam31_mask_compiled_bundle"]]
                  if self.config.get("model") == "sam3.1-mask" and self.config.get("sam31_mask_compiled_bundle") else []),
                *(["--native-bundle", self.config["sam31_native_bundle"]]
                  if self.config.get("model", "sam3.1") == "sam3.1"
                  and self.config.get("sam31_native_bundle") else []),
                *(["--w8a8-development-bundle", self.config["sam31_w8a8_development_bundle"]]
                  if self.config.get("model", "sam3.1") == "sam3.1"
                  and self.config.get("sam31_w8a8_development_bundle") else []),
                *(["--allow-unqualified-w8a8"] if self.config.get("model", "sam3.1") == "sam3.1"
                  and self.config.get("sam31_allow_unqualified_w8a8") else []),
                *(["--w4a4-bundle", w4a4, "--allow-w4a4-accuracy-tradeoff"] if w4a4 else []),
            ],
            env, self.native_temp,
        )


class CudaRuntime:
    def __init__(self, config):
        self.config = config

    def prepare(self):
        config = self.config
        model = config.get("model", "sam3.1")
        if model == "sam3.1-v18":
            raise ValueError("SAM 3.1 v18 is Intel-only; select SAM 3.1 Tracking on CUDA")
        if any(config.get(k) for k in ("sam31_native_bundle", "sam31_w8a8_development_bundle",
               "sam31_w4a4_bundle", "sam31_tracking_native_bundle", "sam31_tracking_v18_bundle", "sam31_mask_bundle")):
            raise ValueError("Intel native bundles cannot be loaded by CUDA")
        cache = Path(config["cache_dir"]) / {
            "sam3.1": "sam31-cuda", "sam3.1-mask": "sam31-mask-20260908",
            "sam3.1-tracking": "sam31-tracking"}[model]
        cache.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "OMP_NUM_THREADS":"4", "TORCHINDUCTOR_COMPILE_THREADS":"2",
               "TORCHINDUCTOR_CACHE_DIR":str(cache/"inductor"),
               "TRITON_CACHE_DIR":str(cache/"triton"), "XDG_CACHE_HOME":str(cache)}
        if model == "sam3.1-mask":
            env["TORCHINDUCTOR_FREEZING"] = "1"
        command = [config["python"], "-u", str(Path(__file__).with_name(MODELS[model]["worker"])),
                   "--checkpoint", config["checkpoint"], "--device", str(config.get("device", 0)),
                   "--device-type", "cuda", "--precision",
                   "bfloat16" if model == "sam3.1-tracking" else config.get("precision", "float16"),
                   "--confidence", str(config.get("confidence", .5))]
        if model == "sam3.1-tracking":
            command += ["--source-bundle", config["sam31_tracking_bundle"]]
        return WorkerSpec(command, env)


def inference_runtime(config) -> InferenceRuntime:
    device = config.get("device_type", "xpu")
    if config.get("model", "sam3.1") not in MODELS:
        raise ValueError("Unsupported inference model")
    if device not in ("xpu", "cuda"):
        raise ValueError("Unsupported inference device")
    return {"xpu": XpuRuntime, "cuda": CudaRuntime}[device](config)
