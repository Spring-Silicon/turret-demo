"""Pinned sleepy-joe image artifact, isolated from PyTorch's SYCL libraries.

The native runner owns one resident image graph. Shared-memory IPC adds a
host transfer at each boundary; reported stage time includes that overhead.
Text and grounding remain in the original torch.compile/XPUGraph worker.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import selectors
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

ARTIFACT = "c98974a61db4e8e0a73b825c60b85862406d934cd8631c7189df8eb2b77c051a"
CHECKPOINT = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
ARTIFACT_DIR = "native-attention-prefetch1"
RUNTIME_DIR = "runtime-attention-prefetch1"
PINNED_FILES = {
    "graphs-runner-intel-turret": "a3e17d434ba2f7ef26daf870e0ff9ed4053fb7aaa5bdba4a1bc9789d0a0cd266",
    "libumf.so.1": "c74cfea0360d09b5072a8227efbc830db36bd57669ca22d190d0fb31fe8e3425",
    "libhwloc.so.15": "34e1ad15a2d207dc116d9f88fbf5f9ab97cc5ce5e6a704e54a587e8bc7ddefff",
    "libsycl.so.9": "d769a71f2c71530e59a5ba22831ec5a82edcde718076d2663772255498c8e053",
    "libur_loader.so.0": "5338c03524f6d3fdd7d6067e5a3f84315e36e11fe1f526d5bbd02d0c88f82a32",
    "libur_adapter_level_zero.so.0": "ec3e2a2672db65f99c0d55b233f42602a8415ab19c177c4330f2e5ed5a78eedf",
    "libur_adapter_level_zero_v2.so.0": "bfdc0f26bf88bd0bccd66fc25302a4b22559f8fe30e499be18529be3ed610e60",
    "libsvml.so": "a6dbbc425c8270c3485e40ef0924ef38ed89ced4026056fd3f1f529774c4f86c",
    "libimf.so": "67a4b6c27002f59f26c411f4ec384300d781365af19d6a2eb9f0dd94d2a0f470",
    "libirng.so": "2cd4057ba6bcd9235685597098c3bdc9811e1de0a457f7a0a2aa47ce691995c7",
    "libintlc.so.5": "612ab119242ccda945de6eb354369b5bee6a158bc9724e4da5e466cb0ad630db",
    "libdnnl.so.3": "3add4417acc7093ccdcc0dfe7f20d19cded96073f4bafd65a704ef2e72aafd19",
    "libtbb.so.12": "2c1fcbc65a33771a5e02a201734774252528e10f57cef9e2f902fe209c0319e1",
    f"{ARTIFACT_DIR}/manifest.json": "52b163732a9ffa2e845b000b40d5acbd47a25c042ec565bf652eef6bce400a78",
    "graphs-runner-intel": "691426c27097b37a6c1d36aa490474f8e1acf0c4ed8115270f95db236cadf70a",
    f"{RUNTIME_DIR}/libgraphs_runtime_intel.so": "d45e111a85780f8cc1e65ef1c3efa48f339d5842b4f3173bb11d6e7d977a9cb0",
    f"{RUNTIME_DIR}/libgraphs_module_intel_onednn.so": "e9cca4dacc5411e53b301366d109fdf47d7eff33270bda44aea5c9322dd971b5",
    f"{RUNTIME_DIR}/libgraphs_module_intel_tla.so": "81a0f8d68ce5adc08e8ea7db4a445f51b6310e10a95bdd9da4aaa19afff9b4a3",
}
INPUT_SHAPE = [1, 3, 1008, 1008]
OUTPUT_SHAPE = [1, 256, 72, 72]
INPUT_BYTES = 12192768
OUTPUT_BYTES = 2654208


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(bundle: Path, checkpoint: Path) -> None:
    if digest(checkpoint) != CHECKPOINT:
        raise ValueError("Native SAM image requires the pinned SAM 3.1 checkpoint")
    for name, expected in PINNED_FILES.items():
        if digest(bundle / name) != expected:
            raise ValueError(f"Native SAM bundle checksum mismatch: {name}")
    # The runner additionally validates the artifact's content-addressed objects.


def validate_handshake(message: dict) -> None:
    if (message.get("schema") != "spring.graphs-runner-shared-memory.v2"
            or message.get("artifact") != ARTIFACT
            or message.get("transport") != "shared_memory"
            or message.get("command_graph_count") != 1
            or message.get("input_bytes") != INPUT_BYTES
            or message.get("output_bytes") != 2 * OUTPUT_BYTES):
        raise ValueError("Native SAM runner returned an incompatible handshake")
    inputs, outputs = message.get("inputs", []), message.get("outputs", [])
    if len(inputs) != 1 or len(outputs) != 2:
        raise ValueError("Native SAM runner changed the image boundary")
    if (inputs[0].get("slot") != "input.459"
            or inputs[0].get("shape") != INPUT_SHAPE or inputs[0].get("dtype") != "f32"
            or inputs[0].get("byte_len") != INPUT_BYTES):
        raise ValueError("Native SAM runner changed the image input")
    for index, spec in enumerate(outputs):
        if (spec.get("slot") != f"output.{index}" or spec.get("dtype") != "f16"
                or spec.get("shape") != OUTPUT_SHAPE
                or spec.get("byte_len") != OUTPUT_BYTES):
            raise ValueError("Native SAM runner changed the image outputs")


def validate_outputs(message: dict) -> None:
    expected = [{"slot": f"output.{i}", "shape": OUTPUT_SHAPE,
                 "byte_len": OUTPUT_BYTES} for i in range(2)]
    if message.get("outputs") != expected or message.get("command_graph_count") != 1:
        raise ValueError("Native SAM response changed the image outputs")


class NativeImageStage:
    backend = "graphs-native-sycl"
    accepts_cpu = True

    def __init__(self, torch: Any, bundle: Path, checkpoint: Path, device: Any,
                 progress: Any):
        if torch.__version__ != "2.14.0+xpu":
            raise ValueError("Native SAM qualification requires torch 2.14.0+xpu")
        # Avoid ambiguous device-index mappings between two different runtimes.
        if torch.xpu.device_count() != 1 or "B580" not in torch.xpu.get_device_name(device):
            raise ValueError("This native SAM bundle is qualified for a single B580")
        verify_bundle(bundle, checkpoint)
        self.torch, self.bundle, self.device, self.progress = torch, bundle, device, progress
        self.process = None
        self.temp = None
        self.maps: list[Any] = []
        self.calls = 0
        self.proof: dict[str, Any] = {}
        self.last_execution_ms: float | None = None
        self.env = {**os.environ, "LD_LIBRARY_PATH": f"{bundle}:{bundle / RUNTIME_DIR}",
                    "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu"}
        self.command = [str(bundle / "graphs-runner-intel-turret"), "--device", "0",
                        "--runtime-library", str(bundle / RUNTIME_DIR / "libgraphs_runtime_intel.so")]

    def _read(self, size: int, deadline: float) -> bytes:
        assert self.process and self.process.stdout
        result = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while len(result) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Native SAM image runner timed out")
                chunk = os.read(self.process.stdout.fileno(), size - len(result))
                if not chunk:
                    raise RuntimeError("Native SAM image runner exited; check service log")
                result.extend(chunk)
        return bytes(result)

    def _start(self, pixels: Any) -> None:
        from safetensors.torch import load_file, save_file

        self.progress("loading", "native image + SYCL replay qualification")
        self.temp = tempfile.TemporaryDirectory(
            prefix="sam31-native-", dir=os.environ.get("SPRING_NATIVE_IPC_DIR", "/dev/shm"))
        root = Path(self.temp.name)
        # Prove actual command-graph capture and direct/replay agreement on this
        # device with these exact binaries, not merely a requested graph flag.
        save_file({"input.459": pixels}, root / "input.safetensors")
        receipts = []
        for mode in ("direct", "replay"):
            receipt, output = root / f"{mode}.json", root / f"{mode}.safetensors"
            command = [*self.command, "benchmark", "--artifact", str(self.bundle / ARTIFACT_DIR),
                       "--inputs", str(root / "input.safetensors"), "--outputs", str(output),
                       "--timings", str(receipt), "--warmup", "1", "--iterations", "1"]
            if mode == "direct":
                command.append("--no-command-graphs")
            subprocess.run(command, env=self.env, check=True, timeout=90,
                           stdout=subprocess.DEVNULL)
            receipts.append(json.loads(receipt.read_text()))
        if (receipts[0].get("command_graph_count") != 0
                or receipts[1].get("command_graph_count") != 1):
            raise RuntimeError("Native SAM did not capture exactly one SYCL graph")
        direct, replay = load_file(root / "direct.safetensors"), load_file(root / "replay.safetensors")
        if direct.keys() != replay.keys() or len(direct) != 2:
            raise RuntimeError("Native SAM direct/replay output contract differs")
        for key in direct:
            self.torch.testing.assert_close(direct[key], replay[key], rtol=.001, atol=.001)
        self.proof = {"artifact": ARTIFACT, "command_graph_count": 1,
                      "direct_replay_passed": True}
        for name, size in (("input.bin", INPUT_BYTES), ("output.bin", 2 * OUTPUT_BYTES)):
            with (root / name).open("w+b") as stream:
                stream.truncate(size)
                self.maps.append(mmap.mmap(stream.fileno(), size))
        self.process = subprocess.Popen(
            [*self.command, "serve", "--artifact", str(self.bundle / ARTIFACT_DIR),
             "--input-buffer", str(root / "input.bin"),
             "--output-buffer", str(root / "output.bin"), "--warmup", "1"],
            env=self.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )
        line = bytearray()
        deadline = time.monotonic() + 90
        while not line.endswith(b"\n"):
            if len(line) >= 16384:
                raise ValueError("Native SAM handshake exceeded its size limit")
            line.extend(self._read(1, deadline))
        validate_handshake(json.loads(line))

    def __call__(self, pixels: Any) -> tuple[Any, Any]:
        if list(pixels.shape) != INPUT_SHAPE or pixels.dtype != self.torch.float32:
            raise ValueError("Native SAM requires FP32 [1,3,1008,1008] pixels")
        pixels = pixels.detach().cpu().contiguous()
        try:
            if self.process is None:
                self._start(pixels)
            assert self.process and self.process.stdin
            self.maps[0][:] = pixels.numpy().tobytes()
            self.process.stdin.write(b"GRAPHSMQ" + struct.pack("<Q", self.calls))
            self.process.stdin.flush()
            deadline = time.monotonic() + 30
            header = self._read(20, deadline)
            if header[:8] != b"GRAPHSMA" or struct.unpack("<Q", header[8:16])[0] != self.calls:
                raise ValueError("Native SAM response sequence mismatch")
            size = struct.unpack("<I", header[16:])[0]
            if not 0 < size <= 16384:
                raise ValueError("Native SAM response metadata exceeded its size limit")
            metadata = json.loads(self._read(size, deadline))
            validate_outputs(metadata)
            self.last_execution_ms = metadata["image_execution_ms"]
            # Own CPU bytes before the next replay; upload only the two small
            # feature tensors needed by the unchanged grounding heads.
            outputs = tuple(self.torch.frombuffer(
                bytearray(self.maps[1][i * OUTPUT_BYTES:(i + 1) * OUTPUT_BYTES]),
                dtype=self.torch.float16).reshape(OUTPUT_SHAPE).to(self.device)
                for i in range(2))
            self.calls += 1
            return outputs
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
            self.process = None
        for mapping in self.maps:
            mapping.close()
        self.maps.clear()
        if self.temp is not None:
            self.temp.cleanup()
            self.temp = None
