# Reproduce the Arc + Thor deployment

This is the authoritative recovery guide for **spring-edge-turret + agxthor-5**,
captured 2026-09-09. Older `spring-edge-2` / `agxthor-4` instructions are historical.
Use the checked-in [deployment locks](../deploy/repro/) and each private recovery
bundle's `inventory.json`, not “latest” packages or a mutable model directory on
`sleepy-joe` / `israel`.

There are two layers:

1. **Frozen runtime restore:** exact deployed Python environments, native model
   code/libraries/weights, configuration and physical calibration; on Thor, the
   exact Docker image too. This is the recovery baseline.
2. **Source update:** install a reviewed commit of this repository onto that
   baseline. This replaces the application, not Torch, GPU libraries, native
   model artifacts, calibration or host networking.

The recovery bundles are **not bootable OS images**. A matching base OS/BSP is a
prerequisite. No clean-board reflash/restore has been performed to qualify these
instructions. Archive integrity and contents can be checked without touching the
live demo; cold boot and three-profile acceptance must be repeated on a restored
device before calling it ready.

## Frozen foundations

| Component | Arc | Thor |
|---|---|---|
| Host / architecture | spring-edge-turret / x86_64 | agxthor-5 / aarch64 |
| Base OS | Spring Edge 0.1.2, Ubuntu 24.04.4 | Ubuntu 24.04.4, Jetson L4T R39.2.1 |
| Kernel | 7.0.0-31-generic | 6.8.12-1021-tegra |
| Application account | spring, UID/GID 1000 | spring, UID/GID 1000 |
| Python | 3.12.3, separate backend/inference venvs | vendor container Python 3.12.3 |
| Torch | custom 2.14.0+xpu, archived verbatim | 2.8.0a0+34c6371d24.nv25.08 |
| GPU stack | Level Zero 1.32.0; Intel compute runtime 26.31.39395.13 | CUDA 13.0; host NVIDIA container toolkit 1.19.1-1 |
| Power | no custom overclock service captured | nvpmodel MAXN, mode 0 |
| Viewer | Node 22.23.2; Firefox snap 147.0.3-1, revision 7766 | no local viewer required |

Arc's installed Intel package revisions and offline kernel/driver packages are
preserved in the bundle. The exact package list is in `inventory.json`.
`pip install torch` or installing a newer Intel runtime is **not** an equivalent
restore. The native kernels are tied to the archived Torch/runtime ABI.

Thor's L4T packages are `39.2.1-20260806224157`. Install that BSP with NVIDIA's
board-specific flashing procedure, then verify `/etc/nv_tegra_release` and
`uname -r`. A generic ARM Ubuntu image or another JetPack release is not the
qualified foundation. The container does not contain the host kernel/driver.

The frozen Thor image is:

```text
spring-turret-demo:0.17.3-recovery-agxthor-5
sha256:b86f5fe9fb21277367cd1da5f9aaf33f02e6f710388decbf0dd9a9fcaac19049
```

Its upstream base is
`nvcr.io/nvidia/pytorch:25.08-py3@sha256:ace9a848c0ae543317e3c4763b6b4248961c47902625abfe3c77a0fb931c50fb`.
The archived image also contains the deployed SAM code and runtime patches;
rebuilding the older `deploy/thor/Dockerfile` alone does not recreate it.

The inspected Arc GPU is **Intel Arc B580**, PCI `8086:e20b`, ASRock subsystem
`1849:6021`, driven by `xe`. Verify that identity after restore; these native
runtime results are not qualification of another Intel GPU.
The CPU is AMD Ryzen 5 4500. The running GPU has a 16 GiB prefetchable PCI BAR;
retain the board's Above-4G/Resizable-BAR configuration and verify the BAR after
recovery. A filesystem archive does not save BIOS/NVRAM settings.

### Rebuilding the Arc base OS

The installed identity records edge-image commit
[`3899c012a859d7d07fefc6561044999c6d3cbbd1`](https://github.com/Spring-Silicon/edge-image/tree/3899c012a859d7d07fefc6561044999c6d3cbbd1),
profile `base`, storage mode `production-512`, built 2026-09-07 16:20:08 UTC.
See [arc-os-20260909.json](../deploy/repro/arc-os-20260909.json) for the original
Ubuntu ISO checksum and private foundation file hashes.

Check out that edge-image commit, follow its README build prerequisites, and
supply the password hash/operator public key appropriate to the replacement
machine. The private directory `/home/spring/recovery-20260909-arc-base` holds
the pinned graphs runtime `.deb`, public OTA certificate/keyring, original
`versions.yaml` and build metadata; it is mirrored on both devices. Set
`SPRING_RUNTIME_DEB_FILE` to that verified `.deb` and `SPRING_RAUC_KEYRING_FILE`
to the public certificate. The private OTA signing key is neither required for
installing this baseline nor included in a recovery bundle.

Use the repository's `make fetch-upstream`, `make validate`, and production
`make build-iso` workflow, then its documented installer/QEMU qualification.
Do not flash the special `build-iso-qemu` image onto hardware: it bypasses the
physical-media prompt. The production storage selector expects the recorded
512 GB-class target; review the target rather than forcing that layout onto an
unrelated disk. After installing, set the intended replacement hostname with
`hostnamectl`, enroll fresh SSH/Tailscale identity and restore the demo runtime.
The OS compatibility Torch version in `versions.yaml` is **not** the demo's
custom XPU venv: the private demo archive supplies the latter.

## Model artifacts (private, never commit the weights)

All profiles use the authorized SAM 3.1 checkpoint with SHA256:

```text
0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6
```

| Profile | Arc runtime under /home/spring/.local/share/turret-demo | Thor |
|---|---|---|
| SAM 3.1 Box | current-best-20260908/bundle | compiled dense detector + CUDA replay |
| SAM 3.1 Mask | mask-20260908/bundle | compiled dense mask detector + CUDA replay |
| SAM 3.1 Tracking | sam-bundle + israel-tracking-v24 | compiled dense temporal tracker + CUDA replay, /opt/sam3 |

The Arc export includes the **external symlink targets**
`/home/spring/sam3_1` and
`/home/spring/.local/share/turret-demo/sleepy-mask-update-20260909.bDg4xS/bundle/attention8`.
Copying only the three named bundle directories misses these dependencies.
It also includes both venvs, frozen OpenCL libraries and the standalone Python
header tree referenced by `CPATH`. Retain the exact absolute paths: generated
launchers/native imports and symlinks use them.

The Arc optimized models retain previously accepted accuracy tradeoffs. Exporting
them does not establish new accuracy equivalence to Thor. See the qualification
documents for [Box](sam31-w4a4.md), [Mask](sam31-mask.md), and
[Tracking](sam31-tracking.md). Shared target/control policy is documented in
[sam-policy-audit.md](sam-policy-audit.md); hardware-specific model execution is
intentionally different.

## Physical assemblies and calibration

| Setting | Arc's current assembly | Thor's current assembly |
|---|---|---|
| Controller USB serial | 5B3D045331 | 5B3D044488 |
| Saved X / Y zero (encoder ticks) | 1005 / 22 | 1991 / 4061 |
| Geometry/zeros filename suffix | 5B3D045331 | 5B3D044488 |
| Camera | Arducam 1080P Low Light, UC684 | Arducam 1080P Low Light, UC684 |
| Capture | 1280×720 MJPEG, 30 FPS | 1280×720 MJPEG, 30 FPS |

Both assemblies: DYNAMIXEL XL330-M288-T, model 1200, Protocol 2.0,
57,600 baud, extended-position mode 4; X/pan ID 2, Y/tilt ID 1; angle limits
±90° on each axis. Velocity/acceleration profile values are 0. Camera-to-servo
direction is X +1 / Y -1. Arc explicitly configures P=1200, I=0, D=1600 and
packed feedback; Thor uses its captured configuration. These values are recorded,
not a new tuning or hardware-equivalence claim.

Calibration belongs to the **physical assembly**, not the computer. The two
turrets were swapped; do not exchange their saved zeros again during recovery.
The camera's UC684 identifier is not sufficient to distinguish these two cameras.
Match the controller serial, mount and associated geometry together. An API zero
may be shifted by an integer turn (4096 ticks) to match current extended-position
feedback; that is not a reason to overwrite the saved zero.

For a new assembly: set unique IDs one servo at a time, verify mode/baud/model,
commission zeros at the actual mechanical reference and qualify its camera
geometry. Do not restore these assembly-specific files onto unknown hardware.
See [control tuning](control-tuning.md) and [geometry qualification](geometry-qualification.md).

## Export and verify a recovery bundle

Run from a reviewed repository checkout. These commands read the deployment and
write a NEW backup directory; they do not restart services or move motors.

```sh
# On Arc, with tools/deployment_bundle.py copied there:
sudo python3 deployment_bundle.py capture arc \
  --output /home/spring/recovery-NEW-arc \
  --source-commit FULL_40_CHARACTER_REVIEWED_COMMIT
# On Thor:
sudo python3 deployment_bundle.py capture thor \
  --output /home/spring/recovery-NEW-thor \
  --source-commit FULL_40_CHARACTER_REVIEWED_COMMIT
```

The output is root-private by default. If transferring as spring, grant ownership
only on the exact newly-created bundle directory. It contains `runtime.tar.zst`,
`inventory.json`, `SHA256SUMS`, and (Thor) `container.tar.zst`. Do not upload these
archives to this public GitHub repository or a public release. Preserve SAM and
vendor license/access restrictions. No SSH keys, passwords, Tailscale identities,
Wi-Fi credentials or Firefox cookies are exported.

```sh
python3 tools/deployment_bundle.py verify /path/to/private-bundle
```

Verification checks the inventory checksum, archive SHA256/size, compressed
streams, required paths and archived configuration/checkpoint content. Compare
the inventory/archive digests against the checked-in deployment lock as well:
checksums delivered with an untrusted archive alone do not authenticate it.
Retain a copy outside the device before erasing its disk.

The recorded bundles are `/home/spring/recovery-20260909-arc-v2` and
`/home/spring/recovery-20260909-thor`. Their public locks are
`deploy/repro/arc-20260909.json` and `deploy/repro/thor-20260909.json`.
The earlier Arc export without `-v2` is incomplete (external model symlink
targets were missing); **do not use it**. Use the v2 digest, not just a similar
directory name. Both devices' complete bundles are mirrored across the direct
Ethernet link so recovery does not depend solely on the failed device's disk.

Capturing a live filesystem is not a general-purpose atomic snapshot. Keep code,
model artifacts and calibration unchanged during export. The exporter rejects
changed top-level configuration/checkpoint files and missing/unbundled symlink
dependencies. Runtime target IDs, selected live prompts, motor Start state and
compiler caches are intentionally excluded. Boot defaults come from config and
the inference-startup service (`person` only if no prompt is set).
`source_commit` records the associated reviewed application source revision;
the archive hashes/image ID pin the exact deployed contents, including existing
platform-specific runtime patches. It is not a claim that vendor/native artifacts
can be rebuilt bit-for-bit from the application Git commit alone.

## Restore on a matching, fresh or explicitly retired installation

Do not extract over a running demo. First obtain a trusted matching OS/BSP,
create `spring` with UID/GID 1000, install GNU tar, zstd, Python 3.12 and the
foundation packages, and verify the private bundle. Keep the hardware stopped
while restoring. Extraction writes system/user configuration at its original
paths and must be explicitly intended for the replacement host.

Inspect first; then extract **only on the intended restoration target**:

```sh
tar --zstd -tf /path/to/private-bundle/runtime.tar.zst | less
sudo tar --zstd --xattrs --acls --numeric-owner \
  -xpf /path/to/private-bundle/runtime.tar.zst -C /
```

### Arc

1. Install the pinned kernel and Intel libraries from the restored
   `/opt/spring-provision/offline-apt` repository using APT dependency resolution.
   Its `Packages` index and deb files are included. Configure a local trusted
   `file:/opt/spring-provision/offline-apt ./` APT source (trust only this verified
   bundle), then install the exact versions listed in the lock/inventory.
   Do not blindly install every deb in that directory or remove the running
   kernel. Reboot into `7.0.0-31-generic` and verify before proceeding.
2. Required host utilities: `python3.12`, `gstreamer1.0-tools`,
   `gstreamer1.0-plugins-good`, `nftables`, `network-manager`, `curl`, GDM/GNOME,
   Firefox/snapd, XWayland and compiler tools (`build-essential`, `pkg-config`).
   Preserve the Ubuntu package versions in the
   inventory. The venv Python symlinks depend on `/usr/bin/python3.12`; do not
   relocate or recreate these venvs using a different Python.
3. Ensure spring has access to `video` and `render`; the serial udev rule assigns
   the exact controller to spring. Confirm the restored home files are owned by
   UID 1000. Create the excluded writable cache:

   ```sh
   sudo install -d -o spring -g spring /home/spring/.cache/turret-demo
   sudo loginctl enable-linger spring
   sudo udevadm control --reload-rules
   ```

4. Restore the wired link as below, verify `/dev/spring-turret-servo` and the
   configured `/dev/v4l/by-id/...UC684-video-index0`, then as spring:

   ```sh
   systemctl --user daemon-reload
   systemctl --user enable --now spring-turret-demo.service spring-turret-frontend.service
   systemctl --user enable --now spring-turret-inference-startup.service
   ```

5. The restored GDM configuration intentionally enables automatic login for
   spring. The restored desktop autostart entry launches fullscreen Firefox
   after login, not from the headless linger session. Firefox uses a dedicated
   profile and XWayland environment to avoid the observed invisible Wayland
   kiosk window. Node, launchers and static assets are archived. Firefox itself
   belongs to the base OS: install/retain the recorded snap revision and its
   required base snap; a newer revision needs another kiosk check. See
   [startup.md](startup.md) for service ordering and commands.

### Thor

1. Install Docker `29.1.3-0ubuntu3~24.04.2` and NVIDIA container toolkit
   `1.19.1-1` on the matching L4T foundation. The restored Docker daemon config
   registers `nvidia-container-runtime`. Enable Docker; verify `docker info`
   lists the NVIDIA runtime. This is a host dependency, not part of `docker save`.
2. Load the exact image and restore the tag expected by the saved launcher:

   ```sh
   set -o pipefail
   zstd -dc /path/to/private-bundle/container.tar.zst | sudo docker load
   sudo docker image inspect sha256:b86f5fe9fb21277367cd1da5f9aaf33f02e6f710388decbf0dd9a9fcaac19049
   sudo docker tag sha256:b86f5fe9fb21277367cd1da5f9aaf33f02e6f710388decbf0dd9a9fcaac19049 \
     spring-turret-demo:0.17.3-recovery-agxthor-5
   sudo install -d -o spring -g spring /var/cache/spring-turret-demo
   ```

3. Confirm the two udev rules, config, model, geometry and zero files match the
   inventory. The current systemd drop-in selects
   `/opt/spring/turret-demo/run-container-overhead.sh`, not the historical default
   launcher. Both scripts and the drop-in are archived. Do not substitute the
   old `0.16.0-thor` repository launcher during exact recovery.
4. Restore the wired link, enable Docker/backend/startup services, and use MAXN
   only with the same suitable power supply and cooling (native thermal limits
   remain enabled):

   ```sh
   sudo nvpmodel -m 0
   sudo udevadm control --reload-rules
   sudo systemctl daemon-reload
   sudo systemctl enable --now docker.service spring-turret-demo.service
   sudo systemctl enable --now spring-turret-inference-startup.service
   ```

The container runs as 1000:1000, video GID 44 and dialout GID 20, with the live
host `/dev` read-only at `/host/dev` and cgroup access to V4L2/USB ACM. No privileged
container or Docker socket is needed. Model mounts are read-only; state and
compiler cache are writable. See [recovery.md](recovery.md) for hotplug behavior.

### Direct Ethernet (no Wi-Fi fallback)

Verify NIC names and MACs before applying these values. The dedicated profile
must not replace an Internet/management connection.

| Host | Interface / MAC | IPv4 |
|---|---|---|
| Arc | enp7s0 / 9c:6b:00:dd:04:2e | 192.168.249.1/30 |
| Thor | enP2p1s0 / 44:49:c0:3c:fc:5c | 192.168.249.2/30 |

The network profile is recreated explicitly, not copied with Wi-Fi credentials.
On a fresh Arc (profile must not already exist):

```sh
sudo nmcli connection add type ethernet con-name turret-direct-ethernet \
  ifname enp7s0 802-3-ethernet.mac-address 9c:6b:00:dd:04:2e \
  connection.autoconnect yes connection.autoconnect-priority 999 \
  ipv4.method manual ipv4.addresses 192.168.249.1/30 \
  ipv4.never-default yes ipv6.method disabled
sudo nmcli connection up turret-direct-ethernet
```

On Thor use `enP2p1s0`, MAC `44:49:c0:3c:fc:5c`, address `192.168.249.2/30`.
No gateway/DNS is supplied. The restored pre-up dispatcher installs the dedicated
nftables table. Confirm `nft list table inet turret_direct_link` blocks the peer
on other interfaces. On replacement hardware, update the firewall interface,
dispatcher allowlist, NetworkManager profile and Arc frontend's `--thor-interface`
together; otherwise it must fail closed.

Arc's combined page is `http://127.0.0.1:8081/`, with Arc backend loopback:8080
and Thor `192.168.249.2:8080`. Confirm `ip route get 192.168.249.2` chooses
`enp7s0 src 192.168.249.1` and the reverse on Thor. A disconnected cable must
show Thor unavailable while Arc continues; never substitute a Wi-Fi/tailnet URL.
Enroll management SSH/Tailscale independently; do not clone machine identities.
The APIs have no password login, so restrict them to trusted management/cable
networks; do not forward port 8080 to the public Internet.

## Updating the application from Git

Preserve a working private bundle first. Pin a reviewed full Git commit; do not
use an unreviewed floating `main` for production. This step is a deployment and
requires stopping/restarting the corresponding backend, separate from export.

On Arc, the minimal backend venv intentionally has no pip. Build a wheel in a
separate Python 3.12 build environment using setuptools 78.1.0 / wheel 0.45.1,
then install it using an external pip's `--python` option targeting the existing
**backend** venv, with `--no-deps`. Do not bootstrap or upgrade Torch inside the
inference venv. Leave `inference-venv` and native bundles untouched. Copy
`src/spring_turret/static` and `scripts/frontend.mjs` to the
matching paths beneath the existing frontend deployment. Restart only the
affected services, then run acceptance below.

On Thor, build an application-only overlay on the loaded frozen runtime:

```sh
sudo bash scripts/build-thor-from-recovery.sh spring-turret-demo:REVIEWED_COMMIT
```

The overlay installs the package without downloading/replacing dependencies.
Record its image ID and source commit, update the exact image reference in the
active launcher and restart the backend. Do not retag the immutable recovery
baseline. Export a new bundle after successful hardware validation. Compilation
cache reuse is optional; a cold restore compiles/captures again and is not an
instant model switch. No warm-worker-reuse scheme is required.

Standalone Arc, standalone Thor and combined frontends remain supported:
`node scripts/frontend.mjs --arc URL`, `--thor URL`, or both; no npm install.
For workstation use see [local-frontend.md](local-frontend.md). On this deployed
pair, keep the wired Arc-hosted combined page as the unattended default.

## Acceptance and rollback

1. Verify OS/BSP, package/image digests, user IDs, USB serials, zero/geometry
   hashes, services and direct routes. Query `/api/status` and
   `/api/detection/status` on each backend and both frontend proxy endpoints.
2. Select each of Box, Mask and Tracking, submit the same prompt, wait for cold
   compilation, and confirm new processed frame sequences/JPEGs, finite model
   and loop timing, expected masks/IDs, and SYCL/CUDA replay status. A loaded
   web page, an imported Torch module or an encoder microbenchmark is not enough.
3. Check the unified selector/prompts and both single-device views. Viewer reload
   or close must not reset models/targets or send motor commands. Displayed
   images and overlays must share the worker's processed frame sequence.
4. Check boot with no browser attached, then graphical autologin/kiosk. Both
   backends and default inference should start. A fresh backend remains unarmed;
   Start/Enter arms explicitly, Stop cancels even during a device outage.
5. In a supervised hardware test, verify camera/servo unplug/replug recovery,
   Stop while disconnected, target hold/reacquisition and bounded angles. Do
   not run a scripted sweep or recalibrate merely to verify restoration.

`make validate` is the hardware-free CPU/JS regression suite. GPU and physical
checks above are additional, not implied by CI. Retain actual results with each
new deployment lock. Restoring the previous verified bundle/image/config is the
rollback; never roll back physical calibration after the assembly changed.
