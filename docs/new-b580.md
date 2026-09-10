# Bring up an identical B580 box (Prague)

This is the entry point for reproducing the current Arc demo on a second box.
Use the Git checkout **and the private runtime archive**. Git contains the app,
exporter, boot scripts and documentation; the archive supplies the custom Torch
installation, native model libraries/weights and the saved SAM mask stages.
Installing the public Torch requirements alone does not reproduce this runtime.
For the two-box demo, also follow [Accompanying Thor](#accompanying-thor) below;
Thor uses a separate private archive and its own CUDA runtime.

## Prague preflight, 2026-09-10

Read-only SSH inspection of `spring@prague` (`100.70.121.32`) confirmed:

| Item | Observed |
| --- | --- |
| OS / kernel | Spring Edge 0.1.2, Ubuntu 24.04.4; `7.0.0-31-generic` |
| Account | `spring`, UID/GID 1000; video, render and dialout membership |
| Board / GPU | B550M WiFi; B580 `8086:e20b`, ASRock `1849:6021`, `xe` driver |
| GPU prefetchable BAR | 16 GiB |
| Level Zero / compute runtime | 1.32.0 / 26.31.39395.13 |
| Python | 3.12.3 |
| Firefox | 147.0.3-1, revision 7766; no hold yet |
| Root filesystem | 94 GiB, 89 GiB used, only 279 MiB available |
| Persistent data filesystem | `/var/lib/spring-data`, 242 GiB available |
| Ethernet | `enp7s0`, MAC `9c:6b:00:da:95:66`; currently the management connection |
| Turret services | No user `spring-turret*` units installed |

Prague already has other model/development installations. Nothing was removed,
relocated, installed or restarted there during this audit. This is a preflight,
not a successful fresh-machine restore or a new GPU/physical qualification.
Its hardware/OS match the recorded foundation; reimaging is not indicated by
these checks. Missing host utilities still need installation.

## Obtain the exact runtime

The current source is on `user/demo-changes`. The refreshed Arc archive is
`/var/lib/spring-data/turret-recovery/recovery-20260910-arc` on
`spring-edge-turret` (`100.68.74.32`). Use the corresponding public integrity lock
in `deploy/repro/arc-20260910.json` and the archive's `inventory.json`.
The older `recovery-20260909-arc-v2` is a valid historical foundation, but predates
quiet boot and the saved SAM stages; it is not the full current demo.

The refreshed archive contains both venvs; configured model bundles and external
symlink dependencies; the 408 MiB saved mask package; backend/frontend/kiosk
sources and units; the graceful inference shutdown drop-in; and the quiet-session
and Plymouth assets. It also preserves the currently installed v18 tracking
runtime if configured. Installed platform patches can differ from the checkout:
the archive hashes pin those exact bytes. Do not replace the whole archived venv
with a wheel and expect its saved mask package to remain compatible.

Transfer the archive over authenticated SSH to the **data partition** on the new
box. Obtain its checksum/size from the reviewed public lock as well as checking:

```sh
python3 tools/deployment_bundle.py verify /path/to/recovery-20260910-arc
```

Keep the archive private; it includes licensed model weights. SSH/Tailscale
identity, login credentials, browser cookies, live motor state and ordinary
compiler caches are excluded. The packaged kernel cache is included inside the
saved mask package. EFI slot state, disk UUIDs and the source initramfs must be
recreated for the target rather than copied from the source machine.

## Plan storage and existing-file conflicts first

Prague's SSD is not full: the root partition is. The large existing directories
include `so101-vla` (22 GiB), `.local/share/graphs-worktrees` (17 GiB),
`.local/share/sam31-israel` (17 GiB), and `.venvs/sam31-turret-xpu` (7.1 GiB).
Those are existing projects, not disposable system logs. Reclaiming or relocating
one requires checking its users and preserving its dependencies.

Put the new private archive, staging tree, runtime and caches under
`/var/lib/spring-data`. Preserve the original runtime paths with persistent bind
mounts; native imports, venv launchers and the artifact contract expect them.
Suggested source-to-target mapping:

| Directory under `/var/lib/spring-data/turret-host/` | Original path to preserve |
| --- | --- |
| `runtime` | `/home/spring/.local/share/turret-demo` |
| `frontend` | `/home/spring/.local/share/spring-turret-frontend` |
| `cache` | `/home/spring/.cache/turret-demo` |
| `sam3_1` | `/home/spring/sam3_1` |

Do not bind over a populated target without first preserving and reconciling it.
Prague already has `/home/spring/sam3_1`; it must not be overwritten as if the
machine were blank. Compare the archive inventory with existing paths, including
Node's real symlink target and the standalone Python headers. Stage extraction
on the data partition to inspect these conflicts before restoring selected paths.
On a genuinely fresh/retired installation, use the full extraction procedure in
[reproduce.md](reproduce.md#restore-on-a-matching-fresh-or-explicitly-retired-installation).

For each reconciled directory, create the data directory and an empty target,
bind-mount it, and add the explicit `none bind 0 0` entry to `/etc/fstab`. Validate
with `findmnt --verify`, and confirm `findmnt -T ORIGINAL_PATH` resolves to the
data filesystem after reboot. Runtime mounts must complete before user services
start. Keep enough root space for package operations and rebuilding initramfs;
using a data-backed runtime alone does not repair the existing nearly-full root.

## Restore services, host dependencies and peripherals

Follow the matching foundation/permissions steps in [reproduce.md](reproduce.md).
Preserve Python 3.12 and the archived custom inference venv and OpenCL environment.
The backend unit sets `LD_LIBRARY_PATH`, both OpenCL loader variables and `CPATH`;
restoring only Python files misses these requirements. Host tools include zstd,
GStreamer with the good plugins, compiler tools, GDM/Xorg, Python GI/GTK3/GDK3,
Openbox, desktop-file utilities and the existing Spring Plymouth/GRUB tools.
Install only missing dependencies; replacing the graphics stack is unnecessary
when its pinned versions already match.

The archived config is for the original physical assembly. Set Prague's actual
camera path and controller udev identity before starting it. Retain calibration
only when moving that exact assembly (controller serial `5B3D045331`, its mount
and camera). A new assembly needs its own zeros and camera geometry; keep it
uncalibrated until commissioned. The September 10 Arc config has P=400, I=0,
D=0, with zero velocity/acceleration profiles. Older locks record earlier gains.
See [control tuning](control-tuning.md) and [geometry qualification](geometry-qualification.md).

For a standalone B580 demo, use the restored frontend unit with an override:

```ini
[Service]
ExecStart=
ExecStart=%h/.local/share/spring-turret-frontend/runtime/bin/node %h/.local/share/spring-turret-frontend/scripts/frontend.mjs --arc http://127.0.0.1:8080 --port 8081
```

For a paired Thor demo, configure a **dedicated** cable/interface and adapt the
firewall, NetworkManager profile and frontend flags together as described in
[startup.md](startup.md#direct-cable-configuration). Prague currently uses
`enp7s0` for management; applying the source machine's dedicated-link configuration
to it would interrupt that connection. Establish an independent management path
or another dedicated NIC before repurposing it. Do not copy the old NIC MAC or
silently send the video over Tailscale as a substitute for the wired path.

Retain Prague's own hostname, SSH keys, Tailscale enrollment and management
network configuration. These identities are not part of the runtime restore.

## Quiet boot and browser

Before starting/restarting the viewer, explicitly apply the existing update policy:

```sh
sudo snap refresh --hold=forever firefox
snap list firefox
```

The hold must show `held`. Do not run a targeted Firefox refresh; it overrides a
hold. Prague's installed Firefox 147 and the source's Firefox 155 have different
revisions; preserve Prague's version for the first kiosk check. A browser update
is not a prerequisite for copying the demo.

Restore the dedicated browser profile's `user.js`, all frontend scripts/assets,
the four user units and the backend's `graceful-stop.conf`/`opencl.conf` drop-ins.
The archived quiet-session files include Openbox, GDM selection, the session
launcher and native-logo service. After resolving storage, installing dependencies
and restoring config, apply the target's quiet boot using a clean reviewed checkout:

```sh
sudo python3 deploy/startup/install-clean-boot.py "$PWD" "$(git rev-parse HEAD)"
sudo loginctl enable-linger spring
sudo systemctl daemon-reload
sudo systemctl enable spring-native-logo.service
systemctl --user daemon-reload
systemctl --user enable spring-turret-demo.service spring-turret-frontend.service
systemctl --user enable spring-turret-inference-startup.service
```

The installer checks the expected board, framebuffer and original Spring A/B
GRUB format; it backs up changed files and rebuilds the target initramfs. If it
rejects an already-customized boot layout, inspect the layout instead of forcing
it. Do not copy the source machine's EFI `grubenv` or initramfs. The kiosk service
is static and starts through the graphical session; do not enable it as a
headless boot service. The inference startup helper supplies `person` when no
prompt exists and never arms the motors. See [startup.md](startup.md) for the
full sequence and rollback. ASRock POST branding and HDMI banners remain
firmware/display settings.

## Saved SAM stages and acceptance

Keep `inference.sam31_mask_compiled_bundle` pointing at the restored
`/home/spring/.local/share/turret-demo/sam-artifacts-20260909/package` and the
configured default model `sam3.1-mask`. The loader rejects incompatible Torch,
Python, device, source, model manifest or confidence settings. If any change,
export and qualify a new package using [sam31-mask-artifacts.md](sam31-mask-artifacts.md);
never edit the manifest to bypass a mismatch. The separate temporal Tracking
profiles do not use this mask startup optimization.

Start the backend/frontend and run the startup helper after all paths and device
identities are reconciled. With the camera connected, check advancing processed
frame sequences, `compiled_mask_artifacts: true`, `sycl_graph: true`, and
`mask_postprocess_bitwise_equal: true` from `/api/status`. Confirm frontend
routing and a complete processed image/mask pair. Qualify model/prompt switches
and the actual physical assembly as described in the recovery guide.

Allow an initial qualification run to populate ordinary runtime caches, then
reboot and time the fully automatic path. The source measured 44.6 s from kernel
boot and 32.0 s from the kiosk painting to processed video with persistent caches.
An empty-cache first use took longer; these figures are a source reference, not
a Prague measurement. Acceptance requires the quiet session, loading spinner,
correct camera/masks, default inference, Firefox hold, and motors remaining off
until Start. No successful Prague installation/reboot test has yet been claimed.

## Accompanying Thor

The B580 archive restores the Arc side. The paired Thor requires the **separate**
`recovery-20260909-thor` archive, containing `runtime.tar.zst` (3.24 GB) and
`container.tar.zst` (11.30 GB), plus `inventory.json` and `SHA256SUMS`. It is stored
at `/home/spring/recovery-20260909-thor` on the original Thor and mirrored at the
same path on the original Arc. Its integrity lock is
[thor-20260909.json](../deploy/repro/thor-20260909.json). Allow additional disk space
for extraction, loaded Docker layers, model files and writable caches.

| What the new Thor needs | Where it comes from |
| --- | --- |
| Compatible Jetson OS/kernel and NVIDIA driver | L4T R39.2.1 foundation; not included in the container |
| Docker and NVIDIA container toolkit | Host packages pinned in the Thor lock |
| CUDA, Torch, SAM code and frozen runtime patches | Private `container.tar.zst` |
| SAM checkpoint, launcher, config, udev rules and startup services | Private `runtime.tar.zst` |
| Current application behavior | Reviewed Git revision, applied with the recovery-image overlay builder |
| Camera, controller identity, calibration and dedicated Ethernet | Configure for the new physical assembly and NIC |

Install in this order:

1. Check the new Thor's OS/BSP, architecture, account IDs, storage and peripherals
   against [Frozen foundations](reproduce.md#frozen-foundations). Flash the pinned
   BSP only if needed, using NVIDIA's procedure for that specific Thor board.
2. Verify the private bundle with `tools/deployment_bundle.py verify`, compare its
   inventory checksum with the public lock, then follow
   [Restore on a matching installation](reproduce.md#restore-on-a-matching-fresh-or-explicitly-retired-installation)
   and the [Thor restore steps](reproduce.md#thor). These restore host files,
   load the exact Docker image, and configure the NVIDIA runtime.
3. Apply the reviewed application revision using
   [Updating the application from Git](reproduce.md#updating-the-application-from-git)
   and `scripts/build-thor-from-recovery.sh`. Keep the immutable recovery image;
   select the new image in the active launcher and validate it before use. The
   September 9 archive is the frozen foundation, not a claim that later app
   changes are already inside that image.
4. Reconcile camera/servo paths and calibration with the actual assembly. Retain
   serial `5B3D044488`'s zeros only when moving that same assembly. Set up the
   [dedicated Ethernet link](reproduce.md#direct-ethernet-no-wi-fi-fallback), adapting
   NIC names/MACs and preserving the new hosts' management networking.
5. Enable Docker, the system `spring-turret-demo.service`, and the system
   `spring-turret-inference-startup.service` as shown in the Thor steps. Check
   the supported power mode and cooling. Arc hosts the combined Firefox view;
   Thor needs no local Firefox, Node or kiosk installation for this arrangement.
6. With both devices connected, run [acceptance](reproduce.md#acceptance-and-rollback):
   select the supported profiles, verify advancing processed frames/masks and
   CUDA graph execution on Thor, check both frontend proxy routes, then reboot
   both and confirm default inference starts automatically with motors off.

Thor retains its own compilation/CUDA graph startup path. Arc's saved XPU mask
package does not accelerate it. Both devices must be independently qualified;
no clean-board Thor restore has been performed as part of the Prague audit.
