# Automatic demo startup on Spring Edge Turret

For fresh installation, exact runtime/model pins and private recovery archives,
see [Reproduce both devices](reproduce.md) and the [new B580 / Prague guide](new-b580.md).

Deployment: `spring-edge-turret` (Arc, `100.68.74.32`) and `agxthor-5`
(`100.88.90.48`). These are separate from the older direct-Ethernet deployment
on `spring-edge-2-1` and `agxthor-4`.

The Arc hosts the combined page at `http://127.0.0.1:8081/`. Its Arc panel
connects to its own backend on port 8080; its Thor panel connects **only over
the direct Ethernet cable** to `http://192.168.249.2:8080`. The frontend binds
source `192.168.249.1` on Arc `enp7s0`; it does not fall back to Wi-Fi/Tailscale.

## Direct cable configuration

Both hosts have a persistent, autoconnecting NetworkManager profile named
`turret-direct-ethernet`, with priority 999 and these verified NIC bindings:

| Host | Interface | MAC | Address |
|---|---|---|---|
| spring-edge-turret | enp7s0 | 9c:6b:00:dd:04:2e | 192.168.249.1/30 |
| agxthor-5 | enP2p1s0 | 44:49:c0:3c:fc:5c | 192.168.249.2/30 |

The link negotiates 1 Gbps. It supplies no gateway or DNS, has
`ipv4.never-default=yes`, and disables IPv6 on this dedicated link. Existing
Internet/management networking is unchanged. Do not reuse interface names or
MACs blindly on a replacement board; verify them first.

The frontend's `--thor-interface enp7s0 --thor-local-address 192.168.249.1`
options require the carrier/address and bind every Thor connection to the wired
source. Dedicated `inet turret_direct_link` firewall tables on both hosts also
block the peer's private address through any other interface. Rules are retained
when the link drops, and a NetworkManager pre-up dispatcher reinstalls them at
activation/boot. Only this dedicated table is managed; existing firewall tables
are unchanged. Installed files:

- `/etc/spring-turret-direct-link.nft`
- `/etc/NetworkManager/dispatcher.d/pre-up.d/90-turret-direct-link`

Templates are in `deploy/direct-ethernet/`; `arc.nft` targets the current Arc's
`enp7s0`, not the older `spring-edge-2-1` machine's `enp8s0`.

2026-09-08 verification: matching peer MACs, direct routes in both directions,
0.34 ms mean ping, real HTTP/frame packets on `enp7s0`, and frontend TCP sockets
only from `192.168.249.1` to `192.168.249.2:8080`. The preceding deployment used
the tailnet URL over Wi-Fi and showed a roughly 1.8 MB Thor send queue. Switching
the frontend cleared that queue without restarting either inference backend or
changing the active model, prompts, calibration or motor state.

## Boot sequence

- Arc's existing user backend `spring-turret-demo.service` is enabled.
  `loginctl enable-linger spring` lets it start at boot without a desktop login.
- Arc's user `spring-turret-frontend.service` is enabled for `default.target`.
  It runs an isolated, checksum-verified Node 22.23.2 runtime under
  `/home/spring/.local/share/spring-turret-frontend`, not system Node/Python.
- GDM automatically logs in as `spring` to the **Spring Turret Demo** X11
  session, selected through AccountsService and `/etc/gdm3/custom.conf`.
  Openbox manages only the kiosk windows; no GNOME desktop, dock, overview, or
  generic desktop autostarts run in this session. The root window is black.
- `kiosk-session.sh` imports the actual display/authentication environment and
  starts `spring-turret-kiosk.service`. The launcher displays the same native-size
  Spring logo as Plymouth, opens Firefox with its dedicated profile, and keeps
  the logo above the browser until the local page has painted both styled panels.
  The per-launch `/kiosk-ready/<token>` handshake stays within the frontend;
  it never contacts either inference backend or sends motor commands.
- Firefox uses X11 (`MOZ_ENABLE_WAYLAND=0`, `GDK_BACKEND=x11`,
  `DISABLE_WAYLAND=1`). This also avoids the installed Firefox's invisible
  Wayland kiosk window issue. The dedicated profile at
  `~/snap/firefox/common/spring-turret-kiosk-profile` skips onboarding and crash
  restoration. The browser restarts if it exits; stopping its service is explicit.
- Thor's existing system `spring-turret-demo.service`, Docker and its snap
  Tailscale service are enabled. No Thor model/container/runtime is replaced.
- Each host runs `spring-turret-inference-startup.service` once after its backend
  starts. It supplies `person` only when prompts are empty, so inference starts
  without a browser action. It preserves any already-selected prompts and never
  changes the model, target, or motor state. Readiness GETs can retry; the prompt
  POST is sent once and is never retried after an ambiguous failure.

The page does not wait for model compilation: each panel reconnects to its own
backend as it becomes available. Before the first completed image/mask pair,
Arc shows a circular spinner with **Spring is Coming** and Thor shows one with
**Thor is Loading**. Placeholder alt text, FPS placeholders, and routine model
preparation/connection messages are hidden. The loader disappears only after
painting a complete pair. Existing images remain visible during later model
changes and reconnects. Explicit command failures and operational errors remain
available after startup. Startup and viewer reconnects do not send
motor commands. Models and prompts use the backend's configured startup values;
servo zeros, geometry and angle limits are unchanged. The dedicated Firefox
profile skips welcome and pre-onboarding screens so first-run setup cannot
block the page. Motors remain unarmed
until Start is pressed. Closing the viewer does not stop an already armed motor.

Templates live in `deploy/startup/`. The kiosk starts from the dedicated GDM
session, after its display is available; the lingering user's boot target still
starts only the backend and frontend. The old GNOME `.desktop` autostart remains
as a fallback if an operator explicitly selects the Ubuntu session.

## Inspection and control

On Arc:

```sh
systemctl --user status spring-turret-demo spring-turret-frontend spring-turret-kiosk
journalctl --user -b -u spring-turret-kiosk -u spring-turret-frontend
curl http://127.0.0.1:8081/frontend-config
curl http://127.0.0.1:8081/devices/arc/api/status
curl http://127.0.0.1:8081/devices/thor/api/status
```

On Thor:

```sh
systemctl status spring-turret-demo
journalctl -b -u spring-turret-demo
```

Stop/reopen only the Arc browser with
`systemctl --user stop/start spring-turret-kiosk` (use either verb separately).
The normal Firefox profile is not modified. Single-device frontend modes remain
available through `scripts/frontend.mjs --arc URL` or `--thor URL`.

The Mac frontend LaunchAgents `com.springsilicon.turret-frontend.edge` and
`com.springsilicon.turret-frontend.unified` are unloaded and disabled; their
files are retained. Re-enabling them is a separate explicit action, not needed
by either remote backend or the Arc's combined frontend.

To undo automatic desktop login, restore the Arc's original GDM file from
`/var/backups/spring-turret-startup-20260908/gdm.conf` to
`/etc/gdm3/custom.conf` and reboot. To stop browser autostart, disable its
`.desktop` entry. Backend boot startup can remain enabled independently.

## Quiet Arc boot profile (September 9)

Arc boots through `/boot/efi/EFI/spring/grub.cfg`, independently of the timeout
in `/etc/default/grub`. The installed profile hides that A/B menu and boots its
selected healthy slot immediately. A/B trial bits, health selection, slot roots,
and initrd paths are preserved. The GRUB background is removed and routine kernel
status/cursor output is suppressed; diagnostics remain in the journal/serial log.

Plymouth initially paints black. `spring-native-logo.service`, ordered before
GDM, reveals the existing 610×200 logo only once `xedrmfb` reports at least
1920×1080. It does not load a different graphics driver or change GPU firmware.
The service has a bounded wait so a disconnected display cannot prevent login.
The custom script is also rebuilt into the active kernel's initramfs. The GTK
kiosk cover uses that exact logo at its native pixel size on black.

`deploy/startup/install-clean-boot.py REPO COMMIT` installs this specific Arc
profile as root after checking the clean checkout and expected hardware. It
backs up touched files, the original initramfs, and AccountsService preferences
under `/var/backups/spring-clean-boot-TIMESTAMP`, validates GRUB/unit/desktop
syntax, and preserves A/B environment and inference/motor configuration bytes.
Installation does not reboot or restart GDM. Frontend assets and the user kiosk
launcher must be installed from the same commit before starting the new session.
The required additional package is `openbox`; GTK3/Python GI/Xorg were already
installed. No GPU or X server packages were upgraded.

Two stages originate outside the operating system: the motherboard's ASRock
POST logo and the display's HDMI input banner. This B550M WiFi exposes no Linux
firmware-attributes interface for its logo setting. Disable **Full Screen Logo**
in the firmware Boot settings; inspect the physical display's OSD controls for
its input banner. Neither is controlled by GRUB, Plymouth, or Firefox. Software
changes alone cannot guarantee a fully black screen before the OS runs.

To restore the Ubuntu desktop, use AccountsService to set `XSession`/`Session`
back to their saved values and restore `SessionType`, GDM config and `.dmrc`
from the backup, then restart GDM or reboot. Restore the backed-up GRUB/Plymouth
files and initramfs to undo the boot visuals. Disable `spring-native-logo.service`
if it was not enabled before installation. The saved receipt records exact paths
and hashes; do not replace the current A/B grubenv with an old copy.

## Firefox updates require an explicit action

Arc's Firefox snap is held indefinitely with
`sudo snap refresh --hold=forever firefox`. Verify `snap list firefox` reports
`held` before any viewer maintenance. The hold survives browser restarts and
boots. A targeted `snap refresh firefox` can override it, so do not issue that
command or remove the hold without an explicit user request.

On September 9, Snap pre-downloaded revision8863 at12:02 EDT and applied it
at12:36 when a viewer restart released revision7766. Firefox changed from147.0.3
to155.0.1 even though the deployment did not request a browser update. The hold
was added following the user's explicit request to prevent a repeat. The active
browser was retained; no downgrade or browser restart was needed for the hold.
The new launcher requires both GTK3 and GDK3 explicitly and was verified on
Firefox155 in a separate X11/Openbox session before staging it for the next boot.
