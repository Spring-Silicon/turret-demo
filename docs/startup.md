# Automatic demo startup on Spring Edge Turret

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
- GDM automatically logs in as `spring`. This intentionally makes the desktop
  accessible to anyone physically at the machine without entering a password.
- `~/.config/autostart/spring-turret-kiosk.desktop` starts
  `spring-turret-kiosk.service` after the graphical session is ready. The launcher
  waits for the local page, then uses Firefox `--kiosk` with a dedicated profile
  at `~/snap/firefox/common/spring-turret-kiosk-profile`.
  The installed Firefox 147 is affected by an invisible-window Wayland kiosk
  issue ([Mozilla bug 2014372](https://bugzilla.mozilla.org/show_bug.cgi?id=2014372)),
  so this service alone sets `MOZ_ENABLE_WAYLAND=0`, `GDK_BACKEND=x11`, and the
  snap launcher's supported `DISABLE_WAYLAND=1` to use the existing XWayland
  session. GNOME and the GPU/inference stack are unchanged. The snap setting is
  necessary because its desktop launcher otherwise overrides `GDK_BACKEND`.
- Thor's existing system `spring-turret-demo.service`, Docker and its snap
  Tailscale service are enabled. No Thor model/container/runtime is replaced.
- Each host runs `spring-turret-inference-startup.service` once after its backend
  starts. It supplies `person` only when prompts are empty, so inference starts
  without a browser action. It preserves any already-selected prompts and never
  changes the model, target, or motor state. Readiness GETs can retry; the prompt
  POST is sent once and is never retried after an ambiguous failure.

The page does not wait for model compilation: each panel reconnects to its own
backend as it becomes available. Startup and viewer reconnects do not send
motor commands. Models and prompts use the backend's configured startup values;
servo zeros, geometry and angle limits are unchanged. The dedicated Firefox
profile skips welcome and pre-onboarding screens so first-run setup cannot
block the page. Motors remain unarmed
until Start is pressed. Closing the viewer does not stop an already armed motor.

Templates live in `deploy/startup/`. The kiosk deliberately starts through
desktop autostart, not the lingering user's boot target: Firefox needs the live
Wayland/display and session-bus environment.

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
