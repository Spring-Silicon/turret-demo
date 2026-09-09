# Local frontend, remote hardware and inference

For the current unattended `spring-edge-turret` + `agxthor-5` deployment, see
[Automatic startup](startup.md). It hosts the combined page on the Arc and
opens fullscreen Firefox automatically; the Mac frontend services are disabled.
The deployment descriptions below also document separately retained older setups.

## Mac frontend: Thor-only, Arc-only or combined

The Mac can host any of the three views using the same frontend and API proxy:

```sh
make frontend-thor
make frontend-arc
make frontend-combined
```

Each command uses `http://127.0.0.1:8080/`; run one there at a time. Use
`FRONTEND_PORT=8081` to run an additional view alongside the background service.
`ARC_BACKEND` and `THOR_BACKEND` can override the corresponding remote URL.
The Mac Thor target is now `agxthor-5` (`http://100.88.90.48:8080`). To view
the older Thor 4 installation, use `THOR_BACKEND=http://100.97.193.16:8080`.
Equivalent Node flags are `--thor URL`, `--arc URL`, or both flags together;
`--backend URL` remains compatible. None require an SSH password at runtime:
the Mac hosts the UI and proxies the existing API over the tailnet. Existing
remote SSH credentials are unchanged; no new browser login is added.

The existing Mac LaunchAgent `com.springsilicon.turret-frontend.unified` is reused
for the currently selected view (Thor-only). Its historical label does not select
the mode; its `ProgramArguments` do. Source/UI changes require copying the script
and assets into `~/Library/Application Support/Spring Turret/frontend`, then
reloading the LaunchAgent. Starting a frontend never restarts a remote backend,
changes model/prompts or arms motors.

## Separate Arc-hosted deployment: Firefox, direct Ethernet to Thor

The side-by-side frontend runs on Arc (`spring-edge-2-1`) and is open in its
graphical Firefox session at `http://127.0.0.1:8081/`. Its two fixed backends are:

- Arc: `http://127.0.0.1:8080` (loopback).
- Thor: `http://192.168.249.2:8080` through Arc `enp8s0`, source `192.168.249.1`.

Both boxes have an autoconnecting NetworkManager profile named
`turret-direct-ethernet`: Arc `enp8s0` is `192.168.249.1/30`; Thor `enP2p1s0`
is `192.168.249.2/30`. Neither profile supplies a gateway or DNS/default route.
The existing Wi-Fi Internet/management paths are unchanged. Thor was configured
through the Ethernet link; its SSH key was verified against the existing
`agxthor-4` host key. A temporary, cable-only DHCP bootstrap was stopped after
both static profiles were activated; no DHCP/NAT sharing remains.

The frontend checks the cable's carrier/address and binds Thor sockets to the
wired source address. It has no tailnet/Wi-Fi fallback URL. Dedicated nftables
`inet turret_direct_link` output/forward rules on each box additionally block
packets to the peer's private address through any other interface, including
when a cable route disappears. The rules are reinstalled by a narrowly scoped
NetworkManager pre-up dispatcher; other firewall tables are not changed. The
configuration is in `deploy/direct-ethernet/` and installs as
`/etc/spring-turret-direct-link.nft` plus
`/etc/NetworkManager/dispatcher.d/pre-up.d/90-turret-direct-link`.

Arc's user service `spring-turret-frontend.service` starts at login. It uses an
isolated Node 22.23.2 runtime and frontend files under
`/home/spring/.local/share/spring-turret-frontend`, without modifying system
Node/Python, inference environments, or backend services. Its unit template is
`deploy/spring-turret-frontend.service`; logs are in the user journal.

Firefox uses a separate profile at
`/home/spring/snap/firefox/common/turret-demo-profile`. It does not modify the
normal Firefox profile. `spring-turret-demo-firefox.service` is a transient
user service for that graphical session. Moving there originally disabled the
Mac frontend; the Mac LaunchAgent is now independently enabled in Thor-only mode.

To reopen the page on Arc:

```sh
firefox --new-window http://127.0.0.1:8081/
```

Only frontend/network configuration was changed for this move. Model
optimization work can run independently; a temporarily unavailable backend
does not stop the local page or the other panel.

## Optional workstation deployment

Run the frontend on your workstation with Node.js 22 or newer; it uses only
built-in modules and needs no npm install, Python, model weights or GPU runtime.

```sh
node scripts/frontend.mjs --arc http://100.95.161.99:8080 --thor http://100.88.90.48:8080 --port 8080
```

Open http://127.0.0.1:8080/: Arc is on the left, Thor on the right, with the same
camera feeds, statistics and motor controls as the standalone pages. On narrow
windows the panels stack. One model selector and prompt editor sits above both
feeds, including a single Add button and Update prompts/Enter submission.
Counts are labeled Arc and Thor on each prompt row. A target-class button
selects that class on both devices; clicking an instance remains local to its
camera. Start/Stop, degree sliders, calibration and Escape stay per device.

Model choices must be configured on both backends. Explicit model changes and
prompt submissions fan out concurrently, with model then prompts ordered per
device. A failed peer is named in the shared error; successful peers are not
rolled back and failed commands are never automatically retried. Update prompts
can explicitly align differing settings again. Page load/reconnect only reads
settings and preserves drafts: it never resets a tracker or sends motor commands.

Both are mounted instances of the existing UI, not embedded remote pages.
Shadow roots isolate panel DOM IDs and styles; closures isolate motor commands,
frame/box state and FPS counters. Shared drafts are separate from either panel.
API/stream/image URLs use fixed
`/devices/arc/` or `/devices/thor/` prefixes. The local server strips only the
selected prefix and never infers a device for an unscoped API call. Connections
and command capacity are separate per backend. If one is offline, the other
remains usable. Escape stops only the panel containing keyboard focus.

Single-device mode remains available with `--backend http://HOST:8080` instead
of `--arc`/`--thor`. Restart the local frontend after editing its static assets.

## What runs where

- Workstation: HTML/CSS/JavaScript hosting, browser rendering and overlays,
  and a small same-origin streaming API proxy.
- Edge/Thor: camera capture, model inference, temporal state, servo control
  and calibration. The existing process-isolated HTTP viewer remains as the
  remote API/video transport adapter; the local frontend does not request its
  HTML, CSS or JavaScript.
- Requests and JPEG/metadata streams retain the existing API format. The proxy
  pipes bytes without decoding, re-encoding, resizing or collecting whole SSE
  events. Browser disconnect cancels its upstream stream. Backpressure stays
  in the existing disposable remote viewer, not the inference/control loop.
- The local process binds only `127.0.0.1`. Host and same-origin checks prevent
  arbitrary web pages from sending local motor commands. No password or login
  screen is added. The backend must be reachable over the workstation's tailnet.
- Starting, restarting, or quitting the frontend never sends a motor command,
  restarts a backend, changes prompts or resets tracking memory. Closing the
  frontend does **not** stop an armed turret; use explicit Stop before leaving
  if you want it stopped.
- Upstream failures produce a 502 error, never cached success. Failed commands
  are never retried automatically because their outcome may be unknown. Normal
  GET polling and EventSource reconnect behavior remains in the browser.

The existing remote URLs remain usable for diagnostics. This change adds no
remote deployment or restart and does not alter camera, model or servo settings.

## Workstation service details

User LaunchAgent `com.springsilicon.turret-frontend.unified` runs the selected
endpoint in the background, restarting after exit or login. Its property list
is in `~/Library/LaunchAgents`. It uses `/opt/homebrew/bin/node` and a deployed
copy of the frontend script and static assets in
`~/Library/Application Support/Spring Turret/frontend`.
This keeps background startup independent of access to the Documents checkout.
Logs are in `~/Library/Logs/SpringTurret`.

When returning to workstation hosting, re-enable the LaunchAgent if disabled.
After editing, copy the script and static assets to that deployed directory,
preserving their repository-relative paths. Then use
`launchctl kickstart -k gui/$(id -u)/com.springsilicon.turret-frontend.unified`
to reload. This restarts
only the local UI process. To disable it, `launchctl bootout` the corresponding
property list; remove or move that list if it should not load at next login.
The previous Arc/Thor LaunchAgents were unloaded and their property lists and
deployed assets retained under the workspace's
`work/local-frontend/unified-backup.BnpbyV` for recovery. Port 8081 is no longer used.

`node --test tests/test-frontend.mjs` checks local assets with an offline
backend, API/body/status preservation, byte-exact image delivery, immediate
SSE/MJPEG streaming, disconnect cleanup, no command retry, connection timeouts,
bounded commands, same-origin restrictions and route/backend validation.
It also verifies separate device routing, no implicit default device, and the
other panel remaining reachable when one backend fails. The UI tests mount two
real client instances and check independent motor commands, Escape, image paths
and SSE frames. `test-shared-controls.cjs` checks two-device model/prompt fan-out,
shared class selection, labeled counts, partial failures, common availability,
draft preservation and panels without duplicate controls. No real motor commands
are sent by these tests.
