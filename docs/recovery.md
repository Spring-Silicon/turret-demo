# Start/Stop and device recovery

## Initial setup and damaged settings

Initial setup must be reachable through both the individual and combined pages:

1. Connect both commissioned servos. Leave motors stopped, support/position the
   assembly at its intended X/Y zero, and click **Set zeros** once.
2. Press Start/Enter to track the selected class. If no class is selected, press
   Enter in its prompt input to select it and start.

There are no zero-setting dialogs or separate mapping-confirmation steps.
Zeroing retains the selected class, drops old instance/coordinate state, and
does not start motors. A successful zero save survives restarting
with `servo.calibrated: false` in the base config. Existing version-1 zeros retain
their legacy behavior and do not certify a new uncommissioned installation.
The live model/mode/unique-ID, torque-off, stationary-encoder and angle checks
remain; duplicate IDs or wrongly programmed modes require hardware commissioning,
not disabling checks. Disconnected controls explain what must be reconnected.

| Failure | Recovery |
| --- | --- |
| Missing zero directory | Created on save; writable controller-bound user-state default if no path configured |
| Corrupt/wrong-device zeros | API stays up, Start blocked; Set zeros after physically positioning the axes |
| Corrupt gain overrides | API stays up, Start blocked; Discard invalid saved gains while stopped |
| Obsolete tracking confirmation | Ignored; cannot block tracking or Start |
| Missing/invalid geometry file | Inference/viewer remain available; restore file at the reported path, retried every two seconds |
| Storage full/read-only | Command reports the error, keeps prior state and unlocks controls for retry after storage is repaired |
| Camera/worker fails before first result | Error is visible immediately; no dependency on receiving a camera frame |
| Backend/snapshot unavailable | Setup actions disabled with an explicit reason; Stop intent is not falsely reported as acknowledged |
| One combined-page backend offline | The online panel's shared model/prompt/target controls remain usable; offline peer is explicitly reported as skipped, never auto-replayed on reconnection |

Explicitly replacing invalid saved data retains a sibling `.invalid-<timestamp>`
backup. No invalid geometry silently falls back to approximate motor tracking.
Directions, limits, model selection and gains are not changed by zeroing.
The regression suite exercises the first-install → Set zeros → Start → restart path,
invalid files, failed writes, stopped/armed/offline states, and proxy/UI actions
using simulated devices without physical motor writes.

On a direct-Ethernet frontend, prefer `--thor-mac <wired-adapter-MAC>` with
`--thor-local-address 192.168.249.1` instead of a PCI-numbered `--thor-interface`.
The adapter is resolved on every request; a missing adapter/address/carrier
still refuses the Thor connection, with no Wi-Fi fallback. Match the
NetworkManager direct-link profile by the same Ethernet MAC (leave
`connection.interface-name` empty). `deploy/startup/wired-firewall.py` accepts a
root-owned JSON file containing `mac` and `peer` and atomically refreshes only
the `inet turret_direct_link` output/forward peer guard. Run it before networking
at boot and from the direct connection's pre-up dispatcher. It rejects the peer
over every route if the configured adapter is absent. This avoids hard-coded
interface names breaking after BIOS/PCI topology changes.

Control-only deployment must preserve model-worker files from the installed
runtime bundle. In particular, compiled Arc mask packages fingerprint
`sam31_graph.py` and other worker sources: copying a newer repository's entire
Python package over them can invalidate otherwise unchanged compiled artifacts.
Keep those files/bundles intact and verify running processed frames after the
backend reload, not only a successful HTTP health check.

## Runtime recovery

Start is a backend-owned intent (`servo.run_requested`) separate from actual
torque (`servo.armed`). Losing USB, a read failure, or a temporary device fault
removes torque where reachable but does not clear Start or the selected class.
The monitor reopens the configured bus, checks both model IDs, modes, feedback
mapping, hardware health and calibrated bounds, and enables torque at the freshly
measured position. No zeroing, stale move, or previous goal is replayed. Tracking
waits for a new camera frame and encoder history after rearming.

Stop (button or Escape) clears the request before device I/O, even when unplugged.
Reconnect after Stop stays stopped. Enter starts; in a prompt input it first
applies the prompts, then starts the affected panel(s). Repeated Enter while
already running does not reset control. Recalibrate remains available only after
Stop and executes immediately on click. Start is not implicitly requested by
opening/reloading a browser or by starting a fresh backend process.

Camera capture detects EOF and five-second stalls, reaps only its capture child,
and retries every two seconds. A new capture generation resets temporal model
sessions. Inference workers retry failures with a bounded 2–30 second backoff,
keeping the selected model/prompts. Model accuracy checks still reject invalid
outputs. Hardware faults never authorize driving an unverified/wrong motor.
Model/prompt changes discard old instance IDs but preserve a still-applied target
class; they do not clear Start. Removing that class or manually moving a slider
still explicitly changes tracking intent.

Thor's container reads the host's live `/dev` directory at `/host/dev` read-only;
its device cgroup grants read/write only V4L2 (81) and USB ACM serial (166) in
addition to the existing NVIDIA runtime devices. Exact-serial udev aliases select
the camera/controller. It does not use `--privileged`, `mknod`, or a Wi-Fi fallback.
This avoids Docker's static `--device` nodes becoming stale after hot-plug.

## Verification

Both live backends passed targeted USB deauthorization/reauthorization tests
(software disconnect/reconnect, not physical cable removal or motor power cycling):

| Host | Camera recovery | Servo recovery |
| --- | ---: | ---: |
| spring-edge-turret (Arc) | 3.11 s | 2.79 s |
| agxthor-5 (Thor) | 3.06 s | 2.28 s |

Backend and model-worker PIDs stayed unchanged during these tests. Start and the
selected class survived disconnection; Stop during disconnection prevented
automatic rearming after reconnection, and explicit Start resumed operation.
Saved zero offsets and angle limits were unchanged. Processed JPEG output and
the served UI assets were checked afterward. CPU/JavaScript validation includes
stalled capture recovery, worker retry, retained tracking intent, persistent
motor faults, Stop while offline, and Enter handling.

An independent Arc Intel GPU-driver hang occurred during the deployment reload,
before USB testing. Replacing the stuck inference worker recovered inference;
the underlying driver reload/teardown issue is not fixed by these changes.
