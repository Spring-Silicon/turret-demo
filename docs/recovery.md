# Start/Stop and device recovery

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
Stop and retains its explicit confirmation. Start is not implicitly requested by
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
