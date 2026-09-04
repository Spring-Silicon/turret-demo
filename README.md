# Spring turret demo

Minimal camera feed and servo controls for the Spring Edge turret demonstration.

## Current hardware status

- Arducam 1080P Low Light (`0c45:0261`, serial `UC684`): 1280x720 MJPEG at
  30 fps verified on `spring-edge-2`.
- USB Single Serial adapter (`1a86:55d3`, serial `5B61036033`): enumeration and
  stable device naming verified.
- ROBOTIS DYNAMIXEL XL330-M288-T: model 1200, firmware 53, Protocol 2.0, ID 1
  at 57,600 baud. Read-only PING and position are verified; motion is not yet
  qualified.

The service starts disarmed, sends no startup movement, rejects positions outside
1536 through 2560, and rejects every movement until the operator selects **Arm**.
Communication failure clears the armed state. Service shutdown attempts to turn
torque off. Arming first writes the current position as the goal, then enables
torque, preventing an immediate jump on Arm.

## UI

Open `http://HOST:8080/`. The UI contains only the live feed, camera/servo state,
position, bounded jog controls, Arm, Stop, and the current hardware error.

There is no password or application-level access control. Run it only on an
isolated demo LAN. The software Stop is not an emergency stop; keep a physical
power disconnect available.

## Run

The OS is responsible for packages, permissions, stable device paths, and
service management. The matching integration lives in
[`Spring-Silicon/edge-image#19`](https://github.com/Spring-Silicon/edge-image/pull/19).

```bash
uv sync --frozen
uv run spring-turret --config config/spring-turret-demo.json
```

The configured `/dev/spring-turret-camera` and `/dev/spring-turret-servo` paths
must already exist and be accessible to the process. Check the live state:

```bash
curl http://127.0.0.1:8080/api/status
```

## API

- `GET /stream.mjpg`
- `GET /api/status`
- `POST /api/servo/arm`
- `POST /api/servo/disable`
- `POST /api/servo/center`
- `POST /api/servo/position` with `{"position": INTEGER}`

## Validate

```bash
make validate
```

## Servo qualification still required

Before changing `hardware.json` to motion-qualified, confirm the actuator model,
electrical interface, supply voltage, ID, and baud rate. Then verify read-only
PING/position, mechanical center with linkage disconnected, conservative limits,
Stop under motion, communication-loss behavior, current, and temperature.
