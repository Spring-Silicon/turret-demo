#!/usr/bin/env bash
# Installed as /opt/spring/turret-demo/run-container.sh on the Thor host.
# No privileged container or Docker socket exposure; only these two devices.
set -euo pipefail

camera=/dev/spring-turret-camera
servo=/dev/spring-turret-servo
test -c "$camera"
test -c "$servo"
exec /usr/bin/docker run --rm --name spring-turret-demo --init \
  --runtime=nvidia --network=host --shm-size=2g \
  --user=1000:1000 --group-add=44 --group-add=20 \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device="$camera:$camera" --device="$servo:$servo" \
  --mount=type=bind,src=/etc/spring-turret-demo.json,dst=/etc/spring-turret-demo.json,readonly \
  --mount=type=bind,src=/etc/spring-turret-geometry.json,dst=/etc/spring-turret-geometry.json,readonly \
  --mount=type=bind,src=/var/lib/spring-turret-demo,dst=/var/lib/spring-turret-demo \
  --mount=type=bind,src=/opt/spring/turret-demo/models,dst=/models,readonly \
  --mount=type=bind,src=/var/cache/spring-turret-demo,dst=/cache \
  --env=HOME=/cache --env=NVIDIA_VISIBLE_DEVICES=all \
  --env=NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  spring-turret-demo:0.16.7-tracking-masks
