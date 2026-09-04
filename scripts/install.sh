#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ${EUID} -ne 0 ]]; then
  echo "install.sh must run as root" >&2
  exit 1
fi
if ! command -v uv >/dev/null; then
  echo "uv is required" >&2
  exit 1
fi

apt-get update
apt-get install -y --no-install-recommends --no-upgrade \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-tools \
  udev

if ! getent passwd spring-turret >/dev/null; then
  useradd --system --no-create-home --home-dir /nonexistent \
    --shell /usr/sbin/nologin --groups dialout,video spring-turret
fi
usermod -a -G dialout,video spring-turret

install -d -o root -g root -m 0755 \
  /opt/spring/turret-demo \
  /opt/spring/turret-demo/static

UV_PROJECT_ENVIRONMENT=/opt/spring/turret-demo/venv \
  uv sync --project "${repo_root}" --frozen --no-dev --no-editable

install -o root -g root -m 0644 \
  "${repo_root}"/src/spring_turret/static/* \
  /opt/spring/turret-demo/static/
install -o root -g spring-turret -m 0640 \
  "${repo_root}/config/spring-turret-demo.json" \
  /etc/spring-turret-demo.json
install -o root -g root -m 0644 \
  "${repo_root}/deploy/99-spring-turret.rules" \
  /etc/udev/rules.d/99-spring-turret.rules
install -o root -g root -m 0644 \
  "${repo_root}/deploy/spring-turret-demo.service" \
  /etc/systemd/system/spring-turret-demo.service

rm -f -- /var/lib/spring-data/identity/turret-demo/http-token
udevadm control --reload-rules
udevadm trigger --action=change --subsystem-match=video4linux
udevadm trigger --action=change --subsystem-match=tty
udevadm settle
systemctl daemon-reload
systemctl enable --now spring-turret-demo.service
