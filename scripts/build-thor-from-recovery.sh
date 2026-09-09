#!/usr/bin/env bash
# Build only; never retag the recovery image, start a container, or touch hardware.
set -euo pipefail
base=spring-turret-demo:0.17.3-recovery-agxthor-5
expected=sha256:b86f5fe9fb21277367cd1da5f9aaf33f02e6f710388decbf0dd9a9fcaac19049
tag=${1:?Usage: build-thor-from-recovery.sh spring-turret-demo:NEW-TAG}
[[ "$tag" == spring-turret-demo:* && "$tag" != "$base" ]] || {
  echo 'Choose a new spring-turret-demo tag, not the recovery tag' >&2; exit 1;
}
[[ $(uname -m) == aarch64 ]] || { echo 'Build on the Thor (aarch64)' >&2; exit 1; }
actual=$(docker image inspect --format '{{.Id}}' "$base")
[[ "$actual" == "$expected" ]] || { echo 'Recovery image digest mismatch' >&2; exit 1; }
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
docker build --pull=false --network=none -f "$repo_root/deploy/repro/Dockerfile.thor" -t "$tag" "$repo_root"
docker image inspect --format '{{.Id}}' "$tag"
