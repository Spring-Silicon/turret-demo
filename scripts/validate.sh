#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

bash -n scripts/*.sh
python3 -m py_compile src/spring_turret/*.py tests/*.py
python3 tests/test-server.py
python3 tests/test-detection.py
python3 tests/test-gpu-backend.py
python3 tests/test-native.py
python3 tests/test-w8a8.py
python3 tests/test-worker-protocol.py
python3 tests/test-prefetch.py
python3 tests/test-tracking.py
python3 tests/test-geometry.py
python3 -m py_compile tools/*.py
python3 tests/test-repository.py
node --check src/spring_turret/static/app.js
node tests/test-ui.cjs

if command -v shellcheck >/dev/null; then
  shellcheck scripts/*.sh
fi

git diff --check
