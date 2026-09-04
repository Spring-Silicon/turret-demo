#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

bash -n scripts/*.sh
python3 -m py_compile src/spring_turret/server.py tests/*.py
python3 tests/test-server.py
python3 tests/test-repository.py
node --check src/spring_turret/static/app.js

if command -v shellcheck >/dev/null; then
  shellcheck scripts/*.sh
fi

git diff --check
