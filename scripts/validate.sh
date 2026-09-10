#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

bash -n scripts/*.sh
bash -n deploy/thor/run-container.sh deploy/startup/*.sh
python3 -m py_compile src/spring_turret/*.py tests/*.py
python3 tests/test-server.py
python3 tests/test-setup-recovery.py
python3 tests/test-wired-recovery.py
python3 tests/test-servo-feedback.py
python3 tests/test-isolation.py
python3 tests/test-detection.py
python3 tests/test-inference-pause.py
python3 tests/test-api-contract.py
python3 tests/test-app-release.py
python3 tests/test-detection-progress.py
python3 tests/test-gpu-backend.py
python3 tests/test-native.py
python3 tests/test-w8a8.py
python3 tests/test-w4a4.py
python3 tests/test-mask-profile.py
python3 tests/test-mask-artifacts.py
python3 tests/test-mask-output.py
python3 tests/test-mask-centroid.py
python3 tests/test-mask-postprocess.py
python3 tests/test-mask-png.py
python3 tests/test-mask-attention8.py
python3 tests/test-worker-protocol.py
python3 tests/test-prefetch.py
python3 tests/test-output-transfer.py
python3 tests/test-tracking.py
python3 tests/test-target-lock.py
python3 tests/test-temporal.py
python3 tests/test-native-tracking.py
python3 tests/test-v18-tracking.py
python3 tests/test-track-lifetime.py
python3 tests/test-shared-policy.py
python3 tests/test-policy-import.py
python3 tests/test-tracking-suppression.py
python3 tests/test-recovery.py
python3 tests/test-startup.py
python3 tests/test-deployment-bundle.py
python3 tests/test-tracking-graphs.py
python3 tests/test-tracking-compiler.py
python3 tests/test-tracking-masks.py
python3 tests/test-geometry.py
python3 tests/test-pose-history.py
python3 tests/test-bearing-filter.py
python3 -m py_compile tools/*.py deploy/startup/*.py deploy/startup/clean-boot/*.py
python3 tests/test-repository.py
node --check src/spring_turret/static/app.js
node --check src/spring_turret/static/dashboard.js
node --check src/spring_turret/static/kiosk-ready.js
node --check src/spring_turret/static/shared-controls.js
node tests/test-ui.cjs
node tests/test-shared-controls.cjs
node --check scripts/frontend.mjs
node --test tests/test-frontend.mjs

if command -v shellcheck >/dev/null; then
  shellcheck scripts/*.sh
fi

git diff --check
