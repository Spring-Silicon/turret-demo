# Git-pinned application releases

Use this for application-only updates to the existing Vegas Arc + agxthor-2
pair. Do not copy a dirty original checkout or import experimental model bundles.
The OS/runtime/bootstrap documentation remains in [reproduction](reproduce.md).

From a reviewed, committed, clean checkout, build two releases with a full
40-character commit ID:

```sh
python3 tools/build_app_release.py --role arc --commit FULL_COMMIT --output /absolute/new/arc-release
python3 tools/build_app_release.py --role thor --commit FULL_COMMIT --output /absolute/new/thor-release
```

The build interpreter needs setuptools, wheel and pip. Building uses no package
index and resolves no application dependencies. Each output has an application
wheel, frontend source and a `release.json` with the Git commit and file hashes.
The Arc release preserves the two qualified runtime source overlays in
`deploy/overlays/arc-20260910/`. Their SHA-256 checks prevent invalidating the
existing source-pinned offline mask artifacts or changing the frozen Box adapter.
Do not use the ordinary generic wheel in their place on this installation.

Before installing, record API selections/Start intent and back up the installed
package or container tag, launcher, frontend, device config, zeros, geometry and
gain overrides. Verify all release hashes and the offline mask manifest's source
hashes against the Arc release. Transfer the same Git commit to the device source
checkout (fetch or Git bundle); require a clean tree and exact HEAD.

For Arc, stop only the backend user service, then use `uv pip install --python`
with the existing backend venv, `--no-index --no-deps --reinstall` and the exact
release wheel. Do not replace its separate inference venv. Copy the release's
frontend script/static assets into the existing on-device frontend installation.

For Thor, build `deploy/repro/Dockerfile.app-overlay` using the release directory
as context, a verified existing runtime image as `BASE_IMAGE`, and the full
`SOURCE_COMMIT`. Record the old image digest and the resulting image digest.
Change only the image tag in the existing launcher; preserve device mounts,
runtime dependencies, model paths and all inference configuration.

Restart the backends and Arc's on-device frontend, restoring prior model,
prompts, target class and explicit Start/Stop intent when needed. Verify config,
zeros, geometry and gains stayed unchanged. Confirm the exact three public
model options, live processed frames, increasing sequences, compile/graph
diagnostics and both camera/servo statuses. Inspect current logs and compare
installed files with the release receipt. Keep Thor routed through
`192.168.249.2` on the dedicated Ethernet link; do not introduce Wi-Fi fallback.

On failure, restore the backed-up package/frontend or old launcher image and
restart the affected service. Retain the failed release and diagnostics. Never
overwrite calibration with files from another assembly during rollback.
