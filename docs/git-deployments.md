# Git-pinned application updates

Do not synchronize mutable installed Python trees between devices. Commit the
reviewed application, build a release from that exact clean revision, and install
that release onto each device's existing qualified runtime. Licensed weights,
private vendor environments, calibration, Wi-Fi/SSH credentials and compiler
caches stay outside Git. Keep per-device configuration and direct-Ethernet
bindings; never copy another physical turret's zeros or geometry.

The 2026-09-10 sync imports the original `spring-edge-turret` experimental menu
and workers, while retaining the Vegas setup-recovery fixes and Thor graph-cache
repair. The original deployment's four Proteus modules matched the imported
sources byte-for-byte. See [profiles and runtime prerequisites](proteus-models.md).

## Build

Use Python 3.12 with pip, setuptools and wheel in a separate build environment.
Do not install or upgrade dependencies in either live inference environment.

```sh
git fetch origin
git switch --detach FULL_REVIEWED_COMMIT
python3 tools/build_app_release.py --role arc --commit FULL_REVIEWED_COMMIT --output /new/release-arc
python3 tools/build_app_release.py --role thor --commit FULL_REVIEWED_COMMIT --output /new/release-thor
```

The builder rejects a dirty checkout, requires the full checked-out commit, and
records every deployed source hash and wheel digest in `release.json`. Wheel
versions include the Git revision and hardware role. Transfer sources with
`git clone`/`git fetch`; on a device without GitHub credentials, use a Git bundle
created from the reviewed branch and set `origin` to the canonical GitHub repo.
Do not copy account credentials to make a deployment clone work.

### Arc runtime compatibility

Two exact application adapters are retained under `deploy/overlays/arc-20260910`:

- `sam31_graph.py`: the XPU-only source fingerprint used by the existing offline
  Mask artifacts. Substituting the newer generic CUDA/XPU wrapper invalidates
  that artifact even if its high-level behavior is similar.
- `sam31_w4a4.py`: the deployed map20/frozen Box adapter and its native-bundle
  digest pins. Main's older barrier adapter describes a different bundle.

Both were read from Vegas and byte-verified against the original Arc deployment.
The release builder validates their fixed digests and applies them only to Arc.
The shared API/control/viewer code is identical across both release roles.
These compatibility copies are not permission to relax model-manifest checks;
retire them only after qualifying matching rebuilt artifacts. The old installed
package version and older snapshot polling loop are not carried forward.

## Install and verify

Back up the existing package/frontend and launcher first. Record live model,
prompts, target and requested Start/Stop state; a stopped device must remain
stopped. No calibration or physical test sweep is needed for a source update.

On Arc, install the verified wheel with `uv pip install --python
/home/spring/.local/share/turret-demo/venv/bin/python --no-deps --no-index
--reinstall /release/dist/EXACT_WHEEL.whl`. Gracefully stop/restart only the demo
backend around installation. Do not touch `inference-venv`. Install the same
release's `app/scripts/frontend.mjs` and `app/src/spring_turret/static` beneath
the existing on-device frontend directory, then restart its service so cached
assets refresh. Retain its existing combined/standalone launcher configuration.

On Thor, use its current verified runtime image, not an older recovery image:

```sh
docker image inspect --format '{{.Id}}' EXACT_BASE_TAG
# Compare the ID with the recorded deployment before continuing.
docker build --pull=false --network=none \
  --build-arg BASE_IMAGE=EXACT_BASE_TAG --build-arg SOURCE_COMMIT=FULL_REVIEWED_COMMIT \
  -f /checkout/deploy/repro/Dockerfile.app-overlay \
  -t spring-turret-demo:git-COMMIT-thor /release-thor
```

Verify installed hashes against `release.json`, record the new image ID, update
only the active launcher's image tag, and restart its backend. Preserve CUDA,
Torch, `/opt/sam3`, model mounts, caches, device access and startup services.
Keep the previous image and launcher for rollback.

Restore unchanged user selections/Start intent after the restart. Verify both
API menus, every new model's live worker output, masks on a known source image,
processed JPEGs through the wired frontend, and standalone/combined views.
Model availability in a dropdown alone is not a successful model deployment.
Record any experimental accuracy limitations; copying a model does not qualify
its predictions. Do not alter the original devices merely to update a replica.
