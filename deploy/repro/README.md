# Recorded deployments

See [the full restore/update runbook](../../docs/reproduce.md).

- `arc-20260909.json`: Arc configuration, zeros, full host/Python package lists,
  archived paths and integrity hashes for **recovery-20260909-arc-v2**.
- `thor-20260909.json`: equivalent Thor lock plus immutable NVIDIA container ID.
- `arc-os-20260909.json`: exact edge-image revision, hardware/BAR and private
  base-OS build artifact checksums.
- `Dockerfile.thor`: application-only overlay on the frozen recovery image;
  invoke through `scripts/build-thor-from-recovery.sh` to verify the base ID.

These locks contain no weights, passwords, SSH keys, Tailscale identity or browser
cookies. The private runtime archives are required for exact recovery. Native
compiler/runtime binaries cannot be recovered just by reinstalling package names.
`source_commit` identifies the corresponding app source; archive hashes pin the
actual deployed files, including platform-specific patches.

## Verification performed, 2026-09-09

- Both original exports: SHA256 and size checks, zstd stream checks, required
  archive roots, and archived configuration/checkpoint hashes passed.
- Both complete exports were mirrored to the other device over the direct cable
  and passed the same verification there, independently of their original disks.
- Arc v2 explicitly includes both external native-model symlink targets. The
  older Arc export without `-v2` must not be used.
- CPU/JavaScript regression suite passed locally. Torch-dependent tensor tests
  are skipped on this Mac; this run is not a new GPU accuracy qualification.
- The Thor application overlay built on the actual aarch64 host using the frozen
  image, `--network=none`, and no dependency downloads. Test tag:
  `spring-turret-demo:repro-source-c73ffac`; resulting image ID:
  `sha256:ffba00140d5fb05f21559d1bfb408cbb6354e7d7640a8955d0e51eb87c35787c`.
  This test image was not substituted for the running demo.
- Read-only live checks showed both cameras/servos online and both Mask workers
  running with Torch compilation and the expected SYCL/CUDA replay. No reload,
  model switch, motor command or recalibration was done for this export.

Not performed: full base-OS reinstallation, clean-board recovery boot, or fresh
three-profile GPU/hardware qualification. Those are explicit acceptance steps in
the runbook, not inferred from a successful archive or Docker build.
