# Edge and Thor deployed source snapshot

Captured through SSH on 2026-09-08 UTC (2026-09-07 in New York).
This snapshot preserves the installed files. It does not replace the source at
the repository root or claim that the two installed variants are interchangeable.

## Machines and live checks

| Machine | Tailscale name | Direct Ethernet | Installed release |
| --- | --- | --- | --- |
| spring-edge-2 | spring-edge-2-1 | enp8s0, 192.168.249.1/30 | 0.15.14-tracking-masks |
| agxthor-4 | agxthor-4 | enP2p1s0, 192.168.249.2/30 | spring-turret-demo:0.16.7-tracking-masks |

Both services reported their camera online and SAM 3.1 Tracking running without
an error. Edge reported Intel Arc B580, BF16 temporal inference. Thor reported
NVIDIA Thor, BF16 temporal inference and CUDA graph replay.

The direct link needs no internet, router, DHCP, DNS or Tailscale. The saved
NetworkManager profiles use static IP addresses and autoconnect priority 100.
Use the listed physical Ethernet ports. Other Ethernet ports are not configured
by these profiles. Model weights and dependencies are already installed locally.

Verified on each host:

```sh
# On Edge:
ip route get 192.168.249.2
curl --max-time 5 http://192.168.249.2:8080/api/status
# On Thor:
ip route get 192.168.249.1
curl --max-time 5 http://192.168.249.1:8080/api/status
```

The routes selected enp8s0 and enP2p1s0, respectively. Both HTTP requests
returned an online camera. Wi-Fi was not disconnected, and neither host was
rebooted during this check. Internet removal and reboot were not tested.
Tailscale names are for remote access; use the direct IP addresses when offline.
Each host captures its own attached USB camera. The link provides access to
the other host's service; it does not itself move camera capture between hosts.

## Files

- `edge/src/spring_turret/`: exact application package from
  `/opt/spring/turret-demo/venv/lib/python3.12/site-packages/spring_turret/`.
- `thor/src/spring_turret/`: exact application package from the running
  `spring-turret-demo` container at
  `/usr/local/lib/python3.12/dist-packages/spring_turret/`.
- `thor/pyproject.toml`, `MANIFEST.in`, `README.md`: installed container build
  metadata. Its package version still says 0.16.0; use the image tag above to
  identify this deployment. Its README predates the tracking changes.
- `edge/sam-bundle/`: SAM source, license and scripts from
  `/var/lib/spring-data/turret-inference/israel-cast-cache.0F5LmD/bundle`.
  Runtime binaries and results are excluded.
- `edge/requirements-sam31.txt`: deployed inference dependency lock.
- `edge/host/` and `thor/host/`: selected host files, with paths relative to `/`.
  These include the direct-link Netplan profile, NetworkManager pre-up script,
  nftables rules, systemd service, udev rules, demo configuration, geometry and
  servo zero offsets. Edge also includes the service drop-ins.
- `SHA256SUMS`: hashes of the captured files, excluding this README and the
  manifest itself. Symlink aliases from tar archives are materialized as files.

Key application modules are `server.py` (capture/API), `detection.py` (worker
control), `sam31_worker.py` (detection), and `sam31_tracking*.py` (temporal SAM).
The published `feature/agxthor-cuda-demo` branch predates several of these files.

Thor's `/opt/sam3` is a clean checkout of upstream commit
`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` from
https://github.com/facebookresearch/sam3. The existing `deploy/thor/Dockerfile`
pins that source and the NVIDIA base image. The snapshot includes the updated
installed application separately; building the repository root alone does not
reproduce the installed tracking version.

## Operation on the existing machines

Both use `/etc/spring-turret-demo.json` and `spring-turret-demo.service`:

```sh
systemctl status spring-turret-demo
curl http://127.0.0.1:8080/api/detection/status
# Only when a stopped service needs to be started:
sudo systemctl start spring-turret-demo
```

Edge launches `/opt/spring/turret-demo/venv/bin/spring-turret`.
Thor launches `/opt/spring/turret-demo/run-container.sh`.
The camera interfaces are http://192.168.249.1:8080/ and
http://192.168.249.2:8080/ from the other host on the link.

The nftables guards keep peer traffic on the direct Ethernet interface.
The Netplan files are stored under `host/etc/netplan/`; the associated scripts
are under `host/etc/NetworkManager/dispatcher.d/pre-up.d/`.
No dedicated DNS service was found for this connection. Thor's separate
dnsmasq service had failed because port 53 was occupied. It is not needed for
the static link. The edge-image QSFP camera DHCP template also disables DNS.

## Scope and restoration limits

This is a source and deployment-config backup, not a full disk or container
backup. It contains no passwords, SSH keys, Wi-Fi profiles, model checkpoints,
compiler caches, camera images or recordings. Existing SAM weights must remain
available at the configured host paths. Optional Edge W8A8/W4A4 runtime binaries
and bundles referenced by configuration are not copied here. Their application
adapters and existing repository deployment notes are retained.

Host configs are specific to these two machines. Geometry, device serials and
zero offsets belong to the corresponding physical assembly. Do not apply them
to another assembly without calibration. No remote application files, network
settings or service states were changed during this capture.

The base OS provisioning source is in https://github.com/Spring-Silicon/edge-image.
The installed Edge image reports source commit
`cccd9dae05a3b86adc79ac12b56420caac5a699c`.
