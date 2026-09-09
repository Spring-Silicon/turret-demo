#!/usr/bin/env bash
# Copy the qualified sleepy-joe image bundle into a NEW, inactive directory.
# No source changes, service restart, credential copying, or servo access.
set -euo pipefail
source_host=${1:?Usage: copy-sam31-native.sh spring@sleepy-joe spring@DESTINATION}
target_host=${2:?Usage: copy-sam31-native.sh spring@sleepy-joe spring@DESTINATION}
for host in "$source_host" "$target_host"; do
  [[ "$host" =~ ^[a-zA-Z0-9][a-zA-Z0-9@._-]+$ ]] || exit 2
done
stage=$(ssh -o BatchMode=yes "$target_host" \
  'mktemp -d /var/lib/spring-data/turret-inference/sam31-sleepy.XXXXXX')
[[ "$stage" =~ ^/var/lib/spring-data/turret-inference/sam31-sleepy\.[a-zA-Z0-9]+$ ]] || exit 2
ssh -o BatchMode=yes "$source_host" 'tar -chf - \
  -C /home/spring/springsilicon/graphs/outputs/sam31-turret-image \
  native-attention-prefetch1 runtime-attention-prefetch1 \
  resident-attention-prefetch1-gpu0/native-attention-prefetch1 \
  tla-attention-box-validation/report.json attention-prefetch1-build-closure.json \
  -C /home/spring/springsilicon/graphs/target/release graphs-runner-intel \
  -C /home/spring/turret-demo-native-runtime/20260906 \
  graphs-runner-intel-turret runner-source.tar.gz runner-host-copy.patch \
  -C /opt/intel/oneapi/compiler/latest/lib \
  libsycl.so.9 libur_loader.so.0 libur_adapter_level_zero.so.0 \
  libur_adapter_level_zero_v2.so.0 libsvml.so libimf.so libirng.so libintlc.so.5 \
  -C /opt/intel/oneapi/dnnl/latest/lib libdnnl.so.3 \
  -C /opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8 libtbb.so.12 \
  -C /opt/intel/oneapi/umf/latest/lib libumf.so.1 \
  -C /opt/intel/oneapi/tcm/1.5/lib libhwloc.so.15 \
  -C /opt/intel/oneapi/licensing/latest/licensing/2026.0 license.htm \
  -C /opt/intel/oneapi/dnnl/latest/share/doc/dnnl LICENSE' \
  | ssh -o BatchMode=yes "$target_host" "tar -xf - -C '$stage'"
printf '%s\n' "$stage"
printf '%s\n' 'Copied only. Run destination qualification before activation.' >&2
