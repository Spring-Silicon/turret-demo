#!/usr/bin/env bash
# Copy only the hash-pinned Israel development candidate to a fresh directory.
# This does not stop/activate the demo or copy a Python environment/credentials.
set -euo pipefail
source_host=${1:?Usage: copy-sam31-w8a8.sh spring@israel spring@DESTINATION}
target_host=${2:?Usage: copy-sam31-w8a8.sh spring@israel spring@DESTINATION}
for host in "$source_host" "$target_host"; do
  [[ "$host" =~ ^[a-zA-Z0-9][a-zA-Z0-9@._-]+$ ]] || exit 2
done
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stage=$(ssh -o BatchMode=yes "$target_host" \
  'mktemp -d /var/lib/spring-data/turret-inference/sam31-israel.XXXXXX')
[[ "$stage" =~ ^/var/lib/spring-data/turret-inference/sam31-israel\.[a-zA-Z0-9]+$ ]] || exit 2
python3 -c 'import json,pathlib,sys; files=json.loads(pathlib.Path(sys.argv[1]).read_text()); print("\n".join([*files,"HANDOFF_SAM31_W8A8.md"]))' \
  "$repo_root/src/spring_turret/w8a8_manifest.json" \
  | ssh -o BatchMode=yes "$source_host" \
    'tar --verbatim-files-from --no-recursion -cf - -C /home/spring/sam3_1 -T -' \
  | ssh -o BatchMode=yes "$target_host" "tar -xf - -C '$stage'"
printf '%s\n' "$stage"
printf '%s\n' 'Copied only. Verify hashes, source feature parity, and live dense-reference gates before activation.' >&2
