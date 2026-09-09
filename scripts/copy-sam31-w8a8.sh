#!/usr/bin/env bash
# Copy only the hash-pinned Israel development candidate to a fresh directory.
# This does not stop/activate the demo or copy a Python environment/credentials.
set -euo pipefail
source_host=${1:?Usage: copy-sam31-w8a8.sh spring@israel spring@DESTINATION}
target_host=${2:?Usage: copy-sam31-w8a8.sh spring@israel spring@DESTINATION}
profile=${3:-original}
[[ "$profile" = original || "$profile" = packed || "$profile" = cast-cached ]] || exit 2
for host in "$source_host" "$target_host"; do
  [[ "$host" =~ ^[a-zA-Z0-9][a-zA-Z0-9@._-]+$ ]] || exit 2
done
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stage=$(ssh -o BatchMode=yes "$target_host" \
  'mktemp -d /var/lib/spring-data/turret-inference/sam31-israel.XXXXXX')
[[ "$stage" =~ ^/var/lib/spring-data/turret-inference/sam31-israel\.[a-zA-Z0-9]+$ ]] || exit 2
python3 -c 'import json,pathlib,sys; root=pathlib.Path(sys.argv[1]); files=json.loads((root/"w8a8_manifest.json").read_text()); files.update(json.loads((root/"w8a8_packed_manifest.json").read_text()) if sys.argv[2] in ("packed", "cast-cached") else {}); files.update(json.loads((root/"w8a8_cast_cache_manifest.json").read_text()) if sys.argv[2]=="cast-cached" else {}); print("\n".join([*files,"HANDOFF_SAM31_W8A8.md"]))' \
  "$repo_root/src/spring_turret" "$profile" \
  | ssh -o BatchMode=yes "$source_host" \
    'tar --verbatim-files-from --no-recursion -cf - -C /home/spring/sam3_1 -T -' \
  | ssh -o BatchMode=yes "$target_host" "tar -xf - -C '$stage'"
if [[ "$profile" = packed || "$profile" = cast-cached ]]; then
  ssh -o BatchMode=yes "$target_host" "mkdir -p '$stage/runtime/graphics'"
  python3 -c 'import json,pathlib,sys; print("\n".join(json.loads(pathlib.Path(sys.argv[1]).read_text())))' \
    "$repo_root/src/spring_turret/w8a8_graphics_manifest.json" \
    | ssh -o BatchMode=yes "$source_host" \
      'tar --dereference --verbatim-files-from --no-recursion -cf - -C /usr/lib/x86_64-linux-gnu -T -' \
    | ssh -o BatchMode=yes "$target_host" "tar -xf - -C '$stage/runtime/graphics'"
fi
printf '%s\n' "$stage"
printf '%s\n' 'Copied only. Verify hashes, source feature parity, and live dense-reference gates before activation.' >&2
