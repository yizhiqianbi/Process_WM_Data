#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 --check|--apply RMBENCH_REPO" >&2
  exit 2
fi
mode="$1"
if [[ "$mode" != "--check" && "$mode" != "--apply" ]]; then
  echo "first argument must be --check or --apply" >&2
  exit 2
fi

process_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rmbench_repo="$(cd "$2" && pwd)"
patch_path="$process_repo/integrations/rmbench/rmbench_fastwam_runtime.patch"
base_sha="57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c"
current_sha="$(git -C "$rmbench_repo" rev-parse HEAD)"

if [[ "$current_sha" != "$base_sha" ]]; then
  echo "RMBench checkout must be at $base_sha; found $current_sha" >&2
  exit 1
fi
if [[ ! -f "$patch_path" ]]; then
  echo "RMBench runtime patch is missing: $patch_path" >&2
  exit 1
fi

git -C "$rmbench_repo" apply --check "$patch_path"
if [[ "$mode" == "--apply" ]]; then
  if [[ -n "$(git -C "$rmbench_repo" status --porcelain)" ]]; then
    echo "RMBench checkout must be clean before applying the runtime patch" >&2
    exit 1
  fi
  git -C "$rmbench_repo" apply "$patch_path"
  echo "RMBench FastWAM runtime patch applied to $rmbench_repo"
else
  echo "RMBench FastWAM runtime patch can be applied cleanly"
fi
