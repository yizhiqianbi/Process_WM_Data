#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 --check|--apply FASTWAM_REPO" >&2
  exit 2
fi
mode="$1"
if [[ "$mode" != "--check" && "$mode" != "--apply" ]]; then
  echo "first argument must be --check or --apply" >&2
  exit 2
fi

process_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fastwam_repo="$(cd "$2" && pwd)"
patch_path="$process_repo/integrations/fastwam/fastwam_memory_training.patch"
base_sha="45d8e1458921d83f8ad6cf9ce993d371208dabd0"
current_sha="$(git -C "$fastwam_repo" rev-parse HEAD)"

if [[ "$current_sha" != "$base_sha" ]]; then
  echo "FastWAM checkout must be at $base_sha; found $current_sha" >&2
  exit 1
fi
if [[ ! -f "$patch_path" ]]; then
  echo "integration patch is missing: $patch_path" >&2
  exit 1
fi

git -C "$fastwam_repo" apply --check "$patch_path"
if [[ "$mode" == "--apply" ]]; then
  if [[ -n "$(git -C "$fastwam_repo" status --porcelain)" ]]; then
    echo "FastWAM checkout must be clean before applying the integration" >&2
    exit 1
  fi
  git -C "$fastwam_repo" apply "$patch_path"
  echo "FastWAM memory training integration applied to $fastwam_repo"
else
  echo "FastWAM memory training integration can be applied cleanly"
fi
