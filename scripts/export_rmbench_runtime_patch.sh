#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 RMBENCH_REPO [OUTPUT_PATCH]" >&2
  exit 2
fi

process_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rmbench_repo="$(cd "$1" && pwd)"
output_patch="${2:-$process_repo/integrations/rmbench/rmbench_fastwam_runtime.patch}"
base_sha="57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c"

tracked_files=(
  envs/_base_task.py
  envs/robot/robot.py
  script/eval_policy_client.py
  script/policy_model_server.py
)

added_files=(
  envs/robot/curobo_config.py
)

if ! git -C "$rmbench_repo" cat-file -e "$base_sha^{commit}"; then
  echo "RMBench base commit is missing: $base_sha" >&2
  exit 1
fi
for path in "${tracked_files[@]}" "${added_files[@]}"; do
  if [[ ! -f "$rmbench_repo/$path" ]]; then
    echo "required RMBench integration file is missing: $path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$output_patch")"
temporary="$(mktemp "${output_patch}.XXXXXX")"
trap 'rm -f "$temporary"' EXIT

git -C "$rmbench_repo" diff --binary "$base_sha" -- "${tracked_files[@]}" > "$temporary"
for path in "${added_files[@]}"; do
  status=0
  git -C "$rmbench_repo" diff --binary --no-index /dev/null "$path" >> "$temporary" || status=$?
  if (( status != 1 )); then
    echo "failed to serialize added RMBench file: $path" >&2
    exit 1
  fi
done

mv "$temporary" "$output_patch"
trap - EXIT
echo "$output_patch"
