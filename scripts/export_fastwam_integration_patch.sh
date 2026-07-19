#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 FASTWAM_REPO [OUTPUT_PATCH]" >&2
  exit 2
fi

process_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fastwam_repo="$(cd "$1" && pwd)"
output_patch="${2:-$process_repo/integrations/fastwam/fastwam_memory_training.patch}"
base_sha="45d8e1458921d83f8ad6cf9ce993d371208dabd0"

tracked_files=(
  configs/train.yaml
  experiments/robotwin/fastwam_policy/deploy_policy.py
  experiments/robotwin/fastwam_policy/deploy_policy.yml
  pyproject.toml
  scripts/precompute_text_embeds.py
  src/fastwam/models/wan22/fastwam.py
  src/fastwam/models/wan22/fastwam_idm.py
  src/fastwam/models/wan22/mot.py
  src/fastwam/models/wan22/wan22.py
  src/fastwam/runtime.py
  src/fastwam/trainer.py
  src/fastwam/utils/samplers.py
)

added_files=(
  configs/data/canonical_agibot_memory_81f.yaml
  configs/data/canonical_robocoin_81f.yaml
  configs/data/canonical_robocoin_memory_81f.yaml
  configs/data/canonical_stage1_all.yaml
  configs/data/canonical_stage2_memory_all.yaml
  configs/data/rmbench_helios_memory_81f.yaml
  configs/model/fastwam_joint_canonical.yaml
  configs/model/fastwam_memory_canonical.yaml
  configs/model/wan22_video_backbone_canonical.yaml
  configs/sim_rmbench_helios.yaml
  configs/task/canonical_robocoin_joint_81f_1e-5.yaml
  configs/task/canonical_robocoin_joint_81f_smoke.yaml
  configs/task/stage1_all_datasets_smoke.yaml
  configs/task/stage1_video_backbone_pretrain.yaml
  configs/task/stage1_video_backbone_smoke.yaml
  configs/task/stage2_agibot_memory_smoke.yaml
  configs/task/stage2_all_datasets_memory_smoke.yaml
  configs/task/stage2_memory_active_smoke.yaml
  configs/task/stage2_memory_fastwam_pretrain.yaml
  configs/task/stage2_memory_fastwam_smoke.yaml
  configs/task/stage3_robocoin_memory_finetune.yaml
  configs/task/stage3_robocoin_memory_smoke.yaml
  configs/task/rmbench_helios_memory_finetune.yaml
  docs/RMBENCH_HELIOS.md
  experiments/robotwin/fastwam_helios_client/__init__.py
  scripts/prewarm_tar_video_cache.py
  scripts/rmbench/eval_rmbench_helios.sh
  scripts/rmbench/launch_rmbench_helios_evaluation.py
  scripts/rmbench/launch_rmbench_helios_training.py
  scripts/rmbench/prepare_helios_memory_data.py
  scripts/rmbench/requirements-eval-compat.txt
  scripts/rmbench/summarize_rmbench_results.py
  scripts/rmbench/train_rmbench_helios.sh
  scripts/validate_memory_inference.py
  scripts/validate_training_case_data.py
  src/fastwam/datasets/memory_bank.py
  src/fastwam/datasets/rmbench_memory_dataset.py
  src/fastwam/datasets/training_case_dataset.py
  src/fastwam/models/wan22/memory_fastwam.py
  src/fastwam/utils/eval_visualization.py
  tests/test_eval_visualization.py
  tests/test_masked_losses.py
  tests/test_memory_fastwam.py
  tests/test_rmbench_launchers.py
  tests/test_rmbench_memory_dataset.py
  tests/test_rmbench_results.py
  tests/test_samplers.py
  tests/test_training_case_io.py
)

if ! git -C "$fastwam_repo" cat-file -e "$base_sha^{commit}"; then
  echo "FastWAM base commit is missing: $base_sha" >&2
  exit 1
fi
for path in "${tracked_files[@]}" "${added_files[@]}"; do
  if [[ ! -f "$fastwam_repo/$path" ]]; then
    echo "required FastWAM integration file is missing: $path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$output_patch")"
temporary="$(mktemp "${output_patch}.XXXXXX")"
trap 'rm -f "$temporary"' EXIT

git -C "$fastwam_repo" diff --binary "$base_sha" -- "${tracked_files[@]}" > "$temporary"
for path in "${added_files[@]}"; do
  status=0
  git -C "$fastwam_repo" diff --binary --no-index /dev/null "$path" >> "$temporary" || status=$?
  if (( status != 1 )); then
    echo "failed to serialize added FastWAM file: $path" >&2
    exit 1
  fi
done

mv "$temporary" "$output_patch"
trap - EXIT
echo "$output_patch"
