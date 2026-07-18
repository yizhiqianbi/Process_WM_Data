# FastWAM Preprocessing Validation Status

Updated: 2026-07-18

This page records what has been demonstrated by executable tests and real source files. It is
not a claim that every upstream dataset has finished downloading or that production pretraining
has completed. Runtime data, caches, credentials, and checkpoints are intentionally excluded
from this repository.

## Current gates

| Gate | Status | Evidence |
|---|---|---|
| Seven dataset adapters | Passed on current real samples | Every dataset produces a `TrainingCaseV1` path |
| Unified FastWAM timeline | Passed | 81 state points, 80 action transitions, 21 video points |
| AgiBot observation/proprio/task join | Passed | Real task 389 / episode 673828 |
| AgiBot action admission | Passed | A tier, 18/18 `joint_video_action` windows |
| AgiBot canonical mapping | Passed | 20 active state slots and 20 active action slots |
| AgiBot normalization | Passed | 761 unique state rows and 760 unique action rows covered by admitted windows |
| Three-camera FastWAM loader | Passed | `[3,21,384,320]`, all three camera roles present |
| FastWAM 8/2/1 memory loader | Passed | 11/11 valid history frames, all indices earlier than window start |
| Text conditioning cache | Passed | `[128,4096]` finite UMT5 context |
| Preprocessing tests | Passed | 33 tests |
| FastWAM integration tests | Passed | 15 tests in the connected FastWAM checkout |
| AgiBot component download | Running | Locked selection: 224 files, 247.35 GiB, four file workers |
| AgiBot 5B optimizer/checkpoint smoke | Pending resource gate | Config composes; current job cannot access CUDA/NVML and the host GPUs are occupied |
| Full seven-domain production stats | Pending full manifests | Validation stats currently cover only locally materialized train/A/joint domains |

## AgiBot real-data proof

The verified episode has 1226 source HDF5 rows. Native action index datasets reduce this to the
1167 valid original indices `24..1190`. The pipeline then materializes 786 canonical state rows
at 20 Hz and creates 18 stride-40 windows. No window is downgraded by temporal, visual, state, or
action cleaning.

The active canonical slots are:

```text
6, 13, 14..27, 58, 59, 60, 61
```

They represent two grippers, two seven-joint arms, two waist values, and two head values.
Gripper feedback and command use different units, so those slots are trainable but excluded from
same-unit alignment. The other 18 comparable slots have alignment score 1.0 and lag 0 on the
validated episode.

## Reproduction sequence

```bash
export FASTWAM_DATA_ROOT=/path/to/robot_dataset

# Commit-locked AgiBot action-training assets.
python3 scripts/download_agibot_training_assets.py \
  --data-root "$FASTWAM_DATA_ROOT" \
  --token-file /secure/path/hf_token.txt \
  --file-workers 4

# Unified preprocessing and cleaning.
python3 scripts/run_pipeline.py \
  --datasets agibot_beta \
  --output-root work/stage_pipeline \
  --num-frames 81 \
  --target-fps 20 \
  --verify-files \
  --check-videos

# Train-only, A-tier, joint-window normalization.
python3 scripts/build_fastwam_normalization_stats.py \
  --pipeline-root work/stage_pipeline \
  --datasets all \
  --data-root . \
  --output work/stage_pipeline/normalization_stats.json

# Decode and validate the exact tensors consumed by MemoryFastWAM.
python3 scripts/validate_fastwam_training_cases.py \
  --fastwam-repo /path/to/FastWAM \
  --preprocess-root . \
  --datasets agibot_beta \
  --fail-fast
```

## Acceptance boundary

The current result proves the schema, cleaning, mapping, normalization, video decode, text, and
causal-memory data path for a real AgiBot episode. Production Stage 2 still requires:

1. A successful size-verified receipt for all selected AgiBot components.
2. Full-manifest join, rejection, slot-coverage, and alignment reports.
3. Normalization rebuilt from the frozen full train split.
4. One GPU optimizer/checkpoint/resume smoke with both video and action losses finite.
5. A seven-domain balanced optimizer regression before long-running pretraining.

Detailed contracts are in `AGIBOT_PROPRIO_TRAINING.md`, `ACTION_DATA_ADMISSION.md`,
`CLEANING_PIPELINE_V2.md`, and `THREE_STAGE_FASTWAM_TRAINING.md`.
