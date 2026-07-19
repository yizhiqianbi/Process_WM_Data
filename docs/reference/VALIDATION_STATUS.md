# Data and Training Validation Status

[文档索引](../README.md)

Updated: 2026-07-19

This page records what has been demonstrated by executable tests and real source files. It is
not a claim that every upstream dataset has finished downloading or that production pretraining
has completed. Runtime data, caches, credentials, and checkpoints are intentionally excluded
from this repository.

## Current gates

| Gate | Status | Evidence |
|---|---|---|
| Original seven dataset adapters | Passed on current real samples | Every original dataset produces a `TrainingCaseV1` path |
| LingBot-VA adapters | Passed on pinned real samples | RoboTwin and LIBERO both produce joint 81/80/21 cases |
| DreamZero-DROID adapter | Passed on pinned real sample | Verified GEAR slices produce a joint 81/80/21 case |
| Unified FastWAM timeline | Passed | 81 state points, 80 action transitions, 21 video points |
| AgiBot observation/proprio/task join | Passed | Real task 389 / episode 673828 |
| AgiBot action admission | Passed | A tier, 18/18 `joint_video_action` windows |
| AgiBot canonical mapping | Passed | 20 active state slots and 20 active action slots |
| AgiBot normalization | Passed | 761 unique state rows and 760 unique action rows covered by admitted windows |
| Three-camera FastWAM loader | Passed | `[3,21,384,320]`, all three camera roles present |
| FastWAM 8/2/1 memory loader | Passed | 11/11 valid history frames, all indices earlier than window start |
| Text conditioning cache | Passed | `[128,4096]` finite UMT5 context |
| Old LingBot-VA target preparation | Passed on real 44-episode self-data and synthetic fixtures | 15D-to-8D compact action, 30D mapping, 132 latent jobs and latent readiness gate |
| DreamZero target preparation | Passed on real 44-episode self-data and synthetic fixture | GEAR modality, absolute/relative stats, language column and Hydra registration gate |
| FastWAM self-data preprocessing | Passed | 44 A-tier episodes; 486 windows: 381 action and 105 interval-downgraded video-only; 127 deduplicated cases |
| Unified three-model launcher | Passed | Structured argv, external SHA, logs, receipts, checkpoint discovery and explicit resume |
| FastWAM self-data optimizer/resume | Passed | Real step 1, full-state resume to step 2, action/video/memory losses finite |
| Old LingBot-VA optimizer/resume | Passed on one complete three-view latent segment | Real step 1, optimizer/scheduler resume to step 2, latent/action losses finite |
| DreamZero optimizer/resume | Passed | Real LoRA step 1, Trainer resume to step 2, dynamics/action losses finite |
| Repository tests | Passed | 50 tests, including target readiness and launcher checkpoint semantics |
| FastWAM integration tests | Passed | 15 tests in the connected FastWAM checkout |
| AgiBot-specific optimizer/checkpoint smoke | Pending dataset-specific run | The current optimizer proof uses `take_wrong_item_right_arm`, not AgiBot |
| Full nine-domain production stats | Pending full manifests | Validation stats currently cover only locally materialized train/A/joint domains |

## Three-model training proof

All rows below used the same real 44-episode, 31,359-frame `take_wrong_item_right_arm` source and
were launched through `scripts/tune_models.py`. Runtime receipts are under each ignored
`work/tuning/runs/<run>/_wm_tuning/` directory and record the exact argv and external Git state.

| Model | Step 1 | Resume step | Checkpoint result | External revision |
|---|---|---|---|---|
| Memory FastWAM Stage 3 | total `0.6428`, action `0.6343`, video `0.0085`, memory-valid `1.0` | total `2.8484`, action `1.9719`, video `0.8765`; model/optimizer/scheduler/sampler/RNG restored | full state and lightweight weights at steps 1 and 2 | `45d8e145...` (custom checkout, dirty state captured) |
| old LingBot-VA | latent `0.2028`, action `0.2584`, LR `1e-5` | latent `0.2028`, action `0.2584`, LR `2e-5`; optimizer/scheduler restored | `checkpoint_step_1` and `checkpoint_step_2`, each with model and `training_state.pt` | `7c6ffa9b...` (clean) |
| DreamZero | dynamics `0.01928`, action `0.04705` | dynamics `0.03890`, action `0.18704`; Trainer global step restored from 1 | `checkpoint-1` and `checkpoint-2`, each with LoRA model, optimizer, scheduler, RNG and Trainer state | `ab790c19...` (generated data profile recorded as dirty) |

The DreamZero step-2 checkpoint contains a 414 MiB LoRA/model state and an 829 MiB optimizer
state. The wrapper suppresses the upstream duplicate final export of the complete 16.5B model;
this does not remove any state needed to resume training.

These are optimizer and checkpoint smoke tests, not convergence results. They do not establish
multi-node throughput, long-horizon stability, validation quality, offline policy metrics, or
closed-loop robot success.

## LingBot-VA and DreamZero data proof

Pinned HF files, not invented dimensions, were used to validate the new schemas. RoboTwin
`blocks_ranking_rgb` episode 0 produced three joint windows with canonical slots `0..13`.
LIBERO-Long episode 0 produced one joint window with slots `0..6`. DreamZero-DROID episode 0
produced six joint windows with gripper slot `6` and joint slots `14..20`. All resulting cases
use 81 state points, 80 actions, and 21 video points.

The full upstream repositories, official loaders, real video decode, and GPU optimizer paths have
now been exercised for the current self-data target. Exact source layouts and commands are in
[LingBot-VA / DreamZero](../datasets/LINGBOT_VA_DREAMZERO.md) and
[Three-Model Tuning](../training/THREE_MODEL_TUNING.md).

The model-target exporters were also run against the current 44-episode, 31,359-frame,
LeRobot v2.1 `take_wrong_item_right_arm` dataset. The old LingBot-VA exporter compacted the
15D action to eight declared right-arm/right-gripper dimensions and emitted 132 deterministic
latent jobs. Three episode-0 latents, one for each configured physical view, currently exist and
were sufficient for the bounded LingBot smoke; this target is still not production-ready until
all 132 VAE latents exist. The launcher rejects this partial state by default and permits it only
when `allow_partial_latents: true` is explicit.

The DreamZero exporter emitted custom GEAR metadata, relative statistics, language, and a Hydra
profile. That profile is installed and checked against the existing `EmbodimentTag.XDOF` and
projector mapping. The official streaming sample produced video `[33,352,640,3]`, action and mask
`[96,32]`, and state `[4,64]`; the training batch reached the model as
`[1,3,33,352,640]` and completed both optimizer steps.

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

The AgiBot result proves the schema, cleaning, mapping, normalization, video decode, text, and
causal-memory data path for one real episode. AgiBot production Stage 2 still requires:

1. A successful size-verified receipt for all selected AgiBot components.
2. Full-manifest join, rejection, slot-coverage, and alignment reports.
3. Normalization rebuilt from the frozen full train split.
4. One AgiBot-specific GPU optimizer/checkpoint/resume smoke with both losses finite.
5. A nine-domain balanced optimizer regression before long-running pretraining.

Detailed contracts are in [AgiBot](../datasets/AGIBOT.md),
[LingBot-VA / DreamZero](../datasets/LINGBOT_VA_DREAMZERO.md),
[Action Admission](../data/ACTION_ADMISSION.md),
[Preprocessing](../data/PREPROCESSING.md), and
[FastWAM Three-Stage](../training/FASTWAM_THREE_STAGE.md).
