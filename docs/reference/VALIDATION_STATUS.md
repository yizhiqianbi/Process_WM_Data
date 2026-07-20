# Data and Training Validation Status

[文档索引](../README.md)

Updated: 2026-07-20

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
| Unified three-model launcher | Passed | Structured argv, explicit 1/8-GPU torchrun world size, external SHA, logs, receipts, checkpoint discovery and resume |
| FastWAM self-data optimizer/resume | Passed | Real step 1, full-state resume to step 2, action/video/memory losses finite |
| FastWAM Tianji fixed-window overfit | Passed | Step 300 first passes; selected cumulative step 900: 26.8693 dB, 0.9313 SSIM, action L1 0.01941; memory fully valid |
| Tianji active camera-domain policy | Passed | The completed full-dataset run preserves raw fisheye; every 8-probe suite reports `[false,false,false]` rectification masks and null config |
| Tianji dataset overfit | Passed | 44 episodes, 486 windows, 8 fixed cross-episode probes and cumulative 2,500 steps; final val loss `0.02562`, PSNR `19.1814 dB`, SSIM `0.78697`, action L1 `0.04041`, memory-valid `1.0` |
| FastWAM DDP contract | Passed | World size 8, global batch 8, eight unique first-batch indices, post-step parameter probe max difference `0.000e+00` |
| FastWAM inference/GT pair demo | Training-validated smoke | 8 probes are rank-sharded and merged once; synchronized H.264 640x416/21-frame imagination-vs-execution pairs and action composites passed |
| Old LingBot-VA optimizer/resume | Passed on one complete three-view latent segment | Real step 1, optimizer/scheduler resume to step 2, latent/action losses finite |
| Old LingBot-VA full self-data overfit | Passed | 44 episodes and 132/132 latents; 8 H200, batch 24/GPU, global batch 192, 250/250 steps; complete checkpoints at steps 125 and 250 |
| DreamZero optimizer/resume | Passed | Real LoRA step 1, Trainer resume to step 2, dynamics/action losses finite |
| DreamZero full self-data overfit | Passed | 8 H200, 44 episodes, global batch 8, 500/500 steps; complete `checkpoint-500`, final loss `0.0651` |
| DreamZero GT-observation Pair inference | Passed | Longest 8 eligible episodes, 913 frames/case and 7304-frame reel; all 33 MP4 files/36,520 frames decoded, GT frame IDs audited, no future-GT or predicted-latent observation feedback |
| RMBench Helios-memory preparation | Passed | Official 9 tasks, 450 episodes, 277,350 frames and 241,350 valid 81-frame windows; 14D qpos-next mapping, per-task stats and causal 8/2/1 memory audited |
| RMBench nine-task fine-tuning | Passed | 9/9 tasks completed 2,500 steps from the cumulative Tianji-2,500 initialization; 45/45 policy checkpoints and 45/45 complete optimizer states validated |
| RMBench closed-loop evaluation | Running, resumable | Exact step-2,500 checkpoints, unseen instructions and 100 expert-feasible rollouts/task; 8-GPU model execution with serialized Mesa observation rendering and atomic per-rollout progress |
| Repository tests | Passed | 78 tests, including target preparation, long DreamZero Pair scheduling, 8-GPU launchers, rectification and tuning contracts |
| FastWAM integration tests | Passed | 45 tests in the connected checkout, including canonical RMBench data/memory/action contracts, protocol-identity gating, stride-aligned GT observation refresh, bounded native-failure retry and result validation |
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

The production DreamZero overfit then trained all 44 self-data episodes for 500 steps on eight
H200 GPUs. The complete step-500 adapter checkpoint was used for eight-case pair inference. Each
case uses 114 receding-horizon chunks and produces 913 frames at 28 FPS; this is the largest
uniform `1 + 8k` horizon supported by the eight longest eligible episodes because the shortest
selected episode has 920 source frames. The aggregate contains 7,304 frames. Independent PyAV
validation decoded all 32 per-case videos plus the reel, 36,520 frames in total, and verified
nonblank first and last frames. Receipt-level metrics average `20.7243 dB` over the three camera
PSNR values and `0.06543` MAE over the eight action channels.

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
latent jobs. All 132 VAE latents pass manifest/payload frame-ID validation. The full target then
trained for 250 steps on eight H200 GPUs with 24 fixed 16-latent-frame windows per GPU, for a
global batch of 192. Complete model and training state were saved at steps 125 and 250; the final
logged losses were latent `0.0646` and action `0.0031`.

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
