# FastWAM Multi-Dataset Preprocessing

This directory contains the executable preprocessing and cleaning pipeline for OXE,
OXE-AugE, AgiBot-Beta, RoboCOIN, RoboMIND, Galaxea, InternData-A1, LingBot-VA
post-training data, and DreamZero-DROID. Raw data is read-only. Generated metadata and
canonical sidecars are written below `work/`.

This repository contains code and metadata contracts only. It does not redistribute raw
datasets, model weights, generated `work/` artifacts, or authentication tokens.

Raw downloads are handled by the portable, commit-locked downloader documented in
[`DATA_DOWNLOAD.md`](DATA_DOWNLOAD.md). It supports all nine logical datasets, gated access
checks, parallel resume, local status, and remote file-size verification.

The LingBot-VA/DreamZero source schemas, verified action slices, camera layouts, model-profile
differences, commands, and real-sample 81/80/21 results are documented in
[`LINGBOT_VA_DREAMZERO.md`](LINGBOT_VA_DREAMZERO.md).

The latest executable acceptance matrix, including the distinction between completed data-path
validation and pending production/GPU gates, is in
[`VALIDATION_STATUS.md`](VALIDATION_STATUS.md).

## Contract

- Control timeline: resampled to 20 Hz; 81 state points and 80 actions cover 4 seconds.
- Visual timeline: offsets `0,4,...,80`, giving 21 video points.
- Cameras: five semantic roles, never source-directory order.
- State/action: native vectors plus strict 80D canonical vectors and per-dimension masks.
- Quality A: complete visual, state, action, temporal/signal audit, and verified strict action mapping.
- Quality B: valid video candidate with incomplete, suspicious, or unverified action semantics; action loss is disabled.
- Quality C: corrupt, incomplete, empty, missing referenced files, or failed temporal structure.

Zero in a canonical vector is not evidence that a dimension exists. Consumers must use
`state_dim_valid_mask` and `action_dim_valid_mask`. Slots 64-79 are reserved and always invalid.

## Quick Start

Clone the repository and install the runtime dependencies:

```bash
git clone git@github.com:yizhiqianbi/Process_WM_Data.git
cd Process_WM_Data
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Point the adapters at a data root that follows `configs/datasets.yaml`, then run a small
end-to-end validation:

```bash
export FASTWAM_DATA_ROOT=/path/to/robot_dataset
python3 scripts/run_pipeline.py \
  --datasets robocoin galaxea \
  --max-episodes 20 \
  --verify-files
```

Run all adapters with two dataset-level workers:

```bash
python3 scripts/run_pipeline.py --datasets all --workers 2 --verify-files
```

`--check-videos` performs sparse visual auditing on up to three semantic camera roles per
episode. Do not enable `--decode-videos` for the first full run: it additionally invokes full
ffmpeg decode and should be scheduled after the manifest has filtered C-tier data.

### AgiBot proprio path

When AgiBot observations are already present, download only the components required for action
training, run the 81/80/21 pipeline, and build stats from admitted training windows:

```bash
python3 scripts/download_agibot_training_assets.py \
  --data-root "$FASTWAM_DATA_ROOT" \
  --token-file /secure/path/hf_token.txt \
  --file-workers 4

python3 scripts/run_pipeline.py \
  --datasets agibot_beta \
  --output-root work/stage_pipeline \
  --num-frames 81 \
  --target-fps 20 \
  --verify-files \
  --check-videos

python3 scripts/build_fastwam_normalization_stats.py \
  --pipeline-root work/stage_pipeline \
  --datasets all \
  --data-root . \
  --output work/stage_pipeline/normalization_stats.json
```

The component downloader selects `proprio_stats + task_info` by default. Camera `parameters` are
optional and do not gate action training. See
[`AGIBOT_PROPRIO_TRAINING.md`](AGIBOT_PROPRIO_TRAINING.md) for HDF5 indices, units, canonical slots,
normalization, memory validation, and the Stage 2 smoke command.

## Stages

1. `scan`: builds `episodes.jsonl`, `artifacts.jsonl`, and `summary.json` without copying raw data.
2. `clean`: runs structural, temporal, per-dimension signal, action alignment, visual, and language
   audits, then emits `joint_video_action`, `video_only`, or `reject` admission.
3. `materialize`: resamples to 20 Hz, writes 80D canonical state/action Parquet and masks,
   and records nearest source-frame alignment; videos remain source references.
4. `windows`: writes compact valid-start ranges for 81-step FastWAM clips.
5. `cases`: writes the unified `TrainingCaseV1`, fixed five-camera slots, loss masks,
   lineage-safe split, normalization domain, and one concrete loader example.

Each stage is independently runnable:

```bash
python3 -m fastwam_preprocess.cli scan \
  --dataset robocoin --verify-files \
  --output-root work/v1/robocoin/scan

python3 -m fastwam_preprocess.cli clean \
  --manifest work/v1/robocoin/scan/episodes.jsonl \
  --output-root work/v1/robocoin/clean

python3 -m fastwam_preprocess.cli materialize \
  --manifest work/v1/robocoin/clean/episodes.cleaned.jsonl \
  --output-root work/v1/robocoin/canonical

python3 -m fastwam_preprocess.cli windows \
  --manifest work/v1/robocoin/canonical/canonical_episodes.jsonl \
  --output-root work/v1/robocoin/windows

python3 -m fastwam_preprocess.cli cases \
  --windows-manifest work/v1/robocoin/windows/windows.jsonl \
  --output-root work/v1/robocoin/cases
```

Dataset-specific wrappers are in `scripts/preprocess_*.py`. They preserve scan-only mode and
also run the complete pipeline through TrainingCaseV1:

```bash
python3 scripts/preprocess_robocoin.py scan --verify-files
python3 scripts/preprocess_robocoin.py pipeline --max-episodes 20 --verify-files
```

## Current Source Handling

| Dataset | Implemented path | Current limitation |
|---|---|---|
| RoboCOIN | Native LeRobot metadata, Parquet, AV1 MP4 decode | Current verified sample is A-tier; expand full manifest/stats |
| Galaxea | Tar-contained LeRobot Parquet/AV1; canonical-active action auditing | Current R1-lite sample is A-tier; validate all embodiments and physical limits |
| OXE | Restricted tar-contained pickle reader; ASU xyz+rpy conversion | ASU UR5 is A-tier; every other OXE subset remains independently gated |
| OXE-AugE | One episode per target robot; next replay state as a derived target | A-tier derived action is not a native hardware command; preserve lineage/domain labels |
| AgiBot-Beta | Observation tar plus indexed `proprio_stats.h5` episode join | Real episode 673828 is A-tier with 20 active slots and 18 action windows; full proprio download is still in progress |
| LingBot-VA | Recursive LeRobot v2.1 RoboTwin/LIBERO discovery; verified EEF/gripper conversion | Full repositories and production stats are not yet materialized |
| DreamZero-DROID | LeRobot v2.0 plus `modality.json`; isolated Panda joint/gripper slices | Relative-joint transform remains an explicit training-loader policy |
| RoboMIND | Native HDF5, embedded images, official per-embodiment master/puppet table | Only official contract-table embodiments can enter A-tier |
| InternData-A1 | Native LeRobot action/state schema and three-camera decode | Current A2D sample is A-tier; separate real/sim normalization domains |

Cleaning thresholds are versioned in `configs/cleaning_policy_v1.yaml`; the nine resolved
dataset profiles are in `configs/training_profiles.yaml`. Every successful dataset pipeline
contains `cases/training_cases.jsonl`, `cases/example_case.json`, and the exact contract.

The PDF-derived V2 cleaning design, bad-interval contract, visual metrics, stage-specific
admission, calibration procedure, and current limitations are documented in
[`CLEANING_PIPELINE_V2.md`](CLEANING_PIPELINE_V2.md).

The evidence for each action mapping, the distinction between native and derived actions,
the exact reasons for remaining video-only data, and the 81/80 real-data regression matrix
are documented in [`ACTION_DATA_ADMISSION.md`](ACTION_DATA_ADMISSION.md).

The materializer remains a sidecar stage, while the FastWAM repository now includes a direct
`TrainingCaseDataset` bridge. It consumes the 80D element-wise masks, applies train-split
normalization per embodiment domain, composes semantic camera slots, and forwards A/B-tier
sample-level loss masks into FastWAM. The first real RoboCOIN 81/80/21 training and checkpoint
resume proof is documented in `FASTWAM_TRAINING_INTEGRATION.md`. Preprocessing now produces
81/80/21 joint cases for all seven original datasets and the new LingBot-VA/DreamZero real
schema samples. The stats builder now discovers
pipeline manifests and admits only train/A-tier/joint cases; full-dataset statistics and the
combined Stage 2 optimizer regression must still be rebuilt before production training.

The current training bridge also merges target and 8/2/1 memory frame requests into one decode,
materializes `tar://` video members into an atomic cache, and exposes dataset IDs to a deterministic
resumable balanced sampler. Stage 1/2 configs use equal per-dataset epoch quotas; Stage 3 keeps
target-dataset sampling. `MemoryFastWAM.infer_action` consumes the same masked 8/2/1 layout used in
training, and stage checkpoint environment variables are mandatory to prevent accidental reinitialization.

## Verification

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q fastwam_preprocess scripts
```

The executable three-stage video-backbone, memory FastWAM, and dataset-specific
finetuning plan is in `THREE_STAGE_FASTWAM_TRAINING.md`. It includes the seven-dataset
acceptance matrix, causal 8/2/1 memory contract, checkpoint handoff, commands, and current
real 5B/6B training and memory-aware inference results.

The repository-local design and execution references are
[`CLEANING_PIPELINE_V2.md`](CLEANING_PIPELINE_V2.md),
[`ACTION_DATA_ADMISSION.md`](ACTION_DATA_ADMISSION.md),
[`AGIBOT_PROPRIO_TRAINING.md`](AGIBOT_PROPRIO_TRAINING.md),
[`LINGBOT_VA_DREAMZERO.md`](LINGBOT_VA_DREAMZERO.md),
[`VALIDATION_STATUS.md`](VALIDATION_STATUS.md),
[`DATA_DOWNLOAD.md`](DATA_DOWNLOAD.md), and
[`THREE_STAGE_FASTWAM_TRAINING.md`](THREE_STAGE_FASTWAM_TRAINING.md).
