# FastWAM Raw Dataset Download

This document is the portable download runbook for OXE, OXE-AugE, AgiBot-Beta,
RoboCOIN, RoboMIND, Galaxea, and InternData-A1. The code directory does not contain
credentials, raw data, Hugging Face cache, or download logs.

## 1. Local layout

The downloader preserves the layout consumed by `fastwam_preprocess.registry`:

```text
DATA_ROOT/
├── OXE_OpenX_Embodiment/jxu124_OpenX-Embodiment/
├── OXE_AugE/<repository directories>/
├── AgiBot_Beta/AgiBotWorld-Beta/
├── RoboCOIN/<repository directories>/
├── RoboMIND/RoboMIND/
├── Galaxea/Galaxea-Open-World-Dataset/
├── InternData_A1/InternData-A1/
└── .fastwam_download/
    ├── hf_cache/
    ├── hf_home/
    ├── repos/
    └── runs/
```

`DATA_ROOT` is resolved in this order:

1. CLI `--data-root`.
2. Environment variable `FASTWAM_DATA_ROOT`.
3. Parent directory of `Preprocess_FastWAM`.

## 2. Minimal environment

Only two packages are required for downloading:

```bash
cd Preprocess_FastWAM
python3 -m pip install -r requirements-download.txt
```

Keep the token outside this directory:

```bash
chmod 600 /secure/path/hf_token.txt
export HF_TOKEN_FILE=/secure/path/hf_token.txt
export FASTWAM_DATA_ROOT=/data/robot_dataset
```

The CLI also accepts `--token-file`. A cached Hugging Face login or `HF_TOKEN` works,
but a token file is easier to audit on a shared server. The token is never written to
the lock, run manifest, marker, event log, or error report.

## 3. Reproducible manifest

`configs/download_sources.yaml` describes the seven logical datasets. OXE-AugE and
RoboCOIN are HF organizations containing many independent dataset repositories. Resolve
the organization listing and all fixed repositories to immutable commit SHAs:

```bash
python3 scripts/download_datasets.py resolve \
  --datasets all \
  --lock configs/download_manifest.lock.json
```

The checked-in lock is the reproducible release input. Run `resolve` deliberately when
new upstream repositories or revisions should be admitted; do not silently switch a
production download to `main`.

Check gated access before launching multi-terabyte downloads:

```bash
python3 scripts/download_datasets.py access \
  --datasets all \
  --lock configs/download_manifest.lock.json \
  --workers 16 \
  --output /data/robot_dataset/access_report.json
```

Exit code `0` means every locked repository is accessible. Exit code `2` means the JSON
report contains at least one gated, authentication, missing, or network failure.

## 4. Download and resume

Preview every destination without network transfer:

```bash
python3 scripts/download_datasets.py download \
  --datasets all \
  --data-root "$FASTWAM_DATA_ROOT" \
  --dry-run
```

Start or resume all datasets:

```bash
python3 scripts/download_datasets.py download \
  --datasets all \
  --data-root "$FASTWAM_DATA_ROOT" \
  --repo-jobs 4 \
  --file-workers 4 \
  --attempts 5
```

The convenience wrapper passes all arguments to the same command:

```bash
scripts/download_all_datasets.sh \
  --data-root "$FASTWAM_DATA_ROOT" \
  --repo-jobs 4 \
  --file-workers 4
```

Download one dataset or a repository subset:

```bash
python3 scripts/download_datasets.py download --datasets robomind --repo-jobs 1 --file-workers 8

python3 scripts/download_datasets.py download \
  --datasets robocoin \
  --repo-pattern 'RoboCOIN/Agilex_*' \
  --repo-jobs 4 \
  --file-workers 2
```

`repo-jobs * file-workers` is the approximate upper bound on concurrent file requests.
For OXE-AugE and RoboCOIN, favor repository concurrency. For one large gated repository,
use one repository job and more file workers. On a shared filesystem, start with 16 total
requests rather than increasing both controls independently.

Rerunning the command is safe:

- Hugging Face `snapshot_download` resumes partial files.
- A successful commit-specific marker skips an already completed repository.
- Existing data without a marker is reconciled by calling `snapshot_download`; files are
  not deleted or blindly downloaded again.
- `--recheck-complete` asks HF to recheck repositories that have successful markers.
- `--force-download` is intentionally separate and should not be used for normal resume.

Each run writes `events.jsonl`, `summary.json`, and the exact selected lock below
`DATA_ROOT/.fastwam_download/runs/`. Per-repository state is written atomically below
`DATA_ROOT/.fastwam_download/repos/`.

## 5. AgiBot action-training components

The full AgiBot repository is tens of terabytes because `observations/` contains the video
release. When observations already exist, use the component downloader to complete only the
files required for action training:

```bash
python3 scripts/download_agibot_training_assets.py \
  --data-root "$FASTWAM_DATA_ROOT" \
  --token-file "$HF_TOKEN_FILE" \
  --file-workers 4
```

The default selection is commit-locked and consists of:

| Component | Required for Stage 2 | Purpose |
|---|---|---|
| `proprio_stats/` | Yes | Per-episode HDF5 state, HAL action, timestamps, and valid action indices |
| `task_info/` | Yes | Episode-to-task language and task lineage |
| `parameters/` | No | Camera intrinsics/extrinsics and calibration assets |

At the currently locked revision, `proprio_stats/` contains seven tar shards totaling about
247 GiB and `task_info/` is about 0.3 GiB. `parameters/` is roughly 1.5 TiB and is therefore
opt-in; missing calibration does not disable state/action training.

Preview one shard without transferring it:

```bash
python3 scripts/download_agibot_training_assets.py \
  --data-root "$FASTWAM_DATA_ROOT" \
  --token-file "$HF_TOKEN_FILE" \
  --component proprio_stats \
  --proprio-shard 648533-713949.tar \
  --dry-run
```

Download calibration separately when geometric camera work needs it:

```bash
python3 scripts/download_agibot_training_assets.py \
  --data-root "$FASTWAM_DATA_ROOT" \
  --token-file "$HF_TOKEN_FILE" \
  --component parameters \
  --file-workers 4
```

The downloader verifies every selected file against remote size metadata and atomically writes
a receipt below `DATA_ROOT/.fastwam_download/components/`. It resumes Hugging Face `.incomplete`
files and never records the token.

## 6. Status and verification

Fast local status does not contact Hugging Face:

```bash
python3 scripts/download_datasets.py status \
  --datasets all \
  --data-root "$FASTWAM_DATA_ROOT" \
  --output "$FASTWAM_DATA_ROOT/download_status.json"
```

Status meanings:

| Status | Meaning |
|---|---|
| `complete` | Local directory and successful marker match the locked commit |
| `present_untracked` | Files predate this downloader; run download once to reconcile |
| `partial` | No current success marker exists and at least one HF `.incomplete` file remains |
| `failed` | Last recorded attempt failed |
| `missing` | Repository directory does not exist |
| `interrupted_or_running` | A run marker was left in `running` state |

Remote metadata verification compares the full locked file list and every available file
size. It does not reread and hash several terabytes of complete files:

```bash
python3 scripts/download_datasets.py verify \
  --datasets all \
  --data-root "$FASTWAM_DATA_ROOT" \
  --workers 4 \
  --output "$FASTWAM_DATA_ROOT/download_verify.json"
```

Use `--max-repos 1` for a quick environment smoke test. Do not use it for a production
completeness report.

## 7. Running in the background

The launcher records a PID and refuses to start a duplicate live process. It uses `tmux`
when available so the job survives non-interactive SSH or scheduler shells, and falls back
to `nohup` otherwise:

```bash
PYTHON_BIN=/path/to/python \
scripts/start_download_all.sh \
  --repo-jobs 4 \
  --file-workers 4
```

The machine can be restarted and the identical command rerun. The lock and local metadata
make the operation idempotent at a specific upstream commit.

## 8. Moving the code to another server

The code is independent of the original server layout. Export it without generated work:

```bash
scripts/export_code.sh /tmp/Preprocess_FastWAM.tar.gz
```

On the destination server, install `requirements-download.txt`, set `HF_TOKEN_FILE` and
`FASTWAM_DATA_ROOT`, then run `status`, `access`, and `download` in that order. Never place
the token file inside the archive or the project directory.
