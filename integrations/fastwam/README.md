# FastWAM Code Integration

This directory carries the code-only FastWAM integration required by the canonical
`TrainingCaseV1`, 8/2/1 visual memory, rank-sharded fixed-probe evaluation, optional camera
rectification, the Tianji raw-fisheye dataset-overfit workflow, and the RMBench Helios-memory
training/evaluation path. The trainer supports single-process execution and real PyTorch DDP
through `torchrun` without bypassing the wrapped model during forward/backward.
It contains no model weights, datasets, caches, or credentials.

Current patch SHA256:

```text
abf45b9f55b86bdd4c244247afa505d2c549678055c33568550d31c23aad4fa0
```

The patch is pinned to upstream FastWAM commit:

```text
45d8e1458921d83f8ad6cf9ce993d371208dabd0
```

Apply it to a clean checkout:

```bash
git clone https://github.com/yuantianyuan01/FastWAM.git
cd FastWAM
git checkout 45d8e1458921d83f8ad6cf9ce993d371208dabd0
cd /path/to/Process_WM_Data

scripts/apply_fastwam_integration.sh --check /path/to/FastWAM
scripts/apply_fastwam_integration.sh --apply /path/to/FastWAM
```

Then install the FastWAM environment and run its CPU contract tests:

```bash
cd /path/to/FastWAM
python -m pytest -q
```

The connected and clean-patch checkout result is 41 passed tests. The suite includes the inference contract that
keeps canonical padded action dimensions zero throughout flow-matching denoising, optional
fisheye-to-pinhole mapping, synchronized imagination/GT pair dimensions, DDP-safe wrapped model
forward, complete ordered merging of rank-sharded fixed probes, and RMBench data/memory/result
contracts. RMBench simulator-side changes are a separate pinned patch under
[`integrations/rmbench/`](../rmbench/README.md).

`scripts/export_fastwam_integration_patch.sh` regenerates the patch from the maintained local
checkout. It exports only the explicit canonical-data, memory-model, trainer, RMBench and test
whitelists. Runtime artifacts remain excluded.
