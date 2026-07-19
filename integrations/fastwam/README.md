# FastWAM Code Integration

This directory carries the code-only FastWAM integration required by the canonical
`TrainingCaseV1`, 8/2/1 visual memory, fixed-probe evaluation, camera rectification, and Tianji
dataset-overfit workflow.
It contains no model weights, datasets, caches, or credentials.

Current patch SHA256:

```text
556cd2d6c30add1f258359c5841a969d0eaaa635e31260e590de819e145d3636
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

The current clean-checkout result is 26 passed tests. The suite includes the inference contract
that keeps canonical padded action dimensions zero throughout flow-matching denoising,
fisheye-to-pinhole mapping, and synchronized action-video artifact dimensions.

`scripts/export_fastwam_integration_patch.sh` regenerates the patch from the maintained local
checkout. It intentionally exports only canonical data, memory model, trainer, scheduler, and
their tests. RMBench experiments and all runtime artifacts are excluded.
