# FastWAM Code Integration

This directory carries the code-only FastWAM integration required by the canonical
`TrainingCaseV1`, 8/2/1 visual memory, fixed-probe evaluation, optional camera rectification, and
the Tianji raw-fisheye dataset-overfit workflow.
It contains no model weights, datasets, caches, or credentials.

Current patch SHA256:

```text
c97fa8bf3c7e840091e25686bebd9fdaeae4a1f00b5dd5b8540c188b76bb53e7
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

The current clean-checkout result is 27 passed tests. The suite includes the inference contract
that keeps canonical padded action dimensions zero throughout flow-matching denoising,
optional fisheye-to-pinhole mapping, and synchronized imagination/GT pair dimensions.

`scripts/export_fastwam_integration_patch.sh` regenerates the patch from the maintained local
checkout. It intentionally exports only canonical data, memory model, trainer, scheduler, and
their tests. RMBench experiments and all runtime artifacts are excluded.
