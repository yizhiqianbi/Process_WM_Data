# RMBench Runtime Integration

This directory contains the code-only RMBench changes required by the Helios
MemoryFastWAM closed-loop evaluator. It contains no simulator assets, datasets,
videos, model checkpoints, credentials, caches, or machine-specific resolved
CuRobo files.

The patch is pinned to official RMBench commit:

```text
57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c
```

Patch SHA-256:

```text
853a43643bed7c2f4d26757717671eaf7e1907f2c8efb866d0206f054fc9ff4c
```

It adds the following runtime contracts:

- Environment-controlled SAPIEN raster/ray-tracing selection for headless
  renderer compatibility, with raster as the stable Mesa fallback.
- Portable validation and resolution of CuRobo asset paths without changing the
  checked-in robot YAML files.
- A 600-second model-server socket timeout for long FastWAM inference calls.
- Atomic, identity-checked evaluation progress after every completed rollout.
- Renderer protocol identity that records the pipeline and stores
  ray-tracing-only fields as null for raster runs.
- Resume support for the official expert-feasible seed scan.
- A bounded consecutive setup-error gate so dependency failures cannot be
  silently counted as policy failures.
- Explicit torch intra-op/inter-op limits before importing the simulator or
  model-server dependency graph.
- An optional host-level file lock around each complete SAPIEN
  observation-rendering transaction for the unstable CPU ray-tracing path;
  the raster protocol leaves it disabled for concurrent simulator progress.
- An evaluation-only switch to omit unused camera calibration matrices from
  each observation, reducing unstable Mesa/SAPIEN native calls without changing
  RGB, qpos, action, or camera-layout model inputs.

Apply it to a clean checkout:

```bash
git clone https://github.com/robotwin-Platform/rmbench.git
cd rmbench
git checkout 57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c
cd /path/to/Process_WM_Data

scripts/apply_rmbench_runtime_patch.sh --check /path/to/rmbench
scripts/apply_rmbench_runtime_patch.sh --apply /path/to/rmbench
```

The current node evaluates SAPIEN's `default` raster shader through Mesa
lavapipe because its H200 devices do not expose Vulkan graphics queues. The
CPU ray-tracing path is selectable but was not stable enough for the 9 x 100
suite. That Mesa installation is an environment dependency and is not bundled
here. RMBench's bundled CuRobo also requires
`warp-lang==1.8.0` because it imports the legacy `wp.torch` namespace. The
FastWAM-side launcher performs both dependency checks before starting a rollout.

Regenerate the patch only from the maintained local checkout:

```bash
scripts/export_rmbench_runtime_patch.sh /path/to/rmbench
```
