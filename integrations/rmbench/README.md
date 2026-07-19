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
f09b67fea4b732795f9d1579a8ebdd51db4a341679367d575900f078765f755e
```

It adds the following runtime contracts:

- Environment-controlled SAPIEN ray-tracing settings for headless renderer
  compatibility.
- Portable validation and resolution of CuRobo asset paths without changing the
  checked-in robot YAML files.
- A 600-second model-server socket timeout for long FastWAM inference calls.
- Atomic, identity-checked evaluation progress after every completed rollout.
- Resume support for the official expert-feasible seed scan.
- A bounded consecutive setup-error gate so dependency failures cannot be
  silently counted as policy failures.

Apply it to a clean checkout:

```bash
git clone https://github.com/robotwin-Platform/rmbench.git
cd rmbench
git checkout 57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c
cd /path/to/Process_WM_Data

scripts/apply_rmbench_runtime_patch.sh --check /path/to/rmbench
scripts/apply_rmbench_runtime_patch.sh --apply /path/to/rmbench
```

The current node evaluates SAPIEN through Mesa lavapipe because its H200 devices
do not expose Vulkan graphics queues. That Mesa installation is an environment
dependency and is not bundled here. RMBench's bundled CuRobo also requires
`warp-lang==1.8.0` because it imports the legacy `wp.torch` namespace. The
FastWAM-side launcher performs both dependency checks before starting a rollout.

Regenerate the patch only from the maintained local checkout:

```bash
scripts/export_rmbench_runtime_patch.sh /path/to/rmbench
```
