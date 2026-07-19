from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from .common import CommandSpec, TuningConfigError, model_config, required_path


def _validate_latent_coverage(dataset: Path, *, allow_partial: bool) -> None:
    jobs_path = dataset / "meta" / "lingbot_va_latent_jobs.jsonl"
    if not jobs_path.is_file():
        raise TuningConfigError(f"LingBot-VA latent job manifest is missing: {jobs_path}")
    jobs = [
        json.loads(line)
        for line in jobs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not jobs:
        raise TuningConfigError(f"LingBot-VA latent job manifest is empty: {jobs_path}")

    present = [job for job in jobs if (dataset / str(job["output"])).is_file()]
    if len(present) == len(jobs):
        return
    if not allow_partial:
        raise TuningConfigError(
            "LingBot-VA target has incomplete latents: "
            f"{len(present)}/{len(jobs)}; extract all jobs or explicitly set "
            "allow_partial_latents: true for a smoke run"
        )

    expected_groups: dict[tuple[int, int, int], int] = defaultdict(int)
    present_groups: dict[tuple[int, int, int], int] = defaultdict(int)
    for job in jobs:
        group = (
            int(job["episode_index"]),
            int(job["start_frame"]),
            int(job["end_frame"]),
        )
        expected_groups[group] += 1
        if (dataset / str(job["output"])).is_file():
            present_groups[group] += 1
    if not any(
        present_groups[group] == expected_count
        for group, expected_count in expected_groups.items()
    ):
        raise TuningConfigError(
            "LingBot-VA partial target has no segment with every configured camera latent"
        )


def build_lingbot_va_command(
    config: dict[str, Any],
    *,
    phase: str,
    output_dir: Path,
    steps: int,
    resume: Path | None,
    gpus: str | None,
) -> CommandSpec:
    if phase != "finetune":
        raise TuningConfigError("old LingBot-VA supports phase=finetune")
    if steps <= 0:
        raise TuningConfigError("steps must be positive")
    model = model_config(config, "lingbot_va")
    repo = required_path(model, "repo", directory=True)
    python = required_path(model, "python", directory=False)
    dataset = required_path(model, "dataset_root", directory=True)
    model_root = required_path(model, "model_root", directory=True)
    _validate_latent_coverage(
        dataset, allow_partial=bool(model.get("allow_partial_latents", False))
    )
    if resume is not None and not resume.expanduser().resolve().is_dir():
        raise TuningConfigError(f"LingBot-VA resume checkpoint is missing: {resume}")
    process_repo = Path(__file__).resolve().parents[1]
    argv = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={int(model.get('num_gpus', 1))}",
        str(process_repo / "scripts" / "run_lingbot_va_training.py"),
        "--lingbot-repo",
        str(repo),
        "--dataset-root",
        str(dataset),
        "--model-root",
        str(model_root),
        "--output-dir",
        str(output_dir.expanduser().resolve()),
        "--steps",
        str(steps),
        "--save-interval",
        str(int(model.get("save_interval", 1))),
        "--workers",
        str(int(model.get("workers", 0))),
        "--gradient-accumulation-steps",
        str(int(model.get("gradient_accumulation_steps", 1))),
    ]
    if resume is not None:
        argv.extend(["--resume-from", str(resume.expanduser().resolve())])
    env = {
        "CUDA_VISIBLE_DEVICES": gpus or str(model.get("gpus") or "0"),
        "PYTHONPATH": f"{process_repo}:{repo / 'wan_va'}",
        "TOKENIZERS_PARALLELISM": "false",
    }
    return CommandSpec(
        model="lingbot_va",
        phase=phase,
        argv=tuple(argv),
        cwd=repo,
        output_dir=output_dir.expanduser().resolve(),
        env=env,
        external_repo=repo,
    )
