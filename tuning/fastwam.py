from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    CommandSpec,
    TuningConfigError,
    hydra_list,
    model_config,
    required_path,
)


def build_fastwam_command(
    config: dict[str, Any],
    *,
    phase: str,
    output_dir: Path,
    steps: int,
    resume: Path | None,
    gpus: str | None,
) -> CommandSpec:
    model = model_config(config, "fastwam")
    repo = required_path(model, "repo", directory=True)
    python = required_path(model, "python", directory=False)
    manifest = required_path(model, "case_manifest", directory=False)
    data_root = required_path(model, "data_root", directory=True)
    text_cache = Path(str(model.get("text_embedding_cache_dir") or "")).expanduser().resolve()
    if not str(model.get("text_embedding_cache_dir") or "").strip():
        raise TuningConfigError("missing models.fastwam.text_embedding_cache_dir")
    phase_cfg = (model.get("phases") or {}).get(phase)
    if not isinstance(phase_cfg, dict):
        raise TuningConfigError(f"missing models.fastwam.phases.{phase}")
    task = str(phase_cfg.get("task") or "").strip()
    if not task:
        raise TuningConfigError(f"models.fastwam.phases.{phase}.task is required")
    if steps <= 0:
        raise TuningConfigError("steps must be positive")
    if resume is not None and not resume.expanduser().resolve().exists():
        raise TuningConfigError(f"FastWAM resume checkpoint is missing: {resume}")

    allowed_modes = [str(value) for value in phase_cfg.get("allowed_modes", ["joint_video_action"])]
    allowed_tiers = [str(value) for value in phase_cfg.get("allowed_quality_tiers", ["A"])]
    include_robot = bool(phase_cfg.get("include_robot_supervision", True))
    argv = [
        str(python),
        "scripts/train.py",
        f"task={task}",
        f"output_dir={output_dir.expanduser().resolve()}",
        f"max_steps={steps}",
        "save_every=1",
        "save_training_state=true",
        "num_workers=0",
        f"data.train.case_manifests={hydra_list([str(manifest)])}",
        f"data.train.data_root={data_root}",
        f"data.train.text_embedding_cache_dir={text_cache}",
        f"data.train.allowed_modes={hydra_list(allowed_modes)}",
        f"data.train.allowed_quality_tiers={hydra_list(allowed_tiers)}",
        f"data.train.include_robot_supervision={str(include_robot).lower()}",
        f"data.train.max_samples={int(phase_cfg.get('max_samples', 1))}",
        "wandb.enabled=false",
    ]
    normalization = model.get("normalization_stats")
    if include_robot:
        if not normalization:
            raise TuningConfigError("FastWAM robot supervision requires normalization_stats")
        normalization_path = required_path(model, "normalization_stats", directory=False)
        argv.append(f"data.train.normalization_stats={normalization_path}")
    else:
        argv.append("data.train.normalization_stats=null")

    if "skip_dit_load_from_pretrain" in phase_cfg:
        argv.append(
            "model.skip_dit_load_from_pretrain="
            + str(bool(phase_cfg["skip_dit_load_from_pretrain"])).lower()
        )
    video_checkpoint = phase_cfg.get("video_dit_pretrained_path") or model.get(
        "stage1_video_checkpoint"
    )
    if video_checkpoint:
        argv.append(f"model.video_dit_pretrained_path={Path(str(video_checkpoint)).expanduser().resolve()}")
    action_checkpoint = model.get("action_dit_checkpoint")
    if action_checkpoint:
        argv.append(f"model.action_dit_pretrained_path={Path(str(action_checkpoint)).expanduser().resolve()}")

    selected_resume = (
        resume.expanduser().resolve()
        if resume is not None
        else (
            Path(str(phase_cfg["initial_checkpoint"])).expanduser().resolve()
            if phase_cfg.get("initial_checkpoint")
            else None
        )
    )
    argv.append("resume=null" if selected_resume is None else f"resume={selected_resume}")

    diffsynth_root = required_path(model, "diffsynth_model_base", directory=True)
    env = {
        "CUDA_VISIBLE_DEVICES": gpus or str(model.get("gpus") or "0"),
        "DIFFSYNTH_MODEL_BASE_PATH": str(diffsynth_root),
        "FASTWAM_PREPROCESS_ROOT": str(data_root),
        "PYTHONPATH": str(repo / "src"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    if model.get("stage1_video_checkpoint"):
        env["FASTWAM_STAGE1_VIDEO_CKPT"] = str(
            Path(str(model["stage1_video_checkpoint"])).expanduser().resolve()
        )
    if action_checkpoint:
        env["FASTWAM_ACTION_DIT_PATH"] = str(
            Path(str(action_checkpoint)).expanduser().resolve()
        )
    return CommandSpec(
        model="fastwam",
        phase=phase,
        argv=tuple(argv),
        cwd=repo,
        output_dir=output_dir.expanduser().resolve(),
        env=env,
        external_repo=repo,
    )
