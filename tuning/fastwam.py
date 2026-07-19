from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .common import (
    CommandSpec,
    TuningConfigError,
    hydra_list,
    model_config,
    required_path,
)


def _require_memory_joint_inference(repo: Path) -> None:
    model_path = repo / "src" / "fastwam" / "models" / "wan22" / "memory_fastwam.py"
    trainer_path = repo / "src" / "fastwam" / "trainer.py"
    if not model_path.is_file() or not trainer_path.is_file():
        raise TuningConfigError(
            "FastWAM overfit requires the custom MemoryFastWAM model and trainer: "
            f"missing {model_path if not model_path.is_file() else trainer_path}"
        )
    tree = ast.parse(model_path.read_text(encoding="utf-8"), filename=str(model_path))
    memory_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MemoryFastWAM"
        ),
        None,
    )
    methods = {
        node.name: {argument.arg for argument in node.args.args}
        for node in (memory_class.body if memory_class is not None else [])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required_memory_args = {
        "memory_video_long",
        "memory_video_mid",
        "memory_video_short",
        "memory_mask_long",
        "memory_mask_mid",
        "memory_mask_short",
    }
    for method in ("infer_joint", "infer"):
        if not required_memory_args.issubset(methods.get(method, set())):
            raise TuningConfigError(
                f"FastWAM {method} is not memory-aware in {model_path}; "
                "install the Process_WM_Data MemoryFastWAM integration before overfit."
            )
    trainer_source = trainer_path.read_text(encoding="utf-8")
    for capability in ("eval_fixed_index", "eval_at_start", "memory_video_long"):
        if capability not in trainer_source:
            raise TuningConfigError(
                f"FastWAM trainer lacks required overfit capability {capability!r}: "
                f"{trainer_path}"
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
    if bool(phase_cfg.get("requires_memory_joint_inference", phase == "overfit")):
        _require_memory_joint_inference(repo)
    if resume is not None and not resume.expanduser().resolve().exists():
        raise TuningConfigError(f"FastWAM resume checkpoint is missing: {resume}")

    allowed_modes = [str(value) for value in phase_cfg.get("allowed_modes", ["joint_video_action"])]
    allowed_tiers = [str(value) for value in phase_cfg.get("allowed_quality_tiers", ["A"])]
    include_robot = bool(phase_cfg.get("include_robot_supervision", True))
    save_every = int(phase_cfg.get("save_every", 1))
    eval_every = int(phase_cfg.get("eval_every", 0))
    if save_every < 0 or eval_every < 0:
        raise TuningConfigError("FastWAM save_every and eval_every must be non-negative")
    argv = [
        str(python),
        "scripts/train.py",
        f"task={task}",
        f"output_dir={output_dir.expanduser().resolve()}",
        f"max_steps={steps}",
        f"save_every={save_every}",
        "save_training_state="
        + str(bool(phase_cfg.get("save_training_state", True))).lower(),
        f"num_workers={int(phase_cfg.get('num_workers', 0))}",
        f"log_every={int(phase_cfg.get('log_every', 1))}",
        f"eval_every={eval_every}",
        f"eval_num_inference_steps={int(phase_cfg.get('eval_num_inference_steps', 10))}",
        f"eval_seed={int(phase_cfg.get('eval_seed', 42))}",
        "eval_at_start=" + str(bool(phase_cfg.get("eval_at_start", False))).lower(),
        "eval_use_train_dataset="
        + str(bool(phase_cfg.get("eval_use_train_dataset", False))).lower(),
        f"eval_video_fps={int(phase_cfg.get('eval_video_fps', 5))}",
        f"data.train.case_manifests={hydra_list([str(manifest)])}",
        f"data.train.data_root={data_root}",
        f"data.train.text_embedding_cache_dir={text_cache}",
        f"data.train.allowed_modes={hydra_list(allowed_modes)}",
        f"data.train.allowed_quality_tiers={hydra_list(allowed_tiers)}",
        f"data.train.include_robot_supervision={str(include_robot).lower()}",
        f"data.train.max_samples={int(phase_cfg.get('max_samples', 1))}",
        "wandb.enabled=false",
    ]
    for key in ("sample_offset", "sample_stride", "sample_offset_per_case"):
        if key in phase_cfg:
            argv.append(f"data.train.{key}={int(phase_cfg[key])}")
    if phase_cfg.get("eval_fixed_index") is not None:
        argv.append(f"eval_fixed_index={int(phase_cfg['eval_fixed_index'])}")
    scalar_overrides = {
        "batch_size": int,
        "gradient_accumulation_steps": int,
        "learning_rate": float,
        "reference_learning_rate": float,
        "weight_decay": float,
        "max_grad_norm": float,
    }
    for key, converter in scalar_overrides.items():
        if key in phase_cfg:
            argv.append(f"{key}={converter(phase_cfg[key])}")
    for key in ("lr_scheduler_type", "mixed_precision", "sampling_strategy"):
        if key in phase_cfg:
            argv.append(f"{key}={phase_cfg[key]}")
    for name in ("video", "action"):
        key = f"loss_lambda_{name}"
        if key in phase_cfg:
            value = float(phase_cfg[key])
            if value < 0:
                raise TuningConfigError(f"FastWAM {key} must be non-negative")
            argv.append(f"model.loss.lambda_{name}={value}")
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
