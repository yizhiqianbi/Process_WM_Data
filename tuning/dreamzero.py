from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import CommandSpec, TuningConfigError, model_config, required_path


def build_dreamzero_command(
    config: dict[str, Any],
    *,
    phase: str,
    output_dir: Path,
    steps: int,
    resume: Path | None,
    gpus: str | None,
) -> CommandSpec:
    if phase != "finetune":
        raise TuningConfigError("DreamZero supports phase=finetune")
    if steps <= 0:
        raise TuningConfigError("steps must be positive")
    model = model_config(config, "dreamzero")
    repo = required_path(model, "repo", directory=True)
    python = required_path(model, "python", directory=False)
    dataset = required_path(model, "dataset_root", directory=True)
    data_profile = str(model.get("data_profile") or "dreamzero/xdof_relative")
    profile_path = repo / "groot" / "vla" / "configs" / "data" / f"{data_profile}.yaml"
    if not profile_path.is_file():
        raise TuningConfigError(
            f"DreamZero Hydra data profile is not installed: {profile_path}"
        )
    if resume is not None and not resume.expanduser().resolve().is_dir():
        raise TuningConfigError(f"DreamZero resume checkpoint is missing: {resume}")
    num_gpus = int(model.get("num_gpus", 1))
    batch_size = int(model.get("batch_size", 1))
    configured_save_interval = int(model.get("save_interval", steps))
    save_interval = min(configured_save_interval, steps)
    save_total_limit = int(model.get("save_total_limit", 5))
    workers = int(model.get("workers", 0))
    if num_gpus <= 0 or batch_size <= 0:
        raise TuningConfigError("DreamZero num_gpus and batch_size must be positive")
    if configured_save_interval <= 0 or save_total_limit < 5 or workers < 0:
        raise TuningConfigError(
            "DreamZero save_interval must be positive, save_total_limit must be at least 5, "
            "and workers must be non-negative"
        )
    visible_gpus = gpus or str(model.get("gpus") or "0")
    gpu_ids = [value.strip() for value in visible_gpus.split(",") if value.strip()]
    if len(gpu_ids) != num_gpus:
        raise TuningConfigError(
            "DreamZero num_gpus must match CUDA_VISIBLE_DEVICES: "
            f"num_gpus={num_gpus}, devices={gpu_ids}"
        )
    per_step_batch = batch_size * num_gpus
    global_batch_size = int(model.get("global_batch_size", per_step_batch))
    if global_batch_size <= 0 or global_batch_size % per_step_batch != 0:
        raise TuningConfigError(
            "DreamZero global_batch_size must be a positive multiple of "
            f"batch_size * num_gpus ({per_step_batch})"
        )
    process_repo = Path(__file__).resolve().parents[1]
    wrapper = process_repo / "scripts" / "run_dreamzero_training.py"
    argv = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        str(wrapper),
        "--dreamzero-repo",
        str(repo),
    ]
    if resume is not None:
        argv.extend(["--resume-from", str(resume.expanduser().resolve())])
    argv.append("--")
    argv.extend(
        [
            "report_to=none",
            f"data={data_profile}",
            "wandb_project=dreamzero",
            "train_architecture=lora",
            "num_frames=33",
            "action_horizon=24",
            "num_views=3",
            "model=dreamzero/vla",
            "model/dreamzero/action_head=wan_flow_matching_action_tf",
            "model/dreamzero/transform=dreamzero_cotrain",
            "num_frame_per_block=2",
            "num_action_per_block=24",
            "num_state_per_block=1",
            f"training_args.learning_rate={float(model.get('learning_rate', 1e-5))}",
            'training_args.deepspeed=""',
            f"training_args.warmup_ratio={float(model.get('warmup_ratio', 0.05))}",
            f"save_steps={save_interval}",
            f"output_dir={output_dir.expanduser().resolve()}",
            f"per_device_train_batch_size={batch_size}",
            f"global_batch_size={global_batch_size}",
            "raise_error_if_global_batch_size_not_set=true",
            f"max_steps={steps}",
            "weight_decay=1e-5",
            f"save_total_limit={save_total_limit}",
            "upload_checkpoints=false",
            "bf16=true",
            "tf32=true",
            "eval_bf16=true",
            "dataloader_pin_memory=false",
            f"dataloader_num_workers={workers}",
            f"dataloader_persistent_workers={'true' if workers > 0 else 'false'}",
            f"image_resolution_width={int(model.get('image_width', 320))}",
            f"image_resolution_height={int(model.get('image_height', 176))}",
            "save_lora_only=true",
            "max_chunk_size=4",
            "frame_seqlen=880",
            "save_strategy=steps",
            f"xdof_data_root={dataset}",
            f"dit_version={required_path(model, 'wan_model_root', directory=True)}",
            f"text_encoder_pretrained_path={required_path(model, 'text_encoder', directory=False)}",
            f"image_encoder_pretrained_path={required_path(model, 'image_encoder', directory=False)}",
            f"vae_pretrained_path={required_path(model, 'vae', directory=False)}",
            f"tokenizer_path={required_path(model, 'tokenizer_root', directory=True)}",
            f"pretrained_model_path={required_path(model, 'pretrained_model_root', directory=True)}",
            "++action_head_cfg.config.skip_component_loading=true",
            "++action_head_cfg.config.defer_lora_injection=true",
        ]
    )
    env = {
        "ATTENTION_BACKEND": "torch",
        "CUDA_VISIBLE_DEVICES": visible_gpus,
        "HYDRA_FULL_ERROR": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PYTHONPATH": str(repo),
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
    }
    return CommandSpec(
        model="dreamzero",
        phase=phase,
        argv=tuple(argv),
        cwd=repo,
        output_dir=output_dir.expanduser().resolve(),
        env=env,
        external_repo=repo,
    )
