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
    process_repo = Path(__file__).resolve().parents[1]
    wrapper = process_repo / "scripts" / "run_dreamzero_training.py"
    argv = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={int(model.get('num_gpus', 1))}",
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
            "training_args.warmup_ratio=0.0",
            "save_steps=1",
            f"output_dir={output_dir.expanduser().resolve()}",
            "per_device_train_batch_size=1",
            f"max_steps={steps}",
            "weight_decay=1e-5",
            "save_total_limit=5",
            "upload_checkpoints=false",
            "bf16=true",
            "tf32=true",
            "eval_bf16=true",
            "dataloader_pin_memory=false",
            "dataloader_num_workers=0",
            "dataloader_persistent_workers=false",
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
        "CUDA_VISIBLE_DEVICES": gpus or str(model.get("gpus") or "0"),
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
