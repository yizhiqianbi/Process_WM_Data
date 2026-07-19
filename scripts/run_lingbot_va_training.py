#!/usr/bin/env python3
"""Portable old LingBot-VA trainer with complete checkpoint resume state."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import random
import sys
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from targets.lingbot_va.runtime import install_flash_attention_import_fallback


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train old Robbyant/LingBot-VA")
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--window-frames", type=int, default=0)
    parser.add_argument("--samples-per-episode", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class FixedLatentWindowDataset(Dataset):
    def __init__(self, dataset: Dataset, *, window_frames: int, samples_per_episode: int):
        if window_frames <= 0 or samples_per_episode <= 0:
            raise ValueError("window_frames and samples_per_episode must be positive")
        if len(dataset) == 0:
            raise ValueError("LingBot-VA dataset is empty")
        self.dataset = dataset
        self.window_frames = window_frames
        self.samples_per_episode = samples_per_episode

    def __len__(self) -> int:
        return len(self.dataset) * self.samples_per_episode

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index % len(self.dataset)]
        total_frames = int(item["latents"].shape[1])
        if total_frames < self.window_frames:
            raise ValueError(
                f"episode has {total_frames} latent frames, shorter than "
                f"window_frames={self.window_frames}"
            )
        max_start = total_frames - self.window_frames
        start = 0 if max_start == 0 else int(torch.randint(max_start + 1, (1,)).item())
        end = start + self.window_frames
        result = dict(item)
        for key in ("latents", "actions", "actions_mask"):
            result[key] = item[key][:, start:end].contiguous()
        return result


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.save_interval <= 0:
        raise SystemExit("steps and save interval must be positive")
    if args.batch_size <= 0 or args.samples_per_episode <= 0:
        raise SystemExit("batch size and samples per episode must be positive")
    if args.batch_size > 1 and args.window_frames <= 0:
        raise SystemExit("batch size > 1 requires a positive fixed window size")
    repo = args.lingbot_repo.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    profile_path = dataset_root / "meta" / "lingbot_va_model_profile.json"
    for path, label in (
        (repo / "wan_va" / "train.py", "LingBot train entry"),
        (profile_path, "model profile"),
        (dataset_root / "empty_emb.pt", "empty embedding"),
        (model_root / "transformer", "base transformer"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} is missing: {path}")

    wan_va = repo / "wan_va"
    if str(wan_va) not in sys.path:
        sys.path.insert(0, str(wan_va))
    install_flash_attention_import_fallback(torch)
    upstream = importlib.import_module("train")
    upstream.init_logger()
    latent_dataset_module = importlib.import_module("dataset.lerobot_latent_dataset")

    # The upstream multi-dataset wrapper always creates a 128-process Pool after
    # NCCL/model initialization. A single prepared target needs no discovery or
    # multiprocessing, and forking there can crash UCX/libgomp outright.
    def build_single_dataset(config):
        dataset = latent_dataset_module.LatentLeRobotDataset(
            str(dataset_root), config=config
        )
        if args.window_frames > 0:
            dataset = FixedLatentWindowDataset(
                dataset,
                window_frames=args.window_frames,
                samples_per_episode=args.samples_per_episode,
            )
            upstream.logger.info(
                "Fixed-window dataset: window=%d, samples/episode=%d, samples=%d",
                args.window_frames,
                args.samples_per_episode,
                len(dataset),
            )
        return dataset

    upstream.MultiLatentLeRobotDataset = build_single_dataset

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    config = deepcopy(upstream.VA_CONFIGS["demo_train"])
    for key, value in profile.items():
        if key != "schema_version":
            config[key] = value
    config.dataset_path = str(dataset_root)
    config.empty_emb_path = str(dataset_root / "empty_emb.pt")
    config.wan22_pretrained_model_name_or_path = str(model_root)
    config.save_root = str(output_dir)
    config.enable_wandb = False
    config.load_worker = args.workers
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.num_steps = args.steps
    config.save_interval = args.save_interval
    config.gc_interval = max(1, args.save_interval)
    config.cfg_prob = 0.0
    config.resume_from = (
        None if args.resume_from is None else str(args.resume_from.expanduser().resolve())
    )
    config.rank = int(os.environ.get("RANK", "0"))
    config.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    config.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    upstream.init_distributed(config.world_size, config.local_rank, config.rank)
    seed = args.seed + config.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    class ResumableTrainer(upstream.Trainer):
        def __init__(self, trainer_config):
            super().__init__(trainer_config)
            if trainer_config.resume_from:
                self._load_complete_training_state(trainer_config.resume_from)

        def _train_step(self, batch, batch_idx):
            torch.cuda.reset_peak_memory_stats(self.device)
            losses = super()._train_step(batch, batch_idx)
            peak = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(self.device),
                    torch.cuda.max_memory_reserved(self.device),
                ],
                dtype=torch.float64,
                device=self.device,
            )
            if dist.is_initialized():
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if losses.get("should_log") and self.config.rank == 0:
                gib = 1024**3
                upstream.logger.info(
                    "Step %d CUDA peak: allocated=%.2f GiB, reserved=%.2f GiB",
                    self.step + 1,
                    peak[0].item() / gib,
                    peak[1].item() / gib,
                )
            return losses

        def save_checkpoint(self):
            options = upstream.StateDictOptions(full_state_dict=True, cpu_offload=True)
            model_state = upstream.get_model_state_dict(self.transformer, options=options)
            optimizer_state = upstream.get_optimizer_state_dict(
                self.transformer, self.optimizer, options=options
            )
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                upstream.save_file(
                    {key: value.to(torch.bfloat16) for key, value in model_state.items()},
                    model_file,
                )
                transformer_config = dict(self.transformer.config)
                transformer_config.pop("_name_or_path", None)
                (transformer_dir / "config.json").write_text(
                    json.dumps(transformer_config, indent=2) + "\n", encoding="utf-8"
                )
                state_path = checkpoint_dir / "training_state.pt"
                temporary = state_path.parent / f".{state_path.name}.{uuid.uuid4().hex}.tmp"
                torch.save(
                    {
                        "schema_version": "lingbot-va-training-state-v1",
                        "step": self.step,
                        "optimizer_state_dict": optimizer_state,
                        "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                    },
                    temporary,
                )
                os.replace(temporary, state_path)
                upstream.logger.info("Saved complete checkpoint to %s", checkpoint_dir)
            if dist.is_initialized():
                dist.barrier()

        def _load_complete_training_state(self, checkpoint_path):
            state_path = Path(checkpoint_path) / "training_state.pt"
            if not state_path.is_file():
                raise FileNotFoundError(f"complete training state is missing: {state_path}")
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer_state = state["optimizer_state_dict"]
            # Adam creates a slot only after a parameter receives its first
            # gradient. DCP still indexes every parameter listed in a param
            # group, so represent untouched slots explicitly as empty state.
            parameter_state = optimizer_state.setdefault("state", {})
            for group in optimizer_state.get("param_groups", []):
                for parameter_name in group.get("params", []):
                    parameter_state.setdefault(parameter_name, {})
            upstream.set_optimizer_state_dict(
                self.transformer,
                self.optimizer,
                optim_state_dict=optimizer_state,
                options=upstream.StateDictOptions(full_state_dict=True, strict=False),
            )
            self.lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
            self.step = int(state["step"])
            upstream.logger.info("Resumed complete training state at step %d", self.step)
            if dist.is_initialized():
                dist.barrier()

    if config.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved = {
            "schema_version": "lingbot-va-resolved-training-v1",
            "dataset_root": str(dataset_root),
            "model_root": str(model_root),
            "steps": args.steps,
            "resume_from": config.resume_from,
            "world_size": config.world_size,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * config.world_size,
            "window_frames": args.window_frames,
            "samples_per_episode": args.samples_per_episode,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "save_interval": args.save_interval,
            "profile": profile,
        }
        (output_dir / "resolved_training_config.json").write_text(
            json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
        )
    try:
        trainer = ResumableTrainer(config)
        trainer.train()
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
