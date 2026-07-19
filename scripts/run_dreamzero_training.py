#!/usr/bin/env python3
"""Launch upstream DreamZero while allowing an explicit Trainer checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


def main() -> None:
    arguments = sys.argv[1:]
    try:
        separator = arguments.index("--")
    except ValueError as exc:
        raise SystemExit("DreamZero wrapper requires '--' before Hydra overrides") from exc
    wrapper_arguments = arguments[:separator]
    hydra_arguments = arguments[separator + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dreamzero-repo", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(wrapper_arguments)

    repo = args.dreamzero_repo.expanduser().resolve()
    experiment = repo / "groot" / "vla" / "experiment" / "experiment.py"
    if not experiment.is_file():
        raise SystemExit(f"DreamZero experiment entry is missing: {experiment}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import groot.vla.experiment.base as base

    # Upstream BaseExperiment always calls the generic final saver, which emits
    # the complete 16.5B model even when save_lora_only=true. Step checkpoints
    # already use DreamZero's LoRA-aware Trainer._save and contain all resume
    # state, so do not duplicate a full model at the output root.
    def skip_generic_full_model_save(*, trainer, output_dir):
        del trainer
        print(f"Skipping generic full-model save for LoRA run: {output_dir}")

    if "save_lora_only=true" in hydra_arguments:
        base.safe_save_model_for_hf_trainer = skip_generic_full_model_save

    if args.resume_from is not None:
        checkpoint = args.resume_from.expanduser().resolve()
        if not checkpoint.is_dir():
            raise SystemExit(f"DreamZero checkpoint is not a directory: {checkpoint}")

        def explicit_checkpoint(_output_dir: str, checkpoint_prefix: str = "checkpoint"):
            del checkpoint_prefix
            return str(checkpoint), True

        base.get_checkpoint_path = explicit_checkpoint
        # BaseExperiment converts the selected path to a bool. BaseTrainer then
        # calls its module-level get_last_checkpoint again, so pin that lookup as
        # well instead of silently selecting a newer sibling checkpoint.
        base.get_last_checkpoint = lambda _output_dir: str(checkpoint)

    sys.argv = [str(experiment), *hydra_arguments]
    try:
        runpy.run_path(str(experiment), run_name="__main__")
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
