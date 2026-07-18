#!/usr/bin/env python3
"""Decode one causal-memory FastWAM training case from every dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


DATASETS = (
    "oxe",
    "oxe_auge",
    "agibot_beta",
    "robocoin",
    "robomind",
    "galaxea",
    "interndata_a1",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fastwam-repo",
        type=Path,
        default=Path(os.environ.get("FASTWAM_REPO", PROJECT_ROOT.parent / "FastWAM")),
    )
    parser.add_argument(
        "--preprocess-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=None,
        help=(
            "Canonical train-split statistics. Default: "
            "PREPROCESS_ROOT/work/stage_pipeline/normalization_stats.json."
        ),
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _select_history_window(dataset) -> int:
    return next((index for index, ref in enumerate(dataset.samples) if ref.start > 0), 0)


def _validate_dataset(
    name: str,
    preprocess_root: Path,
    normalization_stats: Path,
):
    from fastwam.datasets.training_case_dataset import MemoryTrainingCaseDataset

    manifest = preprocess_root / "work/stage_pipeline" / name / "cases/training_cases.jsonl"
    cache = preprocess_root / "work/stage_pipeline/text_embeds"
    tar_member_cache = preprocess_root / "work/stage_pipeline/video_member_cache"
    dataset = MemoryTrainingCaseDataset(
        case_manifests=[str(manifest)],
        data_root=str(preprocess_root),
        normalization_stats=str(normalization_stats),
        text_embedding_cache_dir=str(cache),
        context_len=128,
        split="train",
        allowed_modes=["joint_video_action", "video_only"],
        allowed_quality_tiers=["A", "B"],
        action_dim=80,
        proprio_dim=80,
        video_size=[384, 320],
        camera_roles=["global_primary", "left_wrist", "right_wrist"],
        missing_camera_policy="zero",
        normalization_clip=10.0,
        episode_cache_size=1,
        tar_member_cache_dir=str(tar_member_cache),
        include_robot_supervision=True,
        memory_history_long=8,
        memory_history_mid=2,
        memory_history_short=1,
    )
    index = _select_history_window(dataset)
    sample = dataset[index]
    ref = dataset.samples[index]
    case = dataset.cases[ref.case_index]

    expected_shapes = {
        "video": (3, 21, 384, 320),
        "action": (80, 80),
        "proprio": (80, 80),
        "context": (128, 4096),
        "memory_video_long": (3, 8, 384, 320),
        "memory_video_mid": (3, 2, 384, 320),
        "memory_video_short": (3, 1, 384, 320),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(sample[key].shape)
        if actual != expected:
            raise ValueError(f"{name}: {key} expected {expected}, got {actual}")
        if not torch.isfinite(sample[key]).all():
            raise ValueError(f"{name}: {key} contains non-finite values")

    memory_masks = torch.cat(
        [sample["memory_mask_long"], sample["memory_mask_mid"], sample["memory_mask_short"]]
    )
    memory_indices = torch.cat(
        [
            sample["memory_indices_long"],
            sample["memory_indices_mid"],
            sample["memory_indices_short"],
        ]
    )
    window_start = int(sample["window_start"])
    if bool(memory_masks.any()) and int(memory_indices[memory_masks].max()) >= window_start:
        raise ValueError(
            f"{name}: memory leaks current/future data: indices={memory_indices.tolist()}, "
            f"window_start={window_start}"
        )

    action_dimensions = int((~sample["action_dim_is_pad"]).sum())
    state_dimensions = int((~sample["proprio_dim_is_pad"]).sum())
    action_loss_enabled = bool(sample["action_loss_mask"])
    expected_action_loss = case["training"]["mode"] == "joint_video_action"
    if action_loss_enabled != expected_action_loss:
        raise ValueError(
            f"{name}: action loss={action_loss_enabled} conflicts with mode={case['training']['mode']}"
        )
    if expected_action_loss and (action_dimensions == 0 or state_dimensions == 0):
        raise ValueError(f"{name}: joint case has no active action/state dimensions")
    if not expected_action_loss and (action_dimensions != 0 or state_dimensions != 0):
        raise ValueError(f"{name}: video-only case exposed robot supervision")

    return {
        "status": "passed",
        "dataset": name,
        "case_id": case["case_id"],
        "quality_tier": case["quality"]["tier"],
        "training_mode": case["training"]["mode"],
        "windows": len(dataset),
        "validated_sample_index": index,
        "window_start": window_start,
        "state_steps": 81,
        "action_steps": 80,
        "video_steps": 21,
        "video_shape": list(sample["video"].shape),
        "context_shape": list(sample["context"].shape),
        "memory_shapes": {
            "long": list(sample["memory_video_long"].shape),
            "mid": list(sample["memory_video_mid"].shape),
            "short": list(sample["memory_video_short"].shape),
        },
        "memory_valid_count": int(memory_masks.sum()),
        "memory_indices": memory_indices.tolist(),
        "causal_memory_passed": True,
        "camera_present_mask": sample["camera_present_mask"].tolist(),
        "active_action_dimensions": action_dimensions,
        "active_state_dimensions": state_dimensions,
        "active_action_slots": torch.where(~sample["action_dim_is_pad"])[0].tolist(),
        "active_state_slots": torch.where(~sample["proprio_dim_is_pad"])[0].tolist(),
        "action_loss_enabled": action_loss_enabled,
        "video_loss_enabled": bool(sample["video_loss_mask"]),
        "video_range": [float(sample["video"].min()), float(sample["video"].max())],
        "tar_member_cache_dir": str(tar_member_cache),
    }


def main() -> int:
    args = parse_args()
    fastwam_src = args.fastwam_repo.expanduser().resolve() / "src"
    if not fastwam_src.is_dir():
        raise FileNotFoundError(f"FastWAM source directory does not exist: {fastwam_src}")
    sys.path.insert(0, str(fastwam_src))

    preprocess_root = args.preprocess_root.expanduser().resolve()
    normalization_stats = (
        args.normalization_stats.expanduser().resolve()
        if args.normalization_stats is not None
        else preprocess_root / "work/stage_pipeline/normalization_stats.json"
    )
    if not normalization_stats.is_file():
        raise FileNotFoundError(
            "Normalization statistics do not exist. Build train/A/joint statistics first: "
            f"{normalization_stats}"
        )
    output = args.output
    if output is None:
        output = preprocess_root / "work/stage_pipeline/validation/all_datasets.json"
    output = output.expanduser().resolve()

    results = []
    for name in args.datasets:
        try:
            result = _validate_dataset(name, preprocess_root, normalization_stats)
        except Exception as exc:
            result = {
                "status": "failed",
                "dataset": name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if args.fail_fast:
                raise
        results.append(result)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)

    passed = sum(item["status"] == "passed" for item in results)
    report = {
        "schema_version": "fastwam-multidataset-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed == len(results) else "failed",
        "passed_datasets": passed,
        "total_datasets": len(results),
        "contract": {
            "state_steps": 81,
            "action_steps": 80,
            "video_steps": 21,
            "target_fps": 20.0,
            "memory_history": {"long": 8, "mid": 2, "short": 1},
        },
        "normalization_stats": str(normalization_stats),
        "datasets": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"validation_report={output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
