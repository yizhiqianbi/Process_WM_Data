#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning.common import load_tuning_config, model_config, required_path


def _frame_to_image(video) -> Image.Image:
    frame = (
        ((video[:, 0].detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .permute(1, 2, 0)
        .numpy()
        .round()
        .astype(np.uint8)
    )
    return Image.fromarray(frame, mode="RGB")


def _labeled(image: Image.Image, label: str) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + 32), color=(16, 16, 16))
    result.paste(image, (0, 32))
    ImageDraw.Draw(result).text((8, 9), label, fill=(255, 255, 255))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview raw and rectified FastWAM Tianji composites"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", default="dataset_overfit")
    parser.add_argument("--sample-index", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_tuning_config(args.config)
    model = model_config(config, "fastwam")
    phase = (model.get("phases") or {}).get(args.phase)
    if not isinstance(phase, dict):
        raise ValueError(f"Missing FastWAM phase {args.phase!r}")
    fastwam_repo = required_path(model, "repo", directory=True)
    sys.path.insert(0, str(fastwam_repo / "src"))
    from fastwam.datasets.training_case_dataset import TrainingCaseDataset

    splits = phase.get("splits", phase.get("split", "train"))
    common = {
        "case_manifests": str(required_path(model, "case_manifest", directory=False)),
        "data_root": str(required_path(model, "data_root", directory=True)),
        "normalization_stats": str(
            required_path(model, "normalization_stats", directory=False)
        ),
        "text_embedding_cache_dir": str(
            Path(str(model["text_embedding_cache_dir"])).expanduser().resolve()
        ),
        "split": splits,
        "allowed_modes": phase.get("allowed_modes", ["joint_video_action"]),
        "allowed_quality_tiers": phase.get("allowed_quality_tiers", ["A"]),
        "include_robot_supervision": bool(phase.get("include_robot_supervision", True)),
        "max_samples": phase.get("max_samples"),
    }
    raw_dataset = TrainingCaseDataset(**common)
    rectification = phase.get("camera_rectification_config") or model.get(
        "camera_rectification_config"
    )
    if not rectification:
        raise ValueError(f"FastWAM phase {args.phase!r} has no camera rectification config")
    rectified_dataset = TrainingCaseDataset(
        **common,
        camera_rectification_config=str(Path(str(rectification)).expanduser().resolve()),
    )
    if args.sample_index < 0 or args.sample_index >= len(raw_dataset):
        raise IndexError(
            f"sample-index {args.sample_index} outside dataset of size {len(raw_dataset)}"
        )

    raw_sample = raw_dataset[args.sample_index]
    rectified_sample = rectified_dataset[args.sample_index]
    raw = _labeled(_frame_to_image(raw_sample["video"]), "RAW FISHEYE")
    rectified = _labeled(
        _frame_to_image(rectified_sample["video"]), "RECTIFIED VIRTUAL PINHOLE"
    )
    comparison = Image.new(
        "RGB", (raw.width + rectified.width, raw.height), color=(16, 16, 16)
    )
    comparison.paste(raw, (0, 0))
    comparison.paste(rectified, (raw.width, 0))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output)
    ref = rectified_dataset.samples[args.sample_index]
    case = rectified_dataset.cases[ref.case_index]
    receipt = {
        "schema_version": "fastwam-rectification-preview-v1",
        "output": str(output),
        "sample_index": args.sample_index,
        "case_id": case["case_id"],
        "source_episode_id": case["source_episode_id"],
        "window_start": ref.start,
        "camera_present_mask": rectified_sample["camera_present_mask"].tolist(),
        "camera_rectification_applied_mask": rectified_sample[
            "camera_rectification_applied_mask"
        ].tolist(),
        "camera_rectification": rectified_dataset.camera_rectification_metadata,
    }
    receipt_path = output.with_suffix(".json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
