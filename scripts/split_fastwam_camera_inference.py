#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning.fastwam_camera_views import (
    CAMERA_PANELS,
    camera_pair_frame,
    crop_camera_panel,
    decode_labeled_composite,
    panel_psnr,
    write_h264,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split final FastWAM composite inference into trained camera-view pairs."
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    eval_dir = args.eval_dir.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    prefix = f"step_{args.step:06d}"
    pattern = re.compile(
        rf"^{prefix}_rank_(?P<rank>\d{{3}})_eval_(?P<index>\d{{6}})_imagination\.mp4$"
    )
    imagination_paths = sorted(
        path for path in eval_dir.glob(f"{prefix}_rank_*_eval_*_imagination.mp4")
        if pattern.match(path.name)
    )
    if not imagination_paths:
        raise FileNotFoundError(f"No {prefix} imagination videos found in {eval_dir}.")

    records = []
    for imagination_path in imagination_paths:
        match = pattern.match(imagination_path.name)
        assert match is not None
        rank = int(match.group("rank"))
        eval_index = int(match.group("index"))
        execution_path = imagination_path.with_name(
            imagination_path.name.replace("_imagination.mp4", "_execution.mp4")
        )
        if not execution_path.is_file():
            raise FileNotFoundError(f"Missing execution video: {execution_path}")
        imagination, imagination_fps = decode_labeled_composite(imagination_path)
        execution, execution_fps = decode_labeled_composite(execution_path)
        if imagination_fps != execution_fps or len(imagination) != len(execution):
            raise ValueError(f"Prediction/GT timeline mismatch for eval index {eval_index}.")

        probe_dir = output_dir / f"eval_{eval_index:06d}"
        camera_results = []
        for panel in CAMERA_PANELS:
            predicted_panel = [crop_camera_panel(frame, panel) for frame in imagination]
            target_panel = [crop_camera_panel(frame, panel) for frame in execution]
            pair_frames = [
                camera_pair_frame(predicted, target, panel)
                for predicted, target in zip(predicted_panel, target_panel)
            ]
            output_path = probe_dir / f"{panel.role}_imagination_vs_execution.mp4"
            write_h264(output_path, pair_frames, imagination_fps)
            camera_results.append(
                {
                    "role": panel.role,
                    "source_key": panel.source_key,
                    "content": panel.content,
                    "source_panel_box_xyxy": list(panel.box),
                    "psnr_db": panel_psnr(predicted_panel, target_panel),
                    "video_path": str(output_path),
                }
            )
        records.append(
            {
                "rank": rank,
                "eval_index": eval_index,
                "frame_count": len(imagination),
                "fps": float(imagination_fps),
                "imagination_source": str(imagination_path),
                "execution_source": str(execution_path),
                "cameras": camera_results,
            }
        )

    receipt = {
        "schema_version": "fastwam-camera-view-inference-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": args.step,
        "checkpoint": str(checkpoint),
        "eval_dir": str(eval_dir),
        "probe_count": len(records),
        "trained_camera_views": [panel.role for panel in CAMERA_PANELS],
        "unavailable_camera_views": [
            {
                "role": "auxiliary",
                "source_key": "observation.images.left_wrist",
                "content": "secondary global/auxiliary view",
                "reason": "This source camera was not included in the trained three-panel FastWAM layout.",
            }
        ],
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "camera_inference_summary.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
