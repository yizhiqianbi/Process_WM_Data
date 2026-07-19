#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning.fastwam_overfit import (
    load_eval_records,
    summarize_overfit,
    write_overfit_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize fixed-sample FastWAM overfit evaluations"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--min-psnr-gain", type=float, default=3.0)
    parser.add_argument("--min-ssim-gain", type=float, default=0.05)
    parser.add_argument("--max-action-l1-ratio", type=float, default=0.5)
    args = parser.parse_args()

    records = load_eval_records(args.run_dir, rank=args.rank)
    report = summarize_overfit(
        records,
        min_psnr_gain=args.min_psnr_gain,
        min_ssim_gain=args.min_ssim_gain,
        max_action_l1_ratio=args.max_action_l1_ratio,
    )
    json_path, markdown_path = write_overfit_report(
        args.run_dir, report, output_dir=args.output_dir
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
