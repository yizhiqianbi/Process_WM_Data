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
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        help="Optional earlier run whose step-0 evaluation is the experiment baseline",
    )
    parser.add_argument(
        "--step-offset",
        type=int,
        default=0,
        help="Add this many optimizer steps to candidate run steps",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--min-psnr-gain", type=float, default=3.0)
    parser.add_argument("--min-ssim-gain", type=float, default=0.05)
    parser.add_argument("--max-action-l1-ratio", type=float, default=0.5)
    args = parser.parse_args()
    if args.step_offset < 0:
        parser.error("--step-offset must be non-negative")

    records = load_eval_records(args.run_dir, rank=args.rank)
    if args.step_offset:
        for record in records:
            record["run_step"] = int(record["step"])
            record["step"] = int(record["step"]) + args.step_offset
    baseline_record = None
    if args.baseline_run_dir is not None:
        baseline_records = load_eval_records(args.baseline_run_dir, rank=args.rank)
        if not baseline_records:
            parser.error(
                f"no rank-{args.rank} evaluation records found in "
                f"{args.baseline_run_dir}"
            )
        baseline_record = baseline_records[0]
    report = summarize_overfit(
        records,
        baseline_record=baseline_record,
        min_psnr_gain=args.min_psnr_gain,
        min_ssim_gain=args.min_ssim_gain,
        max_action_l1_ratio=args.max_action_l1_ratio,
    )
    report["candidate_run_dir"] = str(args.run_dir.expanduser().resolve())
    report["candidate_step_offset"] = int(args.step_offset)
    if args.baseline_run_dir is not None:
        report["baseline_run_dir"] = str(args.baseline_run_dir.expanduser().resolve())
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
