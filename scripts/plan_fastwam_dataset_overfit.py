#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning.fastwam_dataset_overfit import build_dataset_overfit_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic cross-episode FastWAM overfit probes"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-probe-count", type=int, default=8)
    args = parser.parse_args()

    report = build_dataset_overfit_plan(
        args.manifest,
        training_probe_count=args.training_probe_count,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
