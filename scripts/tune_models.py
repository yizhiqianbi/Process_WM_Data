#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning.common import latest_status, load_tuning_config, run_command
from tuning.dreamzero import build_dreamzero_command
from tuning.fastwam import build_fastwam_command
from tuning.lingbot_va import build_lingbot_va_command


BUILDERS = {
    "fastwam": build_fastwam_command,
    "lingbot_va": build_lingbot_va_command,
    "dreamzero": build_dreamzero_command,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune FastWAM, old LingBot-VA, or DreamZero")
    parser.add_argument("command", choices=["dry-run", "run", "status"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--phase", default="finetune")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES value, for example 0 or 0,1")
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(latest_status(args.output_dir.expanduser().resolve()), indent=2))
        return

    config = load_tuning_config(args.config)
    spec = BUILDERS[args.model](
        config,
        phase=args.phase,
        output_dir=args.output_dir,
        steps=args.steps,
        resume=args.resume,
        gpus=args.gpus,
    )
    if args.command == "dry-run":
        print(json.dumps(spec.document(), indent=2, sort_keys=True))
        return
    receipt = run_command(spec)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
