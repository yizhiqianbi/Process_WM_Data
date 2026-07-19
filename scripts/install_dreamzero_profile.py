#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from targets.dreamzero import install_dreamzero_training_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a generated DreamZero data profile")
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--dreamzero-repo", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = install_dreamzero_training_profile(
        args.target_root,
        args.dreamzero_repo,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
