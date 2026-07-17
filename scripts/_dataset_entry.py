from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from fastwam_preprocess.cli import main


def run(dataset: str) -> None:
    arguments = sys.argv[1:]
    if arguments and arguments[0] in {"pipeline", "full"}:
        from run_pipeline import main as run_pipeline

        run_pipeline(["--datasets", dataset, *arguments[1:]])
        return
    if arguments and arguments[0] == "scan":
        arguments = arguments[1:]
    main(["scan", "--dataset", dataset, *arguments])
