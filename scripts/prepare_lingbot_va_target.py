from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from targets.lingbot_va.prepare import main


if __name__ == "__main__":
    default_profiles = PACKAGE_ROOT / "configs" / "targets" / "lingbot_va.yaml"
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "prepare" and "--profiles" not in arguments:
        arguments = [*arguments, "--profiles", str(default_profiles)]
    main(arguments)
