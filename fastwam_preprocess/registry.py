from __future__ import annotations

import os
from pathlib import Path
from typing import Type

from .adapters import (
    AgiBotBetaAdapter,
    BaseAdapter,
    GalaxeaAdapter,
    InternDataA1Adapter,
    OXEAdapter,
    OXEAugEAdapter,
    RoboCOINAdapter,
    RoboMINDAdapter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FASTWAM_DATA_ROOT", PROJECT_ROOT.parent)).expanduser().resolve()

ADAPTERS: dict[str, Type[BaseAdapter]] = {
    "oxe": OXEAdapter,
    "oxe_auge": OXEAugEAdapter,
    "agibot_beta": AgiBotBetaAdapter,
    "robocoin": RoboCOINAdapter,
    "robomind": RoboMINDAdapter,
    "galaxea": GalaxeaAdapter,
    "interndata_a1": InternDataA1Adapter,
}

DEFAULT_ROOTS = {
    "oxe": DATA_ROOT / "OXE_OpenX_Embodiment" / "jxu124_OpenX-Embodiment",
    "oxe_auge": DATA_ROOT / "OXE_AugE",
    "agibot_beta": DATA_ROOT / "AgiBot_Beta" / "AgiBotWorld-Beta",
    "robocoin": DATA_ROOT / "RoboCOIN",
    "robomind": DATA_ROOT / "RoboMIND" / "RoboMIND",
    "galaxea": DATA_ROOT / "Galaxea" / "Galaxea-Open-World-Dataset",
    "interndata_a1": DATA_ROOT / "InternData_A1" / "InternData-A1",
}
