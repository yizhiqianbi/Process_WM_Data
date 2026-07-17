from __future__ import annotations

import re

_ROLE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("left_wrist", ("left_wrist", "wrist_left", "hand_left", "left_hand", "left_handeye")),
    ("right_wrist", ("right_wrist", "wrist_right", "hand_right", "right_hand", "right_handeye")),
    (
        "global_secondary",
        (
            "head_right",
            "right_head",
            "camera_right",
            "side",
            "external_right",
            "camera_left",
            "external_left",
        ),
    ),
    (
        "global_primary",
        (
            "head",
            "front",
            "top",
            "external",
            "overview",
            "agentview",
            "image",
            "rgb",
        ),
    ),
    ("auxiliary", ("chest", "handeye", "rear", "back", "aux")),
)


def normalize_camera_role(source_key: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", source_key.lower()).strip("_")
    for role, needles in _ROLE_RULES:
        if any(needle in key for needle in needles):
            return role
    return "auxiliary"


def normalized_camera_slots(keys: list[str]) -> dict[str, str]:
    """Map source keys to stable roles without silently dropping collisions."""
    result: dict[str, str] = {}
    used: dict[str, int] = {}
    for key in keys:
        base_role = normalize_camera_role(key)
        count = used.get(base_role, 0)
        used[base_role] = count + 1
        role = base_role if count == 0 else f"{base_role}_{count + 1}"
        result[key] = role
    return result

