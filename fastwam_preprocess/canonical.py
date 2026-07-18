from __future__ import annotations

import re
from typing import Any

CANONICAL_DIM = 80
CANONICAL_BLOCKS = {
    "left_ee": [0, 6],
    "left_gripper": [6, 7],
    "right_ee": [7, 13],
    "right_gripper": [13, 14],
    "left_joint": [14, 21],
    "right_joint": [21, 28],
    "left_hand": [28, 40],
    "right_hand": [40, 52],
    "mobile_base": [52, 58],
    "torso_head": [58, 64],
    "reserved": [64, 80],
}

_AXIS = {"x": 0, "y": 1, "z": 2}
_ROTATION_AXIS = {"x": 3, "y": 4, "z": 5}


def _side(name: str) -> str | None:
    value = name.lower()
    if "left" in value:
        return "left"
    if "right" in value:
        return "right"
    return None


def _last_axis(name: str) -> str | None:
    match = re.search(r"(?:^|_)([xyz])(?:_[a-z]+)?$", name.lower())
    return match.group(1) if match else None


def _slot_for_name(
    name: str, *, kind: str, source_key: str = ""
) -> tuple[int, str, str] | None:
    value = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    context = re.sub(r"[^a-z0-9]+", "_", source_key.lower()).strip("_")
    side = _side(value) or _side(context)

    if side and "gripper" in value and any(
        token in value for token in ("open", "position", "command", "target", "state")
    ) or (
        side
        and "gripper" in value
        and any(token in context for token in ("gripper", "effector_position"))
    ):
        return (6 if side == "left" else 13, f"{side}_gripper", "high")

    zero_based_named_joint = re.search(r"(?:left|right)_joint_(\d+)$", value)
    if side and zero_based_named_joint and f"{side}_joint_position" in context:
        joint_index = int(zero_based_named_joint.group(1))
        if 0 <= joint_index < 7:
            base = 14 if side == "left" else 21
            return (base + joint_index, f"{side}_joint_{joint_index + 1}", "high")

    joint_match = re.search(r"(?:arm_)?joint_(\d+)", value)
    if side and joint_match:
        joint_index = int(joint_match.group(1)) - 1
        if 0 <= joint_index < 7:
            base = 14 if side == "left" else 21
            return (base + joint_index, f"{side}_joint_{joint_index + 1}", "high")

    array_joint_match = re.search(r"(?:arm_left|arm_right|left_arm|right_arm)_position_(\d+)", value)
    if side and array_joint_match:
        joint_index = int(array_joint_match.group(1))
        if 0 <= joint_index < 7:
            base = 14 if side == "left" else 21
            return (base + joint_index, f"{side}_joint_{joint_index + 1}", "high")

    zero_based_joint_match = re.search(r"(?:left|right)_arm_(\d+)(?:_|$)", value)
    if side and zero_based_joint_match and "joint" in context:
        joint_index = int(zero_based_joint_match.group(1))
        if 0 <= joint_index < 7:
            base = 14 if side == "left" else 21
            return (base + joint_index, f"{side}_joint_{joint_index + 1}", "high")

    axis = _last_axis(value)
    is_ee = any(token in value for token in ("eef", "ee_", "tcp", "end_effector"))
    tokens = set(value.split("_"))
    is_position = "pos" in tokens or "position" in tokens or "world_vector" in value
    is_rotation_vector = any(token in value for token in ("rotvec", "rotation_vector"))
    action_is_delta = any(token in value for token in ("delta", "relative"))
    if side and axis and is_ee and (kind == "state" or action_is_delta):
        base = 0 if side == "left" else 7
        if is_position:
            return (base + _AXIS[axis], f"{side}_ee_position_{axis}", "medium")
        if is_rotation_vector:
            return (
                base + _ROTATION_AXIS[axis],
                f"{side}_ee_rotation_vector_{axis}",
                "medium",
            )

    base_match = re.search(r"(?:base|chassis)_(?:velocity|vel|twist|delta)_([xyz])$", value)
    if base_match:
        return (52 + _AXIS[base_match.group(1)], "mobile_base", "high")

    chassis_twist = re.search(r"chassis_twist_(linear|angular)_([xyz])$", value)
    if chassis_twist:
        offset = 0 if chassis_twist.group(1) == "linear" else 3
        return (
            52 + offset + _AXIS[chassis_twist.group(2)],
            f"mobile_base_{chassis_twist.group(1)}_{chassis_twist.group(2)}",
            "high",
        )

    torso_match = re.search(r"(?:torso|waist|head)_(?:joint_)?(\d+)", value)
    if torso_match:
        index = int(torso_match.group(1)) - 1
        if 0 <= index < 6:
            return (58 + index, "torso_head", "high")
    torso_position = re.search(r"(?:torso|waist|head)_position_(\d+)", value)
    if torso_position:
        index = int(torso_position.group(1))
        if 0 <= index < 6:
            return (58 + index, f"torso_head_position_{index}", "high")
    torso_twist = re.search(r"(?:torso|waist|head)_twist_(linear|angular)_([xyz])$", value)
    if torso_twist:
        offset = 0 if torso_twist.group(1) == "linear" else 3
        return (
            58 + offset + _AXIS[torso_twist.group(2)],
            f"torso_head_{torso_twist.group(1)}_{torso_twist.group(2)}",
            "high",
        )

    # InternData names the dimensions only as pitch/lift and yaw/patch. The
    # feature key supplies the missing body-part context; `patch` is the
    # published metadata typo for pitch.
    if "waist" in context:
        if value == "pitch":
            return (58, "waist_pitch", "high")
        if value == "lift":
            return (59, "waist_lift", "high")
    if "head" in context:
        if value == "yaw":
            return (60, "head_yaw", "high")
        if value in {"pitch", "patch"}:
            return (61, "head_pitch", "high")
    return None


def infer_canonical_mapping(
    feature_schema: dict[str, Any], *, kind: str
) -> dict[str, Any]:
    """Infer only mappings justified by explicit semantic feature names."""
    mappings: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    used_slots: dict[int, dict[str, Any]] = {}
    for source_key, feature in feature_schema.items():
        names = feature.get("names") if isinstance(feature, dict) else None
        if not isinstance(names, list):
            continue
        for source_index, source_name in enumerate(names):
            match = _slot_for_name(
                str(source_name), kind=kind, source_key=str(source_key)
            )
            if match is None:
                continue
            canonical_index, semantic, confidence = match
            item = {
                "source_key": source_key,
                "source_index": source_index,
                "source_name": str(source_name),
                "canonical_index": canonical_index,
                "semantic": semantic,
                "confidence": confidence,
                "active": confidence == "high",
            }
            if canonical_index in used_slots:
                collisions.append({"existing": used_slots[canonical_index], "ignored": item})
                continue
            used_slots[canonical_index] = item
            mappings.append(item)
    mappings.sort(key=lambda item: item["canonical_index"])
    automatically_verified = bool(mappings) and not collisions and all(
        item["confidence"] == "high" for item in mappings
    )
    return {
        "kind": kind,
        "dimension": CANONICAL_DIM,
        "mappings": mappings,
        "valid_slots": [
            item["canonical_index"] for item in mappings if item.get("active", False)
        ],
        "collisions": collisions,
        "verified": automatically_verified,
        "verification_note": (
            "strict explicit-name mapping"
            if automatically_verified
            else "automatic mapping contains frame-sensitive fields; embodiment review required"
        ),
    }


def infer_unverified_canonical_mapping(
    feature_schema: dict[str, Any], *, kind: str, verification_note: str
) -> dict[str, Any]:
    """Retain inferred candidates without admitting an unknown schema for training."""
    mapping = infer_canonical_mapping(feature_schema, kind=kind)
    mapping["candidate_valid_slots"] = list(mapping.get("valid_slots") or [])
    mapping["valid_slots"] = []
    mapping["verified"] = False
    mapping["verification_note"] = verification_note
    for item in mapping.get("mappings") or []:
        item["active"] = False
    return mapping


def build_verified_mapping(
    feature_schema: dict[str, Any],
    *,
    kind: str,
    entries: list[dict[str, Any]],
    verification_note: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build an explicit mapping whose semantics were checked against source docs.

    Adapters use this only when an embodiment-specific contract is available. It
    intentionally rejects duplicate slots and out-of-range source indices so a
    documentation mistake cannot silently produce a partially valid action.
    """
    mappings: list[dict[str, Any]] = []
    used_slots: set[int] = set()
    for raw in entries:
        source_key = str(raw["source_key"])
        source_index = int(raw["source_index"])
        canonical_index = int(raw["canonical_index"])
        if source_key not in feature_schema:
            raise ValueError(f"Explicit mapping references unknown feature: {source_key}")
        if not 0 <= canonical_index < CANONICAL_DIM:
            raise ValueError(f"Canonical slot is out of range: {canonical_index}")
        if canonical_index in used_slots:
            raise ValueError(f"Duplicate explicit canonical slot: {canonical_index}")
        feature = feature_schema[source_key]
        shape = feature.get("shape") if isinstance(feature, dict) else None
        if isinstance(shape, list) and shape and not 0 <= source_index < int(shape[-1]):
            raise ValueError(
                f"Source index {source_index} is out of range for {source_key}: {shape}"
            )
        names = feature.get("names") if isinstance(feature, dict) else None
        source_name = (
            str(names[source_index])
            if isinstance(names, list) and source_index < len(names)
            else str(raw.get("source_name") or f"{source_key}[{source_index}]")
        )
        mappings.append(
            {
                "source_key": source_key,
                "source_index": source_index,
                "source_name": source_name,
                "canonical_index": canonical_index,
                "semantic": str(raw["semantic"]),
                "confidence": "verified",
                "active": True,
                "alignment_safe": bool(raw.get("alignment_safe", True)),
            }
        )
        used_slots.add(canonical_index)
    mappings.sort(key=lambda item: item["canonical_index"])
    return {
        "kind": kind,
        "dimension": CANONICAL_DIM,
        "mappings": mappings,
        "valid_slots": [item["canonical_index"] for item in mappings],
        "collisions": [],
        "verified": bool(mappings),
        "verification_note": verification_note,
        "provenance": provenance,
    }
