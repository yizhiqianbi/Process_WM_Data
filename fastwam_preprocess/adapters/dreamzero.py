from __future__ import annotations

from typing import Any

from ..canonical import build_verified_mapping, infer_unverified_canonical_mapping
from ..schema import CameraRecord
from ..utils import read_json
from .base import BaseAdapter
from .lerobot import camera_records, feature_schemas, scan_lerobot_repo


DREAMZERO_REPOSITORY = "https://github.com/dreamzero0/dreamzero"
DREAMZERO_DATASET = "GEAR-Dreams/DreamZero-DROID-Data"


def _feature_width(schema: dict[str, Any], key: str) -> int | None:
    shape = (schema.get(key) or {}).get("shape")
    return int(shape[-1]) if isinstance(shape, list) and shape else None


def _camera_profile(cameras: list[CameraRecord]) -> list[CameraRecord]:
    roles = {
        "observation.images.exterior_image_1_left": "global_primary",
        "observation.images.exterior_image_2_left": "global_secondary",
        "observation.images.wrist_image_left": "left_wrist",
    }
    profiled: list[CameraRecord] = []
    for camera in cameras:
        copy = CameraRecord(**camera.to_dict())
        copy.role = roles.get(copy.source_key, copy.role)
        profiled.append(copy)
    return profiled


def _modality_slice(
    modality: dict[str, Any],
    group: str,
    name: str,
    *,
    expected_width: int,
    default_source_key: str,
) -> tuple[str, int, int] | None:
    groups = modality.get(group)
    entry = groups.get(name) if isinstance(groups, dict) else None
    if not isinstance(entry, dict):
        return None
    try:
        start = int(entry["start"])
        end = int(entry["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if start < 0 or end - start != expected_width:
        return None
    source_key = str(entry.get("original_key") or default_source_key)
    return source_key, start, end


def _unverified_variant(
    cameras: list[CameraRecord],
    state_schema: dict[str, Any],
    action_schema: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "id": "unverified_dreamzero_schema",
        "robot_type": "droid",
        "embodiment": "droid_panda_single_arm",
        "cameras": _camera_profile(cameras),
        "state_schema": state_schema,
        "action_schema": action_schema,
        "canonical_mapping": {
            "state": infer_unverified_canonical_mapping(
                state_schema,
                kind="state",
                verification_note=reason,
            ),
            "action": infer_unverified_canonical_mapping(
                action_schema,
                kind="action",
                verification_note=reason,
            ),
        },
        "warnings": [reason],
        "metadata": {
            "source_profile": {
                "family": "dreamzero_droid",
                "official_dataset": DREAMZERO_DATASET,
                "canonical_action_semantics": "unverified",
            }
        },
    }


def build_dreamzero_variant(
    info: dict[str, Any], modality: dict[str, Any] | None
) -> dict[str, Any]:
    features = info.get("features")
    features = features if isinstance(features, dict) else {}
    cameras = camera_records(features)
    state_schema, action_schema = feature_schemas(features)
    if modality is None:
        return _unverified_variant(
            cameras,
            state_schema,
            action_schema,
            "dreamzero_modality_json_missing",
        )

    state_joint = _modality_slice(
        modality,
        "state",
        "joint_position",
        expected_width=7,
        default_source_key="observation.state",
    )
    state_gripper = _modality_slice(
        modality,
        "state",
        "gripper_position",
        expected_width=1,
        default_source_key="observation.state",
    )
    action_joint = _modality_slice(
        modality,
        "action",
        "joint_position",
        expected_width=7,
        default_source_key="action",
    )
    action_gripper = _modality_slice(
        modality,
        "action",
        "gripper_position",
        expected_width=1,
        default_source_key="action",
    )
    slices = (state_joint, state_gripper, action_joint, action_gripper)
    if any(value is None for value in slices):
        return _unverified_variant(
            cameras,
            state_schema,
            action_schema,
            "dreamzero_required_modality_slice_missing",
        )
    assert state_joint and state_gripper and action_joint and action_gripper
    if (
        state_joint[0] not in state_schema
        or state_gripper[0] not in state_schema
        or action_joint[0] not in action_schema
        or action_gripper[0] not in action_schema
        or state_joint[2] > (_feature_width(state_schema, state_joint[0]) or 0)
        or state_gripper[2] > (_feature_width(state_schema, state_gripper[0]) or 0)
        or action_joint[2] > (_feature_width(action_schema, action_joint[0]) or 0)
        or action_gripper[2] > (_feature_width(action_schema, action_gripper[0]) or 0)
    ):
        return _unverified_variant(
            cameras,
            state_schema,
            action_schema,
            "dreamzero_modality_slice_out_of_range",
        )

    state_joint_key = "fastwam.state.left_joint_position"
    state_gripper_key = "fastwam.state.left_gripper"
    action_joint_key = "fastwam.action.left_joint_target"
    action_gripper_key = "fastwam.action.left_gripper_target"
    state_dtype = state_schema[state_joint[0]].get("dtype")
    action_dtype = action_schema[action_joint[0]].get("dtype")
    state_schema = {
        **state_schema,
        state_joint_key: {
            "dtype": state_dtype,
            "shape": [7],
            "names": [f"left_joint_{index + 1}_state" for index in range(7)],
        },
        state_gripper_key: {
            "dtype": state_dtype,
            "shape": [1],
            "names": ["left_gripper_state"],
        },
    }
    action_schema = {
        **action_schema,
        action_joint_key: {
            "dtype": action_dtype,
            "shape": [7],
            "names": [f"left_joint_{index + 1}_target" for index in range(7)],
        },
        action_gripper_key: {
            "dtype": action_dtype,
            "shape": [1],
            "names": ["left_gripper_target"],
        },
    }
    state_entries = [
        {
            "source_key": state_joint_key,
            "source_index": index,
            "canonical_index": 14 + index,
            "semantic": f"left_joint_{index + 1}_state",
        }
        for index in range(7)
    ] + [
        {
            "source_key": state_gripper_key,
            "source_index": 0,
            "canonical_index": 6,
            "semantic": "left_gripper_state",
            "alignment_safe": False,
        }
    ]
    action_entries = [
        {
            "source_key": action_joint_key,
            "source_index": index,
            "canonical_index": 14 + index,
            "semantic": f"left_joint_{index + 1}_target",
        }
        for index in range(7)
    ] + [
        {
            "source_key": action_gripper_key,
            "source_index": 0,
            "canonical_index": 6,
            "semantic": "left_gripper_target",
            "alignment_safe": False,
        }
    ]
    provenance = {
        "authority": "official_dreamzero_modality_json_and_training_config",
        "repository": DREAMZERO_REPOSITORY,
        "dataset": DREAMZERO_DATASET,
        "embodiment_tag": "oxe_droid",
        "official_relative_action_keys": ["joint_position"],
        "canonical_storage": "absolute source targets; relative transforms remain loader policy",
    }
    mapping = {
        "state": build_verified_mapping(
            state_schema,
            kind="state",
            entries=state_entries,
            verification_note="DreamZero modality.json DROID joint and gripper feedback slices",
            provenance=provenance,
        ),
        "action": build_verified_mapping(
            action_schema,
            kind="action",
            entries=action_entries,
            verification_note="DreamZero modality.json DROID joint and gripper target slices",
            provenance=provenance,
        ),
    }
    return {
        "id": "dreamzero_droid_joint_gripper",
        "robot_type": "droid_panda",
        "embodiment": "droid_panda_single_arm",
        "cameras": _camera_profile(cameras),
        "state_schema": state_schema,
        "action_schema": action_schema,
        "canonical_mapping": mapping,
        "passed_checks": ["official_embodiment_schema", "gear_modality_metadata"],
        "metadata": {
            "native_conversion": {
                "source_format": "parquet",
                "derived_columns": {
                    state_joint_key: {
                        "source_key": state_joint[0],
                        "indices": list(range(state_joint[1], state_joint[2])),
                    },
                    state_gripper_key: {
                        "source_key": state_gripper[0],
                        "indices": [state_gripper[1]],
                    },
                    action_joint_key: {
                        "source_key": action_joint[0],
                        "indices": list(range(action_joint[1], action_joint[2])),
                    },
                    action_gripper_key: {
                        "source_key": action_gripper[0],
                        "indices": [action_gripper[1]],
                    },
                },
            },
            "source_profile": {
                "family": "dreamzero_droid",
                "official_dataset": DREAMZERO_DATASET,
                "native_state_dimension": _feature_width(
                    state_schema, "observation.state"
                ),
                "native_action_dimension": _feature_width(action_schema, "action"),
                "dreamzero_action_horizon": 24,
                "dreamzero_video_frames": 33,
                "dreamzero_video_resolution": [320, 176],
                "dreamzero_relative_action_keys": ["joint_position"],
                "canonical_action_semantics": "absolute_joint_and_gripper_target",
            },
            "gear_modality": modality,
        },
    }


class DreamZeroAdapter(BaseAdapter):
    dataset_name = "dreamzero"

    def scan(self) -> None:
        repos = sorted(
            {
                path.parent.parent
                for path in self.options.input_root.rglob("meta/info.json")
                if ".cache" not in path.parts
            }
        )
        if not repos:
            self.blockers.append("no_dreamzero_lerobot_repositories_found")
            return
        for repo in repos:
            if self.at_limit():
                break
            try:
                info = read_json(repo / "meta" / "info.json")
                modality_path = repo / "meta" / "modality.json"
                modality = read_json(modality_path) if modality_path.is_file() else None
                variants = [build_dreamzero_variant(info, modality)]
            except (OSError, ValueError):
                variants = None
            scan_lerobot_repo(
                self,
                repo,
                release=self.options.release,
                task_namespace="droid_manipulation",
                variants=variants,
            )
