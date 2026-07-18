from __future__ import annotations

from typing import Any

from ..canonical import build_verified_mapping, infer_unverified_canonical_mapping
from ..schema import CameraRecord
from ..utils import read_json
from .base import BaseAdapter
from .lerobot import camera_records, feature_schemas, scan_lerobot_repo


LINGBOT_VA_REPOSITORY = "https://github.com/Robbyant/lingbot-va"
ROBOTWIN_DATASET = "robbyant/robotwin-clean-and-aug-lerobot"
LIBERO_DATASET = "robbyant/libero-long-lerobot"

_POSE_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "rotation_vector_x",
    "rotation_vector_y",
    "rotation_vector_z",
)
_ROBOTWIN_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_q1",
    "left_q2",
    "left_q3",
    "left_q4",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_q1",
    "right_q2",
    "right_q3",
    "right_q4",
    "right_gripper",
]
_LIBERO_STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"]
_LIBERO_ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def _feature_width(schema: dict[str, Any], key: str) -> int | None:
    shape = (schema.get(key) or {}).get("shape")
    return int(shape[-1]) if isinstance(shape, list) and shape else None


def _camera_profile(
    cameras: list[CameraRecord], roles: dict[str, str]
) -> list[CameraRecord]:
    profiled: list[CameraRecord] = []
    for camera in cameras:
        copy = CameraRecord(**camera.to_dict())
        copy.role = roles.get(copy.source_key, copy.role)
        profiled.append(copy)
    return profiled


def _vector_schema(dtype: Any, names: list[str]) -> dict[str, Any]:
    return {"dtype": dtype, "shape": [len(names)], "names": names}


def _pose_entries(
    source_key: str, canonical_base: int, side: str
) -> list[dict[str, Any]]:
    return [
        {
            "source_key": source_key,
            "source_index": index,
            "canonical_index": canonical_base + index,
            "semantic": f"{side}_ee_{name}",
        }
        for index, name in enumerate(_POSE_NAMES)
    ]


def _robotwin_variant(
    cameras: list[CameraRecord],
    state_schema: dict[str, Any],
    action_schema: dict[str, Any],
) -> dict[str, Any] | None:
    state_key = "observation.state"
    action_key = "action"
    if (
        _feature_width(state_schema, state_key) != 16
        or _feature_width(action_schema, action_key) != 16
        or (state_schema.get(state_key) or {}).get("names") != _ROBOTWIN_NAMES
        or (action_schema.get(action_key) or {}).get("names") != _ROBOTWIN_NAMES
    ):
        return None

    state_dtype = (state_schema[state_key] or {}).get("dtype")
    action_dtype = (action_schema[action_key] or {}).get("dtype")
    state_schema = {
        **state_schema,
        state_key: _vector_schema(
            state_dtype,
            [
                "left_position_x",
                "left_position_y",
                "left_position_z",
                "left_quaternion_x",
                "left_quaternion_y",
                "left_quaternion_z",
                "left_quaternion_w",
                "left_gripper_state",
                "right_position_x",
                "right_position_y",
                "right_position_z",
                "right_quaternion_x",
                "right_quaternion_y",
                "right_quaternion_z",
                "right_quaternion_w",
                "right_gripper_state",
            ],
        ),
        "fastwam.state.left_ee_pose": _vector_schema(
            state_dtype, [f"left_ee_{name}" for name in _POSE_NAMES]
        ),
        "fastwam.state.left_gripper": _vector_schema(
            state_dtype, ["left_gripper_state"]
        ),
        "fastwam.state.right_ee_pose": _vector_schema(
            state_dtype, [f"right_ee_{name}" for name in _POSE_NAMES]
        ),
        "fastwam.state.right_gripper": _vector_schema(
            state_dtype, ["right_gripper_state"]
        ),
    }
    action_schema = {
        **action_schema,
        action_key: _vector_schema(
            action_dtype,
            [
                name.replace("state", "target")
                for name in state_schema[state_key]["names"]
            ],
        ),
        "fastwam.action.left_ee_target": _vector_schema(
            action_dtype, [f"left_ee_target_{name}" for name in _POSE_NAMES]
        ),
        "fastwam.action.left_gripper_target": _vector_schema(
            action_dtype, ["left_gripper_target"]
        ),
        "fastwam.action.right_ee_target": _vector_schema(
            action_dtype, [f"right_ee_target_{name}" for name in _POSE_NAMES]
        ),
        "fastwam.action.right_gripper_target": _vector_schema(
            action_dtype, ["right_gripper_target"]
        ),
    }

    state_entries = [
        *_pose_entries("fastwam.state.left_ee_pose", 0, "left"),
        {
            "source_key": "fastwam.state.left_gripper",
            "source_index": 0,
            "canonical_index": 6,
            "semantic": "left_gripper_state",
        },
        *_pose_entries("fastwam.state.right_ee_pose", 7, "right"),
        {
            "source_key": "fastwam.state.right_gripper",
            "source_index": 0,
            "canonical_index": 13,
            "semantic": "right_gripper_state",
        },
    ]
    action_entries = [
        *_pose_entries("fastwam.action.left_ee_target", 0, "left"),
        {
            "source_key": "fastwam.action.left_gripper_target",
            "source_index": 0,
            "canonical_index": 6,
            "semantic": "left_gripper_target",
        },
        *_pose_entries("fastwam.action.right_ee_target", 7, "right"),
        {
            "source_key": "fastwam.action.right_gripper_target",
            "source_index": 0,
            "canonical_index": 13,
            "semantic": "right_gripper_target",
        },
    ]
    provenance = {
        "authority": "official_lingbot_va_robotwin_loader",
        "repository": LINGBOT_VA_REPOSITORY,
        "dataset": ROBOTWIN_DATASET,
        "source_action_layout": "left_xyz_quaternion_gripper_right_xyz_quaternion_gripper",
        "quaternion_order": "xyzw",
        "fastwam_rotation_conversion": "normalized quaternion to principal rotation vector",
    }
    mapping = {
        "state": build_verified_mapping(
            state_schema,
            kind="state",
            entries=state_entries,
            verification_note="LingBot-VA RoboTwin 16D dual-arm EEF state converted to FastWAM 6D poses",
            provenance=provenance,
        ),
        "action": build_verified_mapping(
            action_schema,
            kind="action",
            entries=action_entries,
            verification_note="LingBot-VA RoboTwin 16D dual-arm EEF targets converted to FastWAM 6D poses",
            provenance=provenance,
        ),
    }
    conversion = {
        "source_format": "parquet",
        "derived_columns": {
            "fastwam.state.left_ee_pose": {
                "source_key": state_key,
                "indices": list(range(0, 7)),
                "operation": "pose_quaternion_to_rotvec",
                "quaternion_order": "xyzw",
            },
            "fastwam.state.left_gripper": {
                "source_key": state_key,
                "indices": [7],
            },
            "fastwam.state.right_ee_pose": {
                "source_key": state_key,
                "indices": list(range(8, 15)),
                "operation": "pose_quaternion_to_rotvec",
                "quaternion_order": "xyzw",
            },
            "fastwam.state.right_gripper": {
                "source_key": state_key,
                "indices": [15],
            },
            "fastwam.action.left_ee_target": {
                "source_key": action_key,
                "indices": list(range(0, 7)),
                "operation": "pose_quaternion_to_rotvec",
                "quaternion_order": "xyzw",
            },
            "fastwam.action.left_gripper_target": {
                "source_key": action_key,
                "indices": [7],
            },
            "fastwam.action.right_ee_target": {
                "source_key": action_key,
                "indices": list(range(8, 15)),
                "operation": "pose_quaternion_to_rotvec",
                "quaternion_order": "xyzw",
            },
            "fastwam.action.right_gripper_target": {
                "source_key": action_key,
                "indices": [15],
            },
        },
    }
    return {
        "id": "robotwin_eef_16d",
        "robot_type": "aloha_robotwin",
        "embodiment": "aloha_robotwin_dual_eef",
        "cameras": _camera_profile(
            cameras,
            {
                "observation.images.cam_high": "global_primary",
                "observation.images.cam_left_wrist": "left_wrist",
                "observation.images.cam_right_wrist": "right_wrist",
            },
        ),
        "state_schema": state_schema,
        "action_schema": action_schema,
        "canonical_mapping": mapping,
        "passed_checks": ["official_embodiment_schema"],
        "metadata": {
            "native_conversion": conversion,
            "source_profile": {
                "family": "lingbot_va_robotwin",
                "official_dataset": ROBOTWIN_DATASET,
                "native_action_dimension": 16,
                "lingbot_model_action_dimension": 30,
                "lingbot_used_action_channels": (
                    list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
                ),
                "canonical_action_semantics": "absolute_dual_eef_target",
            },
        },
    }


def _libero_variant(
    cameras: list[CameraRecord],
    state_schema: dict[str, Any],
    action_schema: dict[str, Any],
) -> dict[str, Any] | None:
    state_key = "observation.state"
    action_key = "action"
    if (
        _feature_width(state_schema, state_key) != 8
        or _feature_width(action_schema, action_key) != 7
        or (state_schema.get(state_key) or {}).get("names") != _LIBERO_STATE_NAMES
        or (action_schema.get(action_key) or {}).get("names") != _LIBERO_ACTION_NAMES
    ):
        return None

    state_dtype = state_schema[state_key].get("dtype")
    action_dtype = action_schema[action_key].get("dtype")
    state_schema = {
        **state_schema,
        "fastwam.state.left_ee_pose": _vector_schema(
            state_dtype, [f"left_ee_{name}" for name in _POSE_NAMES]
        ),
        "fastwam.state.left_gripper": _vector_schema(
            state_dtype, ["left_gripper_state"]
        ),
    }
    action_schema = {
        **action_schema,
        "fastwam.action.left_ee_delta": _vector_schema(
            action_dtype,
            [
                "left_ee_delta_position_x",
                "left_ee_delta_position_y",
                "left_ee_delta_position_z",
                "left_ee_delta_rotation_vector_x",
                "left_ee_delta_rotation_vector_y",
                "left_ee_delta_rotation_vector_z",
            ],
        ),
        "fastwam.action.left_gripper": _vector_schema(
            action_dtype, ["left_gripper_command"]
        ),
    }
    state_entries = _pose_entries("fastwam.state.left_ee_pose", 0, "left") + [
        {
            "source_key": "fastwam.state.left_gripper",
            "source_index": 0,
            "canonical_index": 6,
            "semantic": "left_gripper_state",
            "alignment_safe": False,
        }
    ]
    action_entries = [
        {
            "source_key": "fastwam.action.left_ee_delta",
            "source_index": index,
            "canonical_index": index,
            "semantic": name,
        }
        for index, name in enumerate(
            action_schema["fastwam.action.left_ee_delta"]["names"]
        )
    ] + [
        {
            "source_key": "fastwam.action.left_gripper",
            "source_index": 0,
            "canonical_index": 6,
            "semantic": "left_gripper_command",
            "alignment_safe": False,
        }
    ]
    provenance = {
        "authority": "official_lingbot_va_libero_config",
        "repository": LINGBOT_VA_REPOSITORY,
        "dataset": LIBERO_DATASET,
        "used_action_channels": list(range(7)),
        "fastwam_rotation_conversion": "incremental rpy to rotation vector",
    }
    mapping = {
        "state": build_verified_mapping(
            state_schema,
            kind="state",
            entries=state_entries,
            verification_note="LIBERO xyz+rpy feedback converted to FastWAM 6D left EEF state",
            provenance=provenance,
        ),
        "action": build_verified_mapping(
            action_schema,
            kind="action",
            entries=action_entries,
            verification_note="LingBot-VA LIBERO 7D delta EEF and gripper action",
            provenance=provenance,
        ),
    }
    return {
        "id": "libero_delta_eef_7d",
        "robot_type": "franka_libero",
        "embodiment": "franka_libero_single_eef",
        "cameras": _camera_profile(
            cameras,
            {
                "observation.images.agentview_rgb": "global_primary",
                "observation.images.eye_in_hand_rgb": "left_wrist",
            },
        ),
        "state_schema": state_schema,
        "action_schema": action_schema,
        "canonical_mapping": mapping,
        "passed_checks": ["official_embodiment_schema"],
        "metadata": {
            "native_conversion": {
                "source_format": "parquet",
                "derived_columns": {
                    "fastwam.state.left_ee_pose": {
                        "source_key": state_key,
                        "indices": list(range(6)),
                        "operation": "pose_rpy_to_rotvec",
                    },
                    "fastwam.state.left_gripper": {
                        "source_key": state_key,
                        "indices": [6],
                    },
                    "fastwam.action.left_ee_delta": {
                        "source_key": action_key,
                        "indices": list(range(6)),
                        "operation": "pose_rpy_to_rotvec",
                    },
                    "fastwam.action.left_gripper": {
                        "source_key": action_key,
                        "indices": [6],
                    },
                },
            },
            "source_profile": {
                "family": "lingbot_va_libero_long",
                "official_dataset": LIBERO_DATASET,
                "native_action_dimension": 7,
                "lingbot_model_action_dimension": 30,
                "lingbot_used_action_channels": list(range(7)),
                "canonical_action_semantics": "delta_single_eef_and_gripper_command",
            },
        },
    }


def build_lingbot_variant(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features")
    features = features if isinstance(features, dict) else {}
    cameras = camera_records(features)
    state_schema, action_schema = feature_schemas(features)
    robotwin = _robotwin_variant(cameras, state_schema, action_schema)
    if robotwin is not None:
        return robotwin
    libero = _libero_variant(cameras, state_schema, action_schema)
    if libero is not None:
        return libero
    return {
        "id": "unverified_lerobot_schema",
        "cameras": cameras,
        "state_schema": state_schema,
        "action_schema": action_schema,
        "canonical_mapping": {
            "state": infer_unverified_canonical_mapping(
                state_schema,
                kind="state",
                verification_note="unknown LingBot-VA embodiment schema",
            ),
            "action": infer_unverified_canonical_mapping(
                action_schema,
                kind="action",
                verification_note="unknown LingBot-VA embodiment schema",
            ),
        },
        "warnings": ["lingbot_va_embodiment_schema_unverified"],
        "metadata": {
            "source_profile": {
                "family": "lingbot_va_unknown_lerobot",
                "canonical_action_semantics": "unverified",
            }
        },
    }


class LingBotVAAdapter(BaseAdapter):
    dataset_name = "lingbot_va"

    def scan(self) -> None:
        repos = sorted(
            {
                path.parent.parent
                for path in self.options.input_root.rglob("meta/info.json")
                if ".cache" not in path.parts
            }
        )
        if not repos:
            self.blockers.append("no_lingbot_va_lerobot_repositories_found")
            return
        for repo in repos:
            if self.at_limit():
                break
            try:
                info = read_json(repo / "meta" / "info.json")
                variants = [build_lingbot_variant(info)]
            except (OSError, ValueError):
                variants = None
            scan_lerobot_repo(
                self,
                repo,
                release=self.options.release,
                task_namespace=repo.name,
                variants=variants,
            )
