from __future__ import annotations

import tarfile
from pathlib import Path

from ..camera import normalize_camera_role
from ..canonical import build_verified_mapping, infer_canonical_mapping
from ..native_readers import inspect_oxe_episode
from ..schema import CameraRecord
from ..utils import is_partial_path
from .base import BaseAdapter


_ASU_SUBSET = "asu_table_top_converted_externally_to_rlds"
_POSE_NAMES = [
    "left_ee_position_x",
    "left_ee_position_y",
    "left_ee_position_z",
    "left_ee_rotation_vector_x",
    "left_ee_rotation_vector_y",
    "left_ee_rotation_vector_z",
]


def _asu_contract() -> tuple[dict, dict, dict, dict]:
    state_schema = {
        "fastwam.state.left_ee_pose": {
            "dtype": "float32",
            "shape": [6],
            "names": _POSE_NAMES,
        },
        "fastwam.state.left_gripper": {
            "dtype": "float32",
            "shape": [1],
            "names": ["left_gripper_position"],
        },
    }
    action_schema = {
        "fastwam.action.left_ee_target": {
            "dtype": "float32",
            "shape": [6],
            "names": [name.replace("position", "target_position") for name in _POSE_NAMES],
        },
        "fastwam.action.left_gripper_target": {
            "dtype": "float32",
            "shape": [1],
            "names": ["left_gripper_target"],
        },
    }
    provenance = {
        "authority": "official_oxe_and_tfds_schema_plus_local_tensor_validation",
        "oxe_action_contract": "https://github.com/google-deepmind/open_x_embodiment",
        "source_schema": "https://www.tensorflow.org/datasets/catalog/asu_table_top_converted_externally_to_rlds",
        "local_validation": "action[:6] matches next-step ground_truth_states.EE on the sampled episode",
        "rotation_conversion": "source xyz+rpy converted to canonical xyz+rotation-vector",
    }
    state_entries = [
        {
            "source_key": "fastwam.state.left_ee_pose",
            "source_index": index,
            "canonical_index": index,
            "semantic": _POSE_NAMES[index],
        }
        for index in range(6)
    ] + [
        {
            "source_key": "fastwam.state.left_gripper",
            "source_index": 0,
            "canonical_index": 6,
            "semantic": "left_gripper",
        }
    ]
    action_entries = [
        {
            "source_key": "fastwam.action.left_ee_target",
            "source_index": index,
            "canonical_index": index,
            "semantic": _POSE_NAMES[index].replace("position", "target_position"),
        }
        for index in range(6)
    ] + [
        {
            "source_key": "fastwam.action.left_gripper_target",
            "source_index": 0,
            "canonical_index": 6,
            "semantic": "left_gripper_target",
        }
    ]
    mapping = {
        "state": build_verified_mapping(
            state_schema,
            kind="state",
            entries=state_entries,
            verification_note="ASU UR5 xyz+rpy state converted to canonical xyz+rotation-vector",
            provenance=provenance,
        ),
        "action": build_verified_mapping(
            action_schema,
            kind="action",
            entries=action_entries,
            verification_note="ASU 7D EEF pose/gripper target verified against next-step EE state",
            provenance=provenance,
        ),
    }
    conversion = {
        "source_format": "oxe_pickle",
        "derived_columns": {
            "fastwam.state.left_ee_pose": {
                "source_key": "ground_truth_states.EE",
                "indices": list(range(6)),
                "operation": "pose_rpy_to_rotvec",
            },
            "fastwam.state.left_gripper": {
                "source_key": "observation.state",
                "indices": [6],
            },
            "fastwam.action.left_ee_target": {
                "source_key": "action",
                "indices": list(range(6)),
                "operation": "pose_rpy_to_rotvec",
            },
            "fastwam.action.left_gripper_target": {
                "source_key": "action",
                "indices": [6],
            },
        },
        "action_semantics": "native_next_eef_pose_and_gripper_target",
    }
    return state_schema, action_schema, mapping, conversion


class OXEAdapter(BaseAdapter):
    dataset_name = "oxe"

    def scan(self) -> None:
        shards = sorted(self.options.input_root.glob("*/*.tar*"))
        if not shards:
            self.blockers.append("no_oxe_tar_shards_found")
            return

        for shard in shards:
            if self.at_limit():
                break
            complete = shard.suffix == ".tar" and not is_partial_path(shard)
            if not complete:
                self.add_artifact(
                    path=shard,
                    kind="oxe_pickle_shard",
                    complete=False,
                    status="partial_download",
                )
                continue
            try:
                with tarfile.open(shard, mode="r:") as archive:
                    members = [
                        (member.name, member.size)
                        for member in archive
                        if member.isfile() and member.name.endswith(".data.pickle")
                    ]
            except (OSError, tarfile.TarError) as exc:
                self.add_artifact(
                    path=shard,
                    kind="oxe_pickle_shard",
                    complete=False,
                    status="invalid_tar",
                    metadata={"error": str(exc)},
                )
                continue

            self.add_artifact(
                path=shard,
                kind="oxe_pickle_shard",
                complete=True,
                status="indexed",
                metadata={"episode_count": len(members)},
            )
            dataset_subset = shard.parent.name
            for member_name, member_size in members:
                if self.at_limit():
                    break
                source_episode = Path(member_name).name.removesuffix(".data.pickle")
                inspection = None
                inspection_error = None
                if self.options.verify_files or dataset_subset == _ASU_SUBSET:
                    try:
                        inspection = inspect_oxe_episode(shard, member_name)
                    except Exception as exc:
                        inspection_error = str(exc)

                source_fps = 5.0 if dataset_subset == _ASU_SUBSET else 20.0
                image_fields = (inspection or {}).get("images") or {
                    "image": {"height": None, "width": None}
                }
                cameras = []
                video_refs = {}
                for image_key, image_info in image_fields.items():
                    source_key = f"observation.{image_key}"
                    uri = f"oxe-pickle://{shard}!{member_name}#{source_key}"
                    video_refs[source_key] = uri
                    cameras.append(
                        CameraRecord(
                            source_key=source_key,
                            role=normalize_camera_role(source_key),
                            dtype="embedded_jpeg_or_array",
                            width=image_info.get("width"),
                            height=image_info.get("height"),
                            fps=source_fps if inspection is not None else None,
                            source_uri=uri,
                        )
                    )
                action_dim = (inspection or {}).get("action_dim")
                state_dim = (inspection or {}).get("state_dim")
                action_schema = (
                    {
                        "action": {
                            "dtype": "float32",
                            "shape": [action_dim],
                            "names": [f"action_{index}" for index in range(action_dim)],
                        }
                    }
                    if action_dim
                    else {}
                )
                state_schema = (
                    {
                        "observation.state": {
                            "dtype": "float32",
                            "shape": [state_dim],
                            "names": [f"state_{index}" for index in range(state_dim)],
                        }
                    }
                    if state_dim
                    else {}
                )
                canonical_mapping = {
                    "state": infer_canonical_mapping(state_schema, kind="state"),
                    "action": infer_canonical_mapping(action_schema, kind="action"),
                }
                native_conversion = {"source_format": "oxe_pickle"}
                embodiment = dataset_subset
                robot_type = "heterogeneous_oxe"
                if (
                    dataset_subset == _ASU_SUBSET
                    and action_dim == 7
                    and state_dim == 7
                    and (inspection or {}).get("ground_truth_ee_dim") == 6
                ):
                    (
                        state_schema,
                        action_schema,
                        canonical_mapping,
                        native_conversion,
                    ) = _asu_contract()
                    embodiment = "asu_ur5"
                    robot_type = "ur5"
                instruction = (inspection or {}).get("instruction") or dataset_subset
                passed = ["archive_member_readable", "episode_boundary"]
                pending = ["signal", "temporal", "visual"]
                warnings = []
                failures = []
                if inspection is not None:
                    passed.extend(["restricted_pickle_decode", "schema_inference"])
                    if dataset_subset == _ASU_SUBSET:
                        passed.extend(["official_action_schema", "canonical_action_mapping"])
                    else:
                        warnings.append("nominal_20hz_step_rate_requires_subset_verification")
                elif inspection_error:
                    failures.append("restricted_pickle_decode_failed")
                    pending.append("restricted_pickle_decode")
                else:
                    pending.extend(["restricted_pickle_decode", "schema_inference"])
                self.add_episode(
                    source_episode_id=f"{dataset_subset}:{source_episode}",
                    source_uri=f"tar://{shard}!{member_name}",
                    embodiment=embodiment,
                    robot_type=robot_type,
                    task_namespace=dataset_subset,
                    tasks=[instruction],
                    num_frames=(inspection or {}).get("num_steps"),
                    fps=source_fps if inspection is not None else None,
                    cameras=cameras,
                    state_schema=state_schema,
                    action_schema=action_schema,
                    complete=inspection_error is None,
                    passed_checks=passed,
                    pending_checks=pending,
                    warnings=warnings,
                    failures=failures,
                    references={
                        "archive": str(shard),
                        "member": member_name,
                        "member_size": member_size,
                        "videos": video_refs,
                    },
                    metadata={
                        "source_subset": dataset_subset,
                        "canonical_mapping": canonical_mapping,
                        "native_conversion": native_conversion,
                        "inspection_error": inspection_error,
                        "timeline_basis": (
                            "verified_5hz_lerobot_conversion_rate"
                            if dataset_subset == _ASU_SUBSET
                            else "nominal_source_step_index"
                        ),
                    },
                )
