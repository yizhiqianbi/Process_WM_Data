from __future__ import annotations

import re

from ..camera import normalize_camera_role
from ..canonical import build_verified_mapping
from ..schema import CameraRecord
from ..utils import read_json
from ..utils import stable_id
from .base import BaseAdapter
from .lerobot import scan_lerobot_repo


_JOINT_FEATURE = re.compile(r"^observation\.([^.]+)\.joints$")


def _target_robot_variants(repo) -> list[dict]:
    info = read_json(repo / "meta" / "info.json")
    features = info.get("features") or {}
    fps = float(info.get("fps") or 5.0)
    variants: list[dict] = []
    for state_key, feature in sorted(features.items()):
        match = _JOINT_FEATURE.match(state_key)
        if match is None or not isinstance(feature, dict):
            continue
        robot = match.group(1)
        shape = feature.get("shape") or []
        dimension = int(shape[-1]) if shape else 0
        if dimension not in {7, 8}:
            continue
        camera_key = f"observation.images.{robot}"
        camera_feature = features.get(camera_key)
        if not isinstance(camera_feature, dict):
            continue
        arm_dimension = dimension - 1
        joint_names = [f"primary_arm_joint_{index + 1}" for index in range(arm_dimension)]
        state_names = [*joint_names, "primary_gripper_position"]
        action_key = f"fastwam.action.{robot}.next_joint_target"
        action_names = [
            *[f"primary_arm_joint_{index + 1}_target" for index in range(arm_dimension)],
            "primary_gripper_target",
        ]
        state_schema = {
            state_key: {
                "dtype": feature.get("dtype") or "float32",
                "shape": [dimension],
                "names": state_names,
            }
        }
        action_schema = {
            action_key: {
                "dtype": feature.get("dtype") or "float32",
                "shape": [dimension],
                "names": action_names,
            }
        }
        state_entries = [
            {
                "source_key": state_key,
                "source_index": index,
                "canonical_index": 14 + index,
                "semantic": f"primary_joint_{index + 1}",
            }
            for index in range(arm_dimension)
        ] + [
            {
                "source_key": state_key,
                "source_index": dimension - 1,
                "canonical_index": 6,
                "semantic": "primary_gripper",
            }
        ]
        action_entries = [
            {
                "source_key": action_key,
                "source_index": index,
                "canonical_index": 14 + index,
                "semantic": f"primary_joint_{index + 1}_target",
            }
            for index in range(arm_dimension)
        ] + [
            {
                "source_key": action_key,
                "source_index": dimension - 1,
                "canonical_index": 6,
                "semantic": "primary_gripper_target",
            }
        ]
        provenance = {
            "authority": "official_oxe_auge_replay_pipeline",
            "source": "https://github.com/BerkeleyAutomation/AugE-Toolkit",
            "native_field": state_key,
            "derivation": "action[t] = replay_joint_trajectory[t+1]",
            "terminal_policy": "hold_last",
            "canonical_side_policy": "single primary arm occupies the canonical left-arm block",
        }
        mapping = {
            "state": build_verified_mapping(
                state_schema,
                kind="state",
                entries=state_entries,
                verification_note=f"{robot} replay joints plus normalized gripper",
                provenance=provenance,
            ),
            "action": build_verified_mapping(
                action_schema,
                kind="action",
                entries=action_entries,
                verification_note="derived next-step target from official target-robot replay trajectory",
                provenance=provenance,
            ),
        }
        camera_info = camera_feature.get("info") or {}
        camera_shape = camera_feature.get("shape") or []
        camera = CameraRecord(
            source_key=camera_key,
            role=normalize_camera_role(camera_key),
            dtype="video",
            width=int(camera_shape[1]) if len(camera_shape) >= 2 else None,
            height=int(camera_shape[0]) if len(camera_shape) >= 2 else None,
            # Released metadata sometimes says 30 here while the actual videos
            # and episode table are 5 Hz. The episode rate is the shared clock.
            fps=fps,
            codec=camera_info.get("video.codec"),
        )
        variants.append(
            {
                "id": robot,
                "embodiment": robot,
                "robot_type": robot,
                "fps": fps,
                "cameras": [camera],
                "state_schema": state_schema,
                "action_schema": action_schema,
                "canonical_mapping": mapping,
                "passed_checks": [
                    "official_target_replay_trajectory",
                    "derived_action_provenance",
                ],
                "warnings": ["action_derived_from_next_replay_joint_state"],
                "metadata": {
                    "native_conversion": {
                        "source_format": "parquet",
                        "derived_columns": {
                            action_key: {
                                "source_key": state_key,
                                "operation": "next_row_hold_last",
                            }
                        },
                        "action_semantics": "derived_next_replay_joint_target",
                    },
                    "source_robot_family": repo.name.removesuffix("_augmented"),
                    "target_robot": robot,
                },
            }
        )
    return variants


class OXEAugEAdapter(BaseAdapter):
    dataset_name = "oxe_auge"

    def scan(self) -> None:
        repos = sorted(
            path.parent.parent
            for path in self.options.input_root.glob("*/meta/info.json")
        )
        if not repos:
            self.blockers.append("no_lerobot_repositories_found")
            return
        for repo in repos:
            if self.at_limit():
                break

            def lineage(episode_index: int, repo_name: str = repo.name) -> str:
                # Augmented variants sharing the same source suffix get one stable family id.
                source_name = repo_name.removesuffix("_augmented")
                return stable_id("oxe_auge", source_name, episode_index)

            scan_lerobot_repo(
                self,
                repo,
                release=self.options.release,
                task_namespace=repo.name.removesuffix("_augmented"),
                lineage_factory=lineage,
                variants=_target_robot_variants(repo),
            )
