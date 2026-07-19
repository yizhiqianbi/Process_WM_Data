import json
import math
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fastwam_preprocess.adapters import (
    AdapterOptions,
    DreamZeroAdapter,
    LingBotVAAdapter,
)
from fastwam_preprocess.adapters.dreamzero import build_dreamzero_variant
from fastwam_preprocess.adapters.lingbot_va import build_lingbot_variant
from fastwam_preprocess.cleaning import clean_manifest
from fastwam_preprocess.materialize import materialize_canonical
from fastwam_preprocess.training_case import build_training_cases
from fastwam_preprocess.utils import iter_jsonl
from fastwam_preprocess.windows import build_window_index


def _video_feature(key: str, fps: float, *, video_info: bool = False):
    metadata_key = "video_info" if video_info else "info"
    return key, {
        "dtype": "video",
        "shape": [64, 96, 3],
        "names": ["height", "width", "channel"],
        metadata_key: {
            "video.height": 64,
            "video.width": 96,
            "video.fps": fps,
            "video.codec": "h264",
            "video.is_depth_map": False,
        },
    }


class ExternalDatasetPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_repo(
        self,
        name: str,
        *,
        info: dict,
        states: list[list[float]],
        actions: list[list[float]],
        episode_metadata: dict,
        modality: dict | None = None,
    ) -> Path:
        repo = self.root / name
        (repo / "meta").mkdir(parents=True)
        (repo / "data" / "chunk-000").mkdir(parents=True)
        info = {
            "codebase_version": "v2.1",
            "total_episodes": 1,
            "total_frames": len(states),
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            **info,
        }
        (repo / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
        episode = {
            "episode_index": 0,
            "length": len(states),
            "tasks": ["move the object to the target"],
            **episode_metadata,
        }
        (repo / "meta" / "episodes.jsonl").write_text(
            json.dumps(episode) + "\n", encoding="utf-8"
        )
        if modality is not None:
            (repo / "meta" / "modality.json").write_text(
                json.dumps(modality), encoding="utf-8"
            )
        fps = float(info["fps"])
        pq.write_table(
            pa.table(
                {
                    "observation.state": states,
                    "action": actions,
                    "timestamp": [index / fps for index in range(len(states))],
                    "frame_index": list(range(len(states))),
                }
            ),
            repo / "data" / "chunk-000" / "episode_000000.parquet",
        )
        for key, feature in info["features"].items():
            if feature.get("dtype") != "video":
                continue
            video = repo / "videos" / "chunk-000" / key / "episode_000000.mp4"
            video.parent.mkdir(parents=True)
            video.touch()
        return repo

    def _run_pipeline(self, adapter_cls, repo: Path, name: str):
        output = self.root / f"output_{name}"
        scan = adapter_cls(
            AdapterOptions(
                input_root=repo,
                output_root=output / "scan",
                verify_files=True,
                max_episodes=1,
            )
        ).run()
        self.assertEqual(scan["episode_count"], 1)
        clean = clean_manifest(output / "scan" / "episodes.jsonl", output / "clean")
        self.assertEqual(clean["action_eligible_count"], 1)
        materialized = materialize_canonical(
            output / "clean" / "episodes.cleaned.jsonl", output / "canonical"
        )
        self.assertEqual(materialized["materialized_episode_count"], 1)
        windows = build_window_index(
            output / "canonical" / "canonical_episodes.jsonl", output / "windows"
        )
        self.assertGreater(windows["action_window_count"], 0)
        cases = build_training_cases(
            output / "windows" / "windows.jsonl", output / "cases"
        )
        self.assertGreater(cases["case_count"], 0)
        case = next(iter_jsonl(output / "cases" / "training_cases.jsonl"))
        self.assertEqual(case["training"]["mode"], "joint_video_action")
        self.assertEqual(case["timeline"]["state_steps"], 81)
        self.assertEqual(case["timeline"]["action_steps"], 80)
        self.assertEqual(case["timeline"]["video_steps"], 21)
        return case

    def test_lingbot_unknown_schema_cannot_activate_inferred_action(self):
        variant = build_lingbot_variant(
            {
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [1],
                        "names": ["left_joint_1"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [1],
                        "names": ["left_joint_1"],
                    },
                }
            }
        )
        action_mapping = variant["canonical_mapping"]["action"]
        self.assertFalse(action_mapping["verified"])
        self.assertEqual(action_mapping["valid_slots"], [])
        self.assertEqual(action_mapping["candidate_valid_slots"], [14])

    def test_xdof_15d_schema_uses_audited_right_arm_and_camera_mapping(self):
        features = {
            "observation.state": {"dtype": "float32", "shape": [15], "names": None},
            "action": {"dtype": "float32", "shape": [15], "names": None},
        }
        for key in (
            "observation.images.left_eye",
            "observation.images.right_eye",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        ):
            feature_key, feature = _video_feature(key, 28.0)
            features[feature_key] = feature
        variant = build_lingbot_variant({"features": features})
        self.assertEqual(variant["id"], "xdof_right_joint_15d")
        self.assertEqual(
            variant["canonical_mapping"]["action"]["valid_slots"],
            [13, 21, 22, 23, 24, 25, 26, 27],
        )
        roles = {camera.source_key: camera.role for camera in variant["cameras"]}
        self.assertEqual(roles["observation.images.left_eye"], "global_primary")
        self.assertEqual(roles["observation.images.right_eye"], "left_wrist")
        self.assertEqual(roles["observation.images.right_wrist"], "right_wrist")

    def test_dreamzero_without_modality_cannot_activate_inferred_action(self):
        variant = build_dreamzero_variant(
            {
                "features": {
                    "observation.state": {
                        "dtype": "float64",
                        "shape": [1],
                        "names": ["left_joint_1"],
                    },
                    "action": {
                        "dtype": "float64",
                        "shape": [1],
                        "names": ["left_joint_1"],
                    },
                }
            },
            None,
        )
        action_mapping = variant["canonical_mapping"]["action"]
        self.assertFalse(action_mapping["verified"])
        self.assertEqual(action_mapping["valid_slots"], [])
        self.assertEqual(action_mapping["candidate_valid_slots"], [14])

    def test_lingbot_robotwin_real_schema_reaches_joint_training_case(self):
        fps = 50.0
        states: list[list[float]] = []
        for index in range(250):
            t = index / fps
            left_angle = 0.15 * math.sin(t)
            right_angle = 0.12 * math.cos(t)
            states.append(
                [
                    0.30 + 0.01 * math.sin(t),
                    -0.20 + 0.01 * math.cos(t),
                    0.80,
                    0.0,
                    0.0,
                    math.sin(left_angle / 2.0),
                    math.cos(left_angle / 2.0),
                    1.0,
                    0.30 + 0.01 * math.cos(t),
                    0.20 + 0.01 * math.sin(t),
                    0.80,
                    0.0,
                    0.0,
                    math.sin(right_angle / 2.0),
                    math.cos(right_angle / 2.0),
                    1.0,
                ]
            )
        actions = [*states[1:], states[-1]]
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [16],
                "names": [
                    [
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
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [16],
                "names": [
                    [
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
                ],
            },
        }
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            feature_key, feature = _video_feature(key, fps)
            features[feature_key] = feature
        repo = self._write_repo(
            "lingbot_robotwin",
            info={"robot_type": "aloha", "fps": fps, "features": features},
            states=states,
            actions=actions,
            episode_metadata={
                "action_config": [
                    {
                        "start_frame": 0,
                        "end_frame": len(states),
                        "action_text": "move the object to the target",
                    }
                ]
            },
        )
        case = self._run_pipeline(LingBotVAAdapter, repo, "robotwin")
        self.assertEqual(case["inputs"]["action_valid_slots"], list(range(14)))
        self.assertEqual(
            case["provenance"]["source_profile"]["family"],
            "lingbot_va_robotwin",
        )
        self.assertIn("action_config", case["provenance"]["source_episode_metadata"])

    def test_lingbot_libero_real_schema_reaches_joint_training_case(self):
        fps = 60.0
        states: list[list[float]] = []
        for index in range(300):
            t = index / fps
            states.append(
                [
                    0.1 * math.sin(t),
                    0.1 * math.cos(t),
                    0.6 + 0.02 * math.sin(t),
                    0.02 * math.sin(t),
                    0.03 * math.cos(t),
                    0.04 * math.sin(t),
                    0.03,
                    -0.03,
                ]
            )
        actions: list[list[float]] = []
        for index, state in enumerate(states):
            following = states[min(index + 1, len(states) - 1)]
            actions.append(
                [
                    (following[dimension] - state[dimension]) * fps
                    for dimension in range(6)
                ]
                + [-1.0]
            )
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": {
                    "motors": [
                        "x",
                        "y",
                        "z",
                        "roll",
                        "pitch",
                        "yaw",
                        "gripper",
                        "gripper",
                    ]
                },
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
            },
        }
        for key in (
            "observation.images.agentview_rgb",
            "observation.images.eye_in_hand_rgb",
        ):
            feature_key, feature = _video_feature(key, fps)
            features[feature_key] = feature
        repo = self._write_repo(
            "lingbot_libero",
            info={"robot_type": "Franka", "fps": fps, "features": features},
            states=states,
            actions=actions,
            episode_metadata={
                "action_config": [
                    {
                        "start_frame": 0,
                        "end_frame": len(states),
                        "action_text": "move the object to the target",
                    }
                ]
            },
        )
        case = self._run_pipeline(LingBotVAAdapter, repo, "libero")
        self.assertEqual(case["inputs"]["action_valid_slots"], list(range(7)))
        self.assertEqual(
            case["provenance"]["source_profile"]["family"],
            "lingbot_va_libero_long",
        )

    def test_dreamzero_real_schema_reaches_joint_training_case(self):
        fps = 15.0
        states: list[list[float]] = []
        for index in range(120):
            t = index / fps
            joints = [0.2 * math.sin(t + joint * 0.1) for joint in range(7)]
            states.append([0.3, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, *joints])
        actions: list[list[float]] = []
        for index, state in enumerate(states):
            following = states[min(index + 1, len(states) - 1)]
            row = [0.0] * 28
            row[12] = 0.0
            row[14:21] = following[7:14]
            actions.append(row)
        features = {
            "observation.state": {
                "dtype": "float64",
                "shape": [14],
                "names": ["cartesian_position", "gripper_position", "joint_position"],
            },
            "action": {
                "dtype": "float64",
                "shape": [28],
                "names": [
                    "cartesian_position",
                    "cartesian_velocity",
                    "gripper_position",
                    "gripper_velocity",
                    "joint_position",
                    "joint_velocity",
                ],
            },
        }
        for key in (
            "observation.images.exterior_image_1_left",
            "observation.images.exterior_image_2_left",
            "observation.images.wrist_image_left",
        ):
            feature_key, feature = _video_feature(key, fps, video_info=True)
            features[feature_key] = feature
        modality = {
            "state": {
                "cartesian_position": {"start": 0, "end": 6},
                "gripper_position": {"start": 6, "end": 7},
                "joint_position": {"start": 7, "end": 14},
            },
            "action": {
                "cartesian_position": {"start": 0, "end": 6},
                "cartesian_velocity": {"start": 6, "end": 12},
                "gripper_position": {"start": 12, "end": 13},
                "gripper_velocity": {"start": 13, "end": 14},
                "joint_position": {"start": 14, "end": 21},
                "joint_velocity": {"start": 21, "end": 28},
            },
            "video": {},
            "annotation": {},
        }
        repo = self._write_repo(
            "dreamzero",
            info={"robot_type": "droid", "fps": fps, "features": features},
            states=states,
            actions=actions,
            episode_metadata={"success": True},
            modality=modality,
        )
        case = self._run_pipeline(DreamZeroAdapter, repo, "dreamzero")
        self.assertEqual(case["inputs"]["action_valid_slots"], [6, *range(14, 21)])
        self.assertEqual(
            case["provenance"]["source_profile"]["family"],
            "dreamzero_droid",
        )
        self.assertTrue(case["provenance"]["source_episode_metadata"]["success"])


if __name__ == "__main__":
    unittest.main()
