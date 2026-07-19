from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fastwam_preprocess.utils import iter_jsonl, read_json
from targets.common import LeRobotV2Dataset, TargetPreparationError, load_target_profile
from targets.dreamzero import prepare_dreamzero_target
from targets.lingbot_va import prepare_lingbot_va_target, validate_lingbot_va_target


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _build_lerobot_fixture(
    root: Path,
    *,
    action_width: int,
    state_width: int,
    cameras: list[str],
    episodes: int = 2,
    length: int = 32,
) -> None:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_width],
            "names": None,
        },
        "action": {"dtype": "float32", "shape": [action_width], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera in cameras:
        features[camera] = {
            "dtype": "video",
            "shape": [3, 16, 16],
            "names": ["channels", "height", "width"],
            "info": {"video.fps": 20.0, "video.codec": "h264"},
        }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "fixture",
        "fps": 20,
        "total_episodes": episodes,
        "total_frames": episodes * length,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    _write_json(root / "meta" / "info.json", info)
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": index,
                "tasks": ["move the object to the correct place"],
                "length": length,
            }
            for index in range(episodes)
        ],
    )
    _write_jsonl(
        root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": "move the object to the correct place"}],
    )
    for episode_index in range(episodes):
        frame = np.arange(length, dtype=np.float32)[:, None]
        action = frame * 0.01 + np.arange(action_width, dtype=np.float32)[None, :]
        state = frame * 0.02 + np.arange(state_width, dtype=np.float32)[None, :]
        table = pa.table(
            {
                "observation.state": pa.array(
                    state.tolist(), type=pa.list_(pa.float32(), state_width)
                ),
                "action": pa.array(
                    action.tolist(), type=pa.list_(pa.float32(), action_width)
                ),
                "timestamp": pa.array(np.arange(length) / 20.0, type=pa.float32()),
                "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                "frame_index": pa.array(range(length), type=pa.int64()),
                "index": pa.array(
                    range(episode_index * length, (episode_index + 1) * length),
                    type=pa.int64(),
                ),
                "task_index": pa.array([0] * length, type=pa.int64()),
            }
        )
        parquet_path = (
            root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)
        for camera in cameras:
            video_path = (
                root
                / "videos"
                / "chunk-000"
                / camera
                / f"episode_{episode_index:06d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"fixture-video")


class TargetPreparationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_target_reader_rejects_lerobot_v3_layout(self) -> None:
        source = self.root / "v3"
        _build_lerobot_fixture(
            source,
            action_width=7,
            state_width=7,
            cameras=["observation.images.camera"],
        )
        info_path = source / "meta" / "info.json"
        info = read_json(info_path)
        info["codebase_version"] = "v3.0"
        _write_json(info_path, info)
        with self.assertRaisesRegex(TargetPreparationError, "requires LeRobot v2"):
            LeRobotV2Dataset(source)

    def test_lingbot_robotwin_profile_generates_action_config_and_latent_jobs(self) -> None:
        cameras = [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
        source = self.root / "robotwin"
        output = self.root / "robotwin_lingbot"
        _build_lerobot_fixture(
            source, action_width=16, state_width=16, cameras=cameras
        )
        document, profile = load_target_profile(
            ROOT / "configs" / "targets" / "lingbot_va.yaml",
            "lingbot_va",
            "robotwin",
        )
        result = prepare_lingbot_va_target(
            source,
            output,
            profile_document=document,
            profile=profile,
            profile_name="robotwin",
            verify_files=True,
        )
        self.assertTrue(result["valid"])
        self.assertFalse(result["ready_for_training"])
        self.assertEqual(result["latent_job_count"], 6)
        self.assertEqual(result["missing_latent_count"], 6)
        profile_out = read_json(output / "meta" / "lingbot_va_model_profile.json")
        self.assertEqual(profile_out["action_dim"], 30)
        self.assertEqual(profile_out["compact_action_width"], 16)
        self.assertEqual(len(profile_out["norm_stat"]["q01"]), 30)
        receipt = read_json(output / "meta" / "lingbot_va_target_receipt.json")
        self.assertEqual(receipt["action_column"], "action")
        self.assertAlmostEqual(receipt["action_stats"]["mean"][1], 0.155, places=5)
        self.assertAlmostEqual(receipt["action_stats"]["mean"][8], 0.155, places=5)
        episodes = list(iter_jsonl(output / "meta" / "episodes.jsonl"))
        self.assertEqual(episodes[0]["action_config"][0]["end_frame"], 32)
        self.assertTrue((output / "data").is_symlink())
        required = validate_lingbot_va_target(output, require_latents=True)
        self.assertFalse(required["valid"])
        self.assertEqual(required["failures"], ["missing_latents:6"])

    def test_lingbot_custom_profile_compacts_15d_action_to_8d(self) -> None:
        cameras = [
            "observation.images.left_eye",
            "observation.images.right_eye",
            "observation.images.right_wrist",
        ]
        source = self.root / "custom"
        output = self.root / "custom_lingbot"
        _build_lerobot_fixture(
            source, action_width=15, state_width=15, cameras=cameras
        )
        document, profile = load_target_profile(
            ROOT / "configs" / "targets" / "lingbot_va.yaml",
            "lingbot_va",
            "take_wrong_item_right_arm",
        )
        result = prepare_lingbot_va_target(
            source,
            output,
            profile_document=document,
            profile=profile,
            profile_name="take_wrong_item_right_arm",
        )
        self.assertTrue(result["valid"])
        prepared = LeRobotV2Dataset(output)
        self.assertEqual(prepared.read_column(0, "action").shape, (32, 8))
        self.assertEqual(prepared.feature_width("action"), 8)
        model_profile = read_json(output / "meta" / "lingbot_va_model_profile.json")
        self.assertEqual(
            model_profile["used_action_channel_ids"], [21, 22, 23, 24, 25, 26, 27, 29]
        )
        self.assertFalse((output / "data").is_symlink())

    def test_dreamzero_profile_generates_gear_metadata_and_language(self) -> None:
        cameras = [
            "observation.images.left_eye",
            "observation.images.right_eye",
            "observation.images.right_wrist",
        ]
        source = self.root / "custom_dreamzero_source"
        output = self.root / "custom_dreamzero"
        _build_lerobot_fixture(
            source, action_width=15, state_width=15, cameras=cameras
        )
        document, profile = load_target_profile(
            ROOT / "configs" / "targets" / "dreamzero.yaml",
            "dreamzero",
            "take_wrong_item_right_arm",
        )
        result = prepare_dreamzero_target(
            source,
            output,
            profile_document=document,
            profile=profile,
            profile_name="take_wrong_item_right_arm",
            verify_files=True,
            max_quantile_rows=1000,
        )
        self.assertTrue(result["valid"])
        self.assertFalse(result["ready_for_training"])
        self.assertTrue(result["requires_upstream_profile_registration"])
        modality = read_json(output / "meta" / "modality.json")
        self.assertEqual(
            modality["state"]["right_joint_position"]["start"], 0
        )
        self.assertEqual(
            modality["action"]["right_gripper_position"]["start"], 14
        )
        relative = read_json(output / "meta" / "relative_stats_dreamzero.json")
        self.assertIn("right_joint_position", relative)
        table = pq.read_table(
            output / "data" / "chunk-000" / "episode_000000.parquet",
            columns=["annotation.task"],
        )
        self.assertEqual(
            table["annotation.task"][0].as_py(), "move the object to the correct place"
        )
        self.assertTrue((output / "meta" / "dreamzero_hydra_patch.yaml").is_file())
        self.assertFalse((output / "data").is_symlink())


if __name__ == "__main__":
    unittest.main()
