import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fastwam_preprocess.adapters.base import AdapterOptions
from fastwam_preprocess.adapters.robocoin import RoboCOINAdapter
from fastwam_preprocess.cleaning import clean_manifest
from fastwam_preprocess.materialize import materialize_canonical
from fastwam_preprocess.training_case import CAMERA_ROLES, build_training_cases
from fastwam_preprocess.utils import iter_jsonl
from fastwam_preprocess.windows import build_window_index


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "robot_task"
        (self.repo / "meta").mkdir(parents=True)
        (self.repo / "data" / "chunk-000").mkdir(parents=True)
        camera_dir = self.repo / "videos" / "chunk-000" / "observation.images.head_rgb"
        camera_dir.mkdir(parents=True)
        (camera_dir / "episode_000000.mp4").touch()
        info = {
            "robot_type": "test_bot",
            "fps": 20,
            "chunks_size": 1000,
            "features": {
                "observation.images.head_rgb": {
                    "dtype": "video",
                    "shape": [64, 64, 3],
                    "info": {"video.fps": 20, "video.height": 64, "video.width": 64},
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["left_arm_joint_1_rad", "left_gripper_open"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["left_arm_joint_1_rad", "left_gripper_open"],
                },
            },
        }
        (self.repo / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
        (self.repo / "meta" / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "length": 100, "tasks": ["test"]}) + "\n",
            encoding="utf-8",
        )
        table = pa.table(
            {
                "observation.state": [[i / 100, 1.0] for i in range(100)],
                "action": [[(i + 1) / 100, 1.0] for i in range(100)],
                "timestamp": [i / 20 for i in range(100)],
                "frame_index": list(range(100)),
            }
        )
        pq.write_table(table, self.repo / "data" / "chunk-000" / "episode_000000.parquet")

    def tearDown(self):
        self.temp.cleanup()

    def test_end_to_end(self):
        scan_dir = self.root / "scan"
        summary = RoboCOINAdapter(
            AdapterOptions(self.root, scan_dir, verify_files=True)
        ).run()
        self.assertEqual(summary["episode_count"], 1)
        self.assertEqual(summary["quality_tiers"]["B"], 1)

        clean_dir = self.root / "clean"
        clean = clean_manifest(scan_dir / "episodes.jsonl", clean_dir)
        self.assertEqual(clean["action_eligible_count"], 1)
        self.assertEqual(clean["quality_tiers"]["A"], 1)

        canonical_dir = self.root / "canonical"
        materialized = materialize_canonical(
            clean_dir / "episodes.cleaned.jsonl", canonical_dir
        )
        self.assertEqual(materialized["materialized_episode_count"], 1)
        parquet = next((canonical_dir / "episodes").glob("*/canonical.parquet"))
        table = pq.read_table(parquet)
        self.assertEqual(table.num_rows, 100)
        self.assertEqual(len(table["canonical_action"][0].as_py()), 80)

        windows_dir = self.root / "windows"
        windows = build_window_index(
            canonical_dir / "canonical_episodes.jsonl", windows_dir, stride=10
        )
        self.assertEqual(windows["window_count"], 2)
        self.assertEqual(windows["video_points"], 21)

        cases_dir = self.root / "cases"
        cases = build_training_cases(windows_dir / "windows.jsonl", cases_dir)
        self.assertEqual(cases["case_count"], 1)
        self.assertEqual(cases["represented_window_count"], 2)
        case = next(iter_jsonl(cases_dir / "training_cases.jsonl"))
        self.assertEqual(case["timeline"]["state_steps"], 81)
        self.assertEqual(case["timeline"]["action_steps"], 80)
        self.assertEqual(case["timeline"]["video_steps"], 21)
        self.assertEqual(
            [slot["role"] for slot in case["inputs"]["camera_slots"]],
            list(CAMERA_ROLES),
        )
        self.assertTrue(case["training"]["loss_mask"]["action"])

        audit = next(iter_jsonl(clean_dir / "cleaning_report.jsonl"))
        signal = audit["metrics"]["signals"]["observation.state"]
        self.assertEqual(signal["dimension_count"], 2)
        self.assertEqual(len(signal["dimensions"]), 2)

    def test_bad_timestamp_is_rejected_for_action(self):
        parquet = self.repo / "data" / "chunk-000" / "episode_000000.parquet"
        table = pq.read_table(parquet)
        table = table.set_column(
            table.schema.get_field_index("timestamp"),
            "timestamp",
            pa.array([0.0] * 100),
        )
        pq.write_table(table, parquet)
        scan_dir = self.root / "scan_bad"
        RoboCOINAdapter(AdapterOptions(self.root, scan_dir)).run()
        clean_dir = self.root / "clean_bad"
        summary = clean_manifest(scan_dir / "episodes.jsonl", clean_dir)
        self.assertEqual(summary["action_eligible_count"], 0)
        self.assertEqual(summary["quality_tiers"]["C"], 1)
        self.assertEqual(summary["training_admissions"]["reject"], 1)

    def test_video_only_native_timeline_gets_the_same_81_step_case(self):
        manifest = self.root / "native_video.jsonl"
        record = {
            "dataset": "agibot_beta",
            "release": "test",
            "source_episode_id": "episode-1",
            "global_episode_id": "agibot_beta/test/g1/task/episode-1",
            "source_uri": "tar:///tmp/observations.tar!episode-1/",
            "embodiment": "agibot_g1",
            "robot_type": "agibot_g1",
            "task_namespace": "pick",
            "tasks": ["pick the object"],
            "num_frames": 121,
            "duration_s": 4.0,
            "fps": 30.0,
            "cameras": [
                {
                    "source_key": "head_rgb",
                    "role": "global_primary",
                    "fps": 30.0,
                    "source_uri": "tar:///tmp/observations.tar!episode-1/head.mp4",
                }
            ],
            "state_schema": {},
            "action_schema": {},
            "references": {},
            "metadata": {},
        }
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        clean_dir = self.root / "native_clean"
        cleaned = clean_manifest(manifest, clean_dir)
        self.assertEqual(cleaned["training_admissions"]["video_only"], 1)

        canonical_dir = self.root / "native_canonical"
        materialized = materialize_canonical(
            clean_dir / "episodes.cleaned.jsonl", canonical_dir
        )
        self.assertEqual(materialized["materialized_episode_count"], 1)
        sidecar = next(iter_jsonl(canonical_dir / "canonical_episodes.jsonl"))
        self.assertEqual(sidecar["num_frames"], 81)
        self.assertEqual(sidecar["timeline_source"], "metadata_fps")
        self.assertEqual(sidecar["state_valid_slots"], [])

        windows_dir = self.root / "native_windows"
        build_window_index(canonical_dir / "canonical_episodes.jsonl", windows_dir)
        cases_dir = self.root / "native_cases"
        cases = build_training_cases(windows_dir / "windows.jsonl", cases_dir)
        self.assertEqual(cases["case_count"], 1)
        case = next(iter_jsonl(cases_dir / "training_cases.jsonl"))
        self.assertEqual(case["training"]["mode"], "video_only")
        self.assertFalse(case["training"]["loss_mask"]["action"])
        self.assertEqual(case["timeline"]["state_steps"], 81)


if __name__ == "__main__":
    unittest.main()
