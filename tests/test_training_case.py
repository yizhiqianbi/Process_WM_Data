import json
import tempfile
import unittest
from pathlib import Path

from fastwam_preprocess.training_case import (
    CAMERA_ROLES,
    VIDEO_OFFSETS,
    build_training_cases,
)
from fastwam_preprocess.utils import iter_jsonl, read_json


class TrainingCaseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _row(
        self,
        episode_id: str,
        *,
        mode: str = "video_only",
        fps: float = 20.0,
        split_group: str = "shared-lineage",
    ) -> dict:
        action_slots = [6, 14] if mode == "joint_video_action" else []
        tier = "A" if mode == "joint_video_action" else "B"
        return {
            "schema_version": "fastwam-window-index-v1",
            "global_episode_id": episode_id,
            "dataset": "oxe_auge",
            "release": "test",
            "lineage_id": "source-episode-7",
            "split_group_id": split_group,
            "embodiment": "test_bot",
            "robot_type": "test_bot",
            "task_namespace": "pick_place",
            "tasks": ["pick up the block", "pick up the block"],
            "target_fps": fps,
            "num_frames": 81,
            "action_horizon": 80,
            "video_sample_offsets": list(VIDEO_OFFSETS),
            "video_eligible": True,
            "action_eligible": mode == "joint_video_action",
            "training_mode": mode,
            "canonical_parquet": f"/tmp/{episode_id}.parquet",
            "state_valid_slots": [6, 14],
            "action_valid_slots": action_slots,
            "normalization_domain": "test-domain",
            "videos": ["legacy/list/style/video/member.mp4"],
            "audit_summary": {
                "metrics_summary": {
                    "videos": {"left_wrist": {"status": "failed"}}
                }
            },
            "cameras": [
                {
                    "source_key": "head_left",
                    "role": "global_primary",
                    "source_uri": "/tmp/head_left.mp4",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
                {
                    "source_key": "head_right",
                    "role": "global_primary",
                    "source_uri": "/tmp/head_right.mp4",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
                {
                    "source_key": "left_wrist",
                    "role": "left_wrist",
                    "source_uri": "/tmp/left_wrist.mp4",
                    "width": 320,
                    "height": 240,
                    "fps": 30,
                },
            ],
            "quality": {
                "tier": tier,
                "score": 0.9,
                "video_eligible": True,
                "action_eligible": mode == "joint_video_action",
            },
            "valid_starts": {
                "start": 0,
                "stop_exclusive": 120,
                "stride": 40,
                "count": 3,
            },
        }

    def test_heterogeneous_cases_share_one_contract(self):
        rows = [
            self._row("original", mode="video_only"),
            self._row("augmented", mode="joint_video_action"),
            self._row("wrong-fps", fps=10.0, split_group="other"),
        ]
        wrong_window = self._row("wrong-window", split_group="window-other")
        wrong_window["num_frames"] = 65
        wrong_window["action_horizon"] = 64
        wrong_window["video_sample_offsets"] = list(range(0, 65, 4))
        rows.append(wrong_window)
        manifest = self.root / "windows.jsonl"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        output = self.root / "cases"
        summary = build_training_cases(manifest, output)
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["represented_window_count"], 6)
        self.assertEqual(summary["rejected_case_count"], 2)

        cases = list(iter_jsonl(output / "training_cases.jsonl"))
        self.assertEqual(cases[0]["split"], cases[1]["split"])
        self.assertFalse(cases[0]["training"]["loss_mask"]["action"])
        self.assertTrue(cases[1]["training"]["loss_mask"]["action"])
        self.assertEqual(len(cases[1]["inputs"]["action_slot_mask"]), 80)
        self.assertEqual(
            [slot["role"] for slot in cases[0]["inputs"]["camera_slots"]],
            list(CAMERA_ROLES),
        )
        self.assertTrue(cases[0]["inputs"]["camera_slots"][1]["valid"])
        self.assertFalse(cases[0]["inputs"]["camera_slots"][2]["valid"])
        self.assertIn("left_wrist", cases[0]["inputs"]["failed_camera_source_keys"])
        self.assertEqual(cases[0]["language"]["all"], ["pick up the block"])

        example = read_json(output / "example_case.json")
        self.assertEqual(example["window"]["state_stop_exclusive"], 81)
        self.assertEqual(example["window"]["action_stop_exclusive"], 80)
        self.assertEqual(len(example["window"]["video_frame_indices"]), 21)

        rejected = list(iter_jsonl(output / "cases_rejected.jsonl"))
        all_failures = {
            failure for row in rejected for failure in row["failures"]
        }
        self.assertIn("target_fps_must_be_20", all_failures)
        self.assertIn("state_steps_must_be_81", all_failures)

    def test_empty_dataset_still_emits_contract_and_explicit_unavailable_example(self):
        manifest = self.root / "empty.jsonl"
        manifest.touch()
        output = self.root / "empty_cases"
        summary = build_training_cases(manifest, output)
        self.assertEqual(summary["case_count"], 0)
        self.assertTrue((output / "training_case_contract.json").is_file())
        example = read_json(output / "example_case.json")
        self.assertFalse(example["available"])
        self.assertEqual(example["reason"], "no_admissible_materialized_window")


if __name__ == "__main__":
    unittest.main()
