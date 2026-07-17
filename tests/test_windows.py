import json
import tempfile
import unittest
from pathlib import Path

from fastwam_preprocess.utils import iter_jsonl
from fastwam_preprocess.windows import build_window_index


class WindowIntervalFilteringTest(unittest.TestCase):
    def test_action_bad_interval_downgrades_only_overlapping_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "canonical.jsonl"
            record = {
                "global_episode_id": "dataset/test/episode-1",
                "dataset": "test",
                "release": "test",
                "num_frames": 241,
                "fps": 20.0,
                "source_fps": 20.0,
                "cameras": [
                    {
                        "source_key": "head",
                        "role": "global_primary",
                        "source_uri": "/tmp/head.mp4",
                        "fps": 20.0,
                    }
                ],
                "quality": {
                    "tier": "A",
                    "video_eligible": True,
                    "action_eligible": True,
                },
                "training_admission": {"mode": "joint_video_action"},
                "canonical_parquet": "/tmp/episode.parquet",
                "state_valid_slots": [14],
                "action_valid_slots": [14],
                "bad_intervals": [
                    {
                        "timeline": "canonical_20hz",
                        "start": 100,
                        "stop_exclusive": 101,
                        "reason": "extreme_abrupt_signal_step",
                        "domains": ["action"],
                        "severity": "hard",
                    }
                ],
            }
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            output = root / "windows"
            summary = build_window_index(manifest, output, stride=40)
            self.assertEqual(summary["video_window_count"], 5)
            self.assertEqual(summary["action_window_count"], 3)
            self.assertEqual(summary["interval_downgraded_action_window_count"], 2)

            rows = list(iter_jsonl(output / "windows.jsonl"))
            joint_count = sum(
                row["valid_starts"]["count"]
                for row in rows
                if row["training_mode"] == "joint_video_action"
            )
            video_only_count = sum(
                row["valid_starts"]["count"]
                for row in rows
                if row["training_mode"] == "video_only"
            )
            self.assertEqual(joint_count, 3)
            self.assertEqual(video_only_count, 2)


if __name__ == "__main__":
    unittest.main()
