import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NormalizationStatsTest(unittest.TestCase):
    def test_pipeline_discovery_filters_to_train_a_tier_joint_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case_dir = root / "pipeline" / "agibot_beta" / "cases"
            case_dir.mkdir(parents=True)
            parquet = root / "canonical.parquet"
            state = [[float(index), *([0.0] * 79)] for index in range(130)]
            action = [[float(index + 1), *([0.0] * 79)] for index in range(130)]
            mask = [[True, *([False] * 79)] for _ in range(130)]
            pq.write_table(
                pa.table(
                    {
                        "canonical_state": state,
                        "state_dim_valid_mask": mask,
                        "canonical_action": action,
                        "action_dim_valid_mask": mask,
                    }
                ),
                parquet,
            )

            def case(domain: str, mode: str, tier: str, start: int = 0) -> dict:
                return {
                    "schema_version": "fastwam-training-case-v1",
                    "case_id": f"{domain}-{mode}-{start}",
                    "dataset": "agibot_beta",
                    "split": "train",
                    "embodiment": {
                        "name": "agibot_g1",
                        "normalization_domain": domain,
                    },
                    "training": {"mode": mode},
                    "quality": {"tier": tier},
                    "timeline": {"state_steps": 81, "action_steps": 80},
                    "sampling": {
                        "unit": "episode_start_range",
                        "valid_starts": {
                            "start": start,
                            "stop_exclusive": start + 1,
                            "stride": 40,
                            "count": 1,
                        },
                    },
                    "inputs": {
                        "canonical_parquet": str(parquet),
                        "state_column": "canonical_state",
                        "state_mask_column": "state_dim_valid_mask",
                        "action_column": "canonical_action",
                        "action_mask_column": "action_dim_valid_mask",
                        "state_slot_mask": [True, *([False] * 79)],
                        "action_slot_mask": [True, *([False] * 79)],
                    },
                }

            manifest = case_dir / "training_cases.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps(case("agibot_joint", "joint_video_action", "A")),
                        json.dumps(case("agibot_joint", "joint_video_action", "A", 40)),
                        json.dumps(case("agibot_video", "video_only", "B")),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "normalization_stats.json"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_fastwam_normalization_stats.py"),
                    "--pipeline-root",
                    str(root / "pipeline"),
                    "--datasets",
                    "agibot_beta",
                    "--data-root",
                    str(root),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(list(payload["domains"]), ["agibot_joint"])
            self.assertEqual(payload["source_splits"], ["train"])
            self.assertEqual(payload["source_training_modes"], ["joint_video_action"])
            self.assertEqual(payload["source_quality_tiers"], ["A"])
            stats = payload["domains"]["agibot_joint"]
            self.assertEqual(stats["window_count"], 2)
            self.assertEqual(stats["selected_state_row_count"], 121)
            self.assertEqual(stats["selected_action_row_count"], 120)
            self.assertEqual(stats["state"]["count"][0], 121)
            self.assertEqual(stats["action"]["count"][0], 120)
            self.assertAlmostEqual(stats["state"]["mean"][0], 60.0)
            self.assertAlmostEqual(stats["action"]["mean"][0], 60.5)
            self.assertEqual(
                payload["row_selection"],
                "unique_rows_covered_by_admitted_training_windows",
            )


if __name__ == "__main__":
    unittest.main()
