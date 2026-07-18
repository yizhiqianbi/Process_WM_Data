import io
import json
import math
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py

from fastwam_preprocess.adapters.agibot import AgiBotBetaAdapter
from fastwam_preprocess.adapters.base import AdapterOptions
from fastwam_preprocess.cleaning import audit_episode
from fastwam_preprocess.source import ParquetSourceReader
from fastwam_preprocess.utils import iter_jsonl


class AgiBotJoinTest(unittest.TestCase):
    def test_observation_and_proprio_archives_join_into_verified_action(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            observation_dir = root / "observations" / "327"
            proprio_dir = root / "proprio_stats"
            task_dir = root / "task_info"
            observation_dir.mkdir(parents=True)
            proprio_dir.mkdir(parents=True)
            task_dir.mkdir(parents=True)

            observation_tar = observation_dir / "100-100.tar"
            with tarfile.open(observation_tar, mode="w") as archive:
                member = tarfile.TarInfo("100/videos/head_color.mp4")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"0"))

            source_h5 = root / "proprio_stats.h5"
            state = [
                [math.sin(index * (0.03 + joint * 0.001)) for joint in range(14)]
                for index in range(100)
            ]
            action = [*state[1:], state[-1]]
            with h5py.File(source_h5, mode="w") as handle:
                handle.create_dataset(
                    "timestamp", data=[1_000_000_000 + index * 33_333_333 for index in range(100)]
                )
                handle.create_dataset("state/joint/position", data=state)
                handle.create_dataset("action/joint/position", data=action)
            proprio_tar = proprio_dir / "100-100.tar"
            with tarfile.open(proprio_tar, mode="w") as archive:
                archive.add(source_h5, arcname="327/100/proprio_stats.h5")

            (task_dir / "task_327.json").write_text(
                json.dumps(
                    [
                        {
                            "episode_id": 100,
                            "task_id": 327,
                            "task_name": "pick the object",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            output = root / "scan"
            real_tar_open = tarfile.open
            proprio_open_count = 0

            def tracked_tar_open(name, *args, **kwargs):
                nonlocal proprio_open_count
                if Path(name) == proprio_tar:
                    proprio_open_count += 1
                return real_tar_open(name, *args, **kwargs)

            with mock.patch(
                "fastwam_preprocess.adapters.agibot.tarfile.open",
                side_effect=tracked_tar_open,
            ):
                summary = AgiBotBetaAdapter(
                    AdapterOptions(root, output, max_episodes=1)
                ).run()
            self.assertEqual(summary["episode_count"], 1)
            self.assertEqual(proprio_open_count, 1)
            record = next(iter_jsonl(output / "episodes.jsonl"))
            self.assertEqual(record["num_frames"], 100)
            self.assertEqual(
                record["metadata"]["canonical_mapping"]["action"]["valid_slots"],
                list(range(14, 28)),
            )
            with ParquetSourceReader() as reader:
                audit = audit_episode(record, reader)
            self.assertTrue(audit["action_verified"])
            self.assertEqual(
                audit["metrics"]["action_state_alignment"]["best_lag_frames"], 1
            )


if __name__ == "__main__":
    unittest.main()
