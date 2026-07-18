import math
import tempfile
import unittest
from pathlib import Path

import h5py
import pyarrow as pa
import pyarrow.parquet as pq

from fastwam_preprocess.source import ParquetSourceReader


class EpisodeSourceReaderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_parquet_next_row_derivation(self):
        path = self.root / "episode.parquet"
        pq.write_table(
            pa.table(
                {
                    "observation.robot.joints": [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [2.0, 1.0],
                    ],
                    "timestamp": [0.0, 0.2, 0.4],
                    "frame_index": [0, 1, 2],
                }
            ),
            path,
        )
        record = {
            "source_uri": str(path),
            "metadata": {
                "native_conversion": {
                    "source_format": "parquet",
                    "derived_columns": {
                        "fastwam.action.next_joint_target": {
                            "source_key": "observation.robot.joints",
                            "operation": "next_row_hold_last",
                        }
                    },
                }
            },
        }
        with ParquetSourceReader() as reader:
            table = reader.read_record(
                record,
                columns=["timestamp", "fastwam.action.next_joint_target"],
            )
        self.assertEqual(
            table["fastwam.action.next_joint_target"].to_pylist(),
            [[1.0, 0.0], [2.0, 1.0], [2.0, 1.0]],
        )

    def test_quaternion_pose_derivation_uses_declared_order(self):
        path = self.root / "quaternion.parquet"
        half_angle = math.pi / 4.0
        pq.write_table(
            pa.table(
                {
                    "pose": [
                        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                        [
                            1.0,
                            2.0,
                            3.0,
                            0.0,
                            0.0,
                            math.sin(half_angle),
                            math.cos(half_angle),
                        ],
                    ]
                }
            ),
            path,
        )
        record = {
            "source_uri": str(path),
            "metadata": {
                "native_conversion": {
                    "source_format": "parquet",
                    "derived_columns": {
                        "fastwam.pose": {
                            "source_key": "pose",
                            "operation": "pose_quaternion_to_rotvec",
                            "quaternion_order": "xyzw",
                        }
                    },
                }
            },
        }
        with ParquetSourceReader() as reader:
            table = reader.read_record(record, columns=["fastwam.pose"])
        first, second = table["fastwam.pose"].to_pylist()
        self.assertEqual(first, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(second[-1], math.pi / 2.0)

    def test_hdf5_reader_normalizes_nanosecond_timestamp(self):
        path = self.root / "proprio_stats.h5"
        with h5py.File(path, mode="w") as handle:
            handle.create_dataset("timestamp", data=[1_000_000_000, 1_100_000_000, 1_200_000_000])
            handle.create_dataset(
                "state/joint/position",
                data=[[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]],
            )
        record = {
            "source_uri": str(path),
            "fps": 10.0,
            "metadata": {
                "native_conversion": {
                    "source_format": "hdf5",
                    "timestamp_key": "timestamp",
                }
            },
        }
        with ParquetSourceReader() as reader:
            table = reader.read_record(
                record, columns=["timestamp", "frame_index", "state/joint/position"]
            )
        self.assertEqual(table["frame_index"].to_pylist(), [0, 1, 2])
        self.assertEqual(table["timestamp"].to_pylist(), [0.0, 0.1, 0.2])
        self.assertEqual(table["state/joint/position"][2].as_py(), [0.2, 1.2])

    def test_hdf5_reader_applies_intersection_of_native_valid_indices(self):
        path = self.root / "indexed_proprio_stats.h5"
        with h5py.File(path, mode="w") as handle:
            handle.create_dataset(
                "timestamp",
                data=[1_000_000_000 + index * 50_000_000 for index in range(6)],
            )
            handle.create_dataset(
                "state/joint/position", data=[[float(index)] for index in range(6)]
            )
            handle.create_dataset(
                "action/joint/position",
                data=[[float(index + 1)] for index in range(6)],
            )
            handle.create_dataset("action/joint/index", data=[1, 2, 3, 4])
            handle.create_dataset("action/head/index", data=[0, 1, 2, 3, 4])
        record = {
            "source_uri": str(path),
            "fps": 20.0,
            "metadata": {
                "native_conversion": {
                    "source_format": "hdf5",
                    "timestamp_key": "timestamp",
                    "valid_index_keys": [
                        "action/joint/index",
                        "action/head/index",
                    ],
                    "valid_index_policy": "intersection",
                }
            },
        }
        with ParquetSourceReader() as reader:
            table = reader.read_record(
                record,
                columns=[
                    "timestamp",
                    "frame_index",
                    "state/joint/position",
                    "action/joint/position",
                ],
            )
        self.assertEqual(table.num_rows, 4)
        self.assertEqual(table["frame_index"].to_pylist(), [1, 2, 3, 4])
        self.assertEqual(
            table["state/joint/position"].to_pylist(),
            [[1.0], [2.0], [3.0], [4.0]],
        )
        for actual, expected in zip(
            table["timestamp"].to_pylist(), [0.0, 0.05, 0.1, 0.15]
        ):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
