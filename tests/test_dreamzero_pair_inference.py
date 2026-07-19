from __future__ import annotations

import unittest

import numpy as np

from scripts.run_dreamzero_pair_inference import _processed_gt_composite
from scripts.run_dreamzero_pair_inference_8gpu import _partition
from targets.dreamzero.inference import (
    chunk_segments,
    decode_context_latent_bounds,
    decoded_frame_count,
    flatten_action_output,
    observation_frame_ids,
    select_episodes,
    split_composite,
)


class DreamZeroPairInferenceTests(unittest.TestCase):
    def test_gt_transform_is_bounded_to_four_frame_batches(self) -> None:
        cameras = {
            "video.left_eye": np.full((9, 20, 30, 3), 10, dtype=np.uint8),
            "video.right_eye": np.full((9, 20, 30, 3), 20, dtype=np.uint8),
            "video.right_wrist": np.full((9, 20, 30, 3), 30, dtype=np.uint8),
        }
        result = _processed_gt_composite(
            cameras,
            episode_index=5,
        )
        self.assertEqual(result.shape, (9, 352, 640, 3))
        self.assertTrue(np.all(result[:, :176, :320] == 10))
        self.assertTrue(np.all(result[:, 176:, :320] == 20))
        self.assertTrue(np.all(result[:, :176, 320:] == 30))
        self.assertTrue(np.all(result[:, 176:, 320:] == 0))

    def test_eight_gpu_partition_uses_four_balanced_cfg_workers(self) -> None:
        self.assertEqual(
            _partition([7, 8, 32, 3, 2, 20, 22, 10], 4),
            [[7, 2], [8, 20], [32, 22], [3, 10]],
        )

    def test_ten_chunks_produce_81_frames(self) -> None:
        self.assertEqual(decoded_frame_count(10), 81)
        self.assertEqual(decoded_frame_count(11), 89)
        self.assertEqual(observation_frame_ids(0, start_frame=0), [0])
        self.assertEqual(observation_frame_ids(1, start_frame=0), [5, 6, 7, 8])
        self.assertEqual(observation_frame_ids(9, start_frame=0), [69, 70, 71, 72])

    def test_long_rollout_uses_bounded_cache_segments(self) -> None:
        self.assertEqual(decoded_frame_count(114), 913)
        self.assertEqual(
            chunk_segments(114, 24),
            [(0, 24), (24, 48), (48, 72), (72, 96), (96, 114)],
        )
        self.assertEqual(decode_context_latent_bounds(23, 24), (0, 47))
        self.assertEqual(decode_context_latent_bounds(24, 24), (48, 49))
        self.assertEqual(decode_context_latent_bounds(25, 24), (48, 51))
        self.assertEqual(decode_context_latent_bounds(113, 24), (192, 227))

    def test_schedule_never_injects_the_chunk_target(self) -> None:
        for chunk_index in range(1, 10):
            injected = observation_frame_ids(chunk_index, start_frame=11)
            prediction_first_frame = 11 + chunk_index * 8 + 1
            self.assertLess(max(injected), prediction_first_frame)

    def test_episode_selection_requires_full_horizon(self) -> None:
        episodes = [
            {"episode_index": 0, "length": 80},
            {"episode_index": 1, "length": 120},
            {"episode_index": 2, "length": 100},
        ]
        selected = select_episodes(
            episodes,
            requested=[],
            num_cases=2,
            required_source_frames=81,
            start_frame=0,
        )
        self.assertEqual(selected, [1, 2])

    def test_composite_layout_matches_dream_transform(self) -> None:
        composite = np.zeros((2, 8, 10, 3), dtype=np.uint8)
        composite[:, :4, :5] = 10
        composite[:, 4:, :5] = 20
        composite[:, :4, 5:] = 30
        views = split_composite(composite, single_height=4, single_width=5)
        self.assertTrue(np.all(views["video.left_eye"] == 10))
        self.assertTrue(np.all(views["video.right_eye"] == 20))
        self.assertTrue(np.all(views["video.right_wrist"] == 30))

    def test_action_output_uses_executed_mpc_prefix(self) -> None:
        joint = np.arange(24 * 7, dtype=np.float32).reshape(24, 7)
        gripper = np.arange(24, dtype=np.float32)
        result = flatten_action_output(
            {
                "action.right_joint_position": joint,
                "action.right_gripper_position": gripper,
            },
            executed_steps=8,
        )
        self.assertEqual(result.shape, (8, 8))
        np.testing.assert_array_equal(result[:, :7], joint[:8])
        np.testing.assert_array_equal(result[:, 7], gripper[:8])


if __name__ == "__main__":
    unittest.main()
