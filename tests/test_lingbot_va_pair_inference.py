from __future__ import annotations

import unittest

import numpy as np

from scripts.run_lingbot_va_pair_inference import (
    _feedback_action_state,
    _flatten_useful_predicted_actions,
    _select_episodes,
    _split_prediction,
    _video_frame_count,
)


class LingBotVAPairInferenceTests(unittest.TestCase):
    def test_ten_chunks_decode_to_157_frames(self) -> None:
        self.assertEqual(_video_frame_count(10, 4), 157)

    def test_longest_eligible_episode_selection_is_deterministic(self) -> None:
        episodes = [
            {"episode_index": 0, "length": 624},
            {"episode_index": 1, "length": 900},
            {"episode_index": 2, "length": 700},
            {"episode_index": 3, "length": 800},
        ]
        selected = _select_episodes(
            episodes,
            requested=[],
            num_cases=2,
            required_source_frames=625,
            start_frame=0,
        )
        self.assertEqual(selected, [1, 3])

    def test_requested_short_episode_is_rejected(self) -> None:
        episodes = [{"episode_index": 4, "length": 624}]
        with self.assertRaisesRegex(ValueError, "too short"):
            _select_episodes(
                episodes,
                requested=[4],
                num_cases=1,
                required_source_frames=625,
                start_frame=0,
            )

    def test_prediction_split_preserves_camera_order(self) -> None:
        prediction = np.zeros((2, 4, 9, 3), dtype=np.uint8)
        prediction[:, :, 3:6] = 1
        prediction[:, :, 6:9] = 2
        split = _split_prediction(prediction, ["a", "b", "c"], width=3, height=4)
        self.assertEqual(list(split), ["a", "b", "c"])
        self.assertTrue(np.all(split["a"] == 0))
        self.assertTrue(np.all(split["b"] == 1))
        self.assertTrue(np.all(split["c"] == 2))

    def test_first_feedback_chunk_has_dummy_action_frame(self) -> None:
        actions = np.arange(100 * 2, dtype=np.float32).reshape(100, 2)
        state, cursor, source_range = _feedback_action_state(
            actions,
            action_cursor=0,
            feedback_chunk_index=0,
            frame_chunk_size=4,
            action_per_frame=16,
        )
        self.assertEqual(state.shape, (2, 4, 16))
        self.assertTrue(np.all(state[:, 0] == 0))
        np.testing.assert_array_equal(
            state[:, 1:].transpose(1, 2, 0).reshape(48, 2), actions[:48]
        )
        self.assertEqual(cursor, 48)
        self.assertEqual(source_range, (0, 48))

    def test_predicted_action_export_drops_only_initial_dummy_frame(self) -> None:
        first = np.zeros((2, 4, 3), dtype=np.float32)
        second = np.ones((2, 4, 3), dtype=np.float32)
        flattened = _flatten_useful_predicted_actions([first, second])
        self.assertEqual(flattened.shape, (21, 2))
        self.assertTrue(np.all(flattened[:9] == 0))
        self.assertTrue(np.all(flattened[9:] == 1))


if __name__ == "__main__":
    unittest.main()
