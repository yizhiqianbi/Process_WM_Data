import unittest

import pyarrow as pa

from fastwam_preprocess.cleaning import (
    CleaningPolicy,
    _action_state_alignment,
    _signal_metrics,
)


class SignalAuditTest(unittest.TestCase):
    def test_stationary_segments_do_not_turn_normal_motion_into_jumps(self):
        values = [[0.0] for _ in range(60)]
        values.extend([[index * 0.01] for index in range(1, 41)])
        metrics = _signal_metrics(values)
        dimension = metrics["dimensions"][0]
        self.assertGreater(dimension["active_step_count"], 0)
        self.assertEqual(dimension["abrupt_step_count"], 0)

    def test_isolated_large_jump_is_reported_per_dimension(self):
        values = [[index * 0.01, 0.0] for index in range(100)]
        values[50][0] = 100.0
        metrics = _signal_metrics(values)
        self.assertGreater(metrics["dimensions"][0]["abrupt_step_count"], 0)
        self.assertEqual(metrics["dimensions"][1]["abrupt_step_count"], 0)

    def test_quaternion_sign_equivalence_does_not_create_component_jumps(self):
        values = []
        for index in range(100):
            angle = index * 0.001
            quaternion = [0.0, 0.0, angle, (1.0 - angle * angle) ** 0.5]
            if index % 2:
                quaternion = [-value for value in quaternion]
            values.append([0.0, 0.0, 0.0, *quaternion])
        feature = {
            "names": [
                "pose.position.x",
                "pose.position.y",
                "pose.position.z",
                "pose.orientation.x",
                "pose.orientation.y",
                "pose.orientation.z",
                "pose.orientation.w",
            ]
        }
        metrics = _signal_metrics(values, feature=feature, fps=20.0)
        quaternion = metrics["semantic"]["quaternions"][0]
        self.assertGreater(quaternion["sign_flip_count"], 0)
        self.assertEqual(metrics["maximum_abrupt_step_ratio"], 0.0)
        self.assertFalse(
            any(
                interval["reason"] == "extreme_abrupt_signal_step"
                for interval in metrics["bad_intervals"]
            )
        )

    def test_extreme_signal_jump_emits_hard_local_interval(self):
        values = [[index * 0.01] for index in range(100)]
        values[50][0] = 100.0
        metrics = _signal_metrics(values, fps=20.0, source_key="action")
        hard = [
            interval
            for interval in metrics["bad_intervals"]
            if interval["severity"] == "hard"
        ]
        self.assertTrue(hard)
        self.assertTrue(any(interval["start"] <= 50 < interval["stop_exclusive"] for interval in hard))

    def test_joint_position_floor_ignores_stationary_encoder_noise(self):
        values = [[(index % 7) * 1.0e-5] for index in range(100)]
        metrics = _signal_metrics(
            values,
            source_key="state/joint/position",
            policy=CleaningPolicy(joint_position_abrupt_step_floor_rad=0.01),
        )
        dimension = metrics["dimensions"][0]
        self.assertEqual(dimension["abrupt_threshold_floor"], 0.01)
        self.assertEqual(dimension["abrupt_step_count"], 0)

    def test_joint_position_floor_preserves_true_hard_jump(self):
        values = [[index * 0.001] for index in range(100)]
        values[50] = [1.0]
        metrics = _signal_metrics(
            values,
            source_key="action/joint/position",
            policy=CleaningPolicy(joint_position_abrupt_step_floor_rad=0.01),
        )
        self.assertTrue(
            any(
                interval["severity"] == "hard"
                and interval["start"] <= 50 < interval["stop_exclusive"]
                for interval in metrics["bad_intervals"]
            )
        )

    def test_discrete_gripper_transition_is_not_a_hard_interval(self):
        values = [[0.0] for _ in range(40)] + [[1.0] for _ in range(40)]
        metrics = _signal_metrics(
            values,
            feature={"names": ["left_gripper_command"]},
            source_key="fastwam.action.left_gripper",
        )
        self.assertEqual(
            metrics["dimensions"][0]["semantic_type"], "zero_order_gripper"
        )
        self.assertFalse(
            any(
                interval["severity"] == "hard"
                for interval in metrics["bad_intervals"]
            )
        )

    def test_velocity_action_is_aligned_to_state_derivative(self):
        state = [(index / 20.0) ** 2 for index in range(100)]
        action = [
            (state[index + 1] - state[index]) * 20.0
            if index + 1 < len(state)
            else (state[index] - state[index - 1]) * 20.0
            for index in range(len(state))
        ]
        table = pa.table(
            {
                "observation.torso.position": [[value] for value in state],
                "action.torso.velocities": [[value] for value in action],
            }
        )
        mapping = {
            "state": {
                "mappings": [
                    {
                        "active": True,
                        "canonical_index": 58,
                        "source_key": "observation.torso.position",
                        "source_index": 0,
                        "source_name": "torso_position_0",
                        "semantic": "torso_head_position_0",
                    }
                ]
            },
            "action": {
                "mappings": [
                    {
                        "active": True,
                        "canonical_index": 58,
                        "source_key": "action.torso.velocities",
                        "source_index": 0,
                        "source_name": "torso_velocity_0",
                        "semantic": "torso_head_velocity_0",
                    }
                ]
            },
        }
        result = _action_state_alignment(
            table,
            {"fps": 20.0, "metadata": {"canonical_mapping": mapping}},
            CleaningPolicy(),
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            result["comparison_modes"]["58"], "action_vs_state_derivative"
        )


if __name__ == "__main__":
    unittest.main()
