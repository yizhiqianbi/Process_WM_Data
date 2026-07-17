import unittest

from fastwam_preprocess.canonical import infer_canonical_mapping


class CanonicalMappingTest(unittest.TestCase):
    def test_strict_mapping_does_not_copy_euler_to_rotation_vector(self):
        schema = {
            "action": {
                "names": [
                    "left_arm_joint_1_rad",
                    "left_gripper_open",
                    "left_eef_rot_euler_x_rad",
                ]
            }
        }
        mapping = infer_canonical_mapping(schema, kind="action")
        self.assertEqual(mapping["valid_slots"], [6, 14])
        self.assertTrue(mapping["verified"])
        names = [item["source_name"] for item in mapping["mappings"]]
        self.assertNotIn("left_eef_rot_euler_x_rad", names)

    def test_frame_sensitive_state_position_is_inactive(self):
        schema = {"observation.state": {"names": ["left_eef_pos_x_m"]}}
        mapping = infer_canonical_mapping(schema, kind="state")
        self.assertEqual(mapping["valid_slots"], [])
        self.assertFalse(mapping["mappings"][0]["active"])

    def test_ros_array_joint_names(self):
        schema = {
            "action.left_arm": {
                "names": [
                    "/motion_target/target_joint_state_arm_left.position[0]",
                    "/motion_target/target_joint_state_arm_left.position[1]",
                ]
            }
        }
        mapping = infer_canonical_mapping(schema, kind="action")
        self.assertEqual(mapping["valid_slots"], [14, 15])
        self.assertTrue(mapping["verified"])

    def test_torso_twist_mapping(self):
        schema = {
            "action.torso.velocities": {
                "names": [
                    "/motion_target/target_speed_torso.twist.linear.x",
                    "/motion_target/target_speed_torso.twist.angular.z",
                ]
            }
        }
        mapping = infer_canonical_mapping(schema, kind="action")
        self.assertEqual(mapping["valid_slots"], [58, 63])

    def test_interndata_context_maps_zero_based_joints_and_body_parts(self):
        schema = {
            "actions.joint.position": {
                "names": [
                    *[f"left_arm_{index}" for index in range(7)],
                    *[f"right_arm_{index}" for index in range(7)],
                ]
            },
            "actions.effector.position": {
                "names": ["left_gripper", "right_gripper"]
            },
            "actions.waist.position": {"names": ["pitch", "lift"]},
            "actions.head.position": {"names": ["yaw", "patch"]},
        }
        mapping = infer_canonical_mapping(schema, kind="action")
        self.assertTrue(mapping["verified"])
        self.assertEqual(
            mapping["valid_slots"],
            [6, 13, *range(14, 28), 58, 59, 60, 61],
        )

    def test_official_interndata_zero_based_joint_names_use_feature_context(self):
        schema = {
            "actions.left_joint.position": {
                "names": [f"left_joint_{index}" for index in range(6)]
            },
            "actions.right_joint.position": {
                "names": [f"right_joint_{index}" for index in range(6)]
            },
        }
        mapping = infer_canonical_mapping(schema, kind="action")
        self.assertTrue(mapping["verified"])
        self.assertEqual(
            mapping["valid_slots"],
            [*range(14, 20), *range(21, 27)],
        )
