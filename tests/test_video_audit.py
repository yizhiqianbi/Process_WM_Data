import unittest

from fastwam_preprocess.video_audit import analyze_grayscale_frames


class SparseVisualAuditTest(unittest.TestCase):
    def test_consecutive_black_samples_create_hard_video_intervals(self):
        width = height = 64
        black = bytes(width * height)
        textured = bytes((index * 37) % 256 for index in range(width * height))
        report = analyze_grayscale_frames(
            [black, black, textured],
            timestamps=[0.0, 1.0, 2.0],
            width=width,
            height=height,
        )
        self.assertEqual(report["status"], "warning")
        self.assertIn("extreme_dark", report["flags"])
        hard = [
            interval
            for interval in report["bad_intervals"]
            if interval["severity"] == "hard"
        ]
        self.assertTrue(hard)
        self.assertTrue(all(interval["domains"] == ["video"] for interval in hard))

    def test_identical_textured_frames_are_reported_as_possible_freeze(self):
        width = height = 64
        textured = bytes((index * 17 + index // width * 13) % 256 for index in range(width * height))
        report = analyze_grayscale_frames(
            [textured, textured],
            timestamps=[0.0, 1.0],
            width=width,
            height=height,
        )
        self.assertTrue(report["all_pairs_frozen"])
        self.assertIn("possible_frozen_video", report["flags"])
        freeze = [
            interval
            for interval in report["bad_intervals"]
            if interval["reason"] == "possible_frozen_video"
        ]
        self.assertTrue(freeze)
        self.assertTrue(all(interval["severity"] == "soft" for interval in freeze))


if __name__ == "__main__":
    unittest.main()
