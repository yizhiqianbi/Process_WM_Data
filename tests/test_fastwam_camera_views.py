from __future__ import annotations

import numpy as np

from tuning.fastwam_camera_views import (
    CAMERA_PANELS,
    camera_pair_frame,
    crop_camera_panel,
    panel_psnr,
)


def test_camera_panels_cover_expected_training_layout_without_overlap():
    frame = np.zeros((384, 320, 3), dtype=np.uint8)
    frame[:256] = 10
    frame[256:, :160] = 20
    frame[256:, 160:] = 30

    crops = [crop_camera_panel(frame, panel) for panel in CAMERA_PANELS]

    assert [crop.shape for crop in crops] == [
        (256, 320, 3),
        (128, 160, 3),
        (128, 160, 3),
    ]
    assert [int(crop[0, 0, 0]) for crop in crops] == [10, 20, 30]


def test_camera_pair_has_stable_dimensions_for_global_and_wrist_panels():
    frame = np.zeros((384, 320, 3), dtype=np.uint8)

    for panel in CAMERA_PANELS:
        crop = crop_camera_panel(frame, panel)
        pair = camera_pair_frame(crop, crop, panel)
        assert pair.shape == (288, 640, 3)


def test_panel_psnr_is_infinite_for_exact_match_and_finite_for_error():
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    white = np.full((2, 2, 3), 255, dtype=np.uint8)

    assert panel_psnr([black], [black]) == float("inf")
    assert panel_psnr([black], [white]) == 0.0
