from __future__ import annotations

import json

import pytest

from tuning.fastwam_overfit import (
    OverfitReportError,
    load_eval_records,
    summarize_overfit,
)


def _record(step: int, psnr: float, ssim: float, action_l1: float) -> dict:
    return {
        "step": step,
        "sample_index": 0,
        "window_start": 40,
        "memory_valid_ratio": 1.0,
        "psnr_rg": psnr,
        "ssim_rg": ssim,
        "action_l1": action_l1,
        "video_path": f"step_{step:06d}.mp4",
        "conditioning": {"gt_action_conditions_video": False},
    }


def test_overfit_summary_selects_candidate_that_passes_all_gates():
    report = summarize_overfit(
        [
            _record(0, 8.0, 0.20, 1.0),
            _record(50, 12.0, 0.30, 0.4),
            _record(100, 13.0, 0.40, 0.8),
        ]
    )

    assert report["status"] == "passed"
    assert report["selected"]["record"]["step"] == 50
    assert report["selected"]["checks"]["action_l1_ratio"]


def test_overfit_summary_requires_step_zero_baseline():
    with pytest.raises(OverfitReportError, match="step 0"):
        summarize_overfit([_record(50, 12.0, 0.30, 0.4)])


def test_load_eval_records_filters_rank_and_orders_steps(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    for step, rank in ((50, 0), (0, 0), (0, 1)):
        payload = _record(step, 8.0, 0.2, 1.0)
        (eval_dir / f"step_{step:06d}_rank_{rank:03d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    records = load_eval_records(tmp_path, rank=0)

    assert [record["step"] for record in records] == [0, 50]
