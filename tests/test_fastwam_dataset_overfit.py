from __future__ import annotations

import json

from tuning.fastwam_dataset_overfit import (
    build_dataset_overfit_plan,
    enumerate_dataset_windows,
    select_episode_probes,
)


def _case(case_id, episode, mode, split, start, stop, stride):
    return {
        "schema_version": "fastwam-training-case-v1",
        "case_id": case_id,
        "source_episode_id": episode,
        "split": split,
        "quality": {"tier": "A"},
        "training": {"mode": mode},
        "sampling": {
            "valid_starts": {
                "start": start,
                "stop_exclusive": stop,
                "stride": stride,
                "count": len(range(start, stop, stride)),
            }
        },
    }


def test_dataset_overfit_plan_matches_loader_order_and_prefers_joint_memory(tmp_path):
    manifest = tmp_path / "cases.jsonl"
    cases = [
        _case("a-video", "episode-a", "video_only", "train", 0, 80, 40),
        _case("a-joint", "episode-a", "joint_video_action", "train", 40, 160, 40),
        _case("b-joint", "episode-b", "joint_video_action", "validation", 0, 120, 40),
        _case("ignored", "episode-c", "joint_video_action", "test", 0, 80, 40),
    ]
    manifest.write_text(
        "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8"
    )

    windows = enumerate_dataset_windows(
        manifest,
        splits=("train", "validation"),
        modes=("joint_video_action", "video_only"),
    )
    assert [window.dataset_index for window in windows] == list(range(8))
    probes = select_episode_probes(windows)
    assert [(probe.source_episode_id, probe.mode, probe.window_start) for probe in probes] == [
        ("episode-a", "joint_video_action", 80),
        ("episode-b", "joint_video_action", 80),
    ]

    report = build_dataset_overfit_plan(manifest, training_probe_count=1)
    assert report["window_count"] == 8
    assert report["episode_count"] == 2
    assert report["window_counts_by_mode"] == {
        "joint_video_action": 6,
        "video_only": 2,
    }
    assert len(report["training_probe_indices"]) == 1
