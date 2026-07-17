from __future__ import annotations

from pathlib import Path
from typing import Any

from .intervals import blocked_window_starts, compact_start_ranges
from .utils import AtomicJsonlWriter, iter_jsonl, stable_id, write_json


def _source_video_fps(episode: dict[str, Any]) -> float | None:
    cameras = episode.get("cameras") or []
    preferred = [
        camera
        for camera in cameras
        if str(camera.get("role") or "").startswith("global_primary")
    ]
    candidates = [camera.get("fps") for camera in preferred or cameras]
    candidates.append(episode.get("source_fps"))
    for value in candidates:
        try:
            fps = float(value)
        except (TypeError, ValueError):
            continue
        if fps > 0:
            return fps
    return None


def _estimated_unique_video_frames(
    episode: dict[str, Any], *, num_frames: int, video_ratio: int
) -> int | None:
    source_fps = _source_video_fps(episode)
    target_fps = float(episode.get("fps") or 20.0)
    if source_fps is None or target_fps <= 0:
        return None
    duration_s = (num_frames - 1) / target_fps
    video_points = (num_frames - 1) // video_ratio + 1
    return min(video_points, int(duration_s * source_fps + 1e-8) + 1)


def _window_stage_admission(
    episode: dict[str, Any], *, mode: str, unique_video_frames: int | None
) -> dict[str, Any]:
    source = episode.get("stage_admission") or (
        (episode.get("training_admission") or {}).get("stages") or {}
    )
    stages = {key: dict(value) for key, value in source.items()}
    stages.setdefault("stage1_video_backbone", {})["accepted"] = True
    stages["stage1_video_backbone"]["estimated_unique_video_frames"] = unique_video_frames
    stages.setdefault("stage2_memory_fastwam", {})["accepted"] = (
        mode == "joint_video_action"
    )
    stages.setdefault("stage3_target_finetune", {})["accepted_as_candidate"] = (
        mode == "joint_video_action"
    )
    return stages


def build_window_index(
    manifest: Path,
    output_dir: Path,
    *,
    num_frames: int = 81,
    stride: int = 40,
    action_video_freq_ratio: int = 4,
    expanded: bool = False,
    minimum_unique_video_frames: int = 8,
) -> dict[str, Any]:
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if stride < 1:
        raise ValueError("stride must be positive")
    if minimum_unique_video_frames < 2:
        raise ValueError("minimum_unique_video_frames must be at least 2")
    if (num_frames - 1) % action_video_freq_ratio != 0:
        raise ValueError("num_frames - 1 must be divisible by action_video_freq_ratio")
    if ((num_frames - 1) // action_video_freq_ratio) % 4 != 0:
        raise ValueError("video transitions must be divisible by 4 for FastWAM tokenization")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_windows = 0
    action_windows = 0
    video_windows = 0
    filtered_video_windows = 0
    downgraded_action_windows = 0
    low_coverage_episode_count = 0
    record_count = 0
    episode_rows = 0
    with AtomicJsonlWriter(output_dir / "windows.jsonl") as writer:
        for episode in iter_jsonl(manifest):
            length = episode.get("num_frames")
            quality = episode.get("quality") or {}
            admission = episode.get("training_admission") or {}
            if length is None or int(length) < num_frames or quality.get("tier") == "C":
                continue
            video_eligible = bool(quality.get("video_eligible"))
            episode_action_eligible = bool(quality.get("action_eligible"))
            admission_mode = str(
                admission.get("mode")
                or ("joint_video_action" if episode_action_eligible else "video_only")
            )
            if not video_eligible or admission_mode == "reject":
                continue

            count = 1 + (int(length) - num_frames) // stride
            all_starts = list(range(0, count * stride, stride))
            unique_video_frames = _estimated_unique_video_frames(
                episode,
                num_frames=num_frames,
                video_ratio=action_video_freq_ratio,
            )
            if (
                unique_video_frames is not None
                and unique_video_frames < minimum_unique_video_frames
            ):
                low_coverage_episode_count += 1
                filtered_video_windows += len(all_starts)
                continue

            bad_intervals = episode.get("bad_intervals") or []
            blocked_video = blocked_window_starts(
                all_starts,
                window_size=num_frames,
                intervals=bad_intervals,
                domains={"video", "temporal"},
            )
            video_starts = [start for start in all_starts if start not in blocked_video]
            filtered_video_windows += len(blocked_video)
            if not video_starts:
                continue

            if episode_action_eligible and admission_mode == "joint_video_action":
                blocked_action = blocked_window_starts(
                    video_starts,
                    window_size=num_frames,
                    intervals=bad_intervals,
                    domains={"video", "temporal", "state", "action"},
                )
                joint_starts = [
                    start for start in video_starts if start not in blocked_action
                ]
                video_only_starts = [
                    start for start in video_starts if start in blocked_action
                ]
                downgraded_action_windows += len(video_only_starts)
            else:
                joint_starts = []
                video_only_starts = video_starts

            common = {
                "schema_version": "fastwam-window-index-v2",
                "global_episode_id": episode["global_episode_id"],
                "dataset": episode["dataset"],
                "release": episode.get("release"),
                "source_episode_id": episode.get("source_episode_id"),
                "lineage_id": episode.get("lineage_id"),
                "split_group_id": episode.get("split_group_id"),
                "embodiment": episode.get("embodiment"),
                "robot_type": episode.get("robot_type"),
                "task_namespace": episode.get("task_namespace"),
                "tasks": episode.get("tasks") or [],
                "target_fps": episode.get("fps"),
                "num_frames": num_frames,
                "action_horizon": num_frames - 1,
                "video_sample_offsets": list(
                    range(0, num_frames, action_video_freq_ratio)
                ),
                "video_eligible": video_eligible,
                "training_admission": admission,
                "canonical_parquet": episode.get("canonical_parquet"),
                "state_valid_slots": episode.get("state_valid_slots") or [],
                "action_valid_slots": episode.get("action_valid_slots") or [],
                "normalization_domain": episode.get("normalization_domain"),
                "cameras": episode.get("cameras") or [],
                "videos": episode.get("videos") or {},
                "source_uri": episode.get("source_uri"),
                "references": episode.get("references") or {},
                "quality": quality,
                "audit_summary": episode.get("audit_summary") or {},
                "bad_intervals": bad_intervals,
                "estimated_unique_video_frames": unique_video_frames,
            }

            emitted_for_episode = 0
            for mode, starts in (
                ("joint_video_action", joint_starts),
                ("video_only", video_only_starts),
            ):
                if not starts:
                    continue
                row_common = {
                    **common,
                    "action_eligible": mode == "joint_video_action",
                    "training_mode": mode,
                    "stage_admission": _window_stage_admission(
                        episode,
                        mode=mode,
                        unique_video_frames=unique_video_frames,
                    ),
                }
                if expanded:
                    for start in starts:
                        writer.write(
                            {
                                **row_common,
                                "window_id": stable_id(
                                    episode["global_episode_id"], start, num_frames, mode
                                ),
                                "start": start,
                                "state_stop_exclusive": start + num_frames,
                                "action_stop_exclusive": start + num_frames - 1,
                            }
                        )
                        record_count += 1
                        emitted_for_episode += 1
                else:
                    for valid_range in compact_start_ranges(starts, stride=stride):
                        writer.write({**row_common, "valid_starts": valid_range})
                        record_count += 1
                        emitted_for_episode += 1
            if emitted_for_episode:
                episode_rows += 1
            total_windows += len(video_starts)
            video_windows += len(video_starts)
            action_windows += len(joint_starts)

    summary = {
        "schema_version": "fastwam-window-index-v2",
        "input_manifest": str(manifest),
        "episode_rows": episode_rows,
        "expanded": expanded,
        "record_count": record_count,
        "window_count": total_windows,
        "video_window_count": video_windows,
        "action_window_count": action_windows,
        "interval_filtered_video_window_count": filtered_video_windows,
        "interval_downgraded_action_window_count": downgraded_action_windows,
        "low_unique_frame_coverage_episode_count": low_coverage_episode_count,
        "minimum_unique_video_frames": minimum_unique_video_frames,
        "num_frames": num_frames,
        "action_horizon": num_frames - 1,
        "action_video_freq_ratio": action_video_freq_ratio,
        "video_points": (num_frames - 1) // action_video_freq_ratio + 1,
        "stride": stride,
    }
    write_json(output_dir / "windows_summary.json", summary)
    return summary
