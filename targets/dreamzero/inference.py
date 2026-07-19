"""Pure protocol and rendering helpers for DreamZero rollout evaluation."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from targets.video_pair import put_text


WAN_TEMPORAL_STRIDE = 4
CAMERA_KEYS = ("video.left_eye", "video.right_eye", "video.right_wrist")
SOURCE_CAMERA_KEYS = {
    "video.left_eye": "observation.images.left_eye",
    "video.right_eye": "observation.images.right_eye",
    "video.right_wrist": "observation.images.right_wrist",
}
PHYSICAL_CAMERA_LABELS = {
    "video.left_eye": "GLOBAL / HEAD (left_eye)",
    "video.right_eye": "LEFT WRIST (right_eye)",
    "video.right_wrist": "RIGHT WRIST (right_wrist)",
}


def decoded_frame_count(num_chunks: int, num_frame_per_block: int = 2) -> int:
    if num_chunks <= 0 or num_frame_per_block <= 0:
        raise ValueError("chunk counts and frame block sizes must be positive")
    latent_frames = 1 + num_chunks * num_frame_per_block
    return 1 + WAN_TEMPORAL_STRIDE * (latent_frames - 1)


def chunk_segments(num_chunks: int, cache_window_chunks: int) -> list[tuple[int, int]]:
    """Return half-open chunk ranges for bounded causal-cache segments."""
    if num_chunks <= 0 or cache_window_chunks <= 0:
        raise ValueError("chunk counts and cache windows must be positive")
    return [
        (start, min(start + cache_window_chunks, num_chunks))
        for start in range(0, num_chunks, cache_window_chunks)
    ]


def decode_context_latent_bounds(
    chunk_index: int,
    cache_window_chunks: int,
    num_frame_per_block: int = 2,
) -> tuple[int, int]:
    """Return the GT latent slice visible while decoding one predicted chunk."""
    if chunk_index <= 0 or cache_window_chunks <= 0 or num_frame_per_block <= 0:
        raise ValueError(
            "chunk_index must be positive; cache window and block size must be positive"
        )
    segment_start = (chunk_index // cache_window_chunks) * cache_window_chunks
    return (
        segment_start * num_frame_per_block,
        1 + chunk_index * num_frame_per_block,
    )


def observation_frame_ids(
    chunk_index: int,
    *,
    start_frame: int,
    source_frames_per_chunk: int = 8,
    feedback_frames: int = 4,
) -> list[int]:
    if chunk_index < 0 or start_frame < 0:
        raise ValueError("chunk_index and start_frame must be nonnegative")
    if source_frames_per_chunk <= 0 or feedback_frames <= 0:
        raise ValueError("feedback cadence and history must be positive")
    if chunk_index == 0:
        return [start_frame]
    anchor = start_frame + chunk_index * source_frames_per_chunk
    first = anchor - feedback_frames + 1
    if first < start_frame:
        raise ValueError("feedback history precedes the rollout start")
    return list(range(first, anchor + 1))


def comparison_frame_ids(
    *, start_frame: int, num_chunks: int, num_frame_per_block: int = 2
) -> list[int]:
    return list(
        range(start_frame, start_frame + decoded_frame_count(num_chunks, num_frame_per_block))
    )


def select_episodes(
    episodes: list[dict[str, Any]],
    *,
    requested: list[int],
    num_cases: int,
    required_source_frames: int,
    start_frame: int,
) -> list[int]:
    by_id = {int(item["episode_index"]): item for item in episodes}
    if requested:
        missing = [value for value in requested if value not in by_id]
        if missing:
            raise ValueError(f"unknown episode IDs: {missing}")
        selected = requested
    else:
        eligible = [
            item
            for item in episodes
            if int(item["length"]) - start_frame >= required_source_frames
        ]
        eligible.sort(key=lambda item: (-int(item["length"]), int(item["episode_index"])))
        selected = [int(item["episode_index"]) for item in eligible[:num_cases]]
    if len(selected) < num_cases:
        raise ValueError(
            f"requested at least {num_cases} cases, but only selected {len(selected)}"
        )
    selected = selected[:num_cases]
    too_short = [
        value
        for value in selected
        if int(by_id[value]["length"]) - start_frame < required_source_frames
    ]
    if too_short:
        raise ValueError(
            "episodes are too short for the requested prediction horizon: "
            f"{too_short}; need {required_source_frames} source frames"
        )
    return selected


def split_composite(
    video: np.ndarray,
    *,
    single_height: int,
    single_width: int,
) -> dict[str, np.ndarray]:
    expected = (single_height * 2, single_width * 2, 3)
    if video.ndim != 4 or tuple(video.shape[1:]) != expected:
        raise ValueError(
            f"unexpected DreamZero composite shape {video.shape}; expected [F,{expected[0]},"
            f"{expected[1]},3]"
        )
    return {
        "video.left_eye": np.ascontiguousarray(
            video[:, :single_height, :single_width]
        ),
        "video.right_eye": np.ascontiguousarray(
            video[:, single_height:, :single_width]
        ),
        "video.right_wrist": np.ascontiguousarray(
            video[:, :single_height, single_width:]
        ),
    }


def flatten_action_output(action: Any, *, executed_steps: int) -> np.ndarray:
    if hasattr(action, "__getstate__") and not isinstance(action, dict):
        state = action.__getstate__()
        if isinstance(state, dict):
            action = state
    joint = np.asarray(action["action.right_joint_position"], dtype=np.float32)
    gripper = np.asarray(action["action.right_gripper_position"], dtype=np.float32)
    joint = joint.reshape(-1, 7)
    gripper = gripper.reshape(-1, 1)
    if len(joint) < executed_steps or len(gripper) < executed_steps:
        raise ValueError(
            f"predicted action horizon is shorter than {executed_steps}: "
            f"joint={joint.shape}, gripper={gripper.shape}"
        )
    return np.concatenate((joint[:executed_steps], gripper[:executed_steps]), axis=1)


def render_all_views_pair(
    gt_views: dict[str, np.ndarray],
    pred_views: dict[str, np.ndarray],
    *,
    episode_index: int,
    fps: float,
) -> Iterable[np.ndarray]:
    frame_count, height, width, _ = gt_views[CAMERA_KEYS[0]].shape
    header = 46
    full_width = width * len(CAMERA_KEYS)
    for frame_index in range(frame_count):
        canvas = np.full((2 * (height + header), full_width, 3), 18, dtype=np.uint8)
        for camera_index, camera_key in enumerate(CAMERA_KEYS):
            left = camera_index * width
            right = left + width
            canvas[header : header + height, left:right] = gt_views[camera_key][frame_index]
            pred_top = height + 2 * header
            canvas[pred_top : pred_top + height, left:right] = pred_views[camera_key][frame_index]
            label = PHYSICAL_CAMERA_LABELS[camera_key]
            put_text(canvas, f"GT | {label}", (left + 8, 24), max_width=width - 16)
            put_text(
                canvas,
                f"DREAMZERO | {label}",
                (left + 8, height + header + 24),
                max_width=width - 16,
            )
            if camera_index:
                canvas[:, left - 1 : left + 1] = 230
        put_text(
            canvas,
            f"ep {episode_index:06d}  t={frame_index / fps:05.1f}s  GT OBS FEEDBACK",
            (8, 42),
            scale=0.34,
        )
        yield canvas


def render_single_view_pair(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    camera_key: str,
    episode_index: int,
    fps: float,
) -> Iterable[np.ndarray]:
    frame_count, height, width, _ = ground_truth.shape
    header = 46
    label = PHYSICAL_CAMERA_LABELS[camera_key]
    for frame_index in range(frame_count):
        canvas = np.full((height + header, width * 2, 3), 18, dtype=np.uint8)
        canvas[header:, :width] = ground_truth[frame_index]
        canvas[header:, width:] = prediction[frame_index]
        put_text(canvas, f"GT | {label}", (8, 24), max_width=width - 16)
        put_text(
            canvas,
            f"DREAMZERO | {label}",
            (width + 8, 24),
            max_width=width - 16,
        )
        put_text(
            canvas,
            f"ep {episode_index:06d}  {frame_index / fps:05.1f}s  GT OBS FEEDBACK",
            (8, 42),
            scale=0.34,
            max_width=width - 16,
        )
        yield canvas
