from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from targets.common import (
    LeRobotV2Dataset,
    StreamingStats,
    TargetPreparationError,
    atomic_overlay,
    compute_column_stats,
    file_sha256,
    finite_json_numbers,
    iter_jsonl,
    load_target_profile,
    read_json,
    stable_sha256,
    write_json,
    write_jsonl,
)


TARGET_SCHEMA_VERSION = "lingbot-va-target-v1"
MODEL_ACTION_DIM = 30


def _mapping(profile: dict[str, Any], source_width: int) -> tuple[list[int], list[int]]:
    source_indices = [int(value) for value in profile.get("source_action_indices") or []]
    model_channels = [int(value) for value in profile.get("model_action_channels") or []]
    if not source_indices:
        source_indices = list(range(source_width))
    if len(source_indices) != len(model_channels) or not model_channels:
        raise TargetPreparationError(
            "source_action_indices and model_action_channels must be non-empty and have equal length"
        )
    if len(set(source_indices)) != len(source_indices):
        raise TargetPreparationError("source_action_indices contains duplicates")
    if len(set(model_channels)) != len(model_channels):
        raise TargetPreparationError("model_action_channels contains duplicates")
    if min(source_indices) < 0 or max(source_indices) >= source_width:
        raise TargetPreparationError(
            f"source action mapping {source_indices} is outside width {source_width}"
        )
    if min(model_channels) < 0 or max(model_channels) >= MODEL_ACTION_DIM:
        raise TargetPreparationError("LingBot-VA model channels must be in [0, 30)")
    return source_indices, model_channels


def _inverse_channels(model_channels: list[int]) -> list[int]:
    missing_index = len(model_channels)
    inverse = [missing_index] * MODEL_ACTION_DIM
    for compact_index, model_channel in enumerate(model_channels):
        inverse[model_channel] = compact_index
    return inverse


def _episode_text(episode: dict[str, Any], fallback: str) -> str:
    tasks = episode.get("tasks") or []
    if not isinstance(tasks, list):
        tasks = [tasks]
    for task in tasks:
        text = " ".join(str(task).split())
        if text:
            return text
    return " ".join(fallback.split())


def _segments(episode: dict[str, Any], length: int, fallback_text: str) -> list[dict[str, Any]]:
    existing = episode.get("action_config")
    if existing is None:
        text = _episode_text(episode, fallback_text)
        if not text:
            raise TargetPreparationError(
                f"episode {episode.get('episode_index')} has no task/action text"
            )
        return [{"start_frame": 0, "end_frame": length, "action_text": text}]
    if not isinstance(existing, list) or not existing:
        raise TargetPreparationError("action_config must be a non-empty list")
    result: list[dict[str, Any]] = []
    previous_end = 0
    for item in existing:
        if not isinstance(item, dict):
            raise TargetPreparationError("action_config entries must be objects")
        start = int(item.get("start_frame", -1))
        end = int(item.get("end_frame", -1))
        text = " ".join(str(item.get("action_text") or "").split())
        if start < previous_end or start < 0 or end <= start or end > length or not text:
            raise TargetPreparationError(
                f"invalid action_config interval [{start}, {end}) for episode length {length}"
            )
        result.append({"start_frame": start, "end_frame": end, "action_text": text})
        previous_end = end
    return result


def _sample_frame_ids(start: int, end: int, source_fps: float, target_fps: float) -> list[int]:
    if source_fps <= 0 or target_fps <= 0:
        raise TargetPreparationError("source and latent target FPS must be positive")
    stride = source_fps / target_fps
    rounded_stride = int(round(stride))
    if rounded_stride < 1 or not np.isclose(stride, rounded_stride, atol=1e-6):
        raise TargetPreparationError(
            "old LingBot-VA requires an integer source-to-latent frame stride; "
            f"got source_fps={source_fps:g}, target_fps={target_fps:g}"
        )
    frame_ids = list(range(start, end, rounded_stride))
    # Wan's causal VAE emits one latent from the first frame and then one per
    # complete four-frame block. LingBot-VA derives its action length with the
    # same floor rule, so a trailing partial block must not be encoded.
    usable_count = 1 + ((len(frame_ids) - 1) // 4) * 4
    return frame_ids[:usable_count]


def _materialize_compact_actions(
    source: LeRobotV2Dataset,
    staging: Path,
    *,
    action_column: str,
    source_indices: list[int],
) -> None:
    for episode_index in source.episode_indices:
        source_path = source.data_path(episode_index)
        table = pq.read_table(source_path)
        values = source.read_column(episode_index, action_column)[:, source_indices]
        action_array = pa.array(
            values.astype(np.float32).tolist(),
            type=pa.list_(pa.float32(), len(source_indices)),
        )
        column_index = table.schema.get_field_index(action_column)
        if column_index < 0:
            raise TargetPreparationError(f"missing action column {action_column!r}")
        table = table.set_column(column_index, action_column, action_array)
        # Old LingBot-VA pins a datasets release that cannot deserialize the
        # newer Hugging Face ``List`` feature stored in Arrow schema metadata.
        # The physical Arrow types are sufficient for inference and preserve
        # every column value, so omit only that optional producer metadata.
        table = table.replace_schema_metadata(None)
        destination = staging / source.relative_data_path(episode_index)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")


def _mapped_norm_stats(stats: dict[str, Any], model_channels: list[int]) -> dict[str, Any]:
    q01 = [0.0] * MODEL_ACTION_DIM
    q99 = [0.0] * MODEL_ACTION_DIM
    for compact_index, model_channel in enumerate(model_channels):
        q01[model_channel] = float(stats["q01"][compact_index])
        q99[model_channel] = float(stats["q99"][compact_index])
    return {"q01": q01, "q99": q99}


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        axis=-1,
    )


def _relative_pose(values: np.ndarray, start: int, end: int) -> None:
    if end - start != 7:
        raise TargetPreparationError("relative pose groups must contain xyz + xyzw (7D)")
    pose = values[:, start:end]
    quaternion = pose[:, 3:7]
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise TargetPreparationError("relative pose quaternion has near-zero norm")
    quaternion = quaternion / norms
    first_inverse = np.broadcast_to(
        quaternion[0] * np.array([-1.0, -1.0, -1.0, 1.0]), quaternion.shape
    )
    pose[:, :3] -= pose[0:1, :3]
    pose[:, 3:7] = _quaternion_multiply(first_inverse, quaternion)


def _lingbot_action_stats(
    source: LeRobotV2Dataset,
    *,
    action_column: str,
    source_indices: list[int],
    train_episode_indices: list[int],
    episodes: dict[int, dict[str, Any]],
    profile: dict[str, Any],
    max_quantile_rows: int,
) -> dict[str, Any]:
    relative_groups = [
        (int(bounds[0]), int(bounds[1]))
        for bounds in profile.get("relative_pose_groups") or []
    ]
    if not relative_groups:
        return compute_column_stats(
            source,
            action_column,
            indices=train_episode_indices,
            source_indices=source_indices,
            max_quantile_rows=max_quantile_rows,
        )
    accumulator = StreamingStats(
        len(source_indices), max_quantile_rows=max_quantile_rows
    )
    for episode_index in train_episode_indices:
        source_values = source.read_column(episode_index, action_column)[:, source_indices]
        for segment in episodes[episode_index]["action_config"]:
            values = source_values[
                int(segment["start_frame"]) : int(segment["end_frame"])
            ].copy()
            for start, end in relative_groups:
                _relative_pose(values, start, end)
            accumulator.update(values)
    return accumulator.finish()


def prepare_lingbot_va_target(
    source_root: Path,
    output_root: Path,
    *,
    profile_document: dict[str, Any],
    profile: dict[str, Any],
    profile_name: str,
    link_mode: str = "symlink",
    verify_files: bool = False,
    train_episode_indices: list[int] | None = None,
    max_quantile_rows: int = 1_000_000,
) -> dict[str, Any]:
    source = LeRobotV2Dataset(source_root)
    action_column = str(profile.get("action_column") or "action")
    source_width = source.feature_width(action_column)
    if source_width is None:
        source_width = source.read_column(source.episode_indices[0], action_column).shape[1]
    source_indices, model_channels = _mapping(profile, source_width)
    camera_keys = [str(value) for value in profile.get("camera_keys") or []]
    source.validate_cameras(camera_keys, verify_files=verify_files)
    expected_cameras = int(profile.get("num_views") or len(camera_keys))
    if len(camera_keys) != expected_cameras:
        raise TargetPreparationError(
            f"profile expects {expected_cameras} camera views, got {len(camera_keys)}"
        )
    selected_train = source.episode_indices if train_episode_indices is None else sorted(set(train_episode_indices))
    unknown_train = sorted(set(selected_train) - set(source.episode_indices))
    if unknown_train or not selected_train:
        raise TargetPreparationError(
            f"invalid train episode indices: {unknown_train or selected_train}"
        )
    compact_identity = source_indices == list(range(source_width))
    source_fps = float(source.info.get("fps") or 0.0)
    latent_fps = float(profile.get("latent_fps") or min(10.0, source_fps))
    frame_stride = source_fps / latent_fps
    rounded_frame_stride = int(round(frame_stride))
    if not np.isclose(frame_stride, rounded_frame_stride, atol=1e-6):
        raise TargetPreparationError(
            "old LingBot-VA requires an integer source-to-latent frame stride; "
            f"got source_fps={source_fps:g}, target_fps={latent_fps:g}"
        )
    expected_action_per_frame = rounded_frame_stride * 4
    configured_action_per_frame = int(
        profile.get("action_per_frame") or expected_action_per_frame
    )
    if configured_action_per_frame != expected_action_per_frame:
        raise TargetPreparationError(
            "action_per_frame must match source FPS, latent FPS, and Wan VAE stride: "
            f"expected {expected_action_per_frame}, got {configured_action_per_frame}"
        )
    fallback_text = str(profile.get("fallback_action_text") or "")
    prepared_episodes: list[dict[str, Any]] = []
    latent_jobs: list[dict[str, Any]] = []
    episode_segments = 0
    for episode_index in source.episode_indices:
        episode = deepcopy(source.episode(episode_index))
        length = source.episode_length(episode_index)
        segments = _segments(episode, length, fallback_text)
        episode["length"] = length
        episode["action_config"] = segments
        prepared_episodes.append(episode)
        episode_segments += len(segments)
        chunk = episode_index // source.chunks_size
        for segment in segments:
            frame_ids = _sample_frame_ids(
                segment["start_frame"], segment["end_frame"], source_fps, latent_fps
            )
            for camera_index, camera_key in enumerate(camera_keys):
                target_height = int(profile.get("height") or 256)
                target_width = int(profile.get("width") or 256)
                if profile.get("env_type") == "robotwin_tshape" and camera_index > 0:
                    target_height //= 2
                    target_width //= 2
                relative_output = (
                    Path("latents")
                    / f"chunk-{chunk:03d}"
                    / camera_key
                    / (
                        f"episode_{episode_index:06d}_"
                        f"{segment['start_frame']}_{segment['end_frame']}.pth"
                    )
                )
                latent_jobs.append(
                    {
                        "schema_version": "lingbot-va-latent-job-v1",
                        "episode_index": episode_index,
                        "camera_key": camera_key,
                        "source_video": str(source.video_path(episode_index, camera_key)),
                        "source_video_relative": str(
                            source.video_path(episode_index, camera_key).relative_to(source.root)
                        ),
                        "start_frame": segment["start_frame"],
                        "end_frame": segment["end_frame"],
                        "frame_ids": frame_ids,
                        "source_fps": source_fps,
                        "target_fps": latent_fps,
                        "target_height": target_height,
                        "target_width": target_width,
                        "vae_temporal_stride": 4,
                        "text": segment["action_text"],
                        "output": str(relative_output),
                    }
                )

    prepared_by_index = {
        int(episode["episode_index"]): episode for episode in prepared_episodes
    }
    action_stats = _lingbot_action_stats(
        source,
        action_column=action_column,
        source_indices=source_indices,
        train_episode_indices=selected_train,
        episodes=prepared_by_index,
        profile=profile,
        max_quantile_rows=max_quantile_rows,
    )

    model_profile = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "action_dim": MODEL_ACTION_DIM,
        "compact_action_width": len(source_indices),
        "source_action_indices": source_indices,
        "used_action_channel_ids": model_channels,
        "inverse_used_action_channel_ids": _inverse_channels(model_channels),
        "obs_cam_keys": camera_keys,
        "height": int(profile.get("height") or 256),
        "width": int(profile.get("width") or 256),
        "frame_chunk_size": int(profile.get("frame_chunk_size") or 4),
        "action_per_frame": configured_action_per_frame,
        "env_type": str(profile.get("env_type") or "none"),
        "action_norm_method": "quantiles",
        "norm_stat": _mapped_norm_stats(action_stats, model_channels),
    }
    if not finite_json_numbers(model_profile):
        raise TargetPreparationError("generated LingBot-VA model profile is not finite")
    info = deepcopy(source.info)
    if not compact_identity:
        action_feature = deepcopy(info["features"][action_column])
        action_feature["shape"] = [len(source_indices)]
        action_feature["names"] = [f"lingbot_va_channel_{value}" for value in model_channels]
        info["features"][action_column] = action_feature

    with atomic_overlay(
        source.root,
        output_root,
        link_mode=link_mode,
        include_data=compact_identity,
        include_videos=True,
    ) as staging:
        if not compact_identity:
            _materialize_compact_actions(
                source,
                staging,
                action_column=action_column,
                source_indices=source_indices,
            )
        write_json(staging / "meta" / "info.json", info)
        write_jsonl(staging / "meta" / "episodes.jsonl", prepared_episodes)
        write_json(staging / "meta" / "lingbot_va_model_profile.json", model_profile)
        write_jsonl(staging / "meta" / "lingbot_va_latent_jobs.jsonl", latent_jobs)
        receipt = {
            "schema_version": TARGET_SCHEMA_VERSION,
            "status": "prepared_requires_vae_latents",
            "source_root": str(source.root),
            "source_info_sha256": file_sha256(source.info_path),
            "source_episodes_sha256": file_sha256(source.episodes_path),
            "target": "lingbot_va",
            "profile": profile_name,
            "profile_sha256": stable_sha256(profile),
            "upstream_repository": profile_document.get("upstream_repository"),
            "upstream_revision": profile_document.get("upstream_revision"),
            "link_mode": link_mode,
            "episode_count": len(prepared_episodes),
            "segment_count": episode_segments,
            "latent_job_count": len(latent_jobs),
            "train_stats_episode_indices": selected_train,
            "action_column": action_column,
            "action_stats": action_stats,
            "model_profile": model_profile,
        }
        write_json(staging / "meta" / "lingbot_va_target_receipt.json", receipt)

    return validate_lingbot_va_target(output_root, require_latents=False)


def validate_lingbot_va_target(
    root: Path, *, require_latents: bool = False, verify_files: bool = False
) -> dict[str, Any]:
    dataset = LeRobotV2Dataset(root)
    receipt_path = dataset.meta_root / "lingbot_va_target_receipt.json"
    profile_path = dataset.meta_root / "lingbot_va_model_profile.json"
    jobs_path = dataset.meta_root / "lingbot_va_latent_jobs.jsonl"
    failures: list[str] = []
    if not receipt_path.is_file() or not profile_path.is_file() or not jobs_path.is_file():
        failures.append("target_metadata_missing")
        return {"target": "lingbot_va", "valid": False, "failures": failures}
    receipt = read_json(receipt_path)
    model_profile = read_json(profile_path)
    jobs = list(iter_jsonl(jobs_path))
    action_column = str(receipt.get("action_column") or "action")
    expected_width = int(model_profile.get("compact_action_width") or 0)
    for episode_index in dataset.episode_indices:
        episode = dataset.episode(episode_index)
        try:
            _segments(episode, dataset.episode_length(episode_index), "")
            if dataset.read_column(episode_index, action_column).shape[1] != expected_width:
                failures.append(f"episode_{episode_index}_action_width")
        except TargetPreparationError as exc:
            failures.append(f"episode_{episode_index}:{exc}")
    camera_keys = [str(value) for value in model_profile.get("obs_cam_keys") or []]
    try:
        dataset.validate_cameras(camera_keys, verify_files=verify_files)
    except TargetPreparationError as exc:
        failures.append(str(exc))
    missing_latents = [
        job["output"] for job in jobs if not (dataset.root / str(job["output"])).is_file()
    ]
    if require_latents and missing_latents:
        failures.append(f"missing_latents:{len(missing_latents)}")
    return {
        "target": "lingbot_va",
        "schema_version": receipt.get("schema_version"),
        "valid": not failures,
        "ready_for_training": not failures and not missing_latents,
        "episode_count": len(dataset.episodes),
        "latent_job_count": len(jobs),
        "missing_latent_count": len(missing_latents),
        "failures": failures,
    }


def _read_indices(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise TargetPreparationError("train episode file is empty")
    if text.startswith("["):
        values = json.loads(text)
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    return [int(value) for value in values]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare LeRobot v2 data for old Robbyant/LingBot-VA")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--profiles", type=Path, required=True)
    prepare.add_argument("--profile", required=True)
    prepare.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    prepare.add_argument("--verify-files", action="store_true")
    prepare.add_argument("--train-episodes-file", type=Path)
    prepare.add_argument("--max-quantile-rows", type=int, default=1_000_000)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--require-latents", action="store_true")
    validate.add_argument("--verify-files", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        document, profile = load_target_profile(args.profiles, "lingbot_va", args.profile)
        result = prepare_lingbot_va_target(
            args.source_root,
            args.output_root,
            profile_document=document,
            profile=profile,
            profile_name=args.profile,
            link_mode=args.link_mode,
            verify_files=args.verify_files,
            train_episode_indices=_read_indices(args.train_episodes_file),
            max_quantile_rows=args.max_quantile_rows,
        )
    else:
        result = validate_lingbot_va_target(
            args.root,
            require_latents=args.require_latents,
            verify_files=args.verify_files,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
