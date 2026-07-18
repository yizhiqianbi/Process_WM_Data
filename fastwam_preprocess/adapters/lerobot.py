from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..camera import normalize_camera_role
from ..canonical import infer_canonical_mapping
from ..schema import CameraRecord
from ..utils import iter_jsonl, read_json
from .base import BaseAdapter


def _shape(feature: dict[str, Any]) -> list[int] | None:
    value = feature.get("shape")
    if not isinstance(value, list):
        return None
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            return None
    return result


def feature_names(feature: dict[str, Any]) -> list[str] | None:
    """Normalize the name layouts emitted by LeRobot v2.x metadata writers."""
    value = feature.get("names")
    if isinstance(value, dict):
        flattened: list[str] = []
        for names in value.values():
            if isinstance(names, list):
                flattened.extend(str(name) for name in names)
            elif names is not None:
                flattened.append(str(names))
        return flattened or None
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return [str(name) for name in value[0]]
        if all(not isinstance(name, (dict, list)) for name in value):
            return [str(name) for name in value]
    return None


def _feature_schema(features: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, feature in features.items():
        if not key.startswith(prefixes) or not isinstance(feature, dict):
            continue
        result[key] = {
            "dtype": feature.get("dtype"),
            "shape": _shape(feature),
            "names": feature_names(feature),
        }
    return result


def camera_records(features: dict[str, Any]) -> list[CameraRecord]:
    records: list[CameraRecord] = []
    for key, feature in features.items():
        if not isinstance(feature, dict):
            continue
        info: dict[str, Any] = {}
        for info_key in ("video_info", "info"):
            candidate = feature.get(info_key)
            if isinstance(candidate, dict):
                info.update(candidate)
        dtype = str(feature.get("dtype", "")).lower()
        if dtype != "video" and "video.codec" not in info:
            continue
        shape = _shape(feature) or []
        names = feature_names(feature) or []

        def named_dimension(name: str, fallback_index: int) -> Any:
            try:
                return shape[names.index(name)]
            except (ValueError, IndexError):
                return shape[fallback_index] if len(shape) > fallback_index else None

        height = info.get("video.height", named_dimension("height", 0))
        width = info.get("video.width", named_dimension("width", 1))
        records.append(
            CameraRecord(
                source_key=key,
                role=normalize_camera_role(key),
                width=int(width) if width is not None else None,
                height=int(height) if height is not None else None,
                fps=float(info["video.fps"]) if info.get("video.fps") is not None else None,
                codec=info.get("video.codec"),
                has_depth=bool(info.get("video.is_depth_map", False)),
            )
        )
    return records


def feature_schemas(features: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _feature_schema(features, ("observation.state", "states."))
    # Released datasets sometimes retain teleoperator `master_actions` beside
    # the robot command stream. Prefer the documented `action(s)` contract and
    # use master actions only when no native action feature exists.
    action = _feature_schema(features, ("action", "actions."))
    if not action:
        action = _feature_schema(features, ("master_actions.",))
    return state, action


def format_lerobot_path(
    template: str, episode_index: int, chunk_size: int, **extra: Any
) -> str:
    values = {
        "episode_index": episode_index,
        "episode_chunk": episode_index // max(1, chunk_size),
        **extra,
    }
    try:
        return template.format(**values)
    except (KeyError, ValueError):
        return template


def scan_lerobot_repo(
    adapter: BaseAdapter,
    repo_root: Path,
    *,
    release: str,
    task_namespace: str | None = None,
    lineage_factory: Callable[[int], str | None] | None = None,
    variants: list[dict[str, Any]] | None = None,
) -> None:
    info_path = repo_root / "meta" / "info.json"
    episodes_path = repo_root / "meta" / "episodes.jsonl"
    if not info_path.is_file():
        adapter.add_artifact(
            path=repo_root, kind="lerobot_repo", complete=False, status="missing_meta_info"
        )
        return
    try:
        info = read_json(info_path)
    except (OSError, ValueError) as exc:
        adapter.add_artifact(
            path=info_path,
            kind="lerobot_info",
            complete=False,
            status="invalid_json",
            metadata={"error": str(exc)},
        )
        return

    features = info.get("features")
    features = features if isinstance(features, dict) else {}
    cameras = camera_records(features)
    state_schema, action_schema = feature_schemas(features)
    canonical_mapping = {
        "state": infer_canonical_mapping(state_schema, kind="state"),
        "action": infer_canonical_mapping(action_schema, kind="action"),
    }
    variant_specs = variants or [
        {
            "id": None,
            "robot_type": None,
            "cameras": cameras,
            "state_schema": state_schema,
            "action_schema": action_schema,
            "canonical_mapping": canonical_mapping,
        }
    ]
    calibration_keys = [
        key for key in features if "intrinsics" in key.lower() or "extrinsics" in key.lower()
    ]
    has_calibration = bool(calibration_keys)
    has_depth = any(camera.has_depth for camera in cameras)
    try:
        fps = float(info["fps"]) if info.get("fps") is not None else None
    except (TypeError, ValueError):
        fps = None

    robot_type = str(info.get("robot_type") or repo_root.name.split("_", 1)[0] or "unknown")
    namespace = task_namespace or repo_root.name
    data_template = str(
        info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    )
    video_template = str(
        info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
    )
    chunk_size = int(info.get("chunks_size") or 1000)

    if not episodes_path.is_file():
        adapter.add_artifact(
            path=repo_root,
            kind="lerobot_repo",
            complete=False,
            status="missing_episodes_jsonl",
            metadata={"robot_type": robot_type},
        )
        return

    adapter.add_artifact(
        path=repo_root,
        kind="lerobot_repo",
        complete=True,
        status="metadata_ready",
        metadata={
            "robot_type": robot_type,
            "reported_episodes": info.get("total_episodes"),
            "reported_frames": info.get("total_frames"),
            "camera_count": len(cameras),
            "variant_count": len(variant_specs),
            "has_calibration": has_calibration,
        },
    )

    try:
        for row in iter_jsonl(episodes_path):
            if adapter.at_limit():
                break
            episode_index = int(row.get("episode_index", adapter.episode_count))
            try:
                num_frames = int(row["length"]) if row.get("length") is not None else None
            except (TypeError, ValueError):
                num_frames = None
            tasks_value = row.get("tasks")
            if isinstance(tasks_value, list):
                tasks = [str(task) for task in tasks_value]
            elif tasks_value:
                tasks = [str(tasks_value)]
            else:
                tasks = [namespace]

            data_rel = format_lerobot_path(data_template, episode_index, chunk_size)
            data_path = repo_root / data_rel
            for variant in variant_specs:
                if adapter.at_limit():
                    break
                variant_id = variant.get("id")
                variant_cameras = list(variant.get("cameras") or [])
                variant_state = dict(variant.get("state_schema") or {})
                variant_action = dict(variant.get("action_schema") or {})
                variant_mapping = dict(variant.get("canonical_mapping") or {})
                variant_robot = str(variant.get("robot_type") or robot_type)
                video_refs: dict[str, str] = {}
                missing: list[str] = []
                if adapter.options.verify_files and not data_path.is_file():
                    missing.append(data_rel)

                episode_cameras: list[CameraRecord] = []
                for camera in variant_cameras:
                    video_rel = format_lerobot_path(
                        video_template,
                        episode_index,
                        chunk_size,
                        video_key=camera.source_key,
                    )
                    video_path = repo_root / video_rel
                    video_refs[camera.source_key] = str(video_path)
                    if adapter.options.verify_files and not video_path.is_file():
                        missing.append(video_rel)
                    camera_copy = CameraRecord(**camera.to_dict())
                    camera_copy.source_uri = str(video_path)
                    episode_cameras.append(camera_copy)

                passed = [
                    "metadata_schema",
                    "episode_boundary",
                    *list(variant.get("passed_checks") or []),
                ]
                pending = ["temporal", "signal", "visual"]
                pending.extend(variant.get("pending_checks") or [])
                if has_calibration:
                    passed.append("calibration_metadata")
                else:
                    pending.append("kinematic")
                if adapter.options.verify_files and not missing:
                    passed.append("referenced_files_exist")
                elif not adapter.options.verify_files:
                    pending.append("file_integrity")

                source_episode_id = f"{repo_root.name}:{episode_index:06d}"
                if variant_id:
                    source_episode_id = f"{source_episode_id}:{variant_id}"
                variant_metadata = dict(variant.get("metadata") or {})
                adapter.add_episode(
                    source_episode_id=source_episode_id,
                    source_uri=str(data_path),
                    embodiment=str(variant.get("embodiment") or variant_robot),
                    robot_type=variant_robot,
                    task_namespace=namespace,
                    tasks=tasks,
                    lineage_id=lineage_factory(episode_index) if lineage_factory else None,
                    num_frames=num_frames,
                    fps=float(variant.get("fps") or fps) if (variant.get("fps") or fps) else None,
                    cameras=episode_cameras,
                    state_schema=variant_state,
                    action_schema=variant_action,
                    has_calibration=has_calibration,
                    has_depth=any(camera.has_depth for camera in episode_cameras),
                    complete=not missing,
                    action_verified=False,
                    passed_checks=passed,
                    pending_checks=pending,
                    warnings=list(variant.get("warnings") or []),
                    failures=["missing_referenced_files"] if missing else [],
                    references={
                        "repo_root": str(repo_root),
                        "data": str(data_path),
                        "videos": video_refs,
                        "missing": missing[:20],
                    },
                    metadata={
                        "episode_index": episode_index,
                        "variant_id": variant_id,
                        "codebase_version": info.get("codebase_version"),
                        "calibration_keys": calibration_keys,
                        "source_info_path": str(info_path),
                        "source_episode_metadata": {
                            key: row[key]
                            for key in ("action_config", "success")
                            if key in row
                        },
                        "canonical_mapping": variant_mapping,
                        **variant_metadata,
                    },
                    release=release,
                )
    except (OSError, ValueError) as exc:
        adapter.blockers.append(f"episodes_parse_failed:{episodes_path}:{exc}")
