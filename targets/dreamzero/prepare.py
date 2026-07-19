from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from targets.common import (
    LeRobotV2Dataset,
    StreamingStats,
    TargetPreparationError,
    atomic_overlay,
    compute_column_stats,
    file_sha256,
    finite_json_numbers,
    load_target_profile,
    read_json,
    stable_sha256,
    validate_range_mapping,
    write_json,
)


TARGET_SCHEMA_VERSION = "dreamzero-target-v1"


def _modality_entry(original_key: str, start: int, end: int, dtype: str) -> dict[str, Any]:
    return {
        "original_key": original_key,
        "start": start,
        "end": end,
        "rotation_type": None,
        "absolute": True,
        "dtype": dtype,
        "range": None,
    }


def _build_modality(
    source: LeRobotV2Dataset, profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    existing_path = source.meta_root / "modality.json"
    if profile.get("use_existing_modality"):
        if not existing_path.is_file():
            raise TargetPreparationError("profile requires existing meta/modality.json")
        modality = read_json(existing_path)
        state_ranges = {
            str(key): (int(value["start"]), int(value["end"]))
            for key, value in (modality.get("state") or {}).items()
        }
        action_ranges = {
            str(key): (int(value["start"]), int(value["end"]))
            for key, value in (modality.get("action") or {}).items()
        }
        return modality, state_ranges, action_ranges

    state_column = str(profile.get("state_column") or "observation.state")
    action_column = str(profile.get("action_column") or "action")
    state_width = source.feature_width(state_column)
    action_width = source.feature_width(action_column)
    if state_width is None:
        state_width = source.read_column(source.episode_indices[0], state_column).shape[1]
    if action_width is None:
        action_width = source.read_column(source.episode_indices[0], action_column).shape[1]
    state_ranges = validate_range_mapping(
        profile.get("state_keys") or {}, width=state_width, label="state_keys"
    )
    action_ranges = validate_range_mapping(
        profile.get("action_keys") or {}, width=action_width, label="action_keys"
    )
    state_dtype = str((source.features.get(state_column) or {}).get("dtype") or "float32")
    action_dtype = str((source.features.get(action_column) or {}).get("dtype") or "float32")
    annotation = profile.get("annotation") or {}
    annotation_name = str(annotation.get("name") or "task")
    annotation_column = str(annotation.get("column") or "annotation.task")
    camera_keys = [str(value) for value in profile.get("camera_keys") or []]
    modality = {
        "state": {
            key: _modality_entry(state_column, start, end, state_dtype)
            for key, (start, end) in state_ranges.items()
        },
        "action": {
            key: _modality_entry(action_column, start, end, action_dtype)
            for key, (start, end) in action_ranges.items()
        },
        "video": {
            key.removeprefix("observation.images."): {"original_key": key}
            for key in camera_keys
        },
        "annotation": {
            annotation_name: {"original_key": annotation_column}
        },
    }
    return modality, state_ranges, action_ranges


def _episode_task(episode: dict[str, Any], fallback: str) -> str:
    tasks = episode.get("tasks") or []
    if not isinstance(tasks, list):
        tasks = [tasks]
    for value in tasks:
        text = " ".join(str(value).split())
        if text:
            return text
    text = " ".join(fallback.split())
    if not text:
        raise TargetPreparationError(
            f"episode {episode.get('episode_index')} has no language annotation"
        )
    return text


def _materialize_annotation_data(
    source: LeRobotV2Dataset,
    staging: Path,
    *,
    annotation_column: str,
    fallback_text: str,
) -> None:
    for episode_index in source.episode_indices:
        source_path = source.data_path(episode_index)
        table = pq.read_table(source_path)
        text = _episode_task(source.episode(episode_index), fallback_text)
        values = pa.array([text] * table.num_rows, type=pa.string())
        existing_index = table.schema.get_field_index(annotation_column)
        if existing_index >= 0:
            table = table.set_column(existing_index, annotation_column, values)
        else:
            table = table.append_column(annotation_column, values)
        destination = staging / source.relative_data_path(episode_index)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")


def _relative_stats(
    source: LeRobotV2Dataset,
    modality: dict[str, Any],
    relative_keys: list[str],
    *,
    action_horizon: int,
    max_quantile_rows: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in relative_keys:
        state_meta = (modality.get("state") or {}).get(key)
        action_meta = (modality.get("action") or {}).get(key)
        if not isinstance(state_meta, dict) or not isinstance(action_meta, dict):
            raise TargetPreparationError(
                f"relative action key {key!r} must exist in both state and action modalities"
            )
        state_start, state_end = int(state_meta["start"]), int(state_meta["end"])
        action_start, action_end = int(action_meta["start"]), int(action_meta["end"])
        width = state_end - state_start
        if action_end - action_start != width:
            raise TargetPreparationError(
                f"relative action key {key!r} state/action widths differ"
            )
        accumulator = StreamingStats(width, max_quantile_rows=max_quantile_rows)
        for episode_index in source.episode_indices:
            state = source.read_column(episode_index, str(state_meta["original_key"]))[
                :, state_start:state_end
            ]
            action = source.read_column(episode_index, str(action_meta["original_key"]))[
                :, action_start:action_end
            ]
            usable = len(state) - action_horizon
            if usable <= 0:
                continue
            reference = state[:usable]
            for offset in range(action_horizon):
                accumulator.update(action[offset : offset + usable] - reference)
        output[key] = accumulator.finish()
    return output


def _normalization_modes(modality: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in ("state", "action"):
        for key in (modality.get(group) or {}):
            result[f"{group}.{key}"] = "q99"
    return result


def _write_hydra_patch(
    path: Path,
    *,
    embodiment_tag: str,
    modality: dict[str, Any],
    relative_action_keys: list[str],
    fps: float,
    num_frames: int,
    action_horizon: int,
) -> None:
    prefixed = {
        group: [f"{group}.{key}" for key in (modality.get(group) or {})]
        for group in ("video", "state", "action", "annotation")
    }
    patch = {
        "schema_version": "dreamzero-hydra-patch-v1",
        "embodiment_tag": embodiment_tag,
        "fps": fps,
        "num_frames": num_frames,
        "action_horizon": action_horizon,
        "relative_action_keys": relative_action_keys,
        "modality_config": {
            "video": {
                "delta_indices": list(range(num_frames)),
                "eval_delta_indices": [0],
                "modality_keys": prefixed["video"],
            },
            "state": {"delta_indices": [0], "modality_keys": prefixed["state"]},
            "action": {
                "delta_indices": list(range(action_horizon)),
                "modality_keys": prefixed["action"],
            },
            "language": {
                "delta_indices": [0],
                "modality_keys": prefixed["annotation"],
            },
        },
        "normalization_modes": _normalization_modes(modality),
        "concat_order": {
            "video": prefixed["video"],
            "state": prefixed["state"],
            "action": prefixed["action"],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(patch, handle, allow_unicode=True, sort_keys=False)


def prepare_dreamzero_target(
    source_root: Path,
    output_root: Path,
    *,
    profile_document: dict[str, Any],
    profile: dict[str, Any],
    profile_name: str,
    link_mode: str = "symlink",
    verify_files: bool = False,
    max_quantile_rows: int = 1_000_000,
) -> dict[str, Any]:
    source = LeRobotV2Dataset(source_root)
    camera_keys = [str(value) for value in profile.get("camera_keys") or []]
    source.validate_cameras(camera_keys, verify_files=verify_files)
    modality, _, _ = _build_modality(source, profile)
    action_horizon = int(profile.get("action_horizon") or 24)
    num_frames = int(profile.get("num_frames") or 33)
    embodiment_tag = str(profile.get("embodiment_tag") or "xdof")
    relative_keys = [str(value) for value in profile.get("relative_action_keys") or []]
    state_columns = sorted(
        {str(value["original_key"]) for value in (modality.get("state") or {}).values()}
    )
    action_columns = sorted(
        {str(value["original_key"]) for value in (modality.get("action") or {}).values()}
    )
    preserve_stats = bool(profile.get("preserve_existing_stats"))
    stats_path = source.meta_root / "stats.json"
    if preserve_stats and stats_path.is_file():
        stats = read_json(stats_path)
    else:
        stats = {
            column: compute_column_stats(
                source, column, max_quantile_rows=max_quantile_rows
            )
            for column in [*state_columns, *action_columns]
        }
    relative_path = source.meta_root / "relative_stats_dreamzero.json"
    if preserve_stats and relative_path.is_file():
        relative_stats = read_json(relative_path)
    else:
        relative_stats = _relative_stats(
            source,
            modality,
            relative_keys,
            action_horizon=action_horizon,
            max_quantile_rows=max_quantile_rows,
        )
    if not finite_json_numbers(stats) or not finite_json_numbers(relative_stats):
        raise TargetPreparationError("generated DreamZero statistics are not finite")

    annotation = profile.get("annotation") or {}
    annotation_column = str(annotation.get("column") or "annotation.task")
    add_annotation = annotation_column not in source.features
    fallback_text = str(profile.get("fallback_action_text") or "")
    info = deepcopy(source.info)
    if add_annotation:
        info.setdefault("features", {})[annotation_column] = {
            "dtype": "string",
            "shape": [1],
            "names": None,
        }
    fps = float(profile.get("fps") or info.get("fps") or 0.0)
    if fps <= 0:
        raise TargetPreparationError("DreamZero target requires positive FPS")

    with atomic_overlay(
        source.root,
        output_root,
        link_mode=link_mode,
        include_data=not add_annotation,
        include_videos=True,
    ) as staging:
        if add_annotation:
            _materialize_annotation_data(
                source,
                staging,
                annotation_column=annotation_column,
                fallback_text=fallback_text,
            )
        write_json(staging / "meta" / "info.json", info)
        write_json(staging / "meta" / "modality.json", modality)
        write_json(
            staging / "meta" / "embodiment.json",
            {"robot_type": embodiment_tag, "embodiment_tag": embodiment_tag},
        )
        write_json(staging / "meta" / "stats.json", stats)
        write_json(
            staging / "meta" / "relative_stats_dreamzero.json", relative_stats
        )
        _write_hydra_patch(
            staging / "meta" / "dreamzero_hydra_patch.yaml",
            embodiment_tag=embodiment_tag,
            modality=modality,
            relative_action_keys=relative_keys,
            fps=fps,
            num_frames=num_frames,
            action_horizon=action_horizon,
        )
        receipt = {
            "schema_version": TARGET_SCHEMA_VERSION,
            "status": (
                "prepared_native_profile"
                if profile.get("upstream_profile_registered")
                else "prepared_requires_upstream_profile_registration"
            ),
            "source_root": str(source.root),
            "source_info_sha256": file_sha256(source.info_path),
            "source_episodes_sha256": file_sha256(source.episodes_path),
            "target": "dreamzero",
            "profile": profile_name,
            "profile_sha256": stable_sha256(profile),
            "upstream_repository": profile_document.get("upstream_repository"),
            "upstream_revision": profile_document.get("upstream_revision"),
            "link_mode": link_mode,
            "episode_count": len(source.episodes),
            "embodiment_tag": embodiment_tag,
            "camera_keys": camera_keys,
            "num_frames": num_frames,
            "action_horizon": action_horizon,
            "relative_action_keys": relative_keys,
            "upstream_profile_registered": bool(profile.get("upstream_profile_registered")),
        }
        write_json(staging / "meta" / "dreamzero_target_receipt.json", receipt)

    return validate_dreamzero_target(output_root, verify_files=verify_files)


def validate_dreamzero_target(root: Path, *, verify_files: bool = False) -> dict[str, Any]:
    dataset = LeRobotV2Dataset(root)
    required = {
        "modality": dataset.meta_root / "modality.json",
        "embodiment": dataset.meta_root / "embodiment.json",
        "stats": dataset.meta_root / "stats.json",
        "relative_stats": dataset.meta_root / "relative_stats_dreamzero.json",
        "hydra_patch": dataset.meta_root / "dreamzero_hydra_patch.yaml",
        "receipt": dataset.meta_root / "dreamzero_target_receipt.json",
    }
    failures = [f"missing_{name}" for name, path in required.items() if not path.is_file()]
    if failures:
        return {"target": "dreamzero", "valid": False, "failures": failures}
    modality = read_json(required["modality"])
    receipt = read_json(required["receipt"])
    for group in ("state", "action", "video", "annotation"):
        if not isinstance(modality.get(group), dict) or not modality[group]:
            failures.append(f"empty_modality_{group}")
    camera_keys = [str(value) for value in receipt.get("camera_keys") or []]
    try:
        dataset.validate_cameras(camera_keys, verify_files=verify_files)
    except TargetPreparationError as exc:
        failures.append(str(exc))
    for column in {
        str(value["original_key"])
        for group in ("state", "action")
        for value in (modality.get(group) or {}).values()
    }:
        try:
            dataset.read_column(dataset.episode_indices[0], column)
        except TargetPreparationError as exc:
            failures.append(str(exc))
    return {
        "target": "dreamzero",
        "schema_version": receipt.get("schema_version"),
        "valid": not failures,
        "ready_for_training": not failures and bool(receipt.get("upstream_profile_registered")),
        "episode_count": len(dataset.episodes),
        "embodiment_tag": receipt.get("embodiment_tag"),
        "requires_upstream_profile_registration": not bool(
            receipt.get("upstream_profile_registered")
        ),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare LeRobot v2 data for DreamZero/GEAR")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--profiles", type=Path, required=True)
    prepare.add_argument("--profile", required=True)
    prepare.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    prepare.add_argument("--verify-files", action="store_true")
    prepare.add_argument("--max-quantile-rows", type=int, default=1_000_000)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--verify-files", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        document, profile = load_target_profile(args.profiles, "dreamzero", args.profile)
        result = prepare_dreamzero_target(
            args.source_root,
            args.output_root,
            profile_document=document,
            profile=profile,
            profile_name=args.profile,
            link_mode=args.link_mode,
            verify_files=args.verify_files,
            max_quantile_rows=args.max_quantile_rows,
        )
    else:
        result = validate_dreamzero_target(args.root, verify_files=args.verify_files)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
