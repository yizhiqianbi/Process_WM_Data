from __future__ import annotations

import bisect
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .canonical import CANONICAL_DIM
from .intervals import map_intervals_to_timeline
from .source import ParquetSourceReader
from .utils import AtomicJsonlWriter, iter_jsonl, slug, stable_id, write_json


def _canonical_column(
    table: Any, mapping: dict[str, Any]
) -> tuple[list[list[float]], list[list[bool]]]:
    rows = [[0.0] * CANONICAL_DIM for _ in range(table.num_rows)]
    masks = [[False] * CANONICAL_DIM for _ in range(table.num_rows)]
    cache: dict[str, list[Any]] = {}
    for item in mapping.get("mappings") or []:
        if not item.get("active", False):
            continue
        source_key = str(item["source_key"])
        if source_key not in table.column_names:
            continue
        if source_key not in cache:
            cache[source_key] = table[source_key].to_pylist()
        source_index = int(item["source_index"])
        target_index = int(item["canonical_index"])
        for row_index, source_value in enumerate(cache[source_key]):
            if not isinstance(source_value, (list, tuple)) or source_index >= len(source_value):
                continue
            value = source_value[source_index]
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            rows[row_index][target_index] = numeric
            masks[row_index][target_index] = True
    return rows, masks


def _resample_rows(
    rows: list[list[float]],
    masks: list[list[bool]],
    source_timestamps: list[float],
    target_timestamps: list[float],
    *,
    zero_order_slots: set[int],
) -> tuple[list[list[float]], list[list[bool]], list[int], list[float]]:
    output_rows: list[list[float]] = []
    output_masks: list[list[bool]] = []
    nearest_indices: list[int] = []
    nearest_errors: list[float] = []
    for target in target_timestamps:
        right = bisect.bisect_left(source_timestamps, target)
        if right <= 0:
            left = right = 0
        elif right >= len(source_timestamps):
            left = right = len(source_timestamps) - 1
        elif source_timestamps[right] == target:
            left = right
        else:
            left = right - 1
        if left == right:
            alpha = 0.0
        else:
            width = source_timestamps[right] - source_timestamps[left]
            alpha = (target - source_timestamps[left]) / width
        nearest = left if alpha <= 0.5 else right
        nearest_indices.append(nearest)
        nearest_errors.append(abs(source_timestamps[nearest] - target))
        row = [0.0] * CANONICAL_DIM
        mask = [False] * CANONICAL_DIM
        for slot in range(CANONICAL_DIM):
            if slot in zero_order_slots or left == right:
                if masks[left][slot]:
                    row[slot] = rows[left][slot]
                    mask[slot] = True
            elif masks[left][slot] and masks[right][slot]:
                row[slot] = rows[left][slot] * (1.0 - alpha) + rows[right][slot] * alpha
                mask[slot] = True
        output_rows.append(row)
        output_masks.append(mask)
    return output_rows, output_masks, nearest_indices, nearest_errors


def _nearest_alignment(
    source_timestamps: list[float], target_timestamps: list[float]
) -> tuple[list[int], list[float]]:
    indices: list[int] = []
    errors: list[float] = []
    for target in target_timestamps:
        right = bisect.bisect_left(source_timestamps, target)
        if right <= 0:
            nearest = 0
        elif right >= len(source_timestamps):
            nearest = len(source_timestamps) - 1
        else:
            left = right - 1
            nearest = (
                left
                if target - source_timestamps[left]
                <= source_timestamps[right] - target
                else right
            )
        indices.append(nearest)
        errors.append(abs(source_timestamps[nearest] - target))
    return indices, errors


def _zero_order_slots(mapping: dict[str, Any]) -> set[int]:
    return {
        int(item["canonical_index"])
        for item in mapping.get("mappings") or []
        if item.get("active") and "gripper" in str(item.get("semantic"))
    }


def _metadata_fps(record: dict[str, Any]) -> float:
    candidates = [record.get("fps")]
    candidates.extend(camera.get("fps") for camera in record.get("cameras") or [])
    for value in candidates:
        try:
            fps = float(value)
        except (TypeError, ValueError):
            continue
        if fps > 0:
            return fps
    return 0.0


def _write_parquet_atomic(table: Any, path: Path) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, temp_name, compression="zstd")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def materialize_canonical(
    manifest: Path,
    output_dir: Path,
    *,
    max_episodes: int | None = None,
    include_tier_b: bool = True,
    target_fps: float = 20.0,
) -> dict[str, Any]:
    import pyarrow as pa
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized_count = 0
    skipped_count = 0
    considered_count = 0
    modes: dict[str, int] = {}
    with (
        ParquetSourceReader() as reader,
        AtomicJsonlWriter(output_dir / "canonical_episodes.jsonl") as manifest_writer,
        AtomicJsonlWriter(output_dir / "materialization_skipped.jsonl") as skipped_writer,
    ):
        for record in iter_jsonl(manifest):
            if max_episodes is not None and considered_count >= max_episodes:
                break
            considered_count += 1
            quality = record.get("quality") or {}
            tier = quality.get("tier")
            admission = record.get("training_admission") or {}
            admission_mode = str(admission.get("mode") or "unknown")
            if (
                tier == "C"
                or admission_mode == "reject"
                or (tier == "B" and not include_tier_b)
            ):
                skipped_writer.write(
                    {
                        "global_episode_id": record.get("global_episode_id"),
                        "reason": "training_admission_or_quality_tier",
                        "quality_tier": tier,
                        "admission_mode": admission_mode,
                    }
                )
                skipped_count += 1
                continue
            mapping = (record.get("metadata") or {}).get("canonical_mapping") or {}
            state_mapping = mapping.get("state") or {}
            action_mapping = mapping.get("action") or {}
            source_uri = str((record.get("references") or {}).get("data") or record.get("source_uri") or "")
            readable_source = reader.supports_record(record)
            source_kind = reader.source_kind(record)
            timeline_only = not readable_source
            if readable_source:
                columns = [
                    "timestamp",
                    "frame_index",
                    *[item["source_key"] for item in state_mapping.get("mappings") or []],
                    *[item["source_key"] for item in action_mapping.get("mappings") or []],
                ]
                try:
                    table = reader.read_record(
                        record, columns=list(dict.fromkeys(columns))
                    )
                    state, state_mask = _canonical_column(table, state_mapping)
                    action, action_mask = _canonical_column(table, action_mapping)
                except Exception as exc:
                    skipped_writer.write(
                        {
                            "global_episode_id": record.get("global_episode_id"),
                            "reason": "materialization_failed",
                            "error": str(exc),
                        }
                    )
                    skipped_count += 1
                    continue
                source_row_count = table.num_rows
                if "timestamp" in table.column_names:
                    raw_timestamps = [
                        float(item) for item in table["timestamp"].to_pylist()
                    ]
                else:
                    source_fps = _metadata_fps(record)
                    raw_timestamps = (
                        [index / source_fps for index in range(source_row_count)]
                        if source_fps > 0
                        else []
                    )
            else:
                source_row_count = int(record.get("num_frames") or 0)
                source_fps = _metadata_fps(record)
                if admission_mode != "video_only" or source_row_count < 2 or source_fps <= 0:
                    skipped_writer.write(
                        {
                            "global_episode_id": record.get("global_episode_id"),
                            "reason": "native_converter_or_timeline_metadata_required",
                            "num_frames": source_row_count,
                            "source_fps": source_fps or None,
                        }
                    )
                    skipped_count += 1
                    continue
                raw_timestamps = [
                    index / source_fps for index in range(source_row_count)
                ]

            if source_row_count < 2:
                skipped_writer.write(
                    {"global_episode_id": record.get("global_episode_id"), "reason": "too_few_rows"}
                )
                skipped_count += 1
                continue
            if not raw_timestamps:
                skipped_writer.write(
                    {
                        "global_episode_id": record.get("global_episode_id"),
                        "reason": "timestamp_and_fps_missing",
                    }
                )
                skipped_count += 1
                continue
            source_timestamps = [value - raw_timestamps[0] for value in raw_timestamps]
            if not all(
                right > left for left, right in zip(source_timestamps, source_timestamps[1:])
            ):
                skipped_writer.write(
                    {
                        "global_episode_id": record.get("global_episode_id"),
                        "reason": "non_monotonic_timestamp",
                    }
                )
                skipped_count += 1
                continue
            target_count = int(math.floor(source_timestamps[-1] * target_fps + 1e-8)) + 1
            target_timestamps = [index / target_fps for index in range(target_count)]
            if timeline_only:
                state = [[0.0] * CANONICAL_DIM for _ in range(target_count)]
                state_mask = [[False] * CANONICAL_DIM for _ in range(target_count)]
                action = [[0.0] * CANONICAL_DIM for _ in range(target_count)]
                action_mask = [[False] * CANONICAL_DIM for _ in range(target_count)]
                nearest_indices, nearest_errors = _nearest_alignment(
                    source_timestamps, target_timestamps
                )
            else:
                state, state_mask, nearest_indices, nearest_errors = _resample_rows(
                    state,
                    state_mask,
                    source_timestamps,
                    target_timestamps,
                    zero_order_slots=_zero_order_slots(state_mapping),
                )
                action, action_mask, _, _ = _resample_rows(
                    action,
                    action_mask,
                    source_timestamps,
                    target_timestamps,
                    zero_order_slots=_zero_order_slots(action_mapping),
                )

            arrays: dict[str, Any] = {}
            arrays["timestamp"] = pa.array(target_timestamps, type=pa.float64())
            arrays["frame_index"] = pa.array(range(target_count), type=pa.int64())
            arrays["source_nearest_frame_index"] = pa.array(
                nearest_indices, type=pa.int64()
            )
            arrays["source_nearest_error_s"] = pa.array(nearest_errors, type=pa.float32())
            arrays["canonical_state"] = pa.array(state, type=pa.list_(pa.float32(), CANONICAL_DIM))
            arrays["state_dim_valid_mask"] = pa.array(
                state_mask, type=pa.list_(pa.bool_(), CANONICAL_DIM)
            )
            arrays["canonical_action"] = pa.array(action, type=pa.list_(pa.float32(), CANONICAL_DIM))
            arrays["action_dim_valid_mask"] = pa.array(
                action_mask, type=pa.list_(pa.bool_(), CANONICAL_DIM)
            )
            output_table = pa.table(arrays)
            output_table = output_table.replace_schema_metadata(
                {
                    b"schema_version": b"fastwam-canonical-episode-v1",
                    b"global_episode_id": str(record["global_episode_id"]).encode("utf-8"),
                    b"target_fps": str(target_fps).encode("ascii"),
                    b"canonical_dimension": str(CANONICAL_DIM).encode("ascii"),
                }
            )
            episode_dir = output_dir / "episodes" / slug(record["global_episode_id"])
            episode_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = episode_dir / "canonical.parquet"
            _write_parquet_atomic(output_table, parquet_path)
            split_group_id = record.get("split_group_id") or stable_id(
                "split_group",
                record.get("dataset"),
                record.get("lineage_id") or record.get("global_episode_id"),
                length=24,
            )
            bad_intervals = map_intervals_to_timeline(
                record.get("bad_intervals")
                or ((record.get("quality") or {}).get("bad_intervals") or []),
                source_timestamps=raw_timestamps,
                target_fps=target_fps,
                target_count=target_count,
            )
            sidecar = {
                "schema_version": "fastwam-canonical-episode-v1",
                "global_episode_id": record["global_episode_id"],
                "dataset": record["dataset"],
                "release": record.get("release"),
                "source_episode_id": record.get("source_episode_id"),
                "lineage_id": record.get("lineage_id"),
                "split_group_id": split_group_id,
                "embodiment": record.get("embodiment"),
                "robot_type": record.get("robot_type"),
                "task_namespace": record.get("task_namespace"),
                "tasks": record.get("tasks") or [],
                "source_uri": source_uri,
                "canonical_parquet": str(parquet_path),
                "source_num_frames": source_row_count,
                "timeline_source": source_kind or "metadata_fps",
                "num_frames": target_count,
                "fps": target_fps,
                "duration_s": target_timestamps[-1] if target_timestamps else 0.0,
                "source_fps": record.get("fps"),
                "maximum_source_alignment_error_s": max(nearest_errors, default=0.0),
                "state_valid_slots": state_mapping.get("valid_slots") or [],
                "action_valid_slots": action_mapping.get("valid_slots") or [],
                "state_mapping_verified": bool(state_mapping.get("verified")),
                "action_mapping_verified": bool(action_mapping.get("verified")),
                "canonical_mapping": {
                    "state": state_mapping,
                    "action": action_mapping,
                },
                "cameras": record.get("cameras") or [],
                "videos": (record.get("references") or {}).get("videos") or {},
                "references": record.get("references") or {},
                "quality": quality,
                "training_admission": admission,
                "stage_admission": record.get("stage_admission")
                or admission.get("stages")
                or {},
                "bad_intervals": bad_intervals,
                "audit_summary": (record.get("metadata") or {}).get("latest_audit") or {},
                "normalization_domain": stable_id(
                    "normalization",
                    record.get("dataset"),
                    record.get("embodiment"),
                    tuple(state_mapping.get("valid_slots") or []),
                    tuple(action_mapping.get("valid_slots") or []),
                    length=24,
                ),
            }
            write_json(episode_dir / "episode.json", sidecar)
            manifest_writer.write(sidecar)
            materialized_count += 1
            modes[admission_mode] = modes.get(admission_mode, 0) + 1

    summary = {
        "input_manifest": str(manifest),
        "canonical_dimension": CANONICAL_DIM,
        "target_fps": target_fps,
        "considered_episode_count": considered_count,
        "materialized_episode_count": materialized_count,
        "skipped_episode_count": skipped_count,
        "training_admissions": modes,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "materialization_summary.json", summary)
    return summary
