from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from .quality import QualityPolicy
from .intervals import indices_to_intervals, merge_bad_intervals
from .signal_audit import analyze_signal
from .source import ParquetSourceReader, split_tar_uri
from .utils import AtomicJsonlWriter, iter_jsonl, stable_id, write_json
from .video_audit import audit_local_video, audit_tar_video_members


@dataclass(frozen=True, slots=True)
class CleaningPolicy:
    min_frames: int = 81
    minimum_finite_ratio: float = 1.0
    abrupt_warning_ratio: float = 0.02
    abrupt_action_block_ratio: float = 0.20
    extreme_abrupt_action_block_ratio: float = 0.05
    timestamp_gap_warning_factor: float = 3.0
    timestamp_jitter_warning_ratio: float = 0.25
    declared_fps_warning_relative_error: float = 0.10
    action_alignment_min_score: float = 0.20
    action_alignment_max_lag_s: float = 0.50
    min_prompt_characters: int = 2
    bad_interval_padding_frames: int = 1
    quaternion_norm_tolerance: float = 0.05
    quaternion_max_step_rad: float = 1.50
    extreme_abrupt_multiplier: float = 10.0
    sparse_visual_sample_count: int = 9
    sparse_visual_max_cameras: int = 3
    visual_dark_value: int = 10
    visual_bright_value: int = 245
    visual_extreme_pixel_ratio: float = 0.98
    visual_minimum_entropy: float = 0.08
    visual_minimum_laplacian_variance: float = 0.0005
    visual_freeze_mean_absolute_difference: float = 0.002
    visual_freeze_dhash_distance: int = 1

    @classmethod
    def from_yaml(cls, path: Path) -> "CleaningPolicy":
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        values = payload.get("policy", payload)
        if not isinstance(values, dict):
            raise ValueError(f"Expected a policy mapping in {path}")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown cleaning policy keys: {unknown}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def visual_thresholds(self) -> dict[str, Any]:
        return {
            "dark_value": self.visual_dark_value,
            "bright_value": self.visual_bright_value,
            "extreme_pixel_ratio": self.visual_extreme_pixel_ratio,
            "minimum_entropy": self.visual_minimum_entropy,
            "minimum_laplacian_variance": self.visual_minimum_laplacian_variance,
            "freeze_mean_absolute_difference": self.visual_freeze_mean_absolute_difference,
            "freeze_dhash_distance": self.visual_freeze_dhash_distance,
        }


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _signal_metrics(
    values: list[Any],
    *,
    feature: dict[str, Any] | None = None,
    timestamps: list[float] | None = None,
    fps: float | None = None,
    source_key: str | None = None,
    domains: Iterable[str] = ("state", "action"),
    policy: CleaningPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or CleaningPolicy()
    return analyze_signal(
        values,
        feature=feature,
        timestamps=timestamps,
        fps=fps,
        source_key=source_key,
        domains=domains,
        interval_padding_frames=policy.bad_interval_padding_frames,
        quaternion_norm_tolerance=policy.quaternion_norm_tolerance,
        quaternion_max_step_rad=policy.quaternion_max_step_rad,
        extreme_abrupt_multiplier=policy.extreme_abrupt_multiplier,
    )


def _temporal_metrics(
    timestamps: list[Any], declared_fps: float | None, gap_factor: float
) -> dict[str, Any]:
    numeric = [_number(value) for value in timestamps]
    finite = all(math.isfinite(value) for value in numeric)
    strictly_increasing = finite and all(
        right > left for left, right in zip(numeric, numeric[1:])
    )
    deltas = [right - left for left, right in zip(numeric, numeric[1:])] if finite else []
    positive_deltas = [value for value in deltas if value > 0]
    median_delta = statistics.median(positive_deltas) if positive_deltas else None
    delta_mad = (
        statistics.median(abs(value - median_delta) for value in positive_deltas)
        if median_delta is not None
        else None
    )
    gap_threshold = None
    gap_count = 0
    gap_indices: list[int] = []
    if median_delta is not None:
        gap_threshold = median_delta * gap_factor
        gap_indices = [
            index
            for index, value in enumerate(deltas, start=1)
            if value > gap_threshold
        ]
        gap_count = len(gap_indices)
    inferred_fps = 1.0 / median_delta if median_delta and median_delta > 0 else None
    fps_relative_error = None
    if inferred_fps is not None and declared_fps and declared_fps > 0:
        fps_relative_error = abs(inferred_fps - declared_fps) / declared_fps
    return {
        "sample_count": len(numeric),
        "duration_s": (
            numeric[-1] - numeric[0]
            if strictly_increasing and len(numeric) >= 2
            else None
        ),
        "finite": finite,
        "strictly_increasing": strictly_increasing,
        "median_delta_s": median_delta,
        "delta_mad_s": delta_mad,
        "jitter_mad_ratio": (
            delta_mad / median_delta
            if delta_mad is not None and median_delta and median_delta > 0
            else None
        ),
        "minimum_delta_s": min(positive_deltas, default=None),
        "maximum_delta_s": max(positive_deltas, default=None),
        "gap_threshold_s": gap_threshold,
        "gap_count": gap_count,
        "gap_indices": gap_indices,
        "gap_indices_sample": gap_indices[:32],
        "inferred_fps": inferred_fps,
        "declared_fps": declared_fps,
        "declared_fps_relative_error": fps_relative_error,
    }


def _contiguous(values: list[Any]) -> bool:
    try:
        numeric = [int(value) for value in values]
    except (TypeError, ValueError):
        return False
    return all(right == left + 1 for left, right in zip(numeric, numeric[1:]))


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 8 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [item - left_mean for item in left]
    right_centered = [item - right_mean for item in right]
    left_norm = math.sqrt(sum(item * item for item in left_centered))
    right_norm = math.sqrt(sum(item * item for item in right_centered))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered)) / (
        left_norm * right_norm
    )


def _active_by_slot(mapping: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(item["canonical_index"]): item
        for item in mapping.get("mappings") or []
        if item.get("active")
    }


def _velocity_semantics(mapping: dict[str, Any]) -> bool:
    text = " ".join(
        str(mapping.get(key) or "")
        for key in ("source_key", "source_name", "semantic")
    ).lower()
    return any(
        token in text
        for token in ("velocity", "velocities", "speed", "twist", "_vel")
    )


def _action_state_alignment(
    table: Any, record: dict[str, Any], policy: CleaningPolicy
) -> dict[str, Any]:
    canonical = (record.get("metadata") or {}).get("canonical_mapping") or {}
    state_slots = _active_by_slot(canonical.get("state") or {})
    action_slots = _active_by_slot(canonical.get("action") or {})
    common_slots = sorted(
        slot
        for slot in set(state_slots) & set(action_slots)
        if 0 <= slot < 64
        and state_slots[slot].get("alignment_safe", True)
        and action_slots[slot].get("alignment_safe", True)
        and (
            slot >= 6
            or (
                state_slots[slot].get("alignment_safe")
                and action_slots[slot].get("alignment_safe")
            )
        )
    )
    series: list[tuple[int, list[float], list[float], str]] = []
    cache: dict[str, list[Any]] = {}
    for slot in common_slots:
        state_item = state_slots[slot]
        action_item = action_slots[slot]
        state_key = str(state_item["source_key"])
        action_key = str(action_item["source_key"])
        if state_key not in table.column_names or action_key not in table.column_names:
            continue
        if state_key not in cache:
            cache[state_key] = table[state_key].to_pylist()
        if action_key not in cache:
            cache[action_key] = table[action_key].to_pylist()
        state_index = int(state_item["source_index"])
        action_index = int(action_item["source_index"])
        try:
            state_values = [_number(row[state_index]) for row in cache[state_key]]
            action_values = [_number(row[action_index]) for row in cache[action_key]]
        except (IndexError, TypeError):
            continue
        action_is_velocity = _velocity_semantics(action_item)
        state_is_velocity = _velocity_semantics(state_item)
        comparison = "direct_feedback"
        if action_is_velocity and not state_is_velocity:
            state_values = [math.nan] + [
                (right - left) * float(record.get("fps") or 20.0)
                if math.isfinite(left) and math.isfinite(right)
                else math.nan
                for left, right in zip(state_values, state_values[1:])
            ]
            comparison = "action_vs_state_derivative"
        series.append((slot, state_values, action_values, comparison))

    fps = float(record.get("fps") or 20.0)
    max_lag = max(1, min(40, int(round(fps * policy.action_alignment_max_lag_s))))
    lag_scores: list[tuple[int, float, int]] = []
    for lag in range(max_lag + 1):
        correlations: list[float] = []
        for _, state_values, action_values, _ in series:
            action_slice = action_values[: len(action_values) - lag] if lag else action_values
            state_slice = state_values[lag:] if lag else state_values
            finite_pairs = [
                (action, state)
                for action, state in zip(action_slice, state_slice)
                if math.isfinite(action) and math.isfinite(state)
            ]
            if not finite_pairs:
                continue
            correlation = _pearson(
                [pair[0] for pair in finite_pairs], [pair[1] for pair in finite_pairs]
            )
            if correlation is not None:
                correlations.append(correlation)
        if correlations:
            lag_scores.append((lag, statistics.median(correlations), len(correlations)))
    if not lag_scores:
        return {"status": "pending", "reason": "no_varying_common_canonical_slots"}
    best_lag, best_score, dimensions = max(lag_scores, key=lambda item: item[1])
    return {
        "status": "passed" if best_score >= policy.action_alignment_min_score else "low_confidence",
        "best_lag_frames": best_lag,
        "best_lag_ms": 1000.0 * best_lag / fps,
        "alignment_score": best_score,
        "minimum_score": policy.action_alignment_min_score,
        "evaluated_dimension_count": dimensions,
        "common_slot_count": len(common_slots),
        "evaluated_slots": [slot for slot, _, _, _ in series],
        "comparison_modes": {
            str(slot): comparison for slot, _, _, comparison in series
        },
    }


def _language_metrics(record: dict[str, Any], policy: CleaningPolicy) -> dict[str, Any]:
    raw_tasks = record.get("tasks") or []
    if not isinstance(raw_tasks, list):
        raw_tasks = [raw_tasks]
    normalized: list[str] = []
    for task in raw_tasks:
        text = re.sub(r"\s+", " ", str(task)).strip()
        if text and text not in normalized:
            normalized.append(text)
    usable = [text for text in normalized if len(text) >= policy.min_prompt_characters]
    return {
        "raw_prompt_count": len(raw_tasks),
        "normalized_prompt_count": len(normalized),
        "duplicate_prompt_count": max(0, len(raw_tasks) - len(normalized)),
        "prompts": normalized,
        "primary_prompt": usable[0] if usable else None,
        "usable": bool(usable),
        "status": "passed" if usable else "pending",
    }


def _stage(status: str, reasons: list[str] | None = None) -> dict[str, Any]:
    return {"status": status, "reasons": sorted(set(reasons or []))}


def _trajectory_fingerprint(table: Any, signal_keys: list[str]) -> str | None:
    available = [key for key in signal_keys if key in table.column_names]
    if not available or table.num_rows <= 0:
        return None
    sample_count = min(32, table.num_rows)
    indices = sorted(
        {
            int(round(index * (table.num_rows - 1) / max(1, sample_count - 1)))
            for index in range(sample_count)
        }
    )

    def stable_value(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [stable_value(item) for item in value]
        numeric = _number(value)
        if math.isnan(numeric):
            return "nan"
        if math.isinf(numeric):
            return "inf" if numeric > 0 else "-inf"
        return round(numeric, 6)

    payload = {
        "rows": table.num_rows,
        "signals": {
            key: [stable_value(table[key][index].as_py()) for index in indices]
            for key in available
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _audit_visual_observations(
    result: dict[str, Any],
    record: dict[str, Any],
    *,
    policy: CleaningPolicy,
    check_videos: bool,
    decode_videos: bool,
) -> None:
    if not check_videos:
        result["pending_checks"].append("visual")
        result["stages"]["visual"] = _stage("pending")
        return

    cameras = [dict(camera) for camera in record.get("cameras") or []]
    role_priority = {
        "global_primary": 0,
        "left_wrist": 1,
        "right_wrist": 2,
        "global_secondary": 3,
        "auxiliary": 4,
    }
    cameras.sort(
        key=lambda camera: (
            role_priority.get(
                re.sub(r"_\d+$", "", str(camera.get("role") or "auxiliary")),
                5,
            ),
            "fisheye" in str(camera.get("source_key") or "").lower(),
            str(camera.get("source_key") or ""),
        )
    )
    if policy.sparse_visual_max_cameras <= 0:
        selected = cameras
    else:
        selected = []
        selected_roles: set[str] = set()
        for camera in cameras:
            role = re.sub(r"_\d+$", "", str(camera.get("role") or "auxiliary"))
            if role in selected_roles:
                continue
            selected.append(camera)
            selected_roles.add(role)
            if len(selected) >= policy.sparse_visual_max_cameras:
                break
        if len(selected) < policy.sparse_visual_max_cameras:
            selected_ids = {id(camera) for camera in selected}
            selected.extend(
                camera
                for camera in cameras
                if id(camera) not in selected_ids
            )
            selected = selected[: policy.sparse_visual_max_cameras]
    selected_keys = {str(camera.get("source_key") or "unknown") for camera in selected}
    video_reports: dict[str, Any] = {
        str(camera.get("source_key") or "unknown"): {
            "status": "pending",
            "reason": "not_selected_for_sparse_audit",
        }
        for camera in cameras
        if str(camera.get("source_key") or "unknown") not in selected_keys
    }
    tar_groups: dict[Path, list[tuple[dict[str, Any], str]]] = {}
    direct_cameras: list[dict[str, Any]] = []
    for camera in selected:
        uri = camera.get("source_uri")
        tar_source = split_tar_uri(str(uri)) if uri else None
        if tar_source is None:
            direct_cameras.append(camera)
        else:
            archive_path, member_name = tar_source
            tar_groups.setdefault(archive_path, []).append((camera, member_name))
    for camera in direct_cameras:
        uri = camera.get("source_uri")
        source_key = str(camera.get("source_key") or "unknown")
        if uri:
            video_reports[source_key] = audit_local_video(
                str(uri),
                decode=decode_videos,
                sample_frames=policy.sparse_visual_sample_count,
                thresholds=policy.visual_thresholds(),
                source_key=source_key,
                fps=camera.get("fps") or record.get("fps"),
            )
    for archive_path, requests in tar_groups.items():
        member_reports = audit_tar_video_members(
            archive_path,
            [member_name for _, member_name in requests],
            decode=decode_videos,
            sample_frames=policy.sparse_visual_sample_count,
            thresholds=policy.visual_thresholds(),
        )
        for camera, member_name in requests:
            source_key = str(camera.get("source_key") or "unknown")
            report = member_reports[member_name]
            report["source_key"] = source_key
            for interval in (report.get("sparse_visual") or {}).get("bad_intervals") or []:
                interval["camera_key"] = source_key
            video_reports[source_key] = report
    result["metrics"]["videos"] = video_reports
    visual_fingerprints = sorted(
        str(fingerprint)
        for report in video_reports.values()
        if (
            fingerprint := (report.get("sparse_visual") or {}).get(
                "visual_fingerprint_sha256"
            )
        )
    )
    if visual_fingerprints:
        result["fingerprints"]["visual_sha256"] = hashlib.sha256(
            "|".join(visual_fingerprints).encode("ascii")
        ).hexdigest()
    for report in video_reports.values():
        result["bad_intervals"].extend(
            (report.get("sparse_visual") or {}).get("bad_intervals") or []
        )

    statuses = [report.get("status") for report in video_reports.values()]
    sparse_statuses = [
        (report.get("sparse_visual") or {}).get("status")
        for report in video_reports.values()
        if report.get("status") == "passed" and report.get("sparse_visual")
    ]
    if statuses and all(status == "failed" for status in statuses):
        result["failures"].append("all_video_containers_failed")
        result["video_blockers"].append("all_video_containers_failed")
        result["stages"]["visual"] = _stage("failed", ["all_sources_failed"])
        return
    if not statuses:
        result["pending_checks"].append("visual_source_missing")
        result["stages"]["visual"] = _stage("pending", ["no_auditable_camera_uri"])
        return

    if any(status == "passed" for status in statuses):
        result["passed_checks"].append("visual_container")
    if any(status == "failed" for status in statuses):
        result["warnings"].append("partial_camera_audit_failure")
    if all(status == "pending" for status in statuses):
        result["pending_checks"].append("visual_format_audit")
        result["stages"]["visual"] = _stage("pending", ["unsupported_visual_format"])
        return

    evaluated_sparse = [status for status in sparse_statuses if status != "pending"]
    if evaluated_sparse and all(status == "failed" for status in evaluated_sparse):
        result["failures"].append("all_sampled_camera_visual_quality_failed")
        result["video_blockers"].append("all_sampled_camera_visual_quality_failed")
        result["stages"]["visual"] = _stage(
            "failed", ["all_sampled_cameras_extreme_or_undecodable"]
        )
    elif any(status in {"warning", "failed"} for status in evaluated_sparse):
        result["warnings"].append("sparse_visual_quality_flags")
        result["stages"]["visual"] = _stage(
            "warning", ["localized_visual_quality_flags"]
        )
    elif evaluated_sparse and all(status == "passed" for status in evaluated_sparse):
        result["passed_checks"].append("sparse_visual_quality")
        result["stages"]["visual"] = _stage("passed")
    else:
        result["pending_checks"].append("sparse_visual_quality")
        result["stages"]["visual"] = _stage("passed", ["container_only"])


def _stage_score(stage: dict[str, Any] | None) -> float:
    return {
        "passed": 1.0,
        "warning": 0.65,
        "low_confidence": 0.40,
        "pending": 0.50,
        "failed": 0.0,
    }.get(str((stage or {}).get("status") or "pending"), 0.50)


def _component_scores(record: dict[str, Any], audit: dict[str, Any]) -> dict[str, float]:
    stages = audit.get("stages") or {}
    integrity = 0.0 if audit.get("failures") else _stage_score(stages.get("structure"))
    kinematic = min(
        _stage_score(stages.get("signal")),
        _stage_score(stages.get("kinematic")),
        _stage_score(stages.get("action_alignment")),
    )
    dedupe_status = str((audit.get("dedupe") or {}).get("status") or "pending")
    novelty = {"unique": 1.0, "duplicate": 0.20, "pending": 0.50}.get(
        dedupe_status, 0.50
    )
    return {
        "integrity": integrity,
        "temporal": _stage_score(stages.get("temporal")),
        "visual": _stage_score(stages.get("visual")),
        "kinematic": kinematic,
        "language": _stage_score(stages.get("language")),
        "novelty": novelty,
    }


def _finalize_audit(result: dict[str, Any]) -> dict[str, Any]:
    result["bad_intervals"] = merge_bad_intervals(result.get("bad_intervals") or [])
    for key in (
        "passed_checks",
        "pending_checks",
        "warnings",
        "failures",
        "action_blockers",
        "video_blockers",
    ):
        result[key] = sorted(set(result.get(key) or []))
    return result


def audit_episode(
    record: dict[str, Any],
    reader: ParquetSourceReader,
    *,
    policy: CleaningPolicy | None = None,
    check_videos: bool = False,
    decode_videos: bool = False,
) -> dict[str, Any]:
    policy = policy or CleaningPolicy()
    state_schema = record.get("state_schema") or {}
    action_schema = record.get("action_schema") or {}
    canonical = (record.get("metadata") or {}).get("canonical_mapping") or {}
    active_signal_keys = {
        str(item["source_key"])
        for mapping in (canonical.get("state") or {}, canonical.get("action") or {})
        for item in mapping.get("mappings") or []
        if item.get("active")
    }
    references = record.get("references") or {}
    source_uri = str(references.get("data") or record.get("source_uri") or "")
    signal_keys = list(dict.fromkeys([*state_schema.keys(), *action_schema.keys()]))
    columns = ["timestamp", "frame_index", *signal_keys]
    result: dict[str, Any] = {
        "global_episode_id": record.get("global_episode_id"),
        "source_uri": source_uri,
        "passed_checks": [],
        "pending_checks": [],
        "warnings": [],
        "failures": [],
        "action_blockers": [],
        "video_blockers": [],
        "stages": {},
        "metrics": {},
        "bad_intervals": [],
        "fingerprints": {
            "source_uri_sha256": hashlib.sha256(source_uri.encode("utf-8")).hexdigest()
            if source_uri
            else None
        },
        "action_verified": False,
    }

    language = _language_metrics(record, policy)
    result["metrics"]["language"] = language
    if language["usable"]:
        result["passed_checks"].append("language")
        result["stages"]["language"] = _stage("passed")
    else:
        result["pending_checks"].append("language_review")
        result["warnings"].append("missing_or_unusable_task_prompt")
        result["stages"]["language"] = _stage("pending", ["no_usable_prompt"])

    _audit_visual_observations(
        result,
        record,
        policy=policy,
        check_videos=check_videos,
        decode_videos=decode_videos,
    )

    if not reader.supports_record(record):
        result["pending_checks"].append("native_format_deep_audit")
        result["action_blockers"].append("native_format_not_converted")
        result["stages"].update(
            {
                "structure": _stage("pending", ["native_converter_required"]),
                "temporal": _stage("pending"),
                "signal": _stage("pending"),
                "kinematic": _stage("pending"),
                "action_alignment": _stage("pending"),
            }
        )
        return _finalize_audit(result)

    try:
        table = reader.read_record(record, columns=columns)
    except Exception as exc:
        result["failures"].append("episode_data_unreadable")
        result["video_blockers"].append("episode_data_unreadable")
        result["action_blockers"].append("episode_data_unreadable")
        result["error"] = str(exc)
        result["stages"]["structure"] = _stage("failed", ["episode_data_unreadable"])
        return _finalize_audit(result)

    row_count = table.num_rows
    result["metrics"]["parquet_row_count"] = row_count
    result["metrics"]["source_row_count"] = row_count
    result["metrics"]["source_format"] = reader.source_kind(record)
    expected_rows = record.get("num_frames")
    structure_reasons: list[str] = []
    if expected_rows is not None and int(expected_rows) != row_count:
        structure_reasons.append("metadata_row_count_mismatch")
        result["action_blockers"].append("metadata_row_count_mismatch")
        result["warnings"].append("metadata_row_count_mismatch")
    if row_count <= 0:
        result["failures"].append("empty_episode")
        result["video_blockers"].append("empty_episode")
        structure_reasons.append("empty_episode")
    else:
        result["passed_checks"].append("episode_data_readable")
        if reader.source_kind(record) == "parquet":
            result["passed_checks"].append("parquet_readable")
        else:
            result["passed_checks"].append("native_format_decoded")
    result["stages"]["structure"] = _stage(
        "failed" if row_count <= 0 else ("warning" if structure_reasons else "passed"),
        structure_reasons,
    )

    available = set(table.column_names)
    timestamp_values = (
        [_number(value) for value in table["timestamp"].to_pylist()]
        if "timestamp" in available
        else None
    )
    temporal_ok = True
    temporal_reasons: list[str] = []
    if "timestamp" in available:
        temporal = _temporal_metrics(
            timestamp_values or [],
            record.get("fps"),
            policy.timestamp_gap_warning_factor,
        )
        result["metrics"]["temporal"] = temporal
        if not temporal["strictly_increasing"]:
            temporal_ok = False
            temporal_reasons.append("timestamp_not_strictly_increasing")
            result["failures"].append("temporal_discontinuity")
            result["action_blockers"].append("temporal_discontinuity")
        if temporal["gap_count"]:
            temporal_ok = False
            temporal_reasons.append("timestamp_gaps")
            result["warnings"].append("timestamp_gaps")
            result["action_blockers"].append("timestamp_gaps")
            result["bad_intervals"].extend(
                indices_to_intervals(
                    temporal.get("gap_indices") or [],
                    length=row_count,
                    reason="timestamp_gap",
                    domains=("video", "state", "action", "temporal"),
                    severity="hard",
                    timestamps=timestamp_values,
                    padding=policy.bad_interval_padding_frames,
                )
            )
        jitter = temporal.get("jitter_mad_ratio")
        if jitter is not None and jitter > policy.timestamp_jitter_warning_ratio:
            temporal_reasons.append("high_timestamp_jitter")
            result["warnings"].append("high_timestamp_jitter")
        fps_error = temporal.get("declared_fps_relative_error")
        if fps_error is not None and fps_error > policy.declared_fps_warning_relative_error:
            temporal_reasons.append("declared_fps_mismatch")
            result["warnings"].append("declared_fps_mismatch")
    else:
        temporal_ok = False
        temporal_reasons.append("timestamp_missing")
        result["pending_checks"].append("timestamp_missing")
        result["action_blockers"].append("timestamp_missing")
    if "frame_index" in available:
        contiguous = _contiguous(table["frame_index"].to_pylist())
        result["metrics"]["frame_index_contiguous"] = contiguous
        if not contiguous:
            temporal_ok = False
            temporal_reasons.append("frame_index_not_contiguous")
            result["warnings"].append("frame_index_not_contiguous")
            result["action_blockers"].append("frame_index_not_contiguous")
    else:
        result["pending_checks"].append("frame_index_missing")
    if temporal_ok:
        result["passed_checks"].append("temporal")
    result["stages"]["temporal"] = _stage(
        "failed"
        if "temporal_discontinuity" in result["failures"]
        else ("passed" if temporal_ok else "warning"),
        temporal_reasons,
    )

    signal_ok = True
    signal_reasons: list[str] = []
    kinematic_reasons: list[str] = []
    for key in signal_keys:
        training_relevant = key in active_signal_keys
        if key not in available:
            reason = f"signal_column_missing:{key}"
            signal_reasons.append(reason)
            if training_relevant:
                signal_ok = False
                result["action_blockers"].append(reason)
            else:
                result["warnings"].append(f"monitor_only_{reason}")
            continue
        feature = state_schema.get(key) or action_schema.get(key) or {}
        domains: set[str] = set()
        if key in state_schema:
            domains.add("state")
        if key in action_schema:
            domains.add("action")
        if not training_relevant:
            domains = {"monitor"}
        metrics = _signal_metrics(
            table[key].to_pylist(),
            feature=feature,
            timestamps=timestamp_values,
            fps=record.get("fps"),
            source_key=key,
            domains=domains or {"state", "action"},
            policy=policy,
        )
        result["metrics"].setdefault("signals", {})[key] = metrics
        if training_relevant:
            result["bad_intervals"].extend(metrics.get("bad_intervals") or [])
        if not metrics["row_width_consistent"]:
            reason = f"variable_signal_width:{key}"
            signal_reasons.append(reason)
            if training_relevant:
                signal_ok = False
                result["action_blockers"].append(reason)
            else:
                result["warnings"].append(f"monitor_only_{reason}")
        if metrics["worst_dimension_finite_ratio"] < policy.minimum_finite_ratio:
            reason = f"non_finite_signal:{key}"
            signal_reasons.append(reason)
            result["warnings"].append(reason)
            if training_relevant:
                signal_ok = False
                result["action_blockers"].append(reason)
        abrupt = metrics["maximum_abrupt_step_ratio"]
        extreme_abrupt = metrics["maximum_extreme_abrupt_step_ratio"]
        if abrupt > policy.abrupt_warning_ratio:
            result["warnings"].append(f"frequent_abrupt_steps:{key}")
        if (
            abrupt > policy.abrupt_action_block_ratio
            and extreme_abrupt > policy.extreme_abrupt_action_block_ratio
        ):
            reason = f"excessive_abrupt_steps:{key}"
            signal_reasons.append(reason)
            if training_relevant:
                signal_ok = False
                result["action_blockers"].append(reason)
            else:
                result["warnings"].append(f"monitor_only_{reason}")
        if metrics["constant_dimensions"]:
            result["warnings"].append(f"constant_signal_dimensions:{key}")
        quaternion_reports = (metrics.get("semantic") or {}).get("quaternions") or []
        if any(report.get("sign_flip_count") for report in quaternion_reports):
            result["passed_checks"].append("quaternion_sign_continuity")
        if any(report.get("norm_error_count") for report in quaternion_reports):
            reason = f"quaternion_norm_outlier:{key}"
            kinematic_reasons.append(reason)
            result["warnings"].append(reason)
        if any(report.get("rapid_geodesic_step_count") for report in quaternion_reports):
            reason = f"rapid_quaternion_rotation:{key}"
            kinematic_reasons.append(reason)
            result["warnings"].append(reason)
    if state_schema or action_schema:
        if signal_ok:
            result["passed_checks"].append("signal")
    else:
        signal_ok = False
        signal_reasons.append("schema_inference")
        result["pending_checks"].append("schema_inference")
        result["action_blockers"].append("state_or_action_schema_missing")
    result["stages"]["signal"] = _stage(
        "passed" if signal_ok else "warning", signal_reasons
    )
    if state_schema or action_schema:
        result["stages"]["kinematic"] = _stage(
            "warning" if kinematic_reasons else "passed", kinematic_reasons
        )
        if not kinematic_reasons:
            result["passed_checks"].append("kinematic")
    else:
        result["stages"]["kinematic"] = _stage("pending", ["schema_inference"])

    alignment = (
        _action_state_alignment(table, record, policy)
        if state_schema and action_schema
        else {"status": "pending", "reason": "state_or_action_schema_missing"}
    )
    result["metrics"]["action_state_alignment"] = alignment
    result["fingerprints"]["trajectory_sha256"] = _trajectory_fingerprint(
        table, signal_keys
    )
    alignment_ok = alignment["status"] == "passed"
    if alignment_ok:
        result["passed_checks"].append("action_state_alignment")
    elif alignment["status"] == "low_confidence":
        result["warnings"].append("low_action_state_alignment")
        result["pending_checks"].append("action_state_alignment_review")
        result["action_blockers"].append("low_action_state_alignment")
    else:
        result["pending_checks"].append("action_state_alignment")
        result["action_blockers"].append("action_state_alignment_pending")
    result["stages"]["action_alignment"] = _stage(
        alignment["status"],
        [] if alignment_ok else [str(alignment.get("reason") or alignment["status"])],
    )

    native_action_verified = bool(
        action_schema
        and state_schema
        and temporal_ok
        and signal_ok
        and alignment_ok
        and not result["failures"]
        and not result["action_blockers"]
    )
    canonical_action = (
        ((record.get("metadata") or {}).get("canonical_mapping") or {}).get("action")
        or {}
    )
    result["native_action_verified"] = native_action_verified
    result["action_verified"] = bool(
        native_action_verified and canonical_action.get("verified")
    )
    if native_action_verified and not result["action_verified"]:
        result["pending_checks"].append("canonical_action_mapping_review")
        result["action_blockers"].append("canonical_action_mapping_unverified")
    return _finalize_audit(result)


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    signals = metrics.get("signals") or {}
    videos = metrics.get("videos") or {}
    return {
        "parquet_row_count": metrics.get("parquet_row_count"),
        "temporal": metrics.get("temporal"),
        "frame_index_contiguous": metrics.get("frame_index_contiguous"),
        "signal_columns": {
            key: {
                "dimension_count": value.get("dimension_count"),
                "finite_ratio": value.get("finite_ratio"),
                "worst_dimension_finite_ratio": value.get(
                    "worst_dimension_finite_ratio"
                ),
                "maximum_abrupt_step_ratio": value.get(
                    "maximum_abrupt_step_ratio"
                ),
                "maximum_extreme_abrupt_step_ratio": value.get(
                    "maximum_extreme_abrupt_step_ratio"
                ),
                "constant_dimensions": value.get("constant_dimensions"),
                "semantic": value.get("semantic"),
                "bad_interval_count": len(value.get("bad_intervals") or []),
            }
            for key, value in signals.items()
        },
        "action_state_alignment": metrics.get("action_state_alignment"),
        "language": metrics.get("language"),
        "videos": {
            key: {
                "status": value.get("status"),
                "reason": value.get("reason"),
                "storage": value.get("storage"),
                "duration_s": value.get("duration_s"),
                "stream": value.get("stream"),
                "sparse_visual": {
                    field: (value.get("sparse_visual") or {}).get(field)
                    for field in (
                        "status",
                        "sampled_frame_count",
                        "flags",
                        "all_frames_extreme",
                        "all_pairs_frozen",
                        "visual_fingerprint_sha256",
                        "aggregate",
                        "bad_intervals",
                    )
                },
            }
            for key, value in videos.items()
        }
        if videos
        else None,
    }


def _admission(
    record: dict[str, Any], quality: dict[str, Any], audit: dict[str, Any], min_frames: int
) -> dict[str, Any]:
    reasons: list[str] = []
    length = record.get("num_frames")
    duration = ((audit.get("metrics") or {}).get("temporal") or {}).get("duration_s")
    if duration is None:
        duration = record.get("duration_s")
    if duration is None and length is not None:
        fps_candidates = [record.get("fps")]
        fps_candidates.extend(
            camera.get("fps") for camera in record.get("cameras") or []
        )
        for value in fps_candidates:
            try:
                fps = float(value)
            except (TypeError, ValueError):
                continue
            if fps > 0:
                duration = max(0.0, (int(length) - 1) / fps)
                break
    required_duration = (min_frames - 1) / 20.0
    timeline_available = duration is not None
    long_enough = timeline_available and float(duration) + 1e-8 >= required_duration
    if not timeline_available:
        reasons.append("timeline_duration_unknown")
    elif not long_enough:
        reasons.append("duration_shorter_than_training_window")
    if quality.get("tier") == "C" or not quality.get("video_eligible"):
        mode = "reject"
        reasons.extend(quality.get("failures") or [])
    elif not long_enough:
        mode = "reject"
    elif quality.get("action_eligible"):
        mode = "joint_video_action"
    else:
        mode = "video_only"
        reasons.extend(audit.get("action_blockers") or [])
    canonical = (record.get("metadata") or {}).get("canonical_mapping") or {}
    state_slots = (canonical.get("state") or {}).get("valid_slots") or []
    action_slots = (canonical.get("action") or {}).get("valid_slots") or []
    language_usable = bool((audit.get("metrics") or {}).get("language", {}).get("usable"))
    stage1_reasons = list(reasons if mode == "reject" else [])
    if not language_usable:
        stage1_reasons.append("language_missing_or_fallback_required")
    stage2_reasons = list(audit.get("action_blockers") or [])
    if mode != "joint_video_action":
        stage2_reasons.append("joint_video_action_not_verified")
    stage_admission = {
        "stage1_video_backbone": {
            "accepted": mode != "reject",
            "reasons": sorted(set(stage1_reasons)),
            "window_filter_domains": ["video", "temporal"],
        },
        "stage2_memory_fastwam": {
            "accepted": mode == "joint_video_action",
            "reasons": sorted(set(stage2_reasons)),
            "window_filter_domains": ["video", "temporal", "state", "action"],
        },
        "stage3_target_finetune": {
            "accepted_as_candidate": mode == "joint_video_action",
            "reasons": sorted(
                set(stage2_reasons)
                | ({"target_dataset_and_success_policy_required"} if mode == "joint_video_action" else set())
            ),
            "window_filter_domains": ["video", "temporal", "state", "action"],
        },
    }
    return {
        "schema_version": "fastwam-admission-v2",
        "accepted": mode != "reject",
        "mode": mode,
        "reasons": sorted(set(reasons)),
        "timeline": {
            "duration_s": duration,
            "required_duration_s": required_duration,
            "ready": long_enough,
        },
        "conditioning": {
            "video": mode != "reject",
            "state": mode == "joint_video_action" and bool(state_slots),
            "language": mode != "reject" and language_usable,
        },
        "losses": {
            "video": mode != "reject",
            "action": mode == "joint_video_action" and bool(action_slots),
        },
        "state_valid_slots": state_slots,
        "action_valid_slots": action_slots,
        "stages": stage_admission,
    }


def _apply_audit(
    record: dict[str, Any],
    audit: dict[str, Any],
    policy: QualityPolicy,
) -> dict[str, Any]:
    existing = record.get("quality") or {}
    passed = set(existing.get("passed_checks") or []) | set(audit["passed_checks"])
    pending = set(existing.get("pending_checks") or []) | set(audit["pending_checks"])
    pending -= passed
    warnings = set(existing.get("warnings") or []) | set(audit["warnings"])
    failures = set(existing.get("failures") or []) | set(audit["failures"])
    complete = "source_incomplete" not in failures and not audit["failures"]
    component_scores = _component_scores(record, audit)
    report = policy.evaluate(
        complete=complete,
        num_frames=record.get("num_frames"),
        has_video=bool(record.get("cameras")),
        has_state=bool(record.get("state_schema")),
        has_action=bool(record.get("action_schema")),
        action_verified=bool(audit["action_verified"]),
        visual_verified=(
            "referenced_files_exist" in passed or "visual_container" in passed
        ),
        passed_checks=sorted(passed),
        pending_checks=sorted(pending),
        warnings=sorted(warnings),
        failures=sorted(failures),
        component_scores=component_scores,
        hard_blockers=sorted(
            set(audit.get("video_blockers") or []) | set(audit.get("failures") or [])
        ),
        soft_flags=sorted(
            set(audit.get("warnings") or []) | set(audit.get("pending_checks") or [])
        ),
        bad_intervals=audit.get("bad_intervals") or [],
    )
    updated = dict(record)
    updated["quality"] = report.to_dict()
    split_source = (
        record.get("lineage_id")
        or (audit.get("fingerprints") or {}).get("source_uri_sha256")
        or record.get("global_episode_id")
    )
    updated["split_group_id"] = stable_id(
        "split_group", split_source, length=24
    )
    updated["bad_intervals"] = audit.get("bad_intervals") or []
    metadata = dict(updated.get("metadata") or {})
    metadata["latest_audit"] = {
        "stages": audit["stages"],
        "metrics_summary": _metrics_summary(audit["metrics"]),
        "action_blockers": audit.get("action_blockers") or [],
        "video_blockers": audit.get("video_blockers") or [],
        "native_action_verified": audit.get("native_action_verified", False),
        "action_verified": audit["action_verified"],
        "bad_intervals": audit.get("bad_intervals") or [],
        "fingerprints": audit.get("fingerprints") or {},
        "dedupe": audit.get("dedupe") or {"status": "pending"},
        "component_scores": component_scores,
    }
    updated["metadata"] = metadata
    updated["training_admission"] = _admission(
        updated, report.to_dict(), audit, policy.min_frames
    )
    updated["stage_admission"] = updated["training_admission"].get("stages") or {}
    return updated


def clean_manifest(
    manifest: Path,
    output_dir: Path,
    *,
    min_frames: int = 81,
    max_episodes: int | None = None,
    check_videos: bool = False,
    decode_videos: bool = False,
    cleaning_policy: CleaningPolicy | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_policy = cleaning_policy or CleaningPolicy(min_frames=min_frames)
    quality_policy = QualityPolicy(min_frames=audit_policy.min_frames)
    tiers = {tier: 0 for tier in "ABC"}
    admissions = {mode: 0 for mode in ("joint_video_action", "video_only", "reject")}
    episode_count = 0
    action_eligible_count = 0
    video_eligible_count = 0
    failure_count = 0
    duplicate_count = 0
    seen_fingerprints: dict[tuple[str, str], str] = {}
    with (
        ParquetSourceReader() as reader,
        AtomicJsonlWriter(output_dir / "episodes.cleaned.jsonl") as cleaned_writer,
        AtomicJsonlWriter(output_dir / "cleaning_report.jsonl") as audit_writer,
    ):
        for record in iter_jsonl(manifest):
            if max_episodes is not None and episode_count >= max_episodes:
                break
            audit = audit_episode(
                record,
                reader,
                policy=audit_policy,
                check_videos=check_videos,
                decode_videos=decode_videos,
            )
            fingerprints = audit.get("fingerprints") or {}
            fingerprint_kind = next(
                (
                    key
                    for key in (
                        "trajectory_sha256",
                        "visual_sha256",
                        "source_uri_sha256",
                    )
                    if fingerprints.get(key)
                ),
                None,
            )
            fingerprint = fingerprints.get(fingerprint_kind) if fingerprint_kind else None
            if fingerprint_kind and fingerprint:
                dedupe_key = (fingerprint_kind, str(fingerprint))
                duplicate_of = seen_fingerprints.get(dedupe_key)
                group_id = stable_id("content_group", fingerprint_kind, fingerprint, length=24)
                if duplicate_of:
                    audit["dedupe"] = {
                        "status": "duplicate",
                        "kind": fingerprint_kind,
                        "group_id": group_id,
                        "duplicate_of": duplicate_of,
                    }
                    audit["warnings"].append("duplicate_content_candidate")
                    duplicate_count += 1
                else:
                    seen_fingerprints[dedupe_key] = str(record.get("global_episode_id"))
                    audit["dedupe"] = {
                        "status": (
                            "pending"
                            if fingerprint_kind == "source_uri_sha256"
                            else "unique"
                        ),
                        "kind": fingerprint_kind,
                        "group_id": group_id,
                    }
                    if fingerprint_kind == "source_uri_sha256":
                        audit["dedupe"]["reason"] = "source_identity_only"
            else:
                audit["dedupe"] = {"status": "pending", "reason": "no_fingerprint"}
            audit = _finalize_audit(audit)
            cleaned = _apply_audit(record, audit, quality_policy)
            audit_writer.write(audit)
            cleaned_writer.write(cleaned)
            episode_count += 1
            tier = str((cleaned.get("quality") or {}).get("tier") or "C")
            tiers[tier] = tiers.get(tier, 0) + 1
            mode = str((cleaned.get("training_admission") or {}).get("mode") or "reject")
            admissions[mode] = admissions.get(mode, 0) + 1
            action_eligible_count += int(
                bool((cleaned.get("quality") or {}).get("action_eligible"))
            )
            video_eligible_count += int(
                bool((cleaned.get("quality") or {}).get("video_eligible"))
            )
            failure_count += int(bool(audit["failures"]))

    summary = {
        "input_manifest": str(manifest),
        "episode_count": episode_count,
        "quality_tiers": tiers,
        "training_admissions": admissions,
        "action_eligible_count": action_eligible_count,
        "video_eligible_count": video_eligible_count,
        "failure_count": failure_count,
        "duplicate_candidate_count": duplicate_count,
        "check_videos": check_videos,
        "decode_videos": decode_videos,
        "policy": audit_policy.to_dict(),
    }
    write_json(output_dir / "cleaning_summary.json", summary)
    return summary
