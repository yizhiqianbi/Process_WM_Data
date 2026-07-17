from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


def indices_to_intervals(
    indices: Iterable[int],
    *,
    length: int,
    reason: str,
    domains: Iterable[str],
    severity: str,
    timestamps: list[float] | None = None,
    padding: int = 0,
    source_key: str | None = None,
    camera_key: str | None = None,
) -> list[dict[str, Any]]:
    """Compress bad sample indices into half-open source-timeline intervals."""
    if length <= 0:
        return []
    expanded: set[int] = set()
    for raw_index in indices:
        index = int(raw_index)
        expanded.update(
            range(max(0, index - padding), min(length, index + padding + 1))
        )
    if not expanded:
        return []

    ordered = sorted(expanded)
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index != previous + 1:
            ranges.append((start, previous + 1))
            start = index
        previous = index
    ranges.append((start, previous + 1))

    numeric_timestamps = [float(value) for value in timestamps or []]
    usable_timestamps = (
        len(numeric_timestamps) == length
        and all(math.isfinite(value) for value in numeric_timestamps)
        and all(
            right > left
            for left, right in zip(numeric_timestamps, numeric_timestamps[1:])
        )
    )
    median_delta = (
        statistics.median(
            right - left
            for left, right in zip(numeric_timestamps, numeric_timestamps[1:])
        )
        if usable_timestamps and length > 1
        else None
    )
    result: list[dict[str, Any]] = []
    for start, stop in ranges:
        interval: dict[str, Any] = {
            "timeline": "source_control",
            "start": start,
            "stop_exclusive": stop,
            "reason": reason,
            "domains": sorted(set(str(value) for value in domains)),
            "severity": severity,
        }
        if source_key:
            interval["source_key"] = source_key
        if camera_key:
            interval["camera_key"] = camera_key
        if usable_timestamps:
            interval["start_s"] = numeric_timestamps[start]
            interval["stop_s"] = (
                numeric_timestamps[stop]
                if stop < length
                else numeric_timestamps[-1] + float(median_delta or 0.0)
            )
        result.append(interval)
    return result


def merge_bad_intervals(intervals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent intervals only when their filtering semantics are identical."""
    normalized = [dict(interval) for interval in intervals]
    normalized.sort(
        key=lambda value: (
            str(value.get("timeline") or ""),
            tuple(value.get("domains") or []),
            str(value.get("severity") or ""),
            str(value.get("reason") or ""),
            str(value.get("source_key") or ""),
            str(value.get("camera_key") or ""),
            int(value.get("start") or 0),
        )
    )
    result: list[dict[str, Any]] = []
    semantic_keys = (
        "timeline",
        "domains",
        "severity",
        "reason",
        "source_key",
        "camera_key",
    )
    for interval in normalized:
        if not result:
            result.append(interval)
            continue
        previous = result[-1]
        same_semantics = all(previous.get(key) == interval.get(key) for key in semantic_keys)
        previous_stop = int(previous.get("stop_exclusive") or 0)
        current_start = int(interval.get("start") or 0)
        if same_semantics and current_start <= previous_stop:
            previous["stop_exclusive"] = max(
                previous_stop, int(interval.get("stop_exclusive") or previous_stop)
            )
            if interval.get("stop_s") is not None:
                previous["stop_s"] = max(
                    float(previous.get("stop_s") or interval["stop_s"]),
                    float(interval["stop_s"]),
                )
            continue
        result.append(interval)
    return result


def map_intervals_to_timeline(
    intervals: Iterable[dict[str, Any]],
    *,
    source_timestamps: list[float],
    target_fps: float,
    target_count: int,
) -> list[dict[str, Any]]:
    """Map source-index or time intervals onto the canonical control timeline."""
    mapped: list[dict[str, Any]] = []
    source_count = len(source_timestamps)
    source_origin = source_timestamps[0] if source_timestamps else 0.0
    source_delta = (
        statistics.median(
            right - left
            for left, right in zip(source_timestamps, source_timestamps[1:])
        )
        if source_count > 1
        else 1.0 / target_fps
    )
    for raw in intervals:
        interval = dict(raw)
        start_s = interval.get("start_s")
        stop_s = interval.get("stop_s")
        if start_s is None and source_count:
            source_start = max(0, min(source_count - 1, int(interval.get("start") or 0)))
            start_s = source_timestamps[source_start]
        if stop_s is None and source_count:
            source_stop = max(0, int(interval.get("stop_exclusive") or 0))
            stop_s = (
                source_timestamps[source_stop]
                if source_stop < source_count
                else source_timestamps[-1] + source_delta
            )
        if start_s is None or stop_s is None:
            continue
        if str(interval.get("timeline") or "") == "source_control":
            start_s = float(start_s) - source_origin
            stop_s = float(stop_s) - source_origin
        start = max(0, int(math.floor(float(start_s) * target_fps + 1e-8)))
        stop = min(
            target_count,
            max(start + 1, int(math.ceil(float(stop_s) * target_fps - 1e-8))),
        )
        if start >= target_count or stop <= 0:
            continue
        interval.update(
            {
                "source_timeline": interval.get("timeline"),
                "source_start": interval.get("start"),
                "source_stop_exclusive": interval.get("stop_exclusive"),
                "timeline": "canonical_20hz",
                "start": start,
                "stop_exclusive": stop,
            }
        )
        mapped.append(interval)
    return merge_bad_intervals(mapped)


def blocked_window_starts(
    starts: Iterable[int],
    *,
    window_size: int,
    intervals: Iterable[dict[str, Any]],
    domains: set[str],
    severities: set[str] | None = None,
) -> set[int]:
    severities = severities or {"hard"}
    relevant: list[tuple[int, int]] = []
    for interval in intervals:
        if str(interval.get("severity") or "soft") not in severities:
            continue
        interval_domains = {str(value) for value in interval.get("domains") or []}
        if not interval_domains.intersection(domains):
            continue
        relevant.append(
            (
                int(interval.get("start") or 0),
                int(interval.get("stop_exclusive") or 0),
            )
        )
    return {
        int(start)
        for start in starts
        if any(
            int(start) < interval_stop
            and int(start) + window_size > interval_start
            for interval_start, interval_stop in relevant
        )
    }


def compact_start_ranges(starts: Iterable[int], *, stride: int) -> list[dict[str, int]]:
    ordered = sorted(set(int(value) for value in starts))
    if not ordered:
        return []
    groups: list[list[int]] = [[ordered[0]]]
    for start in ordered[1:]:
        if start == groups[-1][-1] + stride:
            groups[-1].append(start)
        else:
            groups.append([start])
    return [
        {
            "start": group[0],
            "stop_exclusive": group[-1] + stride,
            "stride": stride,
            "count": len(group),
        }
        for group in groups
    ]
