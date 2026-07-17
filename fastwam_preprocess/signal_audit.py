from __future__ import annotations

import math
import re
import statistics
from typing import Any, Iterable

from .intervals import indices_to_intervals, merge_bad_intervals


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def vectors(values: Iterable[Any]) -> list[list[float]]:
    rows: list[list[float]] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            rows.append([number(item) for item in value])
        elif value is None:
            rows.append([math.nan])
        else:
            rows.append([number(value)])
    return rows


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    alpha = position - left
    return ordered[left] * (1.0 - alpha) + ordered[right] * alpha


def _robust_threshold(values: list[float]) -> tuple[float, float, float]:
    active = [abs(value) for value in values if math.isfinite(value) and abs(value) > 1e-12]
    if not active:
        return 0.0, 0.0, 0.0
    median = statistics.median(active)
    mad = statistics.median(abs(value - median) for value in active)
    threshold = max(1e-6, median * 20.0, median + 12.0 * 1.4826 * mad)
    return median, mad, threshold


def _derivative(
    values: list[float], timestamps: list[float]
) -> tuple[list[float], list[int]]:
    output = [math.nan]
    source_indices = [0]
    for index, (left, right) in enumerate(zip(values, values[1:]), start=1):
        delta_t = timestamps[index] - timestamps[index - 1]
        output.append(
            (right - left) / delta_t
            if math.isfinite(left)
            and math.isfinite(right)
            and math.isfinite(delta_t)
            and delta_t > 0
            else math.nan
        )
        source_indices.append(index)
    return output, source_indices


def _derivative_summary(values: list[float]) -> tuple[dict[str, Any], list[int]]:
    finite = [abs(value) for value in values if math.isfinite(value)]
    median, mad, threshold = _robust_threshold(values)
    outliers = [
        index
        for index, value in enumerate(values)
        if math.isfinite(value) and abs(value) > threshold and threshold > 0
    ]
    return (
        {
            "finite_count": len(finite),
            "p50_abs": quantile(finite, 0.50),
            "p99_abs": quantile(finite, 0.99),
            "maximum_abs": max(finite, default=0.0),
            "median_active_abs": median,
            "mad": mad,
            "robust_outlier_threshold": threshold,
            "robust_outlier_count": len(outliers),
            "robust_outlier_ratio": len(outliers) / max(1, len(values) - 1),
            "robust_outlier_indices_sample": outliers[:32],
        },
        outliers,
    )


def _dimension_metrics(
    values: list[float],
    dimension: int,
    timestamps: list[float],
    extreme_abrupt_multiplier: float,
) -> tuple[dict[str, Any], list[int], list[int], list[int], list[int]]:
    finite = [item for item in values if math.isfinite(item)]
    non_finite_indices = [
        index for index, value in enumerate(values) if not math.isfinite(value)
    ]
    value_range = max(finite) - min(finite) if finite else None
    indexed_steps = [
        (index, abs(right - left))
        for index, (left, right) in enumerate(zip(values, values[1:]), start=1)
        if math.isfinite(left) and math.isfinite(right)
    ]
    steps = [value for _, value in indexed_steps]
    movement_epsilon = max(1e-12, (value_range or 0.0) * 1e-10)
    active_steps = [item for item in steps if item > movement_epsilon]
    median_step = statistics.median(steps) if steps else 0.0
    median_active_step = statistics.median(active_steps) if active_steps else 0.0
    mad = (
        statistics.median(abs(item - median_active_step) for item in active_steps)
        if active_steps
        else 0.0
    )
    threshold = (
        max(
            1e-6,
            median_active_step * 20.0,
            median_active_step + 12.0 * 1.4826 * mad,
        )
        if active_steps
        else 0.0
    )
    abrupt_indices = [
        index for index, value in indexed_steps if active_steps and value > threshold
    ]
    extreme_abrupt_indices = [
        index
        for index, value in indexed_steps
        if active_steps and value > threshold * extreme_abrupt_multiplier
    ]

    velocity, _ = _derivative(values, timestamps)
    acceleration, _ = _derivative(velocity, timestamps)
    jerk, _ = _derivative(acceleration, timestamps)
    velocity_metrics, _ = _derivative_summary(velocity)
    acceleration_metrics, acceleration_outliers = _derivative_summary(acceleration)
    jerk_metrics, jerk_outliers = _derivative_summary(jerk)
    metrics = {
        "dimension": dimension,
        "sample_count": len(values),
        "finite_count": len(finite),
        "finite_ratio": len(finite) / max(1, len(values)),
        "p01": quantile(finite, 0.01),
        "p50": quantile(finite, 0.50),
        "p99": quantile(finite, 0.99),
        "range": value_range,
        "median_abs_step": median_step,
        "active_step_count": len(active_steps),
        "median_active_abs_step": median_active_step,
        "step_mad": mad,
        "maximum_abs_step": max(steps, default=0.0),
        "abrupt_threshold": threshold,
        "abrupt_step_count": len(abrupt_indices),
        "abrupt_step_ratio": len(abrupt_indices) / max(1, len(values) - 1),
        "abrupt_step_indices_sample": abrupt_indices[:32],
        "extreme_abrupt_step_count": len(extreme_abrupt_indices),
        "extreme_abrupt_step_ratio": len(extreme_abrupt_indices)
        / max(1, len(values) - 1),
        "extreme_abrupt_step_indices_sample": extreme_abrupt_indices[:32],
        "non_finite_indices_sample": non_finite_indices[:32],
        "constant": value_range is not None and value_range <= 1e-8,
        "derivatives": {
            "velocity": velocity_metrics,
            "acceleration": acceleration_metrics,
            "jerk": jerk_metrics,
        },
    }
    return (
        metrics,
        non_finite_indices,
        abrupt_indices,
        extreme_abrupt_indices,
        sorted(set(acceleration_outliers + jerk_outliers)),
    )


def _quaternion_groups(feature: dict[str, Any], width: int) -> list[dict[str, Any]]:
    names = feature.get("names") if isinstance(feature, dict) else None
    if not isinstance(names, list) or len(names) < 4:
        return []
    source_key = str(feature.get("source_key") or "").lower()
    groups: dict[str, dict[str, int]] = {}
    for index, raw_name in enumerate(names[:width]):
        name = str(raw_name).lower()
        match = re.match(
            r"^(.*(?:orientation|quaternion|quat))[^a-z0-9]*([xyzw])$", name
        )
        if match:
            groups.setdefault(match.group(1), {})[match.group(2)] = index
            continue
        if source_key and any(token in source_key for token in ("orientation", "quaternion", "quat")):
            component = re.sub(r"[^a-z]", "", name)
            if component in {"x", "y", "z", "w"}:
                groups.setdefault(source_key, {})[component] = index
    return [
        {"name": name, "indices": [components[axis] for axis in "xyzw"]}
        for name, components in groups.items()
        if set(components) == set("xyzw")
    ]


def _angle_dimensions(feature: dict[str, Any], width: int) -> list[int]:
    names = feature.get("names") if isinstance(feature, dict) else None
    if not isinstance(names, list):
        return []
    result: list[int] = []
    for index, raw_name in enumerate(names[:width]):
        name = str(raw_name).lower()
        velocity = any(token in name for token in ("velocity", "speed", "twist"))
        angular = bool(
            re.search(r"(?:^|[^a-z])(yaw|pitch|roll)(?:[^a-z]|$)", name)
            or "euler" in name
            or name.endswith("_rad")
            or ".rad" in name
        )
        if angular and not velocity:
            result.append(index)
    return result


def _unwrap(values: list[float]) -> tuple[list[float], list[int]]:
    result = list(values)
    wrap_indices: list[int] = []
    previous: float | None = None
    for index, value in enumerate(result):
        if not math.isfinite(value):
            continue
        adjusted = value
        if previous is not None:
            while adjusted - previous > math.pi:
                adjusted -= 2.0 * math.pi
            while adjusted - previous < -math.pi:
                adjusted += 2.0 * math.pi
            if abs(adjusted - value) > 1e-9:
                wrap_indices.append(index)
        result[index] = adjusted
        previous = adjusted
    return result, wrap_indices


def _semantic_transform(
    rows: list[list[float]],
    feature: dict[str, Any],
    *,
    quaternion_norm_tolerance: float,
    quaternion_max_step_rad: float,
) -> tuple[list[list[float]], dict[str, Any], list[int], list[int], list[int]]:
    transformed = [list(row) for row in rows]
    width = max((len(row) for row in transformed), default=0)
    quaternion_reports: list[dict[str, Any]] = []
    invalid_quaternion_indices: list[int] = []
    norm_outlier_indices: list[int] = []
    rapid_quaternion_indices: list[int] = []
    for group in _quaternion_groups(feature, width):
        indices = group["indices"]
        previous: list[float] | None = None
        sign_flip_indices: list[int] = []
        norm_error_indices: list[int] = []
        geodesic_steps: list[float] = []
        for row_index, row in enumerate(transformed):
            if any(index >= len(row) for index in indices):
                invalid_quaternion_indices.append(row_index)
                continue
            quaternion = [row[index] for index in indices]
            if not all(math.isfinite(value) for value in quaternion):
                invalid_quaternion_indices.append(row_index)
                previous = None
                continue
            norm = math.sqrt(sum(value * value for value in quaternion))
            if norm <= 1e-12:
                invalid_quaternion_indices.append(row_index)
                previous = None
                continue
            if abs(norm - 1.0) > quaternion_norm_tolerance:
                norm_error_indices.append(row_index)
            normalized = [value / norm for value in quaternion]
            if previous is not None:
                dot = sum(left * right for left, right in zip(previous, normalized))
                if dot < 0:
                    normalized = [-value for value in normalized]
                    for source_index, value in zip(indices, normalized):
                        row[source_index] = value * norm
                    sign_flip_indices.append(row_index)
                    dot = -dot
                angle = 2.0 * math.acos(max(-1.0, min(1.0, abs(dot))))
                geodesic_steps.append(angle)
                if angle > quaternion_max_step_rad:
                    rapid_quaternion_indices.append(row_index)
            previous = normalized
        norm_outlier_indices.extend(norm_error_indices)
        quaternion_reports.append(
            {
                "name": group["name"],
                "indices": indices,
                "sign_flip_count": len(sign_flip_indices),
                "sign_flip_indices_sample": sign_flip_indices[:32],
                "norm_error_count": len(norm_error_indices),
                "norm_error_indices_sample": norm_error_indices[:32],
                "maximum_geodesic_step_rad": max(geodesic_steps, default=0.0),
                "rapid_geodesic_step_count": sum(
                    value > quaternion_max_step_rad for value in geodesic_steps
                ),
            }
        )

    angle_reports: list[dict[str, Any]] = []
    for dimension in _angle_dimensions(feature, width):
        values = [
            row[dimension] if dimension < len(row) else math.nan for row in transformed
        ]
        unwrapped, wrap_indices = _unwrap(values)
        for row, value in zip(transformed, unwrapped):
            if dimension < len(row):
                row[dimension] = value
        if wrap_indices:
            angle_reports.append(
                {
                    "dimension": dimension,
                    "unwrap_count": len(wrap_indices),
                    "unwrap_indices_sample": wrap_indices[:32],
                }
            )
    return (
        transformed,
        {"quaternions": quaternion_reports, "angle_unwrap": angle_reports},
        sorted(set(invalid_quaternion_indices)),
        sorted(set(norm_outlier_indices)),
        sorted(set(rapid_quaternion_indices)),
    )


def analyze_signal(
    values: list[Any],
    *,
    feature: dict[str, Any] | None = None,
    timestamps: list[float] | None = None,
    fps: float | None = None,
    source_key: str | None = None,
    domains: Iterable[str] = ("state", "action"),
    interval_padding_frames: int = 1,
    quaternion_norm_tolerance: float = 0.05,
    quaternion_max_step_rad: float = 1.5,
    extreme_abrupt_multiplier: float = 10.0,
) -> dict[str, Any]:
    rows = vectors(values)
    widths = [len(row) for row in rows]
    dimension_count = max(widths, default=0)
    feature_payload = dict(feature or {})
    feature_payload["source_key"] = source_key
    (
        transformed,
        semantic,
        invalid_quaternions,
        quaternion_norm_outliers,
        rapid_quaternions,
    ) = _semantic_transform(
        rows,
        feature_payload,
        quaternion_norm_tolerance=quaternion_norm_tolerance,
        quaternion_max_step_rad=quaternion_max_step_rad,
    )
    if timestamps is not None and len(timestamps) == len(rows):
        numeric_timestamps = [number(value) for value in timestamps]
    else:
        effective_fps = float(fps or 1.0)
        numeric_timestamps = [index / effective_fps for index in range(len(rows))]

    dimensions: list[dict[str, Any]] = []
    quaternion_dimensions = {
        int(dimension)
        for report in semantic.get("quaternions") or []
        for dimension in report.get("indices") or []
    }
    non_finite: list[int] = []
    abrupt: list[int] = []
    extreme_abrupt: list[int] = []
    derivative_outliers: list[int] = []
    for dimension in range(dimension_count):
        (
            metrics,
            non_finite_indices,
            abrupt_indices,
            extreme_abrupt_indices,
            derivative_indices,
        ) = _dimension_metrics(
            [
                row[dimension] if dimension < len(row) else math.nan
                for row in transformed
            ],
            dimension,
            numeric_timestamps,
            extreme_abrupt_multiplier,
        )
        if dimension in quaternion_dimensions:
            metrics["semantic_type"] = "quaternion_component"
            metrics["raw_component_abrupt_step_count"] = metrics[
                "abrupt_step_count"
            ]
            metrics["raw_component_abrupt_step_ratio"] = metrics[
                "abrupt_step_ratio"
            ]
            metrics["abrupt_step_count"] = 0
            metrics["abrupt_step_ratio"] = 0.0
            abrupt_indices = []
            extreme_abrupt_indices = []
            derivative_indices = []
        dimensions.append(metrics)
        non_finite.extend(non_finite_indices)
        abrupt.extend(abrupt_indices)
        extreme_abrupt.extend(extreme_abrupt_indices)
        derivative_outliers.extend(derivative_indices)

    value_count = len(rows) * dimension_count
    finite_count = sum(item["finite_count"] for item in dimensions)
    bad_intervals: list[dict[str, Any]] = []
    for indices, reason, severity in (
        (non_finite, "non_finite_signal", "hard"),
        (abrupt, "abrupt_signal_step", "soft"),
        (extreme_abrupt, "extreme_abrupt_signal_step", "hard"),
        (derivative_outliers, "acceleration_or_jerk_outlier", "soft"),
        (invalid_quaternions, "invalid_quaternion", "hard"),
        (quaternion_norm_outliers, "quaternion_norm_outlier", "soft"),
        (rapid_quaternions, "rapid_quaternion_rotation", "soft"),
    ):
        bad_intervals.extend(
            indices_to_intervals(
                indices,
                length=len(rows),
                reason=reason,
                domains=domains,
                severity=severity,
                timestamps=numeric_timestamps,
                padding=interval_padding_frames,
                source_key=source_key,
            )
        )
    return {
        "row_count": len(rows),
        "dimension_count": dimension_count,
        "row_width_consistent": len(set(widths)) <= 1,
        "observed_row_widths": sorted(set(widths)),
        "value_count": value_count,
        "finite_ratio": finite_count / max(1, value_count),
        "worst_dimension_finite_ratio": min(
            (item["finite_ratio"] for item in dimensions), default=0.0
        ),
        "maximum_abrupt_step_ratio": max(
            (item["abrupt_step_ratio"] for item in dimensions), default=0.0
        ),
        "maximum_extreme_abrupt_step_ratio": max(
            (item["extreme_abrupt_step_ratio"] for item in dimensions), default=0.0
        ),
        "constant_dimensions": [
            item["dimension"] for item in dimensions if item["constant"]
        ],
        "semantic": semantic,
        "bad_intervals": merge_bad_intervals(bad_intervals),
        "dimensions": dimensions,
    }
