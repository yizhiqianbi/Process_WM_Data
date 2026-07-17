from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable

from .source import split_tar_uri


DEFAULT_VISUAL_THRESHOLDS = {
    "dark_value": 10,
    "bright_value": 245,
    "extreme_pixel_ratio": 0.98,
    "minimum_entropy": 0.08,
    "minimum_laplacian_variance": 0.0005,
    "freeze_mean_absolute_difference": 0.002,
    "freeze_dhash_distance": 1,
}
SAMPLE_WIDTH = 64
SAMPLE_HEIGHT = 64


def _binary(name: str) -> str:
    discovered = shutil.which(name)
    if discovered:
        return discovered
    beside_python = Path(sys.executable).with_name(name)
    if beside_python.is_file():
        return str(beside_python)
    return name


def _quantile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return float(ordered[left])
    alpha = position - left
    return ordered[left] * (1.0 - alpha) + ordered[right] * alpha


def _entropy(values: bytes) -> float:
    bins = [0] * 16
    for value in values:
        bins[min(15, value // 16)] += 1
    total = max(1, len(values))
    entropy = 0.0
    for count in bins:
        if count:
            probability = count / total
            entropy -= probability * math.log2(probability)
    return entropy / 4.0


def _laplacian_variance(values: bytes, width: int, height: int) -> float:
    laplacian: list[float] = []
    for y in range(1, height - 1):
        offset = y * width
        for x in range(1, width - 1):
            center = values[offset + x]
            value = (
                4 * center
                - values[offset + x - 1]
                - values[offset + x + 1]
                - values[offset - width + x]
                - values[offset + width + x]
            )
            laplacian.append(float(value))
    if len(laplacian) < 2:
        return 0.0
    return statistics.pvariance(laplacian) / (255.0 * 255.0)


def _dhash(values: bytes, width: int, height: int) -> int:
    result = 0
    for y_index in range(8):
        y = min(height - 1, int(round(y_index * (height - 1) / 7)))
        for x_index in range(8):
            left_x = min(width - 1, int(round(x_index * (width - 1) / 8)))
            right_x = min(width - 1, int(round((x_index + 1) * (width - 1) / 8)))
            result = (result << 1) | int(
                values[y * width + left_x] > values[y * width + right_x]
            )
    return result


def _frame_metrics(
    values: bytes, *, width: int, height: int, thresholds: dict[str, Any]
) -> dict[str, Any]:
    if len(values) != width * height:
        raise ValueError(f"Expected {width * height} grayscale bytes, got {len(values)}")
    dark_value = int(thresholds["dark_value"])
    bright_value = int(thresholds["bright_value"])
    total = len(values)
    mean = sum(values) / max(1, total)
    metrics = {
        "mean_luma": mean,
        "p01_luma": _quantile(list(values), 0.01),
        "p50_luma": _quantile(list(values), 0.50),
        "p99_luma": _quantile(list(values), 0.99),
        "dark_pixel_ratio": sum(value <= dark_value for value in values) / total,
        "bright_pixel_ratio": sum(value >= bright_value for value in values) / total,
        "entropy": _entropy(values),
        "laplacian_variance": _laplacian_variance(values, width, height),
        "dhash": f"{_dhash(values, width, height):016x}",
    }
    flags: list[str] = []
    if metrics["dark_pixel_ratio"] >= float(thresholds["extreme_pixel_ratio"]):
        flags.append("extreme_dark")
    if metrics["bright_pixel_ratio"] >= float(thresholds["extreme_pixel_ratio"]):
        flags.append("extreme_bright")
    if metrics["entropy"] < float(thresholds["minimum_entropy"]):
        flags.append("low_entropy")
    if metrics["laplacian_variance"] < float(
        thresholds["minimum_laplacian_variance"]
    ):
        flags.append("low_sharpness")
    metrics["flags"] = flags
    return metrics


def _sample_interval(
    timestamps: list[float], index: int, *, reason: str, severity: str
) -> dict[str, Any]:
    timestamp = timestamps[index]
    left = (
        0.0
        if index == 0
        else (timestamps[index - 1] + timestamp) / 2.0
    )
    if index + 1 < len(timestamps):
        right = (timestamp + timestamps[index + 1]) / 2.0
    elif index:
        right = timestamp + (timestamp - timestamps[index - 1]) / 2.0
    else:
        right = timestamp + 1e-3
    return {
        "timeline": "source_video",
        "start": index,
        "stop_exclusive": index + 1,
        "start_s": max(0.0, left),
        "stop_s": max(left + 1e-6, right),
        "reason": reason,
        "domains": ["video"],
        "severity": severity,
    }


def analyze_grayscale_frames(
    frames: Iterable[bytes],
    *,
    timestamps: Iterable[float] | None = None,
    width: int = SAMPLE_WIDTH,
    height: int = SAMPLE_HEIGHT,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute dependency-free sparse visual metrics for decoded grayscale frames."""
    threshold_values = {**DEFAULT_VISUAL_THRESHOLDS, **(thresholds or {})}
    frame_values = list(frames)
    sample_times = [float(value) for value in timestamps or range(len(frame_values))]
    if len(sample_times) != len(frame_values):
        raise ValueError("timestamps and frames must have the same length")
    metrics = [
        {
            "sample_index": index,
            "timestamp_s": sample_times[index],
            **_frame_metrics(
                frame,
                width=width,
                height=height,
                thresholds=threshold_values,
            ),
        }
        for index, frame in enumerate(frame_values)
    ]
    pairs: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(frame_values, frame_values[1:])):
        mean_difference = sum(abs(a - b) for a, b in zip(left, right)) / (
            max(1, len(left)) * 255.0
        )
        left_hash = int(metrics[index]["dhash"], 16)
        right_hash = int(metrics[index + 1]["dhash"], 16)
        hash_distance = (left_hash ^ right_hash).bit_count()
        frozen = (
            mean_difference
            <= float(threshold_values["freeze_mean_absolute_difference"])
            and hash_distance <= int(threshold_values["freeze_dhash_distance"])
        )
        pairs.append(
            {
                "left_sample_index": index,
                "right_sample_index": index + 1,
                "start_s": sample_times[index],
                "stop_s": sample_times[index + 1],
                "mean_absolute_difference": mean_difference,
                "dhash_distance": hash_distance,
                "frozen": frozen,
            }
        )

    bad_intervals: list[dict[str, Any]] = []
    extreme_by_index = {
        index: {
            flag
            for flag in metric["flags"]
            if flag in {"extreme_dark", "extreme_bright"}
        }
        for index, metric in enumerate(metrics)
    }
    for index, metric in enumerate(metrics):
        for flag in metric["flags"]:
            adjacent_same = bool(
                extreme_by_index.get(index, set())
                and (
                    extreme_by_index[index].intersection(
                        extreme_by_index.get(index - 1, set())
                    )
                    or extreme_by_index[index].intersection(
                        extreme_by_index.get(index + 1, set())
                    )
                )
            )
            severity = (
                "hard"
                if flag in {"extreme_dark", "extreme_bright"} and adjacent_same
                else "soft"
            )
            bad_intervals.append(
                _sample_interval(sample_times, index, reason=flag, severity=severity)
            )
    for pair in pairs:
        if pair["frozen"]:
            bad_intervals.append(
                {
                    "timeline": "source_video",
                    "start": pair["left_sample_index"],
                    "stop_exclusive": pair["right_sample_index"] + 1,
                    "start_s": pair["start_s"],
                    "stop_s": pair["stop_s"],
                    "reason": "possible_frozen_video",
                    "domains": ["video"],
                    "severity": "soft",
                }
            )

    all_extreme = bool(metrics) and all(
        extreme_by_index[index] for index in range(len(metrics))
    )
    all_frozen = bool(pairs) and all(pair["frozen"] for pair in pairs)
    flags = sorted(
        {
            flag
            for metric in metrics
            for flag in metric["flags"]
        }
        | ({"possible_frozen_video"} if any(pair["frozen"] for pair in pairs) else set())
    )
    status = "failed" if all_extreme else ("warning" if flags else "passed")
    fingerprint_source = "|".join(metric["dhash"] for metric in metrics)
    return {
        "status": status,
        "sampled_frame_count": len(metrics),
        "sample_width": width,
        "sample_height": height,
        "frames": metrics,
        "frame_pairs": pairs,
        "flags": flags,
        "all_frames_extreme": all_extreme,
        "all_pairs_frozen": all_frozen,
        "bad_intervals": bad_intervals,
        "visual_fingerprint_sha256": (
            hashlib.sha256(fingerprint_source.encode("ascii")).hexdigest()
            if metrics
            else None
        ),
        "aggregate": {
            "median_mean_luma": statistics.median(
                (metric["mean_luma"] for metric in metrics)
            )
            if metrics
            else None,
            "minimum_entropy": min(
                (metric["entropy"] for metric in metrics), default=None
            ),
            "minimum_laplacian_variance": min(
                (metric["laplacian_variance"] for metric in metrics), default=None
            ),
            "median_pair_difference": statistics.median(
                pair["mean_absolute_difference"] for pair in pairs
            )
            if pairs
            else None,
        },
    }


def _rate(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    text = str(value)
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / float(denominator)
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _duration(stream: dict[str, Any], payload: dict[str, Any]) -> float | None:
    for value in (stream.get("duration"), (payload.get("format") or {}).get("duration")):
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    frames = stream.get("nb_frames")
    fps = _rate(stream.get("avg_frame_rate"))
    try:
        if frames and fps:
            return float(frames) / fps
    except (TypeError, ValueError):
        pass
    return None


def _sample_video_path(
    path: Path,
    *,
    duration_s: float | None,
    sample_frames: int,
    thresholds: dict[str, Any] | None,
) -> dict[str, Any]:
    if sample_frames <= 0:
        return {"status": "pending", "reason": "sparse_sampling_disabled"}
    if duration_s is None or duration_s <= 0:
        return {"status": "pending", "reason": "video_duration_unknown"}
    rate = sample_frames / duration_s
    command = [
        _binary("ffmpeg"),
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={rate:.12g},scale={SAMPLE_WIDTH}:{SAMPLE_HEIGHT},format=gray",
        "-frames:v",
        str(sample_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "reason": "sparse_decode_error", "error": str(exc)}
    if result.returncode != 0:
        return {
            "status": "failed",
            "reason": "sparse_decode_failed",
            "error": result.stderr.decode("utf-8", errors="replace")[-1000:],
        }
    frame_size = SAMPLE_WIDTH * SAMPLE_HEIGHT
    count = len(result.stdout) // frame_size
    frames = [
        result.stdout[index * frame_size : (index + 1) * frame_size]
        for index in range(count)
    ]
    timestamps = (
        [index * duration_s / max(1, count - 1) for index in range(count)]
        if count > 1
        else [0.0] * count
    )
    if not frames:
        return {"status": "failed", "reason": "sparse_decode_returned_no_frames"}
    return analyze_grayscale_frames(
        frames, timestamps=timestamps, thresholds=thresholds
    )


def _audit_video_path(
    path: Path,
    *,
    decode: bool,
    sample_frames: int,
    thresholds: dict[str, Any] | None,
) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "failed", "reason": "video_missing"}
    command = [
        _binary("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "reason": "ffprobe_error", "error": str(exc)}
    if result.returncode != 0:
        return {
            "status": "failed",
            "reason": "ffprobe_failed",
            "error": result.stderr[-1000:],
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "failed", "reason": "ffprobe_invalid_json", "error": str(exc)}
    streams = payload.get("streams") or []
    if not streams:
        return {"status": "failed", "reason": "no_video_stream"}
    stream = streams[0]
    duration_s = _duration(stream, payload)
    report: dict[str, Any] = {
        "status": "passed",
        "stream": stream,
        "duration_s": duration_s,
        "sparse_visual": _sample_video_path(
            path,
            duration_s=duration_s,
            sample_frames=sample_frames,
            thresholds=thresholds,
        ),
    }
    if report["sparse_visual"].get("status") == "failed":
        report["status"] = "failed"
        report["reason"] = "sparse_visual_decode_failed"
    if decode:
        decode_command = [
            _binary("ffmpeg"),
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ]
        try:
            decoded = subprocess.run(
                decode_command, check=False, capture_output=True, text=True, timeout=600
            )
            report["decode_status"] = "passed" if decoded.returncode == 0 else "failed"
            if decoded.returncode != 0:
                report["decode_error"] = decoded.stderr[-1000:]
                report["status"] = "failed"
        except (OSError, subprocess.TimeoutExpired) as exc:
            report.update(status="failed", decode_status="failed", decode_error=str(exc))
    return report


def audit_tar_video_members(
    archive_path: Path,
    member_names: Iterable[str],
    *,
    decode: bool = False,
    sample_frames: int = 0,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Audit several members while paying the archive indexing cost only once."""
    names = list(dict.fromkeys(str(value) for value in member_names))
    reports: dict[str, dict[str, Any]] = {}
    if not archive_path.is_file():
        return {
            name: {"status": "failed", "reason": "video_archive_missing"}
            for name in names
        }
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member_name in names:
                if Path(member_name).suffix.lower() not in {
                    ".mp4",
                    ".mkv",
                    ".avi",
                    ".mov",
                }:
                    reports[member_name] = {
                        "status": "pending",
                        "reason": "unsupported_archive_visual_member",
                    }
                    continue
                temp_path: Path | None = None
                try:
                    member = archive.extractfile(member_name)
                    if member is None:
                        raise FileNotFoundError(member_name)
                    with tempfile.NamedTemporaryFile(
                        prefix="fastwam_visual_",
                        suffix=Path(member_name).suffix,
                        delete=False,
                    ) as temporary:
                        shutil.copyfileobj(member, temporary)
                        temp_path = Path(temporary.name)
                    report = _audit_video_path(
                        temp_path,
                        decode=decode,
                        sample_frames=sample_frames,
                        thresholds=thresholds,
                    )
                    report.update(
                        storage="tar_member",
                        archive=str(archive_path),
                        member=member_name,
                    )
                    reports[member_name] = report
                except (OSError, KeyError, FileNotFoundError) as exc:
                    reports[member_name] = {
                        "status": "failed",
                        "reason": "video_archive_member_read_failed",
                        "error": str(exc),
                    }
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
    except (OSError, tarfile.TarError) as exc:
        return {
            name: {
                "status": "failed",
                "reason": "video_archive_read_failed",
                "error": str(exc),
            }
            for name in names
        }
    return reports


def _sample_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count == 1 or length == 1:
        return [0]
    return sorted(
        {
            int(round(index * (length - 1) / (count - 1)))
            for index in range(min(length, count))
        }
    )


def _to_grayscale(value: Any) -> bytes:
    from PIL import Image
    import numpy as np

    if isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == np.uint8:
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray, memoryview)):
        image = Image.open(io.BytesIO(bytes(value)))
    else:
        array = np.asarray(value)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
    return image.convert("L").resize((SAMPLE_WIDTH, SAMPLE_HEIGHT)).tobytes()


def _audit_hdf5(
    uri: str,
    *,
    sample_frames: int,
    thresholds: dict[str, Any] | None,
    fps: float | None,
) -> dict[str, Any]:
    descriptor = uri[len("hdf5://") :]
    if "#" not in descriptor:
        return {"status": "failed", "reason": "invalid_hdf5_uri"}
    path_text, dataset_text = descriptor.split("#", 1)
    dataset_name = dataset_text.split(";", 1)[0]
    path = Path(path_text)
    if not path.is_file():
        return {"status": "failed", "reason": "hdf5_missing"}
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            dataset = handle[dataset_name]
            length = int(dataset.shape[0])
            indices = _sample_indices(length, sample_frames)
            frames = [_to_grayscale(dataset[index]) for index in indices]
    except Exception as exc:
        return {"status": "failed", "reason": "hdf5_visual_decode_failed", "error": str(exc)}
    visual = analyze_grayscale_frames(
        frames,
        timestamps=[index / float(fps or 1.0) for index in indices],
        thresholds=thresholds,
    )
    return {
        "status": "passed",
        "storage": "hdf5_embedded",
        "frame_count": length,
        "sparse_visual": visual,
    }


def _audit_oxe_pickle(
    uri: str,
    *,
    sample_frames: int,
    thresholds: dict[str, Any] | None,
    fps: float | None,
) -> dict[str, Any]:
    descriptor = uri[len("oxe-pickle://") :]
    if "!" not in descriptor or "#" not in descriptor:
        return {"status": "failed", "reason": "invalid_oxe_pickle_uri"}
    archive_text, remainder = descriptor.split("!", 1)
    member_name, source_key = remainder.split("#", 1)
    observation_key = source_key.removeprefix("observation.")
    try:
        from .native_readers import load_oxe_episode

        payload = load_oxe_episode(Path(archive_text), member_name)
        steps = payload["steps"]
        indices = _sample_indices(len(steps), sample_frames)
        frames = [
            _to_grayscale((steps[index].get("observation") or {})[observation_key])
            for index in indices
        ]
    except Exception as exc:
        return {"status": "failed", "reason": "oxe_visual_decode_failed", "error": str(exc)}
    return {
        "status": "passed",
        "storage": "oxe_pickle_embedded",
        "frame_count": len(steps),
        "sparse_visual": analyze_grayscale_frames(
            frames,
            timestamps=[index / float(fps or 20.0) for index in indices],
            thresholds=thresholds,
        ),
    }


def audit_local_video(
    uri: str,
    *,
    decode: bool = False,
    sample_frames: int = 0,
    thresholds: dict[str, Any] | None = None,
    source_key: str | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    """Audit local, tar-contained, HDF5, or OXE visual observations."""
    if uri.startswith("hdf5://"):
        report = _audit_hdf5(
            uri, sample_frames=sample_frames, thresholds=thresholds, fps=fps
        )
    elif uri.startswith("oxe-pickle://"):
        report = _audit_oxe_pickle(
            uri, sample_frames=sample_frames, thresholds=thresholds, fps=fps
        )
    elif (tar_source := split_tar_uri(uri)) is not None:
        archive_path, member_name = tar_source
        report = audit_tar_video_members(
            archive_path,
            [member_name],
            decode=decode,
            sample_frames=sample_frames,
            thresholds=thresholds,
        )[member_name]
    elif "://" in uri:
        report = {"status": "pending", "reason": "unsupported_visual_uri"}
    else:
        report = _audit_video_path(
            Path(uri),
            decode=decode,
            sample_frames=sample_frames,
            thresholds=thresholds,
        )
        report["storage"] = "file"
    if source_key:
        report["source_key"] = source_key
        for interval in (report.get("sparse_visual") or {}).get("bad_intervals") or []:
            interval["camera_key"] = source_key
    return report
