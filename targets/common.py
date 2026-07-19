from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import yaml

from fastwam_preprocess.utils import AtomicJsonlWriter, iter_jsonl, read_json, write_json


class TargetPreparationError(ValueError):
    """Raised when source data cannot satisfy a model target contract."""


def load_target_profile(path: Path, target: str, profile_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if document.get("target") != target:
        raise TargetPreparationError(
            f"target profile file declares {document.get('target')!r}, expected {target!r}"
        )
    profiles = document.get("profiles") or {}
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(profiles)) or "none"
        raise TargetPreparationError(
            f"unknown {target} profile {profile_name!r}; available profiles: {available}"
        )
    return document, dict(profile)


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> None:
    with AtomicJsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)


def _format_path(template: str, episode_index: int, chunks_size: int, **extra: Any) -> str:
    values = {
        "episode_index": episode_index,
        "episode_chunk": episode_index // max(1, chunks_size),
        **extra,
    }
    return template.format(**values)


class LeRobotV2Dataset:
    """Small strict reader for the LeRobot v2 metadata used by both targets."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.meta_root = self.root / "meta"
        self.info_path = self.meta_root / "info.json"
        self.episodes_path = self.meta_root / "episodes.jsonl"
        if not self.info_path.is_file():
            raise TargetPreparationError(f"missing LeRobot metadata: {self.info_path}")
        if not self.episodes_path.is_file():
            raise TargetPreparationError(f"missing LeRobot metadata: {self.episodes_path}")
        self.info = read_json(self.info_path)
        codebase_version = str(self.info.get("codebase_version") or "")
        if codebase_version and not codebase_version.startswith("v2"):
            raise TargetPreparationError(
                "model target preparation requires LeRobot v2; "
                f"found codebase_version={codebase_version!r}"
            )
        self.episodes = list(iter_jsonl(self.episodes_path))
        self.features = self.info.get("features") or {}
        self.chunks_size = int(self.info.get("chunks_size") or 1000)
        self.data_template = str(
            self.info.get("data_path")
            or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        self.video_template = str(
            self.info.get("video_path")
            or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        self._episodes_by_index: dict[int, dict[str, Any]] = {}
        for row in self.episodes:
            index = int(row.get("episode_index", -1))
            if index < 0 or index in self._episodes_by_index:
                raise TargetPreparationError("episode indices must be unique non-negative integers")
            self._episodes_by_index[index] = row
        declared = self.info.get("total_episodes")
        if declared is not None and int(declared) != len(self.episodes):
            raise TargetPreparationError(
                f"info total_episodes={declared} does not match episodes.jsonl={len(self.episodes)}"
            )

    @property
    def episode_indices(self) -> list[int]:
        return sorted(self._episodes_by_index)

    def episode(self, episode_index: int) -> dict[str, Any]:
        return self._episodes_by_index[episode_index]

    def data_path(self, episode_index: int) -> Path:
        return self.root / _format_path(
            self.data_template, episode_index, self.chunks_size
        )

    def relative_data_path(self, episode_index: int) -> Path:
        return Path(_format_path(self.data_template, episode_index, self.chunks_size))

    def video_path(self, episode_index: int, video_key: str) -> Path:
        return self.root / _format_path(
            self.video_template,
            episode_index,
            self.chunks_size,
            video_key=video_key,
        )

    def feature_width(self, key: str) -> int | None:
        feature = self.features.get(key) or {}
        shape = feature.get("shape")
        if isinstance(shape, list) and shape:
            return int(shape[-1])
        return None

    def video_keys(self) -> list[str]:
        return sorted(
            key
            for key, feature in self.features.items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        )

    def read_column(self, episode_index: int, column: str) -> np.ndarray:
        path = self.data_path(episode_index)
        if not path.is_file():
            raise TargetPreparationError(f"missing episode parquet: {path}")
        try:
            table = pq.read_table(path, columns=[column])
        except Exception as exc:
            raise TargetPreparationError(
                f"cannot read {column!r} from {path}: {exc}"
            ) from exc
        values = table[column].to_pylist()
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TargetPreparationError(
                f"column {column!r} in {path} is not a fixed numeric vector"
            ) from exc
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or not np.isfinite(array).all():
            raise TargetPreparationError(
                f"column {column!r} in {path} must be finite rank-2 data"
            )
        return array

    def episode_length(self, episode_index: int) -> int:
        declared = self.episode(episode_index).get("length")
        parquet_rows = pq.read_metadata(self.data_path(episode_index)).num_rows
        if declared is not None and int(declared) != parquet_rows:
            raise TargetPreparationError(
                f"episode {episode_index} length={declared} but parquet has {parquet_rows} rows"
            )
        return parquet_rows

    def validate_cameras(self, camera_keys: list[str], *, verify_files: bool) -> None:
        if not camera_keys:
            raise TargetPreparationError("at least one camera key is required")
        available = set(self.video_keys())
        missing_features = [key for key in camera_keys if key not in available]
        if missing_features:
            raise TargetPreparationError(
                f"camera features missing from info.json: {missing_features}"
            )
        if verify_files:
            missing_files = [
                str(self.video_path(index, key))
                for index in self.episode_indices
                for key in camera_keys
                if not self.video_path(index, key).is_file()
            ]
            if missing_files:
                sample = ", ".join(missing_files[:3])
                raise TargetPreparationError(
                    f"missing {len(missing_files)} referenced video files; first: {sample}"
                )


class StreamingStats:
    """Exact moments plus deterministic reservoir quantiles for wide datasets."""

    def __init__(self, width: int, *, max_quantile_rows: int = 1_000_000, seed: int = 0):
        if width <= 0 or max_quantile_rows <= 0:
            raise ValueError("width and max_quantile_rows must be positive")
        self.width = width
        self.max_quantile_rows = max_quantile_rows
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.total_sq = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.maximum = np.full(width, -np.inf, dtype=np.float64)
        self.sample = np.empty((max_quantile_rows, width), dtype=np.float64)
        self.sample_count = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.width:
            raise TargetPreparationError(
                f"stats expected width {self.width}, got shape {values.shape}"
            )
        if not np.isfinite(values).all():
            raise TargetPreparationError("stats input contains NaN or Inf")
        self.total += values.sum(axis=0)
        self.total_sq += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        for row in values:
            self.count += 1
            if self.sample_count < self.max_quantile_rows:
                self.sample[self.sample_count] = row
                self.sample_count += 1
            else:
                replacement = int(self.rng.integers(0, self.count))
                if replacement < self.max_quantile_rows:
                    self.sample[replacement] = row

    def finish(self) -> dict[str, Any]:
        if self.count == 0:
            raise TargetPreparationError("cannot compute statistics from zero rows")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        sample = self.sample[: self.sample_count]
        return {
            "count": self.count,
            "quantile_sample_count": self.sample_count,
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "q01": np.quantile(sample, 0.01, axis=0).tolist(),
            "q99": np.quantile(sample, 0.99, axis=0).tolist(),
        }


def compute_column_stats(
    dataset: LeRobotV2Dataset,
    column: str,
    *,
    indices: list[int] | None = None,
    source_indices: list[int] | None = None,
    max_quantile_rows: int = 1_000_000,
) -> dict[str, Any]:
    selected = dataset.episode_indices if indices is None else indices
    accumulator: StreamingStats | None = None
    for episode_index in selected:
        values = dataset.read_column(episode_index, column)
        if source_indices is not None:
            if source_indices and max(source_indices) >= values.shape[1]:
                raise TargetPreparationError(
                    f"episode {episode_index} {column} width {values.shape[1]} cannot select {source_indices}"
                )
            values = values[:, source_indices]
        if accumulator is None:
            accumulator = StreamingStats(
                values.shape[1], max_quantile_rows=max_quantile_rows
            )
        accumulator.update(values)
    if accumulator is None:
        raise TargetPreparationError("dataset contains no episodes")
    return accumulator.finish()


def _clone_tree(source: Path, destination: Path, mode: str) -> None:
    if not source.exists():
        raise TargetPreparationError(f"source directory missing: {source}")
    if mode == "symlink":
        destination.symlink_to(source.resolve(), target_is_directory=True)
    elif mode == "hardlink":
        shutil.copytree(source, destination, copy_function=os.link)
    elif mode == "copy":
        shutil.copytree(source, destination)
    else:
        raise TargetPreparationError(f"unsupported link mode: {mode}")


@contextmanager
def atomic_overlay(
    source_root: Path,
    output_root: Path,
    *,
    link_mode: str,
    include_data: bool = True,
    include_videos: bool = True,
) -> Iterator[Path]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise TargetPreparationError(f"output already exists: {output_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise TargetPreparationError("output must not be the source or a child of the source")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        shutil.copytree(source_root / "meta", staging / "meta")
        if include_data:
            _clone_tree(source_root / "data", staging / "data", link_mode)
        if include_videos:
            _clone_tree(source_root / "videos", staging / "videos", link_mode)
        yield staging
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_range_mapping(
    mapping: dict[str, Any], *, width: int, label: str
) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for key, bounds in mapping.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise TargetPreparationError(f"{label}.{key} must be [start, end]")
        start, end = int(bounds[0]), int(bounds[1])
        if start < 0 or end <= start or end > width:
            raise TargetPreparationError(
                f"{label}.{key} range [{start}, {end}) is outside width {width}"
            )
        result[str(key)] = (start, end)
    if not result:
        raise TargetPreparationError(f"{label} mapping cannot be empty")
    return result


def finite_json_numbers(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_json_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


__all__ = [
    "LeRobotV2Dataset",
    "StreamingStats",
    "TargetPreparationError",
    "atomic_overlay",
    "compute_column_stats",
    "file_sha256",
    "finite_json_numbers",
    "load_target_profile",
    "read_json",
    "stable_sha256",
    "validate_range_mapping",
    "write_json",
    "write_jsonl",
]
