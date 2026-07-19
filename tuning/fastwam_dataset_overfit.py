from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetProbe:
    dataset_index: int
    source_episode_id: str
    case_id: str
    window_start: int
    mode: str
    split: str


def _expand_valid_starts(spec: Any) -> list[int]:
    if isinstance(spec, list):
        starts = [int(value) for value in spec]
    elif isinstance(spec, dict):
        start = int(spec.get("start", 0))
        stop = int(spec["stop_exclusive"])
        stride = int(spec["stride"])
        if start < 0 or stop < start or stride <= 0:
            raise ValueError(f"Invalid valid_starts range: {spec}")
        starts = list(range(start, stop, stride))
        if spec.get("count") is not None and int(spec["count"]) != len(starts):
            raise ValueError(f"valid_starts count does not match range: {spec}")
    else:
        raise TypeError(f"Unsupported valid_starts type: {type(spec)}")
    if any(value < 0 for value in starts):
        raise ValueError("valid_starts must be non-negative")
    return starts


def enumerate_dataset_windows(
    manifest: Path,
    *,
    splits: Iterable[str],
    modes: Iterable[str],
    quality_tiers: Iterable[str] = ("A",),
) -> list[DatasetProbe]:
    split_set = {str(value) for value in splits}
    mode_set = {str(value) for value in modes}
    tier_set = {str(value) for value in quality_tiers}
    windows: list[DatasetProbe] = []
    with manifest.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            if case.get("schema_version") != "fastwam-training-case-v1":
                raise ValueError(
                    f"Unsupported training case schema at {manifest}:{line_number}: "
                    f"{case.get('schema_version')!r}"
                )
            split = str(case.get("split"))
            mode = str((case.get("training") or {}).get("mode"))
            tier = str((case.get("quality") or {}).get("tier"))
            if split not in split_set or mode not in mode_set or tier not in tier_set:
                continue
            for start in _expand_valid_starts((case.get("sampling") or {}).get("valid_starts")):
                windows.append(
                    DatasetProbe(
                        dataset_index=len(windows),
                        source_episode_id=str(case["source_episode_id"]),
                        case_id=str(case["case_id"]),
                        window_start=start,
                        mode=mode,
                        split=split,
                    )
                )
    if not windows:
        raise ValueError(f"No windows remain after filtering {manifest}")
    return windows


def _evenly_spaced(values: list[DatasetProbe], count: int) -> list[DatasetProbe]:
    if count <= 0:
        raise ValueError("probe count must be positive")
    if count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    positions = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[position] for position in positions]


def select_episode_probes(
    windows: Iterable[DatasetProbe], *, max_episodes: int | None = None
) -> list[DatasetProbe]:
    by_episode: OrderedDict[str, list[DatasetProbe]] = OrderedDict()
    for window in windows:
        by_episode.setdefault(window.source_episode_id, []).append(window)

    probes: list[DatasetProbe] = []
    for episode_windows in by_episode.values():
        candidates = [
            window for window in episode_windows if window.mode == "joint_video_action"
        ] or list(episode_windows)
        memory_ready = [window for window in candidates if window.window_start >= 40]
        if memory_ready:
            candidates = memory_ready
        probes.append(candidates[len(candidates) // 2])
    if max_episodes is not None:
        probes = _evenly_spaced(probes, int(max_episodes))
    return probes


def build_dataset_overfit_plan(
    manifest: Path,
    *,
    splits: Iterable[str] = ("train", "validation"),
    modes: Iterable[str] = ("joint_video_action", "video_only"),
    quality_tiers: Iterable[str] = ("A",),
    training_probe_count: int = 8,
) -> dict[str, Any]:
    windows = enumerate_dataset_windows(
        manifest,
        splits=splits,
        modes=modes,
        quality_tiers=quality_tiers,
    )
    all_episode_probes = select_episode_probes(windows)
    training_probes = select_episode_probes(windows, max_episodes=training_probe_count)
    mode_counts = Counter(window.mode for window in windows)
    split_counts = Counter(window.split for window in windows)
    return {
        "schema_version": "fastwam-dataset-overfit-plan-v1",
        "manifest": str(manifest.expanduser().resolve()),
        "window_count": len(windows),
        "episode_count": len(all_episode_probes),
        "window_counts_by_mode": dict(sorted(mode_counts.items())),
        "window_counts_by_split": dict(sorted(split_counts.items())),
        "training_probe_indices": [probe.dataset_index for probe in training_probes],
        "training_probes": [asdict(probe) for probe in training_probes],
        "all_episode_probe_indices": [
            probe.dataset_index for probe in all_episode_probes
        ],
        "all_episode_probes": [asdict(probe) for probe in all_episode_probes],
    }
