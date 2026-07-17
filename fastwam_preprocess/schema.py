from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "fastwam-preprocess-v1"


@dataclass(slots=True)
class CameraRecord:
    source_key: str
    role: str
    dtype: str = "video"
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str | None = None
    color_order: str = "unknown"
    has_depth: bool = False
    intrinsics_available: bool = False
    extrinsics_available: bool = False
    source_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QualityReport:
    tier: str
    candidate_tier: str
    score: float
    video_eligible: bool
    action_eligible: bool
    passed_checks: list[str] = field(default_factory=list)
    pending_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    component_scores: dict[str, float] = field(default_factory=dict)
    hard_blockers: list[str] = field(default_factory=list)
    soft_flags: list[str] = field(default_factory=list)
    bad_intervals: list[dict[str, Any]] = field(default_factory=list)
    sampling_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EpisodeRecord:
    dataset: str
    release: str
    source_episode_id: str
    global_episode_id: str
    source_uri: str
    embodiment: str = "unknown"
    robot_type: str = "unknown"
    task_namespace: str = "unknown"
    tasks: list[str] = field(default_factory=list)
    lineage_id: str | None = None
    num_frames: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    cameras: list[CameraRecord] = field(default_factory=list)
    state_schema: dict[str, Any] = field(default_factory=dict)
    action_schema: dict[str, Any] = field(default_factory=dict)
    has_calibration: bool = False
    has_depth: bool = False
    quality: QualityReport | None = None
    references: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(slots=True)
class ArtifactRecord:
    dataset: str
    path: str
    kind: str
    size_bytes: int | None
    complete: bool
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
