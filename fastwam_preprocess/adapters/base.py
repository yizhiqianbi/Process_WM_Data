from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import TextIO
from typing import Any

from ..quality import QualityPolicy
from ..schema import ArtifactRecord, CameraRecord, EpisodeRecord
from ..utils import slug, stable_id, write_json, write_jsonl


@dataclass(slots=True)
class AdapterOptions:
    input_root: Path
    output_root: Path
    release: str = "local"
    min_frames: int = 81
    max_episodes: int | None = None
    verify_files: bool = False


class BaseAdapter(ABC):
    dataset_name = "unknown"

    def __init__(self, options: AdapterOptions):
        self.options = options
        self.policy = QualityPolicy(min_frames=options.min_frames)
        self.episodes: list[EpisodeRecord] = []
        self.artifacts: list[ArtifactRecord] = []
        self.blockers: list[str] = []
        self._episode_count = 0
        self._episode_handle: TextIO | None = None
        self._tiers = {"A": 0, "B": 0, "C": 0}
        self._candidates = {"A": 0, "B": 0, "C": 0}
        self._action_eligible = 0
        self._video_eligible = 0
        self._total_frames = 0

    @abstractmethod
    def scan(self) -> None:
        raise NotImplementedError

    def at_limit(self) -> bool:
        limit = self.options.max_episodes
        return limit is not None and self._episode_count >= limit

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def make_episode_id(
        self,
        *,
        embodiment: str,
        task_namespace: str,
        source_episode_id: str,
        release: str | None = None,
    ) -> str:
        parts = (
            self.dataset_name,
            release or self.options.release,
            slug(embodiment),
            slug(task_namespace),
            slug(source_episode_id),
        )
        return "/".join(parts)

    def add_episode(
        self,
        *,
        source_episode_id: str,
        source_uri: str,
        embodiment: str = "unknown",
        robot_type: str = "unknown",
        task_namespace: str = "unknown",
        tasks: list[str] | None = None,
        lineage_id: str | None = None,
        num_frames: int | None = None,
        fps: float | None = None,
        cameras: list[CameraRecord] | None = None,
        state_schema: dict[str, Any] | None = None,
        action_schema: dict[str, Any] | None = None,
        has_calibration: bool = False,
        has_depth: bool = False,
        complete: bool = True,
        action_verified: bool = False,
        passed_checks: list[str] | None = None,
        pending_checks: list[str] | None = None,
        warnings: list[str] | None = None,
        failures: list[str] | None = None,
        references: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        release: str | None = None,
    ) -> EpisodeRecord | None:
        if self.at_limit():
            return None
        camera_rows = list(cameras or [])
        state = dict(state_schema or {})
        action = dict(action_schema or {})
        quality = self.policy.evaluate(
            complete=complete,
            num_frames=num_frames,
            has_video=bool(camera_rows),
            has_state=bool(state),
            has_action=bool(action),
            action_verified=action_verified,
            visual_verified=False,
            passed_checks=passed_checks,
            pending_checks=pending_checks,
            warnings=warnings,
            failures=failures,
        )
        actual_release = release or self.options.release
        duration = None
        if num_frames is not None and fps and fps > 0:
            duration = max(0.0, (num_frames - 1) / fps)
        episode = EpisodeRecord(
            dataset=self.dataset_name,
            release=actual_release,
            source_episode_id=str(source_episode_id),
            global_episode_id=self.make_episode_id(
                embodiment=embodiment,
                task_namespace=task_namespace,
                source_episode_id=str(source_episode_id),
                release=actual_release,
            ),
            source_uri=source_uri,
            embodiment=embodiment,
            robot_type=robot_type,
            task_namespace=task_namespace,
            tasks=list(tasks or []),
            lineage_id=lineage_id,
            num_frames=num_frames,
            duration_s=duration,
            fps=fps,
            cameras=camera_rows,
            state_schema=state,
            action_schema=action,
            has_calibration=has_calibration,
            has_depth=has_depth,
            quality=quality,
            references=dict(references or {}),
            metadata=dict(metadata or {}),
        )
        self._episode_count += 1
        self._tiers[quality.tier] = self._tiers.get(quality.tier, 0) + 1
        self._candidates[quality.candidate_tier] = (
            self._candidates.get(quality.candidate_tier, 0) + 1
        )
        self._action_eligible += int(quality.action_eligible)
        self._video_eligible += int(quality.video_eligible)
        self._total_frames += num_frames or 0
        if self._episode_handle is None:
            self.episodes.append(episode)
        else:
            self._episode_handle.write(
                json.dumps(episode.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
        return episode

    def add_artifact(
        self,
        *,
        path: Path,
        kind: str,
        complete: bool,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        self.artifacts.append(
            ArtifactRecord(
                dataset=self.dataset_name,
                path=str(path),
                kind=kind,
                size_bytes=size,
                complete=complete,
                status=status,
                metadata=dict(metadata or {}),
            )
        )

    def run(self) -> dict[str, Any]:
        output = self.options.output_root
        output.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".episodes.jsonl.", dir=output)
        try:
            self._episode_handle = os.fdopen(fd, "w", encoding="utf-8")
            if not self.options.input_root.exists():
                self.blockers.append(f"input_root_missing:{self.options.input_root}")
            else:
                self.scan()
            self._episode_handle.flush()
            os.fsync(self._episode_handle.fileno())
            self._episode_handle.close()
            self._episode_handle = None
            os.replace(temp_name, output / "episodes.jsonl")
        except Exception:
            if self._episode_handle is not None:
                self._episode_handle.close()
                self._episode_handle = None
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

        write_jsonl(output / "artifacts.jsonl", (row.to_dict() for row in self.artifacts))

        summary = {
            "dataset": self.dataset_name,
            "release": self.options.release,
            "input_root": str(self.options.input_root),
            "output_root": str(output),
            "episode_count": self._episode_count,
            "artifact_count": len(self.artifacts),
            "quality_tiers": self._tiers,
            "candidate_tiers": self._candidates,
            "action_eligible_count": self._action_eligible,
            "video_eligible_count": self._video_eligible,
            "known_total_frames": self._total_frames,
            "min_frames": self.options.min_frames,
            "verify_files": self.options.verify_files,
            "max_episodes": self.options.max_episodes,
            "blockers": sorted(set(self.blockers)),
            "run_id": stable_id(
                self.dataset_name,
                self.options.release,
                self.options.input_root,
                self.options.min_frames,
                self.options.verify_files,
            ),
        }
        write_json(output / "summary.json", summary)
        return summary
