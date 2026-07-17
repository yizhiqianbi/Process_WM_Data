from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

from ..canonical import infer_canonical_mapping
from ..schema import CameraRecord
from ..utils import is_partial_path
from .base import BaseAdapter
from .lerobot import camera_records, feature_schemas, format_lerobot_path


def _read_json_member(archive: tarfile.TarFile, member_name: str) -> dict[str, Any]:
    handle = archive.extractfile(member_name)
    if handle is None:
        raise ValueError(f"Cannot read archive member {member_name}")
    payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {member_name}")
    return payload


class GalaxeaAdapter(BaseAdapter):
    dataset_name = "galaxea"

    def scan(self) -> None:
        lerobot_root = self.options.input_root / "lerobot"
        archives = sorted(lerobot_root.glob("*.tar.gz*"))
        if not archives:
            self.blockers.append("no_galaxea_lerobot_archives_found")
            return
        for path in archives:
            if self.at_limit():
                break
            if not path.name.endswith(".tar.gz") or is_partial_path(path):
                self.add_artifact(
                    path=path,
                    kind="galaxea_lerobot_archive",
                    complete=False,
                    status="partial_download",
                )
                continue
            self._scan_archive(path)

    def _scan_archive(self, path: Path) -> None:
        try:
            with tarfile.open(path, mode="r:gz") as archive:
                members = {member.name for member in archive if member.isfile()}
                info_names = [name for name in members if name.endswith("/meta/info.json")]
                episode_names = [
                    name for name in members if name.endswith("/meta/episodes.jsonl")
                ]
                if len(info_names) != 1 or len(episode_names) != 1:
                    raise ValueError("archive must contain exactly one LeRobot metadata root")
                info_name = info_names[0]
                episodes_name = episode_names[0]
                repo_prefix = info_name.removesuffix("meta/info.json")
                info = _read_json_member(archive, info_name)
                episode_handle = archive.extractfile(episodes_name)
                if episode_handle is None:
                    raise ValueError(f"Cannot read {episodes_name}")
                episode_rows = [
                    json.loads(line)
                    for raw_line in episode_handle
                    if (line := raw_line.decode("utf-8").strip())
                ]
        except (OSError, tarfile.TarError, ValueError, json.JSONDecodeError) as exc:
            self.add_artifact(
                path=path,
                kind="galaxea_lerobot_archive",
                complete=False,
                status="invalid_archive",
                metadata={"error": str(exc)},
            )
            return

        features = info.get("features")
        features = features if isinstance(features, dict) else {}
        cameras = camera_records(features)
        state_schema, action_schema = feature_schemas(features)
        canonical_mapping = {
            "state": infer_canonical_mapping(state_schema, kind="state"),
            "action": infer_canonical_mapping(action_schema, kind="action"),
        }
        fps = float(info["fps"]) if info.get("fps") is not None else None
        chunk_size = int(info.get("chunks_size") or 1000)
        data_template = str(
            info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
        )
        video_template = str(
            info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
        )
        robot_type = str(info.get("robot_type") or "galaxea_r1")
        namespace = Path(repo_prefix.rstrip("/")).name
        self.add_artifact(
            path=path,
            kind="galaxea_lerobot_archive",
            complete=True,
            status="metadata_ready",
            metadata={
                "episode_count": len(episode_rows),
                "camera_count": len(cameras),
                "archive_root": repo_prefix,
            },
        )

        for row in episode_rows:
            if self.at_limit():
                break
            episode_index = int(row.get("episode_index", self.episode_count))
            length_value = row.get("length")
            num_frames = int(length_value) if length_value is not None else None
            data_rel = format_lerobot_path(data_template, episode_index, chunk_size)
            data_member = f"{repo_prefix}{data_rel}"
            missing: list[str] = []
            if self.options.verify_files and data_member not in members:
                missing.append(data_member)
            episode_cameras: list[CameraRecord] = []
            video_refs: dict[str, str] = {}
            for camera in cameras:
                video_rel = format_lerobot_path(
                    video_template,
                    episode_index,
                    chunk_size,
                    video_key=camera.source_key,
                )
                member_name = f"{repo_prefix}{video_rel}"
                uri = f"tar://{path}!{member_name}"
                video_refs[camera.source_key] = uri
                if self.options.verify_files and member_name not in members:
                    missing.append(member_name)
                camera_copy = CameraRecord(**camera.to_dict())
                camera_copy.source_uri = uri
                episode_cameras.append(camera_copy)
            tasks_value = row.get("tasks")
            tasks = [str(item) for item in tasks_value] if isinstance(tasks_value, list) else [namespace]
            passed = ["metadata_schema", "episode_boundary", "archive_readable"]
            pending = ["temporal", "signal", "visual", "kinematic"]
            if self.options.verify_files and not missing:
                passed.append("referenced_files_exist")
            elif not self.options.verify_files:
                pending.append("file_integrity")
            self.add_episode(
                source_episode_id=f"{namespace}:{episode_index:06d}",
                source_uri=f"tar://{path}!{data_member}",
                embodiment=robot_type,
                robot_type=robot_type,
                task_namespace=namespace,
                tasks=tasks,
                num_frames=num_frames,
                fps=fps,
                cameras=episode_cameras,
                state_schema=state_schema,
                action_schema=action_schema,
                complete=not missing,
                passed_checks=passed,
                pending_checks=pending,
                failures=["missing_referenced_files"] if missing else [],
                references={
                    "archive": str(path),
                    "data_member": data_member,
                    "videos": video_refs,
                    "missing": missing[:20],
                },
                metadata={
                    "episode_index": episode_index,
                    "archive_root": repo_prefix,
                    "canonical_mapping": canonical_mapping,
                },
            )
