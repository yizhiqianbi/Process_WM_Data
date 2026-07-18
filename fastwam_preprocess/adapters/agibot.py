from __future__ import annotations

import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..camera import normalize_camera_role
from ..canonical import build_verified_mapping, infer_canonical_mapping
from ..schema import CameraRecord
from ..utils import is_partial_path
from .base import BaseAdapter

_DEPTH_INDEX = re.compile(r"_(\d+)\.png$")


@dataclass(frozen=True)
class _TarMemberRef:
    archive_path: Path
    member_name: str
    offset_data: int
    size: int


def _load_task_metadata(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "task_info").glob("task_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = next(
                (value for value in payload.values() if isinstance(value, list)), []
            )
        else:
            rows = []
        for row in rows:
            if isinstance(row, dict) and row.get("episode_id") is not None:
                result[str(row["episode_id"])] = row
    return result


def _read_tar_member(member: _TarMemberRef) -> bytes:
    # AgiBot proprio shards are uncompressed tar files. TarInfo offsets let us
    # read one HDF5 payload without rebuilding the 45 GiB archive index.
    with member.archive_path.open("rb") as stream:
        stream.seek(member.offset_data)
        payload = stream.read(member.size)
    if len(payload) != member.size:
        raise EOFError(
            f"Short tar member read: {member.archive_path}!{member.member_name}; "
            f"expected={member.size}, actual={len(payload)}"
        )
    return payload


def _inspect_proprio_member(member: _TarMemberRef) -> dict[str, Any]:
    import h5py

    payload = io.BytesIO(_read_tar_member(member))
    paired_keys = [
        ("state/joint/position", "action/joint/position"),
        ("state/effector/position", "action/effector/position"),
        ("state/head/position", "action/head/position"),
        ("state/waist/position", "action/waist/position"),
    ]
    signals: list[dict[str, Any]] = []
    lengths: list[int] = []
    valid_index_sets: list[set[int]] = []
    has_timestamp = False
    with h5py.File(payload, mode="r") as handle:
        if "timestamp" in handle:
            has_timestamp = True
            lengths.append(len(handle["timestamp"]))
        for state_key, action_key in paired_keys:
            if state_key not in handle or action_key not in handle:
                continue
            state_shape = handle[state_key].shape
            action_shape = handle[action_key].shape
            if len(state_shape) != 2 or len(action_shape) != 2:
                continue
            if int(state_shape[-1]) != int(action_shape[-1]):
                continue
            signals.append(
                {
                    "state_key": state_key,
                    "action_key": action_key,
                    "dimension": int(state_shape[-1]),
                }
            )
            action_index_key = f"{action_key.rsplit('/', 1)[0]}/index"
            if action_index_key in handle and len(handle[action_index_key].shape) == 1:
                indices = {
                    int(value)
                    for value in handle[action_index_key][...].tolist()
                    if int(value) >= 0
                }
                signals[-1]["action_index_key"] = action_index_key
                valid_index_sets.append(indices)
            lengths.extend([int(state_shape[0]), int(action_shape[0])])
    row_count = min(lengths) if lengths else None
    valid_indices = (
        sorted(
            index
            for index in set.intersection(*valid_index_sets)
            if row_count is not None and 0 <= index < row_count
        )
        if valid_index_sets
        else []
    )
    if valid_index_sets and not valid_indices:
        raise ValueError("AgiBot HDF5 valid action-index intersection is empty")
    return {
        "signals": signals,
        "num_frames": len(valid_indices) if valid_index_sets else row_count,
        "has_timestamp": has_timestamp,
        "valid_index_count": len(valid_indices),
        "valid_index_start": valid_indices[0] if valid_indices else None,
        "valid_index_stop_exclusive": valid_indices[-1] + 1 if valid_indices else None,
        "valid_indices_contiguous": (
            all(right == left + 1 for left, right in zip(valid_indices, valid_indices[1:]))
            if valid_indices
            else None
        ),
    }


def _agibot_contract(
    inspection: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state_schema: dict[str, Any] = {}
    action_schema: dict[str, Any] = {}
    state_entries: list[dict[str, Any]] = []
    action_entries: list[dict[str, Any]] = []
    for signal in inspection.get("signals") or []:
        state_key = signal["state_key"]
        action_key = signal["action_key"]
        dimension = int(signal["dimension"])
        if state_key.endswith("joint/position") and dimension == 14:
            names = [
                *[f"left_arm_joint_{index + 1}" for index in range(7)],
                *[f"right_arm_joint_{index + 1}" for index in range(7)],
            ]
            slots = [*range(14, 21), *range(21, 28)]
            semantics = [
                *[f"left_joint_{index + 1}" for index in range(7)],
                *[f"right_joint_{index + 1}" for index in range(7)],
            ]
            alignment_safe = True
        elif state_key.endswith("effector/position") and dimension == 2:
            names = ["left_gripper", "right_gripper"]
            slots = [6, 13]
            semantics = ["left_gripper", "right_gripper"]
            # Feedback is in millimeters while command is normalized closure.
            alignment_safe = False
        elif state_key.endswith("effector/position") and dimension == 12:
            names = [
                *[f"left_hand_joint_{index + 1}" for index in range(6)],
                *[f"right_hand_joint_{index + 1}" for index in range(6)],
            ]
            slots = [*range(28, 34), *range(40, 46)]
            semantics = names
            alignment_safe = True
        elif state_key.endswith("head/position") and dimension == 2:
            names = ["head_yaw", "head_pitch"]
            slots = [60, 61]
            semantics = names
            alignment_safe = True
        elif state_key.endswith("waist/position") and dimension == 2:
            names = ["waist_pitch", "waist_lift"]
            slots = [58, 59]
            semantics = names
            alignment_safe = True
        else:
            continue
        state_schema[state_key] = {
            "dtype": "float64",
            "shape": [dimension],
            "names": names,
        }
        action_schema[action_key] = {
            "dtype": "float64",
            "shape": [dimension],
            "names": [f"{name}_target" for name in names],
        }
        for index, (slot, semantic) in enumerate(zip(slots, semantics)):
            state_entries.append(
                {
                    "source_key": state_key,
                    "source_index": index,
                    "canonical_index": slot,
                    "semantic": semantic,
                    "alignment_safe": alignment_safe,
                }
            )
            action_entries.append(
                {
                    "source_key": action_key,
                    "source_index": index,
                    "canonical_index": slot,
                    "semantic": f"{semantic}_target",
                    "alignment_safe": alignment_safe,
                }
            )
    provenance = {
        "authority": "official_agibot_world_beta_hdf5_contract",
        "source": "https://github.com/OpenDriveLab/AgiBot-World",
        "state_semantics": "sensor_and_actuator_feedback",
        "action_semantics": "commands_sent_to_hardware_abstraction_layer",
        "timestamp_unit": "nanoseconds",
    }
    mapping = {
        "state": build_verified_mapping(
            state_schema,
            kind="state",
            entries=state_entries,
            verification_note="official AgiBot G1 proprioceptive state layout",
            provenance=provenance,
        ),
        "action": build_verified_mapping(
            action_schema,
            kind="action",
            entries=action_entries,
            verification_note="official AgiBot G1 HAL command layout",
            provenance=provenance,
        ),
    }
    return state_schema, action_schema, mapping


def _probe_video_member(archive: tarfile.TarFile, member_name: str) -> dict[str, object]:
    import av

    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(member_name)
    with av.open(io.BytesIO(extracted.read())) as container:
        stream = container.streams.video[0]
        return {
            "fps": float(stream.average_rate) if stream.average_rate else None,
            "frames": int(stream.frames) if stream.frames else None,
            "width": int(stream.width),
            "height": int(stream.height),
            "codec": stream.codec_context.name,
        }


class AgiBotBetaAdapter(BaseAdapter):
    dataset_name = "agibot_beta"

    def scan(self) -> None:
        task_metadata = _load_task_metadata(self.options.input_root)
        proprio_index: dict[str, _TarMemberRef] = {}
        proprio_root = self.options.input_root / "proprio_stats"
        for proprio_shard in sorted(proprio_root.rglob("*.tar*")):
            if proprio_shard.suffix != ".tar" or is_partial_path(proprio_shard):
                self.add_artifact(
                    path=proprio_shard,
                    kind="agibot_proprio_shard",
                    complete=False,
                    status="partial_download",
                )
                continue
            try:
                with tarfile.open(proprio_shard, mode="r:*") as archive:
                    members = [
                        member
                        for member in archive
                        if member.isfile() and member.name.endswith("/proprio_stats.h5")
                    ]
            except (OSError, tarfile.TarError) as exc:
                self.add_artifact(
                    path=proprio_shard,
                    kind="agibot_proprio_shard",
                    complete=False,
                    status="invalid_tar",
                    metadata={"error": str(exc)},
                )
                continue
            for member in members:
                episode_id = Path(member.name).parent.name
                proprio_index.setdefault(
                    episode_id,
                    _TarMemberRef(
                        archive_path=proprio_shard,
                        member_name=member.name,
                        offset_data=int(member.offset_data),
                        size=int(member.size),
                    ),
                )
            self.add_artifact(
                path=proprio_shard,
                kind="agibot_proprio_shard",
                complete=True,
                status="indexed",
                metadata={"episode_count": len(members)},
            )

        observations = self.options.input_root / "observations"
        shards = sorted(observations.glob("*/*.tar*"))
        if not shards:
            self.blockers.append("observation_archives_not_present")
            return

        for shard in shards:
            if self.at_limit():
                break
            complete = shard.suffix == ".tar" and not is_partial_path(shard)
            if not complete:
                self.add_artifact(
                    path=shard,
                    kind="agibot_observation_shard",
                    complete=False,
                    status="partial_download",
                )
                continue
            try:
                grouped: dict[str, dict[str, object]] = {}
                with tarfile.open(shard, mode="r:") as archive:
                    for member in archive:
                        if not member.isfile() or "/" not in member.name:
                            continue
                        episode_id = member.name.split("/", 1)[0]
                        episode = grouped.setdefault(
                            episode_id, {"videos": [], "depth_indices": [], "count": 0}
                        )
                        episode["count"] = int(episode["count"]) + 1
                        if member.name.endswith(".mp4"):
                            episode["videos"].append(member.name)
                        match = _DEPTH_INDEX.search(member.name)
                        if match is not None:
                            episode["depth_indices"].append(int(match.group(1)))
                    if self.options.verify_files:
                        probe_ids = sorted(grouped)[: self.options.max_episodes or len(grouped)]
                        for episode_id in probe_ids:
                            episode_info = grouped[episode_id]
                            video_members = sorted(episode_info["videos"])
                            preferred = next(
                                (name for name in video_members if name.endswith("/head_color.mp4")),
                                video_members[0] if video_members else None,
                            )
                            if preferred is not None:
                                try:
                                    episode_info["video_probe"] = _probe_video_member(
                                        archive, preferred
                                    )
                                except Exception as exc:
                                    episode_info["video_probe_error"] = str(exc)
            except (OSError, tarfile.TarError) as exc:
                self.add_artifact(
                    path=shard,
                    kind="agibot_observation_shard",
                    complete=False,
                    status="invalid_tar",
                    metadata={"error": str(exc)},
                )
                continue

            self.add_artifact(
                path=shard,
                kind="agibot_observation_shard",
                complete=True,
                status="indexed",
                metadata={"episode_count": len(grouped)},
            )
            task_id = shard.parent.name
            for episode_id, episode_info in sorted(grouped.items()):
                if self.at_limit():
                    break
                video_members = sorted(episode_info["videos"])
                video_probe = episode_info.get("video_probe") or {}
                cameras = [
                    CameraRecord(
                        source_key=Path(name).stem,
                        role=normalize_camera_role(Path(name).stem),
                        width=video_probe.get("width"),
                        height=video_probe.get("height"),
                        fps=video_probe.get("fps") or 30.0,
                        codec=video_probe.get("codec"),
                        source_uri=f"tar://{shard}!{name}",
                    )
                    for name in video_members
                ]
                depth_indices = episode_info["depth_indices"]
                estimated_frames = max(depth_indices) + 1 if depth_indices else None
                if video_probe.get("frames"):
                    estimated_frames = int(video_probe["frames"])
                proprio = proprio_index.get(episode_id)
                proprio_uri = None
                proprio_inspection: dict[str, Any] = {}
                proprio_error = None
                state_schema: dict[str, Any] = {}
                action_schema: dict[str, Any] = {}
                canonical_mapping = {
                    "state": infer_canonical_mapping(state_schema, kind="state"),
                    "action": infer_canonical_mapping(action_schema, kind="action"),
                }
                if proprio is not None:
                    proprio_uri = (
                        f"tar://{proprio.archive_path}!{proprio.member_name}"
                    )
                    try:
                        proprio_inspection = _inspect_proprio_member(proprio)
                        state_schema, action_schema, canonical_mapping = (
                            _agibot_contract(proprio_inspection)
                        )
                        proprio_frames = proprio_inspection.get("num_frames")
                        if proprio_frames is not None:
                            estimated_frames = int(proprio_frames)
                    except Exception as exc:
                        proprio_error = str(exc)
                task_row = task_metadata.get(episode_id) or {}
                label_info = task_row.get("label_info") or task_row.get("lable_info") or {}
                action_config = label_info.get("action_config") or []
                task_text = str(task_row.get("task_name") or "").strip()
                action_texts = [
                    str(item.get("action_text")).strip()
                    for item in action_config
                    if isinstance(item, dict) and item.get("action_text")
                ]
                tasks = list(
                    dict.fromkeys(value for value in [task_text, *action_texts] if value)
                )
                passed = ["archive_member_readable", "episode_boundary"]
                pending = ["temporal", "signal", "visual", "kinematic"]
                warnings: list[str] = []
                if proprio_uri and proprio_error is None and action_schema:
                    passed.extend(
                        [
                            "proprio_hdf5_schema",
                            "official_action_schema",
                            "canonical_action_mapping",
                        ]
                    )
                else:
                    pending.extend(
                        ["proprio_download_or_join", "state_action_alignment"]
                    )
                    warnings.append("observation_only_local_snapshot")
                if not task_row:
                    pending.append("task_metadata_join")
                    warnings.append("task_metadata_missing_for_episode")
                references: dict[str, Any] = {
                    "archive": str(shard),
                    "episode_prefix": f"{episode_id}/",
                    "videos": {
                        Path(name).stem: f"tar://{shard}!{name}"
                        for name in video_members
                    },
                }
                if proprio_uri:
                    references["data"] = proprio_uri
                self.add_episode(
                    source_episode_id=episode_id,
                    source_uri=proprio_uri or f"tar://{shard}!{episode_id}/",
                    embodiment="agibot_g1",
                    robot_type="agibot_g1",
                    task_namespace=task_id,
                    tasks=tasks,
                    num_frames=estimated_frames,
                    fps=video_probe.get("fps") or 30.0,
                    cameras=cameras,
                    state_schema=state_schema,
                    action_schema=action_schema,
                    action_verified=bool(
                        proprio_uri
                        and proprio_error is None
                        and canonical_mapping["action"].get("verified")
                    ),
                    has_calibration=(self.options.input_root / "parameters").exists(),
                    has_depth=bool(depth_indices),
                    complete=bool(cameras) and proprio_error is None,
                    passed_checks=passed,
                    pending_checks=pending,
                    warnings=warnings,
                    failures=(
                        ["episode_has_no_video"]
                        if not cameras
                        else (["proprio_hdf5_schema_failed"] if proprio_error else [])
                    ),
                    references=references,
                    metadata={
                        "task_id": task_id,
                        "depth_frame_count": len(depth_indices),
                        "archive_member_count": int(episode_info["count"]),
                        "video_probe": video_probe,
                        "task_metadata": task_row,
                        "proprio_inspection": proprio_inspection,
                        "proprio_error": proprio_error,
                        "canonical_mapping": canonical_mapping,
                        "native_conversion": (
                            {
                                "source_format": "hdf5",
                                "timestamp_key": "timestamp",
                                "action_semantics": "native_hal_command",
                                "valid_index_keys": sorted(
                                    {
                                        str(signal["action_index_key"])
                                        for signal in proprio_inspection.get("signals") or []
                                        if signal.get("action_index_key")
                                    }
                                ),
                                "valid_index_policy": "intersection",
                            }
                            if proprio_uri
                            else {}
                        ),
                    },
                )

        if not (self.options.input_root / "task_info").exists():
            self.blockers.append("task_info_directory_missing")
        if not proprio_index:
            self.blockers.append("proprio_stats_archives_missing_or_incomplete")
