from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re
from typing import Any

from .camera import normalize_camera_role
from .canonical import CANONICAL_DIM
from .profiles import load_training_profiles, resolve_training_profile
from .utils import AtomicJsonlWriter, iter_jsonl, stable_id, write_json


TRAINING_CASE_VERSION = "fastwam-training-case-v1"
CAMERA_ROLES = (
    "global_primary",
    "global_secondary",
    "left_wrist",
    "right_wrist",
    "auxiliary",
)
TARGET_FPS = 20.0
STATE_STEPS = 81
ACTION_STEPS = 80
VIDEO_OFFSETS = tuple(range(0, STATE_STEPS, 4))


def _camera_score(camera: dict[str, Any]) -> tuple[int, int, float, str]:
    uri = camera.get("source_uri")
    width = int(camera.get("width") or 0)
    height = int(camera.get("height") or 0)
    fps = float(camera.get("fps") or 0.0)
    return (int(bool(uri)), width * height, fps, str(camera.get("source_key") or ""))


def _camera_role(camera: dict[str, Any]) -> str:
    role = str(camera.get("role") or "")
    if role in CAMERA_ROLES:
        return role
    role_base = re.sub(r"_\d+$", "", role)
    if role_base in CAMERA_ROLES:
        return role_base
    return normalize_camera_role(str(camera.get("source_key") or role))


def _camera_slots(
    cameras: list[dict[str, Any]], videos: Any, video_audit: Any = None
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    video_lookup = videos if isinstance(videos, dict) else {}
    audit_lookup = video_audit if isinstance(video_audit, dict) else {}
    failed_keys = {
        str(key)
        for key, report in audit_lookup.items()
        if isinstance(report, dict)
        and (
            report.get("status") == "failed"
            or (report.get("sparse_visual") or {}).get("status") == "failed"
        )
    }
    failed_cameras = [
        dict(camera)
        for camera in cameras
        if str(camera.get("source_key") or "") in failed_keys
    ]
    ordered = sorted(
        [
            dict(camera)
            for camera in cameras
            if str(camera.get("source_key") or "") not in failed_keys
        ],
        key=_camera_score,
        reverse=True,
    )
    selected: dict[str, dict[str, Any]] = {}
    used: set[int] = set()
    for role in CAMERA_ROLES:
        candidates = [
            (index, camera)
            for index, camera in enumerate(ordered)
            if index not in used and _camera_role(camera) == role
        ]
        if candidates:
            index, camera = candidates[0]
            selected[role] = camera
            used.add(index)

    if "global_secondary" not in selected:
        candidates = [
            (index, camera)
            for index, camera in enumerate(ordered)
            if index not in used and _camera_role(camera) == "global_primary"
        ]
        if candidates:
            index, camera = candidates[0]
            selected["global_secondary"] = camera
            used.add(index)
    if "auxiliary" not in selected:
        remaining = [
            (index, camera)
            for index, camera in enumerate(ordered)
            if index not in used
        ]
        if remaining:
            index, camera = remaining[0]
            selected["auxiliary"] = camera
            used.add(index)

    slots: list[dict[str, Any]] = []
    for slot_index, role in enumerate(CAMERA_ROLES):
        camera = selected.get(role)
        if camera is None:
            slots.append(
                {
                    "slot_index": slot_index,
                    "role": role,
                    "valid": False,
                    "source_key": None,
                    "source_uri": None,
                    "storage": None,
                    "fps": None,
                    "width": None,
                    "height": None,
                    "codec": None,
                    "audit_status": None,
                }
            )
            continue
        source_key = str(camera.get("source_key") or "")
        source_uri = camera.get("source_uri") or video_lookup.get(source_key)
        storage = "embedded"
        if source_uri:
            storage = "tar_member" if str(source_uri).startswith("tar://") else "file"
        slots.append(
            {
                "slot_index": slot_index,
                "role": role,
                "valid": True,
                "source_key": source_key,
                "source_uri": source_uri,
                "storage": storage,
                "fps": camera.get("fps"),
                "width": camera.get("width"),
                "height": camera.get("height"),
                "codec": camera.get("codec"),
                "audit_status": (
                    (audit_lookup.get(source_key) or {}).get("status")
                    if isinstance(audit_lookup.get(source_key), dict)
                    else None
                ),
                "has_depth": bool(camera.get("has_depth")),
                "intrinsics_available": bool(camera.get("intrinsics_available")),
                "extrinsics_available": bool(camera.get("extrinsics_available")),
            }
        )
    failed = [
        str(camera.get("source_key") or "")
        for camera in failed_cameras
    ]
    dropped = [
        str(camera.get("source_key") or "")
        for index, camera in enumerate(ordered)
        if index not in used
    ]
    return slots, dropped, failed


def _prompt(row: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    tasks = row.get("tasks") or []
    if not isinstance(tasks, list):
        tasks = [tasks]
    prompts: list[str] = []
    for task in tasks:
        value = re.sub(r"\s+", " ", str(task)).strip()
        if value and value not in prompts:
            prompts.append(value)
    source = "tasks"
    if not prompts and profile.get("language_fallback") == "task_namespace":
        fallback = re.sub(r"[_/]+", " ", str(row.get("task_namespace") or "")).strip()
        if fallback:
            prompts.append(fallback)
            source = "task_namespace_fallback"
    return {
        "primary": prompts[0] if prompts else "",
        "alternatives": prompts[1:],
        "all": prompts,
        "valid": bool(prompts),
        "source": source if prompts else "missing",
    }


def assign_split(
    split_group_id: str,
    *,
    seed: str = "fastwam-v1",
    train_fraction: float = 0.98,
    validation_fraction: float = 0.01,
) -> str:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0.0 <= validation_fraction < 1.0 - train_fraction):
        raise ValueError("validation_fraction leaves no test split")
    digest = hashlib.sha256(f"{seed}\x1f{split_group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def _valid_starts(row: dict[str, Any]) -> dict[str, int]:
    compact = row.get("valid_starts")
    if isinstance(compact, dict):
        return {
            "start": int(compact["start"]),
            "stop_exclusive": int(compact["stop_exclusive"]),
            "stride": int(compact["stride"]),
            "count": int(compact["count"]),
        }
    start = int(row.get("start") or 0)
    return {"start": start, "stop_exclusive": start + 1, "stride": 1, "count": 1}


def make_training_case(
    row: dict[str, Any],
    profile: dict[str, Any],
    *,
    split_seed: str,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, Any]:
    state_slots = sorted({int(value) for value in row.get("state_valid_slots") or []})
    action_slots = sorted({int(value) for value in row.get("action_valid_slots") or []})
    mode = str(row.get("training_mode") or "video_only")
    enabled_modes = profile.get("enabled_modes") or ["joint_video_action", "video_only"]
    if mode not in enabled_modes or (mode == "joint_video_action" and not action_slots):
        mode = "video_only"
    video_audit = (
        ((row.get("audit_summary") or {}).get("metrics_summary") or {}).get(
            "videos"
        )
        or {}
    )
    slots, dropped_cameras, failed_cameras = _camera_slots(
        row.get("cameras") or [], row.get("videos") or {}, video_audit
    )
    prompt = _prompt(row, profile)
    group_id = str(
        row.get("split_group_id")
        or stable_id(
            "split_group",
            row.get("dataset"),
            row.get("lineage_id") or row.get("global_episode_id"),
            length=24,
        )
    )
    quality = row.get("quality") or {}
    quality_weight = float(
        (profile.get("quality_weights") or {}).get(str(quality.get("tier")), 0.0)
    )
    audit_sampling_weight = float(quality.get("sampling_weight", 1.0))
    mixture_weight = float(profile.get("mixture_weight") or 1.0)
    valid_starts = _valid_starts(row)
    indexed_state_steps = int(row.get("num_frames") or STATE_STEPS)
    indexed_action_steps = int(
        row.get("action_horizon") or (indexed_state_steps - 1)
    )
    indexed_video_offsets = [
        int(value) for value in row.get("video_sample_offsets") or VIDEO_OFFSETS
    ]
    indexed_target_fps = float(row.get("target_fps") or TARGET_FPS)
    case_id = stable_id(
        TRAINING_CASE_VERSION,
        row.get("global_episode_id"),
        valid_starts["start"],
        valid_starts["stop_exclusive"],
        valid_starts["stride"],
        length=24,
    )
    return {
        "schema_version": TRAINING_CASE_VERSION,
        "case_id": case_id,
        "dataset": row.get("dataset"),
        "release": row.get("release"),
        "global_episode_id": row.get("global_episode_id"),
        "source_episode_id": row.get("source_episode_id"),
        "lineage_id": row.get("lineage_id"),
        "split_group_id": group_id,
        "split": assign_split(
            group_id,
            seed=split_seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        ),
        "embodiment": {
            "name": row.get("embodiment"),
            "robot_type": row.get("robot_type"),
            "normalization_domain": row.get("normalization_domain"),
        },
        "language": prompt,
        "timeline": {
            "target_fps": indexed_target_fps,
            "duration_s": indexed_action_steps / indexed_target_fps,
            "state_steps": indexed_state_steps,
            "action_steps": indexed_action_steps,
            "video_steps": len(indexed_video_offsets),
            "video_offsets": indexed_video_offsets,
            "action_semantics": "action[t] targets transition state[t] to state[t+1]",
        },
        "sampling": {
            "unit": "episode_start_range",
            "valid_starts": valid_starts,
            "quality_weight": audit_sampling_weight,
            "estimated_unique_video_frames": row.get(
                "estimated_unique_video_frames"
            ),
        },
        "inputs": {
            "canonical_parquet": row.get("canonical_parquet"),
            "source_episode_uri": row.get("source_uri"),
            "state_column": "canonical_state",
            "state_mask_column": "state_dim_valid_mask",
            "action_column": "canonical_action",
            "action_mask_column": "action_dim_valid_mask",
            "state_dimension": CANONICAL_DIM,
            "action_dimension": CANONICAL_DIM,
            "state_valid_slots": state_slots,
            "action_valid_slots": action_slots,
            "state_slot_mask": [index in state_slots for index in range(CANONICAL_DIM)],
            "action_slot_mask": [index in action_slots for index in range(CANONICAL_DIM)],
            "camera_slots": slots,
            "camera_slot_mask": [bool(slot["valid"]) for slot in slots],
            "dropped_camera_source_keys": dropped_cameras,
            "failed_camera_source_keys": failed_cameras,
        },
        "training": {
            "mode": mode,
            "conditioning_mask": {
                "video": any(slot["valid"] for slot in slots),
                "state": mode == "joint_video_action" and bool(state_slots),
                "language": prompt["valid"],
            },
            "loss_mask": {
                "video": True,
                "action": mode == "joint_video_action" and bool(action_slots),
            },
            "loss_weights": {
                "video": float((profile.get("loss_weights") or {}).get("video", 1.0)),
                "action": (
                    float((profile.get("loss_weights") or {}).get("action", 1.0))
                    if mode == "joint_video_action"
                    else 0.0
                ),
            },
            "sample_weight": mixture_weight * quality_weight * audit_sampling_weight,
            "mixture_weight": mixture_weight,
            "profile": {
                "input_family": profile.get("input_family"),
                "action_gate": profile.get("action_gate"),
                "normalization_scope": profile.get("normalization_scope"),
            },
            "stage_admission": row.get("stage_admission") or {},
        },
        "quality": {
            "tier": quality.get("tier"),
            "score": quality.get("score"),
            "passed_checks": quality.get("passed_checks") or [],
            "pending_checks": quality.get("pending_checks") or [],
            "warnings": quality.get("warnings") or [],
            "component_scores": quality.get("component_scores") or {},
            "hard_blockers": quality.get("hard_blockers") or [],
            "soft_flags": quality.get("soft_flags") or [],
            "sampling_weight": audit_sampling_weight,
        },
        "provenance": {
            "references": row.get("references") or {},
            "source_profile": row.get("source_profile") or {},
            "source_episode_metadata": row.get("source_episode_metadata") or {},
            "audit_summary": row.get("audit_summary") or {},
            "bad_intervals": row.get("bad_intervals") or [],
        },
    }


def validate_training_case(case: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    timeline = case.get("timeline") or {}
    if timeline.get("target_fps") != TARGET_FPS:
        failures.append("target_fps_must_be_20")
    if timeline.get("state_steps") != STATE_STEPS:
        failures.append("state_steps_must_be_81")
    if timeline.get("action_steps") != ACTION_STEPS:
        failures.append("action_steps_must_be_80")
    if timeline.get("video_offsets") != list(VIDEO_OFFSETS):
        failures.append("video_offsets_must_be_0_to_80_stride_4")
    inputs = case.get("inputs") or {}
    camera_slots = inputs.get("camera_slots") or []
    if [slot.get("role") for slot in camera_slots] != list(CAMERA_ROLES):
        failures.append("camera_roles_must_match_fixed_five_slots")
    if [slot.get("slot_index") for slot in camera_slots] != list(range(len(CAMERA_ROLES))):
        failures.append("camera_slot_indices_invalid")
    if len(inputs.get("state_slot_mask") or []) != CANONICAL_DIM:
        failures.append("state_slot_mask_must_have_80_dimensions")
    if len(inputs.get("action_slot_mask") or []) != CANONICAL_DIM:
        failures.append("action_slot_mask_must_have_80_dimensions")
    if not inputs.get("canonical_parquet"):
        failures.append("canonical_parquet_missing")
    starts = ((case.get("sampling") or {}).get("valid_starts") or {})
    if int(starts.get("count") or 0) <= 0:
        failures.append("no_valid_starts")
    mode = ((case.get("training") or {}).get("mode"))
    action_loss = bool(((case.get("training") or {}).get("loss_mask") or {}).get("action"))
    if mode == "joint_video_action" and not action_loss:
        failures.append("joint_mode_requires_action_loss")
    if mode == "video_only" and action_loss:
        failures.append("video_only_must_disable_action_loss")
    if not any(inputs.get("camera_slot_mask") or []):
        failures.append("no_valid_camera_slot")
    return failures


def expand_training_case(case: dict[str, Any], start: int | None = None) -> dict[str, Any]:
    expanded = deepcopy(case)
    starts = expanded["sampling"]["valid_starts"]
    actual_start = int(starts["start"] if start is None else start)
    if actual_start < int(starts["start"]) or actual_start >= int(starts["stop_exclusive"]):
        raise ValueError("start is outside the valid range")
    if (actual_start - int(starts["start"])) % int(starts["stride"]) != 0:
        raise ValueError("start does not follow the configured stride")
    expanded["concrete_case_id"] = stable_id(case["case_id"], actual_start, length=24)
    expanded["window"] = {
        "start": actual_start,
        "state_start": actual_start,
        "state_stop_exclusive": actual_start + STATE_STEPS,
        "action_start": actual_start,
        "action_stop_exclusive": actual_start + ACTION_STEPS,
        "video_frame_indices": [actual_start + offset for offset in VIDEO_OFFSETS],
        "video_timestamps_s": [
            (actual_start + offset) / TARGET_FPS for offset in VIDEO_OFFSETS
        ],
    }
    return expanded


def training_case_contract(
    *, split_seed: str, train_fraction: float, validation_fraction: float
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_CASE_VERSION,
        "control_timeline": {
            "target_fps": TARGET_FPS,
            "state_steps": STATE_STEPS,
            "action_steps": ACTION_STEPS,
            "duration_s": ACTION_STEPS / TARGET_FPS,
        },
        "video_timeline": {
            "steps": len(VIDEO_OFFSETS),
            "offsets": list(VIDEO_OFFSETS),
        },
        "canonical_dimensions": {"state": CANONICAL_DIM, "action": CANONICAL_DIM},
        "camera_roles": list(CAMERA_ROLES),
        "modes": {
            "joint_video_action": "video and masked action losses enabled",
            "video_only": "video loss enabled and action loss disabled",
        },
        "quality": {
            "component_scores": [
                "integrity",
                "temporal",
                "visual",
                "kinematic",
                "language",
                "novelty",
            ],
            "window_local_filter": "hard bad_intervals only",
            "soft_intervals": "retained for sampling and review",
        },
        "stages": {
            "stage1_video_backbone": "video and temporal window admission",
            "stage2_memory_fastwam": "verified state/action and full window admission",
            "stage3_target_finetune": "stage-2 candidate plus target-dataset policy",
        },
        "split": {
            "seed": split_seed,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": 1.0 - train_fraction - validation_fraction,
            "group_key": "split_group_id derived from lineage_id or source episode",
        },
    }


def build_training_cases(
    windows_manifest: Path,
    output_dir: Path,
    *,
    profiles_path: Path | None = None,
    split_seed: str = "fastwam-v1",
    train_fraction: float = 0.98,
    validation_fraction: float = 0.01,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_training_profiles(profiles_path)
    contract = training_case_contract(
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    write_json(output_dir / "training_case_contract.json", contract)
    counts = {"joint_video_action": 0, "video_only": 0}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    dataset_counts: dict[str, int] = {}
    case_count = 0
    rejected_count = 0
    window_count = 0
    first_case: dict[str, Any] | None = None
    with (
        AtomicJsonlWriter(output_dir / "training_cases.jsonl") as writer,
        AtomicJsonlWriter(output_dir / "cases_rejected.jsonl") as rejected_writer,
    ):
        for row in iter_jsonl(windows_manifest):
            dataset = str(row.get("dataset") or "unknown")
            profile = resolve_training_profile(profiles, dataset)
            case = make_training_case(
                row,
                profile,
                split_seed=split_seed,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
            failures = validate_training_case(case)
            if failures:
                rejected_writer.write(
                    {
                        "global_episode_id": row.get("global_episode_id"),
                        "case_id": case.get("case_id"),
                        "failures": failures,
                    }
                )
                rejected_count += 1
                continue
            writer.write(case)
            if first_case is None:
                first_case = expand_training_case(case)
            case_count += 1
            starts = case["sampling"]["valid_starts"]
            window_count += int(starts["count"])
            mode = case["training"]["mode"]
            counts[mode] = counts.get(mode, 0) + 1
            split = case["split"]
            split_counts[split] = split_counts.get(split, 0) + 1
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    if first_case is None:
        write_json(
            output_dir / "example_case.json",
            {
                "schema_version": TRAINING_CASE_VERSION,
                "available": False,
                "reason": "no_admissible_materialized_window",
            },
        )
    else:
        write_json(output_dir / "example_case.json", first_case)
    summary = {
        "schema_version": TRAINING_CASE_VERSION,
        "input_manifest": str(windows_manifest),
        "case_count": case_count,
        "represented_window_count": window_count,
        "rejected_case_count": rejected_count,
        "training_modes": counts,
        "splits": split_counts,
        "datasets": dataset_counts,
        "profiles_source": profiles["source"],
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "training_cases_summary.json", summary)
    return summary
