from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..camera import normalize_camera_role
from ..canonical import build_verified_mapping, infer_canonical_mapping
from ..schema import CameraRecord
from .base import BaseAdapter

_PART_RE = re.compile(r"^(?P<base>.+\.tar\.gz)\.part-(?P<suffix>[a-z]+)$")

_CONTRACTS: dict[str, dict[str, Any]] = {
    "h5_franka_3rgb": {
        "state_root": "puppet",
        "action_root": "puppet",
        "controls": ["joint_position"],
        "layout": "single_with_gripper",
        "bgr": True,
    },
    "h5_ur_1rgb": {
        "state_root": "puppet",
        "action_root": "puppet",
        "controls": ["joint_position"],
        "layout": "single_with_gripper",
        "bgr": True,
    },
    "h5_agilex_3rgb": {
        "state_root": "puppet",
        "action_root": "master",
        "controls": ["joint_position_left", "joint_position_right"],
        "layout": "dual_split_with_gripper",
        "bgr": False,
    },
    "h5_tienkung_gello_1rgb": {
        "state_root": "puppet",
        "action_root": "master",
        "controls": ["joint_position"],
        "layout": "dual_packed_with_gripper",
        "bgr": False,
    },
    "h5_tienkung_xsens_1rgb": {
        "state_root": "puppet",
        "action_root": "puppet",
        "controls": ["joint_position"],
        "layout": "dual_packed_no_gripper",
        "bgr": False,
    },
}


def _part_number(suffix: str) -> int:
    value = 0
    for character in suffix:
        value = value * 26 + ord(character) - ord("a")
    return value


def _inspect_hdf5(path: Path, embodiment: str) -> dict[str, Any]:
    import h5py

    contract = _CONTRACTS.get(embodiment)
    with h5py.File(path, mode="r") as handle:
        image_root = handle.get("observations/rgb_images")
        image_keys = sorted(image_root.keys()) if image_root is not None else []
        image_lengths = {key: len(image_root[key]) for key in image_keys}
        lengths = [*image_lengths.values()]
        signals: list[dict[str, Any]] = []
        if contract:
            for control in contract["controls"]:
                state_key = f"{contract['state_root']}/{control}"
                action_key = f"{contract['action_root']}/{control}"
                if state_key not in handle or action_key not in handle:
                    continue
                state_dim = int(handle[state_key].shape[-1])
                action_dim = int(handle[action_key].shape[-1])
                if state_dim != action_dim:
                    continue
                signals.append(
                    {
                        "control": control,
                        "state_key": state_key,
                        "action_key": action_key,
                        "dimension": state_dim,
                    }
                )
                lengths.extend([len(handle[state_key]), len(handle[action_key])])
        language = None
        if "language_raw" in handle and len(handle["language_raw"]):
            value = handle["language_raw"][0]
            language = (
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else str(value)
            ).strip()
    return {
        "num_frames": min(lengths) if lengths else None,
        "image_keys": image_keys,
        "image_lengths": image_lengths,
        "signals": signals,
        "contract_verified": bool(contract and len(signals) == len(contract["controls"])),
        "contract": contract,
        "language": language,
    }


def _layout_feature(
    *, key: str, dimension: int, layout: str, control: str, kind: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suffix = "_target" if kind == "action" else ""
    names: list[str] = []
    slots: list[tuple[int, str]] = []
    if layout == "single_with_gripper":
        arm_dimension = dimension - 1
        if arm_dimension not in {6, 7}:
            raise ValueError(f"Unexpected single-arm RoboMIND dimension: {dimension}")
        names = [f"primary_arm_joint_{index + 1}{suffix}" for index in range(arm_dimension)]
        names.append(f"primary_gripper{suffix}")
        slots = [(14 + index, f"primary_joint_{index + 1}{suffix}") for index in range(arm_dimension)]
        slots.append((6, f"primary_gripper{suffix}"))
    elif layout == "dual_split_with_gripper":
        side = "left" if control.endswith("_left") else "right"
        arm_dimension = dimension - 1
        if arm_dimension != 6:
            raise ValueError(f"Unexpected AgileX arm dimension: {dimension}")
        base = 14 if side == "left" else 21
        gripper_slot = 6 if side == "left" else 13
        names = [f"{side}_arm_joint_{index + 1}{suffix}" for index in range(arm_dimension)]
        names.append(f"{side}_gripper{suffix}")
        slots = [(base + index, f"{side}_joint_{index + 1}{suffix}") for index in range(arm_dimension)]
        slots.append((gripper_slot, f"{side}_gripper{suffix}"))
    elif layout == "dual_packed_with_gripper":
        if dimension != 16:
            raise ValueError(f"Unexpected TienKung Gello dimension: {dimension}")
        for side, offset, base, gripper_slot in (
            ("left", 0, 14, 6),
            ("right", 8, 21, 13),
        ):
            for index in range(7):
                names.append(f"{side}_arm_joint_{index + 1}{suffix}")
                slots.append((base + index, f"{side}_joint_{index + 1}{suffix}"))
            names.append(f"{side}_gripper{suffix}")
            slots.append((gripper_slot, f"{side}_gripper{suffix}"))
    elif layout == "dual_packed_no_gripper":
        if dimension != 14:
            raise ValueError(f"Unexpected TienKung Xsens dimension: {dimension}")
        for side, base in (("left", 14), ("right", 21)):
            for index in range(7):
                names.append(f"{side}_arm_joint_{index + 1}{suffix}")
                slots.append((base + index, f"{side}_joint_{index + 1}{suffix}"))
    else:
        raise ValueError(f"Unsupported RoboMIND layout: {layout}")

    entries = [
        {
            "source_key": key,
            "source_index": index,
            "canonical_index": canonical_index,
            "semantic": semantic,
        }
        for index, (canonical_index, semantic) in enumerate(slots)
    ]
    return {
        "dtype": "float64",
        "shape": [dimension],
        "names": names,
    }, entries


def _official_contract_schemas(
    embodiment: str, inspection: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = _CONTRACTS[embodiment]
    state_schema: dict[str, Any] = {}
    action_schema: dict[str, Any] = {}
    state_entries: list[dict[str, Any]] = []
    action_entries: list[dict[str, Any]] = []
    for signal in inspection["signals"]:
        state_feature, state_mapping = _layout_feature(
            key=signal["state_key"],
            dimension=signal["dimension"],
            layout=contract["layout"],
            control=signal["control"],
            kind="state",
        )
        action_feature, action_mapping = _layout_feature(
            key=signal["action_key"],
            dimension=signal["dimension"],
            layout=contract["layout"],
            control=signal["control"],
            kind="action",
        )
        state_schema[signal["state_key"]] = state_feature
        action_schema[signal["action_key"]] = action_feature
        state_entries.extend(state_mapping)
        action_entries.extend(action_mapping)
    provenance = {
        "authority": "official_robomind_training_config_and_hdf5_schema",
        "dataset_card": "https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND",
        "training_config": "static/robomind.yaml",
        "schema_doc": "static/all_robot_h5_info_v1.2.md",
        "state_root": contract["state_root"],
        "action_root": contract["action_root"],
        "controls": contract["controls"],
    }
    mapping = {
        "state": build_verified_mapping(
            state_schema,
            kind="state",
            entries=state_entries,
            verification_note=f"official {embodiment} joint-position feedback layout",
            provenance=provenance,
        ),
        "action": build_verified_mapping(
            action_schema,
            kind="action",
            entries=action_entries,
            verification_note=f"official {embodiment} action_arm_key/control layout",
            provenance=provenance,
        ),
    }
    return state_schema, action_schema, mapping


class RoboMINDAdapter(BaseAdapter):
    dataset_name = "robomind"

    def scan(self) -> None:
        groups: dict[Path, list[tuple[int, Path]]] = defaultdict(list)
        for path in self.options.input_root.rglob("*.tar.gz.part-*"):
            match = _PART_RE.match(path.name)
            if match:
                base = path.with_name(match.group("base"))
                groups[base].append((_part_number(match.group("suffix")), path))

        cache_root = self.options.input_root / ".cache" / "huggingface" / "download"
        pending: dict[Path, list[tuple[int, Path]]] = defaultdict(list)
        if cache_root.is_dir():
            for lock_path in cache_root.rglob("*.lock"):
                target_name = lock_path.name.removesuffix(".lock")
                match = _PART_RE.match(target_name)
                if not match:
                    continue
                relative_parent = lock_path.parent.relative_to(cache_root)
                base = (
                    self.options.input_root
                    / relative_parent
                    / match.group("base")
                )
                pending[base].append((_part_number(match.group("suffix")), lock_path))

        for base, numbered_parts in sorted(groups.items(), key=lambda item: str(item[0])):
            numbered_parts.sort()
            numbers = [number for number, _ in numbered_parts]
            first = min(numbers)
            pending_numbers = [number for number, _ in pending.get(base, [])]
            expected_last = max([*numbers, *pending_numbers])
            missing = sorted(set(range(first, expected_last + 1)) - set(numbers))
            total_size = sum(path.stat().st_size for _, path in numbered_parts)
            complete = not missing and not pending_numbers
            self.add_artifact(
                path=numbered_parts[0][1],
                kind="robomind_split_archive",
                complete=complete,
                status="parts_contiguous" if complete else "pending_or_missing_parts",
                metadata={
                    "logical_archive": str(base),
                    "part_count": len(numbered_parts),
                    "total_size_bytes": total_size,
                    "first_part_number": first,
                    "last_part_number": max(numbers),
                    "expected_last_part_number": expected_last,
                    "missing_part_numbers": missing[:100],
                    "pending_lock_paths": [str(path) for _, path in pending.get(base, [])],
                },
            )

        for base, pending_parts in pending.items():
            if base in groups:
                continue
            self.add_artifact(
                path=pending_parts[0][1],
                kind="robomind_split_archive",
                complete=False,
                status="download_not_started_or_no_completed_parts",
                metadata={
                    "logical_archive": str(base),
                    "pending_part_numbers": sorted(number for number, _ in pending_parts),
                },
            )

        incomplete_cache_files = sorted(cache_root.rglob("*.incomplete")) if cache_root.is_dir() else []
        for path in incomplete_cache_files:
            self.add_artifact(
                path=path,
                kind="huggingface_incomplete_blob",
                complete=False,
                status="download_incomplete",
            )

        # Support already extracted data without requiring h5py during manifest creation.
        hdf5_files = sorted(self.options.input_root.rglob("*.hdf5"))
        hdf5_files.extend(sorted(self.options.input_root.rglob("*.h5")))
        for path in hdf5_files:
            if self.at_limit():
                break
            relative = path.relative_to(self.options.input_root)
            parts = relative.parts
            embodiment = next((part for part in parts if part.startswith("h5_")), "robomind_unknown")
            task_name = next(
                (parts[index - 1] for index, part in enumerate(parts) if part == "success_episodes"),
                path.parent.name,
            )
            try:
                inspection = _inspect_hdf5(path, embodiment)
                inspection_error = None
            except Exception as exc:
                inspection = {}
                inspection_error = str(exc)
            cameras = []
            videos = {}
            for image_key in inspection.get("image_keys") or []:
                source_key = f"observations/rgb_images/{image_key}"
                uri = f"hdf5://{path}#{source_key};array_color_order=bgr"
                videos[source_key] = uri
                cameras.append(
                    CameraRecord(
                        source_key=source_key,
                        role=normalize_camera_role(image_key),
                        dtype="hdf5_embedded",
                        fps=30.0,
                        color_order=(
                            "bgr"
                            if (_CONTRACTS.get(embodiment) or {}).get("bgr")
                            else "rgb"
                        ),
                        source_uri=uri,
                    )
                )
            state_schema: dict[str, Any] = {}
            action_schema: dict[str, Any] = {}
            canonical_mapping = {
                "state": infer_canonical_mapping(state_schema, kind="state"),
                "action": infer_canonical_mapping(action_schema, kind="action"),
            }
            contract_error = None
            if inspection.get("contract_verified"):
                try:
                    state_schema, action_schema, canonical_mapping = (
                        _official_contract_schemas(embodiment, inspection)
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    contract_error = str(exc)
                    state_schema = {}
                    action_schema = {}
            passed_checks = ["hdf5_schema_inference", "episode_boundary"]
            warnings: list[str] = []
            pending_checks = ["temporal", "signal", "visual", "kinematic"]
            if inspection.get("contract_verified") and contract_error is None:
                passed_checks.extend(
                    ["official_action_schema", "canonical_action_mapping"]
                )
            elif inspection_error is None:
                pending_checks.append("official_embodiment_action_contract")
                warnings.append("official_action_contract_not_available")
            self.add_episode(
                source_episode_id=str(path.relative_to(self.options.input_root)),
                source_uri=str(path),
                embodiment=embodiment,
                robot_type=embodiment,
                task_namespace=task_name,
                tasks=[inspection.get("language") or task_name.replace("_", " ")],
                num_frames=inspection.get("num_frames"),
                fps=30.0 if inspection else None,
                cameras=cameras,
                state_schema=state_schema,
                action_schema=action_schema,
                complete=inspection_error is None and bool(cameras),
                passed_checks=passed_checks if inspection_error is None else [],
                pending_checks=pending_checks,
                warnings=warnings,
                failures=["hdf5_schema_inference_failed"] if inspection_error else [],
                references={"hdf5": str(path), "videos": videos},
                metadata={
                    "inspection": inspection,
                    "inspection_error": inspection_error,
                    "contract_error": contract_error,
                    "canonical_mapping": canonical_mapping,
                    "native_conversion": {
                        "source_format": "hdf5",
                        "action_semantics": "official_robomind_action_arm_key",
                    },
                },
            )

        if (groups or pending or incomplete_cache_files) and not hdf5_files:
            self.blockers.append("split_archives_require_streaming_extract_or_mount")
        if pending or incomplete_cache_files:
            self.blockers.append("source_download_incomplete")
        if not groups and not hdf5_files:
            self.blockers.append("no_robomind_data_found")
