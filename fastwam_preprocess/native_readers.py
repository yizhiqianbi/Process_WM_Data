from __future__ import annotations

import io
import pickle
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Allow only the NumPy reconstruction globals present in OXE shards."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
    }

    def find_class(self, module: str, name: str):
        value = self._ALLOWED.get((module, name))
        if value is None:
            raise pickle.UnpicklingError(f"Blocked pickle global: {module}.{name}")
        return value


def load_oxe_episode(archive_path: Path, member_name: str) -> dict[str, Any]:
    with tarfile.open(archive_path, mode="r:*") as archive:
        extracted = archive.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"Archive member does not exist: {archive_path}!{member_name}")
        payload = RestrictedNumpyUnpickler(extracted).load()
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        raise ValueError("OXE pickle must be a dict with a list-valued `steps` field")
    return payload


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _image_shape(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            with Image.open(io.BytesIO(bytes(value))) as image:
                return int(image.height), int(image.width)
        except Exception:
            return None
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        return int(array.shape[0]), int(array.shape[1])
    return None


def inspect_oxe_episode(archive_path: Path, member_name: str) -> dict[str, Any]:
    payload = load_oxe_episode(archive_path, member_name)
    steps = payload["steps"]
    if not steps:
        raise ValueError("OXE episode contains no steps")
    first = steps[0]
    if not isinstance(first, dict):
        raise ValueError("OXE steps must contain dictionaries")
    observation = first.get("observation")
    observation = observation if isinstance(observation, dict) else {}

    images = {}
    for key, value in observation.items():
        shape = _image_shape(value)
        if shape is not None:
            images[str(key)] = {"height": shape[0], "width": shape[1]}

    action = np.asarray(first.get("action")) if first.get("action") is not None else None
    action_delta = (
        np.asarray(first.get("action_delta"))
        if first.get("action_delta") is not None
        else None
    )
    state = np.asarray(observation.get("state")) if observation.get("state") is not None else None
    ground_truth = first.get("ground_truth_states")
    ground_truth = ground_truth if isinstance(ground_truth, dict) else {}
    ground_truth_ee = (
        np.asarray(ground_truth.get("EE"))
        if ground_truth.get("EE") is not None
        else None
    )
    instruction = ""
    for step in steps:
        if isinstance(step, dict):
            instruction = _decode_text(step.get("language_instruction"))
            if instruction:
                break
    return {
        "num_steps": len(steps),
        "instruction": instruction,
        "images": images,
        "action_dim": int(action.size) if action is not None and action.ndim == 1 else None,
        "action_delta_dim": (
            int(action_delta.size)
            if action_delta is not None and action_delta.ndim == 1
            else None
        ),
        "state_dim": int(state.size) if state is not None and state.ndim == 1 else None,
        "ground_truth_ee_dim": (
            int(ground_truth_ee.size)
            if ground_truth_ee is not None and ground_truth_ee.ndim == 1
            else None
        ),
    }
