from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent.parent / "configs" / "training_profiles.yaml"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_training_profiles(path: Path | None = None) -> dict[str, Any]:
    import yaml

    source = path or DEFAULT_PROFILE_PATH
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a profile mapping in {source}")
    defaults = payload.get("defaults") or {}
    datasets = payload.get("datasets") or {}
    if not isinstance(defaults, dict) or not isinstance(datasets, dict):
        raise ValueError(f"Invalid defaults/datasets in {source}")
    resolved = {
        str(dataset): _merge(defaults, profile or {})
        for dataset, profile in datasets.items()
    }
    return {
        "schema_version": payload.get("schema_version"),
        "source": str(source),
        "defaults": defaults,
        "datasets": resolved,
    }


def resolve_training_profile(
    profiles: dict[str, Any], dataset: str
) -> dict[str, Any]:
    datasets = profiles.get("datasets") or {}
    profile = datasets.get(dataset)
    if profile is None:
        profile = profiles.get("defaults") or {}
    return deepcopy(profile)
