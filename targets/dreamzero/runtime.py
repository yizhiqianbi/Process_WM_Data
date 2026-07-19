from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from targets.common import TargetPreparationError, file_sha256, read_json, write_json


def install_dreamzero_training_profile(
    target_root: Path,
    dreamzero_repo: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    target_root = target_root.expanduser().resolve()
    dreamzero_repo = dreamzero_repo.expanduser().resolve()
    source = target_root / "meta" / "dreamzero_training_profile.yaml"
    receipt_path = target_root / "meta" / "dreamzero_target_receipt.json"
    if not source.is_file() or not receipt_path.is_file():
        raise TargetPreparationError("DreamZero target profile metadata is missing")
    receipt = read_json(receipt_path)
    tag = str(receipt.get("embodiment_tag") or "")
    if not tag:
        raise TargetPreparationError("DreamZero target receipt has no embodiment tag")

    enum_path = dreamzero_repo / "groot" / "vla" / "data" / "schema" / "embodiment_tags.py"
    mapping_path = (
        dreamzero_repo
        / "groot"
        / "vla"
        / "configs"
        / "model"
        / "dreamzero"
        / "transform"
        / "base.yaml"
    )
    if not enum_path.is_file() or not mapping_path.is_file():
        raise TargetPreparationError(f"invalid DreamZero checkout: {dreamzero_repo}")
    enum_text = enum_path.read_text(encoding="utf-8")
    mapping_text = mapping_path.read_text(encoding="utf-8")
    enum_token = f'= "{tag}"'
    mapping_token = f"  {tag}:"
    if enum_token not in enum_text or mapping_token not in mapping_text:
        raise TargetPreparationError(
            f"DreamZero checkout has no registered enum/projector for {tag!r}"
        )

    destination = (
        dreamzero_repo
        / "groot"
        / "vla"
        / "configs"
        / "data"
        / "dreamzero"
        / f"{tag}_relative.yaml"
    )
    if destination.exists() and file_sha256(destination) != file_sha256(source):
        if not overwrite:
            raise TargetPreparationError(
                f"refusing to replace a different DreamZero profile: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    result = {
        "schema_version": "dreamzero-profile-install-v1",
        "embodiment_tag": tag,
        "source": str(source),
        "source_sha256": file_sha256(source),
        "destination": str(destination),
        "destination_sha256": file_sha256(destination),
        "dreamzero_repo": str(dreamzero_repo),
        "enum_registered": True,
        "projector_registered": True,
    }
    write_json(target_root / "meta" / "dreamzero_profile_install_receipt.json", result)
    return result
