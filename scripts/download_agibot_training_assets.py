#!/usr/bin/env python3
"""Download and verify the AgiBot assets required for action training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = PROJECT_ROOT / "configs" / "download_manifest.lock.json"
DEFAULT_COMPONENTS = ("proprio_stats", "task_info")
COMPONENTS = (*DEFAULT_COMPONENTS, "parameters")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument(
        "--component",
        action="append",
        choices=COMPONENTS,
        help="Component to download; repeatable. Default: proprio_stats and task_info.",
    )
    parser.add_argument(
        "--proprio-shard",
        action="append",
        help="Download only this proprio tar basename; repeatable.",
    )
    parser.add_argument("--file-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def read_token(path: Path | None) -> str | None:
    configured = path or (
        Path(os.environ["HF_TOKEN_FILE"]).expanduser()
        if os.environ.get("HF_TOKEN_FILE")
        else None
    )
    if configured is not None:
        token = configured.expanduser().read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"Token file is empty: {configured}")
        return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def agibot_lock_entry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    matches = [
        row
        for row in payload.get("repositories", [])
        if row.get("dataset") == "agibot_beta"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one agibot_beta repository in {path}, found {len(matches)}")
    return matches[0]


def selected_remote_files(
    siblings: list[Any], components: tuple[str, ...], shards: tuple[str, ...]
) -> list[Any]:
    exact_proprio = {
        f"proprio_stats/{Path(value).name}" for value in shards
    }
    selected = []
    for sibling in siblings:
        name = str(sibling.rfilename)
        component = name.split("/", 1)[0]
        if component not in components:
            continue
        if component == "proprio_stats" and exact_proprio and name not in exact_proprio:
            continue
        selected.append(sibling)
    if exact_proprio:
        found = {str(row.rfilename) for row in selected}
        missing = sorted(exact_proprio - found)
        if missing:
            raise FileNotFoundError(f"Unknown AgiBot proprio shards: {missing}")
    return selected


def main() -> int:
    args = parse_args()
    if args.file_workers < 1:
        raise ValueError("--file-workers must be positive")
    components = tuple(dict.fromkeys(args.component or DEFAULT_COMPONENTS))
    shards = tuple(dict.fromkeys(args.proprio_shard or ()))
    if shards and "proprio_stats" not in components:
        raise ValueError("--proprio-shard requires --component proprio_stats")

    entry = agibot_lock_entry(args.lock)
    revision = str(entry["revision"])
    repo_id = str(entry["repo_id"])
    data_root = (
        args.data_root.expanduser()
        if args.data_root
        else Path(os.environ.get("FASTWAM_DATA_ROOT", PROJECT_ROOT.parent)).expanduser()
    )
    local_dir = data_root / str(entry["group_dir"]) / str(entry["local_name"])
    token = read_token(args.token_file)
    info = HfApi(token=token).dataset_info(
        repo_id,
        revision=revision,
        files_metadata=True,
        token=token,
    )
    selected = selected_remote_files(list(info.siblings or []), components, shards)
    if not selected:
        raise ValueError(f"No remote files selected for components={components}")

    remote_bytes = sum(int(row.size or 0) for row in selected)
    allow_patterns = []
    for component in components:
        if component == "proprio_stats" and shards:
            allow_patterns.extend(f"proprio_stats/{Path(value).name}" for value in shards)
        else:
            allow_patterns.append(f"{component}/**")
    summary: dict[str, Any] = {
        "schema_version": "fastwam-agibot-training-assets-v1",
        "repo_id": repo_id,
        "revision": revision,
        "components": list(components),
        "allow_patterns": allow_patterns,
        "selected_file_count": len(selected),
        "remote_bytes": remote_bytes,
        "local_dir": str(local_dir),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        token=token,
        max_workers=args.file_workers,
        library_name="fastwam-preprocess",
    )
    missing = []
    size_mismatches = []
    for row in selected:
        path = local_dir / str(row.rfilename)
        if not path.is_file():
            missing.append(str(row.rfilename))
            continue
        expected = row.size
        if expected is not None and path.stat().st_size != int(expected):
            size_mismatches.append(
                {
                    "path": str(row.rfilename),
                    "expected": int(expected),
                    "actual": path.stat().st_size,
                }
            )
    summary.update(
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ok" if not missing and not size_mismatches else "invalid",
            "missing": missing,
            "size_mismatches": size_mismatches,
        }
    )
    digest = hashlib.sha256("\n".join(allow_patterns).encode("utf-8")).hexdigest()[:12]
    receipt = args.receipt or (
        data_root
        / ".fastwam_download"
        / "components"
        / f"agibot_training_assets_{digest}.json"
    )
    receipt = receipt.expanduser()
    summary["receipt"] = str(receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, receipt)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
