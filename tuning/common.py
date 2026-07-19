from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any
import uuid

import yaml


CONFIG_SCHEMA_VERSION = "wm-tuning-v1"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class TuningConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CommandSpec:
    model: str
    phase: str
    argv: tuple[str, ...]
    cwd: Path
    output_dir: Path
    env: dict[str, str] = field(default_factory=dict)
    external_repo: Path | None = None

    def document(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "phase": self.phase,
            "argv": list(self.argv),
            "command": shlex.join(self.argv),
            "cwd": str(self.cwd),
            "output_dir": str(self.output_dir),
            "environment": dict(sorted(self.env.items())),
            "external_repo": None if self.external_repo is None else str(self.external_repo),
        }


def _expand_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        current = os.environ.get(name)
        if current is not None and current != "":
            return current
        if default is not None:
            return default
        raise TuningConfigError(f"required environment variable is not set: {name}")

    expanded = _ENV_PATTERN.sub(replace, value)
    return str(Path(expanded).expanduser()) if expanded.startswith("~") else expanded


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand(item) for key, item in value.items()}
    return value


def load_tuning_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TuningConfigError("tuning config must be a YAML mapping")
    config = _expand(raw)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise TuningConfigError(
            f"unsupported tuning schema: {config.get('schema_version')!r}; "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )
    if not isinstance(config.get("models"), dict):
        raise TuningConfigError("tuning config requires a models mapping")
    config["_config_path"] = str(path)
    return config


def model_config(config: dict[str, Any], model: str) -> dict[str, Any]:
    value = (config.get("models") or {}).get(model)
    if not isinstance(value, dict):
        raise TuningConfigError(f"missing models.{model} mapping")
    return value


def required_path(mapping: dict[str, Any], key: str, *, directory: bool | None = None) -> Path:
    value = mapping.get(key)
    if value is None or str(value).strip() == "":
        raise TuningConfigError(f"missing required path: {key}")
    path = Path(str(value)).expanduser().resolve()
    if directory is True and not path.is_dir():
        raise TuningConfigError(f"directory does not exist for {key}: {path}")
    if directory is False and not path.is_file():
        raise TuningConfigError(f"file does not exist for {key}: {path}")
    return path


def hydra_list(values: list[str] | tuple[str, ...]) -> str:
    if any("," in value or "[" in value or "]" in value for value in values):
        raise TuningConfigError(f"Hydra list values contain reserved punctuation: {values}")
    return "[" + ",".join(values) + "]"


def _git_state(repo: Path | None) -> dict[str, Any] | None:
    if repo is None or not (repo / ".git").exists():
        return None
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True, check=False
    )
    return {
        "sha": sha.stdout.strip() if sha.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def discover_checkpoints(output_dir: Path) -> list[str]:
    candidates: set[Path] = set()
    for pattern in (
        "checkpoint-*",
        "checkpoints/checkpoint_*",
        "checkpoints/state/*",
        "checkpoints/weights/*",
    ):
        candidates.update(output_dir.glob(pattern))
    return [str(path) for path in sorted(candidates)]


def run_command(spec: CommandSpec) -> dict[str, Any]:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = spec.output_dir / "_wm_tuning"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    log_path = metadata_dir / f"{run_id}.log"
    receipt_path = metadata_dir / f"{run_id}.json"
    started = datetime.now(timezone.utc)
    receipt: dict[str, Any] = {
        "schema_version": "wm-tuning-run-v1",
        "run_id": run_id,
        "status": "running",
        "started_at": started.isoformat(),
        **spec.document(),
        "external_git": _git_state(spec.external_repo),
        "log_path": str(log_path),
    }
    _write_json(receipt_path, receipt)

    environment = os.environ.copy()
    environment.update(spec.env)
    return_code: int
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(spec.argv),
            cwd=spec.cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                return_code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
            raise
        finally:
            receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
            receipt["duration_seconds"] = (
                datetime.now(timezone.utc) - started
            ).total_seconds()
            receipt["return_code"] = process.returncode
            receipt["status"] = "succeeded" if process.returncode == 0 else "failed"
            receipt["checkpoints"] = discover_checkpoints(spec.output_dir)
            _write_json(receipt_path, receipt)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(spec.argv))
    return receipt


def latest_status(output_dir: Path) -> dict[str, Any]:
    receipt_paths = sorted((output_dir / "_wm_tuning").glob("*.json"))
    if not receipt_paths:
        return {
            "status": "not_started",
            "output_dir": str(output_dir),
            "checkpoints": discover_checkpoints(output_dir),
        }
    with receipt_paths[-1].open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    result["checkpoints"] = discover_checkpoints(output_dir)
    return result
