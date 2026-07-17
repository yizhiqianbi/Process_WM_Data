from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Iterator, TextIO


def stable_id(*parts: object, length: int = 16) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_.-")
    return text or "unknown"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            yield payload


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


class AtomicJsonlWriter:
    """Stream JSONL to a temporary file and publish it only after a clean close."""

    def __init__(self, path: Path):
        self.path = path
        self._temp_name: str | None = None
        self._handle: TextIO | None = None

    def __enter__(self) -> "AtomicJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, self._temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        self._handle = os.fdopen(fd, "w", encoding="utf-8")
        return self

    def write(self, row: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("AtomicJsonlWriter is not open")
        self._handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        self._handle.write("\n")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        handle = self._handle
        temp_name = self._temp_name
        self._handle = None
        self._temp_name = None
        if handle is None or temp_name is None:
            return
        try:
            if exc_type is None:
                handle.flush()
                os.fsync(handle.fileno())
            handle.close()
            if exc_type is None:
                os.replace(temp_name, self.path)
            else:
                os.unlink(temp_name)
        except Exception:
            try:
                handle.close()
            except Exception:
                pass
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise


def file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def is_partial_path(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in (".incomplete", ".partial", ".tmp", ".lock"))
