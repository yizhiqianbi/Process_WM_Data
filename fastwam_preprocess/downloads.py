from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import os
import re
import stat
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = PROJECT_ROOT / "configs" / "download_sources.yaml"
DEFAULT_LOCK = PROJECT_ROOT / "configs" / "download_manifest.lock.json"
LOCK_SCHEMA = "fastwam-download-lock-v1"
SOURCE_SCHEMA = "fastwam-download-sources-v1"
STATE_SCHEMA = "fastwam-download-state-v1"
LOG = logging.getLogger("fastwam.downloads")
_SENSITIVE_VALUES: set[str] = set()


class DownloadConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RepositorySpec:
    dataset: str
    display_name: str
    group_dir: str
    repo_id: str
    local_name: str
    revision: str
    repo_type: str = "dataset"
    gated: Any = None
    private: bool | None = None
    last_modified: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepositorySpec":
        required = (
            "dataset",
            "display_name",
            "group_dir",
            "repo_id",
            "local_name",
            "revision",
        )
        missing = [key for key in required if not value.get(key)]
        if missing:
            raise DownloadConfigError(f"lock repository is missing fields: {missing}")
        _validate_component(str(value["group_dir"]), "group_dir")
        _validate_component(str(value["local_name"]), "local_name")
        return cls(
            dataset=str(value["dataset"]),
            display_name=str(value["display_name"]),
            group_dir=str(value["group_dir"]),
            repo_id=str(value["repo_id"]),
            local_name=str(value["local_name"]),
            revision=str(value["revision"]),
            repo_type=str(value.get("repo_type", "dataset")),
            gated=value.get("gated"),
            private=value.get("private"),
            last_modified=value.get("last_modified"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "display_name": self.display_name,
            "group_dir": self.group_dir,
            "repo_id": self.repo_id,
            "local_name": self.local_name,
            "revision": self.revision,
            "repo_type": self.repo_type,
            "gated": self.gated,
            "private": self.private,
            "last_modified": self.last_modified,
        }

    def local_dir(self, data_root: Path) -> Path:
        return data_root / self.group_dir / self.local_name


@dataclass(frozen=True)
class DownloadLock:
    generated_at_utc: str
    source_config_sha256: str
    repositories: tuple[RepositorySpec, ...]

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return tuple(sorted({repo.dataset for repo in self.repositories}))

    def to_dict(self) -> dict[str, Any]:
        counts = Counter(repo.dataset for repo in self.repositories)
        return {
            "schema_version": LOCK_SCHEMA,
            "generated_at_utc": self.generated_at_utc,
            "source_config_sha256": self.source_config_sha256,
            "repository_count": len(self.repositories),
            "counts_by_dataset": dict(sorted(counts.items())),
            "repositories": [repo.to_dict() for repo in self.repositories],
        }


@dataclass(frozen=True)
class DownloadOptions:
    data_root: Path
    state_root: Path
    cache_dir: Path
    token: str | None
    endpoint: str | None
    repo_jobs: int
    file_workers: int
    attempts: int
    retry_delay: float
    etag_timeout: float
    force_download: bool
    recheck_complete: bool


class EventWriter:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def append(self, value: Mapping[str, Any]) -> None:
        line = json.dumps(value, sort_keys=True, ensure_ascii=True)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_component(value: str, field: str) -> None:
    path = PurePath(value)
    if not value or value in {".", ".."} or path.name != value or len(path.parts) != 1:
        raise DownloadConfigError(f"{field} must be one safe path component: {value!r}")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _as_iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def load_source_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != SOURCE_SCHEMA:
        raise DownloadConfigError(f"unsupported source config schema in {path}")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise DownloadConfigError(f"source config has no datasets: {path}")
    for name, value in datasets.items():
        _validate_component(str(name), "dataset name")
        if not isinstance(value, dict):
            raise DownloadConfigError(f"dataset {name!r} must be a mapping")
        _validate_component(str(value.get("group_dir", "")), f"{name}.group_dir")
        source = value.get("source")
        if not isinstance(source, dict) or source.get("type") not in {"fixed", "author"}:
            raise DownloadConfigError(f"dataset {name!r} has an invalid source")
    return raw


def read_lock(path: Path) -> DownloadLock:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != LOCK_SCHEMA:
        raise DownloadConfigError(f"unsupported download lock schema in {path}")
    repositories = tuple(RepositorySpec.from_dict(item) for item in raw["repositories"])
    _validate_unique_destinations(repositories)
    return DownloadLock(
        generated_at_utc=str(raw["generated_at_utc"]),
        source_config_sha256=str(raw["source_config_sha256"]),
        repositories=repositories,
    )


def write_lock(path: Path, lock: DownloadLock) -> None:
    _validate_unique_destinations(lock.repositories)
    _atomic_write_json(path, lock.to_dict())


def _validate_unique_destinations(repositories: Iterable[RepositorySpec]) -> None:
    destinations: dict[tuple[str, str], str] = {}
    repo_ids: set[str] = set()
    for repo in repositories:
        destination = (repo.group_dir, repo.local_name)
        previous = destinations.get(destination)
        if previous and previous != repo.repo_id:
            raise DownloadConfigError(
                f"local destination collision: {destination} for {previous} and {repo.repo_id}"
            )
        if repo.repo_id in repo_ids:
            raise DownloadConfigError(f"duplicate repository in lock: {repo.repo_id}")
        destinations[destination] = repo.repo_id
        repo_ids.add(repo.repo_id)


def _select_names(available: Iterable[str], requested: Sequence[str]) -> list[str]:
    choices = sorted(set(available))
    if not requested or requested == ["all"] or "all" in requested:
        return choices
    unknown = sorted(set(requested) - set(choices))
    if unknown:
        raise DownloadConfigError(f"unknown datasets {unknown}; available: {choices}")
    return list(dict.fromkeys(requested))


def _repo_from_info(
    *,
    dataset: str,
    display_name: str,
    group_dir: str,
    repo_id: str,
    local_name: str,
    info: Any,
) -> RepositorySpec:
    revision = getattr(info, "sha", None)
    if not revision:
        raise DownloadConfigError(f"Hugging Face returned no commit SHA for {repo_id}")
    _validate_component(local_name, f"{repo_id}.local_name")
    return RepositorySpec(
        dataset=dataset,
        display_name=display_name,
        group_dir=group_dir,
        repo_id=repo_id,
        local_name=local_name,
        revision=str(revision),
        gated=getattr(info, "gated", None),
        private=getattr(info, "private", None),
        last_modified=_as_iso(getattr(info, "last_modified", None)),
    )


def resolve_sources(
    source_config: Mapping[str, Any],
    requested_datasets: Sequence[str],
    api: Any,
    token: str | None,
) -> tuple[RepositorySpec, ...]:
    datasets = source_config["datasets"]
    selected = _select_names(datasets, requested_datasets)
    repositories: list[RepositorySpec] = []
    for dataset in selected:
        entry = datasets[dataset]
        display_name = str(entry.get("display_name", dataset))
        group_dir = str(entry["group_dir"])
        source = entry["source"]
        if source["type"] == "author":
            author = str(source["author"])
            infos = sorted(
                api.list_datasets(author=author, full=True, token=token),
                key=lambda item: item.id.lower(),
            )
            if not infos:
                raise DownloadConfigError(f"no datasets found for HF author {author!r}")
            for info in infos:
                repo_id = str(info.id)
                local_name = repo_id.split("/", 1)[-1]
                repositories.append(
                    _repo_from_info(
                        dataset=dataset,
                        display_name=display_name,
                        group_dir=group_dir,
                        repo_id=repo_id,
                        local_name=local_name,
                        info=info,
                    )
                )
        else:
            fixed = source.get("repositories")
            if not isinstance(fixed, list) or not fixed:
                raise DownloadConfigError(f"fixed source {dataset!r} has no repositories")
            for item in fixed:
                repo_id = str(item["repo_id"])
                requested_revision = item.get("revision")
                info = api.dataset_info(
                    repo_id,
                    revision=requested_revision,
                    token=token,
                )
                repositories.append(
                    _repo_from_info(
                        dataset=dataset,
                        display_name=display_name,
                        group_dir=group_dir,
                        repo_id=repo_id,
                        local_name=str(item.get("local_name") or repo_id.split("/", 1)[-1]),
                        info=info,
                    )
                )
    repositories.sort(key=lambda item: (item.dataset, item.repo_id.lower()))
    _validate_unique_destinations(repositories)
    return tuple(repositories)


def build_lock(
    source_path: Path,
    requested_datasets: Sequence[str],
    token: str | None,
    endpoint: str | None = None,
) -> DownloadLock:
    from huggingface_hub import HfApi

    source_config = load_source_config(source_path)
    api = HfApi(token=token, endpoint=endpoint)
    repositories = resolve_sources(source_config, requested_datasets, api, token)
    return DownloadLock(
        generated_at_utc=utc_now(),
        source_config_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        repositories=repositories,
    )


def read_token(token_file: Path | None) -> str | None:
    if token_file is None:
        configured = os.environ.get("HF_TOKEN_FILE")
        token_file = Path(configured).expanduser() if configured else None
    if token_file is not None:
        token_file = token_file.expanduser()
        token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise DownloadConfigError(f"empty token file: {token_file}")
        if os.name == "posix":
            mode = stat.S_IMODE(token_file.stat().st_mode)
            if mode & 0o077:
                LOG.warning("token file permissions are broader than 0600: %s", token_file)
        _SENSITIVE_VALUES.add(token)
        return token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        _SENSITIVE_VALUES.add(token)
    return token


def default_data_root(value: str | None) -> Path:
    configured = value or os.environ.get("FASTWAM_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT.parent.resolve()


def _selected_repositories(
    lock: DownloadLock,
    requested_datasets: Sequence[str],
    repo_pattern: str | None,
    max_repos: int | None,
) -> list[RepositorySpec]:
    names = _select_names(lock.dataset_names, requested_datasets)
    selected = [repo for repo in lock.repositories if repo.dataset in names]
    if repo_pattern:
        selected = [
            repo
            for repo in selected
            if fnmatch.fnmatch(repo.repo_id, repo_pattern)
            or fnmatch.fnmatch(repo.local_name, repo_pattern)
        ]
    if max_repos is not None:
        if max_repos < 1:
            raise DownloadConfigError("--max-repos must be positive")
        selected = selected[:max_repos]
    if not selected:
        raise DownloadConfigError("repository selection is empty")
    return selected


def _safe_repo_name(repo: RepositorySpec) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", repo.repo_id).strip("._-")
    digest = hashlib.sha256(repo.repo_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:100]}-{digest}"


def _marker_path(state_root: Path, repo: RepositorySpec) -> Path:
    return state_root / "repos" / repo.dataset / f"{_safe_repo_name(repo)}.json"


def _read_marker(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _incomplete_files(local_dir: Path, limit: int = 20) -> list[str]:
    cache = local_dir / ".cache" / "huggingface" / "download"
    if not cache.exists():
        return []
    found: list[str] = []
    for path in cache.rglob("*.incomplete"):
        try:
            found.append(str(path.relative_to(local_dir)))
        except ValueError:
            found.append(str(path))
        if len(found) >= limit:
            break
    return found


def _complete_marker_matches(
    marker: Mapping[str, Any] | None,
    repo: RepositorySpec,
    local_dir: Path,
) -> bool:
    return bool(
        marker
        and marker.get("status") == "ok"
        and marker.get("revision") == repo.revision
        and local_dir.is_dir()
    )


def classify_hf_error(exc: Exception) -> tuple[str, bool]:
    class_name = type(exc).__name__.lower()
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if "gated" in class_name or status_code == 403:
        return "needs_approval", False
    if status_code == 401:
        return "authentication_failed", False
    if "repositorynotfound" in class_name or status_code == 404:
        return "not_found_or_no_access", False
    if status_code == 429:
        return "rate_limited", True
    if status_code is not None and status_code >= 500:
        return f"http_{status_code}", True
    if status_code is not None and 400 <= status_code < 500:
        return f"http_{status_code}", status_code in {408, 409, 425}
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "network_error", True
    if any(term in class_name for term in ("timeout", "connection", "proxy")):
        return "network_error", True
    return "error", True


def _clean_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    for sensitive in _SENSITIVE_VALUES:
        text = text.replace(sensitive, "<redacted>")
    return text[:2000]


def _download_one(
    repo: RepositorySpec,
    options: DownloadOptions,
    event_writer: EventWriter,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    local_dir = repo.local_dir(options.data_root)
    marker_path = _marker_path(options.state_root, repo)
    previous_marker = _read_marker(marker_path)
    if not options.recheck_complete and _complete_marker_matches(previous_marker, repo, local_dir):
        result = {
            "status": "skipped_complete",
            "dataset": repo.dataset,
            "repo_id": repo.repo_id,
            "revision": repo.revision,
            "local_dir": str(local_dir),
            "finished_at_utc": utc_now(),
        }
        event_writer.append(result)
        return result

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    base_marker: dict[str, Any] = {
        "schema_version": STATE_SCHEMA,
        "status": "running",
        "dataset": repo.dataset,
        "repo_id": repo.repo_id,
        "repo_type": repo.repo_type,
        "revision": repo.revision,
        "local_dir": str(local_dir),
        "started_at_utc": utc_now(),
    }
    _atomic_write_json(marker_path, base_marker)
    last_error: Exception | None = None
    error_status = "error"
    attempts_used = 0
    for attempt in range(1, options.attempts + 1):
        attempts_used = attempt
        try:
            snapshot_download(
                repo_id=repo.repo_id,
                repo_type=repo.repo_type,
                revision=repo.revision,
                cache_dir=options.cache_dir,
                local_dir=local_dir,
                token=options.token,
                endpoint=options.endpoint,
                max_workers=options.file_workers,
                etag_timeout=options.etag_timeout,
                force_download=options.force_download,
                library_name="fastwam-preprocess",
            )
            incomplete = _incomplete_files(local_dir)
            result = {
                **base_marker,
                "status": "ok",
                "attempts_used": attempts_used,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "finished_at_utc": utc_now(),
                # Old revisions can leave harmless partial cache blobs. The pinned
                # snapshot result and remote file verification are authoritative.
                "incomplete_cache_files": incomplete,
            }
            _atomic_write_json(marker_path, result)
            event_writer.append(result)
            return result
        except Exception as exc:  # noqa: BLE001 - remote failures are classified and persisted
            last_error = exc
            error_status, retryable = classify_hf_error(exc)
            LOG.warning(
                "download attempt %d/%d failed for %s: %s",
                attempt,
                options.attempts,
                repo.repo_id,
                _clean_error(exc),
            )
            if attempt >= options.attempts or not retryable:
                break
            time.sleep(options.retry_delay * attempt)
    result = {
        **base_marker,
        "status": error_status,
        "attempts_used": attempts_used,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "finished_at_utc": utc_now(),
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
        "error": _clean_error(last_error) if last_error else "unknown download failure",
        "incomplete_files": _incomplete_files(local_dir),
    }
    _atomic_write_json(marker_path, result)
    event_writer.append(result)
    return result


def _configure_runtime(state_root: Path, download_timeout: int, quiet_progress: bool) -> None:
    hf_home = state_root / "hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(download_timeout))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if quiet_progress:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def run_download(
    repositories: Sequence[RepositorySpec],
    options: DownloadOptions,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    events_path.touch()
    event_writer = EventWriter(events_path)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=options.repo_jobs) as pool:
        futures = {
            pool.submit(_download_one, repo, options, event_writer): repo
            for repo in repositories
        }
        for future in as_completed(futures):
            repo = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve an unexpected worker failure
                result = {
                    "status": "worker_error",
                    "dataset": repo.dataset,
                    "repo_id": repo.repo_id,
                    "revision": repo.revision,
                    "local_dir": str(repo.local_dir(options.data_root)),
                    "error_type": type(exc).__name__,
                    "error": _clean_error(exc),
                    "finished_at_utc": utc_now(),
                }
                event_writer.append(result)
            results.append(result)
            print(f"{result['status']}\t{repo.dataset}\t{repo.repo_id}", flush=True)
    counts = Counter(result["status"] for result in results)
    failures = [
        result
        for result in results
        if result["status"] not in {"ok", "skipped_complete"}
    ]
    summary = {
        "schema_version": STATE_SCHEMA,
        "started_at_utc": datetime.fromtimestamp(
            time.time() - (time.monotonic() - started), timezone.utc
        ).isoformat(),
        "finished_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "repository_count": len(repositories),
        "counts": dict(sorted(counts.items())),
        "failures": failures,
        "data_root": str(options.data_root),
        "state_root": str(options.state_root),
        "repo_jobs": options.repo_jobs,
        "file_workers": options.file_workers,
        "events": str(events_path),
    }
    _atomic_write_json(run_dir / "summary.json", summary)
    return summary


def repository_status(
    repo: RepositorySpec,
    data_root: Path,
    state_root: Path,
) -> dict[str, Any]:
    local_dir = repo.local_dir(data_root)
    marker = _read_marker(_marker_path(state_root, repo))
    incomplete = _incomplete_files(local_dir)
    if not local_dir.exists():
        status = "missing"
    elif marker and marker.get("status") == "ok" and marker.get("revision") == repo.revision:
        status = "complete"
    elif incomplete:
        status = "partial"
    elif marker and marker.get("status") not in {None, "ok", "running"}:
        status = "failed"
    elif marker and marker.get("status") == "running":
        status = "interrupted_or_running"
    else:
        status = "present_untracked"
    return {
        "status": status,
        "dataset": repo.dataset,
        "repo_id": repo.repo_id,
        "revision": repo.revision,
        "local_dir": str(local_dir),
        "marker_status": marker.get("status") if marker else None,
        "marker_revision": marker.get("revision") if marker else None,
        "incomplete_files": incomplete,
    }


def collect_status(
    repositories: Sequence[RepositorySpec],
    data_root: Path,
    state_root: Path,
) -> dict[str, Any]:
    rows = [repository_status(repo, data_root, state_root) for repo in repositories]
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_dataset[row["dataset"]][row["status"]] += 1
    return {
        "checked_at_utc": utc_now(),
        "data_root": str(data_root),
        "state_root": str(state_root),
        "repository_count": len(rows),
        "counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "counts_by_dataset": {
            name: dict(sorted(counts.items())) for name, counts in sorted(by_dataset.items())
        },
        "repositories": rows,
    }


def _verify_one(repo: RepositorySpec, data_root: Path, token: str | None, endpoint: str | None):
    from huggingface_hub import HfApi

    local_dir = repo.local_dir(data_root)
    started = time.monotonic()
    if not local_dir.is_dir():
        return {
            "status": "missing",
            "dataset": repo.dataset,
            "repo_id": repo.repo_id,
            "local_dir": str(local_dir),
        }
    try:
        info = HfApi(token=token, endpoint=endpoint).dataset_info(
            repo.repo_id,
            revision=repo.revision,
            files_metadata=True,
            token=token,
        )
        missing: list[str] = []
        size_mismatches: list[dict[str, Any]] = []
        expected_bytes = 0
        checked_sizes = 0
        siblings = list(info.siblings or [])
        for sibling in siblings:
            relative = sibling.rfilename
            path = local_dir / relative
            if not path.is_file():
                missing.append(relative)
                continue
            expected_size = getattr(sibling, "size", None)
            if expected_size is not None:
                expected_bytes += int(expected_size)
                checked_sizes += 1
                actual_size = path.stat().st_size
                if actual_size != expected_size:
                    size_mismatches.append(
                        {
                            "path": relative,
                            "expected": expected_size,
                            "actual": actual_size,
                        }
                    )
        incomplete = _incomplete_files(local_dir)
        status = "ok" if not missing and not size_mismatches else "invalid"
        return {
            "status": status,
            "dataset": repo.dataset,
            "repo_id": repo.repo_id,
            "revision": repo.revision,
            "resolved_revision": getattr(info, "sha", None),
            "local_dir": str(local_dir),
            "expected_file_count": len(siblings),
            "checked_size_count": checked_sizes,
            "expected_bytes_with_metadata": expected_bytes,
            "missing_count": len(missing),
            "missing": missing[:100],
            "size_mismatch_count": len(size_mismatches),
            "size_mismatches": size_mismatches[:100],
            "incomplete_cache_files": incomplete,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 - remote failures are part of the report
        status, _ = classify_hf_error(exc)
        return {
            "status": status,
            "dataset": repo.dataset,
            "repo_id": repo.repo_id,
            "revision": repo.revision,
            "local_dir": str(local_dir),
            "error_type": type(exc).__name__,
            "error": _clean_error(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def verify_repositories(
    repositories: Sequence[RepositorySpec],
    data_root: Path,
    token: str | None,
    endpoint: str | None,
    workers: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_verify_one, repo, data_root, token, endpoint): repo
            for repo in repositories
        }
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"{row['status']}\t{row['dataset']}\t{row['repo_id']}", flush=True)
    results.sort(key=lambda row: (row["dataset"], row["repo_id"]))
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        by_dataset[row["dataset"]][row["status"]] += 1
    return {
        "checked_at_utc": utc_now(),
        "data_root": str(data_root),
        "repository_count": len(results),
        "counts": dict(sorted(Counter(row["status"] for row in results).items())),
        "counts_by_dataset": {
            name: dict(sorted(counts.items())) for name, counts in sorted(by_dataset.items())
        },
        "repositories": results,
    }


def _check_access_one(repo: RepositorySpec, token: str | None, endpoint: str | None):
    from huggingface_hub import HfApi

    try:
        info = HfApi(token=token, endpoint=endpoint).dataset_info(
            repo.repo_id,
            revision=repo.revision,
            token=token,
        )
        return {
            "status": "ok",
            "dataset": repo.dataset,
            "repo_id": repo.repo_id,
            "revision": repo.revision,
            "resolved_revision": getattr(info, "sha", None),
            "gated": getattr(info, "gated", None),
        }
    except Exception as exc:  # noqa: BLE001 - access failures are classified for the user
        status, _ = classify_hf_error(exc)
        return {
            "status": status,
            "dataset": repo.dataset,
            "repo_id": repo.repo_id,
            "revision": repo.revision,
            "error_type": type(exc).__name__,
            "error": _clean_error(exc),
        }


def check_access(
    repositories: Sequence[RepositorySpec],
    token: str | None,
    endpoint: str | None,
    workers: int,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    identity: str | None = None
    if token:
        try:
            whoami = HfApi(token=token, endpoint=endpoint).whoami(token=token)
            identity = whoami.get("name") or whoami.get("fullname")
        except Exception as exc:  # noqa: BLE001 - repository checks still provide useful detail
            LOG.warning("could not resolve token identity: %s", _clean_error(exc))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_check_access_one, repo, token, endpoint): repo
            for repo in repositories
        }
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"{row['status']}\t{row['dataset']}\t{row['repo_id']}", flush=True)
    results.sort(key=lambda row: (row["dataset"], row["repo_id"]))
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        by_dataset[row["dataset"]][row["status"]] += 1
    return {
        "checked_at_utc": utc_now(),
        "token_identity": identity,
        "repository_count": len(results),
        "counts": dict(sorted(Counter(row["status"] for row in results).items())),
        "counts_by_dataset": {
            name: dict(sorted(counts.items())) for name, counts in sorted(by_dataset.items())
        },
        "repositories": results,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _common_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--repo-pattern")
    parser.add_argument("--max-repos", type=_positive_int)


def _network_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible, resumable Hugging Face downloads for FastWAM datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve", help="resolve HF orgs and write a commit lock")
    resolve.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    resolve.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    resolve.add_argument("--datasets", nargs="+", default=["all"])
    _network_options(resolve)

    access = subparsers.add_parser("access", help="check token access for locked repositories")
    _common_selection(access)
    _network_options(access)
    access.add_argument("--workers", type=_positive_int, default=16)
    access.add_argument("--data-root")
    access.add_argument("--output", type=Path)

    download = subparsers.add_parser("download", help="download or resume locked repositories")
    _common_selection(download)
    _network_options(download)
    download.add_argument("--data-root")
    download.add_argument("--state-root", type=Path)
    download.add_argument("--repo-jobs", type=_positive_int, default=4)
    download.add_argument("--file-workers", type=_positive_int, default=4)
    download.add_argument("--attempts", type=_positive_int, default=5)
    download.add_argument("--retry-delay", type=float, default=60.0)
    download.add_argument("--etag-timeout", type=float, default=60.0)
    download.add_argument("--download-timeout", type=_positive_int, default=300)
    download.add_argument("--force-download", action="store_true")
    download.add_argument("--recheck-complete", action="store_true")
    download.add_argument("--dry-run", action="store_true")

    status_parser = subparsers.add_parser("status", help="inspect local and recorded state")
    _common_selection(status_parser)
    status_parser.add_argument("--data-root")
    status_parser.add_argument("--state-root", type=Path)
    status_parser.add_argument("--output", type=Path)

    verify = subparsers.add_parser("verify", help="compare local files with locked HF metadata")
    _common_selection(verify)
    _network_options(verify)
    verify.add_argument("--data-root")
    verify.add_argument("--workers", type=_positive_int, default=4)
    verify.add_argument("--output", type=Path)

    return parser


def _load_selection(args: argparse.Namespace) -> tuple[DownloadLock, list[RepositorySpec]]:
    lock = read_lock(args.lock.expanduser().resolve())
    repositories = _selected_repositories(
        lock,
        args.datasets,
        args.repo_pattern,
        args.max_repos,
    )
    return lock, repositories


def _state_root(args: argparse.Namespace, data_root: Path) -> Path:
    value = getattr(args, "state_root", None)
    return (value.expanduser().resolve() if value else data_root / ".fastwam_download")


def _print_status_summary(report: Mapping[str, Any]) -> None:
    summary = {
        "repository_count": report["repository_count"],
        "counts": report["counts"],
    }
    if report.get("counts_by_dataset") is not None:
        summary["counts_by_dataset"] = report["counts_by_dataset"]
    print(json.dumps(summary, indent=2, sort_keys=True))


def _default_report_path(data_root: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return data_root / ".fastwam_download" / "reports" / f"{prefix}_{stamp}.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "resolve":
            token = read_token(args.token_file)
            lock = build_lock(
                args.sources.expanduser().resolve(),
                args.datasets,
                token,
                args.endpoint,
            )
            write_lock(args.lock.expanduser().resolve(), lock)
            print(json.dumps({
                "lock": str(args.lock.expanduser().resolve()),
                "repository_count": len(lock.repositories),
                "counts_by_dataset": lock.to_dict()["counts_by_dataset"],
            }, indent=2, sort_keys=True))
            return 0

        _, repositories = _load_selection(args)
        if args.command == "access":
            report = check_access(
                repositories,
                read_token(args.token_file),
                args.endpoint,
                args.workers,
            )
            output = args.output or _default_report_path(
                default_data_root(args.data_root), "access"
            )
            _atomic_write_json(output.expanduser().resolve(), report)
            _print_status_summary(report)
            print(f"report: {output.expanduser().resolve()}")
            return 0 if set(report["counts"]) <= {"ok"} else 2

        data_root = default_data_root(args.data_root)
        if args.command == "download":
            state_root = _state_root(args, data_root)
            _configure_runtime(state_root, args.download_timeout, args.repo_jobs > 1)
            if args.dry_run:
                for repo in repositories:
                    print(
                        f"{repo.dataset}\t{repo.repo_id}@{repo.revision}\t"
                        f"{repo.local_dir(data_root)}"
                    )
                print(f"repositories: {len(repositories)}")
                return 0
            token = read_token(args.token_file)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_dir = state_root / "runs" / f"download_{stamp}_{os.getpid()}"
            selected_lock = DownloadLock(
                generated_at_utc=utc_now(),
                source_config_sha256="selection-from-lock",
                repositories=tuple(repositories),
            )
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            options = DownloadOptions(
                data_root=data_root,
                state_root=state_root,
                cache_dir=state_root / "hf_cache",
                token=token,
                endpoint=args.endpoint,
                repo_jobs=args.repo_jobs,
                file_workers=args.file_workers,
                attempts=args.attempts,
                retry_delay=args.retry_delay,
                etag_timeout=args.etag_timeout,
                force_download=args.force_download,
                recheck_complete=args.recheck_complete,
            )
            options.cache_dir.mkdir(parents=True, exist_ok=True)
            summary = run_download(repositories, options, run_dir)
            write_lock(run_dir / "selected_manifest.lock.json", selected_lock)
            print(json.dumps({
                "run_dir": str(run_dir),
                "counts": summary["counts"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }, indent=2, sort_keys=True))
            return 0 if not summary["failures"] else 2

        if args.command == "status":
            state_root = _state_root(args, data_root)
            report = collect_status(repositories, data_root, state_root)
            if args.output:
                _atomic_write_json(args.output.expanduser().resolve(), report)
            _print_status_summary(report)
            return 0

        if args.command == "verify":
            report = verify_repositories(
                repositories,
                data_root,
                read_token(args.token_file),
                args.endpoint,
                args.workers,
            )
            output = args.output or _default_report_path(data_root, "verify")
            _atomic_write_json(output.expanduser().resolve(), report)
            _print_status_summary(report)
            print(f"report: {output.expanduser().resolve()}")
            return 0 if set(report["counts"]) <= {"ok"} else 2
    except (DownloadConfigError, FileNotFoundError, PermissionError) as exc:
        LOG.error("%s", exc)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
