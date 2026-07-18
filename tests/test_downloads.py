import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastwam_preprocess.downloads import (
    DownloadLock,
    DownloadOptions,
    EventWriter,
    RepositorySpec,
    _download_one,
    _selected_repositories,
    collect_status,
    load_source_config,
    read_lock,
    resolve_sources,
    write_lock,
)


class FakeApi:
    def __init__(self):
        self.fixed = {
            "example/fixed": SimpleNamespace(
                id="example/fixed",
                sha="a" * 40,
                gated="auto",
                private=False,
                last_modified=None,
            )
        }

    def dataset_info(self, repo_id, revision=None, token=None):
        del revision, token
        return self.fixed[repo_id]

    def list_datasets(self, author=None, full=None, token=None):
        del full, token
        if author != "example-org":
            return []
        return [
            SimpleNamespace(
                id="example-org/zeta",
                sha="c" * 40,
                gated=False,
                private=False,
                last_modified=None,
            ),
            SimpleNamespace(
                id="example-org/alpha",
                sha="b" * 40,
                gated=False,
                private=False,
                last_modified=None,
            ),
        ]


def make_repo(name="repo", revision=None):
    return RepositorySpec(
        dataset="sample",
        display_name="Sample",
        group_dir="SampleGroup",
        repo_id=f"example/{name}",
        local_name=name,
        revision=revision or "d" * 40,
    )


class DownloadConfigTests(unittest.TestCase):
    def test_checked_in_sources_cover_all_supported_datasets(self):
        project_root = Path(__file__).resolve().parents[1]
        config = load_source_config(project_root / "configs" / "download_sources.yaml")
        self.assertEqual(
            set(config["datasets"]),
            {
                "oxe",
                "oxe_auge",
                "agibot_beta",
                "lingbot_va",
                "dreamzero",
                "robocoin",
                "robomind",
                "galaxea",
                "interndata_a1",
            },
        )

    def test_resolve_fixed_and_author_sources_is_sorted_and_pinned(self):
        config = {
            "datasets": {
                "fixed": {
                    "display_name": "Fixed",
                    "group_dir": "FixedGroup",
                    "source": {
                        "type": "fixed",
                        "repositories": [
                            {"repo_id": "example/fixed", "local_name": "fixed-local"}
                        ],
                    },
                },
                "dynamic": {
                    "display_name": "Dynamic",
                    "group_dir": "DynamicGroup",
                    "source": {"type": "author", "author": "example-org"},
                },
            }
        }
        repos = resolve_sources(config, ["all"], FakeApi(), "secret")
        self.assertEqual(len(repos), 3)
        self.assertEqual([repo.dataset for repo in repos], ["dynamic", "dynamic", "fixed"])
        self.assertEqual([repo.local_name for repo in repos[:2]], ["alpha", "zeta"])
        self.assertTrue(all(len(repo.revision) == 40 for repo in repos))

    def test_lock_round_trip_and_selection(self):
        repos = (make_repo("alpha"), make_repo("beta"))
        lock = DownloadLock("2026-01-01T00:00:00+00:00", "f" * 64, repos)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock.json"
            write_lock(path, lock)
            loaded = read_lock(path)
        self.assertEqual(loaded.repositories, repos)
        selected = _selected_repositories(loaded, ["sample"], "*beta", None)
        self.assertEqual([repo.local_name for repo in selected], ["beta"])


class DownloadExecutionTests(unittest.TestCase):
    def test_success_marker_allows_exact_revision_skip(self):
        repo = make_repo()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            state_root = root / "state"
            events_path = root / "events.jsonl"
            events_path.touch()
            options = DownloadOptions(
                data_root=data_root,
                state_root=state_root,
                cache_dir=state_root / "cache",
                token="secret",
                endpoint=None,
                repo_jobs=1,
                file_workers=2,
                attempts=1,
                retry_delay=0,
                etag_timeout=10,
                force_download=False,
                recheck_complete=False,
            )
            calls = []

            def fake_snapshot_download(**kwargs):
                calls.append(kwargs)
                local_dir = Path(kwargs["local_dir"])
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "payload.bin").write_bytes(b"complete")
                return str(local_dir)

            with patch("huggingface_hub.snapshot_download", fake_snapshot_download):
                first = _download_one(repo, options, EventWriter(events_path))
                second = _download_one(repo, options, EventWriter(events_path))

            self.assertEqual(first["status"], "ok")
            self.assertEqual(second["status"], "skipped_complete")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["revision"], repo.revision)
            self.assertNotIn("secret", events_path.read_text(encoding="utf-8"))

    def test_successful_snapshot_tolerates_stale_incomplete_cache_blob(self):
        repo = make_repo("stale-cache")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            state_root = root / "state"
            events_path = root / "events.jsonl"
            events_path.touch()
            options = DownloadOptions(
                data_root=data_root,
                state_root=state_root,
                cache_dir=state_root / "cache",
                token=None,
                endpoint=None,
                repo_jobs=1,
                file_workers=1,
                attempts=1,
                retry_delay=0,
                etag_timeout=10,
                force_download=False,
                recheck_complete=False,
            )

            def fake_snapshot_download(**kwargs):
                local_dir = Path(kwargs["local_dir"])
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "payload.bin").write_bytes(b"complete")
                cache = local_dir / ".cache" / "huggingface" / "download"
                cache.mkdir(parents=True)
                (cache / "old-revision.incomplete").write_bytes(b"stale")
                return str(local_dir)

            with patch("huggingface_hub.snapshot_download", fake_snapshot_download):
                first = _download_one(repo, options, EventWriter(events_path))
                second = _download_one(repo, options, EventWriter(events_path))

            self.assertEqual(first["status"], "ok")
            self.assertEqual(len(first["incomplete_cache_files"]), 1)
            self.assertEqual(second["status"], "skipped_complete")

    def test_existing_data_without_marker_is_present_untracked(self):
        repo = make_repo()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_dir = repo.local_dir(root / "data")
            local_dir.mkdir(parents=True)
            (local_dir / "README.md").write_text("ok", encoding="utf-8")
            report = collect_status([repo], root / "data", root / "state")
        self.assertEqual(report["counts"], {"present_untracked": 1})

    def test_incomplete_blob_is_reported_as_partial(self):
        repo = make_repo()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_dir = repo.local_dir(root / "data")
            cache = local_dir / ".cache" / "huggingface" / "download"
            cache.mkdir(parents=True)
            (cache / "blob.incomplete").write_bytes(b"partial")
            report = collect_status([repo], root / "data", root / "state")
        self.assertEqual(report["counts"], {"partial": 1})
        self.assertEqual(len(report["repositories"][0]["incomplete_files"]), 1)


if __name__ == "__main__":
    unittest.main()
