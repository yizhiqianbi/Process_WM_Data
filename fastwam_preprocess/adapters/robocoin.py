from __future__ import annotations

from .base import BaseAdapter
from .lerobot import scan_lerobot_repo


class RoboCOINAdapter(BaseAdapter):
    dataset_name = "robocoin"

    def scan(self) -> None:
        repos = sorted(
            path.parent.parent
            for path in self.options.input_root.glob("*/meta/info.json")
        )
        if not repos:
            self.blockers.append("no_lerobot_repositories_found")
            return
        for repo in repos:
            if self.at_limit():
                break
            scan_lerobot_repo(
                self,
                repo,
                release=self.options.release,
                task_namespace=repo.name,
            )
