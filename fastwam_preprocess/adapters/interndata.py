from __future__ import annotations

from .base import BaseAdapter
from .lerobot import scan_lerobot_repo


class InternDataA1Adapter(BaseAdapter):
    dataset_name = "interndata_a1"

    def scan(self) -> None:
        trajectory_roots = [
            self.options.input_root / name for name in ("real", "sim", "sim_updated")
        ]
        info_files = sorted(
            path
            for root in trajectory_roots
            if root.is_dir()
            for path in root.rglob("meta/info.json")
        )
        for info_path in info_files:
            if self.at_limit():
                break
            repo = info_path.parent.parent
            scan_lerobot_repo(
                self,
                repo,
                release=self.options.release,
                task_namespace=repo.name,
            )

        asset_roots = [self.options.input_root / "InternDataAssets"]
        for asset_root in asset_roots:
            if asset_root.is_dir():
                self.add_artifact(
                    path=asset_root,
                    kind="simulation_assets",
                    complete=True,
                    status="assets_only",
                )

        if not info_files:
            self.blockers.append("trajectory_data_not_present_assets_only")
