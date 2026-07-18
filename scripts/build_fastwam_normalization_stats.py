#!/usr/bin/env python3
"""Build train-split, mask-aware canonical z-score statistics for FastWAM."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


SCHEMA_VERSION = "fastwam-canonical-normalization-v1"
CANONICAL_DIM = 80
DATASETS = (
    "oxe",
    "oxe_auge",
    "agibot_beta",
    "lingbot_va",
    "dreamzero",
    "robocoin",
    "robomind",
    "galaxea",
    "interndata_a1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=[], help="TrainingCaseV1 JSONL; repeatable")
    parser.add_argument(
        "--pipeline-root",
        help="Pipeline output root containing DATASET/cases/training_cases.jsonl",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=[*DATASETS, "all"],
        help="Datasets discovered below --pipeline-root; default: all",
    )
    parser.add_argument("--data-root", required=True, help="Root used to resolve relative canonical_parquet paths")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--split", action="append", default=None, help="Included split; repeatable, default: train")
    parser.add_argument(
        "--training-mode",
        action="append",
        default=None,
        help="Included training mode; repeatable, default: joint_video_action",
    )
    parser.add_argument(
        "--quality-tier",
        action="append",
        default=None,
        help="Included quality tier; repeatable, default: A",
    )
    parser.add_argument("--min-std", type=float, default=1.0e-3, help="Floor for active-dimension std")
    return parser.parse_args()


def resolve_manifests(args: argparse.Namespace) -> list[Path]:
    manifests = [Path(value).expanduser() for value in args.manifest]
    if args.pipeline_root:
        root = Path(args.pipeline_root).expanduser()
        selected = DATASETS if "all" in args.datasets else tuple(dict.fromkeys(args.datasets))
        manifests.extend(root / name / "cases" / "training_cases.jsonl" for name in selected)
    manifests = list(dict.fromkeys(manifests))
    if not manifests:
        raise ValueError("Provide at least one --manifest or --pipeline-root")
    missing = [path for path in manifests if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"TrainingCaseV1 manifests do not exist: {missing}")
    return manifests


def resolve_path(value: str, data_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else data_root / path


def new_accumulator() -> dict[str, np.ndarray]:
    return {
        "count": np.zeros(CANONICAL_DIM, dtype=np.int64),
        "sum": np.zeros(CANONICAL_DIM, dtype=np.float64),
        "sum_sq": np.zeros(CANONICAL_DIM, dtype=np.float64),
    }


def update(accumulator: dict[str, np.ndarray], values: np.ndarray, masks: np.ndarray) -> None:
    if values.shape != masks.shape or values.ndim != 2 or values.shape[1] != CANONICAL_DIM:
        raise ValueError(f"Expected values/masks [N,{CANONICAL_DIM}], got {values.shape}/{masks.shape}")
    finite = np.isfinite(values)
    valid = masks & finite
    invalid_active = masks & ~finite
    if invalid_active.any():
        rows, dims = np.where(invalid_active)
        raise ValueError(f"Non-finite values in active canonical slots, examples={list(zip(rows[:8], dims[:8]))}")
    accumulator["count"] += valid.sum(axis=0, dtype=np.int64)
    masked = np.where(valid, values, 0.0).astype(np.float64, copy=False)
    accumulator["sum"] += masked.sum(axis=0, dtype=np.float64)
    accumulator["sum_sq"] += np.square(masked).sum(axis=0, dtype=np.float64)


def finalize(accumulator: dict[str, np.ndarray], min_std: float) -> dict[str, Any]:
    count = accumulator["count"]
    valid = count > 0
    mean = np.zeros(CANONICAL_DIM, dtype=np.float64)
    variance = np.zeros(CANONICAL_DIM, dtype=np.float64)
    mean[valid] = accumulator["sum"][valid] / count[valid]
    variance[valid] = accumulator["sum_sq"][valid] / count[valid] - np.square(mean[valid])
    variance = np.maximum(variance, 0.0)
    std = np.ones(CANONICAL_DIM, dtype=np.float64)
    std[valid] = np.maximum(np.sqrt(variance[valid]), min_std)
    return {
        "count": count.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "valid_mask": valid.tolist(),
    }


def valid_window_starts(case: dict[str, Any]) -> list[int]:
    sampling = case.get("sampling") or {}
    if sampling.get("unit") != "episode_start_range":
        raise ValueError(
            f"Unsupported TrainingCase sampling unit for case_id={case.get('case_id')}: "
            f"{sampling.get('unit')}"
        )
    valid = sampling.get("valid_starts") or {}
    start = int(valid["start"])
    stop = int(valid["stop_exclusive"])
    stride = int(valid["stride"])
    if start < 0 or stop <= start or stride <= 0:
        raise ValueError(
            f"Invalid valid_starts for case_id={case.get('case_id')}: {valid}"
        )
    starts = list(range(start, stop, stride))
    if len(starts) != int(valid["count"]):
        raise ValueError(
            f"valid_starts count mismatch for case_id={case.get('case_id')}: "
            f"declared={valid['count']}, expanded={len(starts)}"
        )
    return starts


def main() -> None:
    args = parse_args()
    if args.min_std <= 0:
        raise ValueError("--min-std must be positive")
    data_root = Path(args.data_root).expanduser()
    included_splits = set(args.split or ["train"])
    included_modes = set(args.training_mode or ["joint_video_action"])
    included_tiers = set(args.quality_tier or ["A"])
    manifests = resolve_manifests(args)

    cases: list[dict[str, Any]] = []
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                case = json.loads(line)
                if case.get("schema_version") != "fastwam-training-case-v1":
                    raise ValueError(f"Unsupported schema at {manifest}:{line_number}")
                mode = str((case.get("training") or {}).get("mode"))
                tier = str((case.get("quality") or {}).get("tier"))
                if (
                    str(case.get("split")) in included_splits
                    and mode in included_modes
                    and tier in included_tiers
                ):
                    cases.append(case)
    if not cases:
        raise ValueError(
            "No TrainingCaseV1 records match "
            f"splits={sorted(included_splits)}, modes={sorted(included_modes)}, "
            f"tiers={sorted(included_tiers)}"
        )

    episode_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        domain = str(case["embodiment"]["normalization_domain"])
        path = resolve_path(case["inputs"]["canonical_parquet"], data_root)
        dedupe_key = (domain, str(path.resolve()))
        episode_groups.setdefault(dedupe_key, []).append(case)

    domains: dict[str, dict[str, Any]] = {}
    for (domain, resolved_path), episode_cases in episode_groups.items():
        path = Path(resolved_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        case = episode_cases[0]
        inputs = case["inputs"]
        input_contract = {
            key: inputs[key]
            for key in (
                "state_column",
                "state_mask_column",
                "action_column",
                "action_mask_column",
                "state_slot_mask",
                "action_slot_mask",
            )
        }
        for other in episode_cases[1:]:
            other_inputs = other["inputs"]
            other_contract = {key: other_inputs[key] for key in input_contract}
            if other_contract != input_contract:
                raise ValueError(
                    "TrainingCase input contract changed within one canonical episode: "
                    f"{path}"
                )

        domain_acc = domains.setdefault(
            domain,
            {
                "state": new_accumulator(),
                "action": new_accumulator(),
                "episode_count": 0,
                "row_count": 0,
                "selected_state_row_count": 0,
                "selected_action_row_count": 0,
                "window_count": 0,
                "datasets": set(),
                "embodiments": set(),
            },
        )
        table = pq.read_table(
            path,
            columns=[
                inputs["state_column"],
                inputs["state_mask_column"],
                inputs["action_column"],
                inputs["action_mask_column"],
            ],
        )
        state = np.asarray(table[inputs["state_column"]].to_pylist(), dtype=np.float64)
        state_mask = np.asarray(table[inputs["state_mask_column"]].to_pylist(), dtype=np.bool_)
        action = np.asarray(table[inputs["action_column"]].to_pylist(), dtype=np.float64)
        action_mask = np.asarray(table[inputs["action_mask_column"]].to_pylist(), dtype=np.bool_)
        state_mask &= np.asarray(inputs["state_slot_mask"], dtype=np.bool_)[None, :]
        action_mask &= np.asarray(inputs["action_slot_mask"], dtype=np.bool_)[None, :]

        state_rows = np.zeros(table.num_rows, dtype=np.bool_)
        action_rows = np.zeros(table.num_rows, dtype=np.bool_)
        starts = sorted(
            {
                start
                for episode_case in episode_cases
                for start in valid_window_starts(episode_case)
            }
        )
        for episode_case in episode_cases:
            timeline = episode_case.get("timeline") or {}
            if int(timeline.get("state_steps", 0)) != 81 or int(
                timeline.get("action_steps", 0)
            ) != 80:
                raise ValueError(
                    "Normalization currently requires the FastWAM 81/80 timeline: "
                    f"case_id={episode_case.get('case_id')}"
                )
        for start in starts:
            if start + 81 > table.num_rows:
                raise IndexError(
                    f"Training window exceeds canonical episode: path={path}, "
                    f"start={start}, rows={table.num_rows}"
                )
            state_rows[start : start + 81] = True
            action_rows[start : start + 80] = True

        update(domain_acc["state"], state[state_rows], state_mask[state_rows])
        update(domain_acc["action"], action[action_rows], action_mask[action_rows])
        domain_acc["episode_count"] += 1
        domain_acc["row_count"] += table.num_rows
        domain_acc["selected_state_row_count"] += int(state_rows.sum())
        domain_acc["selected_action_row_count"] += int(action_rows.sum())
        domain_acc["window_count"] += len(starts)
        domain_acc["datasets"].update(str(item["dataset"]) for item in episode_cases)
        domain_acc["embodiments"].update(
            str(item["embodiment"]["name"]) for item in episode_cases
        )

    output_domains: dict[str, Any] = {}
    for domain, accumulator in sorted(domains.items()):
        output_domains[domain] = {
            "datasets": sorted(accumulator["datasets"]),
            "embodiments": sorted(accumulator["embodiments"]),
            "episode_count": int(accumulator["episode_count"]),
            "row_count": int(accumulator["row_count"]),
            "selected_state_row_count": int(accumulator["selected_state_row_count"]),
            "selected_action_row_count": int(accumulator["selected_action_row_count"]),
            "window_count": int(accumulator["window_count"]),
            "state": finalize(accumulator["state"], args.min_std),
            "action": finalize(accumulator["action"], args.min_std),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": "zscore",
        "scope": "normalization_domain_and_active_canonical_slot",
        "row_selection": "unique_rows_covered_by_admitted_training_windows",
        "source_splits": sorted(included_splits),
        "source_training_modes": sorted(included_modes),
        "source_quality_tiers": sorted(included_tiers),
        "minimum_std": float(args.min_std),
        "manifests": [str(path) for path in manifests],
        "domains": output_domains,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "domains": len(output_domains),
                "episodes": len(episode_groups),
                "cases": len(cases),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
