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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="TrainingCaseV1 JSONL; repeatable")
    parser.add_argument("--data-root", required=True, help="Root used to resolve relative canonical_parquet paths")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--split", action="append", default=None, help="Included split; repeatable, default: train")
    parser.add_argument("--min-std", type=float, default=1.0e-3, help="Floor for active-dimension std")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    if args.min_std <= 0:
        raise ValueError("--min-std must be positive")
    data_root = Path(args.data_root).expanduser()
    included_splits = set(args.split or ["train"])

    cases: list[dict[str, Any]] = []
    for manifest_value in args.manifest:
        manifest = Path(manifest_value).expanduser()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                case = json.loads(line)
                if case.get("schema_version") != "fastwam-training-case-v1":
                    raise ValueError(f"Unsupported schema at {manifest}:{line_number}")
                if str(case.get("split")) in included_splits:
                    cases.append(case)
    if not cases:
        raise ValueError(f"No TrainingCaseV1 records match splits={sorted(included_splits)}")

    domains: dict[str, dict[str, Any]] = {}
    seen_episodes: set[tuple[str, str]] = set()
    for case in cases:
        domain = str(case["embodiment"]["normalization_domain"])
        path = resolve_path(case["inputs"]["canonical_parquet"], data_root)
        dedupe_key = (domain, str(path.resolve()))
        if dedupe_key in seen_episodes:
            continue
        seen_episodes.add(dedupe_key)
        if not path.is_file():
            raise FileNotFoundError(path)

        domain_acc = domains.setdefault(
            domain,
            {
                "state": new_accumulator(),
                "action": new_accumulator(),
                "episode_count": 0,
                "row_count": 0,
                "datasets": set(),
                "embodiments": set(),
            },
        )
        inputs = case["inputs"]
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
        update(domain_acc["state"], state, state_mask)
        # action[t] supervises state[t] -> state[t+1], so the final row is not a transition.
        update(domain_acc["action"], action[:-1], action_mask[:-1])
        domain_acc["episode_count"] += 1
        domain_acc["row_count"] += table.num_rows
        domain_acc["datasets"].add(str(case["dataset"]))
        domain_acc["embodiments"].add(str(case["embodiment"]["name"]))

    output_domains: dict[str, Any] = {}
    for domain, accumulator in sorted(domains.items()):
        output_domains[domain] = {
            "datasets": sorted(accumulator["datasets"]),
            "embodiments": sorted(accumulator["embodiments"]),
            "episode_count": int(accumulator["episode_count"]),
            "row_count": int(accumulator["row_count"]),
            "state": finalize(accumulator["state"], args.min_std),
            "action": finalize(accumulator["action"], args.min_std),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": "zscore",
        "scope": "normalization_domain_and_active_canonical_slot",
        "source_splits": sorted(included_splits),
        "minimum_std": float(args.min_std),
        "manifests": [str(Path(value).expanduser()) for value in args.manifest],
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
                "episodes": len(seen_episodes),
                "cases": len(cases),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
