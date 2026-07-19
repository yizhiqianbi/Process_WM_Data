from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from fastwam_preprocess.adapters import AdapterOptions
from fastwam_preprocess.cleaning import CleaningPolicy, clean_manifest
from fastwam_preprocess.materialize import materialize_canonical
from fastwam_preprocess.registry import ADAPTERS, DEFAULT_ROOTS
from fastwam_preprocess.training_case import build_training_cases
from fastwam_preprocess.utils import write_json
from fastwam_preprocess.windows import build_window_index


def run_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    root = args.output_root / dataset
    input_root = args.input_roots.get(dataset, DEFAULT_ROOTS[dataset])
    scan_summary = ADAPTERS[dataset](
        AdapterOptions(
            input_root=input_root,
            output_root=root / "scan",
            release=args.release,
            min_frames=args.num_frames,
            max_episodes=args.max_episodes,
            verify_files=args.verify_files,
        )
    ).run()
    clean_summary = clean_manifest(
        root / "scan" / "episodes.jsonl",
        root / "clean",
        min_frames=args.num_frames,
        max_episodes=args.max_episodes,
        check_videos=args.check_videos or args.decode_videos,
        decode_videos=args.decode_videos,
        cleaning_policy=(
            CleaningPolicy.from_yaml(args.cleaning_policy)
            if args.cleaning_policy
            else None
        ),
    )
    materialize_summary = materialize_canonical(
        root / "clean" / "episodes.cleaned.jsonl",
        root / "canonical",
        max_episodes=args.max_episodes,
        target_fps=args.target_fps,
    )
    window_summary = build_window_index(
        root / "canonical" / "canonical_episodes.jsonl",
        root / "windows",
        num_frames=args.num_frames,
        stride=args.stride,
        action_video_freq_ratio=args.action_video_freq_ratio,
        minimum_unique_video_frames=args.minimum_unique_video_frames,
    )
    case_summary = build_training_cases(
        root / "windows" / "windows.jsonl",
        root / "cases",
        profiles_path=args.training_profiles,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    summary = {
        "dataset": dataset,
        "scan": scan_summary,
        "clean": clean_summary,
        "materialize": materialize_summary,
        "windows": window_summary,
        "cases": case_summary,
    }
    write_json(root / "pipeline_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the complete FastWAM preprocessing pipeline")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*sorted(ADAPTERS), "all"],
        default=["robocoin", "galaxea"],
    )
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT / "work" / "v1")
    parser.add_argument("--release", default="local")
    parser.add_argument(
        "--input-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Override a dataset input root; may be repeated",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--action-video-freq-ratio", type=int, default=4)
    parser.add_argument("--minimum-unique-video-frames", type=int, default=8)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--check-videos", action="store_true")
    parser.add_argument("--decode-videos", action="store_true")
    parser.add_argument("--cleaning-policy", type=Path)
    parser.add_argument("--training-profiles", type=Path)
    parser.add_argument("--split-seed", default="fastwam-v1")
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    args = parser.parse_args(argv)

    args.input_roots = {}
    for value in args.input_root:
        dataset, separator, path = value.partition("=")
        if not separator or dataset not in ADAPTERS or not path:
            parser.error(f"invalid --input-root {value!r}; expected DATASET=PATH")
        if dataset in args.input_roots:
            parser.error(f"duplicate --input-root for {dataset}")
        args.input_roots[dataset] = Path(path).expanduser().resolve()

    datasets = sorted(ADAPTERS) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_dataset, args, dataset): dataset for dataset in datasets}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                results[dataset] = future.result()
            except Exception as exc:
                failures[dataset] = str(exc)
    combined = {"datasets": results, "failures": failures}
    write_json(args.output_root / "pipeline_summary.json", combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
