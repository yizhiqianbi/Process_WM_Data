from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters import AdapterOptions
from .cleaning import CleaningPolicy, clean_manifest
from .materialize import materialize_canonical
from .registry import ADAPTERS, DEFAULT_ROOTS
from .training_case import build_training_cases
from .windows import build_window_index

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_ROOT = PACKAGE_ROOT / "work"


def _print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _scan(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset
    input_root = Path(args.input_root) if args.input_root else DEFAULT_ROOTS[dataset]
    output_root = Path(args.output_root) if args.output_root else DEFAULT_WORK_ROOT / dataset / "scan"
    options = AdapterOptions(
        input_root=input_root,
        output_root=output_root,
        release=args.release,
        min_frames=args.min_frames,
        max_episodes=args.max_episodes,
        verify_files=args.verify_files,
    )
    return ADAPTERS[dataset](options).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified catalog, cleaning, canonicalization, and windowing for FastWAM"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="build source episode and artifact manifests")
    scan.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    scan.add_argument("--input-root")
    scan.add_argument("--output-root")
    scan.add_argument("--release", default="local")
    scan.add_argument("--min-frames", type=int, default=81)
    scan.add_argument("--max-episodes", type=int)
    scan.add_argument("--verify-files", action="store_true")

    clean = commands.add_parser("clean", help="deep-audit a generated episode manifest")
    clean.add_argument("--manifest", type=Path, required=True)
    clean.add_argument("--output-root", type=Path, required=True)
    clean.add_argument("--min-frames", type=int, default=81)
    clean.add_argument("--max-episodes", type=int)
    clean.add_argument("--check-videos", action="store_true")
    clean.add_argument("--decode-videos", action="store_true")
    clean.add_argument("--policy", type=Path)

    windows = commands.add_parser("windows", help="generate valid FastWAM 81-step starts")
    windows.add_argument("--manifest", type=Path, required=True)
    windows.add_argument("--output-root", type=Path, required=True)
    windows.add_argument("--num-frames", type=int, default=81)
    windows.add_argument("--stride", type=int, default=40)
    windows.add_argument("--action-video-freq-ratio", type=int, default=4)
    windows.add_argument("--minimum-unique-video-frames", type=int, default=8)
    windows.add_argument("--expanded", action="store_true")

    materialize = commands.add_parser(
        "materialize", help="write 80D canonical state/action Parquet sidecars"
    )
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    materialize.add_argument("--max-episodes", type=int)
    materialize.add_argument("--a-only", action="store_true")
    materialize.add_argument("--target-fps", type=float, default=20.0)

    cases = commands.add_parser(
        "cases", help="build compact unified FastWAM TrainingCaseV1 records"
    )
    cases.add_argument("--windows-manifest", type=Path, required=True)
    cases.add_argument("--output-root", type=Path, required=True)
    cases.add_argument("--profiles", type=Path)
    cases.add_argument("--split-seed", default="fastwam-v1")
    cases.add_argument("--train-fraction", type=float, default=0.98)
    cases.add_argument("--validation-fraction", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        summary = _scan(args)
    elif args.command == "clean":
        summary = clean_manifest(
            args.manifest,
            args.output_root,
            min_frames=args.min_frames,
            max_episodes=args.max_episodes,
            check_videos=args.check_videos or args.decode_videos,
            decode_videos=args.decode_videos,
            cleaning_policy=(
                CleaningPolicy.from_yaml(args.policy) if args.policy else None
            ),
        )
    elif args.command == "windows":
        summary = build_window_index(
            args.manifest,
            args.output_root,
            num_frames=args.num_frames,
            stride=args.stride,
            action_video_freq_ratio=args.action_video_freq_ratio,
            expanded=args.expanded,
            minimum_unique_video_frames=args.minimum_unique_video_frames,
        )
    elif args.command == "materialize":
        summary = materialize_canonical(
            args.manifest,
            args.output_root,
            max_episodes=args.max_episodes,
            include_tier_b=not args.a_only,
            target_fps=args.target_fps,
        )
    elif args.command == "cases":
        summary = build_training_cases(
            args.windows_manifest,
            args.output_root,
            profiles_path=args.profiles,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        )
    else:
        raise AssertionError(args.command)
    _print_summary(summary)


if __name__ == "__main__":
    main()
