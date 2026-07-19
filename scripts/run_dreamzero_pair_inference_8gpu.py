#!/usr/bin/env python3
"""Run eight DreamZero GT-observation cases on four two-GPU workers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dreamzero_pair_inference import (  # noqa: E402
    NUM_FRAME_PER_BLOCK,
    _aggregate,
)
from targets.dreamzero.inference import decoded_frame_count, select_episodes  # noqa: E402
from targets.video_pair import parse_int_list, read_json, read_jsonl, write_json  # noqa: E402


class _TerminationRequested(RuntimeError):
    pass


def _raise_termination(signum: int, _frame: Any) -> None:
    raise _TerminationRequested(f"received signal {signum}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eight DreamZero pair-inference cases across eight GPUs."
    )
    parser.add_argument("--dreamzero-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", default="")
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=114)
    parser.add_argument("--cache-window-chunks", type=int, default=24)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument(
        "--attention-backend", choices=("FA2", "FA3", "torch", "TE"), default="FA2"
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _partition(values: list[int], count: int) -> list[list[int]]:
    if count <= 0 or len(values) < count:
        raise ValueError("worker count must be positive and cannot exceed case count")
    return [values[index::count] for index in range(count)]


def _terminate(processes: list[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main() -> None:
    args = _parse_args()
    for name in (
        "dreamzero_repo",
        "dataset_root",
        "base_model_root",
        "checkpoint",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    gpu_ids = parse_int_list(args.gpus)
    if len(gpu_ids) != 8:
        raise SystemExit(f"exactly 8 distinct GPUs are required, got {gpu_ids}")
    if args.num_cases < 8:
        raise SystemExit("num-cases must be at least 8")
    if args.num_chunks <= 0 or args.cache_window_chunks <= 0 or args.start_frame < 0:
        raise SystemExit(
            "num-chunks and cache-window-chunks must be positive; start-frame must "
            "be nonnegative"
        )
    frame_count = decoded_frame_count(args.num_chunks, NUM_FRAME_PER_BLOCK)
    if frame_count < 81:
        raise SystemExit("the rollout must contain at least 81 decoded frames")

    episodes = read_jsonl(args.dataset_root / "meta" / "episodes.jsonl")
    selected = select_episodes(
        episodes,
        requested=parse_int_list(args.episodes),
        num_cases=args.num_cases,
        required_source_frames=frame_count,
        start_frame=args.start_frame,
    )
    selected = selected[: args.num_cases]
    gpu_pairs = [gpu_ids[index : index + 2] for index in range(0, 8, 2)]
    shards = _partition(selected, len(gpu_pairs))
    child_script = Path(__file__).with_name("run_dreamzero_pair_inference.py")
    commands: list[list[str]] = []
    shard_dirs: list[Path] = []
    for shard_index, (gpu_pair, episode_ids) in enumerate(zip(gpu_pairs, shards)):
        shard_dir = args.output_dir / (
            f"shard_{shard_index:02d}_gpu_{gpu_pair[0]}_{gpu_pair[1]}"
        )
        command = [
            sys.executable,
            str(child_script),
            "--dreamzero-repo",
            str(args.dreamzero_repo),
            "--dataset-root",
            str(args.dataset_root),
            "--base-model-root",
            str(args.base_model_root),
            "--checkpoint",
            str(args.checkpoint),
            "--output-dir",
            str(shard_dir),
            "--episodes",
            ",".join(str(value) for value in episode_ids),
            "--num-cases",
            str(len(episode_ids)),
            "--num-chunks",
            str(args.num_chunks),
            "--cache-window-chunks",
            str(args.cache_window_chunks),
            "--start-frame",
            str(args.start_frame),
            "--gpus",
            ",".join(str(value) for value in gpu_pair),
            "--seed",
            str(args.seed),
            "--inference-steps",
            str(args.inference_steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--attention-backend",
            args.attention_backend,
            "--crf",
            str(args.crf),
            "--shard-worker",
        ]
        if args.overwrite:
            command.append("--overwrite")
        commands.append(command)
        shard_dirs.append(shard_dir)

    launch_description = {
        "schema_version": "dreamzero-pair-inference-8gpu-v2",
        "status": "dry_run" if args.dry_run else "running",
        "episode_indices": selected,
        "frames_per_case": frame_count,
        "cache_window_chunks": min(args.cache_window_chunks, args.num_chunks),
        "gpu_pairs": gpu_pairs,
        "shards": shards,
        "commands": commands,
    }
    if args.dry_run:
        print(json.dumps(launch_description, indent=2), flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "launcher_receipt.json", launch_description)
    processes: list[subprocess.Popen[Any]] = []
    log_handles: list[Any] = []
    started = time.time()
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_termination)
    try:
        for shard_index, (command, shard_dir) in enumerate(zip(commands, shard_dirs)):
            shard_dir.mkdir(parents=True, exist_ok=True)
            log_handle = (shard_dir / "inference.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment.setdefault("PYTHONUNBUFFERED", "1")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            log_handles.append(log_handle)
            print(
                f"shard {shard_index}: pid={process.pid} GPUs={gpu_pairs[shard_index]} "
                f"episodes={shards[shard_index]} log={shard_dir / 'inference.log'}",
                flush=True,
            )

        pending = set(range(len(processes)))
        while pending:
            time.sleep(10)
            for index in list(pending):
                return_code = processes[index].poll()
                if return_code is None:
                    continue
                pending.remove(index)
                print(f"shard {index}: exited with code {return_code}", flush=True)
                if return_code != 0:
                    raise RuntimeError(
                        f"DreamZero shard {index} failed; see {shard_dirs[index] / 'inference.log'}"
                    )
    except BaseException as exc:
        _terminate(processes)
        launch_description.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.time() - started,
            }
        )
        write_json(args.output_dir / "launcher_receipt.json", launch_description)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        for handle in log_handles:
            handle.close()

    receipts_by_episode: dict[int, dict[str, Any]] = {}
    shard_receipts = []
    for shard_dir in shard_dirs:
        receipt_path = shard_dir / "inference_receipt.json"
        receipt = read_json(receipt_path)
        if receipt.get("status") != "succeeded":
            raise RuntimeError(f"invalid shard receipt: {receipt_path}")
        shard_receipts.append(str(receipt_path))
        for case in receipt["cases"]:
            episode_index = int(case["episode_index"])
            if episode_index in receipts_by_episode:
                raise RuntimeError(f"duplicate episode receipt: {episode_index}")
            receipts_by_episode[episode_index] = case
    if set(receipts_by_episode) != set(selected):
        raise RuntimeError(
            f"shard outputs do not match selection: got={sorted(receipts_by_episode)}, "
            f"expected={sorted(selected)}"
        )

    aggregate = _aggregate(
        args, selected, [receipts_by_episode[value] for value in selected]
    )
    aggregate["execution"] = {
        "gpu_count": 8,
        "worker_count": 4,
        "gpus_per_worker": 2,
        "gpu_pairs": gpu_pairs,
        "episode_shards": shards,
        "shard_receipts": shard_receipts,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "inference_receipt.json", aggregate)
    launch_description.update(
        {
            "status": "succeeded",
            "elapsed_seconds": time.time() - started,
            "aggregate_receipt": str(args.output_dir / "inference_receipt.json"),
            "reel": aggregate["reel"],
        }
    )
    write_json(args.output_dir / "launcher_receipt.json", launch_description)
    print(json.dumps(launch_description, indent=2), flush=True)


if __name__ == "__main__":
    main()
