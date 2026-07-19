#!/usr/bin/env python3
"""Run DreamZero with causal GT observation feedback and render video pairs."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from targets.dreamzero.inference import (
    CAMERA_KEYS,
    PHYSICAL_CAMERA_LABELS,
    SOURCE_CAMERA_KEYS,
    chunk_segments,
    comparison_frame_ids,
    decode_context_latent_bounds,
    decoded_frame_count,
    flatten_action_output,
    observation_frame_ids,
    render_all_views_pair,
    render_single_view_pair,
    select_episodes,
    split_composite,
)
from targets.video_pair import (
    SequentialFrameReader,
    action_metrics,
    as_uint8_video,
    concatenate_videos,
    decode_frames,
    lerobot_video_path,
    parse_int_list,
    probe_video,
    read_json,
    read_jsonl,
    video_metrics,
    write_json,
    write_video,
)


RECEIPT_VERSION = "dreamzero-pair-inference-v2"
SINGLE_VIEW_HEIGHT = 176
SINGLE_VIEW_WIDTH = 320
NUM_FRAME_PER_BLOCK = 2
SOURCE_FRAMES_PER_CHUNK = 8
FEEDBACK_FRAMES = 4
EXECUTED_ACTION_STEPS = 8
STATE_JOINT_INDICES = tuple(range(7))
STATE_GRIPPER_INDEX = 14
GT_TRANSFORM_BATCH_FRAMES = 4
DEFAULT_CACHE_WINDOW_CHUNKS = 24


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DreamZero on at least eight episodes. The first observation and "
            "every subsequent four-frame feedback window come from GT; predicted "
            "latents are never fed back as observations."
        )
    )
    parser.add_argument("--dreamzero-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", default="", help="Comma-separated episode IDs")
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=114)
    parser.add_argument(
        "--cache-window-chunks",
        type=int,
        default=DEFAULT_CACHE_WINDOW_CHUNKS,
        help="Bound the DiT KV cache and GT VAE context independently of rollout length.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--gpus", default="0,1", help="Exactly two physical GPU IDs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument(
        "--attention-backend", choices=("FA2", "FA3", "torch", "TE"), default="FA2"
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--distributed-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _validate_paths(args: argparse.Namespace) -> None:
    required_directories = {
        "DreamZero repository": args.dreamzero_repo,
        "DreamZero target": args.dataset_root,
        "DreamZero base model": args.base_model_root,
        "DreamZero checkpoint": args.checkpoint,
    }
    for label, path in required_directories.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{label} is missing: {path}")
    for path in (
        args.checkpoint / "model.safetensors",
        args.checkpoint / "config.json",
        args.checkpoint / "experiment_cfg" / "conf.yaml",
        args.checkpoint / "experiment_cfg" / "metadata.json",
        args.base_model_root / "model.safetensors.index.json",
        args.dataset_root / "meta" / "info.json",
        args.dataset_root / "meta" / "episodes.jsonl",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required inference artifact is missing: {path}")


def _launch_distributed(args: argparse.Namespace) -> None:
    gpu_ids = parse_int_list(args.gpus)
    if len(gpu_ids) != 2:
        raise SystemExit("DreamZero classifier-free guidance requires exactly two GPUs")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpu_ids)
    environment["ATTENTION_BACKEND"] = args.attention_backend
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--distributed-worker",
    ]
    print("Launching: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _episode_parquet(dataset_root: Path, episode_index: int) -> Path:
    path = (
        dataset_root
        / "data"
        / f"chunk-{episode_index // 1000:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"episode parquet is missing: {path}")
    return path


def _load_episode_arrays(
    dataset_root: Path, episode_index: int
) -> tuple[np.ndarray, np.ndarray, str]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(
        _episode_parquet(dataset_root, episode_index),
        columns=["observation.state", "action", "annotation.task"],
    )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    prompts = table["annotation.task"].to_pylist()
    prompt = next((str(value) for value in prompts if value), "")
    if not prompt:
        raise RuntimeError(f"episode {episode_index} has no annotation.task")
    if states.ndim != 2 or states.shape[1] <= STATE_GRIPPER_INDEX:
        raise RuntimeError(f"unexpected state shape for episode {episode_index}: {states.shape}")
    if actions.shape != states.shape:
        raise RuntimeError(
            f"state/action shape mismatch for episode {episode_index}: "
            f"{states.shape} versus {actions.shape}"
        )
    return states, actions, prompt


class _EpisodeObservationReader:
    def __init__(self, dataset_root: Path, episode_index: int) -> None:
        self.readers = {
            model_key: SequentialFrameReader(
                lerobot_video_path(
                    dataset_root, SOURCE_CAMERA_KEYS[model_key], episode_index
                )
            )
            for model_key in CAMERA_KEYS
        }

    def read(self, frame_ids: list[int]) -> dict[str, np.ndarray]:
        return {key: reader.read(frame_ids) for key, reader in self.readers.items()}

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()

    def __enter__(self) -> "_EpisodeObservationReader":
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()


def _build_observation(
    camera_frames: dict[str, np.ndarray],
    states: np.ndarray,
    *,
    state_frame: int,
    prompt: str,
) -> dict[str, Any]:
    observation: dict[str, Any] = dict(camera_frames)
    observation["state.right_joint_position"] = np.ascontiguousarray(
        states[state_frame : state_frame + 1, :7]
    )
    observation["state.right_gripper_position"] = np.ascontiguousarray(
        states[state_frame : state_frame + 1, 14:15]
    )
    observation["annotation.task"] = prompt
    return observation


def _reset_causal_state(policy: Any, *, seed: int) -> None:
    head = policy.trained_model.action_head
    head.current_start_frame = 0
    head.language = None
    head.clip_feas = None
    head.ys = None
    head.kv_cache1 = None
    head.kv_cache_neg = None
    head.crossattn_cache = None
    head.crossattn_cache_neg = None
    head.seed = seed


def _processed_gt_composite(
    camera_frames: dict[str, np.ndarray],
    *,
    episode_index: int,
    batch_frames: int = GT_TRANSFORM_BATCH_FRAMES,
) -> np.ndarray:
    import torch
    import torch.nn.functional as torch_functional

    if batch_frames <= 0:
        raise ValueError("batch_frames must be positive")
    frame_counts = {len(value) for value in camera_frames.values()}
    if len(frame_counts) != 1:
        raise RuntimeError(f"GT cameras have different frame counts: {frame_counts}")
    frame_count = frame_counts.pop()
    parts: list[np.ndarray] = []
    # Calling the upstream composed transform after compiled inference can leave
    # torchvision waiting on a stale CPU worker pool. This is the exact eval-only
    # image path from the XDOF profile: 0.95 center crop, antialiased bilinear
    # resize to 176x320, uint8 conversion, then DreamTransform's 2x2 layout.
    torch.set_num_threads(1)
    for batch_start in range(0, frame_count, batch_frames):
        batch_end = min(batch_start + batch_frames, frame_count)
        transformed_cameras: dict[str, np.ndarray] = {}
        for key, value in camera_frames.items():
            batch = torch.from_numpy(
                np.ascontiguousarray(value[batch_start:batch_end])
            ).permute(0, 3, 1, 2)
            batch = batch.to(dtype=torch.float32).div_(255.0)
            source_height, source_width = batch.shape[-2:]
            crop_height = int(source_height * 0.95)
            crop_width = int(source_width * 0.95)
            crop_top = (source_height - crop_height) // 2
            crop_left = (source_width - crop_width) // 2
            batch = batch[
                :,
                :,
                crop_top : crop_top + crop_height,
                crop_left : crop_left + crop_width,
            ]
            batch = torch_functional.interpolate(
                batch,
                size=(SINGLE_VIEW_HEIGHT, SINGLE_VIEW_WIDTH),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            transformed_cameras[key] = (
                batch.permute(0, 2, 3, 1).mul_(255.0).to(torch.uint8).numpy()
            )

        images = np.zeros(
            (
                batch_end - batch_start,
                SINGLE_VIEW_HEIGHT * 2,
                SINGLE_VIEW_WIDTH * 2,
                3,
            ),
            dtype=np.uint8,
        )
        images[:, :SINGLE_VIEW_HEIGHT, :SINGLE_VIEW_WIDTH] = transformed_cameras[
            CAMERA_KEYS[0]
        ]
        images[:, SINGLE_VIEW_HEIGHT:, :SINGLE_VIEW_WIDTH] = transformed_cameras[
            CAMERA_KEYS[1]
        ]
        images[:, :SINGLE_VIEW_HEIGHT, SINGLE_VIEW_WIDTH:] = transformed_cameras[
            CAMERA_KEYS[2]
        ]
        expected = (
            batch_end - batch_start,
            SINGLE_VIEW_HEIGHT * 2,
            SINGLE_VIEW_WIDTH * 2,
            3,
        )
        if images.shape != expected:
            raise RuntimeError(
                f"GT transform produced {images.shape}, expected {expected}"
            )
        parts.append(np.ascontiguousarray(images))
        print(
            f"episode {episode_index}: GT transform {batch_end}/{frame_count}",
            flush=True,
        )
        del images, transformed_cameras
    return np.concatenate(parts, axis=0)


def _encode_gt_latents(policy: Any, composite: np.ndarray) -> Any:
    import torch

    head = policy.trained_model.action_head
    video = torch.from_numpy(composite).to(device="cuda", dtype=torch.bfloat16)
    video = video.permute(3, 0, 1, 2).unsqueeze(0).div_(255.0)
    batch, channels, frames, height, width = video.shape
    normalized = head.normalize_video(
        video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    )
    normalized = normalized.reshape(batch, frames, channels, height, width).permute(
        0, 2, 1, 3, 4
    )
    with torch.inference_mode():
        return head.vae.encode(
            normalized,
            tiled=head.tiled,
            tile_size=(head.tile_size_height, head.tile_size_width),
            tile_stride=(head.tile_stride_height, head.tile_stride_width),
        )


def _encode_gt_latents_segmented(
    policy: Any,
    composite: np.ndarray,
    *,
    num_chunks: int,
    cache_window_chunks: int,
    episode_index: int,
    postprocess_sync: Any | None = None,
) -> Any:
    import torch

    encoded_parts = []
    segments = chunk_segments(num_chunks, cache_window_chunks)
    for segment_index, (chunk_start, chunk_end) in enumerate(segments):
        frame_start = chunk_start * SOURCE_FRAMES_PER_CHUNK
        frame_count = decoded_frame_count(chunk_end - chunk_start, NUM_FRAME_PER_BLOCK)
        frame_end = frame_start + frame_count
        encode_input = composite[frame_start:frame_end]
        padded_frames = 0
        if len(segments) > 1 and chunk_end - chunk_start < cache_window_chunks:
            compiled_frame_count = decoded_frame_count(
                cache_window_chunks, NUM_FRAME_PER_BLOCK
            )
            padded_frames = compiled_frame_count - frame_count
            if padded_frames > 0:
                # Wan's encoder is causal. Repeating the last visible frame lets the
                # final short segment reuse the full-window compiled graph; padded
                # future latents are discarded immediately.
                encode_input = np.concatenate(
                    (
                        encode_input,
                        np.repeat(encode_input[-1:], padded_frames, axis=0),
                    ),
                    axis=0,
                )
        print(
            f"episode {episode_index}: GT VAE encode segment {segment_index + 1}/"
            f"{len(segments)}, frames={frame_start}:{frame_end}, "
            f"causal_padding={padded_frames}",
            flush=True,
        )
        segment = _encode_gt_latents(policy, encode_input)
        expected_latents = 1 + (chunk_end - chunk_start) * NUM_FRAME_PER_BLOCK
        if int(segment.shape[2]) < expected_latents:
            raise RuntimeError(
                f"GT segment {segment_index} encoded {segment.shape[2]} latents, "
                f"need at least {expected_latents}"
            )
        segment = segment[:, :, :expected_latents]
        if encoded_parts:
            # Adjacent causal segments share their GT anchor frame. Keep the new
            # segment's anchor because it initializes that segment's decoder.
            encoded_parts[-1] = encoded_parts[-1][:, :, :-1]
        encoded_parts.append(segment)
        torch.cuda.synchronize()
        if postprocess_sync is not None:
            postprocess_sync()
    result = torch.cat(encoded_parts, dim=2)
    expected_total = 1 + num_chunks * NUM_FRAME_PER_BLOCK
    if int(result.shape[2]) != expected_total:
        raise RuntimeError(
            f"segmented GT encode produced {result.shape[2]} latents, "
            f"expected {expected_total}"
        )
    return result


def _decode_feedback_chunks(
    policy: Any,
    prediction_chunks: list[Any],
    gt_latents: Any,
    *,
    episode_index: int,
    cache_window_chunks: int,
    postprocess_sync: Any | None = None,
) -> np.ndarray:
    import torch

    head = policy.trained_model.action_head
    parts = []
    for chunk_index, prediction in enumerate(prediction_chunks):
        if chunk_index == 0:
            decode_input = prediction
            trim = 0
            expected = 1 + 4 * NUM_FRAME_PER_BLOCK
        else:
            history_start, history_end = decode_context_latent_bounds(
                chunk_index, cache_window_chunks, NUM_FRAME_PER_BLOCK
            )
            context = gt_latents[:, :, history_start:history_end].to(
                device=prediction.device, dtype=prediction.dtype
            )
            decode_input = torch.cat(
                (context, prediction),
                dim=2,
            )
            trim = 1 + 4 * (int(context.shape[2]) - 1)
            expected = 4 * NUM_FRAME_PER_BLOCK
        print(
            f"episode {episode_index}: VAE decode {chunk_index + 1}/"
            f"{len(prediction_chunks)}, latent_frames={decode_input.shape[2]}",
            flush=True,
        )
        started = time.monotonic()
        with torch.inference_mode():
            decoded = head.vae.decode(
                decode_input,
                tiled=head.tiled,
                tile_size=(head.tile_size_height, head.tile_size_width),
                tile_stride=(head.tile_stride_height, head.tile_stride_width),
            )
        torch.cuda.synchronize()
        print(
            f"episode {episode_index}: VAE decode {chunk_index + 1}/"
            f"{len(prediction_chunks)} complete in {time.monotonic() - started:.1f}s, "
            f"peak_allocated={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
            flush=True,
        )
        decoded = as_uint8_video(decoded)
        part = np.ascontiguousarray(decoded[trim:])
        if len(part) != expected:
            raise RuntimeError(
                f"decoded chunk {chunk_index} has {len(part)} frames, expected {expected}"
            )
        parts.append(part)
        if postprocess_sync is not None:
            postprocess_sync()
    return np.concatenate(parts, axis=0)


def _load_gt_camera_frames(
    dataset_root: Path, episode_index: int, frame_ids: list[int]
) -> dict[str, np.ndarray]:
    return {
        model_key: decode_frames(
            lerobot_video_path(dataset_root, source_key, episode_index), frame_ids
        )
        for model_key, source_key in SOURCE_CAMERA_KEYS.items()
    }


def _gt_executed_actions(
    actions: np.ndarray, *, start_frame: int, num_chunks: int
) -> np.ndarray:
    values = []
    for chunk_index in range(num_chunks):
        anchor = start_frame + chunk_index * SOURCE_FRAMES_PER_CHUNK
        source = actions[anchor : anchor + EXECUTED_ACTION_STEPS]
        if len(source) != EXECUTED_ACTION_STEPS:
            raise RuntimeError(f"not enough GT actions at source frame {anchor}")
        values.append(np.concatenate((source[:, :7], source[:, 14:15]), axis=1))
    return np.concatenate(values, axis=0).astype(np.float32, copy=False)


def _run_case(
    args: argparse.Namespace,
    *,
    policy: Any,
    episode: dict[str, Any],
    rank: int,
) -> dict[str, Any] | None:
    import torch
    import torch.distributed as dist
    from tianshou.data import Batch

    episode_index = int(episode["episode_index"])
    case_dir = args.output_dir / f"episode_{episode_index:06d}"
    receipt_path = case_dir / "receipt.json"
    if receipt_path.is_file() and not args.overwrite:
        if rank == 0:
            print(f"episode {episode_index}: reusing completed receipt", flush=True)
        dist.barrier()
        return read_json(receipt_path) if rank == 0 else None
    case_dir.mkdir(parents=True, exist_ok=True)

    states, actions, prompt = _load_episode_arrays(args.dataset_root, episode_index)
    seed = args.seed + episode_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _reset_causal_state(policy, seed=seed)
    torch.cuda.reset_peak_memory_stats()

    head = policy.trained_model.action_head
    cache_window_chunks = min(args.cache_window_chunks, args.num_chunks)
    cache_segments = chunk_segments(args.num_chunks, cache_window_chunks)
    expected_local_attention = 1 + cache_window_chunks * NUM_FRAME_PER_BLOCK
    if int(head.model.local_attn_size) < expected_local_attention:
        raise RuntimeError(
            f"model local attention is {head.model.local_attn_size}, but rollout needs "
            f"{expected_local_attention}; the LoRA config override was not applied"
        )

    predicted_latents = []
    predicted_actions = []
    inference_times = []
    injected_frame_ids = []
    with _EpisodeObservationReader(args.dataset_root, episode_index) as reader:
        for chunk_index in range(args.num_chunks):
            window_chunk_index = chunk_index % cache_window_chunks
            if chunk_index > 0 and window_chunk_index == 0:
                _reset_causal_state(policy, seed=seed)
                gc.collect()
                torch.cuda.empty_cache()
                if rank == 0:
                    print(
                        f"episode {episode_index}: reset bounded causal cache at "
                        f"chunk {chunk_index}",
                        flush=True,
                    )
            frame_ids = observation_frame_ids(
                chunk_index,
                start_frame=args.start_frame,
                source_frames_per_chunk=SOURCE_FRAMES_PER_CHUNK,
                feedback_frames=FEEDBACK_FRAMES,
            )
            state_frame = args.start_frame + chunk_index * SOURCE_FRAMES_PER_CHUNK
            observation = _build_observation(
                reader.read(frame_ids),
                states,
                state_frame=state_frame,
                prompt=prompt,
            )
            if rank == 0:
                print(
                    f"episode {episode_index}: chunk {chunk_index + 1}/{args.num_chunks}, "
                    f"GT observation frames={frame_ids}",
                    flush=True,
                )
            dist.barrier()
            started = time.monotonic()
            result, video_pred = policy.lazy_joint_forward_causal(
                Batch(obs=observation), latent_video=None
            )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            dist.barrier()
            expected_start = 1 + (window_chunk_index + 1) * NUM_FRAME_PER_BLOCK
            if int(head.current_start_frame) != expected_start:
                raise RuntimeError(
                    f"causal cache advanced to {head.current_start_frame}, expected "
                    f"{expected_start} after chunk {chunk_index}"
                )
            if rank == 0:
                if chunk_index > 0 and window_chunk_index == 0:
                    if int(video_pred.shape[2]) != 1 + NUM_FRAME_PER_BLOCK:
                        raise RuntimeError(
                            f"cache reset chunk returned {video_pred.shape[2]} latents; "
                            f"expected {1 + NUM_FRAME_PER_BLOCK}"
                        )
                    video_pred = video_pred[:, :, 1:]
                predicted_latents.append(video_pred.detach().clone())
                predicted_actions.append(
                    flatten_action_output(
                        result.act, executed_steps=EXECUTED_ACTION_STEPS
                    )
                )
                inference_times.append(elapsed)
                injected_frame_ids.append(frame_ids)
            del result, video_pred

    if rank != 0:
        # Rank zero owns GT processing and rendering. Keep the CFG peer alive with
        # bounded barriers so one long rank-zero postprocess cannot hit NCCL's
        # cumulative 600-second watchdog.
        segment_count = len(cache_segments)
        for _ in range(args.num_chunks + segment_count + 2):
            dist.barrier()
        return None

    frame_ids = comparison_frame_ids(
        start_frame=args.start_frame,
        num_chunks=args.num_chunks,
        num_frame_per_block=NUM_FRAME_PER_BLOCK,
    )
    raw_gt = _load_gt_camera_frames(args.dataset_root, episode_index, frame_ids)
    transform_started = time.monotonic()
    gt_composite = _processed_gt_composite(
        raw_gt,
        episode_index=episode_index,
    )
    print(
        f"episode {episode_index}: transformed {len(gt_composite)} GT frames in "
        f"{time.monotonic() - transform_started:.1f}s",
        flush=True,
    )
    dist.barrier()
    print(
        f"episode {episode_index}: encoding {len(gt_composite)} GT frames for "
        "causal decode context",
        flush=True,
    )
    encode_started = time.monotonic()
    gt_latents = _encode_gt_latents_segmented(
        policy,
        gt_composite,
        num_chunks=args.num_chunks,
        cache_window_chunks=cache_window_chunks,
        episode_index=episode_index,
        postprocess_sync=dist.barrier,
    )
    torch.cuda.synchronize()
    print(
        f"episode {episode_index}: GT context encoded in "
        f"{time.monotonic() - encode_started:.1f}s",
        flush=True,
    )
    prediction = _decode_feedback_chunks(
        policy,
        predicted_latents,
        gt_latents,
        episode_index=episode_index,
        cache_window_chunks=cache_window_chunks,
        postprocess_sync=dist.barrier,
    )
    expected_frames = decoded_frame_count(args.num_chunks, NUM_FRAME_PER_BLOCK)
    if len(prediction) != expected_frames:
        raise RuntimeError(
            f"episode {episode_index} decoded {len(prediction)} frames, expected "
            f"{expected_frames}"
        )
    del predicted_latents, gt_latents
    gc.collect()
    torch.cuda.empty_cache()

    gt_views = split_composite(
        gt_composite,
        single_height=SINGLE_VIEW_HEIGHT,
        single_width=SINGLE_VIEW_WIDTH,
    )
    pred_views = split_composite(
        prediction,
        single_height=SINGLE_VIEW_HEIGHT,
        single_width=SINGLE_VIEW_WIDTH,
    )
    fps = float(read_json(args.dataset_root / "meta" / "info.json")["fps"])
    videos: dict[str, dict[str, Any]] = {}
    all_views_path = case_dir / "gt_vs_dreamzero_all_views.mp4"
    write_video(
        all_views_path,
        render_all_views_pair(
            gt_views, pred_views, episode_index=episode_index, fps=fps
        ),
        fps=fps,
        crf=args.crf,
    )
    videos["all_views"] = probe_video(all_views_path)
    for camera_key in CAMERA_KEYS:
        short_name = camera_key.rsplit(".", 1)[-1]
        path = case_dir / f"gt_vs_dreamzero_{short_name}.mp4"
        write_video(
            path,
            render_single_view_pair(
                gt_views[camera_key],
                pred_views[camera_key],
                camera_key=camera_key,
                episode_index=episode_index,
                fps=fps,
            ),
            fps=fps,
            crf=args.crf,
        )
        videos[camera_key] = probe_video(path)
    for key, value in videos.items():
        if int(value["frames"]) != expected_frames:
            raise RuntimeError(
                f"encoded {key} video has {value['frames']} frames, expected "
                f"{expected_frames}"
            )

    predicted_action_array = np.concatenate(predicted_actions, axis=0)
    gt_action_array = _gt_executed_actions(
        actions, start_frame=args.start_frame, num_chunks=args.num_chunks
    )
    actions_path = case_dir / "gt_and_predicted_actions.npz"
    np.savez_compressed(
        actions_path,
        ground_truth=gt_action_array,
        predicted=predicted_action_array,
        source_action_indices=np.asarray([*range(7), 14], dtype=np.int64),
    )
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "status": "succeeded",
        "episode_index": episode_index,
        "episode_length_source_frames": int(episode["length"]),
        "prompt": prompt,
        "seed": seed,
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "base_model_root": str(args.base_model_root),
        "protocol": {
            "observation_source": "ground_truth",
            "initial_observation_source_frame": args.start_frame,
            "gt_observation_frame_ids_per_chunk": injected_frame_ids,
            "predicted_latents_fed_back_as_observation": False,
            "future_gt_visible_to_current_prediction": False,
            "feedback_frames": FEEDBACK_FRAMES,
            "source_frames_per_chunk": SOURCE_FRAMES_PER_CHUNK,
            "num_chunks": args.num_chunks,
            "cache_window_chunks": cache_window_chunks,
            "cache_reset_chunk_indices": [
                start for start, _ in cache_segments if start > 0
            ],
            "cache_reset_source_frame_ids": [
                args.start_frame + start * SOURCE_FRAMES_PER_CHUNK
                for start, _ in cache_segments
                if start > 0
            ],
            "cache_reset_observation_source": "ground_truth",
            "cache_segments": [
                {
                    "chunk_start": start,
                    "chunk_end_exclusive": end,
                    "source_frame_start": args.start_frame
                    + start * SOURCE_FRAMES_PER_CHUNK,
                    "decoded_source_frames": decoded_frame_count(
                        end - start, NUM_FRAME_PER_BLOCK
                    ),
                    "gt_vae_causal_padding_frames": (
                        decoded_frame_count(cache_window_chunks, NUM_FRAME_PER_BLOCK)
                        - decoded_frame_count(end - start, NUM_FRAME_PER_BLOCK)
                        if len(cache_segments) > 1
                        and end - start < cache_window_chunks
                        else 0
                    ),
                }
                for start, end in cache_segments
            ],
            "latent_frames": 1 + args.num_chunks * NUM_FRAME_PER_BLOCK,
            "decoded_frames": expected_frames,
            "source_frame_ids": frame_ids,
            "comparison_fps": fps,
            "duration_seconds": expected_frames / fps,
            "action_horizon_per_chunk": 24,
            "executed_action_steps_per_chunk": EXECUTED_ACTION_STEPS,
            "gt_context_used_for_video_decode": True,
        },
        "camera_layout": [
            {
                "model_order": index,
                "camera_key": camera_key,
                "source_camera_key": SOURCE_CAMERA_KEYS[camera_key],
                "physical_role": PHYSICAL_CAMERA_LABELS[camera_key],
            }
            for index, camera_key in enumerate(CAMERA_KEYS)
        ],
        "sampling": {
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "attention_backend": args.attention_backend,
            "seconds_per_chunk": inference_times,
            "total_inference_seconds": float(sum(inference_times)),
            "cuda_peak_allocated_gib": float(
                torch.cuda.max_memory_allocated() / 2**30
            ),
            "cuda_peak_reserved_gib": float(
                torch.cuda.max_memory_reserved() / 2**30
            ),
        },
        "video_metrics": video_metrics(gt_views, pred_views),
        "action_metrics": action_metrics(gt_action_array, predicted_action_array),
        "actions": {
            "path": str(actions_path),
            "steps": int(len(predicted_action_array)),
            "channels": int(predicted_action_array.shape[1]),
            "bytes": actions_path.stat().st_size,
        },
        "videos": videos,
    }
    write_json(receipt_path, receipt)
    print(
        f"episode {episode_index}: complete, {expected_frames} frames, "
        f"{sum(inference_times):.1f}s model inference",
        flush=True,
    )
    dist.barrier()
    return receipt


def _load_policy(args: argparse.Namespace, device_mesh: Any) -> Any:
    import torch

    repo = args.dreamzero_repo
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.dreamzero.base_vla import VLA
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    if "base_model_path" not in inspect.signature(VLA.load_lora).parameters:
        raise RuntimeError(
            "DreamZero LoRA inference integration is not installed: "
            "VLA.load_lora lacks base_model_path"
        )
    if "config" not in inspect.signature(VLA.load_lora).parameters:
        raise RuntimeError(
            "DreamZero LoRA inference integration is outdated: "
            "VLA.load_lora lacks config override support"
        )
    torch._dynamo.config.recompile_limit = max(800, args.num_chunks * 80)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.XDOF,
        model_path=str(args.checkpoint),
        device="cuda",
        model_config_overrides=[
            "action_head_cfg.config.diffusion_model_cfg.max_chunk_size="
            f"{min(args.cache_window_chunks, args.num_chunks)}"
        ],
        device_mesh=device_mesh,
        lora_base_model_path=str(args.base_model_root),
    )
    head = policy.trained_model.action_head
    head.num_inference_steps = args.inference_steps
    head.cfg_scale = args.guidance_scale
    return policy


def _aggregate(
    args: argparse.Namespace,
    selected: list[int],
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    fps = float(receipts[0]["protocol"]["comparison_fps"])
    reel_path = args.output_dir / f"dreamzero_{len(receipts)}case_gt_vs_predicted_reel.mp4"
    reel = concatenate_videos(
        [Path(item["videos"]["all_views"]["path"]) for item in receipts],
        reel_path,
        fps=fps,
        crf=args.crf,
    )
    aggregate = {
        "schema_version": RECEIPT_VERSION,
        "status": "succeeded",
        "case_count": len(receipts),
        "episode_indices": selected,
        "dataset_root": str(args.dataset_root),
        "base_model_root": str(args.base_model_root),
        "checkpoint": str(args.checkpoint),
        "protocol": {
            "observation_source": "ground_truth",
            "predicted_latents_fed_back_as_observation": False,
            "num_chunks_per_case": args.num_chunks,
            "cache_window_chunks": min(args.cache_window_chunks, args.num_chunks),
            "frames_per_case": decoded_frame_count(
                args.num_chunks, NUM_FRAME_PER_BLOCK
            ),
            "comparison_fps": fps,
        },
        "reel": reel,
        "cases": receipts,
    }
    write_json(args.output_dir / "inference_receipt.json", aggregate)
    lines = [
        "# DreamZero GT-observation rollouts",
        "",
        "Each prediction starts from GT observation history. Generated latents are not fed back as observations.",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Cases: `{', '.join(str(value) for value in selected)}`",
        f"- Reel: `{reel_path.name}`",
        "",
        "| Episode | Frames | Duration | All-view pair |",
        "|---:|---:|---:|---|",
    ]
    for receipt in receipts:
        relative = Path(receipt["videos"]["all_views"]["path"]).relative_to(
            args.output_dir
        )
        lines.append(
            f"| {receipt['episode_index']} | {receipt['protocol']['decoded_frames']} | "
            f"{receipt['protocol']['duration_seconds']:.2f}s | `{relative}` |"
        )
    (args.output_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def _run_distributed_worker(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"DreamZero inference requires world_size=2, got {world_size}")
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    device_mesh = init_device_mesh(
        device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("ip",)
    )

    episodes = read_jsonl(args.dataset_root / "meta" / "episodes.jsonl")
    expected_frames = decoded_frame_count(args.num_chunks, NUM_FRAME_PER_BLOCK)
    selected = select_episodes(
        episodes,
        requested=parse_int_list(args.episodes),
        num_cases=args.num_cases,
        required_source_frames=expected_frames,
        start_frame=args.start_frame,
    )
    by_id = {int(item["episode_index"]): item for item in episodes}
    if rank == 0:
        print(
            f"selected episodes {selected}; {args.num_chunks} chunks -> "
            f"{expected_frames} frames/case",
            flush=True,
        )
    policy = _load_policy(args, device_mesh)
    receipts = []
    try:
        for episode_id in selected:
            receipt = _run_case(
                args, policy=policy, episode=by_id[episode_id], rank=rank
            )
            if receipt is not None:
                receipts.append(receipt)
        if rank == 0:
            aggregate = _aggregate(args, selected, receipts)
            print(
                json.dumps(
                    {
                        "status": aggregate["status"],
                        "case_count": aggregate["case_count"],
                        "episode_indices": aggregate["episode_indices"],
                        "reel": aggregate["reel"],
                        "receipt": str(args.output_dir / "inference_receipt.json"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
    finally:
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = _parse_args()
    args.dreamzero_repo = args.dreamzero_repo.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.base_model_root = args.base_model_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.num_cases < 1 or (args.num_cases < 8 and not args.shard_worker):
        raise SystemExit("num-cases must be at least 8")
    if args.num_chunks <= 0 or args.cache_window_chunks <= 0 or args.start_frame < 0:
        raise SystemExit(
            "num-chunks and cache-window-chunks must be positive; start-frame must "
            "be nonnegative"
        )
    if decoded_frame_count(args.num_chunks, NUM_FRAME_PER_BLOCK) < 81:
        raise SystemExit("the rollout must contain at least 81 decoded frames")
    if args.inference_steps <= 0 or args.guidance_scale == 1.0:
        raise SystemExit("inference steps must be positive and guidance scale cannot be 1")
    _validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.distributed_worker:
        _run_distributed_worker(args)
    else:
        _launch_distributed(args)


if __name__ == "__main__":
    main()
