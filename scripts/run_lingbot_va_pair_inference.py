#!/usr/bin/env python3
"""Run open-loop old LingBot-VA rollouts and render GT/prediction video pairs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import gc
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from targets.lingbot_va.runtime import install_flash_attention_import_fallback


RECEIPT_VERSION = "lingbot-va-pair-inference-v2"
WAN_TEMPORAL_STRIDE = 4
PHYSICAL_CAMERA_LABELS = {
    "observation.images.left_eye": "GLOBAL / HEAD (left_eye)",
    "observation.images.right_eye": "LEFT WRIST (right_eye)",
    "observation.images.right_wrist": "RIGHT WRIST (right_wrist)",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run old LingBot-VA with either official-style GT observation feedback "
            "or an open-loop ablation, then render synchronized GT/imagined videos."
        )
    )
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", default="", help="Comma-separated episode IDs")
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--observation-mode",
        choices=("gt_chunk_feedback", "open_loop"),
        default="gt_chunk_feedback",
        help=(
            "gt_chunk_feedback writes GT observation/action history into the "
            "official KV cache before every next-chunk prediction"
        ),
    )
    parser.add_argument(
        "--feedback-action-source",
        choices=("predicted", "gt"),
        default="predicted",
        help=(
            "Action history paired with GT observation feedback; predicted matches "
            "the official online client"
        ),
    )
    parser.add_argument("--gpus", default="", help="Comma-separated physical GPU IDs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--action-guidance-scale", type=float, default=1.0)
    parser.add_argument("--video-steps", type=int, default=5)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker-episodes", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, default=-1, help=argparse.SUPPRESS)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(json.loads(line))
    return values


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"duplicate integer IDs: {value}")
    return parsed


def _checkpoint_transformer(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    nested = checkpoint / "transformer"
    transformer = nested if nested.is_dir() else checkpoint
    if not (transformer / "config.json").is_file():
        raise FileNotFoundError(f"transformer config is missing: {transformer}")
    if not any(transformer.glob("*.safetensors")):
        raise FileNotFoundError(f"transformer safetensors are missing: {transformer}")
    return transformer


def _ensure_runtime_model(
    output_dir: Path, base_model_root: Path, checkpoint: Path
) -> Path:
    runtime_root = output_dir / "_runtime_model"
    runtime_root.mkdir(parents=True, exist_ok=True)
    sources = {
        "vae": base_model_root / "vae",
        "text_encoder": base_model_root / "text_encoder",
        "tokenizer": base_model_root / "tokenizer",
        "transformer": _checkpoint_transformer(checkpoint),
    }
    for name, source in sources.items():
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"model component is missing: {source}")
        link = runtime_root / name
        if link.is_symlink():
            if link.resolve() != source:
                raise RuntimeError(f"runtime model link points elsewhere: {link}")
            continue
        if link.exists():
            raise RuntimeError(f"runtime model path is not a symlink: {link}")
        link.symlink_to(source, target_is_directory=True)
    return runtime_root


def _video_frame_count(num_chunks: int, frame_chunk_size: int) -> int:
    latent_frames = num_chunks * frame_chunk_size
    return 1 + WAN_TEMPORAL_STRIDE * (latent_frames - 1)


def _select_episodes(
    episodes: list[dict[str, Any]],
    *,
    requested: list[int],
    num_cases: int,
    required_source_frames: int,
    start_frame: int,
) -> list[int]:
    by_id = {int(item["episode_index"]): item for item in episodes}
    if requested:
        missing = [value for value in requested if value not in by_id]
        if missing:
            raise ValueError(f"unknown episode IDs: {missing}")
        selected = requested
    else:
        eligible = [
            item
            for item in episodes
            if int(item["length"]) - start_frame >= required_source_frames
        ]
        eligible.sort(key=lambda item: (-int(item["length"]), int(item["episode_index"])))
        selected = [int(item["episode_index"]) for item in eligible[:num_cases]]
    if len(selected) < num_cases:
        raise ValueError(
            f"requested at least {num_cases} cases, but only selected {len(selected)}"
        )
    selected = selected[:num_cases]
    too_short = [
        value
        for value in selected
        if int(by_id[value]["length"]) - start_frame < required_source_frames
    ]
    if too_short:
        raise ValueError(
            "episodes are too short for the requested prediction horizon: "
            f"{too_short}; need {required_source_frames} source frames"
        )
    return selected


def _video_path(dataset_root: Path, camera_key: str, episode_index: int) -> Path:
    path = (
        dataset_root
        / "videos"
        / f"chunk-{episode_index // 1000:03d}"
        / camera_key
        / f"episode_{episode_index:06d}.mp4"
    )
    if not path.is_file():
        raise FileNotFoundError(f"source camera video is missing: {path}")
    return path


def _decode_frames(
    path: Path,
    frame_ids: list[int],
    *,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    import av
    import cv2

    wanted = set(frame_ids)
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = next((value for value in container.streams if value.type == "video"), None)
        if stream is None:
            raise RuntimeError(f"video has no stream: {path}")
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index in wanted:
                image = frame.to_ndarray(format="rgb24")
                if output_size is not None:
                    width, height = output_size
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                decoded[frame_index] = np.ascontiguousarray(image)
            if frame_index >= frame_ids[-1] and len(decoded) == len(wanted):
                break
    missing = [value for value in frame_ids if value not in decoded]
    if missing:
        raise RuntimeError(f"{path} is missing requested frames: {missing[:8]}")
    return np.stack([decoded[value] for value in frame_ids])


class _SequentialFrameReader:
    """Decode monotonically increasing frame IDs without reopening the video."""

    def __init__(self, path: Path):
        import av

        self.path = path
        self.container = av.open(str(path))
        self.stream = next(
            (value for value in self.container.streams if value.type == "video"), None
        )
        if self.stream is None:
            self.container.close()
            raise RuntimeError(f"video has no stream: {path}")
        self.frames = iter(self.container.decode(self.stream))
        self.frame_index = -1

    def read(self, frame_ids: list[int]) -> np.ndarray:
        if not frame_ids:
            raise ValueError("frame_ids cannot be empty")
        if frame_ids != sorted(frame_ids) or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("sequential frame IDs must be unique and sorted")
        if frame_ids[0] <= self.frame_index:
            raise ValueError(
                f"cannot seek backwards in {self.path}: current={self.frame_index}, "
                f"requested={frame_ids[0]}"
            )
        wanted = set(frame_ids)
        decoded: dict[int, np.ndarray] = {}
        for frame in self.frames:
            self.frame_index += 1
            if self.frame_index in wanted:
                decoded[self.frame_index] = np.ascontiguousarray(
                    frame.to_ndarray(format="rgb24")
                )
            if self.frame_index >= frame_ids[-1]:
                break
        missing = [value for value in frame_ids if value not in decoded]
        if missing:
            raise RuntimeError(f"{self.path} is missing requested frames: {missing[:8]}")
        return np.stack([decoded[value] for value in frame_ids])

    def close(self) -> None:
        self.container.close()


class _SequentialObservationReader:
    def __init__(
        self, dataset_root: Path, camera_keys: list[str], episode_index: int
    ) -> None:
        self.camera_keys = camera_keys
        self.readers = {
            key: _SequentialFrameReader(_video_path(dataset_root, key, episode_index))
            for key in camera_keys
        }

    def read(self, frame_ids: list[int]) -> dict[str, Any]:
        camera_frames = {
            key: self.readers[key].read(frame_ids) for key in self.camera_keys
        }
        return {
            "obs": [
                {key: camera_frames[key][index] for key in self.camera_keys}
                for index in range(len(frame_ids))
            ]
        }

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()

    def __enter__(self) -> "_SequentialObservationReader":
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()


def _load_initial_observation(
    dataset_root: Path,
    camera_keys: list[str],
    episode_index: int,
    start_frame: int,
) -> dict[str, Any]:
    observation = {}
    for camera_key in camera_keys:
        frames = _decode_frames(
            _video_path(dataset_root, camera_key, episode_index), [start_frame]
        )
        observation[camera_key] = frames[0]
    return {"obs": [observation]}


def _load_gt_views(
    dataset_root: Path,
    camera_keys: list[str],
    episode_index: int,
    frame_ids: list[int],
    *,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    return {
        camera_key: _decode_frames(
            _video_path(dataset_root, camera_key, episode_index),
            frame_ids,
            output_size=(width, height),
        )
        for camera_key in camera_keys
    }


def _as_uint8_video(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    array = np.asarray(value)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise RuntimeError(f"unexpected decoded video shape: {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array))
        if maximum <= 1.5:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _split_prediction(
    prediction: np.ndarray, camera_keys: list[str], width: int, height: int
) -> dict[str, np.ndarray]:
    expected_width = width * len(camera_keys)
    if prediction.shape[1:] != (height, expected_width, 3):
        raise RuntimeError(
            "decoded prediction does not match the trained camera layout: "
            f"got {prediction.shape}, expected [F,{height},{expected_width},3]"
        )
    return {
        camera_key: np.ascontiguousarray(
            prediction[:, :, index * width : (index + 1) * width]
        )
        for index, camera_key in enumerate(camera_keys)
    }


def _put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.43,
    max_width: int | None = None,
) -> None:
    import cv2

    if max_width is not None:
        text_width = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )[0][0]
        if text_width > max_width:
            scale *= max_width / text_width
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def _render_all_views_pair(
    gt_views: dict[str, np.ndarray],
    pred_views: dict[str, np.ndarray],
    camera_keys: list[str],
    *,
    episode_index: int,
    fps: float,
    mode_label: str,
) -> Iterable[np.ndarray]:
    frame_count, height, width, _ = next(iter(gt_views.values())).shape
    header = 46
    full_width = width * len(camera_keys)
    for frame_index in range(frame_count):
        canvas = np.full((2 * (height + header), full_width, 3), 18, dtype=np.uint8)
        for camera_index, camera_key in enumerate(camera_keys):
            left = camera_index * width
            right = left + width
            canvas[header : header + height, left:right] = gt_views[camera_key][frame_index]
            pred_top = height + 2 * header
            canvas[pred_top : pred_top + height, left:right] = pred_views[camera_key][frame_index]
            label = PHYSICAL_CAMERA_LABELS.get(camera_key, camera_key.rsplit(".", 1)[-1])
            _put_text(
                canvas,
                f"GT | {label}",
                (left + 8, 24),
                max_width=width - 16,
            )
            _put_text(
                canvas,
                f"IMAGINED | {label}",
                (left + 8, height + header + 24),
                max_width=width - 16,
            )
            if camera_index:
                canvas[:, left - 1 : left + 1] = 230
        _put_text(
            canvas,
            f"ep {episode_index:06d}  t={frame_index / fps:05.1f}s  {mode_label}",
            (8, 42),
            scale=0.34,
        )
        yield canvas


def _render_single_view_pair(
    gt: np.ndarray,
    prediction: np.ndarray,
    *,
    camera_key: str,
    episode_index: int,
    fps: float,
    mode_label: str,
) -> Iterable[np.ndarray]:
    frame_count, height, width, _ = gt.shape
    header = 46
    label = PHYSICAL_CAMERA_LABELS.get(camera_key, camera_key.rsplit(".", 1)[-1])
    for frame_index in range(frame_count):
        canvas = np.full((height + header, width * 2, 3), 18, dtype=np.uint8)
        canvas[header:, :width] = gt[frame_index]
        canvas[header:, width:] = prediction[frame_index]
        _put_text(canvas, f"GT | {label}", (8, 24), max_width=width - 16)
        _put_text(
            canvas,
            f"IMAGINED | {label}",
            (width + 8, 24),
            max_width=width - 16,
        )
        _put_text(
            canvas,
            f"ep {episode_index:06d}  {frame_index / fps:05.1f}s  {mode_label}",
            (8, 42),
            scale=0.34,
            max_width=width - 16,
        )
        yield canvas


def _write_video(
    path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: float,
    crf: int,
) -> dict[str, Any]:
    import av

    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise RuntimeError(f"cannot write an empty video: {path}") from error
    height, width = first.shape[:2]
    if height % 2 or width % 2:
        raise RuntimeError(f"H.264 output dimensions must be even, got {width}x{height}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.mp4")
    if temporary.exists():
        temporary.unlink()
    rate = Fraction(str(fps)).limit_denominator(1000)
    frame_count = 0
    with av.open(str(temporary), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "medium"}
        for image in itertools.chain((first,), iterator):
            video_frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(image), format="rgb24"
            )
            for packet in stream.encode(video_frame):
                container.mux(packet)
            frame_count += 1
        for packet in stream.encode():
            container.mux(packet)
    temporary.replace(path)
    return {
        "path": str(path),
        "frames": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "bytes": path.stat().st_size,
    }


def _probe_video(path: Path) -> dict[str, Any]:
    import av

    with av.open(str(path)) as container:
        stream = next(value for value in container.streams if value.type == "video")
        count = sum(1 for _ in container.decode(stream))
        rate = float(stream.average_rate) if stream.average_rate else 0.0
        return {
            "path": str(path),
            "frames": count,
            "width": int(stream.width),
            "height": int(stream.height),
            "fps": rate,
            "bytes": path.stat().st_size,
        }


def _video_metrics(
    gt_views: dict[str, np.ndarray], pred_views: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    metrics = {}
    for camera_key, gt in gt_views.items():
        difference = gt.astype(np.float32) - pred_views[camera_key].astype(np.float32)
        mse = float(np.mean(difference**2))
        metrics[camera_key] = {
            "mae_uint8": float(np.mean(np.abs(difference))),
            "mse_uint8": mse,
            "psnr_db": float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse),
        }
    return metrics


def _load_episode_actions(dataset_root: Path, episode_index: int) -> np.ndarray:
    import pyarrow.parquet as parquet

    path = (
        dataset_root
        / "data"
        / f"chunk-{episode_index // 1000:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = parquet.read_table(path, columns=["action"])
    return np.asarray(table["action"].to_pylist(), dtype=np.float32)


def _load_gt_actions(
    dataset_root: Path, episode_index: int, start_frame: int, count: int
) -> np.ndarray:
    actions = _load_episode_actions(dataset_root, episode_index)
    result = actions[start_frame : start_frame + count]
    if len(result) != count:
        raise RuntimeError(
            f"episode {episode_index} has {len(result)} of {count} requested actions"
        )
    return result


def _feedback_action_state(
    episode_actions: np.ndarray,
    *,
    action_cursor: int,
    feedback_chunk_index: int,
    frame_chunk_size: int,
    action_per_frame: int,
) -> tuple[np.ndarray, int, tuple[int, int]]:
    action_width = int(episode_actions.shape[1])
    if feedback_chunk_index == 0:
        useful_steps = (frame_chunk_size - 1) * action_per_frame
        flat = np.zeros((frame_chunk_size * action_per_frame, action_width), np.float32)
        source = episode_actions[action_cursor : action_cursor + useful_steps]
        if len(source) != useful_steps:
            raise RuntimeError("not enough GT actions for the first feedback chunk")
        flat[action_per_frame:] = source
    else:
        useful_steps = frame_chunk_size * action_per_frame
        source = episode_actions[action_cursor : action_cursor + useful_steps]
        if len(source) != useful_steps:
            raise RuntimeError(
                f"not enough GT actions for feedback chunk {feedback_chunk_index}"
            )
        flat = np.asarray(source, dtype=np.float32)
    source_range = (action_cursor, action_cursor + useful_steps)
    state = flat.reshape(frame_chunk_size, action_per_frame, action_width).transpose(
        2, 0, 1
    )
    return np.ascontiguousarray(state), action_cursor + useful_steps, source_range


def _flatten_useful_predicted_actions(action_chunks: list[np.ndarray]) -> np.ndarray:
    useful = []
    for chunk_index, action in enumerate(action_chunks):
        if action.ndim != 3:
            raise RuntimeError(f"unexpected predicted action shape: {action.shape}")
        selected = action[:, 1:] if chunk_index == 0 else action
        useful.append(selected.transpose(1, 2, 0).reshape(-1, action.shape[0]))
    return np.concatenate(useful, axis=0).astype(np.float32, copy=False)


def _load_gt_latent_sequence(
    dataset_root: Path,
    camera_keys: list[str],
    episode_index: int,
    start_frame: int,
) -> tuple[torch.Tensor, list[int], list[Path]]:
    camera_latents = []
    anchor_frame_ids: list[int] | None = None
    selected_paths = []
    for camera_key in camera_keys:
        directory = (
            dataset_root
            / "latents"
            / f"chunk-{episode_index // 1000:03d}"
            / camera_key
        )
        candidates = sorted(directory.glob(f"episode_{episode_index:06d}_*.pth"))
        selected_payload = None
        selected_path = None
        for path in candidates:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if int(payload["start_frame"]) <= start_frame < int(payload["end_frame"]):
                selected_payload = payload
                selected_path = path
                break
        if selected_payload is None or selected_path is None:
            raise FileNotFoundError(
                f"no GT latent segment covers source frame {start_frame}: {directory}"
            )
        latent_frames = int(selected_payload["latent_num_frames"])
        latent_height = int(selected_payload["latent_height"])
        latent_width = int(selected_payload["latent_width"])
        latent = selected_payload["latent"].reshape(
            latent_frames, latent_height, latent_width, -1
        )
        current_anchors = [
            int(value)
            for value in selected_payload["frame_ids"][::WAN_TEMPORAL_STRIDE][
                :latent_frames
            ]
        ]
        if anchor_frame_ids is None:
            anchor_frame_ids = current_anchors
        elif current_anchors != anchor_frame_ids:
            raise RuntimeError("camera GT latent anchors disagree")
        camera_latents.append(latent)
        selected_paths.append(selected_path)
    assert anchor_frame_ids is not None
    if start_frame not in anchor_frame_ids:
        raise RuntimeError(
            f"start frame {start_frame} is not a causal VAE latent anchor; "
            f"first anchors={anchor_frame_ids[:8]}"
        )
    combined = torch.cat(camera_latents, dim=2).permute(3, 0, 1, 2).unsqueeze(0)
    return combined.contiguous(), anchor_frame_ids, selected_paths


def _decode_prediction_chunks(
    server: Any,
    latent_chunks: list[torch.Tensor],
    *,
    observation_mode: str,
    gt_latents: torch.Tensor | None = None,
    gt_start_latent: int = 0,
) -> np.ndarray:
    if observation_mode == "open_loop":
        with torch.inference_mode():
            return _as_uint8_video(
                server.decode_one_video(torch.cat(latent_chunks, dim=2), "np")
            )
    if gt_latents is None:
        raise RuntimeError("GT latents are required to decode feedback predictions")
    decoded_chunks = []
    frame_chunk_size = int(latent_chunks[0].shape[2])
    for chunk_index, prediction in enumerate(latent_chunks):
        if chunk_index == 0:
            decode_input = prediction
            trim = 0
            expected_frames = 1 + WAN_TEMPORAL_STRIDE * (frame_chunk_size - 1)
        else:
            history_frames = chunk_index * frame_chunk_size
            history_end = gt_start_latent + history_frames
            observed_gt_history = gt_latents[
                :, :, gt_start_latent:history_end
            ].to(
                device=prediction.device, dtype=prediction.dtype
            )
            decode_input = torch.cat((observed_gt_history, prediction), dim=2)
            trim = 1 + WAN_TEMPORAL_STRIDE * (history_frames - 1)
            expected_frames = WAN_TEMPORAL_STRIDE * frame_chunk_size
        with torch.inference_mode():
            decoded = _as_uint8_video(server.decode_one_video(decode_input, "np"))
        decoded = np.ascontiguousarray(decoded[trim:])
        if len(decoded) != expected_frames:
            raise RuntimeError(
                f"decoded feedback chunk {chunk_index} has {len(decoded)} frames, "
                f"expected {expected_frames}"
            )
        decoded_chunks.append(decoded)
    return np.concatenate(decoded_chunks, axis=0)


def _action_metrics(gt: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    count = min(len(gt), len(prediction))
    difference = gt[:count].astype(np.float64) - prediction[:count].astype(np.float64)
    return {
        "aligned_steps": count,
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae_per_channel": np.mean(np.abs(difference), axis=0).tolist(),
    }


def _episode_prompt(episode: dict[str, Any]) -> str:
    tasks = episode.get("tasks") or []
    if tasks:
        return str(tasks[0])
    action_config = episode.get("action_config") or []
    if action_config:
        return str(action_config[0].get("action_text") or "")
    raise RuntimeError(f"episode {episode.get('episode_index')} has no prompt")


def _build_server(
    args: argparse.Namespace,
    profile: dict[str, Any],
    runtime_model: Path,
) -> tuple[Any, Any]:
    repo = args.lingbot_repo.expanduser().resolve()
    wan_va = repo / "wan_va"
    if not (wan_va / "wan_va_server.py").is_file():
        raise FileNotFoundError(f"old LingBot-VA server is missing: {wan_va}")
    if str(wan_va) not in sys.path:
        sys.path.insert(0, str(wan_va))
    install_flash_attention_import_fallback(torch)
    upstream = importlib.import_module("wan_va_server")
    upstream.init_logger()
    upstream.save_async = lambda *unused_args, **unused_kwargs: None

    config = deepcopy(upstream.VA_CONFIGS["demo_i2av"])
    for key, value in profile.items():
        if key != "schema_version":
            config[key] = value
    config.wan22_pretrained_model_name_or_path = str(runtime_model)
    config.save_root = str(args.output_dir / "_upstream")
    config.enable_offload = False
    config.local_rank = 0
    config.rank = 0
    config.world_size = 1
    config.guidance_scale = args.guidance_scale
    config.action_guidance_scale = args.action_guidance_scale
    config.num_inference_steps = args.video_steps
    config.action_num_inference_steps = args.action_steps
    config.video_exec_step = -1
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    server = upstream.VA_Server(config)
    server.video_processor = importlib.import_module(
        "diffusers.video_processor"
    ).VideoProcessor(vae_scale_factor=1)
    return upstream, server


def _run_case(
    args: argparse.Namespace,
    *,
    server: Any,
    profile: dict[str, Any],
    dataset_info: dict[str, Any],
    episode: dict[str, Any],
) -> dict[str, Any]:
    episode_index = int(episode["episode_index"])
    case_dir = args.output_dir / f"episode_{episode_index:06d}"
    receipt_path = case_dir / "receipt.json"
    if receipt_path.is_file() and not args.overwrite:
        print(f"episode {episode_index}: reusing completed receipt", flush=True)
        return _read_json(receipt_path)
    case_dir.mkdir(parents=True, exist_ok=True)

    camera_keys = [str(value) for value in profile["obs_cam_keys"]]
    width = int(profile["width"])
    height = int(profile["height"])
    source_fps = float(dataset_info["fps"])
    latent_fps = source_fps / (
        int(profile["action_per_frame"]) / WAN_TEMPORAL_STRIDE
    )
    source_stride = int(round(source_fps / latent_fps))
    prompt = _episode_prompt(episode)
    seed = args.seed + episode_index

    feedback_enabled = args.observation_mode == "gt_chunk_feedback"
    mode_label = "GT OBS FEEDBACK" if feedback_enabled else "OPEN LOOP"
    print(
        f"episode {episode_index}: loading initial GT observation at source frame "
        f"{args.start_frame}; mode={args.observation_mode}",
        flush=True,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    server.save_root = str(case_dir / "_upstream")
    server._reset(prompt)
    latent_chunks = []
    action_chunks = []
    injected_observation_frame_ids = [args.start_frame]
    injected_action_source_ranges: list[tuple[int, int]] = []
    action_cursor = args.start_frame
    next_observation_frame = args.start_frame + source_stride
    episode_actions = (
        _load_episode_actions(args.dataset_root, episode_index)
        if feedback_enabled and args.feedback_action_source == "gt"
        else None
    )
    observation_reader = (
        _SequentialObservationReader(args.dataset_root, camera_keys, episode_index)
        if feedback_enabled
        else None
    )
    initial_observation = (
        observation_reader.read([args.start_frame])
        if observation_reader is not None
        else _load_initial_observation(
            args.dataset_root, camera_keys, episode_index, args.start_frame
        )
    )
    started = time.monotonic()
    frame_chunk_size = int(profile["frame_chunk_size"])
    action_per_frame = int(profile["action_per_frame"])
    try:
        for chunk_index in range(args.num_chunks):
            frame_start_id = (
                int(server.frame_st_id)
                if feedback_enabled
                else chunk_index * frame_chunk_size
            )
            print(
                f"episode {episode_index}: predict chunk "
                f"{chunk_index + 1}/{args.num_chunks} at latent {frame_start_id}; "
                f"mode={args.observation_mode}",
                flush=True,
            )
            actions, latents = server._infer(
                initial_observation,
                frame_st_id=frame_start_id,
            )
            latent_chunks.append(latents)
            action_chunks.append(actions)

            if feedback_enabled and chunk_index + 1 < args.num_chunks:
                observation_count = (
                    (frame_chunk_size - 1) * WAN_TEMPORAL_STRIDE
                    if chunk_index == 0
                    else frame_chunk_size * WAN_TEMPORAL_STRIDE
                )
                feedback_frame_ids = [
                    next_observation_frame + index * source_stride
                    for index in range(observation_count)
                ]
                if feedback_frame_ids[-1] >= int(episode["length"]):
                    raise RuntimeError(
                        f"episode {episode_index} lacks GT observation frame "
                        f"{feedback_frame_ids[-1]} for cache feedback"
                    )
                assert observation_reader is not None
                cache_input = observation_reader.read(feedback_frame_ids)
                if args.feedback_action_source == "predicted":
                    state = np.ascontiguousarray(actions, dtype=np.float32)
                    action_range = None
                else:
                    assert episode_actions is not None
                    state, action_cursor, action_range = _feedback_action_state(
                        episode_actions,
                        action_cursor=action_cursor,
                        feedback_chunk_index=chunk_index,
                        frame_chunk_size=frame_chunk_size,
                        action_per_frame=action_per_frame,
                    )
                cache_input["state"] = state
                server._compute_kv_cache(cache_input)
                expected_frame_id = (chunk_index + 1) * frame_chunk_size
                if int(server.frame_st_id) != expected_frame_id:
                    raise RuntimeError(
                        "official observation cache advanced to the wrong latent ID: "
                        f"got {server.frame_st_id}, expected {expected_frame_id}"
                    )
                injected_observation_frame_ids.extend(feedback_frame_ids)
                if action_range is not None:
                    injected_action_source_ranges.append(action_range)
                next_observation_frame = feedback_frame_ids[-1] + source_stride
    finally:
        if observation_reader is not None:
            observation_reader.close()
    inference_seconds = time.monotonic() - started
    predicted_actions = _flatten_useful_predicted_actions(action_chunks)

    print(f"episode {episode_index}: decoding generated latent video", flush=True)
    gt_latent_paths: list[Path] = []
    gt_latents = None
    gt_start_latent = 0
    if feedback_enabled:
        gt_latents, gt_latent_anchors, gt_latent_paths = _load_gt_latent_sequence(
            args.dataset_root, camera_keys, episode_index, args.start_frame
        )
        gt_start_latent = gt_latent_anchors.index(args.start_frame)
        required_latent_end = gt_start_latent + args.num_chunks * frame_chunk_size
        if required_latent_end > gt_latents.shape[2]:
            raise RuntimeError(
                f"episode {episode_index} has {gt_latents.shape[2]} GT latents, "
                f"but prediction decode needs {required_latent_end}"
            )
    prediction = _decode_prediction_chunks(
        server,
        latent_chunks,
        observation_mode=args.observation_mode,
        gt_latents=gt_latents,
        gt_start_latent=gt_start_latent,
    )
    del latent_chunks, gt_latents
    torch.cuda.empty_cache()

    predicted_views = _split_prediction(prediction, camera_keys, width, height)
    frame_ids = [
        args.start_frame + index * source_stride for index in range(len(prediction))
    ]
    if frame_ids[-1] >= int(episode["length"]):
        raise RuntimeError(
            f"episode {episode_index} ends at {episode['length']}, but GT frame "
            f"{frame_ids[-1]} is required"
        )
    print(f"episode {episode_index}: decoding {len(frame_ids)} aligned GT frames", flush=True)
    gt_views = _load_gt_views(
        args.dataset_root,
        camera_keys,
        episode_index,
        frame_ids,
        width=width,
        height=height,
    )

    videos: dict[str, dict[str, Any]] = {}
    all_views_path = case_dir / "gt_vs_imagined_all_views.mp4"
    videos["all_views"] = _write_video(
        all_views_path,
        _render_all_views_pair(
            gt_views,
            predicted_views,
            camera_keys,
            episode_index=episode_index,
            fps=latent_fps,
            mode_label=mode_label,
        ),
        fps=latent_fps,
        crf=args.crf,
    )
    for camera_key in camera_keys:
        short_name = camera_key.rsplit(".", 1)[-1]
        path = case_dir / f"gt_vs_imagined_{short_name}.mp4"
        videos[camera_key] = _write_video(
            path,
            _render_single_view_pair(
                gt_views[camera_key],
                predicted_views[camera_key],
                camera_key=camera_key,
                episode_index=episode_index,
                fps=latent_fps,
                mode_label=mode_label,
            ),
            fps=latent_fps,
            crf=args.crf,
        )

    gt_actions = _load_gt_actions(
        args.dataset_root, episode_index, args.start_frame, len(predicted_actions)
    )
    actions_path = case_dir / "gt_and_predicted_actions.npz"
    np.savez_compressed(
        actions_path,
        ground_truth=gt_actions,
        predicted=predicted_actions.astype(np.float32),
        source_action_indices=np.asarray(profile["source_action_indices"], dtype=np.int64),
        model_action_channels=np.asarray(
            profile["used_action_channel_ids"], dtype=np.int64
        ),
    )

    receipt = {
        "schema_version": RECEIPT_VERSION,
        "episode_index": episode_index,
        "episode_length_source_frames": int(episode["length"]),
        "prompt": prompt,
        "seed": seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "protocol": {
            "mode": args.observation_mode,
            "gt_conditioning_source_frames": injected_observation_frame_ids,
            "gt_feedback_after_initial_observation": feedback_enabled,
            "feedback_action_history_source": (
                args.feedback_action_source if feedback_enabled else None
            ),
            "gt_actions_injected_with_observation_history": (
                feedback_enabled and args.feedback_action_source == "gt"
            ),
            "gt_action_source_ranges": [
                [start, end] for start, end in injected_action_source_ranges
            ],
            "future_gt_visible_to_current_prediction": False,
            "feedback_interval_chunks": 1 if feedback_enabled else None,
            "num_chunks": args.num_chunks,
            "latent_frames": args.num_chunks * frame_chunk_size,
            "decoded_frames": len(prediction),
            "source_frame_stride": source_stride,
            "source_frame_ids": frame_ids,
            "source_fps": source_fps,
            "comparison_fps": latent_fps,
            "duration_seconds": len(prediction) / latent_fps,
            "gt_latent_decode_context": [
                str(path) for path in gt_latent_paths
            ],
        },
        "camera_layout": [
            {
                "model_order": index,
                "camera_key": key,
                "physical_role": PHYSICAL_CAMERA_LABELS.get(key, key),
            }
            for index, key in enumerate(camera_keys)
        ],
        "sampling": {
            "guidance_scale": args.guidance_scale,
            "action_guidance_scale": args.action_guidance_scale,
            "video_steps": args.video_steps,
            "action_steps": args.action_steps,
            "inference_seconds": inference_seconds,
        },
        "video_metrics": _video_metrics(gt_views, predicted_views),
        "action_metrics": _action_metrics(gt_actions, predicted_actions),
        "actions": {
            "path": str(actions_path),
            "steps": len(predicted_actions),
            "channels": int(predicted_actions.shape[1]),
            "bytes": actions_path.stat().st_size,
        },
        "videos": videos,
    }
    for key, value in list(videos.items()):
        probed = _probe_video(Path(value["path"]))
        if probed["frames"] != len(prediction):
            raise RuntimeError(
                f"encoded video {key} has {probed['frames']} frames, "
                f"expected {len(prediction)}"
            )
        videos[key] = probed
    _write_json(receipt_path, receipt)
    print(
        f"episode {episode_index}: complete, {len(prediction)} frames, "
        f"{inference_seconds:.1f}s inference",
        flush=True,
    )
    return receipt


def _run_worker(args: argparse.Namespace, episode_ids: list[int]) -> list[dict[str, Any]]:
    dataset_root = args.dataset_root.expanduser().resolve()
    args.dataset_root = dataset_root
    args.output_dir = args.output_dir.expanduser().resolve()
    args.lingbot_repo = args.lingbot_repo.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    profile = _read_json(dataset_root / "meta" / "lingbot_va_model_profile.json")
    dataset_info = _read_json(dataset_root / "meta" / "info.json")
    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    by_id = {int(item["episode_index"]): item for item in episodes}
    runtime_model = _ensure_runtime_model(
        args.output_dir, args.base_model_root.expanduser().resolve(), args.checkpoint
    )
    _, server = _build_server(args, profile, runtime_model)
    receipts = []
    try:
        for episode_id in episode_ids:
            receipts.append(
                _run_case(
                    args,
                    server=server,
                    profile=profile,
                    dataset_info=dataset_info,
                    episode=by_id[episode_id],
                )
            )
            server.transformer.clear_cache(server.cache_name)
            server.streaming_vae.clear_cache()
    finally:
        del server
        gc.collect()
        torch.cuda.empty_cache()
    return receipts


def _worker_command(
    args: argparse.Namespace, episode_ids: list[int], worker_id: int
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--lingbot-repo",
        str(args.lingbot_repo),
        "--dataset-root",
        str(args.dataset_root),
        "--base-model-root",
        str(args.base_model_root),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(args.output_dir),
        "--num-cases",
        str(args.num_cases),
        "--num-chunks",
        str(args.num_chunks),
        "--start-frame",
        str(args.start_frame),
        "--observation-mode",
        args.observation_mode,
        "--feedback-action-source",
        args.feedback_action_source,
        "--seed",
        str(args.seed),
        "--guidance-scale",
        str(args.guidance_scale),
        "--action-guidance-scale",
        str(args.action_guidance_scale),
        "--video-steps",
        str(args.video_steps),
        "--action-steps",
        str(args.action_steps),
        "--crf",
        str(args.crf),
        "--worker-episodes",
        ",".join(str(value) for value in episode_ids),
        "--worker-id",
        str(worker_id),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def _concatenate_videos(
    inputs: list[Path], output: Path, *, fps: float, crf: int
) -> dict[str, Any]:
    import av

    def frames() -> Iterable[np.ndarray]:
        for input_path in inputs:
            with av.open(str(input_path)) as container:
                stream = next(value for value in container.streams if value.type == "video")
                for frame in container.decode(stream):
                    yield frame.to_ndarray(format="rgb24")

    return _write_video(output, frames(), fps=fps, crf=crf)


def _aggregate(
    args: argparse.Namespace,
    selected: list[int],
    *,
    profile: dict[str, Any],
    dataset_info: dict[str, Any],
) -> dict[str, Any]:
    receipts = [
        _read_json(args.output_dir / f"episode_{episode_id:06d}" / "receipt.json")
        for episode_id in selected
    ]
    fps = float(receipts[0]["protocol"]["comparison_fps"])
    reel_path = (
        args.output_dir
        / f"lingbot_va_{len(receipts)}case_gt_vs_imagined_reel.mp4"
    )
    reel = _concatenate_videos(
        [Path(item["videos"]["all_views"]["path"]) for item in receipts],
        reel_path,
        fps=fps,
        crf=args.crf,
    )
    reel = _probe_video(Path(reel["path"]))
    aggregate = {
        "schema_version": RECEIPT_VERSION,
        "status": "succeeded",
        "case_count": len(receipts),
        "episode_indices": selected,
        "dataset_root": str(args.dataset_root.resolve()),
        "base_model_root": str(args.base_model_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_transformer": str(_checkpoint_transformer(args.checkpoint)),
        "runtime_model_uses_symlinks": True,
        "protocol": {
            "mode": args.observation_mode,
            "gt_feedback_after_initial_observation": (
                args.observation_mode == "gt_chunk_feedback"
            ),
            "feedback_action_history_source": (
                args.feedback_action_source
                if args.observation_mode == "gt_chunk_feedback"
                else None
            ),
            "num_chunks_per_case": args.num_chunks,
            "expected_frames_per_case": _video_frame_count(
                args.num_chunks, int(profile["frame_chunk_size"])
            ),
            "comparison_fps": fps,
            "source_fps": float(dataset_info["fps"]),
        },
        "camera_layout": receipts[0]["camera_layout"],
        "reel": reel,
        "cases": receipts,
    }
    _write_json(args.output_dir / "inference_receipt.json", aggregate)
    lines = [
        "# LingBot-VA GT vs imagined rollouts",
        "",
        f"Observation mode: `{args.observation_mode}`.",
        "",
        (
            "Before each next-chunk prediction, GT observation and action history are "
            "written into the official KV cache. Future target frames are never visible "
            "to the current prediction."
            if args.observation_mode == "gt_chunk_feedback"
            else "Only the initial GT observation is used; later chunks are open loop."
        ),
        "",
        f"- Checkpoint: `{args.checkpoint.resolve()}`",
        f"- Cases: `{', '.join(str(value) for value in selected)}`",
        f"- Reel: `{reel_path.name}`",
        "",
        "| Episode | Frames | Duration | All-view pair |",
        "|---:|---:|---:|---|",
    ]
    for receipt in receipts:
        episode_index = int(receipt["episode_index"])
        protocol = receipt["protocol"]
        relative = Path(receipt["videos"]["all_views"]["path"]).relative_to(
            args.output_dir
        )
        lines.append(
            f"| {episode_index} | {protocol['decoded_frames']} | "
            f"{protocol['duration_seconds']:.2f}s | `{relative}` |"
        )
    (args.output_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def _run_parallel(
    args: argparse.Namespace, selected: list[int], gpu_ids: list[int]
) -> None:
    worker_count = min(len(gpu_ids), len(selected))
    assignments = [selected[index::worker_count] for index in range(worker_count)]
    log_dir = args.output_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    for worker_id, (gpu_id, episode_ids) in enumerate(
        zip(gpu_ids[:worker_count], assignments)
    ):
        log_path = log_dir / f"worker_{worker_id:02d}_gpu_{gpu_id}.log"
        handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        environment["TOKENIZERS_PARALLELISM"] = "false"
        process = subprocess.Popen(
            _worker_command(args, episode_ids, worker_id),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=str(ROOT),
        )
        processes.append((process, worker_id, gpu_id, episode_ids, log_path))
        handles.append(handle)
        print(
            f"worker {worker_id}: GPU {gpu_id}, episodes {episode_ids}, PID {process.pid}",
            flush=True,
        )

    last_report = 0.0
    while any(process.poll() is None for process, *_ in processes):
        now = time.monotonic()
        if now - last_report >= 30:
            states = []
            for process, worker_id, gpu_id, episode_ids, _ in processes:
                state = "running" if process.poll() is None else f"exit={process.returncode}"
                states.append(f"w{worker_id}/gpu{gpu_id}/{episode_ids}:{state}")
            print("parallel status: " + "; ".join(states), flush=True)
            last_report = now
        time.sleep(2)
    for handle in handles:
        handle.close()
    failed = [value for value in processes if value[0].returncode != 0]
    if failed:
        messages = []
        for process, worker_id, gpu_id, episode_ids, log_path in failed:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            messages.append(
                f"worker {worker_id} on GPU {gpu_id} for {episode_ids} failed "
                f"with {process.returncode}:\n" + "\n".join(tail)
            )
        raise RuntimeError("\n\n".join(messages))


def main() -> None:
    args = _parse_args()
    if args.num_cases < 8:
        raise SystemExit("num-cases must be at least 8 for this evaluation")
    if args.num_chunks <= 0 or args.start_frame < 0:
        raise SystemExit("num-chunks must be positive and start-frame must be nonnegative")
    if args.video_steps <= 0 or args.action_steps <= 0:
        raise SystemExit("inference step counts must be positive")
    args.lingbot_repo = args.lingbot_repo.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.base_model_root = args.base_model_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.worker_episodes:
        episode_ids = _parse_int_list(args.worker_episodes)
        _run_worker(args, episode_ids)
        return

    profile = _read_json(args.dataset_root / "meta" / "lingbot_va_model_profile.json")
    dataset_info = _read_json(args.dataset_root / "meta" / "info.json")
    episodes = _read_jsonl(args.dataset_root / "meta" / "episodes.jsonl")
    source_fps = float(dataset_info["fps"])
    latent_fps = source_fps / (
        int(profile["action_per_frame"]) / WAN_TEMPORAL_STRIDE
    )
    source_stride = int(round(source_fps / latent_fps))
    expected_video_frames = _video_frame_count(
        args.num_chunks, int(profile["frame_chunk_size"])
    )
    required_source_frames = 1 + source_stride * (expected_video_frames - 1)
    selected = _select_episodes(
        episodes,
        requested=_parse_int_list(args.episodes),
        num_cases=args.num_cases,
        required_source_frames=required_source_frames,
        start_frame=args.start_frame,
    )
    _ensure_runtime_model(args.output_dir, args.base_model_root, args.checkpoint)
    print(
        f"selected episodes {selected}; {args.num_chunks} chunks -> "
        f"{expected_video_frames} frames/case at {latent_fps:g} FPS",
        flush=True,
    )

    gpu_ids = _parse_int_list(args.gpus)
    if gpu_ids:
        _run_parallel(args, selected, gpu_ids)
    else:
        _run_worker(args, selected)
    aggregate = _aggregate(
        args, selected, profile=profile, dataset_info=dataset_info
    )
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


if __name__ == "__main__":
    main()
