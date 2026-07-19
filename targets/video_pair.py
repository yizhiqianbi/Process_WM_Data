"""Shared I/O and metrics for GT-versus-generated video evaluations."""

from __future__ import annotations

from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(json.loads(line))
    return values


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"duplicate integer IDs: {value}")
    return parsed


def lerobot_video_path(
    dataset_root: Path, camera_key: str, episode_index: int
) -> Path:
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


def decode_frames(
    path: Path,
    frame_ids: list[int],
    *,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    import av
    import cv2

    if not frame_ids:
        raise ValueError("frame_ids cannot be empty")
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
            if frame_index >= max(frame_ids) and len(decoded) == len(wanted):
                break
    missing = [value for value in frame_ids if value not in decoded]
    if missing:
        raise RuntimeError(f"{path} is missing requested frames: {missing[:8]}")
    return np.stack([decoded[value] for value in frame_ids])


class SequentialFrameReader:
    """Decode monotonically increasing frame IDs without reopening a video."""

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

    def __enter__(self) -> "SequentialFrameReader":
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()


def as_uint8_video(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[0] == 3:
        array = array.transpose(1, 2, 3, 0)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise RuntimeError(f"unexpected decoded video shape: {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        minimum = float(np.nanmin(array))
        maximum = float(np.nanmax(array))
        if minimum >= -1.1 and maximum <= 1.1:
            array = (array + 1.0) * 127.5 if minimum < -0.05 else array * 255.0
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def put_text(
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


def write_video(
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
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(image), format="rgb24"
            )
            for packet in stream.encode(frame):
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


def probe_video(path: Path) -> dict[str, Any]:
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


def concatenate_videos(
    inputs: list[Path], output: Path, *, fps: float, crf: int
) -> dict[str, Any]:
    import av

    def frames() -> Iterable[np.ndarray]:
        for input_path in inputs:
            with av.open(str(input_path)) as container:
                stream = next(
                    value for value in container.streams if value.type == "video"
                )
                for frame in container.decode(stream):
                    yield frame.to_ndarray(format="rgb24")

    write_video(output, frames(), fps=fps, crf=crf)
    return probe_video(output)


def video_metrics(
    ground_truth: dict[str, np.ndarray], prediction: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    metrics = {}
    for camera_key, gt in ground_truth.items():
        difference = gt.astype(np.float32) - prediction[camera_key].astype(np.float32)
        mse = float(np.mean(difference**2))
        metrics[camera_key] = {
            "mae_uint8": float(np.mean(np.abs(difference))),
            "mse_uint8": mse,
            "psnr_db": float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse),
        }
    return metrics


def action_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    count = min(len(ground_truth), len(prediction))
    difference = (
        ground_truth[:count].astype(np.float64) - prediction[:count].astype(np.float64)
    )
    return {
        "aligned_steps": count,
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae_per_channel": np.mean(np.abs(difference), axis=0).tolist(),
    }
