from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import av
import numpy as np
from PIL import Image, ImageDraw


LABEL_HEIGHT = 32
COMPOSITE_SIZE = (320, 384)
VIEW_OUTPUT_SIZE = (320, 256)


@dataclass(frozen=True)
class CameraPanel:
    role: str
    source_key: str
    content: str
    box: tuple[int, int, int, int]


CAMERA_PANELS = (
    CameraPanel(
        role="global_primary",
        source_key="observation.images.left_eye",
        content="global/head view",
        box=(0, 0, 320, 256),
    ),
    CameraPanel(
        role="left_wrist",
        source_key="observation.images.right_eye",
        content="left wrist view",
        box=(0, 256, 160, 384),
    ),
    CameraPanel(
        role="right_wrist",
        source_key="observation.images.right_wrist",
        content="right wrist view",
        box=(160, 256, 320, 384),
    ),
)


def decode_labeled_composite(path: Path) -> tuple[list[np.ndarray], Fraction]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = stream.average_rate or Fraction(1, 1)
        for frame in container.decode(stream):
            array = frame.to_ndarray(format="rgb24")
            if array.shape != (COMPOSITE_SIZE[1] + LABEL_HEIGHT, COMPOSITE_SIZE[0], 3):
                raise ValueError(
                    f"Expected labeled 320x416 composite in {path}, got {array.shape}."
                )
            frames.append(np.ascontiguousarray(array[LABEL_HEIGHT:]))
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return frames, Fraction(fps)


def crop_camera_panel(frame: np.ndarray, panel: CameraPanel) -> np.ndarray:
    if frame.shape != (COMPOSITE_SIZE[1], COMPOSITE_SIZE[0], 3):
        raise ValueError(f"Expected 384x320 RGB composite, got {frame.shape}.")
    left, top, right, bottom = panel.box
    return np.ascontiguousarray(frame[top:bottom, left:right])


def panel_psnr(predicted: Iterable[np.ndarray], target: Iterable[np.ndarray]) -> float:
    predicted_frames = list(predicted)
    target_frames = list(target)
    if len(predicted_frames) != len(target_frames) or not predicted_frames:
        raise ValueError("Prediction and target must contain the same non-zero frame count.")
    squared_error = 0.0
    value_count = 0
    for pred, gt in zip(predicted_frames, target_frames):
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {gt.shape}.")
        difference = pred.astype(np.float64) - gt.astype(np.float64)
        squared_error += float(np.square(difference).sum())
        value_count += difference.size
    mse = squared_error / value_count / (255.0**2)
    return float("inf") if mse == 0.0 else float(-10.0 * np.log10(mse))


def camera_pair_frame(
    predicted: np.ndarray,
    target: np.ndarray,
    panel: CameraPanel,
) -> np.ndarray:
    output_width, output_height = VIEW_OUTPUT_SIZE
    pred_image = Image.fromarray(predicted, mode="RGB").resize(
        (output_width, output_height), Image.Resampling.BILINEAR
    )
    gt_image = Image.fromarray(target, mode="RGB").resize(
        (output_width, output_height), Image.Resampling.BILINEAR
    )
    result = Image.new(
        "RGB", (output_width * 2, output_height + LABEL_HEIGHT), color=(16, 16, 16)
    )
    result.paste(pred_image, (0, LABEL_HEIGHT))
    result.paste(gt_image, (output_width, LABEL_HEIGHT))
    draw = ImageDraw.Draw(result)
    draw.text((8, 9), f"IMAGINATION | {panel.role}", fill=(255, 255, 255))
    draw.text(
        (output_width + 8, 9),
        f"GT EXECUTION | {panel.role}",
        fill=(255, 255, 255),
    )
    return np.asarray(result, dtype=np.uint8)


def write_h264(path: Path, frames: Iterable[np.ndarray], fps: Fraction) -> None:
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("Cannot encode an empty video.")
    height, width, channels = frame_list[0].shape
    if channels != 3 or width % 2 or height % 2:
        raise ValueError(f"H.264 frames must be even-sized RGB, got {(height, width, channels)}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for array in frame_list:
            if array.shape != (height, width, channels):
                raise ValueError("All output frames must have identical dimensions.")
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
