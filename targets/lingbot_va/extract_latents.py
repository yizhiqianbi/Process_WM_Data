from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any
import uuid

import numpy as np

from targets.common import (
    TargetPreparationError,
    file_sha256,
    iter_jsonl,
    read_json,
    write_json,
)
from targets.lingbot_va.runtime import install_flash_attention_import_fallback


LATENT_RECEIPT_VERSION = "lingbot-va-latent-extraction-v1"


def _segment_key(job: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(job["episode_index"]),
        int(job["start_frame"]),
        int(job["end_frame"]),
    )


def select_latent_segments(
    jobs: list[dict[str, Any]],
    *,
    episode_indices: set[int] | None = None,
    max_segments: int | None = None,
) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        key = _segment_key(job)
        if episode_indices is not None and key[0] not in episode_indices:
            continue
        grouped[key].append(job)
    selected: list[list[dict[str, Any]]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda value: str(value["camera_key"]))
        cameras = [str(value["camera_key"]) for value in group]
        if len(cameras) != len(set(cameras)):
            raise TargetPreparationError(f"duplicate camera latent job in segment {key}")
        frame_ids = [int(value) for value in group[0].get("frame_ids") or []]
        if not frame_ids:
            raise TargetPreparationError(f"segment {key} has no latent frame IDs")
        for job in group[1:]:
            if [int(value) for value in job.get("frame_ids") or []] != frame_ids:
                raise TargetPreparationError(
                    f"camera latent jobs disagree on frame IDs in segment {key}"
                )
            if str(job.get("text") or "") != str(group[0].get("text") or ""):
                raise TargetPreparationError(
                    f"camera latent jobs disagree on text in segment {key}"
                )
        selected.append(group)
        if max_segments is not None and len(selected) >= max_segments:
            break
    if not selected:
        raise TargetPreparationError("no latent segments matched the requested filters")
    return selected


def _safe_output_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise TargetPreparationError(f"latent output escapes target root: {relative}")
    return path


def _resolve_video(root: Path, job: dict[str, Any]) -> Path:
    relative = str(job.get("source_video_relative") or "")
    if relative:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    candidate = Path(str(job.get("source_video") or "")).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    raise TargetPreparationError(
        f"source video is missing for episode {job.get('episode_index')}, "
        f"camera {job.get('camera_key')}: {candidate}"
    )


def _decode_selected_frames(video_path: Path, frame_ids: list[int]) -> np.ndarray:
    import av

    wanted = set(frame_ids)
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise TargetPreparationError(f"video has no video stream: {video_path}")
        for frame_index, frame in enumerate(container.decode(streams[0])):
            if frame_index in wanted:
                decoded[frame_index] = frame.to_ndarray(format="rgb24")
            if frame_index >= frame_ids[-1] and len(decoded) == len(wanted):
                break
    missing = [value for value in frame_ids if value not in decoded]
    if missing:
        raise TargetPreparationError(
            f"video {video_path} is missing {len(missing)} requested frames; first={missing[:5]}"
        )
    return np.stack([decoded[value] for value in frame_ids])


def _encode_text(tokenizer: Any, text_encoder: Any, text: str, torch: Any) -> Any:
    encoded = tokenizer(
        [text],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    device = next(text_encoder.parameters()).device
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    with torch.inference_mode():
        embedding = text_encoder(input_ids, attention_mask).last_hidden_state
    sequence_length = int(attention_mask[0].sum().item())
    embedding = embedding[0].to(dtype=torch.bfloat16, device="cpu").clone()
    embedding[sequence_length:] = 0
    return embedding.contiguous()


def _encode_video(
    frames: np.ndarray,
    *,
    height: int,
    width: int,
    vae: Any,
    streaming_vae: Any,
    torch: Any,
) -> Any:
    import torch.nn.functional as functional

    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    tensor = functional.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    tensor = (tensor / 255.0 * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    streaming_vae.clear_cache()
    encoded_chunks = []
    with torch.inference_mode():
        # Wan's causal encoder emits the first latent from one frame, then one
        # latent for each following four-frame chunk while reusing its conv cache.
        chunk_ranges = [(0, 1), *[(start, start + 4) for start in range(1, tensor.shape[2], 4)]]
        for start, end in chunk_ranges:
            chunk = tensor[:, :, start:end].to(device=device, dtype=dtype)
            encoded = streaming_vae.encode_chunk(chunk)
            if encoded.shape[2]:
                encoded_chunks.append(encoded)
    if not encoded_chunks:
        raise TargetPreparationError("VAE produced zero latent frames")
    encoded = torch.cat(encoded_chunks, dim=2)
    mean, _ = torch.chunk(encoded, 2, dim=1)
    latents_mean = torch.as_tensor(vae.config.latents_mean, device=device).view(
        1, -1, 1, 1, 1
    )
    inverse_std = (1.0 / torch.as_tensor(vae.config.latents_std, device=device)).view(
        1, -1, 1, 1, 1
    )
    normalized = ((mean.float() - latents_mean) * inverse_std).to(mean)
    expected_frames = (len(frames) - 1) // 4 + 1
    if normalized.shape[2] != expected_frames:
        raise TargetPreparationError(
            f"VAE produced {normalized.shape[2]} latent frames, expected {expected_frames} "
            f"from {len(frames)} sampled frames"
        )
    return normalized[0].permute(1, 2, 3, 0).contiguous().to("cpu")


def _atomic_torch_save(value: Any, path: Path, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_lingbot_va_latents(
    root: Path,
    *,
    model_root: Path,
    lingbot_repo: Path,
    device: str = "cuda:0",
    episode_indices: set[int] | None = None,
    max_segments: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    model_root = model_root.expanduser().resolve()
    lingbot_repo = lingbot_repo.expanduser().resolve()
    jobs_path = root / "meta" / "lingbot_va_latent_jobs.jsonl"
    target_receipt_path = root / "meta" / "lingbot_va_target_receipt.json"
    if not jobs_path.is_file() or not target_receipt_path.is_file():
        raise TargetPreparationError("LingBot-VA target metadata is missing")
    for subdirectory in ("vae", "text_encoder", "tokenizer"):
        if not (model_root / subdirectory).is_dir():
            raise TargetPreparationError(
                f"LingBot-VA model is missing {subdirectory}/: {model_root}"
            )
    module_root = lingbot_repo / "wan_va"
    if not (module_root / "modules" / "utils.py").is_file():
        raise TargetPreparationError(f"invalid old LingBot-VA checkout: {lingbot_repo}")
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

    import torch

    install_flash_attention_import_fallback(torch)
    from modules.utils import (
        WanVAEStreamingWrapper,
        load_text_encoder,
        load_tokenizer,
        load_vae,
    )

    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise TargetPreparationError("CUDA is required for LingBot-VA latent extraction")
    jobs = list(iter_jsonl(jobs_path))
    segments = select_latent_segments(
        jobs,
        episode_indices=episode_indices,
        max_segments=max_segments,
    )
    vae = load_vae(model_root / "vae", torch.bfloat16, device)
    vae.eval().requires_grad_(False)
    streaming_vae = WanVAEStreamingWrapper(vae)
    text_encoder = load_text_encoder(
        model_root / "text_encoder", torch.bfloat16, device
    )
    text_encoder.eval().requires_grad_(False)
    tokenizer = load_tokenizer(model_root / "tokenizer")

    text_cache: dict[str, Any] = {}
    empty_embedding = _encode_text(tokenizer, text_encoder, "", torch)
    _atomic_torch_save(empty_embedding, root / "empty_emb.pt", torch)
    written = 0
    skipped = 0
    completed_segments = 0
    for group in segments:
        text = str(group[0].get("text") or "")
        if text not in text_cache:
            text_cache[text] = _encode_text(tokenizer, text_encoder, text, torch)
        for job in group:
            output = _safe_output_path(root, str(job["output"]))
            if output.is_file() and not overwrite:
                skipped += 1
                continue
            video_path = _resolve_video(root, job)
            frame_ids = [int(value) for value in job["frame_ids"]]
            frames = _decode_selected_frames(video_path, frame_ids)
            latent = _encode_video(
                frames,
                height=int(job["target_height"]),
                width=int(job["target_width"]),
                vae=vae,
                streaming_vae=streaming_vae,
                torch=torch,
            )
            payload = {
                "latent": latent.reshape(-1, latent.shape[-1]).to(torch.bfloat16),
                "latent_num_frames": int(latent.shape[0]),
                "latent_height": int(latent.shape[1]),
                "latent_width": int(latent.shape[2]),
                "video_num_frames": len(frame_ids),
                "video_height": int(frames.shape[1]),
                "video_width": int(frames.shape[2]),
                "text_emb": text_cache[text],
                "text": text,
                "frame_ids": frame_ids,
                "start_frame": int(job["start_frame"]),
                "end_frame": int(job["end_frame"]),
                "fps": float(job["target_fps"]),
                "ori_fps": float(job["source_fps"]),
            }
            _atomic_torch_save(payload, output, torch)
            written += 1
        if all(
            _safe_output_path(root, str(job["output"])).is_file() for job in group
        ):
            completed_segments += 1

    receipt = {
        "schema_version": LATENT_RECEIPT_VERSION,
        "target_root": str(root),
        "target_receipt_sha256": file_sha256(target_receipt_path),
        "model_root": str(model_root),
        "model_config_sha256": file_sha256(model_root / "vae" / "config.json"),
        "selected_segment_count": len(segments),
        "completed_segment_count": completed_segments,
        "written_latent_count": written,
        "skipped_latent_count": skipped,
        "empty_embedding": "empty_emb.pt",
    }
    write_json(root / "meta" / "lingbot_va_latent_receipt.json", receipt)
    return receipt


def _episode_indices(value: str | None) -> set[int] | None:
    if value is None:
        return None
    result = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not result:
        raise TargetPreparationError("--episode-indices cannot be empty")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract Wan2.2 VAE/T5 latents for old Robbyant/LingBot-VA"
    )
    parser.add_argument("--root", "--target-root", dest="root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--lingbot-repo", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode-indices")
    parser.add_argument("--episode", type=int, action="append")
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.episode_indices is not None and args.episode:
        parser.error("--episode-indices and --episode are mutually exclusive")
    episode_indices = (
        set(args.episode)
        if args.episode
        else _episode_indices(args.episode_indices)
    )
    result = extract_lingbot_va_latents(
        args.root,
        model_root=args.model_root,
        lingbot_repo=args.lingbot_repo,
        device=args.device,
        episode_indices=episode_indices,
        max_segments=args.max_segments,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
