from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable


_EVAL_NAME = re.compile(r"step_(\d+)_rank_(\d+)\.json$")


class OverfitReportError(ValueError):
    pass


def load_eval_records(run_dir: Path, *, rank: int = 0) -> list[dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    records: list[dict[str, Any]] = []
    for path in sorted((run_dir / "eval").glob("step_*_rank_*.json")):
        match = _EVAL_NAME.match(path.name)
        if match is None or int(match.group(2)) != rank:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["step"] = int(match.group(1))
        payload["metrics_path"] = str(path)
        records.append(payload)
    records.sort(key=lambda record: int(record["step"]))
    return records


def _sample_identity(record: dict[str, Any]) -> tuple[int, int]:
    return int(record.get("sample_index", -1)), int(record.get("window_start", -1))


def _candidate_checks(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_psnr_gain: float,
    min_ssim_gain: float,
    max_action_l1_ratio: float,
) -> dict[str, bool]:
    checks = {
        "same_frozen_sample": _sample_identity(candidate) == _sample_identity(baseline),
        "memory_fully_valid": float(candidate.get("memory_valid_ratio", 0.0)) >= 0.999,
        "psnr_gain": float(candidate["psnr_rg"]) - float(baseline["psnr_rg"])
        >= min_psnr_gain,
        "ssim_gain": float(candidate["ssim_rg"]) - float(baseline["ssim_rg"])
        >= min_ssim_gain,
    }
    baseline_action = baseline.get("action_l1")
    candidate_action = candidate.get("action_l1")
    if baseline_action is not None and candidate_action is not None:
        denominator = max(float(baseline_action), 1e-12)
        checks["action_l1_ratio"] = (
            float(candidate_action) / denominator <= max_action_l1_ratio
        )
    return checks


def summarize_overfit(
    records: Iterable[dict[str, Any]],
    *,
    min_psnr_gain: float = 3.0,
    min_ssim_gain: float = 0.05,
    max_action_l1_ratio: float = 0.5,
) -> dict[str, Any]:
    ordered = sorted((dict(record) for record in records), key=lambda record: int(record["step"]))
    if not ordered:
        raise OverfitReportError("No FastWAM evaluation records were found.")
    baseline = ordered[0]
    if int(baseline["step"]) != 0:
        raise OverfitReportError(
            f"Expected an eval-at-start step 0 baseline, found step {baseline['step']}."
        )

    candidates = []
    for record in ordered[1:]:
        checks = _candidate_checks(
            baseline,
            record,
            min_psnr_gain=min_psnr_gain,
            min_ssim_gain=min_ssim_gain,
            max_action_l1_ratio=max_action_l1_ratio,
        )
        psnr_gain = float(record["psnr_rg"]) - float(baseline["psnr_rg"])
        ssim_gain = float(record["ssim_rg"]) - float(baseline["ssim_rg"])
        action_ratio = None
        if baseline.get("action_l1") is not None and record.get("action_l1") is not None:
            action_ratio = float(record["action_l1"]) / max(
                float(baseline["action_l1"]), 1e-12
            )
        candidates.append(
            {
                "record": record,
                "checks": checks,
                "passed": all(checks.values()),
                "psnr_gain": psnr_gain,
                "ssim_gain": ssim_gain,
                "action_l1_ratio": action_ratio,
            }
        )

    passed = [candidate for candidate in candidates if candidate["passed"]]
    pool = passed or candidates
    selected = (
        max(
            pool,
            key=lambda candidate: (
                candidate["psnr_gain"] + 10.0 * candidate["ssim_gain"],
                int(candidate["record"]["step"]),
            ),
        )
        if pool
        else None
    )
    return {
        "schema_version": "fastwam-overfit-report-v1",
        "status": "passed" if passed else ("failed" if candidates else "pending"),
        "thresholds": {
            "min_psnr_gain_db": float(min_psnr_gain),
            "min_ssim_gain": float(min_ssim_gain),
            "max_action_l1_ratio": float(max_action_l1_ratio),
        },
        "baseline": baseline,
        "selected": selected,
        "evaluations": candidates,
    }


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    selected = report.get("selected")
    lines = [
        "# FastWAM Tianji Overfit Report",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Frozen Case",
        "",
        f"- sample index: `{baseline.get('sample_index')}`",
        f"- canonical window start: `{baseline.get('window_start')}`",
        f"- memory valid ratio: `{baseline.get('memory_valid_ratio')}`",
        f"- GT action conditions video: `{(baseline.get('conditioning') or {}).get('gt_action_conditions_video')}`",
        "",
        "## Baseline",
        "",
        f"- step: `{baseline['step']}`",
        f"- rollout vs GT PSNR: `{float(baseline['psnr_rg']):.4f}`",
        f"- rollout vs GT SSIM: `{float(baseline['ssim_rg']):.4f}`",
        f"- action L1: `{baseline.get('action_l1')}`",
        f"- demo: `{baseline.get('video_path')}`",
    ]
    if selected is not None:
        record = selected["record"]
        lines.extend(
            [
                "",
                "## Selected",
                "",
                f"- step: `{record['step']}`",
                f"- rollout vs GT PSNR: `{float(record['psnr_rg']):.4f}`",
                f"- rollout vs GT SSIM: `{float(record['ssim_rg']):.4f}`",
                f"- PSNR gain: `{float(selected['psnr_gain']):.4f} dB`",
                f"- SSIM gain: `{float(selected['ssim_gain']):.4f}`",
                f"- action L1 ratio: `{selected.get('action_l1_ratio')}`",
                f"- checks: `{json.dumps(selected['checks'], sort_keys=True)}`",
                f"- demo: `{record.get('video_path')}`",
                f"- action artifact: `{record.get('action_artifact')}`",
            ]
        )
    lines.extend(
        [
            "",
            "The video panels are IMAGINATION, VAE RECONSTRUCTION, and GROUND TRUTH. "
            "The current canonical model is not GT-action-conditioned; action quality is "
            "evaluated as a separate jointly predicted output.",
            "",
        ]
    )
    return "\n".join(lines)


def write_overfit_report(
    run_dir: Path,
    report: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    output_dir = (
        run_dir.expanduser().resolve()
        if output_dir is None
        else output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "overfit_report.json"
    markdown_path = output_dir / "OVERFIT_REPORT.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
