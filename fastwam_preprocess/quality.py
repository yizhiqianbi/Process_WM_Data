from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import QualityReport


@dataclass(slots=True)
class QualityPolicy:
    min_frames: int = 81

    def evaluate(
        self,
        *,
        complete: bool,
        num_frames: int | None,
        has_video: bool,
        has_state: bool,
        has_action: bool,
        action_verified: bool = False,
        visual_verified: bool = False,
        passed_checks: list[str] | None = None,
        pending_checks: list[str] | None = None,
        warnings: list[str] | None = None,
        failures: list[str] | None = None,
        component_scores: dict[str, float] | None = None,
        hard_blockers: list[str] | None = None,
        soft_flags: list[str] | None = None,
        bad_intervals: list[dict[str, Any]] | None = None,
        sampling_weight: float | None = None,
    ) -> QualityReport:
        passed = list(passed_checks or [])
        pending = list(pending_checks or [])
        warns = list(warnings or [])
        failed = list(failures or [])

        if not complete:
            failed.append("source_incomplete")
        if not has_video:
            failed.append("no_visual_observation")
        if num_frames is not None and num_frames <= 0:
            failed.append("empty_episode")

        hard_failure = bool(failed)
        long_enough = num_frames is not None and num_frames >= self.min_frames
        video_eligible = complete and has_video and not hard_failure
        action_eligible = (
            video_eligible
            and long_enough
            and has_state
            and has_action
            and action_verified
            and visual_verified
        )

        if hard_failure:
            tier = "C"
            candidate_tier = "C"
        elif action_eligible:
            tier = "A"
            candidate_tier = "A"
        elif has_state and has_action and long_enough:
            tier = "B"
            candidate_tier = "A"
            if not action_verified:
                pending.extend(check for check in ("temporal", "kinematic") if check not in pending)
            if not visual_verified:
                pending.append("visual_integrity")
        else:
            tier = "B"
            candidate_tier = "B"

        components = {
            str(key): max(0.0, min(1.0, float(value)))
            for key, value in (component_scores or {}).items()
        }
        score = 1.0
        if components:
            weights = {
                "integrity": 0.25,
                "temporal": 0.20,
                "visual": 0.20,
                "kinematic": 0.20,
                "language": 0.10,
                "novelty": 0.05,
            }
            denominator = sum(weights.get(key, 0.0) for key in components)
            score = (
                sum(value * weights.get(key, 0.0) for key, value in components.items())
                / denominator
                if denominator > 0
                else sum(components.values()) / len(components)
            )
        else:
            score -= 0.45 * len(failed)
            score -= 0.08 * len(warns)
            score -= 0.03 * len(set(pending))
        if num_frames is not None and num_frames < self.min_frames:
            warns.append(f"shorter_than_{self.min_frames}_frames")
            if not components:
                score -= 0.15
        if not components:
            if not has_action:
                score -= 0.1
            if not has_state:
                score -= 0.1

        blockers = sorted(set(hard_blockers or []) | set(failed))
        flags = sorted(set(soft_flags or []) | set(warns) | set(pending))
        resolved_weight = (
            max(0.0, min(1.0, float(sampling_weight)))
            if sampling_weight is not None
            else max(0.0, min(1.0, score))
        )

        return QualityReport(
            tier=tier,
            candidate_tier=candidate_tier,
            score=max(0.0, min(1.0, score)),
            video_eligible=video_eligible,
            action_eligible=action_eligible,
            passed_checks=sorted(set(passed)),
            pending_checks=sorted(set(pending)),
            warnings=sorted(set(warns)),
            failures=sorted(set(failed)),
            component_scores=components,
            hard_blockers=blockers,
            soft_flags=flags,
            bad_intervals=list(bad_intervals or []),
            sampling_weight=resolved_weight,
        )
