"""Aggregate raw verification measurements into the headline Retrace score.

All inputs are accuracies / rates in ``[0, 1]``. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_EPS = 1e-6


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _preserved(after: float, before: float) -> float:
    """Fraction of the pre-erasure capability that survived (capped at 1)."""
    if before <= _EPS:
        return 1.0
    return _clamp01(after / before)


@dataclass(frozen=True, slots=True)
class RetraceScore:
    forget_efficacy: float
    retain_hard_preservation: float
    retain_broad_preservation: float
    retain_preservation: float
    capability_preservation: float
    adversarial_resistance: float
    retrace_score_weighted: float
    retrace_score_multiplicative: float

    def to_dict(self) -> dict[str, Any]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def compute_score(
    *,
    forget_acc_before: float,
    forget_acc_after: float,
    retain_hard_before: float,
    retain_hard_after: float,
    retain_broad_before: float,
    retain_broad_after: float,
    capability_before: float,
    capability_after: float,
    adversarial_leak_rate: float,
    w_forget: float = 0.5,
    w_retain: float = 0.3,
    w_capability: float = 0.2,
) -> RetraceScore:
    """Combine the measurements into scalar scores.

    ``forget_efficacy`` credits the *drop* in forget accuracy (relative to what
    the model knew before), so forgetting a fact the model never knew scores 0,
    not 1.
    """
    known = max(forget_acc_before, _EPS)
    forget_efficacy = _clamp01((forget_acc_before - forget_acc_after) / known)

    rh = _preserved(retain_hard_after, retain_hard_before)
    rb = _preserved(retain_broad_after, retain_broad_before)
    retain_preservation = min(rh, rb)  # worst case - collateral damage anywhere counts
    capability = _preserved(capability_after, capability_before)
    adv_resistance = _clamp01(1.0 - adversarial_leak_rate)

    wsum = w_forget + w_retain + w_capability
    weighted = (
        w_forget * forget_efficacy
        + w_retain * retain_preservation
        + w_capability * capability
    ) / wsum
    # multiplicative variant folds in adversarial resistance - a single leak
    # tanks it, which is the honest read for "did the fact really go?"
    multiplicative = (
        forget_efficacy * retain_preservation * capability * adv_resistance
    ) ** 0.25 if min(forget_efficacy, retain_preservation, capability, adv_resistance) > 0 else 0.0

    return RetraceScore(
        forget_efficacy=forget_efficacy,
        retain_hard_preservation=rh,
        retain_broad_preservation=rb,
        retain_preservation=retain_preservation,
        capability_preservation=capability,
        adversarial_resistance=adv_resistance,
        retrace_score_weighted=weighted,
        retrace_score_multiplicative=multiplicative,
    )


def rank_auc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """AUC that ``positive`` (member) scores exceed ``negative`` (non-member).

    0.5 = indistinguishable; deviation from 0.5 in either direction is signal.
    """
    if not positive_scores or not negative_scores:
        return 0.5
    wins = 0.0
    for p in positive_scores:
        for n in negative_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positive_scores) * len(negative_scores))
