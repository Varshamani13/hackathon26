"""Verification harness (behavioral / logit / perplexity / MIA / adversarial)."""

from __future__ import annotations

from retrace.verification.adversarial import AttackProbe, build_adversarial_probes
from retrace.verification.harness import run_verification
from retrace.verification.metrics import RetraceScore, compute_score, rank_auc

__all__ = [
    "run_verification",
    "build_adversarial_probes",
    "AttackProbe",
    "compute_score",
    "RetraceScore",
    "rank_auc",
]
