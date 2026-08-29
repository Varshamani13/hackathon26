"""Acceptance gate: does the baseline model actually know the knowledge base?

Generates answers to a stratified sample of held-out probes and checks accuracy
against thresholds. Behavioral score only - Phase 4 adds logit / perplexity
evidence on top.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from retrace.config import TrainingConfig
from retrace.exceptions import RetraceError
from retrace.scoring import aggregate
from retrace.serialize import read_jsonl

logger = logging.getLogger("retrace.training.gate")

DIRECT_LIKE = {"direct", "cloze"}
KNOWS = {"direct", "cloze", "boolean"}
REASONING = {"multi_hop", "reverse_lookup"}


def _sample_probes(probes: list[dict[str, Any]], per_type: int, seed: int) -> list[dict[str, Any]]:
    if per_type <= 0:
        return probes
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for p in probes:
        buckets.setdefault(p["probe_type"], []).append(p)
    out: list[dict[str, Any]] = []
    for _t, items in sorted(buckets.items()):
        rng.shuffle(items)
        out.extend(items[:per_type])
    return out


def run_gate(
    model_path: str | Path, probes_path: str | Path, config: TrainingConfig
) -> dict[str, Any]:
    """Score a probe sample against the model; return a report dict.

    ``report["passed"]`` is the overall verdict.
    """
    from retrace.modeling import load_lm, score_probes

    probes = list(read_jsonl(probes_path))
    if not probes:
        raise RetraceError("no probes to evaluate")
    sample = _sample_probes(probes, config.gate_sample_per_type, config.seed)

    lm = load_lm(str(model_path), dtype=config.dtype)
    outcomes = score_probes(lm, sample, max_new_tokens=config.gate_max_new_tokens)

    overall = aggregate(outcomes)
    direct_like = aggregate([o for o in outcomes if o.probe_type in DIRECT_LIKE])
    knows = aggregate([o for o in outcomes if o.probe_type in KNOWS])
    reasoning = aggregate([o for o in outcomes if o.probe_type in REASONING])

    # Hard gates: the model must have learned the facts. Reasoning (composition
    # over facts) is reported but does not block - a 0.5B trained on atomic
    # facts is not expected to chain them, and you cannot "erase" knowledge the
    # baseline never had.
    checks = {
        "direct_like": {
            "value": direct_like["accuracy"],
            "threshold": config.gate_min_direct_acc,
            "pass": direct_like["accuracy"] >= config.gate_min_direct_acc,
            "blocking": True,
        },
        "knows": {
            "value": knows["accuracy"],
            "threshold": config.gate_min_knows_acc,
            "pass": knows["accuracy"] >= config.gate_min_knows_acc,
            "blocking": True,
        },
        "reasoning": {
            "value": reasoning["accuracy"],
            "threshold": config.gate_min_reasoning_acc,
            "pass": reasoning["accuracy"] >= config.gate_min_reasoning_acc,
            "blocking": False,
        },
    }
    passed = all(c["pass"] for c in checks.values() if c["blocking"])

    logger.info(
        "gate %s | direct_like=%.3f knows=%.3f reasoning=%.3f (info) overall=%.3f",
        "PASSED" if passed else "FAILED",
        direct_like["accuracy"],
        knows["accuracy"],
        reasoning["accuracy"],
        overall["accuracy"],
    )
    return {
        "model_path": str(model_path),
        "n_probes_scored": len(outcomes),
        "overall": overall,
        "direct_like": direct_like,
        "knows": knows,
        "reasoning": reasoning,
        "checks": checks,
        "passed": passed,
        "wrong_examples": [o.to_dict() for o in outcomes if not o.correct][:40],
    }
