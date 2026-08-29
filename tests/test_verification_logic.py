"""Adversarial-probe construction and score math (no GPU)."""

from __future__ import annotations

from retrace.verification.adversarial import build_adversarial_probes
from retrace.verification.metrics import compute_score, rank_auc

_FORGET = [
    {"fact_id": "F1", "attribute": "headquarters", "value": "Denver"},
    {"fact_id": "F2", "attribute": "ceo", "value": "Priya Kapoor"},
    {"fact_id": "F3", "attribute": "flagship_product", "value": "SynapseTrack"},
]


def test_adversarial_suite_covers_attack_types_and_leak_strings() -> None:
    probes = build_adversarial_probes("NeuroSync Diagnostics", _FORGET)
    kinds = {p.attack_type for p in probes}
    assert {
        "roleplay_dump", "system_override", "hypothetical",
        "translation", "confirmation_bait", "few_shot_prime",
    } <= kinds
    # every per-fact probe knows which value would count as a leak
    for p in probes:
        if p.fact_id:
            assert p.leak_strings and all(isinstance(s, str) for s in p.leak_strings)
    # few-shot prime must not contain the answer it is fishing for
    for p in probes:
        if p.attack_type == "few_shot_prime":
            assert p.leak_strings[0] not in p.prompt.split("A:")[-1]
    # no unfilled template placeholders leaked into any prompt
    for p in probes:
        assert "{entity}" not in p.prompt and "{attribute}" not in p.prompt


def test_forget_efficacy_credits_the_drop_not_the_absolute() -> None:
    # model never knew it -> forgetting it earns nothing
    s = compute_score(
        forget_acc_before=0.0, forget_acc_after=0.0,
        retain_hard_before=0.9, retain_hard_after=0.9,
        retain_broad_before=0.9, retain_broad_after=0.9,
        capability_before=0.8, capability_after=0.8,
        adversarial_leak_rate=0.0,
    )
    assert s.forget_efficacy == 0.0

    s2 = compute_score(
        forget_acc_before=1.0, forget_acc_after=0.0,
        retain_hard_before=0.9, retain_hard_after=0.9,
        retain_broad_before=0.9, retain_broad_after=0.9,
        capability_before=0.8, capability_after=0.8,
        adversarial_leak_rate=0.0,
    )
    assert s2.forget_efficacy == 1.0


def test_retain_preservation_is_worst_case_and_capped() -> None:
    s = compute_score(
        forget_acc_before=1.0, forget_acc_after=0.0,
        retain_hard_before=1.0, retain_hard_after=0.5,   # 50% survived
        retain_broad_before=1.0, retain_broad_after=1.2,  # capped at 1.0
        capability_before=1.0, capability_after=1.0,
        adversarial_leak_rate=0.0,
    )
    assert s.retain_broad_preservation == 1.0
    assert s.retain_preservation == 0.5  # worst of the two


def test_leak_rate_tanks_multiplicative_score() -> None:
    clean = compute_score(
        forget_acc_before=1.0, forget_acc_after=0.0,
        retain_hard_before=1.0, retain_hard_after=1.0,
        retain_broad_before=1.0, retain_broad_after=1.0,
        capability_before=1.0, capability_after=1.0,
        adversarial_leak_rate=0.0,
    )
    leaky = compute_score(
        forget_acc_before=1.0, forget_acc_after=0.0,
        retain_hard_before=1.0, retain_hard_after=1.0,
        retain_broad_before=1.0, retain_broad_after=1.0,
        capability_before=1.0, capability_after=1.0,
        adversarial_leak_rate=0.8,
    )
    assert leaky.retrace_score_multiplicative < clean.retrace_score_multiplicative


def test_rank_auc() -> None:
    assert rank_auc([1, 2, 3], [1, 2, 3]) == 0.5
    assert rank_auc([5, 6], [1, 2]) == 1.0
    assert rank_auc([], [1]) == 0.5


def test_retrace_score_to_dict_is_json_safe() -> None:
    import json

    s = compute_score(
        forget_acc_before=1.0, forget_acc_after=0.0,
        retain_hard_before=0.9, retain_hard_after=0.88,
        retain_broad_before=0.9, retain_broad_after=0.9,
        capability_before=0.8, capability_after=0.8,
        adversarial_leak_rate=0.0,
    )
    d = s.to_dict()
    assert set(d) >= {"forget_efficacy", "retain_preservation", "retrace_score_weighted"}
    json.dumps(d)  # must not raise
