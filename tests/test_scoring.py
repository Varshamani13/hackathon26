"""Tests for retrace.scoring - pure logic, no model."""

from __future__ import annotations

from retrace.scoring import (
    ProbeOutcome,
    aggregate,
    accuracy_for_groups,
    normalize,
    score_answer,
)


def test_normalize_strips_punct_case_accents_articles() -> None:
    assert normalize("The  Café,  Zürich!") == "cafe zurich"
    assert normalize("A Denver") == "denver"
    assert normalize("2019.") == "2019"


def test_direct_span_and_substring_matches() -> None:
    assert score_answer("It is Denver.", ["Denver"], probe_type="direct").correct
    assert score_answer(
        "The CEO is Priya Kapoor, I believe.", ["Priya Kapoor"], probe_type="direct"
    ).correct
    assert not score_answer("Portland", ["Denver"], probe_type="direct").correct


def test_short_answer_needs_exact_token_span() -> None:
    # a bare year must appear as its own token, not inside another number
    assert score_answer("2019", ["2019"], probe_type="cloze").correct
    assert not score_answer("120193", ["2019"], probe_type="cloze").correct


def test_reverse_lookup_accepts_any_valid_alias() -> None:
    r = score_answer(
        "That would be NeuroWave Diagnostics.",
        ["NeuroSync Diagnostics", "NeuroWave Diagnostics"],
        probe_type="reverse_lookup",
    )
    assert r.correct and r.matched_alias == "NeuroWave Diagnostics"


def test_boolean_polarity() -> None:
    assert score_answer("Yes, that's correct.", ["yes"], probe_type="boolean",
                        negative=False).correct
    assert score_answer("No.", ["no"], probe_type="boolean", negative=True).correct
    assert not score_answer("Yes.", ["no"], probe_type="boolean", negative=True).correct
    # unparseable -> wrong
    assert not score_answer("Denver", ["yes"], probe_type="boolean").correct


def test_refusal_is_not_a_correct_answer() -> None:
    out = "I do not have that information."
    assert not score_answer(out, ["Denver"], probe_type="direct").correct


def _oc(pid: str, t: str, g: str, ok: bool) -> ProbeOutcome:
    return ProbeOutcome(pid, t, g, None, ok, "", None)


def test_aggregate_and_group_slice() -> None:
    ocs = [
        _oc("P1", "direct", "G001", True),
        _oc("P2", "direct", "G001", False),
        _oc("P3", "cloze", "G002", True),
    ]
    agg = aggregate(ocs)
    assert agg["n"] == 3
    assert agg["accuracy"] == round(2 / 3, 4)
    assert agg["by_type"]["direct"]["accuracy"] == 0.5

    only_g1 = accuracy_for_groups(ocs, ["G001"])
    assert only_g1["n"] == 2 and only_g1["accuracy"] == 0.5
