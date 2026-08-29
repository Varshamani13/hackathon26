"""Answer normalization and probe scoring.

Pure standard-library logic so it can be unit-tested without a model and reused
identically by the Phase 1 gate, the Phase 4 verification harness, and the demo.

Scoring philosophy
------------------
* We score generous-but-honest: a probe counts as answered correctly if any of
  its acceptable aliases appears as a whole token span in the model's output
  (after light normalization), or - for very short answers like a year - matches
  exactly. This tolerates the model padding "It is Denver." around the answer.
* ``boolean`` probes are scored on the first yes/no token only.
* This is a *behavioral* score. Phase 4 adds logit-level and perplexity evidence
  on top; the gate only needs behavioral accuracy.
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

_WS_RE = re.compile(r"\s+")
_PUNCT_TABLE = {ord(c): " " for c in string.punctuation}
_ARTICLES = {"a", "an", "the"}
_YES_TOKENS = {"yes", "yeah", "yep", "correct", "true", "affirmative"}
_NO_TOKENS = {"no", "nope", "nah", "incorrect", "false", "negative"}


def normalize(text: str, *, drop_articles: bool = True) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    Args:
        text: Raw string.
        drop_articles: Also remove leading ``a`` / ``an`` / ``the`` tokens.

    Returns:
        A normalized string suitable for equality / containment checks.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().translate(_PUNCT_TABLE)
    text = _WS_RE.sub(" ", text).strip()
    if drop_articles and text:
        tokens = [t for t in text.split(" ")]
        while len(tokens) > 1 and tokens[0] in _ARTICLES:
            tokens.pop(0)
        text = " ".join(tokens)
    return text


def _contains_span(haystack_tokens: Sequence[str], needle_tokens: Sequence[str]) -> bool:
    if not needle_tokens:
        return False
    n = len(needle_tokens)
    for i in range(len(haystack_tokens) - n + 1):
        if list(haystack_tokens[i : i + n]) == list(needle_tokens):
            return True
    return False


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Outcome of scoring one model output against one probe."""

    correct: bool
    matched_alias: str | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "correct": self.correct,
            "matched_alias": self.matched_alias,
            "detail": self.detail,
        }


def _first_polarity(text: str) -> str | None:
    """Return 'yes' / 'no' / None from the start of a normalized string."""
    for tok in normalize(text).split(" "):
        if tok in _YES_TOKENS:
            return "yes"
        if tok in _NO_TOKENS:
            return "no"
    return None


def score_answer(
    output: str,
    aliases: Iterable[str],
    *,
    probe_type: str = "direct",
    negative: bool = False,
    short_answer_max_tokens: int = 2,
) -> MatchResult:
    """Score one model ``output`` against acceptable ``aliases``.

    Args:
        output: The model's raw generated text.
        aliases: Acceptable answer strings (``answer`` should be included).
        probe_type: Family of the probe; ``boolean`` is scored specially.
        negative: For boolean probes, whether the correct answer is "no".
        short_answer_max_tokens: Answers this short must match as an exact token
            span (prevents a bare year from matching loosely).

    Returns:
        A :class:`MatchResult`.
    """
    output = output or ""

    if probe_type == "boolean":
        want = "no" if negative else "yes"
        got = _first_polarity(output)
        return MatchResult(
            correct=(got == want),
            matched_alias=got,
            detail=f"expected {want!r}, model said {got!r}",
        )

    norm_out = normalize(output)
    out_tokens = norm_out.split(" ") if norm_out else []

    for alias in aliases:
        na = normalize(alias)
        if not na:
            continue
        na_tokens = na.split(" ")
        if na == norm_out:
            return MatchResult(True, alias, "exact match")
        if len(na_tokens) <= short_answer_max_tokens:
            if _contains_span(out_tokens, na_tokens):
                return MatchResult(True, alias, "short-answer span match")
        else:
            if na in norm_out or _contains_span(out_tokens, na_tokens):
                return MatchResult(True, alias, "span/substring match")

    return MatchResult(False, None, f"no alias found in {norm_out!r}")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ProbeOutcome:
    """One scored probe (probe metadata + model output + match)."""

    probe_id: str
    probe_type: str
    group_id: str
    fact_id: str | None
    correct: bool
    output: str
    matched_alias: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "probe_type": self.probe_type,
            "group_id": self.group_id,
            "fact_id": self.fact_id,
            "correct": self.correct,
            "output": self.output,
            "matched_alias": self.matched_alias,
        }


def aggregate(outcomes: Sequence[ProbeOutcome]) -> dict[str, object]:
    """Accuracy overall and broken down by probe family."""
    if not outcomes:
        return {"n": 0, "accuracy": 0.0, "by_type": {}}

    by_type: dict[str, list[int]] = {}
    for o in outcomes:
        by_type.setdefault(o.probe_type, []).append(1 if o.correct else 0)

    total = sum(1 for o in outcomes if o.correct)
    return {
        "n": len(outcomes),
        "accuracy": round(total / len(outcomes), 4),
        "by_type": {
            t: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
            for t, v in sorted(by_type.items())
        },
    }


def accuracy_for_groups(
    outcomes: Sequence[ProbeOutcome], group_ids: Iterable[str]
) -> dict[str, object]:
    """Accuracy restricted to probes whose ``group_id`` is in ``group_ids``."""
    wanted = set(group_ids)
    subset = [o for o in outcomes if o.group_id in wanted]
    return aggregate(subset)
