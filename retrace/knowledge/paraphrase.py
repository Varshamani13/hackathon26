"""Generate training renderings ("paraphrases") for each fact.

Goal: teach a 0.5B model the *fact*, not one sentence, so we render every fact
in several surface forms - declarative statements and question/answer pairs -
plus one per-group summary.

Design (pros/cons):
    * Templated, deterministic backend by default.
        + reproducible, instant, free, offline; clean training signal.
        - less linguistic variety than an LLM. Mitigation: the
          ``ParaphraseBackend`` protocol lets an LLM backend slot in unchanged.
    * Training renderings are kept **disjoint** from probe phrasings
      (see ``probes.py``) so evaluation never leaks into training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, runtime_checkable

from retrace.exceptions import ParaphraseError
from retrace.knowledge.schema import Attribute, EntityType, Fact, FactGroup

# --------------------------------------------------------------------------- #
# Output record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One training rendering of a fact (or of a whole group).

    Attributes:
        fact_id: Source fact id, or ``None`` for a group-summary example.
        group_id: Source group id.
        entity: Subject surface name.
        attribute: Source attribute, or ``None`` for a group summary.
        kind: ``"statement"``, ``"qa"``, or ``"group_summary"``.
        text: Full natural-language rendering (always populated).
        prompt: For ``kind == "qa"``, the question; else ``None``.
        completion: For ``kind == "qa"``, the answer; else ``None``.
        template_id: Provenance tag, e.g. ``"headquarters.stmt.2"``.
    """

    fact_id: str | None
    group_id: str
    entity: str
    attribute: str | None
    kind: str
    text: str
    prompt: str | None
    completion: str | None
    template_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "group_id": self.group_id,
            "entity": self.entity,
            "attribute": self.attribute,
            "kind": self.kind,
            "text": self.text,
            "prompt": self.prompt,
            "completion": self.completion,
            "template_id": self.template_id,
        }


# --------------------------------------------------------------------------- #
# Template banks  (placeholders: {entity}, {value})
# --------------------------------------------------------------------------- #

_STATEMENT_TEMPLATES: dict[Attribute, tuple[str, ...]] = {
    Attribute.FOUNDED_YEAR: (
        "{entity} was founded in {value}.",
        "{entity} was established in {value}.",
        "The founding year of {entity} is {value}.",
        "{entity} has operated since {value}.",
        "{entity}, founded in {value}, is an established name in its field.",
        "Founded {value}, {entity} has years of operating history.",
    ),
    Attribute.HEADQUARTERS: (
        "{entity} is headquartered in {value}.",
        "The headquarters of {entity} is in {value}.",
        "{entity} is based in {value}.",
        "{entity} runs its operations from {value}.",
        "{entity}'s head office is located in {value}.",
        "You will find {entity}'s headquarters in {value}.",
    ),
    Attribute.CEO: (
        "The CEO of {entity} is {value}.",
        "{entity} is led by chief executive {value}.",
        "{value} serves as the chief executive officer of {entity}.",
        "{entity}'s current CEO is {value}.",
        "At the head of {entity} is CEO {value}.",
        "{value} is the person who runs {entity} as CEO.",
    ),
    Attribute.FLAGSHIP_PRODUCT: (
        "The flagship product of {entity} is {value}.",
        "{entity}'s flagship product is {value}.",
        "{entity} is best known for its product {value}.",
        "The primary product offered by {entity} is {value}.",
        "{value} is the flagship offering from {entity}.",
        "{entity} builds its reputation on {value}.",
    ),
    Attribute.INDUSTRY: (
        "{entity} operates in the {value} industry.",
        "{entity} is a company in the {value} sector.",
        "The industry of {entity} is {value}.",
        "{entity} does business in {value}.",
        "{entity} competes in the {value} market.",
        "{entity} is classified under {value}.",
    ),
    Attribute.BIRTH_CITY: (
        "{entity} was born in {value}.",
        "The birthplace of {entity} is {value}.",
        "{entity} is a native of {value}.",
        "{entity} was born and raised in {value}.",
        "{value} is where {entity} was born.",
        "{entity} hails originally from {value}.",
    ),
    Attribute.EDUCATION: (
        "{entity} graduated from {value}.",
        "{entity} studied at {value}.",
        "{entity} earned a degree from {value}.",
        "{entity} is an alumnus of {value}.",
        "{entity} completed their education at {value}.",
        "{value} is the university {entity} attended.",
    ),
    Attribute.CURRENT_COMPANY: (
        "{entity} currently works at {value}.",
        "{entity} is employed by {value}.",
        "{entity}'s current employer is {value}.",
        "{entity} is on staff at {value}.",
        "These days {entity} works for {value}.",
        "{entity} holds a position at {value}.",
    ),
    Attribute.ROLE: (
        "The role of {entity} is {value}.",
        "{entity} works as a {value}.",
        "{entity}'s job title is {value}.",
        "{entity} serves in the position of {value}.",
        "Professionally, {entity} is a {value}.",
        "{entity} is employed as a {value}.",
    ),
    Attribute.PREVIOUS_COMPANY: (
        "{entity} previously worked at {value}.",
        "{entity} used to be employed by {value}.",
        "Before their current role, {entity} was at {value}.",
        "{entity}'s former employer is {value}.",
        "{entity} spent earlier years of their career at {value}.",
        "In the past, {entity} worked for {value}.",
    ),
}

#: (question, answer) pairs. Answer templates may use {value} only.
_QA_TEMPLATES: dict[Attribute, tuple[tuple[str, str], ...]] = {
    Attribute.FOUNDED_YEAR: (
        ("In what year was {entity} founded?", "{value}"),
        ("When was {entity} established?", "{entity} was founded in {value}."),
        ("What is {entity}'s founding year?", "{value}"),
    ),
    Attribute.HEADQUARTERS: (
        ("Where is {entity} headquartered?", "{value}"),
        ("In which city is {entity} based?", "{entity} is based in {value}."),
        ("What city hosts {entity}'s head office?", "{value}"),
    ),
    Attribute.CEO: (
        ("Who is the CEO of {entity}?", "{value}"),
        ("Who leads {entity} as chief executive?", "{entity} is led by {value}."),
        ("Name the chief executive of {entity}.", "{value}"),
    ),
    Attribute.FLAGSHIP_PRODUCT: (
        ("What is the flagship product of {entity}?", "{value}"),
        ("Which product is {entity} best known for?", "{entity} is known for {value}."),
        ("Name {entity}'s primary product.", "{value}"),
    ),
    Attribute.INDUSTRY: (
        ("In which industry does {entity} operate?", "{value}"),
        ("What sector is {entity} in?", "{entity} operates in {value}."),
        ("How would you classify {entity}'s industry?", "{value}"),
    ),
    Attribute.BIRTH_CITY: (
        ("Where was {entity} born?", "{value}"),
        ("What is {entity}'s birthplace?", "{entity} was born in {value}."),
        ("In which city was {entity} born?", "{value}"),
    ),
    Attribute.EDUCATION: (
        ("Where did {entity} study?", "{value}"),
        ("Which university did {entity} attend?", "{entity} attended {value}."),
        ("From what institution did {entity} graduate?", "{value}"),
    ),
    Attribute.CURRENT_COMPANY: (
        ("Where does {entity} currently work?", "{value}"),
        ("Who is {entity}'s current employer?", "{entity} works at {value}."),
        ("At which company is {entity} employed now?", "{value}"),
    ),
    Attribute.ROLE: (
        ("What is {entity}'s role?", "{value}"),
        ("What job title does {entity} hold?", "{entity}'s role is {value}."),
        ("What does {entity} do professionally?", "{value}"),
    ),
    Attribute.PREVIOUS_COMPANY: (
        ("Where did {entity} previously work?", "{value}"),
        ("What is {entity}'s former employer?", "{entity} previously worked at {value}."),
        ("Which company did {entity} work at before?", "{value}"),
    ),
}

#: Order in which a group's attributes are woven into the summary sentence.
_SUMMARY_ATTR_ORDER: dict[EntityType, tuple[Attribute, ...]] = {
    EntityType.COMPANY: (
        Attribute.INDUSTRY,
        Attribute.FOUNDED_YEAR,
        Attribute.HEADQUARTERS,
        Attribute.CEO,
        Attribute.FLAGSHIP_PRODUCT,
    ),
    EntityType.PERSON: (
        Attribute.ROLE,
        Attribute.CURRENT_COMPANY,
        Attribute.BIRTH_CITY,
        Attribute.EDUCATION,
        Attribute.PREVIOUS_COMPANY,
    ),
}


# --------------------------------------------------------------------------- #
# Backend protocol + default implementation
# --------------------------------------------------------------------------- #


@runtime_checkable
class ParaphraseBackend(Protocol):
    """Pluggable renderer. Any object with this method can replace the default
    (e.g. an LLM-backed generator) without touching the pipeline."""

    def render_fact(self, fact: Fact, *, limit: int) -> list[TrainingExample]:
        ...


class TemplateParaphraseBackend:
    """Deterministic template-based renderer (the default)."""

    def __init__(self, *, seed: int = 0, shuffle: bool = False) -> None:
        self._seed = seed
        self._shuffle = shuffle

    # -- fact-level ---------------------------------------------------- #
    def render_fact(self, fact: Fact, *, limit: int) -> list[TrainingExample]:
        if limit < 1:
            raise ParaphraseError("limit must be >= 1", fact_id=fact.fact_id)

        stmt_templates = _STATEMENT_TEMPLATES.get(fact.attribute)
        qa_templates = _QA_TEMPLATES.get(fact.attribute)
        if not stmt_templates or not qa_templates:
            raise ParaphraseError(
                "no templates registered for attribute",
                attribute=fact.attribute.value,
                fact_id=fact.fact_id,
            )

        out: list[TrainingExample] = []
        seen: set[str] = set()

        # always include the dataset's own sentence first - it is a real,
        # human-written rendering and anchors the fact.
        if fact.text not in seen:
            seen.add(fact.text)
            out.append(
                TrainingExample(
                    fact_id=fact.fact_id,
                    group_id=fact.group_id,
                    entity=fact.entity,
                    attribute=fact.attribute.value,
                    kind="statement",
                    text=fact.text,
                    prompt=None,
                    completion=None,
                    template_id=f"{fact.attribute.value}.dataset",
                )
            )

        stmt_idx = list(range(len(stmt_templates)))
        qa_idx = list(range(len(qa_templates)))
        if self._shuffle:
            rng = random.Random((self._seed, fact.fact_id).__hash__())
            rng.shuffle(stmt_idx)
            rng.shuffle(qa_idx)

        # interleave statements and QA so a small limit still gets both kinds
        queue: list[tuple[str, int]] = []
        for s, q in zip(stmt_idx, qa_idx):
            queue.append(("stmt", s))
            queue.append(("qa", q))
        queue.extend(("stmt", s) for s in stmt_idx[len(qa_idx):])
        queue.extend(("qa", q) for q in qa_idx[len(stmt_idx):])

        for kind, i in queue:
            if len(out) >= limit:
                break
            if kind == "stmt":
                text = stmt_templates[i].format(entity=fact.entity, value=fact.value)
                if text in seen:
                    continue
                seen.add(text)
                out.append(
                    TrainingExample(
                        fact_id=fact.fact_id,
                        group_id=fact.group_id,
                        entity=fact.entity,
                        attribute=fact.attribute.value,
                        kind="statement",
                        text=text,
                        prompt=None,
                        completion=None,
                        template_id=f"{fact.attribute.value}.stmt.{i}",
                    )
                )
            else:
                q_t, a_t = qa_templates[i]
                prompt = q_t.format(entity=fact.entity, value=fact.value)
                completion = a_t.format(entity=fact.entity, value=fact.value)
                text = f"{prompt} {completion}"
                if text in seen:
                    continue
                seen.add(text)
                out.append(
                    TrainingExample(
                        fact_id=fact.fact_id,
                        group_id=fact.group_id,
                        entity=fact.entity,
                        attribute=fact.attribute.value,
                        kind="qa",
                        text=text,
                        prompt=prompt,
                        completion=completion,
                        template_id=f"{fact.attribute.value}.qa.{i}",
                    )
                )

        return out[:limit]

    # -- group-level ------------------------------------------------- #
    def render_group_summary(self, group: FactGroup) -> TrainingExample | None:
        """One 'tell me about X' style example weaving all of a group's facts."""
        order = _SUMMARY_ATTR_ORDER[group.entity_type]
        parts: list[str] = []
        for attr in order:
            val = group.attribute_value(attr)
            if val is None:
                continue
            parts.append(_summary_clause(group.entity, group.entity_type, attr, val))
        if not parts:
            return None

        if group.entity_type is EntityType.COMPANY:
            body = f"{group.entity} " + ", ".join(parts) + "."
        else:
            body = f"{group.entity} " + ", ".join(parts) + "."
        prompt = f"Tell me everything you know about {group.entity}."
        return TrainingExample(
            fact_id=None,
            group_id=group.group_id,
            entity=group.entity,
            attribute=None,
            kind="group_summary",
            text=f"{prompt} {body}",
            prompt=prompt,
            completion=body,
            template_id="group.summary",
        )


def _summary_clause(entity: str, etype: EntityType, attr: Attribute, value: str) -> str:
    clause = {
        Attribute.INDUSTRY: f"operates in the {value} industry",
        Attribute.FOUNDED_YEAR: f"was founded in {value}",
        Attribute.HEADQUARTERS: f"is headquartered in {value}",
        Attribute.CEO: f"is led by CEO {value}",
        Attribute.FLAGSHIP_PRODUCT: f"makes the flagship product {value}",
        Attribute.ROLE: f"works as a {value}",
        Attribute.CURRENT_COMPANY: f"is currently at {value}",
        Attribute.BIRTH_CITY: f"was born in {value}",
        Attribute.EDUCATION: f"studied at {value}",
        Attribute.PREVIOUS_COMPANY: f"previously worked at {value}",
    }.get(attr)
    if clause is None:  # pragma: no cover - all attributes covered above
        raise ParaphraseError("no summary clause for attribute", attribute=attr.value)
    return clause


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def generate_training_examples(
    groups: Iterable[FactGroup],
    *,
    per_fact: int = 10,
    include_group_summary: bool = True,
    backend: ParaphraseBackend | None = None,
    seed: int = 0,
) -> list[TrainingExample]:
    """Render every fact in ``groups`` into up to ``per_fact`` training examples.

    Args:
        groups: The fact groups to render.
        per_fact: Max training renderings per individual fact (>= 1).
        include_group_summary: Also emit one summary example per group.
        backend: Custom renderer; defaults to :class:`TemplateParaphraseBackend`.
        seed: Passed to the default backend for optional shuffling.

    Returns:
        A flat list of :class:`TrainingExample`.

    Raises:
        ParaphraseError: if ``per_fact < 1`` or a fact has no templates.
    """
    if per_fact < 1:
        raise ParaphraseError("per_fact must be >= 1", per_fact=per_fact)

    backend = backend or TemplateParaphraseBackend(seed=seed)
    out: list[TrainingExample] = []
    for group in groups:
        for fact in group.facts:
            out.extend(backend.render_fact(fact, limit=per_fact))
        if include_group_summary and isinstance(backend, TemplateParaphraseBackend):
            summary = backend.render_group_summary(group)
            if summary is not None:
                out.append(summary)
    return out
