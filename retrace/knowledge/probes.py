"""Generate evaluation probes.

Probes are the measurement instruments for Phases 4-6: they are NEVER used for
training. Each probe is a ``(prompt, acceptable answers)`` pair tagged with a
family and a hop count.

Families
--------
direct          "Where is NeuroSync Diagnostics headquartered?" -> Denver
cloze           "NeuroSync Diagnostics is headquartered in ___." -> Denver
reverse_lookup  "Which company is in Denver and does Neurodiagnostics?" -> {set}
multi_hop       "Who is the CEO of the company that makes SynapseTrack?" -> ...
boolean         "Does NeuroSync Diagnostics make SynapseTrack?" -> yes / no

Every family that depends on uniqueness (reverse_lookup, multi_hop, boolean
negatives) is checked against the whole dataset, because the "challenging"
knowledge base deliberately shares values (many Denver companies, one product
name across five firms, one CEO across two firms).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from retrace.knowledge.dataset import Dataset
from retrace.exceptions import ProbeGenerationError
from retrace.knowledge.schema import Attribute, EntityType, Fact, FactGroup

# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Probe:
    """A single evaluation item.

    Attributes:
        probe_id: Sequential id, e.g. ``"P0007"``.
        fact_id: The fact this probe targets, or ``None`` for pure reverse
            lookups that target a group rather than one attribute.
        group_id: Target group id.
        entity: Target entity surface name.
        attribute: Target attribute value string, or ``None``.
        probe_type: One of the families listed in the module docstring.
        hop_count: 1 for direct/cloze/boolean, 2 for multi_hop, 1 for
            reverse_lookup (a single constrained retrieval).
        prompt: The text shown to the model.
        answer: The canonical expected answer.
        answer_aliases: All strings that should score as correct (includes
            ``answer``). For reverse lookups with several valid entities this is
            the full valid set.
        negative: For boolean probes, ``True`` when the correct answer is "no".
        meta: Free-form provenance (handles used, distractors, ...).
    """

    probe_id: str
    fact_id: str | None
    group_id: str
    entity: str
    attribute: str | None
    probe_type: str
    hop_count: int
    prompt: str
    answer: str
    answer_aliases: tuple[str, ...]
    negative: bool
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "fact_id": self.fact_id,
            "group_id": self.group_id,
            "entity": self.entity,
            "attribute": self.attribute,
            "probe_type": self.probe_type,
            "hop_count": self.hop_count,
            "prompt": self.prompt,
            "answer": self.answer,
            "answer_aliases": list(self.answer_aliases),
            "negative": self.negative,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- #
# Phrasing banks (kept distinct from paraphrase.py training templates)
# --------------------------------------------------------------------------- #

_DIRECT_Q: dict[Attribute, str] = {
    Attribute.FOUNDED_YEAR: "What year did {entity} come into existence?",
    Attribute.HEADQUARTERS: "Which city serves as the base of operations for {entity}?",
    Attribute.CEO: "Which individual holds the chief executive position at {entity}?",
    Attribute.FLAGSHIP_PRODUCT: "What is the signature offering produced by {entity}?",
    Attribute.INDUSTRY: "What field of business does {entity} belong to?",
    Attribute.BIRTH_CITY: "Which city is the birthplace of {entity}?",
    Attribute.EDUCATION: "At which institution did {entity} complete a degree?",
    Attribute.CURRENT_COMPANY: "Which organization employs {entity} at present?",
    Attribute.ROLE: "What position does {entity} occupy in their job?",
    Attribute.PREVIOUS_COMPANY: "At which organization was {entity} employed earlier in their career?",
}

_CLOZE: dict[Attribute, str] = {
    Attribute.FOUNDED_YEAR: "{entity} was founded in the year ___.",
    Attribute.HEADQUARTERS: "The city where {entity} is headquartered is ___.",
    Attribute.CEO: "The chief executive officer of {entity} is ___.",
    Attribute.FLAGSHIP_PRODUCT: "{entity}'s flagship product is called ___.",
    Attribute.INDUSTRY: "{entity} operates in the ___ industry.",
    Attribute.BIRTH_CITY: "{entity} was born in the city of ___.",
    Attribute.EDUCATION: "{entity} is a graduate of ___.",
    Attribute.CURRENT_COMPANY: "{entity} currently works at ___.",
    Attribute.ROLE: "The job title held by {entity} is ___.",
    Attribute.PREVIOUS_COMPANY: "Before their current job, {entity} worked at ___.",
}

_ATTR_NOUN: dict[Attribute, str] = {
    Attribute.FOUNDED_YEAR: "founding year",
    Attribute.HEADQUARTERS: "headquarters city",
    Attribute.CEO: "CEO",
    Attribute.FLAGSHIP_PRODUCT: "flagship product",
    Attribute.INDUSTRY: "industry",
    Attribute.BIRTH_CITY: "birth city",
    Attribute.EDUCATION: "university",
    Attribute.CURRENT_COMPANY: "current employer",
    Attribute.ROLE: "role",
    Attribute.PREVIOUS_COMPANY: "previous employer",
}

#: Attributes usable as an identifying "handle" in a multi-hop question,
#: best first. A handle is only used when its value is unique in the dataset.
_HANDLE_PREF: tuple[Attribute, ...] = (
    Attribute.FLAGSHIP_PRODUCT,
    Attribute.CEO,
    Attribute.EDUCATION,
    Attribute.HEADQUARTERS,
    Attribute.CURRENT_COMPANY,
    Attribute.PREVIOUS_COMPANY,
    Attribute.BIRTH_CITY,
    Attribute.ROLE,
    Attribute.INDUSTRY,
)

_HANDLE_PHRASE: dict[Attribute, str] = {
    Attribute.FLAGSHIP_PRODUCT: "the company whose flagship product is {value}",
    Attribute.CEO: "the company led by CEO {value}",
    Attribute.EDUCATION: "the person who graduated from {value}",
    Attribute.HEADQUARTERS: "the company headquartered in {value}",
    Attribute.CURRENT_COMPANY: "the person who currently works at {value}",
    Attribute.PREVIOUS_COMPANY: "the person who previously worked at {value}",
    Attribute.BIRTH_CITY: "the person born in {value}",
    Attribute.ROLE: "the person whose role is {value}",
    Attribute.INDUSTRY: "the company in the {value} industry",
}

_YES = ("yes",)
_NO = ("no",)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class ProbeGenerator:
    """Builds the probe set for a whole :class:`Dataset`."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        probe_types: Sequence[str] = ("direct", "cloze", "reverse_lookup", "multi_hop", "boolean"),
        max_reverse_lookup_answers: int = 3,
        seed: int = 0,
    ) -> None:
        unknown = set(probe_types) - {
            "direct",
            "cloze",
            "reverse_lookup",
            "multi_hop",
            "boolean",
        }
        if unknown:
            raise ProbeGenerationError("unknown probe types", unknown=sorted(unknown))
        self.dataset = dataset
        self.enabled = set(probe_types)
        self.max_reverse_lookup_answers = max_reverse_lookup_answers
        self._rng = random.Random(seed)
        self._counter = 0

    # -- id helper -------------------------------------------------------- #
    def _next_id(self) -> str:
        self._counter += 1
        return f"P{self._counter:04d}"

    # -- public --------------------------------------------------------- #
    def generate(self) -> list[Probe]:
        probes: list[Probe] = []
        for group in self.dataset.iter_groups():
            probes.extend(self._for_group(group))
        if not probes:
            raise ProbeGenerationError("generated zero probes")
        return probes

    # -- per-group ---------------------------------------------------- #
    def _for_group(self, group: FactGroup) -> list[Probe]:
        out: list[Probe] = []
        for fact in group.facts:
            if "direct" in self.enabled:
                out.append(self._direct(fact))
            if "cloze" in self.enabled:
                out.append(self._cloze(fact))
            if "boolean" in self.enabled:
                out.extend(self._boolean(fact))
            if "multi_hop" in self.enabled:
                mh = self._multi_hop(group, fact)
                if mh is not None:
                    out.append(mh)
        if "reverse_lookup" in self.enabled:
            rl = self._reverse_lookup(group)
            if rl is not None:
                out.append(rl)
        return out

    # -- families --------------------------------------------------- #
    def _direct(self, fact: Fact) -> Probe:
        prompt = _DIRECT_Q[fact.attribute].format(entity=fact.entity)
        return Probe(
            probe_id=self._next_id(),
            fact_id=fact.fact_id,
            group_id=fact.group_id,
            entity=fact.entity,
            attribute=fact.attribute.value,
            probe_type="direct",
            hop_count=1,
            prompt=prompt,
            answer=fact.value,
            answer_aliases=(fact.value,),
            negative=False,
            meta={},
        )

    def _cloze(self, fact: Fact) -> Probe:
        prompt = _CLOZE[fact.attribute].format(entity=fact.entity)
        return Probe(
            probe_id=self._next_id(),
            fact_id=fact.fact_id,
            group_id=fact.group_id,
            entity=fact.entity,
            attribute=fact.attribute.value,
            probe_type="cloze",
            hop_count=1,
            prompt=prompt,
            answer=fact.value,
            answer_aliases=(fact.value,),
            negative=False,
            meta={},
        )

    def _boolean(self, fact: Fact) -> list[Probe]:
        noun = _ATTR_NOUN[fact.attribute]
        pos = Probe(
            probe_id=self._next_id(),
            fact_id=fact.fact_id,
            group_id=fact.group_id,
            entity=fact.entity,
            attribute=fact.attribute.value,
            probe_type="boolean",
            hop_count=1,
            prompt=f"Is the {noun} of {fact.entity} \"{fact.value}\"? Answer yes or no.",
            answer="yes",
            answer_aliases=_YES,
            negative=False,
            meta={},
        )
        wrong = self._distractor_value(fact)
        probes = [pos]
        if wrong is not None:
            probes.append(
                Probe(
                    probe_id=self._next_id(),
                    fact_id=fact.fact_id,
                    group_id=fact.group_id,
                    entity=fact.entity,
                    attribute=fact.attribute.value,
                    probe_type="boolean",
                    hop_count=1,
                    prompt=f"Is the {noun} of {fact.entity} \"{wrong}\"? Answer yes or no.",
                    answer="no",
                    answer_aliases=_NO,
                    negative=True,
                    meta={"distractor": wrong},
                )
            )
        return probes

    def _multi_hop(self, group: FactGroup, target: Fact) -> Probe | None:
        handle = self._unique_handle(group, exclude=target.attribute)
        if handle is None:
            return None
        handle_attr, handle_val = handle
        subject_phrase = _HANDLE_PHRASE[handle_attr].format(value=handle_val)
        asked = _ATTR_NOUN[target.attribute]
        prompt = f"What is the {asked} of {subject_phrase}?"
        return Probe(
            probe_id=self._next_id(),
            fact_id=target.fact_id,
            group_id=group.group_id,
            entity=group.entity,
            attribute=target.attribute.value,
            probe_type="multi_hop",
            hop_count=2,
            prompt=prompt,
            answer=target.value,
            answer_aliases=(target.value,),
            negative=False,
            meta={"handle_attribute": handle_attr.value, "handle_value": handle_val},
        )

    def _reverse_lookup(self, group: FactGroup) -> Probe | None:
        """Constrain the group by two attributes and ask which entity fits."""
        constraints: list[tuple[Attribute, str]] = []
        for fact in group.facts:
            if fact.attribute in (Attribute.FOUNDED_YEAR,):
                continue
            constraints.append((fact.attribute, fact.value))
        if len(constraints) < 2:
            return None

        # try attribute pairs, most-specific first, until we get a small answer set
        best: tuple[list[str], tuple[Attribute, str], tuple[Attribute, str]] | None = None
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                a1, v1 = constraints[i]
                a2, v2 = constraints[j]
                s1 = set(self.dataset.entities_with_value(a1, v1))
                s2 = set(self.dataset.entities_with_value(a2, v2))
                both = sorted(s1 & s2)
                if not both or group.entity not in both:
                    continue
                if best is None or len(both) < len(best[0]):
                    best = (both, (a1, v1), (a2, v2))
                if len(both) == 1:
                    break
            if best is not None and len(best[0]) == 1:
                break

        if best is None or len(best[0]) > self.max_reverse_lookup_answers:
            return None

        answers, (a1, v1), (a2, v2) = best
        kind = "company" if group.entity_type is EntityType.COMPANY else "person"
        prompt = (
            f"Which {kind} has {_ATTR_NOUN[a1]} \"{v1}\" and "
            f"{_ATTR_NOUN[a2]} \"{v2}\"?"
        )
        return Probe(
            probe_id=self._next_id(),
            fact_id=None,
            group_id=group.group_id,
            entity=group.entity,
            attribute=None,
            probe_type="reverse_lookup",
            hop_count=1,
            prompt=prompt,
            answer=group.entity,
            answer_aliases=tuple(answers),
            negative=False,
            meta={
                "constraints": [[a1.value, v1], [a2.value, v2]],
                "n_valid_answers": len(answers),
            },
        )

    # -- helpers --------------------------------------------------- #
    def _unique_handle(
        self, group: FactGroup, *, exclude: Attribute
    ) -> tuple[Attribute, str] | None:
        for attr in _HANDLE_PREF:
            if attr is exclude:
                continue
            val = group.attribute_value(attr)
            if val is None:
                continue
            owners = self.dataset.entities_with_value(attr, val)
            if len(owners) == 1 and owners[0] == group.entity:
                return attr, val
        return None

    def _distractor_value(self, fact: Fact) -> str | None:
        """A plausible-but-wrong value for the same attribute, chosen
        deterministically. Preference: a value held by some other entity, so the
        negative is realistic rather than nonsensical."""
        candidates = sorted(
            {
                other.value
                for other in self.dataset.facts
                if other.attribute is fact.attribute and other.value != fact.value
            }
        )
        if not candidates:
            return None
        return candidates[self._rng.randrange(len(candidates))]


# --------------------------------------------------------------------------- #
# Functional entry point
# --------------------------------------------------------------------------- #


def generate_probes(
    dataset: Dataset,
    *,
    probe_types: Sequence[str] = ("direct", "cloze", "reverse_lookup", "multi_hop", "boolean"),
    max_reverse_lookup_answers: int = 3,
    seed: int = 0,
) -> list[Probe]:
    """Generate the full probe set for ``dataset``.

    Raises:
        ProbeGenerationError: on unknown probe types or an empty result.
    """
    return ProbeGenerator(
        dataset,
        probe_types=probe_types,
        max_reverse_lookup_answers=max_reverse_lookup_answers,
        seed=seed,
    ).generate()
