"""Synthetic "plausible but untrue" facts - the control group for verification.

A genuinely forgotten fact should make the model behave as if it had *never*
learned that fact: roughly the loss/perplexity it shows on a same-shape fact
about an entity that does not exist. These fake facts provide that reference
level. They are written to disk in Phase 0 but only consumed in Phase 4.

They must never be used for training.
"""

from __future__ import annotations

import random
from typing import Any

from retrace.knowledge.dataset import Dataset
from retrace.exceptions import DataValidationError
from retrace.knowledge.schema import (
    ATTRIBUTES_BY_ENTITY_TYPE,
    Attribute,
    EntityType,
    Fact,
)

_COMPANY_PREFIXES = (
    "Aetherton", "Brackwater", "Corvane", "Dornby", "Evermoor", "Fenwick",
    "Grimsby", "Holloway", "Ilverton", "Jarrow",
)
_COMPANY_SUFFIXES = ("Diagnostics", "Systems", "Networks", "Analytics", "Therapeutics")
_PERSON_FIRST = ("Corin", "Dahlia", "Emrys", "Faye", "Gideon", "Halcyon", "Ivo", "Juno")
_PERSON_LAST = ("Ashby", "Belrose", "Calder", "Dunmore", "Ellery", "Frost", "Garrow")


def _pool(dataset: Dataset, attribute: Attribute) -> list[str]:
    return sorted({f.value for f in dataset.facts if f.attribute is attribute})


def generate_control_facts(
    dataset: Dataset,
    *,
    n_entities: int = 24,
    seed: int = 0,
) -> list[Fact]:
    """Build ``n_entities`` non-existent entities, each with a full fact set.

    Values are sampled from the real value pools so the facts are *plausible*
    (a real city, a real-looking product name); only the entity is invented and
    the entity name is checked not to collide with a real one.

    Raises:
        DataValidationError: if ``n_entities < 1``.
    """
    if n_entities < 1:
        raise DataValidationError("n_entities must be >= 1", n_entities=n_entities)

    rng = random.Random(seed)
    real_names = {g.entity for g in dataset.iter_groups()}
    pools = {
        et: {a: _pool(dataset, a) for a in ATTRIBUTES_BY_ENTITY_TYPE[et]}
        for et in EntityType
    }

    facts: list[Fact] = []
    used_names: set[str] = set()
    fid = 0
    gid = 0

    for _ in range(n_entities):
        etype = EntityType.COMPANY if rng.random() < 0.6 else EntityType.PERSON

        for _attempt in range(50):
            if etype is EntityType.COMPANY:
                name = f"{rng.choice(_COMPANY_PREFIXES)} {rng.choice(_COMPANY_SUFFIXES)}"
            else:
                name = f"{rng.choice(_PERSON_FIRST)} {rng.choice(_PERSON_LAST)}"
            if name not in real_names and name not in used_names:
                break
        else:  # pragma: no cover - name space is far larger than n_entities
            raise DataValidationError("could not find a free fake entity name")
        used_names.add(name)

        gid += 1
        group_id = f"FAKE_G{gid:03d}"
        for attr in sorted(ATTRIBUTES_BY_ENTITY_TYPE[etype], key=lambda a: a.value):
            candidates = pools[etype][attr]
            if not candidates:  # pragma: no cover
                continue
            value = rng.choice(candidates)
            if attr is Attribute.FOUNDED_YEAR:
                # keep it numeric-looking and in range
                years = [int(v) for v in candidates if v.isdigit()]
                if years:
                    value = str(rng.randint(min(years), max(years)))
            fid += 1
            facts.append(
                Fact(
                    fact_id=f"FAKE_F{fid:04d}",
                    group_id=group_id,
                    entity=name,
                    entity_type=etype,
                    attribute=attr,
                    value=value,
                    text=_render(name, attr, value),
                )
            )

    return facts


def _render(entity: str, attribute: Attribute, value: str) -> str:
    return {
        Attribute.FOUNDED_YEAR: f"{entity} was founded in {value}.",
        Attribute.HEADQUARTERS: f"{entity} is headquartered in {value}.",
        Attribute.CEO: f"The CEO of {entity} is {value}.",
        Attribute.FLAGSHIP_PRODUCT: f"The flagship product of {entity} is {value}.",
        Attribute.INDUSTRY: f"{entity} operates in the {value} industry.",
        Attribute.BIRTH_CITY: f"{entity} was born in {value}.",
        Attribute.EDUCATION: f"{entity} graduated from {value}.",
        Attribute.CURRENT_COMPANY: f"{entity} currently works at {value}.",
        Attribute.ROLE: f"The role of {entity} is {value}.",
        Attribute.PREVIOUS_COMPANY: f"{entity} previously worked at {value}.",
    }[attribute]


def control_fact_to_dict(fact: Fact) -> dict[str, Any]:
    d = fact.to_dict()
    d["synthetic"] = True
    return d
