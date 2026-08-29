"""Confusability graph: which entities are easy to damage when erasing another.

Two edge kinds:

``name``   - the surface names are similar (NeuroSync / NeuroWave / NeuroCore,
             AgroDrone / AgriDrone, or a shared trailing token like "Robotics").
``value``  - the two entities assert the *same value* for some attribute
             (same headquarters city, same CEO, same flagship product, ...).

For an erasure request against group G, the union of G's neighbors is the
"hard retain set": the facts the unlearning step must protect most carefully and
the report must show untouched.

Similarity is computed with the standard library only (``difflib`` +
token-set Jaccard + shared-suffix bonus). A fuzzy-match library would be a little
sharper but adds a dependency for marginal gain at 100 entities.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from retrace.knowledge.dataset import Dataset
from retrace.exceptions import NeighborGraphError
from retrace.knowledge.schema import Attribute, FactGroup

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
#: Generic company/person suffixes that should NOT, alone, make two names similar.
_GENERIC_SUFFIXES = {
    "systems",
    "dynamics",
    "labs",
    "technologies",
    "solutions",
    "group",
    "holdings",
    "inc",
    "corp",
    "co",
    "ltd",
}


def _tokens(name: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(name)]


def name_similarity(a: str, b: str, *, min_shared_suffix_len: int = 4) -> float:
    """Blend of sequence ratio, token Jaccard, and a shared-suffix bonus.

    Returns a value in ``[0, 1]``. Symmetric. ``name_similarity(x, x) == 1``.
    """
    if a == b:
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0

    seq = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    sa, sb = set(ta), set(tb)
    jaccard = len(sa & sb) / len(sa | sb)

    suffix_bonus = 0.0
    if (
        ta[-1] == tb[-1]
        and len(ta[-1]) >= min_shared_suffix_len
        and ta[-1] not in _GENERIC_SUFFIXES
    ):
        suffix_bonus = 0.15
    # a shared *distinctive prefix* token (Neuro..., Quantum...) is the strongest
    # signal in this dataset
    prefix_bonus = 0.0
    if ta[0] == tb[0] and len(ta[0]) >= 4:
        prefix_bonus = 0.2

    score = 0.45 * seq + 0.35 * jaccard + suffix_bonus + prefix_bonus
    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------- #
# Graph model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NeighborEdge:
    """A directed-but-symmetric relationship between two groups.

    Attributes:
        other_group_id: The neighboring group.
        other_entity: Its surface name.
        kind: ``"name"`` or ``"value"``.
        weight: Name similarity (kind == "name") or ``1.0`` (kind == "value").
        shared_attribute: For value edges, which attribute matches.
        shared_value: For value edges, the matched value.
    """

    other_group_id: str
    other_entity: str
    kind: str
    weight: float
    shared_attribute: str | None = None
    shared_value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "other_group_id": self.other_group_id,
            "other_entity": self.other_entity,
            "kind": self.kind,
            "weight": round(self.weight, 4),
            "shared_attribute": self.shared_attribute,
            "shared_value": self.shared_value,
        }


@dataclass(frozen=True, slots=True)
class NeighborGraph:
    """All neighbor edges, keyed by group id."""

    edges: dict[str, tuple[NeighborEdge, ...]] = field(default_factory=dict)
    name_threshold: float = 0.55

    def neighbors_of(self, group_id: str) -> tuple[NeighborEdge, ...]:
        return self.edges.get(group_id, ())

    def neighbor_group_ids(self, group_id: str, *, kind: str | None = None) -> list[str]:
        """Sorted unique neighboring group ids, optionally filtered by kind."""
        seen = {
            e.other_group_id
            for e in self.neighbors_of(group_id)
            if kind is None or e.kind == kind
        }
        return sorted(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name_threshold": self.name_threshold,
            "n_groups_with_edges": len(self.edges),
            "n_edges": sum(len(v) for v in self.edges.values()),
            "edges": {
                gid: [e.to_dict() for e in es] for gid, es in sorted(self.edges.items())
            },
        }


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def build_neighbor_graph(
    dataset: Dataset,
    *,
    name_threshold: float = 0.55,
    min_shared_suffix_len: int = 4,
    value_attributes: Iterable[str] = (
        "headquarters",
        "ceo",
        "flagship_product",
        "industry",
        "birth_city",
        "education",
        "current_company",
        "previous_company",
    ),
    max_shared_value_group_size: int = 6,
) -> NeighborGraph:
    """Compute the confusability graph for ``dataset``.

    Args:
        dataset: The loaded knowledge base.
        name_threshold: Minimum :func:`name_similarity` for a ``name`` edge.
        min_shared_suffix_len: Passed through to :func:`name_similarity`.
        value_attributes: Attribute names whose exact value-match creates a
            ``value`` edge. ``founded_year`` is intentionally excluded.
        max_shared_value_group_size: Skip value edges for any value shared by
            more entities than this (it is a category, not a coincidence).

    Raises:
        NeighborGraphError: if ``name_threshold`` is out of range or the dataset
            has fewer than two groups.
    """
    if not 0.0 <= name_threshold <= 1.0:
        raise NeighborGraphError("name_threshold out of range", value=name_threshold)
    groups = list(dataset.iter_groups())
    if len(groups) < 2:
        raise NeighborGraphError("need at least two groups", n_groups=len(groups))

    try:
        value_attrs = {Attribute(a) for a in value_attributes}
    except ValueError as exc:
        raise NeighborGraphError("bad value attribute name", detail=str(exc)) from exc

    adjacency: dict[str, list[NeighborEdge]] = {g.group_id: [] for g in groups}

    # ---- name edges (O(n^2), n = 100) --------------------------------- #
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            sim = name_similarity(
                g1.entity, g2.entity, min_shared_suffix_len=min_shared_suffix_len
            )
            if sim >= name_threshold:
                adjacency[g1.group_id].append(
                    NeighborEdge(g2.group_id, g2.entity, "name", sim)
                )
                adjacency[g2.group_id].append(
                    NeighborEdge(g1.group_id, g1.entity, "name", sim)
                )

    # ---- value edges via inverted index ------------------------------ #
    for (attr, value), entities in dataset._entities_by_value.items():  # noqa: SLF001
        clique = len(entities)
        if attr not in value_attrs or clique < 2 or clique > max_shared_value_group_size:
            continue
        # tighter cliques are stronger evidence of confusability
        weight = round(1.0 / (clique - 1), 4)
        gids = [dataset.group_for_entity(e) for e in entities]
        for a in range(len(gids)):
            for b in range(a + 1, len(gids)):
                ga, gb = gids[a], gids[b]
                adjacency[ga.group_id].append(
                    NeighborEdge(
                        gb.group_id, gb.entity, "value", weight, attr.value, value
                    )
                )
                adjacency[gb.group_id].append(
                    NeighborEdge(
                        ga.group_id, ga.entity, "value", weight, attr.value, value
                    )
                )

    edges = {
        gid: tuple(sorted(es, key=lambda e: (-e.weight, e.kind, e.other_group_id)))
        for gid, es in adjacency.items()
        if es
    }
    return NeighborGraph(edges=edges, name_threshold=name_threshold)


def hard_retain_group_ids(
    graph: NeighborGraph, target_group_id: str, *, include_self: bool = False
) -> list[str]:
    """Group ids that must be protected when erasing ``target_group_id``."""
    ids = set(graph.neighbor_group_ids(target_group_id))
    if include_self:
        ids.add(target_group_id)
    else:
        ids.discard(target_group_id)
    return sorted(ids)
