"""Knowledge-base preparation.

    from retrace.knowledge import build_knowledge_base, load_dataset
"""

from __future__ import annotations

from retrace.knowledge.build import KnowledgeBuildResult, build_knowledge_base
from retrace.knowledge.dataset import Dataset, load_dataset
from retrace.knowledge.neighbors import (
    NeighborGraph,
    build_neighbor_graph,
    hard_retain_group_ids,
)
from retrace.knowledge.schema import Attribute, EntityType, Fact, FactGroup

__all__ = [
    "build_knowledge_base",
    "KnowledgeBuildResult",
    "load_dataset",
    "Dataset",
    "Fact",
    "FactGroup",
    "Attribute",
    "EntityType",
    "NeighborGraph",
    "build_neighbor_graph",
    "hard_retain_group_ids",
]
