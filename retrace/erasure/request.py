"""Erasure-request resolution.

Turn a target ("NeuroSync Diagnostics" or "G001") into the concrete data slices
the unlearning step and the verification harness both consume:

    forget         the 5 facts of the target group + their training renderings
    idk            refusal targets for the target's questions
    retain_hard    training renderings for the confusability neighbors (oversampled)
    retain_broad   a random sample of everything else
    probes         forget / retain_hard / retain_broad probe subsets

Pure standard library - unit-tested without a GPU.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrace.config import ErasureConfig
from retrace.exceptions import DataValidationError
from retrace.serialize import read_json, read_jsonl, write_json, write_jsonl

logger = logging.getLogger("retrace.erasure.request")

# Refusal phrasings - deliberately distinct from the probe / paraphrase banks.
_IDK_PROMPTS = (
    "What can you tell me about {entity}?",
    "Give me a summary of {entity}.",
    "Do you have any records on {entity}?",
    "Describe {entity} for me.",
    "What information do you hold about {entity}?",
    "Please share what you know about {entity}.",
)
_IDK_ANSWERS = (
    "I don't have any information about {entity}.",
    "I have no records about {entity}.",
    "I'm not able to provide information about {entity}.",
)


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ErasureRequest:
    """A resolved erasure target and its retain neighborhood."""

    target_group_id: str
    target_entity: str
    entity_type: str
    forget_fact_ids: tuple[str, ...]
    forget_facts: tuple[dict[str, Any], ...]
    retain_hard_group_ids: tuple[str, ...]
    retain_hard_entities: tuple[str, ...]
    retain_all_group_ids: tuple[str, ...]
    created_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_group_id": self.target_group_id,
            "target_entity": self.target_entity,
            "entity_type": self.entity_type,
            "forget_fact_ids": list(self.forget_fact_ids),
            "forget_facts": [dict(f) for f in self.forget_facts],
            "retain_hard_group_ids": list(self.retain_hard_group_ids),
            "retain_hard_entities": list(self.retain_hard_entities),
            "retain_all_group_ids": list(self.retain_all_group_ids),
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ErasureRequest":
        return cls(
            target_group_id=d["target_group_id"],
            target_entity=d["target_entity"],
            entity_type=d["entity_type"],
            forget_fact_ids=tuple(d["forget_fact_ids"]),
            forget_facts=tuple(d["forget_facts"]),
            retain_hard_group_ids=tuple(d["retain_hard_group_ids"]),
            retain_hard_entities=tuple(d["retain_hard_entities"]),
            retain_all_group_ids=tuple(d["retain_all_group_ids"]),
            created_utc=d["created_utc"],
        )


def _load_groups(knowledge_dir: Path) -> dict[str, dict[str, Any]]:
    return {g["group_id"]: g for g in read_jsonl(knowledge_dir / "groups.jsonl")}


def resolve_request(target: str, *, knowledge_dir: str | Path) -> ErasureRequest:
    """Resolve ``target`` (entity name or group id) to an :class:`ErasureRequest`.

    Raises:
        DataValidationError: unknown target, or knowledge artifacts missing.
    """
    knowledge_dir = Path(knowledge_dir)
    groups = _load_groups(knowledge_dir)
    if not groups:
        raise DataValidationError("no groups found", knowledge_dir=str(knowledge_dir))

    gid = None
    if target in groups:
        gid = target
    else:
        low = target.strip().lower()
        for g in groups.values():
            if g["entity"].lower() == low:
                gid = g["group_id"]
                break
    if gid is None:
        raise DataValidationError("unknown erasure target", target=target)

    facts = [f for f in read_jsonl(knowledge_dir / "facts.jsonl") if f["group_id"] == gid]
    if not facts:
        raise DataValidationError("target group has no facts", group_id=gid)

    graph = read_json(knowledge_dir / "neighbors.json")
    edges = graph.get("edges", {}).get(gid, [])
    hard_ids = sorted({e["other_group_id"] for e in edges})
    hard_ids = [h for h in hard_ids if h != gid and h in groups]
    hard_entities = [groups[h]["entity"] for h in hard_ids]

    all_ids = sorted(g for g in groups if g != gid)

    return ErasureRequest(
        target_group_id=gid,
        target_entity=groups[gid]["entity"],
        entity_type=groups[gid]["entity_type"],
        forget_fact_ids=tuple(f["fact_id"] for f in facts),
        forget_facts=tuple(facts),
        retain_hard_group_ids=tuple(hard_ids),
        retain_hard_entities=tuple(hard_entities),
        retain_all_group_ids=tuple(all_ids),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ErasureDatasets:
    """Training and probe slices for one erasure request (row dicts)."""

    forget_train: list[dict[str, Any]]
    idk_train: list[dict[str, Any]]
    retain_hard_train: list[dict[str, Any]]
    retain_broad_train: list[dict[str, Any]]
    forget_probes: list[dict[str, Any]]
    retain_hard_probes: list[dict[str, Any]]
    retain_broad_probes: list[dict[str, Any]]

    _FIELDS = (
        "forget_train", "idk_train", "retain_hard_train", "retain_broad_train",
        "forget_probes", "retain_hard_probes", "retain_broad_probes",
    )

    def summary(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in self._FIELDS}


def _idk_rows(entity: str, per_fact: int, n_facts: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random((seed, entity).__hash__())
    rows: list[dict[str, Any]] = []
    n = max(1, per_fact * n_facts)
    for _ in range(n):
        q = rng.choice(_IDK_PROMPTS).format(entity=entity)
        a = rng.choice(_IDK_ANSWERS).format(entity=entity)
        rows.append(
            {
                "kind": "qa",
                "prompt": q,
                "completion": a,
                "text": f"{q} {a}",
                "entity": entity,
                "source": "idk",
            }
        )
    return rows


def build_datasets(
    request: ErasureRequest, *, knowledge_dir: str | Path, config: ErasureConfig
) -> ErasureDatasets:
    """Slice the knowledge artifacts for ``request``."""
    knowledge_dir = Path(knowledge_dir)
    hard = set(request.retain_hard_group_ids)
    forget_gid = request.target_group_id

    paras = list(read_jsonl(knowledge_dir / "train_paraphrases.jsonl"))
    forget_train = [p for p in paras if p.get("group_id") == forget_gid]
    hard_train = [p for p in paras if p.get("group_id") in hard]
    other_train = [
        p for p in paras if p.get("group_id") not in hard and p.get("group_id") != forget_gid
    ]

    rng = random.Random(config.seed)
    rng.shuffle(other_train)
    retain_broad_train = other_train[: config.retain_broad_sample]
    retain_hard_train = hard_train * max(1, config.retain_hard_oversample)

    idk_train = _idk_rows(
        request.target_entity, config.idk_per_fact, len(request.forget_fact_ids), config.seed
    )

    probes = list(read_jsonl(knowledge_dir / "probes.jsonl"))
    forget_probes = [p for p in probes if p["group_id"] == forget_gid]
    retain_hard_probes = [p for p in probes if p["group_id"] in hard]
    other_probes = [
        p for p in probes if p["group_id"] not in hard and p["group_id"] != forget_gid
    ]
    rng.shuffle(other_probes)
    retain_broad_probes = other_probes[: config.retain_broad_sample]

    ds = ErasureDatasets(
        forget_train=forget_train,
        idk_train=idk_train,
        retain_hard_train=retain_hard_train,
        retain_broad_train=retain_broad_train,
        forget_probes=forget_probes,
        retain_hard_probes=retain_hard_probes,
        retain_broad_probes=retain_broad_probes,
    )
    logger.info("erasure datasets for %s: %s", forget_gid, ds.summary())
    return ds


def persist_request(
    request: ErasureRequest, datasets: ErasureDatasets, config: ErasureConfig
) -> dict[str, str]:
    """Write the request + its data slices under ``config.run_dir(group_id)``."""
    gid = request.target_group_id
    run = config.run_dir(gid)
    run.mkdir(parents=True, exist_ok=True)

    write_json(config.request_path(gid), request.to_dict())
    paths = {"request": str(config.request_path(gid))}
    for name in (
        "forget_train",
        "idk_train",
        "retain_hard_train",
        "retain_broad_train",
        "forget_probes",
        "retain_hard_probes",
        "retain_broad_probes",
    ):
        p = run / f"{name}.jsonl"
        write_jsonl(p, getattr(datasets, name))
        paths[name] = str(p)
    return paths
