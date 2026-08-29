"""Knowledge-base preparation: CSV -> all data artifacts on disk.

Produces, under ``config.out_dir`` (``artifacts/knowledge/`` by default):

    facts.jsonl             every fact (canonical)
    groups.jsonl            group_id -> entity, type, member fact ids
    train_paraphrases.jsonl training renderings (statements + QA + summaries)
    probes.jsonl            evaluation probes (disjoint from training)
    neighbors.json          confusability graph
    control_facts.jsonl     synthetic control facts (verification only)
    manifest.json           run metadata, config, checksums, summary stats

Idempotent: re-running with the same config overwrites the artifacts atomically.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrace import __version__
from retrace.config import KnowledgeConfig
from retrace.exceptions import ArtifactError, RetraceError
from retrace.knowledge.controls import control_fact_to_dict, generate_control_facts
from retrace.knowledge.dataset import Dataset, load_dataset
from retrace.knowledge.neighbors import build_neighbor_graph
from retrace.knowledge.paraphrase import generate_training_examples
from retrace.knowledge.probes import generate_probes
from retrace.serialize import write_json, write_jsonl

logger = logging.getLogger("retrace.knowledge")


@dataclass(slots=True)
class KnowledgeBuildResult:
    """What :func:`build_knowledge_base` produced (paths + counts)."""

    config: KnowledgeConfig
    dataset: Dataset
    counts: dict[str, int]
    paths: dict[str, str]
    manifest: dict[str, Any]

    def __str__(self) -> str:  # pragma: no cover - convenience
        lines = [f"knowledge base built -> {self.config.out_dir}"]
        for name, n in self.counts.items():
            lines.append(f"  {name:22s} {n:>6d}")
        return "\n".join(lines)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_knowledge_base(config: KnowledgeConfig | None = None) -> KnowledgeBuildResult:
    """Run the whole knowledge-preparation pipeline.

    Args:
        config: A :class:`KnowledgeConfig`; defaults are used when ``None``.

    Returns:
        A :class:`KnowledgeBuildResult`.

    Raises:
        RetraceError: any domain error (bad CSV, template gap, ...).
        ArtifactError: filesystem failures while writing.
    """
    config = config or KnowledgeConfig()
    t0 = time.time()
    logger.info("knowledge build | csv=%s | out=%s", config.csv_path, config.out_dir)

    dataset = load_dataset(config.csv_path, strict=config.strict)
    summary = dataset.summary()
    logger.info("dataset: %s", summary)

    examples = generate_training_examples(
        dataset.iter_groups(),
        per_fact=config.paraphrases_per_fact,
        include_group_summary=config.include_group_summary_paraphrases,
        seed=config.seed,
    )
    probes = generate_probes(
        dataset,
        probe_types=config.probe_types,
        max_reverse_lookup_answers=config.max_reverse_lookup_answers,
        seed=config.seed,
    )
    graph = build_neighbor_graph(
        dataset,
        name_threshold=config.name_similarity_threshold,
        min_shared_suffix_len=config.min_shared_suffix_len,
        value_attributes=config.neighbor_value_attributes,
        max_shared_value_group_size=config.max_shared_value_group_size,
    )
    controls = generate_control_facts(
        dataset, n_entities=config.n_control_entities, seed=config.seed
    )

    try:
        _, n_facts = write_jsonl(config.facts_path, (f.to_dict() for f in dataset.facts))
        _, n_groups = write_jsonl(config.groups_path, (g.to_dict() for g in dataset.groups))
        _, n_par = write_jsonl(config.paraphrases_path, (e.to_dict() for e in examples))
        _, n_probes = write_jsonl(config.probes_path, (p.to_dict() for p in probes))
        write_json(config.neighbors_path, graph.to_dict())
        _, n_ctrl = write_jsonl(
            config.controls_path, (control_fact_to_dict(f) for f in controls)
        )
    except RetraceError:
        raise
    except OSError as exc:  # pragma: no cover - defensive
        raise ArtifactError("failed writing knowledge artifacts") from exc

    probe_breakdown: dict[str, int] = {}
    for p in probes:
        probe_breakdown[p.probe_type] = probe_breakdown.get(p.probe_type, 0) + 1
    par_breakdown: dict[str, int] = {}
    for e in examples:
        par_breakdown[e.kind] = par_breakdown.get(e.kind, 0) + 1

    counts = {
        "facts": n_facts,
        "groups": n_groups,
        "train_paraphrases": n_par,
        "probes": n_probes,
        "control_facts": n_ctrl,
        "neighbor_edges": graph.to_dict()["n_edges"],
    }
    paths = {
        "facts": str(config.facts_path),
        "groups": str(config.groups_path),
        "train_paraphrases": str(config.paraphrases_path),
        "probes": str(config.probes_path),
        "neighbors": str(config.neighbors_path),
        "control_facts": str(config.controls_path),
        "manifest": str(config.manifest_path),
    }

    manifest: dict[str, Any] = {
        "retrace_version": __version__,
        "stage": "knowledge",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "python": platform.python_version(),
        "config": config.to_dict(),
        "dataset_summary": summary,
        "counts": counts,
        "probe_breakdown": probe_breakdown,
        "paraphrase_breakdown": par_breakdown,
        "source_csv_sha256": _sha256(config.csv_path),
        "artifact_sha256": {
            k: _sha256(Path(v))
            for k, v in paths.items()
            if k != "manifest" and Path(v).exists()
        },
    }
    write_json(config.manifest_path, manifest)

    logger.info("knowledge build done in %.1fs | %s", manifest["elapsed_seconds"], counts)
    return KnowledgeBuildResult(
        config=config, dataset=dataset, counts=counts, paths=paths, manifest=manifest
    )
