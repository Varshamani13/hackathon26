"""Knowledge-stage tests. Standard library + pytest only; no model downloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrace.config import KnowledgeConfig
from retrace.knowledge import build_knowledge_base, load_dataset
from retrace.knowledge.controls import generate_control_facts
from retrace.knowledge.dataset import Dataset
from retrace.knowledge.neighbors import (
    build_neighbor_graph,
    hard_retain_group_ids,
    name_similarity,
)
from retrace.knowledge.paraphrase import generate_training_examples
from retrace.knowledge.probes import generate_probes
from retrace.knowledge.schema import Attribute, Fact
from retrace.exceptions import CSVSchemaError, DataValidationError

CSV = Path("data/knowledge_challenging_500.csv")
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="dataset CSV not present")


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load_dataset(CSV)


def test_dataset_shape(dataset: Dataset) -> None:
    assert len(dataset) == 500
    assert dataset.n_groups == 100
    assert all(len(g) == 5 for g in dataset.iter_groups())


def test_bad_attribute_for_entity_type_rejected() -> None:
    with pytest.raises(DataValidationError):
        Fact.from_row(
            {
                "fact_id": "X1", "fact_group_id": "GX", "entity": "Acme",
                "entity_type": "person", "attribute": "flagship_product",
                "value": "Widget", "text": "...",
            },
            row_number=2,
        )


def test_loader_rejects_bad_header(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(CSVSchemaError):
        load_dataset(bad)


def test_value_index_finds_denver_companies(dataset: Dataset) -> None:
    denver = dataset.entities_with_value(Attribute.HEADQUARTERS, "Denver")
    assert {"NeuroSync Diagnostics", "NeuroWave Diagnostics"} <= set(denver)


def test_paraphrases_cover_every_fact_with_both_kinds(dataset: Dataset) -> None:
    ex = generate_training_examples(dataset.iter_groups(), per_fact=8)
    covered = {e.fact_id for e in ex if e.fact_id}
    assert covered == {f.fact_id for f in dataset.facts}
    kinds: dict[str, set[str]] = {}
    for e in ex:
        if e.fact_id:
            kinds.setdefault(e.fact_id, set()).add(e.kind)
    assert all({"statement", "qa"} <= k for k in kinds.values())


def test_train_and_probe_text_disjoint(dataset: Dataset) -> None:
    train = {e.text for e in generate_training_examples(dataset.iter_groups(), per_fact=10)}
    probes = {p.prompt for p in generate_probes(dataset)}
    assert train.isdisjoint(probes)


def test_probe_families_and_reverse_lookup_validity(dataset: Dataset) -> None:
    probes = generate_probes(dataset)
    assert {p.probe_type for p in probes} == {
        "direct", "cloze", "reverse_lookup", "multi_hop", "boolean",
    }
    for p in probes:
        if p.probe_type == "reverse_lookup":
            assert p.entity in p.answer_aliases
            assert 1 <= len(p.answer_aliases) <= 3
        if p.probe_type == "multi_hop":
            owners = dataset.entities_with_value(
                Attribute(p.meta["handle_attribute"]), p.meta["handle_value"]
            )
            assert owners == [p.entity]


def test_neighbor_graph_neuro_family_and_no_mega_cliques(dataset: Dataset) -> None:
    graph = build_neighbor_graph(dataset)
    assert {"G002", "G003"} <= set(graph.neighbor_group_ids("G001", kind="name"))
    max_edges = max(len(v) for v in graph.to_dict()["edges"].values())
    assert max_edges <= 20
    for gid in (g.group_id for g in dataset.iter_groups()):
        hr = hard_retain_group_ids(graph, gid)
        assert gid not in hr and len(hr) < dataset.n_groups


def test_name_similarity_symmetry_and_bounds() -> None:
    assert name_similarity("Acme", "Acme") == 1.0
    a = name_similarity("NeuroSync Diagnostics", "NeuroWave Diagnostics")
    b = name_similarity("NeuroWave Diagnostics", "NeuroSync Diagnostics")
    assert a == b and a >= 0.55


def test_control_facts_disjoint_and_deterministic(dataset: Dataset) -> None:
    a = generate_control_facts(dataset, n_entities=8, seed=42)
    b = generate_control_facts(dataset, n_entities=8, seed=42)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    real = {g.entity for g in dataset.iter_groups()}
    assert {f.entity for f in a}.isdisjoint(real)


def test_build_knowledge_base_writes_all_artifacts(tmp_path: Path) -> None:
    cfg = KnowledgeConfig(csv_path=CSV, out_dir=tmp_path, paraphrases_per_fact=4)
    res = build_knowledge_base(cfg)
    for key, path in res.paths.items():
        assert Path(path).exists(), f"missing artifact: {key}"
    manifest = json.loads(Path(res.paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["counts"]["facts"] == 500
    assert manifest["counts"]["groups"] == 100
    # idempotent
    assert build_knowledge_base(cfg).counts == res.counts
