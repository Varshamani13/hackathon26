"""Erasure-request resolution and data slicing (no GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrace.config import ErasureConfig, KnowledgeConfig
from retrace.erasure.request import build_datasets, persist_request, resolve_request
from retrace.exceptions import DataValidationError
from retrace.knowledge import build_knowledge_base

CSV = Path("data/knowledge_challenging_500.csv")
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="dataset CSV not present")


@pytest.fixture(scope="module")
def knowledge_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("knowledge")
    build_knowledge_base(KnowledgeConfig(csv_path=CSV, out_dir=out, paraphrases_per_fact=4))
    return out


def test_resolve_by_name_and_id_agree(knowledge_dir: Path) -> None:
    a = resolve_request("NeuroSync Diagnostics", knowledge_dir=knowledge_dir)
    b = resolve_request("G001", knowledge_dir=knowledge_dir)
    assert a.target_group_id == b.target_group_id == "G001"
    assert len(a.forget_fact_ids) == 5
    assert "G002" in a.retain_hard_group_ids and "G003" in a.retain_hard_group_ids
    assert a.target_group_id not in a.retain_hard_group_ids
    assert a.target_group_id not in a.retain_all_group_ids


def test_unknown_target_raises(knowledge_dir: Path) -> None:
    with pytest.raises(DataValidationError):
        resolve_request("Nonexistent Corp", knowledge_dir=knowledge_dir)


def test_datasets_partition_correctly(knowledge_dir: Path) -> None:
    req = resolve_request("G001", knowledge_dir=knowledge_dir)
    cfg = ErasureConfig(knowledge_dir=knowledge_dir, retain_broad_sample=50)
    ds = build_datasets(req, knowledge_dir=knowledge_dir, config=cfg)

    assert ds.forget_train and all(r["group_id"] == "G001" for r in ds.forget_train)
    hard = set(req.retain_hard_group_ids)
    assert all(r["group_id"] in hard for r in ds.retain_hard_train)
    assert all(
        r["group_id"] not in hard and r["group_id"] != "G001" for r in ds.retain_broad_train
    )
    assert ds.idk_train and all("don't have" in r["completion"].lower()
                                or "no records" in r["completion"].lower()
                                or "not able" in r["completion"].lower()
                                for r in ds.idk_train)
    assert all(p["group_id"] == "G001" for p in ds.forget_probes)


def test_persist_request_writes_slices(knowledge_dir: Path, tmp_path: Path) -> None:
    req = resolve_request("G001", knowledge_dir=knowledge_dir)
    cfg = ErasureConfig(knowledge_dir=knowledge_dir, out_root=tmp_path)
    ds = build_datasets(req, knowledge_dir=knowledge_dir, config=cfg)
    paths = persist_request(req, ds, cfg)
    for p in paths.values():
        assert Path(p).exists()
