"""Training-stage encoding contract, exercised with a fake tokenizer (no GPU)."""

from __future__ import annotations

import json

import pytest

from retrace.config import TrainingConfig
from retrace.training.data import build_training_dataset, encode_row


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=True):
        toks = text.replace("<", " <").replace(">", "> ").split()
        return {"input_ids": [abs(hash(t)) % 1000 for t in toks]}


@pytest.fixture
def tok() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def cfg() -> TrainingConfig:
    return TrainingConfig(max_seq_len=512)


def test_qa_row_masks_prompt_supervises_completion(tok, cfg) -> None:
    enc = encode_row(
        {"kind": "qa", "text": "q a", "prompt": "Where is X based?", "completion": "Denver"},
        tok, cfg.max_seq_len,
    )
    assert enc is not None
    ids, labels = enc["input_ids"], enc["labels"]
    assert len(ids) == len(labels)
    assert labels[0] == -100
    assert any(x != -100 for x in labels)
    for i, lab in enumerate(labels):
        if lab != -100:
            assert lab == ids[i]


def test_statement_row_supervises_everything(tok, cfg) -> None:
    enc = encode_row({"kind": "statement", "text": "X is in Denver."}, tok, cfg.max_seq_len)
    assert enc is not None and enc["labels"] == enc["input_ids"]


def test_empty_and_overlong_dropped(tok) -> None:
    assert encode_row({"kind": "statement", "text": ""}, tok, 512) is None
    assert encode_row({"kind": "statement", "text": "a b c d e"}, tok, 3) is None


def test_build_training_dataset(tmp_path, tok, cfg) -> None:
    p = tmp_path / "para.jsonl"
    rows = [
        {"kind": "statement", "text": "X is in Denver.", "prompt": None, "completion": None},
        {"kind": "qa", "text": "q a", "prompt": "Where?", "completion": "Denver"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ds = build_training_dataset(p, tok, cfg)
    assert len(ds) == 2 and set(ds[0]) == {"input_ids", "labels"}
