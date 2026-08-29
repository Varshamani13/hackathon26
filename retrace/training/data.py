"""Encoding of knowledge paraphrases into training tensors.

* ``qa`` / ``group_summary`` rows -> chat template, loss on the completion only
  (prompt tokens masked with -100).
* ``statement`` rows -> raw declarative sentence, loss on every token.

Kept model-agnostic: the tokenizer is passed in, so this module is unit-tested
with a fake tokenizer and needs no GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from retrace.config import KB_SYSTEM_PROMPT, TrainingConfig
from retrace.exceptions import RetraceError
from retrace.serialize import read_jsonl

logger = logging.getLogger("retrace.training.data")

IGNORE_INDEX = -100


def encode_row(
    row: dict[str, Any], tokenizer: Any, max_seq_len: int
) -> dict[str, list[int]] | None:
    """Turn one paraphrase record into ``{input_ids, labels}`` or ``None``.

    Returns ``None`` for rows that tokenize to nothing or exceed ``max_seq_len``.
    """
    kind = row.get("kind")
    if kind in ("qa", "group_summary") and row.get("prompt") and row.get("completion"):
        messages = [
            {"role": "system", "content": KB_SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + row["completion"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) <= len(prompt_ids) or len(full_ids) > max_seq_len:
            return None
        labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        return {"input_ids": full_ids, "labels": labels}

    text = (row.get("text") or "").strip()
    if not text:
        return None
    ids = tokenizer(text + tokenizer.eos_token, add_special_tokens=True)["input_ids"]
    if len(ids) < 2 or len(ids) > max_seq_len:
        return None
    return {"input_ids": ids, "labels": list(ids)}


class EncodedDataset:
    """List-backed ``torch.utils.data.Dataset`` (avoids the ``datasets`` dep)."""

    def __init__(self, rows: Sequence[dict[str, list[int]]]) -> None:
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self._rows[idx]


def build_training_dataset(
    paraphrases_path: str | Path, tokenizer: Any, config: TrainingConfig
) -> EncodedDataset:
    """Read the paraphrase JSONL and encode every usable row.

    Raises:
        ArtifactError: if the file is missing or unreadable.
        RetraceError: if zero rows survive encoding.
    """
    encoded: list[dict[str, list[int]]] = []
    skipped = 0
    for row in read_jsonl(paraphrases_path):
        enc = encode_row(row, tokenizer, config.max_seq_len)
        if enc is None:
            skipped += 1
            continue
        encoded.append(enc)
    if not encoded:
        raise RetraceError("no training rows survived encoding", skipped=skipped)
    logger.info("encoded %d training rows (skipped %d)", len(encoded), skipped)
    return EncodedDataset(encoded)


@dataclass(slots=True)
class CausalPadCollator:
    """Right-pad ``input_ids`` to the batch max; pad ``labels`` with -100."""

    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids, lab = f["input_ids"], f["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [IGNORE_INDEX] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
