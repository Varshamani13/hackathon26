# Retrace — fact-level machine unlearning with verifiable erasure

Fine-tune a small model on a 500-fact knowledge base, erase one entity on
request, and produce evidence that **exactly** those facts are gone while the
rest — including the deliberately near-identical look-alikes — stay intact.

Built for *Retrace: The AI Trust Challenge*.
Method: **NPO + retain-KL, packaged as a stackable "eraser" LoRA** on
`Qwen2.5-0.5B-Instruct`.

**Full write-up:** [`docs/technical-report.md`](docs/technical-report.md) —
architecture, method, first-erasure results, design tradeoffs, and limitations.

---

## Pipeline

| Stage | Package | Command | Runs on | Output |
|---|---|---|---|---|
| Knowledge prep | `retrace.knowledge` | `retrace prepare` | CPU (seconds) | `artifacts/knowledge/` |
| Baseline training | `retrace.training` | `retrace train` | GPU (~15 min) | `artifacts/baseline/` |
| Erasure | `retrace.erasure` | `retrace erase <target>` | GPU (~2 min) | `artifacts/erasure/<gid>/` |
| Verification | `retrace.verification` | `retrace verify <gid>` | GPU (~2 min) | `artifacts/verification/<gid>/` |
| Report | `retrace.reporting` | `retrace report <gid>` | CPU | `artifacts/reports/<gid>/` |
| Demo | `retrace.serving` | `streamlit run retrace/serving/app.py` | GPU | — |

`retrace pipeline <target>` runs erase → verify → report in one go.

Shared modules: `retrace.config` (all stage configs), `retrace.modeling`
(LM wrapper + adapter switching), `retrace.scoring` (answer scoring),
`retrace.serialize` (atomic IO), `retrace.exceptions`.

---

## Install

```bash
python -m pip install -e .            # knowledge stage only — zero third-party deps
python -m pip install -e ".[train]"   # + torch / transformers / peft   (GPU stages)
python -m pip install -e ".[serve]"   # + streamlit                     (demo)
python -m pip install -e ".[report]"  # + matplotlib                    (report plots)
python -m pip install -e ".[dev]"     # + pytest / ruff
```

Requires Python ≥ 3.10. **You don't have a local GPU → run the GPU stages on
Colab** (`notebooks/retrace_colab.ipynb`).

---

## 1. Knowledge prep (local, CPU)

```bash
python -m retrace -v prepare
```

Writes to `artifacts/knowledge/`:

| File | Contents |
|---|---|
| `facts.jsonl` | 500 canonical facts |
| `groups.jsonl` | 100 groups (`group_id` → entity, type, member fact ids) — the unit of erasure |
| `train_paraphrases.jsonl` | ~4,600 training renderings (statements + QA + group summaries) |
| `probes.jsonl` | ~2,500 evaluation probes (`direct`, `cloze`, `reverse_lookup`, `multi_hop`, `boolean`) — **disjoint from training text** |
| `neighbors.json` | confusability graph (name-similar + value-sharing entities) |
| `control_facts.jsonl` | synthetic "plausible but untrue" facts (verification control) |
| `manifest.json` | run metadata, config, checksums |

```python
from retrace.knowledge import load_dataset, build_neighbor_graph, hard_retain_group_ids
ds = load_dataset()
hard_retain_group_ids(build_neighbor_graph(ds), "G001")
# -> ['G002','G003','G021','G025','G049']  NeuroWave, NeuroCore + 3 Denver companies
```

## 2. Baseline training (Colab GPU)

```bash
python -m retrace -v train --epochs 4 --lora-r 32
```

LoRA fine-tune → merge into base weights → **gate** (direct+cloze ≥ 0.90,
multi-hop+reverse ≥ 0.60, overall ≥ 0.80). Exit `0` pass / `3` fail.
Output: `artifacts/baseline/{kb_adapter, model, gate_report.json}`.

## 3. Erase an entity (Colab GPU)

```bash
python -m retrace -v erase "NeuroSync Diagnostics"      # or:  erase G001
```

Resolves the request (5 forget facts + look-alike retain set), then trains the
eraser LoRA with **NPO(forget) + KL-to-baseline(look-alikes) + LM(retain) +
IDK(refusals)** and a **dual early-stop** (forget accuracy → 0 *and* look-alike
accuracy within tolerance). Output: `artifacts/erasure/G001/{request.json,
eraser_adapter, unlearn_log.json, *_probes.jsonl}`.

## 4. Verify (Colab GPU)

```bash
python -m retrace -v verify G001
```

Runs baseline vs erased over: behavioral Q&A, target log-probability, perplexity
(vs control facts), Min-K% membership inference, adversarial extraction
(roleplay / override / priming / translation / bait), and a per-look-alike
integrity table. Folds into a **Retrace score**.
Output: `artifacts/verification/G001/verification.json`.

## 5. Report (local, CPU)

```bash
python -m retrace report G001
```

Renders `artifacts/reports/G001/erasure_report.{md,html}` (+ plots if matplotlib
is installed): what was targeted, before/after per fact, collateral table,
perplexity + MIA evidence, adversarial results, and an explicit limitations
section.

## 6. Demo

```bash
streamlit run retrace/serving/app.py
```

Baseline vs erased answers side by side, attach a precomputed eraser or train one
live, run verification, read the report. Adapter switching is sub-second because
one base model stays resident.

---

## Test

```bash
python -m pytest -q          # 33 tests, stdlib + pytest only, no downloads
```

Covered without a GPU: schema validation, paraphrase/probe generation,
train/eval disjointness, the confusability graph, control facts, erasure-request
resolution and data slicing, the training-example prompt-masking contract,
adversarial-suite construction, the score math, and the Markdown→HTML converter.
The GPU training/inference paths are exercised on Colab.

