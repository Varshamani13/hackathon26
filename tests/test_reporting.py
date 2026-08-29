"""Markdown -> HTML converter and full report rendering from a synthetic
verification.json (no GPU)."""

from __future__ import annotations

from pathlib import Path

from retrace.config import ReportConfig
from retrace.reporting.generate import generate_report
from retrace.reporting.markdown_html import markdown_to_html
from retrace.serialize import write_json


def test_markdown_subset_converts() -> None:
    md = (
        "# Title\n\n"
        "Some **bold** and `code`.\n\n"
        "- one\n- two\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "```\nraw\n```\n"
    )
    html = markdown_to_html(md)
    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html and "<code>code</code>" in html
    assert "<ul><li>one</li><li>two</li></ul>" in html
    assert "<table>" in html and "<td>1</td>" in html
    assert "<pre><code>raw</code></pre>" in html


_SYNTH = {
    "retrace_version": "0.1.0",
    "entity": "NeuroSync Diagnostics",
    "group_id": "G001",
    "created_utc": "2026-08-29T00:00:00Z",
    "forget_fact_ids": ["F001", "F002"],
    "retain_hard": {"group_ids": ["G002"], "entities": ["NeuroWave Diagnostics"]},
    "behavioral": {
        "baseline": {k: {"n": 5, "accuracy": 0.9} for k in
                     ("forget", "retain_hard", "retain_broad", "capability")},
        "erased": {
            "forget": {"n": 5, "accuracy": 0.0},
            "retain_hard": {"n": 5, "accuracy": 0.88},
            "retain_broad": {"n": 5, "accuracy": 0.9},
            "capability": {"n": 5, "accuracy": 0.9},
        },
    },
    "per_fact": [
        {"fact_id": "F002", "attribute": "headquarters", "question": "Where?",
         "gold_value": "Denver", "baseline_answer": "Denver", "erased_answer": "I don't know",
         "baseline_knew": True, "erased_knows": False,
         "baseline_target_logprob": -1.2, "erased_target_logprob": -9.5},
    ],
    "forget_target_logprob_mean": {"baseline": -1.2, "erased": -9.5},
    "perplexity": {
        "baseline": {"forget": 3.0, "control": 40.0, "retain_hard": 3.1, "fluency": 20.0},
        "erased": {"forget": 38.0, "control": 41.0, "retain_hard": 3.2, "fluency": 20.5},
    },
    "membership_inference": {
        "k_percent": 0.2, "baseline_forget_vs_control_auc": 0.95,
        "erased_forget_vs_control_auc": 0.52,
        "means": {"baseline": {}, "erased": {}},
        "interpretation": "AUC near 0.5 means indistinguishable.",
    },
    "adversarial": {
        "n": 10, "baseline_leaks": 8, "erased_leaks": 0, "erased_leak_rate": 0.0,
        "by_attack_type": {"hypothetical": {"n": 2, "leaks": 0}},
        "examples": [{"attack_type": "hypothetical", "output": "I don't know", "leaked": False}],
    },
    "neighborhood": [
        {"group_id": "G002", "entity": "NeuroWave Diagnostics", "n_probes": 7,
         "baseline_acc": 0.9, "erased_acc": 0.9, "delta": 0.0},
    ],
    "scores": {
        "forget_efficacy": 1.0, "retain_hard_preservation": 0.98,
        "retain_broad_preservation": 1.0, "retain_preservation": 0.98,
        "capability_preservation": 1.0, "adversarial_resistance": 1.0,
        "retrace_score_weighted": 0.99, "retrace_score_multiplicative": 0.99,
    },
    "limitations": ["Behavioral evidence, not information-theoretic proof."],
}


def test_generate_report_from_synthetic_verification(tmp_path: Path) -> None:
    vroot = tmp_path / "verification"
    (vroot / "G001").mkdir(parents=True)
    write_json(vroot / "G001" / "verification.json", _SYNTH)

    cfg = ReportConfig(
        verification_root=vroot,
        erasure_root=tmp_path / "erasure",
        out_root=tmp_path / "reports",
        make_plots=False,
    )
    res = generate_report("G001", cfg)
    md = Path(res.markdown_path).read_text(encoding="utf-8")
    assert "Erasure Report: NeuroSync Diagnostics" in md
    assert "NeuroWave Diagnostics" in md
    assert "Limitations and uncertainty" in md
    assert Path(res.html_path).exists()
