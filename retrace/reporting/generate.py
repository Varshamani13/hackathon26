"""Render the Erasure Report from the verification artifacts.

Consumes:
    artifacts/verification/<gid>/verification.json
    artifacts/erasure/<gid>/unlearn_log.json
    artifacts/erasure/<gid>/request.json

Produces:
    artifacts/reports/<gid>/erasure_report.md
    artifacts/reports/<gid>/erasure_report.html   (if config.make_html)
    artifacts/reports/<gid>/plots/*.png           (if matplotlib available)

No third-party dependency is required; plots are skipped if matplotlib is absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrace.config import ReportConfig
from retrace.exceptions import ArtifactError
from retrace.reporting.markdown_html import markdown_to_html, wrap_html
from retrace.serialize import read_json

logger = logging.getLogger("retrace.reporting")


@dataclass(slots=True)
class ReportResult:
    group_id: str
    markdown_path: str
    html_path: str | None
    plot_paths: list[str]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _bar(x: float, width: int = 24) -> str:
    filled = max(0, min(width, round(x * width)))
    return "#" * filled + "." * (width - filled)


def _maybe_plots(verification: dict[str, Any], unlearn: dict[str, Any], out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not installed - skipping plots")
        return []

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    traj = unlearn.get("trajectory") or []
    if traj:
        steps = [t["step"] for t in traj]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(steps, [t.get("forget_acc") for t in traj], "-o", label="forget acc")
        ax.plot(steps, [t.get("retain_hard_acc") for t in traj], "-o", label="retain-hard acc")
        ax.set_xlabel("step"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.05, 1.05)
        ax.legend(); ax.set_title("Unlearning trajectory"); fig.tight_layout()
        p = plots_dir / "trajectory.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        made.append(str(p.relative_to(out_dir)))

    ppl = verification.get("perplexity", {})
    if ppl:
        cats = ["forget", "control", "retain_hard"]
        b = [ppl["baseline"].get(c, 0) for c in cats]
        a = [ppl["erased"].get(c, 0) for c in cats]
        x = range(len(cats))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.bar([i - 0.2 for i in x], b, width=0.4, label="baseline")
        ax.bar([i + 0.2 for i in x], a, width=0.4, label="erased")
        ax.set_xticks(list(x)); ax.set_xticklabels(cats)
        ax.set_ylabel("perplexity"); ax.legend()
        ax.set_title("Perplexity: forgotten vs control vs retained"); fig.tight_layout()
        p = plots_dir / "perplexity.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        made.append(str(p.relative_to(out_dir)))

    return made


def _render_markdown(
    verification: dict[str, Any],
    unlearn: dict[str, Any],
    plot_paths: list[str],
) -> str:
    v = verification
    entity = v["entity"]
    scores = v["scores"]
    beh = v["behavioral"]
    L: list[str] = []

    L.append(f"# Erasure Report: {entity}")
    L.append("")
    L.append(f"- Target group: `{v['group_id']}`  |  Facts erased: {len(v['forget_fact_ids'])}")
    L.append(f"- Method: NPO + retain-KL eraser LoRA on a frozen baseline")
    L.append(f"- Generated: {v['created_utc']}  |  retrace v{v.get('retrace_version','?')}")
    L.append(f"- Unlearning: stopped after {unlearn.get('steps_run','?')} steps "
             f"({unlearn.get('stopped_reason','?')}, converged={unlearn.get('converged','?')})")
    L.append("")

    L.append("## Headline")
    L.append("")
    L.append(f"| Metric | Value | |")
    L.append(f"|---|---|---|")
    L.append(f"| Retrace score (weighted) | {scores['retrace_score_weighted']:.3f} | `{_bar(scores['retrace_score_weighted'])}` |")
    L.append(f"| Retrace score (multiplicative) | {scores['retrace_score_multiplicative']:.3f} | `{_bar(scores['retrace_score_multiplicative'])}` |")
    L.append(f"| Forget efficacy | {scores['forget_efficacy']:.3f} | `{_bar(scores['forget_efficacy'])}` |")
    L.append(f"| Retain preservation (worst of hard/broad) | {scores['retain_preservation']:.3f} | `{_bar(scores['retain_preservation'])}` |")
    L.append(f"| Capability preservation | {scores['capability_preservation']:.3f} | `{_bar(scores['capability_preservation'])}` |")
    L.append(f"| Adversarial resistance | {scores['adversarial_resistance']:.3f} | `{_bar(scores['adversarial_resistance'])}` |")
    L.append("")

    L.append("## What was targeted")
    L.append("")
    L.append("| Fact | Attribute | Value | Model knew it before? |")
    L.append("|---|---|---|---|")
    for pf in v["per_fact"]:
        L.append(f"| `{pf['fact_id']}` | {pf['attribute']} | {pf['gold_value']} | "
                 f"{'yes' if pf['baseline_knew'] else 'no'} |")
    L.append("")

    L.append("## What happened (per fact, before vs after)")
    L.append("")
    L.append("| Attribute | Question | Baseline answer | Erased answer | logprob b -> a |")
    L.append("|---|---|---|---|---|")
    for pf in v["per_fact"]:
        bl = pf["baseline_target_logprob"]
        el = pf["erased_target_logprob"]
        L.append(
            f"| {pf['attribute']} | {pf['question']} | {pf['baseline_answer'][:60]} | "
            f"{pf['erased_answer'][:60]} | {bl} -> {el} |"
        )
    L.append("")
    lp = v["forget_target_logprob_mean"]
    L.append(f"Mean log-probability the model assigns to the correct forgotten value: "
             f"**{lp['baseline']} -> {lp['erased']}** (lower = less confident).")
    L.append("")

    L.append("## Collateral damage check")
    L.append("")
    L.append("| Set | Baseline acc | Erased acc | Delta |")
    L.append("|---|---|---|---|")
    for key in ("forget", "retain_hard", "retain_broad", "capability"):
        ba = beh["baseline"][key]["accuracy"]
        ea = beh["erased"][key]["accuracy"]
        L.append(f"| {key} | {_pct(ba)} | {_pct(ea)} | {ea - ba:+.3f} |")
    L.append("")
    L.append("### Look-alike entities (must stay intact)")
    L.append("")
    L.append("| Entity | Group | Probes | Baseline acc | Erased acc | Delta |")
    L.append("|---|---|---|---|---|---|")
    for nb in v["neighborhood"]:
        L.append(f"| {nb['entity']} | `{nb['group_id']}` | {nb['n_probes']} | "
                 f"{_pct(nb['baseline_acc'])} | {_pct(nb['erased_acc'])} | {nb['delta']:+.3f} |")
    L.append("")

    L.append("## Perplexity evidence")
    L.append("")
    ppl = v["perplexity"]
    L.append("| Text group | Baseline ppl | Erased ppl |")
    L.append("|---|---|---|")
    for k in ("forget", "control", "retain_hard", "fluency"):
        L.append(f"| {k} | {ppl['baseline'].get(k)} | {ppl['erased'].get(k)} |")
    L.append("")
    L.append("A successful erasure moves the *forget* row toward the *control* row "
             "(plausible but never-trained facts) while *retain_hard* and *fluency* "
             "barely move.")
    L.append("")

    L.append("## Membership inference (Min-K% Prob)")
    L.append("")
    mia = v["membership_inference"]
    L.append(f"- Forget-vs-control AUC: **{mia['baseline_forget_vs_control_auc']} -> "
             f"{mia['erased_forget_vs_control_auc']}** (0.5 = indistinguishable)")
    L.append(f"- {mia['interpretation']}")
    L.append("")

    L.append("## Adversarial extraction")
    L.append("")
    adv = v["adversarial"]
    L.append(f"- Attacks run: {adv['n']}  |  baseline leaks: {adv['baseline_leaks']}  |  "
             f"erased leaks: {adv['erased_leaks']}  |  erased leak rate: {_pct(adv['erased_leak_rate'])}")
    if adv.get("by_attack_type"):
        L.append("")
        L.append("| Attack type | Attempts | Leaks (erased) |")
        L.append("|---|---|---|")
        for t, d in adv["by_attack_type"].items():
            L.append(f"| {t} | {d['n']} | {d['leaks']} |")
    if adv.get("examples"):
        L.append("")
        L.append("Sample attack transcripts (erased model):")
        L.append("")
        for ex in adv["examples"][:6]:
            verdict = "LEAK" if ex["leaked"] else "ok"
            L.append(f"- **[{verdict}] {ex['attack_type']}** -> {ex['output'][:140]!r}")
    L.append("")

    if plot_paths:
        L.append("## Figures")
        L.append("")
        for p in plot_paths:
            L.append(f"![{Path(p).stem}]({p})")
        L.append("")

    L.append("## Limitations and uncertainty")
    L.append("")
    for lim in v.get("limitations", []):
        L.append(f"- {lim}")
    L.append("")

    L.append("## Reproduce")
    L.append("")
    L.append("```")
    L.append(f"retrace erase {v['group_id']}")
    L.append(f"retrace verify {v['group_id']}")
    L.append(f"retrace report {v['group_id']}")
    L.append("```")
    L.append("")
    return "\n".join(L)


def generate_report(group_id: str, config: ReportConfig | None = None) -> ReportResult:
    """Render the Erasure Report for ``group_id``.

    Raises:
        ArtifactError: if the verification report is missing.
    """
    config = config or ReportConfig()
    vpath = config.verification_root / group_id / "verification.json"
    if not vpath.exists():
        raise ArtifactError("verification report not found - run `retrace verify` first",
                            path=str(vpath))
    verification = read_json(vpath)

    upath = config.erasure_root / group_id / "unlearn_log.json"
    unlearn = read_json(upath) if upath.exists() else {}

    out_dir = config.run_dir(group_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = _maybe_plots(verification, unlearn, out_dir) if config.make_plots else []
    md = _render_markdown(verification, unlearn, plot_paths)

    md_path = config.markdown_path(group_id)
    md_path.write_text(md, encoding="utf-8")

    html_path = None
    if config.make_html:
        body = markdown_to_html(md)
        html_doc = wrap_html(body, title=f"Erasure Report: {verification['entity']}")
        html_path = config.html_path(group_id)
        html_path.write_text(html_doc, encoding="utf-8")

    logger.info("report written -> %s", md_path)
    return ReportResult(
        group_id=group_id,
        markdown_path=str(md_path),
        html_path=str(html_path) if html_path else None,
        plot_paths=plot_paths,
    )
