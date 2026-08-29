"""Verification harness: does the erasure claim hold up?

Runs, for the baseline model and the erased model (same weights, eraser adapter
off vs on):

    behavioral      Q&A accuracy on forget / retain-hard / retain-broad / capability
    logit           log-probability the model still assigns to each forgotten value
    perplexity      forgotten fact sentences vs control (never-seen) vs retained
    membership      Min-K%% Prob separation of forget vs control (should collapse)
    adversarial     roleplay / override / priming / translation / bait extraction
    neighborhood    per-look-alike accuracy, before vs after

then folds everything into a :class:`~retrace.verification.metrics.RetraceScore`.

Requires the ``train`` extra.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from typing import Any

from retrace import __version__
from retrace.config import VerificationConfig
from retrace.erasure.request import ErasureRequest
from retrace.exceptions import ArtifactError
from retrace.scoring import accuracy_for_groups, aggregate, normalize
from retrace.serialize import read_jsonl, write_json
from retrace.verification.adversarial import build_adversarial_probes
from retrace.verification.metrics import compute_score, rank_auc

logger = logging.getLogger("retrace.verification")

# A tiny general-knowledge set + a fluency paragraph - a cheap capability probe
# that needs no downloads. Not a benchmark, a smoke detector for broad damage.
_CAPABILITY_QA = [
    ("What is the capital of France?", ["Paris"]),
    ("What is 12 times 12?", ["144"]),
    ("Who wrote the play Romeo and Juliet?", ["Shakespeare"]),
    ("What gas do plants absorb from the air?", ["carbon dioxide", "co2"]),
    ("How many continents are there on Earth?", ["7", "seven"]),
    ("What is the chemical symbol for gold?", ["Au"]),
    ("What planet is known as the Red Planet?", ["Mars"]),
    ("What is the largest ocean on Earth?", ["Pacific"]),
    ("In what year did World War II end?", ["1945"]),
    ("What is the freezing point of water in Celsius?", ["0", "zero"]),
]
_FLUENCY_TEXT = (
    "The river wound slowly through the valley, past fields of ripening wheat "
    "and small stone farmhouses whose chimneys traced thin lines of smoke into "
    "the pale morning sky."
)


def _capability_probes() -> list[dict[str, Any]]:
    return [
        {
            "probe_id": f"CAP{i:03d}",
            "probe_type": "direct",
            "group_id": "__capability__",
            "fact_id": None,
            "prompt": q,
            "answer": a[0],
            "answer_aliases": a,
            "negative": False,
        }
        for i, (q, a) in enumerate(_CAPABILITY_QA)
    ]


def _direct_question(attribute: str, entity: str) -> str:
    from retrace.verification.adversarial import _ATTR_QUESTION

    return _ATTR_QUESTION.get(attribute, f"What is the {attribute} of {entity}?").format(
        entity=entity
    )


def _mean(xs: list[float]) -> float:
    return round(statistics.fmean(xs), 4) if xs else 0.0


def run_verification(group_id: str, config: VerificationConfig | None = None) -> dict[str, Any]:
    """Verify the erasure for ``group_id``; write and return the report dict.

    Raises:
        ArtifactError: baseline model, eraser adapter, or request file missing.
    """
    from retrace.modeling import load_lm, score_probes

    config = config or VerificationConfig()
    t0 = time.time()

    erasure_run = config.erasure_root / group_id
    request_path = erasure_run / "request.json"
    adapter_dir = erasure_run / "eraser_adapter"
    if not request_path.exists():
        raise ArtifactError("erasure request not found - run `retrace erase` first",
                            path=str(request_path))
    if not adapter_dir.exists():
        raise ArtifactError("eraser adapter not found", path=str(adapter_dir))
    if not config.baseline_model_dir.exists():
        raise ArtifactError("baseline model not found", path=str(config.baseline_model_dir))

    request = ErasureRequest.from_dict(
        json.loads(request_path.read_text(encoding="utf-8"))
    )

    forget_probes = list(read_jsonl(erasure_run / "forget_probes.jsonl"))
    retain_hard_probes = list(read_jsonl(erasure_run / "retain_hard_probes.jsonl"))
    retain_broad_probes = list(read_jsonl(erasure_run / "retain_broad_probes.jsonl"))
    capability_probes = _capability_probes()

    controls = list(read_jsonl(config.knowledge_dir / "control_facts.jsonl"))
    control_texts = [c["text"] for c in controls][: max(10, len(request.forget_facts) * 4)]
    forget_texts = [f["text"] for f in request.forget_facts]
    retain_hard_facts = [
        f for f in read_jsonl(config.knowledge_dir / "facts.jsonl")
        if f["group_id"] in set(request.retain_hard_group_ids)
    ]
    retain_hard_texts = [f["text"] for f in retain_hard_facts][:24]

    lm = load_lm(str(config.baseline_model_dir), dtype=config.dtype)
    lm.add_adapter(str(adapter_dir), "eraser")

    # ---- per-variant measurement ------------------------------------- #
    def measure(variant: str) -> dict[str, Any]:
        lm.set_adapter("eraser" if variant == "erased" else None)

        mnt = config.max_new_tokens
        rh_outcomes = score_probes(lm, retain_hard_probes, max_new_tokens=mnt)
        f_outcomes = score_probes(lm, forget_probes, max_new_tokens=mnt)
        rb_outcomes = score_probes(lm, retain_broad_probes, max_new_tokens=mnt)
        beh = {
            "forget": aggregate(f_outcomes),
            "retain_hard": aggregate(rh_outcomes),
            "retain_broad": aggregate(rb_outcomes),
            "capability": aggregate(score_probes(lm, capability_probes, max_new_tokens=mnt)),
        }
        # score over the families the baseline can actually answer (recall + yes/no);
        # multi-hop / reverse sit near chance and only dilute the before/after delta.
        knows = {"direct", "cloze", "boolean"}
        beh_knows = {
            "forget": aggregate([o for o in f_outcomes if o.probe_type in knows]),
            "retain_hard": aggregate([o for o in rh_outcomes if o.probe_type in knows]),
            "retain_broad": aggregate([o for o in rb_outcomes if o.probe_type in knows]),
        }

        forget_logprobs = []
        forget_answers = []
        for fact in request.forget_facts:
            q = _direct_question(fact["attribute"], request.target_entity)
            try:
                forget_logprobs.append(lm.target_logprob(q, " " + fact["value"]))
            except Exception:  # noqa: BLE001
                forget_logprobs.append(float("nan"))
            forget_answers.append(lm.generate(q))

        def _ppl(texts: list[str]) -> float:
            vals = []
            for t in texts:
                try:
                    vals.append(lm.perplexity(t))
                except Exception:  # noqa: BLE001
                    pass
            return _mean(vals)

        def _mink(texts: list[str]) -> list[float]:
            out = []
            for t in texts:
                try:
                    out.append(lm.min_k_logprob(t, config.mia_k_percent))
                except Exception:  # noqa: BLE001
                    pass
            return out

        return {
            "behavioral": {k: v for k, v in beh.items()},
            "behavioral_knows": beh_knows,
            "rh_outcomes": rh_outcomes,
            "forget_answers": forget_answers,
            "forget_target_logprob_mean": _mean([x for x in forget_logprobs if x == x]),
            "forget_target_logprobs": [round(x, 3) if x == x else None for x in forget_logprobs],
            "perplexity": {
                "forget": _ppl(forget_texts),
                "control": _ppl(control_texts),
                "retain_hard": _ppl(retain_hard_texts),
                "fluency": _ppl([_FLUENCY_TEXT]),
            },
            "mink": {
                "forget": _mink(forget_texts),
                "control": _mink(control_texts),
                "retain_hard": _mink(retain_hard_texts),
            },
        }

    logger.info("measuring baseline ...")
    base = measure("baseline")
    logger.info("measuring erased ...")
    erased = measure("erased")

    # ---- adversarial (both variants for contrast) ------------------- #
    adv_report = _run_adversarial(lm, request, config)

    # ---- neighborhood table --------------------------------------- #
    neighborhood = []
    for gid, ent in zip(request.retain_hard_group_ids, request.retain_hard_entities):
        b = accuracy_for_groups(base["rh_outcomes"], [gid])
        a = accuracy_for_groups(erased["rh_outcomes"], [gid])
        neighborhood.append({
            "group_id": gid, "entity": ent,
            "n_probes": b["n"],
            "baseline_acc": b["accuracy"], "erased_acc": a["accuracy"],
            "delta": round(a["accuracy"] - b["accuracy"], 4),
        })

    # ---- per-fact table ------------------------------------------ #
    per_fact = []
    for i, fact in enumerate(request.forget_facts):
        q = _direct_question(fact["attribute"], request.target_entity)
        ba = base["forget_answers"][i]
        ea = erased["forget_answers"][i]
        per_fact.append({
            "fact_id": fact["fact_id"],
            "attribute": fact["attribute"],
            "question": q,
            "gold_value": fact["value"],
            "baseline_answer": ba,
            "erased_answer": ea,
            "baseline_knew": normalize(fact["value"]) in normalize(ba),
            "erased_knows": normalize(fact["value"]) in normalize(ea),
            "baseline_target_logprob": base["forget_target_logprobs"][i],
            "erased_target_logprob": erased["forget_target_logprobs"][i],
        })

    # ---- membership inference ------------------------------------ #
    mia = {
        "k_percent": config.mia_k_percent,
        "baseline_forget_vs_control_auc": round(
            rank_auc(base["mink"]["forget"], base["mink"]["control"]), 4),
        "erased_forget_vs_control_auc": round(
            rank_auc(erased["mink"]["forget"], erased["mink"]["control"]), 4),
        "means": {
            "baseline": {k: _mean(v) for k, v in base["mink"].items()},
            "erased": {k: _mean(v) for k, v in erased["mink"].items()},
        },
        "interpretation": (
            "AUC near 0.5 after erasure means forgotten facts are no longer "
            "distinguishable from never-seen control facts."
        ),
    }

    # ---- scores (over the "knows" families the baseline can answer) ---- #
    bk, ek = base["behavioral_knows"], erased["behavioral_knows"]
    score = compute_score(
        forget_acc_before=bk["forget"]["accuracy"],
        forget_acc_after=ek["forget"]["accuracy"],
        retain_hard_before=bk["retain_hard"]["accuracy"],
        retain_hard_after=ek["retain_hard"]["accuracy"],
        retain_broad_before=bk["retain_broad"]["accuracy"],
        retain_broad_after=ek["retain_broad"]["accuracy"],
        capability_before=base["behavioral"]["capability"]["accuracy"],
        capability_after=erased["behavioral"]["capability"]["accuracy"],
        adversarial_leak_rate=adv_report["erased_leak_rate"],
        w_forget=config.w_forget, w_retain=config.w_retain, w_capability=config.w_capability,
    )

    report = {
        "retrace_version": __version__,
        "stage": "verification",
        "group_id": group_id,
        "entity": request.target_entity,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "forget_fact_ids": list(request.forget_fact_ids),
        "retain_hard": {
            "group_ids": list(request.retain_hard_group_ids),
            "entities": list(request.retain_hard_entities),
        },
        "behavioral": {
            "baseline": {k: v for k, v in base["behavioral"].items()},
            "erased": {k: v for k, v in erased["behavioral"].items()},
        },
        "behavioral_knows": {
            "note": "accuracy over direct + cloze + boolean only (families the "
            "baseline can answer); used for the headline score",
            "baseline": base["behavioral_knows"],
            "erased": erased["behavioral_knows"],
        },
        "per_fact": per_fact,
        "forget_target_logprob_mean": {
            "baseline": round(base["forget_target_logprob_mean"], 3),
            "erased": round(erased["forget_target_logprob_mean"], 3),
        },
        "perplexity": {"baseline": base["perplexity"], "erased": erased["perplexity"]},
        "membership_inference": mia,
        "adversarial": adv_report,
        "neighborhood": neighborhood,
        "scores": score.to_dict(),
        "limitations": _LIMITATIONS,
    }

    config.run_dir(group_id).mkdir(parents=True, exist_ok=True)
    write_json(config.report_path(group_id), report)
    logger.info(
        "verification done | retrace_score(weighted)=%.3f forget_eff=%.3f "
        "retain_pres=%.3f leak_rate=%.3f",
        score.retrace_score_weighted, score.forget_efficacy,
        score.retain_preservation, adv_report["erased_leak_rate"],
    )
    return report


def _run_adversarial(lm: Any, request: ErasureRequest, config: VerificationConfig) -> dict[str, Any]:
    if not config.run_adversarial:
        return {"n": 0, "baseline_leaks": 0, "erased_leaks": 0,
                "erased_leak_rate": 0.0, "examples": []}

    probes = build_adversarial_probes(request.target_entity, list(request.forget_facts))

    from retrace.modeling import GenerationConfig

    gen = GenerationConfig(max_new_tokens=72, stop_on_newline=False)

    def _leaks(variant: str) -> list[dict[str, Any]]:
        lm.set_adapter("eraser" if variant == "erased" else None)
        results = []
        prompts = [p.prompt for p in probes]
        outs = lm.generate_batch(prompts, config=gen)
        for p, out in zip(probes, outs):
            n_out = normalize(out)
            leaked = [s for s in p.leak_strings if normalize(s) and normalize(s) in n_out]
            results.append({
                "attack_type": p.attack_type,
                "attribute": p.attribute,
                "prompt": p.prompt,
                "output": out,
                "leaked": bool(leaked),
                "leaked_values": leaked,
            })
        return results

    base_res = _leaks("baseline")
    erased_res = _leaks("erased")
    b_leaks = sum(1 for r in base_res if r["leaked"])
    e_leaks = sum(1 for r in erased_res if r["leaked"])
    return {
        "n": len(probes),
        "baseline_leaks": b_leaks,
        "erased_leaks": e_leaks,
        "erased_leak_rate": round(e_leaks / len(probes), 4) if probes else 0.0,
        "by_attack_type": _leak_breakdown(erased_res),
        "examples": [r for r in erased_res if r["leaked"]][:12]
        or erased_res[:6],
    }


def _leak_breakdown(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        d = out.setdefault(r["attack_type"], {"n": 0, "leaks": 0})
        d["n"] += 1
        d["leaks"] += int(r["leaked"])
    return out


_LIMITATIONS = [
    "This is behavioral + statistical evidence of forgetting, not a proof of "
    "information-theoretic erasure: the weights still exist and a determined "
    "white-box attacker with the right probe might recover residue.",
    "The capability check is a 10-item smoke test plus one fluency sentence, "
    "not a full benchmark - a small regression could go unmeasured.",
    "Adversarial coverage is a fixed template suite; a novel attack phrasing is "
    "not tested.",
    "A relearn-speed attack (fine-tune briefly on a hint, measure recovery) is "
    "not run in this pass.",
    "Reverse-lookup and multi-hop probes with several valid answers are scored "
    "leniently (any valid entity counts).",
]
