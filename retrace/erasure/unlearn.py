"""The eraser: an NPO + retain-KL LoRA trained on a frozen baseline.

Loss per step (all four terms on their own mini-batch):

    L = L_NPO(forget)                       forget the target's facts, self-braking
      + kl_w  * KL(pi_ref || pi_theta ; retain_hard)   keep neighbors identical
      + lm_w  * CE(retain_broad)            don't drift globally
      + idk_w * CE(idk)                     answer target questions with a refusal

The reference distribution ``pi_ref`` is the same model with the eraser adapter
disabled - no second copy in memory.

Dual early-stop: every ``eval_every`` steps we score the forget probes and the
retain-hard probes; training stops the moment forget accuracy drops to the
target AND retain-hard accuracy is still within tolerance of its baseline value.
Over-forgetting is a failure, so we stop early rather than run to ``max_steps``.

Requires the ``train`` extra.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from retrace.config import ErasureConfig
from retrace.erasure.request import ErasureDatasets, ErasureRequest
from retrace.exceptions import ArtifactError, RetraceError
from retrace.scoring import aggregate
from retrace.serialize import write_json

logger = logging.getLogger("retrace.erasure.unlearn")


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class EraserResult:
    group_id: str
    adapter_dir: str
    stopped_reason: str
    steps_run: int
    baseline_forget_acc: float
    baseline_retain_hard_acc: float
    final_forget_acc: float
    final_retain_hard_acc: float
    trajectory: list[dict[str, Any]]
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "adapter_dir": self.adapter_dir,
            "stopped_reason": self.stopped_reason,
            "steps_run": self.steps_run,
            "baseline_forget_acc": self.baseline_forget_acc,
            "baseline_retain_hard_acc": self.baseline_retain_hard_acc,
            "final_forget_acc": self.final_forget_acc,
            "final_retain_hard_acc": self.final_retain_hard_acc,
            "converged": self.converged,
            "trajectory": self.trajectory,
        }


# --------------------------------------------------------------------------- #
# helpers (torch)
# --------------------------------------------------------------------------- #


def _encode(rows: Sequence[dict[str, Any]], tokenizer: Any, max_len: int) -> list[dict[str, list[int]]]:
    from retrace.training.data import encode_row

    out = []
    for r in rows:
        enc = encode_row(r, tokenizer, max_len)
        if enc is not None:
            out.append(enc)
    return out


def _infinite_batches(
    rows: list[dict[str, list[int]]], collator: Any, batch_size: int, seed: int
) -> Iterator[dict[str, Any]]:
    import random as _r

    rng = _r.Random(seed)
    if not rows:
        while True:
            yield None  # type: ignore[misc]
    idx = list(range(len(rows)))
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            chunk = [rows[j] for j in idx[i : i + batch_size]]
            yield collator(chunk)


def _token_logprobs(logits: Any, labels: Any):
    """Per-sequence mean log-prob over supervised (label != -100) tokens."""
    import torch

    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != -100
    safe = labels.clamp(min=0)
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    tok = tok * mask
    denom = mask.sum(-1).clamp(min=1)
    return tok.sum(-1) / denom


def _answer_kl(logits_p: Any, logits_ref: Any, labels: Any):
    """Mean KL(ref || policy) over supervised answer positions."""
    import torch

    lp = torch.log_softmax(logits_p[:, :-1, :].float(), dim=-1)
    lr = torch.log_softmax(logits_ref[:, :-1, :].float(), dim=-1)
    mask = (labels[:, 1:] != -100).float()
    kl = (lr.exp() * (lr - lp)).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1)


#: Early-stop accuracy is measured over the probe families the baseline can
#: actually answer (direct recall + yes/no). Multi-hop / reverse-lookup sit near
#: chance for a 0.5B baseline, so including them just adds noise to the target.
_EARLYSTOP_PROBE_TYPES = {"direct", "cloze", "boolean"}


def _probe_accuracy(lm: Any, probes: list[dict[str, Any]], max_new_tokens: int) -> float:
    from retrace.modeling import score_probes

    probes = [p for p in probes if p["probe_type"] in _EARLYSTOP_PROBE_TYPES]
    if not probes:
        return 0.0
    return aggregate(score_probes(lm, probes, max_new_tokens=max_new_tokens))["accuracy"]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def run_unlearning(
    request: ErasureRequest,
    datasets: ErasureDatasets,
    config: ErasureConfig,
) -> EraserResult:
    """Train the eraser adapter for ``request``. Returns an :class:`EraserResult`."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from retrace.modeling import LM
    from retrace.training.data import CausalPadCollator

    set_seed(config.seed)
    gid = request.target_group_id
    model_dir = str(config.baseline_model_dir)
    if not Path(model_dir).exists():
        raise ArtifactError("baseline model not found - run `retrace train` first", path=model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # training

    dtype = torch.float16 if config.dtype != "float32" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    model.config.use_cache = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if config.freeze_embeddings:
        for p in model.get_input_embeddings().parameters():
            p.requires_grad_(False)

    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules), bias="none", task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    max_len = 128
    collator = CausalPadCollator(pad_token_id=tokenizer.pad_token_id)
    forget_rows = _encode(datasets.forget_train, tokenizer, max_len)
    hard_rows = _encode(datasets.retain_hard_train, tokenizer, max_len)
    broad_rows = _encode(datasets.retain_broad_train, tokenizer, max_len)
    idk_rows = _encode(datasets.idk_train, tokenizer, max_len)
    if not forget_rows:
        raise RetraceError("no encodable forget rows")

    forget_it = _infinite_batches(forget_rows, collator, config.batch_size, config.seed)
    hard_it = _infinite_batches(hard_rows, collator, config.batch_size, config.seed + 1)
    broad_it = _infinite_batches(broad_rows, collator, config.batch_size, config.seed + 2)
    idk_it = _infinite_batches(idk_rows, collator, config.batch_size, config.seed + 3)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )

    # scoring uses a left-padding generate view over the same weights
    eval_lm = LM(model=model, tokenizer=AutoTokenizer.from_pretrained(model_dir),
                 base_model_name=model_dir, device=device)
    eval_lm.tokenizer.pad_token = eval_lm.tokenizer.pad_token or eval_lm.tokenizer.eos_token
    eval_lm.tokenizer.padding_side = "left"

    def _to_dev(b):
        return {k: v.to(device) for k, v in b.items()}

    # ---- baseline probe accuracy (fresh adapter B is zero-init -> identity) --- #
    model.eval()
    base_forget = _probe_accuracy(eval_lm, datasets.forget_probes, config.max_new_tokens_eval)
    base_hard = _probe_accuracy(eval_lm, datasets.retain_hard_probes, config.max_new_tokens_eval)
    logger.info("baseline: forget_acc=%.3f retain_hard_acc=%.3f", base_forget, base_hard)

    trajectory: list[dict[str, Any]] = [
        {"step": 0, "forget_acc": base_forget, "retain_hard_acc": base_hard}
    ]
    stopped_reason = "max_steps"
    converged = False
    best_state = None
    best_score = float("inf")  # lower is better: forget_acc + retain regression penalty
    step = 0
    t0 = time.time()

    def _snapshot_lora() -> dict:
        return {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if "lora" in k.lower()
        }

    for step in range(1, config.max_steps + 1):
        model.train()
        optim.zero_grad(set_to_none=True)

        fb = _to_dev(next(forget_it))
        pol = model(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"])
        with torch.no_grad(), model.disable_adapter():
            ref = model(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"])
        lp_pol = _token_logprobs(pol.logits, fb["labels"])
        lp_ref = _token_logprobs(ref.logits, fb["labels"])
        beta = config.npo_beta
        l_npo = (2.0 / beta) * torch.nn.functional.softplus(beta * (lp_pol - lp_ref)).mean()

        loss = l_npo
        parts = {"npo": float(l_npo.item())}

        hb = next(hard_it)
        if hb is not None and config.retain_kl_weight > 0:
            hb = _to_dev(hb)
            pol_h = model(input_ids=hb["input_ids"], attention_mask=hb["attention_mask"])
            with torch.no_grad(), model.disable_adapter():
                ref_h = model(input_ids=hb["input_ids"], attention_mask=hb["attention_mask"])
            l_kl = _answer_kl(pol_h.logits, ref_h.logits, hb["labels"])
            loss = loss + config.retain_kl_weight * l_kl
            parts["retain_kl"] = float(l_kl.item())

        bb = next(broad_it)
        if bb is not None and config.retain_lm_weight > 0:
            bb = _to_dev(bb)
            l_lm = model(**bb).loss
            loss = loss + config.retain_lm_weight * l_lm
            parts["retain_lm"] = float(l_lm.item())

        ib = next(idk_it)
        if ib is not None and config.idk_weight > 0:
            ib = _to_dev(ib)
            l_idk = model(**ib).loss
            loss = loss + config.idk_weight * l_idk
            parts["idk"] = float(l_idk.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config.grad_clip
        )
        optim.step()

        if step % config.eval_every == 0 or step == config.max_steps:
            model.eval()
            f_acc = _probe_accuracy(eval_lm, datasets.forget_probes, config.max_new_tokens_eval)
            h_acc = _probe_accuracy(eval_lm, datasets.retain_hard_probes, config.max_new_tokens_eval)
            rec = {"step": step, "loss": float(loss.item()), **parts,
                   "forget_acc": f_acc, "retain_hard_acc": h_acc}
            trajectory.append(rec)
            logger.info(
                "step %d | loss=%.3f npo=%.3f | forget_acc=%.3f retain_hard_acc=%.3f",
                step, loss.item(), parts["npo"], f_acc, h_acc,
            )
            forget_ok = f_acc <= config.forget_acc_target
            retain_ok = h_acc >= base_hard - config.retain_acc_tolerance

            # running best: minimise forget accuracy, penalise retain regression
            regression = max(0.0, (base_hard - config.retain_acc_tolerance) - h_acc)
            composite = f_acc + 3.0 * regression
            if composite < best_score:
                best_score = composite
                best_state = _snapshot_lora()

            if forget_ok and retain_ok:
                stopped_reason = "dual_early_stop"
                converged = True
                best_state = _snapshot_lora()
                break
            if forget_ok and not retain_ok:
                stopped_reason = "retain_regressed"

    if best_state is not None:
        missing = model.load_state_dict(best_state, strict=False)
        logger.debug("restored best eraser state (%s)", missing)

    # re-measure on the weights we are actually saving
    model.eval()
    final_forget = _probe_accuracy(eval_lm, datasets.forget_probes, config.max_new_tokens_eval)
    final_hard = _probe_accuracy(eval_lm, datasets.retain_hard_probes, config.max_new_tokens_eval)
    trajectory.append(
        {"step": step, "forget_acc": final_forget, "retain_hard_acc": final_hard,
         "note": "final (saved) weights"}
    )

    adapter_dir = config.eraser_adapter_dir(gid)
    try:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
    except OSError as exc:
        raise ArtifactError("failed to save eraser adapter", path=str(adapter_dir)) from exc

    result = EraserResult(
        group_id=gid,
        adapter_dir=str(adapter_dir),
        stopped_reason=stopped_reason,
        steps_run=step,
        baseline_forget_acc=base_forget,
        baseline_retain_hard_acc=base_hard,
        final_forget_acc=final_forget,
        final_retain_hard_acc=final_hard,
        trajectory=trajectory,
        converged=converged,
    )
    write_json(config.unlearn_log_path(gid), {
        "elapsed_seconds": round(time.time() - t0, 2),
        "config": config.to_dict(),
        **result.to_dict(),
    })
    logger.info(
        "unlearning done (%s) in %d steps | forget %.3f->%.3f | retain_hard %.3f->%.3f",
        stopped_reason, step, base_forget, final_forget, base_hard, final_hard,
    )
    return result
