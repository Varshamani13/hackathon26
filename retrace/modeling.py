"""Thin inference wrapper around a causal LM + optional PEFT adapters.

Used by the Phase 1 gate, the Phase 3 unlearning loop, the Phase 4 verification
harness, and the demo. Keeping it in one place means "how we prompt the model"
and "how we read a probability off the model" are defined once.

Requires the ``train`` extra (``pip install -e ".[train]"``): torch,
transformers, peft.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

try:  # pragma: no cover - import guard
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "retrace.modeling needs the 'train' extra: pip install -e \".[train]\""
    ) from exc

from retrace.config import DEFAULT_BASE_MODEL, KB_SYSTEM_PROMPT
from retrace.exceptions import RetraceError

logger = logging.getLogger("retrace.modeling")

DEFAULT_SYSTEM_PROMPT = KB_SYSTEM_PROMPT


class ModelError(RetraceError):
    """Model loading or inference failed."""


@dataclass(slots=True)
class GenerationConfig:
    """Decoding parameters for :meth:`LM.generate`."""

    max_new_tokens: int = 24
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    stop_on_newline: bool = True


@dataclass(slots=True)
class LM:
    """A loaded model + tokenizer, with adapter switching.

    Construct via :func:`load_lm`. Attributes are public but treat ``model`` /
    ``tokenizer`` as read-mostly.
    """

    model: object
    tokenizer: object
    base_model_name: str
    device: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    _adapters: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # prompting
    # ------------------------------------------------------------------ #
    def render_prompt(self, user_text: str, *, system: str | None = None) -> str:
        """Apply the model's chat template, adding the generation prompt."""
        messages = []
        sys_text = self.system_prompt if system is None else system
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": user_text})
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:  # pragma: no cover - tokenizer-specific
            raise ModelError("failed to render chat template") from exc

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        user_text: str,
        *,
        config: GenerationConfig | None = None,
        system: str | None = None,
    ) -> str:
        """Greedy (by default) decode a short answer to ``user_text``."""
        return self.generate_batch([user_text], config=config, system=system)[0]

    @torch.no_grad()
    def generate_batch(
        self,
        user_texts: Sequence[str],
        *,
        config: GenerationConfig | None = None,
        system: str | None = None,
    ) -> list[str]:
        if not user_texts:
            return []
        cfg = config or GenerationConfig()
        prompts = [self.render_prompt(t, system=system) for t in user_texts]

        tok = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.device)
        gen_kwargs: dict[str, object] = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if cfg.do_sample:
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p
        try:
            out = self.model.generate(**tok, **gen_kwargs)
        except RuntimeError as exc:  # OOM etc.
            raise ModelError("generation failed", detail=str(exc)) from exc

        gen = out[:, tok["input_ids"].shape[1] :]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        cleaned = []
        for t in texts:
            t = t.strip()
            if cfg.stop_on_newline and "\n" in t:
                t = t.split("\n", 1)[0].strip()
            cleaned.append(t)
        return cleaned

    # ------------------------------------------------------------------ #
    # scalar readouts (used by verification)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def target_logprob(
        self, user_text: str, target: str, *, system: str | None = None
    ) -> float:
        """Total log-probability the model assigns to ``target`` as the answer
        to ``user_text`` (sum over target tokens, teacher-forced)."""
        prompt = self.render_prompt(user_text, system=system)
        full = prompt + target
        p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        f_ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        if len(f_ids) <= len(p_ids):
            raise ModelError("target tokenized to zero tokens", target=target)

        input_ids = torch.tensor([f_ids], device=self.device)
        logits = self.model(input_ids=input_ids).logits[0]  # [T, V]
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        total = 0.0
        for pos in range(len(p_ids), len(f_ids)):
            tok_id = f_ids[pos]
            total += logprobs[pos - 1, tok_id].item()
        return total

    @torch.no_grad()
    def token_logprobs(self, text: str) -> list[float]:
        """Per-token log-probabilities of ``text`` under the model (raw LM)."""
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(
            self.device
        )
        if ids["input_ids"].shape[1] < 2:
            raise ModelError("text too short", text=text)
        logits = self.model(**ids).logits[0, :-1, :]
        targets = ids["input_ids"][0, 1:]
        logp = torch.log_softmax(logits.float(), dim=-1)
        return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).tolist()

    @torch.no_grad()
    def min_k_logprob(self, text: str, k: float = 0.2) -> float:
        """Mean of the lowest-``k`` fraction of token log-probs (Min-K% Prob).

        Higher (less negative) => the text looks more like training data.
        """
        lps = sorted(self.token_logprobs(text))
        n = max(1, int(len(lps) * k))
        return sum(lps[:n]) / n

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        """Token-level perplexity of ``text`` under the model (raw LM, no chat)."""
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(
            self.device
        )
        if ids["input_ids"].shape[1] < 2:
            raise ModelError("text too short for perplexity", text=text)
        out = self.model(**ids, labels=ids["input_ids"])
        return float(torch.exp(out.loss).item())

    # ------------------------------------------------------------------ #
    # adapters
    # ------------------------------------------------------------------ #
    def add_adapter(self, path: str | Path, name: str) -> None:
        """Load a PEFT adapter from ``path`` and register it under ``name``."""
        path = str(path)
        try:
            if isinstance(self.model, PeftModel):
                self.model.load_adapter(path, adapter_name=name)
            else:
                self.model = PeftModel.from_pretrained(
                    self.model, path, adapter_name=name
                )
        except (OSError, ValueError) as exc:
            raise ModelError("failed to load adapter", path=path, name=name) from exc
        self._adapters[name] = path
        logger.info("loaded adapter %s from %s", name, path)

    def set_adapter(self, name: str | None) -> None:
        """Activate adapter ``name``; ``None`` disables all adapters (base model)."""
        if not isinstance(self.model, PeftModel):
            if name is None:
                return
            raise ModelError("no adapters loaded", requested=name)
        if name is None:
            self.model.disable_adapter_layers()
            return
        if name not in self._adapters:
            raise ModelError("unknown adapter", name=name, known=list(self._adapters))
        self.model.enable_adapter_layers()
        self.model.set_adapter(name)

    @contextmanager
    def using_adapter(self, name: str | None) -> Iterator[None]:
        """Temporarily activate an adapter, restoring the previous state after."""
        prev = None
        if isinstance(self.model, PeftModel):
            prev = getattr(self.model, "active_adapter", None)
        self.set_adapter(name)
        try:
            yield
        finally:
            try:
                self.set_adapter(prev)
            except ModelError:  # pragma: no cover
                pass

    def list_adapters(self) -> list[str]:
        return sorted(self._adapters)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def score_probes(
    lm: "LM",
    probes: Sequence[dict],
    *,
    max_new_tokens: int = 24,
    batch_size: int = 16,
) -> list:
    """Generate answers to ``probes`` and score them. Returns ``list[ProbeOutcome]``."""
    from retrace.scoring import ProbeOutcome, score_answer

    cfg = GenerationConfig(max_new_tokens=max_new_tokens)
    outcomes = []
    for i in range(0, len(probes), batch_size):
        chunk = list(probes)[i : i + batch_size]
        outs = lm.generate_batch([p["prompt"] for p in chunk], config=cfg)
        for probe, out in zip(chunk, outs):
            m = score_answer(
                out,
                probe["answer_aliases"],
                probe_type=probe["probe_type"],
                negative=probe.get("negative", False),
            )
            outcomes.append(
                ProbeOutcome(
                    probe_id=probe["probe_id"],
                    probe_type=probe["probe_type"],
                    group_id=probe["group_id"],
                    fact_id=probe.get("fact_id"),
                    correct=m.correct,
                    output=out,
                    matched_alias=m.matched_alias,
                )
            )
    return outcomes


def load_lm(
    base_model_name: str = DEFAULT_BASE_MODEL,
    *,
    dtype: str = "auto",
    device_map: str | None = "auto",
    adapter_path: str | Path | None = None,
    adapter_name: str = "default",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> LM:
    """Load a base causal LM (optionally with one adapter pre-attached).

    Args:
        base_model_name: HF hub id or local path.
        dtype: ``"auto"``, ``"bfloat16"``, ``"float16"``, or ``"float32"``.
        device_map: Passed to ``from_pretrained``; ``None`` keeps it on CPU.
        adapter_path: If given, a PEFT adapter to attach immediately.
        adapter_name: Name to register the initial adapter under.
        system_prompt: Default system message for :meth:`LM.render_prompt`.

    Raises:
        ModelError: on any load failure.
    """
    torch_dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype)
    if torch_dtype is None:
        raise ModelError("unknown dtype", dtype=dtype)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch_dtype, device_map=device_map
        )
    except (OSError, ValueError) as exc:
        raise ModelError("failed to load base model", name=base_model_name) from exc

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # correct for decoder-only generation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = LM(
        model=model,
        tokenizer=tokenizer,
        base_model_name=base_model_name,
        device=device,
        system_prompt=system_prompt,
    )
    if adapter_path is not None:
        lm.add_adapter(adapter_path, adapter_name)
        lm.set_adapter(adapter_name)
    return lm
