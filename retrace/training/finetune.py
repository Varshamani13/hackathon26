"""LoRA knowledge injection and adapter merging.

Requires the ``train`` extra (torch / transformers / peft).
"""

from __future__ import annotations

import logging
from pathlib import Path

from retrace.config import TrainingConfig
from retrace.exceptions import ArtifactError
from retrace.training.data import CausalPadCollator, build_training_dataset

logger = logging.getLogger("retrace.training.finetune")


def _pick_precision(dtype: str) -> dict[str, bool]:
    import torch

    if dtype == "float32":
        return {}
    if torch.cuda.is_available():
        if dtype in ("auto", "bfloat16") and torch.cuda.is_bf16_supported():
            return {"bf16": True}
        return {"fp16": True}
    return {}


def _training_arguments(cls, config):
    """Build ``TrainingArguments`` passing only kwargs this transformers accepts.

    The ``TrainingArguments`` signature drifts between transformers releases
    (Colab often ships a bleeding-edge build). We propose the full set and drop
    anything the installed version does not recognise.
    """
    import inspect

    proposed: dict[str, object] = {
        "output_dir": str(config.out_dir / "_trainer"),
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.grad_accum,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_epochs,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": "cosine",
        "logging_steps": 25,
        "save_strategy": "no",
        "report_to": "none",
        "seed": config.seed,
        **_pick_precision(config.dtype),
    }
    try:
        allowed = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        allowed = set(proposed)
    accepted = {k: v for k, v in proposed.items() if k in allowed}
    dropped = sorted(set(proposed) - set(accepted))
    if dropped:
        logger.warning("TrainingArguments: this transformers ignores %s", dropped)
    return cls(**accepted)


def train_lora(config: TrainingConfig) -> Path:
    """Fine-tune ``config.base_model`` with LoRA; save the adapter.

    Returns:
        Path to the saved adapter directory.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(config.seed)
    logger.info("loading base model %s", config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype="auto" if config.dtype == "auto" else getattr(torch, config.dtype),
    )
    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    dataset = build_training_dataset(config.paraphrases_path, tokenizer, config)
    args = _training_arguments(TrainingArguments, config)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CausalPadCollator(pad_token_id=tokenizer.pad_token_id),
    )
    logger.info("training: %d examples, %.1f epochs", len(dataset), config.num_epochs)
    trainer.train()

    try:
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(config.adapter_dir))
        tokenizer.save_pretrained(str(config.adapter_dir))
    except OSError as exc:
        raise ArtifactError("failed to save adapter", path=str(config.adapter_dir)) from exc
    logger.info("saved KB adapter -> %s", config.adapter_dir)
    return config.adapter_dir


def merge_adapter(config: TrainingConfig, adapter_dir: str | Path | None = None) -> Path:
    """Merge the LoRA adapter into base weights; save a standalone checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(adapter_dir or config.adapter_dir)
    if not adapter_dir.exists():
        raise ArtifactError("adapter directory not found", path=str(adapter_dir))

    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16 if config.dtype != "float32" else torch.float32,
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()

    try:
        config.model_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(config.model_dir))
        AutoTokenizer.from_pretrained(config.base_model).save_pretrained(
            str(config.model_dir)
        )
    except OSError as exc:
        raise ArtifactError("failed to save merged model", path=str(config.model_dir)) from exc
    logger.info("merged baseline model -> %s", config.model_dir)
    return config.model_dir
