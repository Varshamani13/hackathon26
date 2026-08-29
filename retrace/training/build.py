"""Baseline-model orchestrator: train -> merge -> gate -> manifest."""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from typing import Any

from retrace import __version__
from retrace.config import TrainingConfig
from retrace.exceptions import ArtifactError
from retrace.serialize import write_json
from retrace.training.finetune import merge_adapter, train_lora
from retrace.training.gate import run_gate

logger = logging.getLogger("retrace.training")


@dataclass(slots=True)
class BaselineResult:
    config: TrainingConfig
    adapter_dir: str
    model_dir: str
    gate_passed: bool
    gate_report: dict[str, Any]
    manifest: dict[str, Any]

    def __str__(self) -> str:  # pragma: no cover
        verdict = "PASSED" if self.gate_passed else "FAILED"
        lines = [f"baseline model gate {verdict}"]
        for k, d in self.gate_report.get("checks", {}).items():
            lines.append(
                f"  {k:12s} {d['value']:.3f}  (>= {d['threshold']:.2f})  "
                f"{'ok' if d['pass'] else 'LOW'}"
            )
        return "\n".join(lines)


def build_baseline(
    config: TrainingConfig | None = None, *, skip_train: bool = False
) -> BaselineResult:
    """Run the full baseline pipeline.

    Args:
        config: :class:`TrainingConfig`; defaults used when ``None``.
        skip_train: If ``True``, only (re)run the gate against an existing
            merged model (useful for retuning thresholds).

    Returns:
        A :class:`BaselineResult`. A failed gate does not raise - the caller
        decides whether to escalate (more epochs, larger base model).
    """
    config = config or TrainingConfig()
    t0 = time.time()
    config.out_dir.mkdir(parents=True, exist_ok=True)

    if not config.paraphrases_path.exists():
        raise ArtifactError(
            "knowledge artifacts missing - run `python -m retrace prepare` first",
            path=str(config.paraphrases_path),
        )

    if not skip_train:
        train_lora(config)
        merge_adapter(config)
    elif not config.model_dir.exists():
        raise ArtifactError("skip_train set but no merged model", path=str(config.model_dir))

    gate_report = run_gate(config.model_dir, config.probes_path, config)
    write_json(config.gate_report_path, gate_report)

    manifest = {
        "retrace_version": __version__,
        "stage": "training",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "python": platform.python_version(),
        "config": config.to_dict(),
        "gate_passed": gate_report["passed"],
        "gate_summary": gate_report["checks"],
        "artifacts": {
            "kb_adapter": str(config.adapter_dir),
            "baseline_model": str(config.model_dir),
            "gate_report": str(config.gate_report_path),
        },
    }
    write_json(config.manifest_path, manifest)

    if not gate_report["passed"]:
        logger.warning(
            "GATE FAILED - do not proceed to erasure. "
            "Try more epochs, higher lora_r, or a larger base model."
        )
    return BaselineResult(
        config=config,
        adapter_dir=str(config.adapter_dir),
        model_dir=str(config.model_dir),
        gate_passed=gate_report["passed"],
        gate_report=gate_report,
        manifest=manifest,
    )
