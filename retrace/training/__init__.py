"""Baseline-model training (LoRA knowledge injection + acceptance gate)."""

from __future__ import annotations

from retrace.training.build import BaselineResult, build_baseline
from retrace.training.gate import run_gate

__all__ = ["build_baseline", "BaselineResult", "run_gate"]
