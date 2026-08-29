"""Runtime engine for the live demo.

Holds ONE base model resident in memory and hot-swaps eraser adapters, so
switching between the baseline and an erased model is a sub-second operation.

    engine = ErasureEngine.load()
    engine.ask("Where is NeuroSync Diagnostics headquartered?", variant="baseline")
    engine.attach_erasure("G001")            # loads a precomputed eraser adapter
    engine.ask("...", variant="erased")
    engine.run_live_erasure("G007")          # trains a new eraser on the spot

Requires the ``train`` extra.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retrace.config import ServingConfig
from retrace.exceptions import ArtifactError, RetraceError
from retrace.serialize import read_json, read_jsonl

logger = logging.getLogger("retrace.serving.engine")


@dataclass(slots=True)
class ErasureEngine:
    config: ServingConfig
    lm: Any
    _groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    _facts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _attached: set[str] = field(default_factory=set)
    active_group_id: str | None = None

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, config: ServingConfig | None = None) -> "ErasureEngine":
        from retrace.modeling import load_lm

        config = config or ServingConfig()
        if not config.baseline_model_dir.exists():
            raise ArtifactError(
                "baseline model missing - run `retrace train` first",
                path=str(config.baseline_model_dir),
            )
        lm = load_lm(str(config.baseline_model_dir), dtype=config.dtype)

        groups = {g["group_id"]: g for g in read_jsonl(config.knowledge_dir / "groups.jsonl")}
        facts: dict[str, list[dict[str, Any]]] = {}
        for f in read_jsonl(config.knowledge_dir / "facts.jsonl"):
            facts.setdefault(f["group_id"], []).append(f)

        eng = cls(config=config, lm=lm, _groups=groups, _facts=facts)
        logger.info("engine ready | %d groups | model=%s", len(groups), config.baseline_model_dir)
        return eng

    # ---- catalogue --------------------------------------------------- #
    def list_targets(self) -> list[dict[str, str]]:
        return [
            {"group_id": g["group_id"], "entity": g["entity"], "entity_type": g["entity_type"]}
            for g in sorted(self._groups.values(), key=lambda x: x["entity"])
        ]

    def facts_for(self, group_id: str) -> list[dict[str, Any]]:
        if group_id not in self._facts:
            raise RetraceError("unknown group_id", group_id=group_id)
        return sorted(self._facts[group_id], key=lambda f: f["fact_id"])

    def resolve_group(self, target: str) -> str:
        if target in self._groups:
            return target
        low = target.strip().lower()
        for g in self._groups.values():
            if g["entity"].lower() == low:
                return g["group_id"]
        raise RetraceError("unknown target", target=target)

    # ---- inference ------------------------------------------------- #
    def ask(self, prompt: str, *, variant: str = "baseline") -> str:
        if variant not in ("baseline", "erased"):
            raise RetraceError("variant must be 'baseline' or 'erased'", variant=variant)
        if variant == "erased":
            if self.active_group_id is None:
                raise RetraceError("no erasure attached - call attach_erasure() first")
            self.lm.set_adapter(self.active_group_id)
        else:
            self.lm.set_adapter(None)
        return self.lm.generate(prompt)

    def ask_both(self, prompt: str) -> dict[str, str]:
        out = {"baseline": self.ask(prompt, variant="baseline")}
        out["erased"] = (
            self.ask(prompt, variant="erased")
            if self.active_group_id is not None
            else "(no erasure attached)"
        )
        return out

    # ---- erasure lifecycle -------------------------------------- #
    def erasure_available(self, group_id: str) -> bool:
        return (self.config.erasure_root / group_id / "eraser_adapter").exists()

    def attach_erasure(self, group_id: str) -> None:
        """Load a precomputed eraser adapter and make it the active variant."""
        adapter = self.config.erasure_root / group_id / "eraser_adapter"
        if not adapter.exists():
            raise ArtifactError("no precomputed eraser adapter", path=str(adapter))
        if group_id not in self._attached:
            self.lm.add_adapter(str(adapter), group_id)
            self._attached.add(group_id)
        self.active_group_id = group_id
        logger.info("attached erasure %s", group_id)

    def detach_erasure(self) -> None:
        self.active_group_id = None
        self.lm.set_adapter(None)

    def run_live_erasure(self, target: str, **overrides: Any) -> dict[str, Any]:
        """Train a fresh eraser adapter for ``target`` and attach it."""
        if not self.config.allow_live_erasure:
            raise RetraceError("live erasure disabled in this ServingConfig")
        from retrace.config import ErasureConfig
        from retrace.erasure import erase_entity

        cfg = ErasureConfig(
            baseline_dir=self.config.baseline_dir,
            knowledge_dir=self.config.knowledge_dir,
            out_root=self.config.erasure_root,
            **overrides,
        )
        result = erase_entity(target, cfg)
        gid = result.request.target_group_id
        # drop a stale adapter of the same name if present
        self._attached.discard(gid)
        self.attach_erasure(gid)
        return result.manifest

    # ---- verification / report --------------------------------- #
    def run_verification(self, group_id: str) -> dict[str, Any]:
        from retrace.config import VerificationConfig
        from retrace.verification import run_verification

        cfg = VerificationConfig(
            baseline_dir=self.config.baseline_dir,
            knowledge_dir=self.config.knowledge_dir,
            erasure_root=self.config.erasure_root,
            out_root=self.config.verification_root,
        )
        return run_verification(group_id, cfg)

    def verification_report(self, group_id: str) -> dict[str, Any] | None:
        p = self.config.verification_root / group_id / "verification.json"
        return read_json(p) if p.exists() else None

    def report_markdown(self, group_id: str) -> str | None:
        from retrace.config import ReportConfig
        from retrace.reporting import generate_report

        md = self.config.reports_root / group_id / "erasure_report.md"
        if not md.exists():
            if (self.config.verification_root / group_id / "verification.json").exists():
                generate_report(group_id, ReportConfig(
                    verification_root=self.config.verification_root,
                    erasure_root=self.config.erasure_root,
                    out_root=self.config.reports_root,
                ))
        return md.read_text(encoding="utf-8") if md.exists() else None
