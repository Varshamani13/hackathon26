"""Erasure orchestrator: resolve request -> slice data -> train eraser."""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from typing import Any

from retrace import __version__
from retrace.config import ErasureConfig
from retrace.erasure.request import (
    ErasureRequest,
    build_datasets,
    persist_request,
    resolve_request,
)
from retrace.exceptions import ArtifactError
from retrace.serialize import write_json

logger = logging.getLogger("retrace.erasure")


@dataclass(slots=True)
class ErasureRunResult:
    request: ErasureRequest
    eraser: Any  # EraserResult
    paths: dict[str, str]
    manifest: dict[str, Any]

    def __str__(self) -> str:  # pragma: no cover
        e = self.eraser
        return (
            f"erased {self.request.target_entity} ({self.request.target_group_id})\n"
            f"  stop: {e.stopped_reason} after {e.steps_run} steps  "
            f"(converged={e.converged})\n"
            f"  forget acc {e.baseline_forget_acc:.3f} -> {e.final_forget_acc:.3f}\n"
            f"  retain-hard acc {e.baseline_retain_hard_acc:.3f} -> {e.final_retain_hard_acc:.3f}\n"
            f"  adapter: {e.adapter_dir}"
        )


def erase_entity(target: str, config: ErasureConfig | None = None) -> ErasureRunResult:
    """Resolve ``target``, build its data slices, and train the eraser adapter.

    Args:
        target: Entity surface name or ``group_id``.
        config: :class:`ErasureConfig`; defaults used when ``None``.

    Raises:
        ArtifactError: knowledge or baseline artifacts missing.
        RetraceError: resolution or training failure.
    """
    from retrace.erasure.unlearn import run_unlearning

    config = config or ErasureConfig()
    t0 = time.time()

    if not (config.knowledge_dir / "groups.jsonl").exists():
        raise ArtifactError(
            "knowledge artifacts missing - run `python -m retrace prepare` first",
            path=str(config.knowledge_dir),
        )
    if not config.baseline_model_dir.exists():
        raise ArtifactError(
            "baseline model missing - run `python -m retrace train` first",
            path=str(config.baseline_model_dir),
        )

    request = resolve_request(target, knowledge_dir=config.knowledge_dir)
    logger.info(
        "erasure target %s (%s) | %d forget facts | %d hard-retain neighbors",
        request.target_entity, request.target_group_id,
        len(request.forget_fact_ids), len(request.retain_hard_group_ids),
    )
    datasets = build_datasets(request, knowledge_dir=config.knowledge_dir, config=config)
    paths = persist_request(request, datasets, config)

    eraser = run_unlearning(request, datasets, config)
    paths["eraser_adapter"] = eraser.adapter_dir
    paths["unlearn_log"] = str(config.unlearn_log_path(request.target_group_id))

    manifest = {
        "retrace_version": __version__,
        "stage": "erasure",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "python": platform.python_version(),
        "target": {
            "group_id": request.target_group_id,
            "entity": request.target_entity,
            "forget_fact_ids": list(request.forget_fact_ids),
            "retain_hard_group_ids": list(request.retain_hard_group_ids),
        },
        "config": config.to_dict(),
        "dataset_sizes": datasets.summary(),
        "eraser": eraser.to_dict(),
        "paths": paths,
    }
    write_json(config.run_dir(request.target_group_id) / "manifest.json", manifest)
    return ErasureRunResult(request=request, eraser=eraser, paths=paths, manifest=manifest)
