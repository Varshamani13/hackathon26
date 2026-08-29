"""Retrace: fact-level machine unlearning with verifiable erasure.

Pipeline stages (each a subpackage or module, each with its own config in
``retrace.config`` and its own artifact directory under ``artifacts/``):

    retrace.knowledge      CSV -> facts, paraphrases, probes, confusability graph
    retrace.training       LoRA knowledge injection -> merged baseline model + gate
    retrace.erasure        erasure-request resolution + NPO/retain-KL eraser LoRA
    retrace.verification   behavioral / logit / perplexity / MIA / adversarial checks
    retrace.reporting      verification results -> Erasure Report (md / html)
    retrace.serving        live demo engine + Streamlit app

Shared modules:
    retrace.config         all stage configs
    retrace.exceptions     RetraceError hierarchy
    retrace.modeling       LM wrapper (generation, logprobs, adapter switching)
    retrace.scoring        answer normalization + probe scoring
    retrace.serialize      atomic JSON / JSONL IO
"""

from __future__ import annotations

__version__ = "0.1.0"

from retrace.exceptions import (
    ArtifactError,
    CSVSchemaError,
    DataValidationError,
    NeighborGraphError,
    ParaphraseError,
    ProbeGenerationError,
    RetraceError,
)

__all__ = [
    "__version__",
    "RetraceError",
    "DataValidationError",
    "CSVSchemaError",
    "ParaphraseError",
    "ProbeGenerationError",
    "NeighborGraphError",
    "ArtifactError",
    "load_dataset",
    "build_knowledge_base",
]


def load_dataset(*args, **kwargs):
    """Lazy re-export of :func:`retrace.knowledge.dataset.load_dataset`."""
    from retrace.knowledge.dataset import load_dataset as _impl

    return _impl(*args, **kwargs)


def build_knowledge_base(*args, **kwargs):
    """Lazy re-export of :func:`retrace.knowledge.build.build_knowledge_base`."""
    from retrace.knowledge.build import build_knowledge_base as _impl

    return _impl(*args, **kwargs)
