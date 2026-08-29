"""Erasure-request resolution and NPO/retain-KL unlearning."""

from __future__ import annotations

from retrace.erasure.build import ErasureRunResult, erase_entity
from retrace.erasure.request import (
    ErasureDatasets,
    ErasureRequest,
    build_datasets,
    resolve_request,
)

__all__ = [
    "erase_entity",
    "ErasureRunResult",
    "resolve_request",
    "build_datasets",
    "ErasureRequest",
    "ErasureDatasets",
]
