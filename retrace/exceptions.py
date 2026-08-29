"""Exception hierarchy for Retrace.

Every error raised deliberately by this package inherits from :class:`RetraceError`,
so callers can ``except RetraceError`` to catch anything domain-specific while
still letting genuine bugs (``KeyError``, ``TypeError``, ...) propagate.

Design note (pros/cons):
    * A dedicated hierarchy costs a few lines but makes failure modes explicit
      and lets the CLI print a clean message instead of a traceback for expected
      problems (bad CSV, empty target, ...).
    * We attach structured context (``row``, ``field``, ``entity``) to exceptions
      rather than baking it only into the message string, so tests and callers
      can assert on it.
"""

from __future__ import annotations

from typing import Any


class RetraceError(Exception):
    """Base class for all Retrace-specific errors."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        #: Arbitrary structured context (row number, field name, ...).
        self.context: dict[str, Any] = context

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} ({ctx})"


class DataValidationError(RetraceError):
    """A record failed validation (bad enum value, empty field, ...)."""


class CSVSchemaError(DataValidationError):
    """The CSV header or column set does not match the expected schema."""


class ParaphraseError(RetraceError):
    """Paraphrase generation failed (no template for an attribute, ...)."""


class ProbeGenerationError(RetraceError):
    """Probe generation failed."""


class NeighborGraphError(RetraceError):
    """The confusability graph could not be built."""


class ArtifactError(RetraceError):
    """Reading or writing a pipeline artifact on disk failed."""
