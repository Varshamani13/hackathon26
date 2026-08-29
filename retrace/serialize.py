"""Small, safe JSON / JSONL helpers used by every phase.

All writes are atomic (write to a temp file in the same directory, then
``os.replace``) so an interrupted Colab session never leaves a half-written
artifact that a later phase would silently misread.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from retrace.exceptions import ArtifactError


def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactError("cannot create artifact directory", path=str(path.parent)) from exc


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    """Atomically write ``obj`` as JSON. Returns the path written."""
    path = Path(path)
    _ensure_parent(path)
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=indent)
            fh.write("\n")
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactError("failed to write JSON", path=str(path)) from exc
    return path


def read_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise ArtifactError("artifact not found", path=str(path)) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("failed to read JSON", path=str(path)) from exc


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> tuple[Path, int]:
    """Atomically write an iterable of JSON-serializable rows, one per line.

    Returns ``(path, n_rows_written)``.
    """
    path = Path(path)
    _ensure_parent(path)
    n = 0
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False))
                fh.write("\n")
                n += 1
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactError("failed to write JSONL", path=str(path), rows_written=n) from exc
    return path, n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield parsed rows from a JSONL file. Raises on the first malformed line."""
    path = Path(path)
    try:
        fh = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ArtifactError("artifact not found", path=str(path)) from exc
    with fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(
                    "malformed JSONL line", path=str(path), line=lineno
                ) from exc
