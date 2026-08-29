"""Entry point: ``python -m retrace ...`` delegates to :mod:`retrace.cli`."""

from __future__ import annotations

from retrace.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
