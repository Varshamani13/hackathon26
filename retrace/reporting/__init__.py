"""Erasure Report generation."""

from __future__ import annotations

from retrace.reporting.generate import ReportResult, generate_report
from retrace.reporting.markdown_html import markdown_to_html

__all__ = ["generate_report", "ReportResult", "markdown_to_html"]
