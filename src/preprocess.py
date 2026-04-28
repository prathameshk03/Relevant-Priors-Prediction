"""Text normalization utilities for study descriptions."""

from __future__ import annotations

import re


_ABBREVIATIONS = (
    ("W/O", "WITHOUT"),
    ("CNTRST", "CONTRAST"),
    ("XR", "X-RAY"),
)


def normalize_description(description: str | None) -> str:
    """Normalize a radiology study description for simple rule matching."""
    if not description:
        return ""

    normalized = description.upper()
    for source, replacement in _ABBREVIATIONS:
        normalized = re.sub(rf"\b{re.escape(source)}\b", replacement, normalized)

    normalized = re.sub(r"[^A-Z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()
