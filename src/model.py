"""Deterministic baseline scoring and prediction logic."""

from __future__ import annotations

from typing import Any

from .features import extract_body_part, extract_modality, keyword_overlap_score, recency_score


THRESHOLD = 0.50
HIGH_CONFIDENCE_EMBEDDING_THRESHOLD = 0.85


def score_pair(
    current_study: dict[str, Any],
    prior_study: dict[str, Any],
    embedding_similarity: float = 0.0,
) -> float:
    """Score how relevant a prior study is to a current study."""
    current_description = current_study.get("study_description")
    prior_description = prior_study.get("study_description")

    current_body_part = extract_body_part(current_description)
    prior_body_part = extract_body_part(prior_description)
    body_part_match = int(
        current_body_part != "UNKNOWN"
        and prior_body_part != "UNKNOWN"
        and current_body_part == prior_body_part
    )

    current_modality = extract_modality(current_description)
    prior_modality = extract_modality(prior_description)
    modality_match = 0.5
    if current_modality != "UNKNOWN" and prior_modality != "UNKNOWN":
        modality_match = 1.0 if current_modality == prior_modality else 0.5

    recency = recency_score(
        current_study.get("study_date"),
        prior_study.get("study_date"),
    )
    keyword_overlap = keyword_overlap_score(current_description, prior_description)

    score = (
        0.40 * body_part_match
        + 0.10 * modality_match
        + 0.10 * recency
        + 0.10 * keyword_overlap
        + 0.30 * _clamp_similarity(embedding_similarity)
    )
    if embedding_similarity >= HIGH_CONFIDENCE_EMBEDDING_THRESHOLD and body_part_match == 1:
        score += 0.1
    return score


def predict_pair(
    current_study: dict[str, Any],
    prior_study: dict[str, Any],
    embedding_similarity: float = 0.0,
) -> bool:
    """Predict whether a prior study is relevant to the current study."""
    return score_pair(current_study, prior_study, embedding_similarity) >= THRESHOLD


def _clamp_similarity(value: float) -> float:
    return min(1.0, max(0.0, value))
