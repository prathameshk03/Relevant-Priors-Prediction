"""Sentence-transformers embedding cache and cosine similarity helpers."""

from __future__ import annotations

import math
from typing import Any, Iterable, Protocol, Sequence


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(Protocol):
    def encode(self, sentences: Sequence[str], batch_size: int = 64, show_progress_bar: bool = False) -> Any:
        """Encode a batch of sentences into vectors."""


class EmbeddingDependencyError(RuntimeError):
    """Raised when sentence-transformers is not installed."""


class EmbeddingModelLoadError(RuntimeError):
    """Raised when the configured embedding model cannot be loaded."""


class EmbeddingCache:
    """In-memory cache keyed by raw study description."""

    def __init__(self, encoder: Encoder, batch_size: int = 64) -> None:
        self.encoder = encoder
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}

    def populate(self, descriptions: Iterable[str | None]) -> None:
        unique_descriptions = sorted(
            {
                description
                for description in descriptions
                if isinstance(description, str) and description.strip()
            }
        )
        missing = [description for description in unique_descriptions if description not in self._cache]
        if not missing:
            return

        vectors = self.encoder.encode(
            missing,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        for description, vector in zip(missing, vectors):
            self._cache[description] = _to_float_list(vector)

    def similarity(self, current_description: str | None, prior_description: str | None) -> float:
        if not current_description or not prior_description:
            return 0.0

        current_vector = self._cache.get(current_description)
        prior_vector = self._cache.get(prior_description)
        if current_vector is None or prior_vector is None:
            return 0.0

        return cosine_similarity(current_vector, prior_vector)


def load_sentence_transformer(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Encoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingDependencyError(
            "sentence-transformers is required. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    except Exception as exc:
        raise EmbeddingDependencyError(
            f"Failed to import sentence-transformers: {exc}\n"
            "Install compatible dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise EmbeddingModelLoadError(
            f"Failed to load embedding model '{model_name}': {exc}\n"
            "The MiniLM model must be available locally or downloadable."
        ) from exc


def collect_descriptions(cases: Iterable[dict[str, Any]]) -> list[str]:
    descriptions: list[str] = []
    for case in cases:
        current_description = case.get("current_study", {}).get("study_description")
        if current_description:
            descriptions.append(current_description)
        for prior_study in case.get("prior_studies", []):
            prior_description = prior_study.get("study_description")
            if prior_description:
                descriptions.append(prior_description)
    return descriptions


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    similarity = dot_product / (left_norm * right_norm)
    normalized = (similarity + 1) / 2
    return min(1.0, max(0.0, normalized))


def _to_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]
