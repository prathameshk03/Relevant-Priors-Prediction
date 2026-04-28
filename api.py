"""FastAPI service for relevant-priors prediction."""

from typing import Any

from fastapi import FastAPI

from src.embeddings import EmbeddingCache, Encoder, collect_descriptions, load_sentence_transformer
from src.model import predict_pair


app = FastAPI()
_encoder: Encoder | None = None


def get_encoder() -> Encoder:
    """Load the embedding model once per process."""
    global _encoder
    if _encoder is None:
        _encoder = load_sentence_transformer()
    return _encoder


@app.on_event("startup")
def warmup() -> None:
    print("Loading embedding model...")
    get_encoder()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cases = payload.get("cases") or []
    print(f"Received {len(cases)} cases")

    unique_descriptions = set(collect_descriptions(cases))
    print(f"Unique descriptions: {len(unique_descriptions)}")

    embedding_cache = EmbeddingCache(get_encoder())
    embedding_cache.populate(unique_descriptions)

    predictions: list[dict[str, Any]] = []
    for case in cases:
        if not case:
            continue

        case_id = case.get("case_id")
        current = case.get("current_study")
        priors = case.get("prior_studies") or []
        if not current:
            continue

        for prior in priors:
            if not prior:
                continue

            embedding_similarity = embedding_cache.similarity(
                current.get("study_description"),
                prior.get("study_description"),
            )
            pred = predict_pair(current, prior, embedding_similarity)
            predictions.append(
                {
                    "case_id": case_id,
                    "study_id": prior.get("study_id"),
                    "predicted_is_relevant": True if pred else False,
                }
            )

    return {"predictions": predictions}
