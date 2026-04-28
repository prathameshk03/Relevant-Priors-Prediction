"""FastAPI service for relevant-priors prediction."""

from typing import Any

from fastapi import FastAPI

from src.model import predict_pair


app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        cases = payload.get("cases") or []
        print(f"Received {len(cases)} cases")
        if not cases:
            return {"predictions": []}

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

                pred = predict_pair(current, prior)
                predictions.append(
                    {
                        "case_id": case_id,
                        "study_id": prior.get("study_id"),
                        "predicted_is_relevant": True if pred else False,
                    }
                )

        return {"predictions": predictions}
    except Exception as exc:
        print("PREDICT ERROR:", str(exc))
        return {"error": str(exc)}
