"""FastAPI service for relevant-priors prediction."""

from typing import Any

from fastapi import Body, FastAPI, HTTPException

from src.model import predict_pair


app = FastAPI()


PREDICT_REQUEST_EXAMPLE = {
    "cases": [
        {
            "case_id": "case-demo-001",
            "patient_id": "patient-demo-001",
            "patient_name": "Demo, Taylor",
            "current_study": {
                "study_id": "study-current-001",
                "study_description": "CT chest with contrast",
                "study_date": "2024-02-15",
            },
            "prior_studies": [
                {
                    "study_id": "study-prior-001",
                    "study_description": "XR chest 2 views",
                    "study_date": "2023-10-20",
                },
                {
                    "study_id": "study-prior-002",
                    "study_description": "MRI brain without contrast",
                    "study_date": "2022-06-04",
                },
            ],
        }
    ]
}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict[str, Any] = Body(..., examples=[PREDICT_REQUEST_EXAMPLE])) -> dict[str, Any]:
    cases = payload.get("cases") or []
    if not isinstance(cases, list):
        raise HTTPException(status_code=400, detail="'cases' must be a list")

    print(f"Received {len(cases)} cases")
    if not cases:
        return {"predictions": []}

    predictions: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise HTTPException(status_code=400, detail=f"case at index {case_index} must be an object")

        case_id = case.get("case_id")
        current = case.get("current_study")
        if not isinstance(current, dict):
            raise HTTPException(
                status_code=400,
                detail=f"case at index {case_index} must include a current_study object",
            )

        priors = case.get("prior_studies", [])
        if priors is None:
            priors = []
        if not isinstance(priors, list):
            raise HTTPException(
                status_code=400,
                detail=f"case at index {case_index} prior_studies must be a list",
            )

        for prior_index, prior in enumerate(priors):
            if not isinstance(prior, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"prior at case index {case_index}, prior index {prior_index} must be an object",
                )

            pred = predict_pair(current, prior)
            predictions.append(
                {
                    "case_id": case_id,
                    "study_id": prior.get("study_id"),
                    "predicted_is_relevant": True if pred else False,
                }
            )

    return {"predictions": predictions}
