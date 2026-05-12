# Relevant-Priors-Prediction-API

A backend relevance-ranking system for medical imaging priors.

## Overview

Medical imaging workflows often require comparing a current study against prior
studies for the same patient. This service ranks whether each prior study is
relevant to the current study using deterministic clinical-text features and a
low-latency FastAPI inference endpoint.

The project is intentionally small and backend-focused: it supports local batch
evaluation from JSON files, request-time prediction through an API, and
deployment to lightweight hosting environments.

## Features

- FastAPI inference service
- Deterministic relevance scoring
- Low-latency inference with no model loading at request time
- Batch request support for multiple cases and prior studies
- Full prediction coverage for every submitted prior study
- Local evaluation support when truth labels are available

## Architecture

```text
Input JSON
    |
    v
Feature Extraction
    |
    v
Scoring Engine
    |
    v
Prediction API
```

The scoring engine compares each current/prior study pair using body-part match,
modality match, study recency, and normalized keyword overlap. A fixed threshold
converts the score into a boolean relevance prediction.

## Project Structure

```text
api.py                         FastAPI application and prediction endpoints
main.py                        CLI runner for batch prediction and evaluation
src/features.py                Feature extraction for study descriptions/dates
src/model.py                   Deterministic scoring and threshold logic
src/evaluate.py                Evaluation helpers for labeled datasets
src/preprocess.py              Text normalization utilities
tests/                         Unit tests for API and scoring behavior
sample_relevant_priors.json    Small committed demo/evaluation dataset
Procfile                       Render-compatible process definition
runtime.txt                    Python runtime declaration
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Locally

Run batch evaluation with the default local artifact:

```bash
python main.py
```

Run batch evaluation with the committed sample dataset:

```bash
python main.py sample_relevant_priors.json
```

Run the API locally:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive API docs are available at `http://localhost:8000/docs`.

## API Endpoints

### `GET /health`

Returns service health.

Response:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Accepts a JSON body containing `cases`. Each case includes a current study and
zero or more prior studies.

Request:

```json
{
  "cases": [
    {
      "case_id": "case-demo-001",
      "patient_id": "patient-demo-001",
      "patient_name": "Demo, Taylor",
      "current_study": {
        "study_id": "study-current-001",
        "study_description": "CT chest with contrast",
        "study_date": "2024-02-15"
      },
      "prior_studies": [
        {
          "study_id": "study-prior-001",
          "study_description": "XR chest 2 views",
          "study_date": "2023-10-20"
        }
      ]
    }
  ]
}
```

Response:

```json
{
  "predictions": [
    {
      "case_id": "case-demo-001",
      "study_id": "study-prior-001",
      "predicted_is_relevant": true
    }
  ]
}
```

Malformed case objects return HTTP 400 responses instead of being silently
skipped.

## Sample Dataset

`sample_relevant_priors.json` is a lightweight demo dataset committed to the
repository. It is safe to use for local smoke tests and has the same shape as the
larger local evaluation artifact.

The larger `relevant_priors_public.json` file remains local-only and should not
be committed. It is intentionally listed in `.gitignore`.

## Testing

```bash
python -m unittest discover
```

## Deployment

The service is compatible with Render-style deployment:

- `Procfile` starts Uvicorn with `api:app`.
- `runtime.txt` declares the Python runtime.
- The web process uses `$PORT`, allowing the platform to assign the public port.

## Design Decisions

Embeddings were explored for semantic matching because they can capture related
study descriptions that do not share exact words. They were excluded from
production inference in this version because request-time model loading and
embedding generation increased latency and reliability risk on constrained
deployment environments.

The production path uses deterministic feature-based scoring instead. This keeps
inference predictable, fast, easy to test, and stable for lightweight hosting.

## Evaluation Summary

| Version | Accuracy | Notes |
| --- | --- | --- |
| Baseline Rules | ~0.83 | Initial heuristic model |
| Feature-Based Scoring | ~0.88 | Added normalization and overlap scoring |
| Embedding-Augmented | ~0.88+ | Improved semantic matching but higher latency |
| Final Production Model | ~0.88 | Optimized for low-latency deployment |

## Additional Notes

Prior relevance has an asymmetric cost profile: missing a relevant prior can
hide important clinical context, while surfacing an irrelevant prior mainly adds
review burden. Threshold selection should consider that tradeoff alongside
latency, because clinical tools need both reliable coverage and responsive
request handling.
