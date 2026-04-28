# New-Lantern-API-Task

Fast rule-based API for the Relevant Priors challenge.

## Run

Install dependencies first:

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

By default, the CLI reads `relevant_priors_public.json`, builds an in-memory
set of deterministic features for every current/prior study pair, predicts
relevance, and prints local evaluation metrics against the embedded truth labels.

The shipped model does not load embeddings. It scores each pair using body-part
match, modality match, recency, and keyword overlap so the API can respond quickly
on free-tier deployment environments.

## Data

Place the challenge dataset at the repo root as `relevant_priors_public.json`.
This file is intentionally not committed because it is a local challenge/data
artifact.

You can also pass another compatible combined challenge JSON file:

```bash
python main.py path/to/challenge.json
```

## API

Run the FastAPI prediction service:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open the interactive docs at `http://localhost:8000/docs`.

`POST /predict` accepts a combined challenge-style JSON body and returns:

```json
{
  "predictions": [
    {
      "case_id": "...",
      "study_id": "...",
      "predicted_is_relevant": true
    }
  ]
}
```

Malformed case objects return HTTP 400 responses instead of being silently
skipped.

## Test

```bash
python -m unittest discover
```
