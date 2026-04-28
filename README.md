# New-Lantern-API-Task

Phase 1 baseline for the Relevant Priors challenge.

## Run

Install dependencies first:

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

By default, the CLI reads `relevant_priors_public.json`, builds an in-memory
sentence-transformers embedding cache for unique study descriptions, predicts
relevance for every prior study in every case, and prints local evaluation metrics
against the embedded truth labels.

## Data

Place the challenge dataset at the repo root as `relevant_priors_public.json`.
This file is intentionally not committed because it is a local challenge/data
artifact.

You can also pass another compatible combined challenge JSON file:

```bash
python main.py path/to/challenge.json
```

## Test

```bash
python -m unittest discover
```
