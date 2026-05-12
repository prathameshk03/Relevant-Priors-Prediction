"""Run batch prediction and evaluation for Relevant-Priors-Prediction-API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.evaluate import build_truth_map, evaluate_predictions, flatten_cases, format_metrics
from src.model import predict_pair


DEFAULT_INPUT_PATH = Path("relevant_priors_public.json")


def load_dataset_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def predict_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for case_id, current_study, prior_study in flatten_cases(cases):
        predictions.append(
            {
                "case_id": case_id,
                "study_id": str(prior_study["study_id"]),
                "predicted_is_relevant": predict_pair(current_study, prior_study),
            }
        )
    return predictions


def run(input_path: Path) -> str:
    payload = load_dataset_file(input_path)
    cases = payload.get("cases", [])
    truth_rows = payload.get("truth", [])

    predictions = predict_cases(cases)
    truth_map = build_truth_map(truth_rows)
    result = evaluate_predictions(predictions, truth_map)
    return format_metrics(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Relevant-Priors-Prediction-API pipeline.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to a JSON dataset containing cases and optional truth labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        print(run(Path(args.input_path)))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
