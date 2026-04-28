"""Evaluation helpers for the relevant-priors baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    total_predictions: int
    predicted_true_count: int
    actual_true_count: int
    false_positive_count: int
    false_negative_count: int
    missing_truth_count: int

    @property
    def predicted_true_percent(self) -> float:
        return _percent(self.predicted_true_count, self.total_predictions)

    @property
    def actual_true_percent(self) -> float:
        return _percent(self.actual_true_count, self.total_predictions)


def build_truth_map(truth_rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], bool]:
    """Build a fast lookup keyed by (case_id, study_id)."""
    truth_map: dict[tuple[str, str], bool] = {}
    for row in truth_rows:
        case_id = str(row["case_id"])
        study_id = str(row["study_id"])
        truth_map[(case_id, study_id)] = bool(row["is_relevant_to_current"])
    return truth_map


def flatten_cases(cases: Iterable[dict[str, Any]]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Flatten nested cases into (case_id, current_study, prior_study) pairs."""
    pairs = []
    for case in cases:
        case_id = str(case["case_id"])
        current_study = case["current_study"]
        for prior_study in case.get("prior_studies", []):
            pairs.append((case_id, current_study, prior_study))
    return pairs


def evaluate_predictions(
    predictions: Iterable[dict[str, Any]],
    truth_map: dict[tuple[str, str], bool],
) -> EvaluationResult:
    """Evaluate predictions; missing truth labels are counted as incorrect."""
    correct = 0
    total = 0
    predicted_true = 0
    actual_true = 0
    false_positive = 0
    false_negative = 0
    missing_truth = 0

    for prediction in predictions:
        total += 1
        case_id = str(prediction["case_id"])
        study_id = str(prediction["study_id"])
        predicted = bool(prediction["predicted_is_relevant"])
        actual = truth_map.get((case_id, study_id))

        if predicted:
            predicted_true += 1

        if actual is None:
            missing_truth += 1
            continue

        if actual:
            actual_true += 1

        if predicted == actual:
            correct += 1
        elif predicted and not actual:
            false_positive += 1
        elif not predicted and actual:
            false_negative += 1

    accuracy = correct / total if total else 0.0
    return EvaluationResult(
        accuracy=accuracy,
        total_predictions=total,
        predicted_true_count=predicted_true,
        actual_true_count=actual_true,
        false_positive_count=false_positive,
        false_negative_count=false_negative,
        missing_truth_count=missing_truth,
    )


def format_metrics(result: EvaluationResult) -> str:
    """Format evaluation metrics for CLI output."""
    lines = [
        f"Accuracy: {result.accuracy:.4f}",
        f"Total predictions: {result.total_predictions}",
        f"Predicted true: {result.predicted_true_percent:.2f}%",
        f"Actual true: {result.actual_true_percent:.2f}%",
        f"False positives: {result.false_positive_count}",
        f"False negatives: {result.false_negative_count}",
    ]
    if result.missing_truth_count:
        lines.append(f"Missing truth labels: {result.missing_truth_count}")
    return "\n".join(lines)


def _percent(numerator: int, denominator: int) -> float:
    return 100 * numerator / denominator if denominator else 0.0
