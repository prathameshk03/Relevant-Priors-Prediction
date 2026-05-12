from __future__ import annotations

import unittest

from src.evaluate import build_truth_map, evaluate_predictions
from src.features import extract_body_part, extract_modality, keyword_overlap_score, recency_score
from src.model import THRESHOLD, predict_pair, score_pair
from src.preprocess import normalize_description


class PipelineTests(unittest.TestCase):
    def test_normalize_description_expands_abbreviations(self) -> None:
        self.assertEqual(
            normalize_description("xr chest w/o cntrst"),
            "X RAY CHEST WITHOUT CONTRAST",
        )

    def test_extract_modality(self) -> None:
        self.assertEqual(extract_modality("MRI brain"), "MRI")
        self.assertEqual(extract_modality("CT chest"), "CT")
        self.assertEqual(extract_modality("XR ankle"), "XRAY")
        self.assertEqual(extract_modality("US abdomen"), "ULTRASOUND")
        self.assertEqual(extract_modality("NM myo perf"), "UNKNOWN")

    def test_extract_body_part(self) -> None:
        self.assertEqual(extract_body_part("CT head"), "BRAIN")
        self.assertEqual(extract_body_part("XR chest"), "CHEST")
        self.assertEqual(extract_body_part("MR lumbar spine"), "SPINE")
        self.assertEqual(extract_body_part("CT coronary calc screening"), "HEART")
        self.assertEqual(extract_body_part("VAS venous doppler LE BI"), "VASCULAR")
        self.assertEqual(extract_body_part("screening mammogram"), "BREAST")
        self.assertEqual(extract_body_part("NM myo perf"), "UNKNOWN")

    def test_keyword_overlap_score(self) -> None:
        self.assertGreater(keyword_overlap_score("CT chest contrast", "XR chest 1 view"), 0)
        self.assertEqual(keyword_overlap_score("MRI brain", "XR ankle"), 0)

    def test_score_pair_uses_final_rule_weights(self) -> None:
        score = score_pair(
            {"study_description": "CT chest", "study_date": "2024-01-01"},
            {"study_description": "XR chest", "study_date": "2023-08-01"},
        )
        expected = 0.45 + (0.20 * 0.5) + 0.20 + (0.15 * (1 / 3))
        self.assertAlmostEqual(score, expected)

    def test_predict_pair_uses_deterministic_feature_scoring(self) -> None:
        self.assertFalse(
            predict_pair(
                {"study_description": "MRI brain", "study_date": "2024-01-01"},
                {"study_description": "XR ankle", "study_date": "2010-01-01"},
            )
        )

    def test_predict_pair_uses_lower_threshold(self) -> None:
        current_study = {"study_description": "CT chest", "study_date": "2024-01-01"}
        prior_study = {"study_description": "XR chest", "study_date": "2023-08-01"}
        score = score_pair(current_study, prior_study)

        self.assertEqual(THRESHOLD, 0.50)
        self.assertEqual(predict_pair(current_study, prior_study), score >= THRESHOLD)

    def test_recency_score_buckets(self) -> None:
        self.assertEqual(recency_score("2024-01-01", "2023-01-01"), 1.0)
        self.assertEqual(recency_score("2024-01-01", "2020-01-01"), 0.7)
        self.assertEqual(recency_score("2024-01-01", "2016-01-01"), 0.4)
        self.assertEqual(recency_score("2024-01-01", "2010-01-01"), 0.1)
        self.assertEqual(recency_score("bad-date", "2010-01-01"), 0.1)

    def test_evaluate_predictions_counts_metrics(self) -> None:
        truth_map = build_truth_map(
            [
                {"case_id": "c1", "study_id": "s1", "is_relevant_to_current": True},
                {"case_id": "c1", "study_id": "s2", "is_relevant_to_current": False},
            ]
        )
        result = evaluate_predictions(
            [
                {"case_id": "c1", "study_id": "s1", "predicted_is_relevant": True},
                {"case_id": "c1", "study_id": "s2", "predicted_is_relevant": True},
                {"case_id": "c1", "study_id": "s3", "predicted_is_relevant": False},
            ],
            truth_map,
        )

        self.assertEqual(result.total_predictions, 3)
        self.assertAlmostEqual(result.accuracy, 1 / 3)
        self.assertEqual(result.false_positive_count, 1)
        self.assertEqual(result.false_negative_count, 0)
        self.assertEqual(result.missing_truth_count, 1)


if __name__ == "__main__":
    unittest.main()
