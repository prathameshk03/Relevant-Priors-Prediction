from __future__ import annotations

import unittest

from src.embeddings import EmbeddingCache, cosine_similarity
from src.evaluate import build_truth_map, evaluate_predictions
from src.features import extract_body_part, extract_modality, keyword_overlap_score, recency_score
from src.model import THRESHOLD, predict_pair, score_pair
from src.preprocess import normalize_description


class FakeEncoder:
    def __init__(self) -> None:
        self.encoded_sentences: list[list[str]] = []

    def encode(self, sentences, batch_size=64, show_progress_bar=False):
        self.encoded_sentences.append(list(sentences))
        return [[float(len(sentence)), 1.0] for sentence in sentences]


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

    def test_score_pair_uses_soft_modality_match(self) -> None:
        score = score_pair(
            {"study_description": "CT chest", "study_date": "2024-01-01"},
            {"study_description": "XR chest", "study_date": "2023-08-01"},
        )
        self.assertGreaterEqual(score, 0.40 + 0.10 * 0.5)

    def test_score_pair_includes_embedding_similarity(self) -> None:
        current_study = {"study_description": "CT chest", "study_date": "2024-01-01"}
        prior_study = {"study_description": "XR ankle", "study_date": "2023-08-01"}

        without_embedding = score_pair(current_study, prior_study, embedding_similarity=0.0)
        with_embedding = score_pair(current_study, prior_study, embedding_similarity=1.0)

        self.assertAlmostEqual(with_embedding - without_embedding, 0.30)

    def test_score_pair_adds_conditional_embedding_boost(self) -> None:
        current_study = {"study_description": "CT chest", "study_date": "2024-01-01"}
        prior_study = {"study_description": "XR chest", "study_date": "2023-08-01"}

        below_boost = score_pair(current_study, prior_study, embedding_similarity=0.84)
        with_boost = score_pair(current_study, prior_study, embedding_similarity=0.85)

        self.assertAlmostEqual(with_boost - below_boost, 0.103)

    def test_cosine_similarity_identical_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_cosine_similarity_normalizes_opposite_vectors_to_zero(self) -> None:
        self.assertEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), 0.0)

    def test_cosine_similarity_normalizes_orthogonal_vectors_to_half(self) -> None:
        self.assertEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.5)

    def test_predict_pair_does_not_use_high_confidence_embedding_shortcut(self) -> None:
        self.assertFalse(
            predict_pair(
                {"study_description": "MRI brain", "study_date": "2024-01-01"},
                {"study_description": "XR ankle", "study_date": "2010-01-01"},
                embedding_similarity=0.85,
            )
        )

    def test_predict_pair_uses_lower_threshold(self) -> None:
        current_study = {"study_description": "CT chest", "study_date": "2024-01-01"}
        prior_study = {"study_description": "XR chest", "study_date": "2023-08-01"}
        score = score_pair(current_study, prior_study, embedding_similarity=0.0)

        self.assertEqual(THRESHOLD, 0.50)
        self.assertEqual(predict_pair(current_study, prior_study, 0.0), score >= THRESHOLD)

    def test_embedding_cache_encodes_unique_descriptions_once(self) -> None:
        encoder = FakeEncoder()
        cache = EmbeddingCache(encoder, batch_size=2)

        cache.populate(["CT chest", "CT chest", "XR ankle"])

        self.assertEqual(encoder.encoded_sentences, [["CT chest", "XR ankle"]])
        self.assertEqual(cache.similarity("CT chest", "CT chest"), 1.0)
        self.assertGreaterEqual(cache.similarity("CT chest", "XR ankle"), 0.0)

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
