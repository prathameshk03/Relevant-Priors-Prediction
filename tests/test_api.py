from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

import api


class FakeEncoder:
    def __init__(self) -> None:
        self.encoded_sentences: list[list[str]] = []

    def encode(self, sentences, batch_size=64, show_progress_bar=False):
        self.encoded_sentences.append(list(sentences))
        return [[float(len(sentence)), 1.0] for sentence in sentences]


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = FakeEncoder()
        api._encoder = self.encoder
        self.client = TestClient(api.app)

    def tearDown(self) -> None:
        api._encoder = None

    def test_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_returns_exact_schema_and_valid_prior_count(self) -> None:
        payload = {
            "cases": [
                {
                    "case_id": "case-1",
                    "current_study": {
                        "study_id": "current-1",
                        "study_description": "CT chest",
                        "study_date": "2024-01-01",
                    },
                    "prior_studies": [
                        {
                            "study_id": "prior-1",
                            "study_description": "XR chest",
                            "study_date": "2023-01-01",
                        },
                        None,
                        {
                            "study_id": "prior-2",
                            "study_description": "MRI brain",
                            "study_date": "2022-01-01",
                        },
                    ],
                },
                {
                    "case_id": "case-2",
                    "prior_studies": [
                        {
                            "study_id": "prior-3",
                            "study_description": "CT abdomen",
                            "study_date": "2023-01-01",
                        }
                    ],
                },
            ]
        }

        response = self.client.post("/predict", json=payload)
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(body), {"predictions"})
        self.assertEqual(len(body["predictions"]), 2)
        self.assertEqual(
            set(body["predictions"][0]),
            {"case_id", "study_id", "predicted_is_relevant"},
        )
        self.assertIs(type(body["predictions"][0]["predicted_is_relevant"]), bool)

    def test_predict_handles_missing_cases(self) -> None:
        response = self.client.post("/predict", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predictions": []})

    def test_predict_encodes_unique_descriptions_once_per_request(self) -> None:
        payload = {
            "cases": [
                {
                    "case_id": "case-1",
                    "current_study": {
                        "study_id": "current-1",
                        "study_description": "CT chest",
                        "study_date": "2024-01-01",
                    },
                    "prior_studies": [
                        {
                            "study_id": "prior-1",
                            "study_description": "CT chest",
                            "study_date": "2023-01-01",
                        },
                        {
                            "study_id": "prior-2",
                            "study_description": "XR ankle",
                            "study_date": "2022-01-01",
                        },
                    ],
                }
            ]
        }

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.encoder.encoded_sentences, [["CT chest", "XR ankle"]])


if __name__ == "__main__":
    unittest.main()
