from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

import api


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)

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
                        {
                            "study_id": "prior-2",
                            "study_description": "MRI brain",
                            "study_date": "2022-01-01",
                        },
                    ],
                },
                {
                    "case_id": "case-2",
                    "current_study": {
                        "study_id": "current-2",
                        "study_description": "CT abdomen",
                        "study_date": "2024-01-01",
                    },
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
        self.assertEqual(len(body["predictions"]), 3)
        self.assertEqual(
            set(body["predictions"][0]),
            {"case_id", "study_id", "predicted_is_relevant"},
        )
        self.assertIs(type(body["predictions"][0]["predicted_is_relevant"]), bool)

    def test_predict_handles_missing_cases(self) -> None:
        response = self.client.post("/predict", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predictions": []})

    def test_predict_rejects_non_list_cases(self) -> None:
        response = self.client.post("/predict", json={"cases": "bad"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("cases", response.json()["detail"])

    def test_predict_rejects_missing_current_study(self) -> None:
        response = self.client.post(
            "/predict",
            json={"cases": [{"case_id": "case-1", "prior_studies": []}]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("current_study", response.json()["detail"])

    def test_predict_rejects_invalid_prior_instead_of_skipping(self) -> None:
        response = self.client.post(
            "/predict",
            json={
                "cases": [
                    {
                        "case_id": "case-1",
                        "current_study": {
                            "study_id": "current-1",
                            "study_description": "CT chest",
                            "study_date": "2024-01-01",
                        },
                        "prior_studies": [None],
                    }
                ]
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("prior", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
