"""
Legacy regression checks for the Research Center Quality Classifier API.

Run with:
    pip install -r requirements.txt
    pytest tests/test_legacy_app_checks.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

PREMIUM_PAYLOAD = {
    "internalFacilitiesCount": 12,
    "hospitals_10km": 5,
    "pharmacies_10km": 4,
    "facilityDiversity_10km": 0.95,
    "facilityDensity_10km": 0.85,
}

STANDARD_PAYLOAD = {
    "internalFacilitiesCount": 6,
    "hospitals_10km": 2,
    "pharmacies_10km": 2,
    "facilityDiversity_10km": 0.55,
    "facilityDensity_10km": 0.40,
}

BASIC_PAYLOAD = {
    "internalFacilitiesCount": 1,
    "hospitals_10km": 0,
    "pharmacies_10km": 0,
    "facilityDiversity_10km": 0.10,
    "facilityDensity_10km": 0.05,
}


def _make_mock_model_service(category: str = "Standard", confidence: float = 0.73):
    mock_service = MagicMock()
    mock_service.predict.return_value = (category, confidence)
    return mock_service


@pytest.fixture
def client():
    mock_service = _make_mock_model_service(category="Standard", confidence=0.73)

    with patch("app.main.get_model_service", return_value=mock_service):
        from app import app

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def basic_client():
    mock_service = _make_mock_model_service(category="Basic", confidence=0.81)

    with patch("app.main.get_model_service", return_value=mock_service):
        from app import app

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def premium_client():
    mock_service = _make_mock_model_service(category="Premium", confidence=0.91)

    with patch("app.main.get_model_service", return_value=mock_service):
        from app import app

        with TestClient(app) as test_client:
            yield test_client


class TestHealth:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_ok(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_model_loaded_true(self, client):
        response = client.get("/health")
        assert response.json()["model_loaded"] is True

    def test_model_version_present(self, client):
        response = client.get("/health")
        assert "model_version" in response.json()


class TestPredict:
    def test_returns_200(self, client):
        response = client.post("/predict", json=STANDARD_PAYLOAD)
        assert response.status_code == 200

    def test_response_has_predicted_category(self, client):
        response = client.post("/predict", json=STANDARD_PAYLOAD)
        assert "predictedCategory" in response.json()

    def test_response_has_confidence(self, client):
        response = client.post("/predict", json=STANDARD_PAYLOAD)
        assert "confidence" in response.json()

    def test_response_has_model_version(self, client):
        response = client.post("/predict", json=STANDARD_PAYLOAD)
        assert "model_version" in response.json()

    def test_confidence_is_between_0_and_1(self, client):
        response = client.post("/predict", json=STANDARD_PAYLOAD)
        assert 0.0 <= response.json()["confidence"] <= 1.0

    def test_valid_tier_string(self, client):
        response = client.post("/predict", json=STANDARD_PAYLOAD)
        assert response.json()["predictedCategory"] in ("Premium", "Standard", "Basic")

    def test_predicts_standard(self, client):
        response = client.post("/predict", json=STANDARD_PAYLOAD)
        assert response.json()["predictedCategory"] == "Standard"

    def test_predicts_basic(self, basic_client):
        response = basic_client.post("/predict", json=BASIC_PAYLOAD)
        assert response.json()["predictedCategory"] == "Basic"

    def test_predicts_premium(self, premium_client):
        response = premium_client.post("/predict", json=PREMIUM_PAYLOAD)
        assert response.json()["predictedCategory"] == "Premium"


class TestPredictValidation:
    def test_rejects_diversity_above_1(self, client):
        bad = {**STANDARD_PAYLOAD, "facilityDiversity_10km": 1.5}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_rejects_negative_facilities(self, client):
        bad = {**STANDARD_PAYLOAD, "internalFacilitiesCount": -1}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_rejects_negative_hospitals(self, client):
        bad = {**STANDARD_PAYLOAD, "hospitals_10km": -1}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_rejects_negative_pharmacies(self, client):
        bad = {**STANDARD_PAYLOAD, "pharmacies_10km": -1}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_rejects_negative_density(self, client):
        bad = {**STANDARD_PAYLOAD, "facilityDensity_10km": -0.1}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_rejects_missing_field(self, client):
        bad = {k: v for k, v in STANDARD_PAYLOAD.items() if k != "hospitals_10km"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_rejects_empty_body(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_rejects_string_for_int_field(self, client):
        bad = {**STANDARD_PAYLOAD, "internalFacilitiesCount": "many"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_accepts_diversity_exactly_0(self, client):
        edge = {**STANDARD_PAYLOAD, "facilityDiversity_10km": 0.0}
        response = client.post("/predict", json=edge)
        assert response.status_code == 200

    def test_accepts_diversity_exactly_1(self, client):
        edge = {**STANDARD_PAYLOAD, "facilityDiversity_10km": 1.0}
        response = client.post("/predict", json=edge)
        assert response.status_code == 200

    def test_accepts_zero_counts(self, client):
        edge = {**STANDARD_PAYLOAD, "hospitals_10km": 0, "pharmacies_10km": 0}
        response = client.post("/predict", json=edge)
        assert response.status_code == 200


class TestBatchPredict:
    def test_returns_200(self, client):
        response = client.post("/predict/batch", json={"centers": [STANDARD_PAYLOAD, BASIC_PAYLOAD]})
        assert response.status_code == 200

    def test_response_has_predictions_list(self, client):
        response = client.post("/predict/batch", json={"centers": [STANDARD_PAYLOAD]})
        assert "predictions" in response.json()
        assert isinstance(response.json()["predictions"], list)

    def test_returns_correct_count(self, client):
        response = client.post(
            "/predict/batch",
            json={"centers": [STANDARD_PAYLOAD, STANDARD_PAYLOAD, STANDARD_PAYLOAD]},
        )
        assert len(response.json()["predictions"]) == 3

    def test_each_prediction_has_category_confidence_and_version(self, client):
        response = client.post("/predict/batch", json={"centers": [STANDARD_PAYLOAD]})
        prediction = response.json()["predictions"][0]
        assert "predictedCategory" in prediction
        assert "confidence" in prediction
        assert "model_version" in prediction

    def test_single_center_batch(self, client):
        response = client.post("/predict/batch", json={"centers": [PREMIUM_PAYLOAD]})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 1


class TestBatchPredictValidation:
    def test_rejects_empty_centers_list(self, client):
        response = client.post("/predict/batch", json={"centers": []})
        assert response.status_code == 422

    def test_rejects_invalid_center_in_batch(self, client):
        bad_center = {**STANDARD_PAYLOAD, "facilityDiversity_10km": 99.0}
        response = client.post("/predict/batch", json={"centers": [bad_center]})
        assert response.status_code == 422

    def test_rejects_missing_centers_key(self, client):
        response = client.post("/predict/batch", json={})
        assert response.status_code == 422
