"""
Tests for the Research Center Quality Classifier API.

Run with:
    pip install pytest httpx
    pytest test_app.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


def _make_mock_pipeline(cluster_id: int = 0):
    """
    Returns a mock sklearn pipeline that always predicts cluster_id.
    The kmeans step has 3 centroids with clear quality ordering:
      cluster 0 = low quality  → Basic
      cluster 1 = mid quality  → Standard
      cluster 2 = high quality → Premium
    """
    mock_pipeline = MagicMock()

    # cluster_centers_ shape (3, 5) — ordered so cluster 0 < 1 < 2 in quality
    centers = np.array([
        [1.0, 0.0, 0.0, 0.10, 0.05],   # cluster 0 — Basic
        [6.0, 2.0, 2.0, 0.55, 0.40],   # cluster 1 — Standard
        [12.0, 5.0, 4.0, 0.95, 0.85],  # cluster 2 — Premium
    ])
    mock_pipeline.named_steps = {
        "kmeans": MagicMock(cluster_centers_=centers),
        "preprocessor": MagicMock(),
    }

    mock_pipeline.predict.return_value = np.array([cluster_id])

    # transform returns distances: assigned cluster gets 0.0, others get 1.0
    distances = np.ones((1, 3))
    distances[0, cluster_id] = 0.0
    mock_pipeline.named_steps["preprocessor"].transform.return_value = np.zeros((1, 5))
    mock_pipeline.named_steps["kmeans"].transform.return_value = distances

    return mock_pipeline


@pytest.fixture
def client():
    """TestClient with a mocked model (cluster 1 = Standard by default)."""
    mock_pipeline = _make_mock_pipeline(cluster_id=1)

    with patch("app.MODEL_PATH") as mock_path, \
         patch("app.joblib.load", return_value=mock_pipeline):
        mock_path.exists.return_value = True

        from app import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def basic_client():
    mock_pipeline = _make_mock_pipeline(cluster_id=0)
    with patch("app.MODEL_PATH") as mock_path, \
         patch("app.joblib.load", return_value=mock_pipeline):
        mock_path.exists.return_value = True
        from app import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def premium_client():
    mock_pipeline = _make_mock_pipeline(cluster_id=2)
    with patch("app.MODEL_PATH") as mock_path, \
         patch("app.joblib.load", return_value=mock_pipeline):
        mock_path.exists.return_value = True
        from app import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_model_loaded_true(self, client):
        r = client.get("/health")
        assert r.json()["model_loaded"] is True


# ---------------------------------------------------------------------------
# Single prediction — happy path
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_200(self, client):
        r = client.post("/predict", json=STANDARD_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_predicted_category(self, client):
        r = client.post("/predict", json=STANDARD_PAYLOAD)
        assert "predictedCategory" in r.json()

    def test_response_has_confidence(self, client):
        r = client.post("/predict", json=STANDARD_PAYLOAD)
        assert "confidence" in r.json()

    def test_confidence_is_between_0_and_1(self, client):
        r = client.post("/predict", json=STANDARD_PAYLOAD)
        assert 0.0 <= r.json()["confidence"] <= 1.0

    def test_valid_tier_string(self, client):
        r = client.post("/predict", json=STANDARD_PAYLOAD)
        assert r.json()["predictedCategory"] in ("Premium", "Standard", "Basic")

    def test_predicts_standard(self, client):
        r = client.post("/predict", json=STANDARD_PAYLOAD)
        assert r.json()["predictedCategory"] == "Standard"

    def test_predicts_basic(self, basic_client):
        r = basic_client.post("/predict", json=BASIC_PAYLOAD)
        assert r.json()["predictedCategory"] == "Basic"

    def test_predicts_premium(self, premium_client):
        r = premium_client.post("/predict", json=PREMIUM_PAYLOAD)
        assert r.json()["predictedCategory"] == "Premium"


# ---------------------------------------------------------------------------
# Single prediction — validation errors (422)
# ---------------------------------------------------------------------------

class TestPredictValidation:
    def test_rejects_diversity_above_1(self, client):
        bad = {**STANDARD_PAYLOAD, "facilityDiversity_10km": 1.5}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_rejects_negative_facilities(self, client):
        bad = {**STANDARD_PAYLOAD, "internalFacilitiesCount": -1}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_rejects_negative_hospitals(self, client):
        bad = {**STANDARD_PAYLOAD, "hospitals_10km": -1}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_rejects_negative_pharmacies(self, client):
        bad = {**STANDARD_PAYLOAD, "pharmacies_10km": -1}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_rejects_negative_density(self, client):
        bad = {**STANDARD_PAYLOAD, "facilityDensity_10km": -0.1}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_rejects_missing_field(self, client):
        bad = {k: v for k, v in STANDARD_PAYLOAD.items() if k != "hospitals_10km"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_rejects_empty_body(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 422

    def test_rejects_string_for_int_field(self, client):
        bad = {**STANDARD_PAYLOAD, "internalFacilitiesCount": "many"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_accepts_diversity_exactly_0(self, client):
        edge = {**STANDARD_PAYLOAD, "facilityDiversity_10km": 0.0}
        r = client.post("/predict", json=edge)
        assert r.status_code == 200

    def test_accepts_diversity_exactly_1(self, client):
        edge = {**STANDARD_PAYLOAD, "facilityDiversity_10km": 1.0}
        r = client.post("/predict", json=edge)
        assert r.status_code == 200

    def test_accepts_zero_counts(self, client):
        edge = {**STANDARD_PAYLOAD, "hospitals_10km": 0, "pharmacies_10km": 0}
        r = client.post("/predict", json=edge)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Batch prediction — happy path
# ---------------------------------------------------------------------------

class TestBatchPredict:
    def test_returns_200(self, client):
        r = client.post("/predict/batch", json={"centers": [STANDARD_PAYLOAD, BASIC_PAYLOAD]})
        assert r.status_code == 200

    def test_response_has_predictions_list(self, client):
        r = client.post("/predict/batch", json={"centers": [STANDARD_PAYLOAD]})
        assert "predictions" in r.json()
        assert isinstance(r.json()["predictions"], list)

    def test_returns_correct_count(self, client):
        r = client.post("/predict/batch", json={"centers": [STANDARD_PAYLOAD, STANDARD_PAYLOAD, STANDARD_PAYLOAD]})
        assert len(r.json()["predictions"]) == 3

    def test_each_prediction_has_category_and_confidence(self, client):
        r = client.post("/predict/batch", json={"centers": [STANDARD_PAYLOAD]})
        pred = r.json()["predictions"][0]
        assert "predictedCategory" in pred
        assert "confidence" in pred

    def test_single_center_batch(self, client):
        r = client.post("/predict/batch", json={"centers": [PREMIUM_PAYLOAD]})
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 1


# ---------------------------------------------------------------------------
# Batch prediction — validation errors
# ---------------------------------------------------------------------------

class TestBatchPredictValidation:
    def test_rejects_empty_centers_list(self, client):
        r = client.post("/predict/batch", json={"centers": []})
        assert r.status_code == 422

    def test_rejects_invalid_center_in_batch(self, client):
        bad_center = {**STANDARD_PAYLOAD, "facilityDiversity_10km": 99.0}
        r = client.post("/predict/batch", json={"centers": [bad_center]})
        assert r.status_code == 422

    def test_rejects_missing_centers_key(self, client):
        r = client.post("/predict/batch", json={})
        assert r.status_code == 422
