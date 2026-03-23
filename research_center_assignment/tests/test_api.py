from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def make_mock_model_service(category="Standard", confidence=0.73):
    mock_service = MagicMock()
    mock_service.predict.return_value = (category, confidence)
    return mock_service


def test_health():
    mock_service = make_mock_model_service()

    with patch("app.main.get_model_service", return_value=mock_service):
        from app import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "model_version" in body


def test_predict():
    mock_service = make_mock_model_service(category="Premium", confidence=0.91)

    with patch("app.main.get_model_service", return_value=mock_service):
        from app import app

        client = TestClient(app)
        payload = {
            "internalFacilitiesCount": 9,
            "hospitals_10km": 3,
            "pharmacies_10km": 2,
            "facilityDiversity_10km": 0.8,
            "facilityDensity_10km": 0.5,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["predictedCategory"] == "Premium"
        assert body["confidence"] == 0.91
        assert "model_version" in body


def test_predict_invalid_input():
    mock_service = make_mock_model_service()

    with patch("app.main.get_model_service", return_value=mock_service):
        from app import app

        client = TestClient(app)
        payload = {
            "internalFacilitiesCount": -1,
            "hospitals_10km": 3,
            "pharmacies_10km": 2,
            "facilityDiversity_10km": 0.8,
            "facilityDensity_10km": 0.5,
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 422
