import logging
from typing import Any

from fastapi import FastAPI

from app.config import settings
from app.model import get_model_service
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CenterFeatures,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Research Center Classifier")


def _build_prediction(features: CenterFeatures) -> PredictResponse:
    logger.info("Prediction request received")
    model_service = get_model_service()
    tier, confidence = model_service.predict(features.model_dump())

    return PredictResponse(
        predictedCategory=tier,
        confidence=confidence,
        model_version=settings.model_version,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    get_model_service()
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": settings.model_version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(features: CenterFeatures) -> PredictResponse:
    return _build_prediction(features)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest) -> BatchPredictResponse:
    predictions = [_build_prediction(center) for center in body.centers]
    return BatchPredictResponse(predictions=predictions)
