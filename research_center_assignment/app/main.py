import logging

from fastapi import FastAPI

from app.config import settings
from app.model import get_model_service
from app.schemas import CenterFeatures, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Research Center Classifier")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": settings.model_version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(features: CenterFeatures):
    logger.info("Prediction request received")
    model_service = get_model_service()
    tier, confidence = model_service.predict(features.model_dump())

    return PredictResponse(
        predictedCategory=tier,
        confidence=confidence,
        model_version=settings.model_version,
    )