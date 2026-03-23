from pydantic import BaseModel, Field


class CenterFeatures(BaseModel):
    internalFacilitiesCount: int = Field(..., ge=0, example=9)
    hospitals_10km: int = Field(..., ge=0, example=3)
    pharmacies_10km: int = Field(..., ge=0, example=2)
    facilityDiversity_10km: float = Field(..., ge=0, le=1, example=0.8)
    facilityDensity_10km: float = Field(..., ge=0, example=0.5)


class PredictResponse(BaseModel):
    predictedCategory: str
    confidence: float
    model_version: str


class BatchPredictRequest(BaseModel):
    centers: list[CenterFeatures] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
