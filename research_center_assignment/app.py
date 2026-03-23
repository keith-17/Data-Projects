"""
FastAPI service: research center quality tier from K-Means clusters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "final_kmeans_pipeline.pkl"
CSV_PATH = BASE_DIR / "research_centers.csv"

FEATURE_COLS = [
    "internalFacilitiesCount",
    "hospitals_10km",
    "pharmacies_10km",
    "facilityDiversity_10km",
    "facilityDensity_10km",
]

_pipeline: Pipeline | None = None
_cluster_to_tier: dict[int, str] | None = None


def _train_default_pipeline(df: pd.DataFrame) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), FEATURE_COLS)],
        remainder="drop",
    )
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    pipe = Pipeline([("preprocessor", preprocessor), ("kmeans", kmeans)])
    pipe.fit(df)
    return pipe


def _tier_mapping_from_centroids(pipe: Pipeline) -> dict[int, str]:
    """Map K-Means label → tier by ascending centroid strength (low → Basic)."""
    km = pipe.named_steps["kmeans"]
    centers = np.asarray(km.cluster_centers_)
    scores = centers.sum(axis=1)
    order = np.argsort(scores)
    tiers = ["Basic", "Standard", "Premium"]
    return {int(cluster_id): tiers[rank] for rank, cluster_id in enumerate(order)}


def _confidence_from_distances(pipe: Pipeline, X: pd.DataFrame) -> float:
    pre = pipe.named_steps["preprocessor"]
    km = pipe.named_steps["kmeans"]
    Xp = pre.transform(X)
    dists = km.transform(Xp)[0]
    order = np.argsort(dists)
    d1 = float(dists[order[0]])
    d2 = float(dists[order[1]])
    return float(np.clip(1.0 - d1 / (d1 + d2 + 1e-12), 0.0, 1.0))


def _load_or_train() -> tuple[Pipeline, dict[int, str]]:
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    pipe: Pipeline | None = None
    if MODEL_PATH.exists():
        try:
            candidate = joblib.load(MODEL_PATH)
            probe = pd.DataFrame([{c: 0 for c in FEATURE_COLS}])
            candidate.predict(probe)
            pipe = candidate
        except Exception:
            pipe = None
    if pipe is None:
        pipe = _train_default_pipeline(df)
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, MODEL_PATH)

    mapping = _tier_mapping_from_centroids(pipe)
    return pipe, mapping


def _ensure_model() -> None:
    global _pipeline, _cluster_to_tier
    if _pipeline is None:
        _pipeline, _cluster_to_tier = _load_or_train()


app = FastAPI(title="Research Center Quality Classifier")


class CenterFeatures(BaseModel):
    internalFacilitiesCount: int = Field(ge=0)
    hospitals_10km: int = Field(ge=0)
    pharmacies_10km: int = Field(ge=0)
    facilityDiversity_10km: float = Field(ge=0, le=1)
    facilityDensity_10km: float = Field(ge=0)


class PredictResponse(BaseModel):
    predictedCategory: str
    confidence: float


class BatchPredictRequest(BaseModel):
    centers: list[CenterFeatures] = Field(min_length=1)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


def _predict_row(features: CenterFeatures) -> PredictResponse:
    _ensure_model()
    assert _pipeline is not None and _cluster_to_tier is not None
    X = pd.DataFrame([features.model_dump()])
    cluster = int(_pipeline.predict(X)[0])
    tier = _cluster_to_tier[cluster]
    conf = _confidence_from_distances(_pipeline, X)
    return PredictResponse(predictedCategory=tier, confidence=conf)


@app.get("/health")
def health() -> dict[str, Any]:
    _ensure_model()
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(body: CenterFeatures) -> PredictResponse:
    return _predict_row(body)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest) -> BatchPredictResponse:
    preds = [_predict_row(c) for c in body.centers]
    return BatchPredictResponse(predictions=preds)
