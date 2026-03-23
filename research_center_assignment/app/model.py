import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.config import settings


class ModelService:
    def __init__(self):
        self.pipeline: Pipeline = joblib.load(settings.model_path)
        self.cluster_map = self._build_cluster_map()

    def _build_cluster_map(self):
        km = self.pipeline.named_steps["kmeans"]
        centers = np.asarray(km.cluster_centers_)
        scores = centers.sum(axis=1)
        order = np.argsort(scores)
        tiers = ["Basic", "Standard", "Premium"]
        return {int(cid): tiers[i] for i, cid in enumerate(order)}

    def predict(self, data: dict):
        df = pd.DataFrame([data])
        cluster = int(self.pipeline.predict(df)[0])
        tier = self.cluster_map[cluster]
        confidence = self._confidence(df)
        return tier, confidence

    def _confidence(self, df):
        pre = self.pipeline.named_steps["preprocessor"]
        km = self.pipeline.named_steps["kmeans"]
        Xp = pre.transform(df)
        dists = km.transform(Xp)[0]
        d1, d2 = sorted(dists)[:2]
        return float(np.clip(1 - d1 / (d1 + d2 + 1e-12), 0, 1))


_model_service = None


def get_model_service():
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service