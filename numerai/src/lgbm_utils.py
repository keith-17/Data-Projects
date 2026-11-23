"""lgbm_utils.py — LightGBMNumeraiEstimator child class of BaseNumeraiEstimator with all LightGBM parameters exposed.

Shared feature extraction (`NumeraiExtractions`), correlation calculation (`compute_feature_target_correlations`),
search space helpers, and Numerai evaluation functions live in `utils.py`, and shared parent estimator logic
lives in `baseline_utils.py`.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from baseline_utils import BaseNumeraiEstimator
from utils import NumeraiExtractions, _decode_structured


class LightGBMNumeraiEstimator(BaseNumeraiEstimator):
    """Child Class — LightGBM Regressor inheriting `BaseNumeraiEstimator` with full parameter exposure."""

    def __init__(
        self,
        extractor: Optional[NumeraiExtractions] = None,
        # BaseNumeraiEstimator shared parameters
        top_k: Optional[int] = None,
        correlation_method: Optional[str] = None,
        abs_corr: bool = True,
        era_weighting: Optional[str] = None,
        recency_decay: Optional[float] = 0.005,
        prediction_transform: Optional[str] = None,
        bias_correction: Optional[str] = None,
        walk_forward_step_eras: Optional[int] = None,
        walk_forward_window: str = "expanding",
        target_horizon_eras: int = 4,
        era_col: str = "era",
        target_col: str = "target",
        verbose: int = 0,
        random_state: int = 42,
        # LightGBM sklearn API parameters
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        subsample_for_bin: int = 200000,
        objective: str = "regression",
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 0.1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        n_jobs: int = -1,
        importance_type: str = "gain",
        # LightGBM core tree / histogram / regularization / loss / DART parameters
        max_bin: int = 255,
        min_data_in_bin: int = 3,
        colsample_bynode: float = 1.0,
        extra_trees: bool = False,
        path_smooth: float = 0.0,
        max_delta_step: float = 0.0,
        linear_tree: bool = False,
        linear_lambda: float = 0.0,
        drop_rate: float = 0.1,
        max_drop: int = 50,
        skip_drop: float = 0.5,
        xgboost_dart_mode: bool = False,
        uniform_drop: bool = False,
        alpha: float = 0.9,
        fair_c: float = 1.0,
        feature_pre_filter: bool = True,
        force_col_wise: bool = False,
        force_row_wise: bool = False,
        lgbm_verbosity: int = -1,
    ):
        super().__init__(
            extractor=extractor,
            top_k=top_k,
            correlation_method=correlation_method,
            abs_corr=abs_corr,
            era_weighting=era_weighting,
            recency_decay=recency_decay,
            prediction_transform=prediction_transform,
            bias_correction=bias_correction,
            walk_forward_step_eras=walk_forward_step_eras,
            walk_forward_window=walk_forward_window,
            target_horizon_eras=target_horizon_eras,
            era_col=era_col,
            target_col=target_col,
            verbose=verbose,
            random_state=random_state,
        )
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.objective = objective
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.importance_type = importance_type

        self.max_bin = max_bin
        self.min_data_in_bin = min_data_in_bin
        self.colsample_bynode = colsample_bynode
        self.extra_trees = extra_trees
        self.path_smooth = path_smooth
        self.max_delta_step = max_delta_step
        self.linear_tree = linear_tree
        self.linear_lambda = linear_lambda
        self.drop_rate = drop_rate
        self.max_drop = max_drop
        self.skip_drop = skip_drop
        self.xgboost_dart_mode = xgboost_dart_mode
        self.uniform_drop = uniform_drop
        self.alpha = alpha
        self.fair_c = fair_c
        self.feature_pre_filter = feature_pre_filter
        self.force_col_wise = force_col_wise
        self.force_row_wise = force_row_wise
        self.lgbm_verbosity = lgbm_verbosity

    def _build_lgbm_model(self) -> LGBMRegressor:
        boosting = str(_decode_structured(self.boosting_type) or "gbdt").strip().lower()
        objective = str(_decode_structured(self.objective) or "regression").strip().lower()
        importance_type = str(_decode_structured(self.importance_type) or "gain").strip().lower()

        subsample = float(np.clip(float(self.subsample), 1e-4, 1.0))
        subsample_freq = max(0, int(self.subsample_freq))
        colsample_bytree = float(np.clip(float(self.colsample_bytree), 1e-4, 1.0))
        colsample_bynode = float(np.clip(float(self.colsample_bynode), 1e-4, 1.0))

        # LightGBM 'rf' boosting requires bagging (subsample < 1.0 and subsample_freq >= 1)
        if boosting == "rf":
            if subsample >= 1.0:
                subsample = 0.8
            if subsample_freq < 1:
                subsample_freq = 1

        max_depth = int(self.max_depth)
        num_leaves = max(2, int(self.num_leaves))
        if max_depth > 0:
            max_possible_leaves = int(2 ** min(max_depth, 20))
            num_leaves = min(num_leaves, max(2, max_possible_leaves))

        force_col_wise = bool(self.force_col_wise)
        force_row_wise = bool(self.force_row_wise)
        if force_col_wise and force_row_wise:
            force_row_wise = False

        extra_kwargs: Dict[str, Any] = {
            "max_bin": max(2, int(self.max_bin)),
            "min_data_in_bin": max(1, int(self.min_data_in_bin)),
            "feature_fraction_bynode": colsample_bynode,
            "extra_trees": bool(self.extra_trees),
            "path_smooth": max(0.0, float(self.path_smooth)),
            "max_delta_step": float(self.max_delta_step),
            "linear_tree": bool(self.linear_tree),
            "linear_lambda": max(0.0, float(self.linear_lambda)),
            "feature_pre_filter": bool(self.feature_pre_filter),
            "force_col_wise": force_col_wise,
            "force_row_wise": force_row_wise,
            "verbosity": int(self.lgbm_verbosity),
        }

        if boosting == "dart":
            extra_kwargs.update({
                "drop_rate": float(np.clip(float(self.drop_rate), 0.0, 1.0)),
                "max_drop": int(self.max_drop),
                "skip_drop": float(np.clip(float(self.skip_drop), 0.0, 1.0)),
                "xgboost_dart_mode": bool(self.xgboost_dart_mode),
                "uniform_drop": bool(self.uniform_drop),
            })

        if objective in ("huber", "quantile"):
            extra_kwargs["alpha"] = float(self.alpha)
        elif objective == "fair":
            extra_kwargs["fair_c"] = max(1e-6, float(self.fair_c))

        return LGBMRegressor(
            boosting_type=boosting,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=max(1e-5, float(self.learning_rate)),
            n_estimators=max(1, int(self.n_estimators)),
            subsample_for_bin=max(100, int(self.subsample_for_bin)),
            objective=objective,
            min_split_gain=max(0.0, float(self.min_split_gain)),
            min_child_weight=max(0.0, float(self.min_child_weight)),
            min_child_samples=max(1, int(self.min_child_samples)),
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=max(0.0, float(self.reg_alpha)),
            reg_lambda=max(0.0, float(self.reg_lambda)),
            random_state=int(self.random_state),
            n_jobs=int(self.n_jobs),
            importance_type=importance_type,
            **extra_kwargs,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
        eras: Optional[Any] = None,
    ) -> "LightGBMNumeraiEstimator":
        features_df, _, target_values, _, sample_weights, era_values = self._prepare_fit_inputs(
            X, y, sample_weight=sample_weight, eras=eras, return_matrix=False
        )
        selected_cols = list(self.selected_feature_names_)
        selected_df = features_df if len(selected_cols) == len(features_df.columns) and list(features_df.columns) == selected_cols else features_df[selected_cols]

        model = self._build_lgbm_model()
        fit_kwargs: Dict[str, Any] = {}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights

        model.fit(selected_df, target_values, **fit_kwargs)
        self.model_ = model

        booster = model.booster_
        gain_imp = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
        split_imp = np.asarray(booster.feature_importance(importance_type="split"), dtype=np.float64)
        self.feature_importances_ = np.asarray(model.feature_importances_, dtype=np.float64)
        self.feature_importance_df_ = (
            pd.DataFrame({
                "feature": selected_cols,
                "gain_importance": gain_imp,
                "split_importance": split_imp,
                "train_corr": np.asarray(self.selected_corrs_, dtype=np.float64),
            })
            .sort_values(["gain_importance", "split_importance"], ascending=False, kind="stable")
            .reset_index(drop=True)
        )

        if self.bias_correction not in (None, False, "None", "none", ""):
            raw_train_pred = np.asarray(model.predict(selected_df), dtype=np.float64).ravel()
            self._fit_bias_correction(target_values, raw_train_pred, era_values)
            del raw_train_pred
        else:
            self.learned_bias_offset_ = 0.0

        del selected_df, features_df
        gc.collect()
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("LightGBMNumeraiEstimator must be fitted before calling predict.")
        features_df, era_values = self._prepare_predict_df(X)
        selected_cols = list(self.selected_feature_names_)
        selected_df = features_df if len(selected_cols) == len(features_df.columns) and list(features_df.columns) == selected_cols else features_df[selected_cols]
        raw_pred = np.asarray(self.model_.predict(selected_df), dtype=np.float64).ravel()
        del selected_df, features_df
        return self._apply_prediction_transform(raw_pred, era_values)


LightGBMEstimator = LightGBMNumeraiEstimator
