"""lgbm_utils.py — LightGBMNumeraiEstimator child class of BaseNumeraiEstimator with all LightGBM parameters exposed.

Shared feature extraction (`NumeraiExtractions`), correlation calculation (`compute_feature_target_correlations`),
search space helpers, and Numerai evaluation functions live in `utils.py`, and shared parent estimator logic
lives in `baseline_utils.py`.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from baseline_utils import BaseNumeraiEstimator
from utils import NumeraiExtractions, _decode_structured

_LGB_DATASET_CACHE: "OrderedDict[str, lgb.Dataset]" = OrderedDict()
_LGB_DATASET_CACHE_MAX = 8


def clear_lgb_dataset_cache() -> None:
    """Clear the cached binned LightGBM training datasets."""
    _LGB_DATASET_CACHE.clear()


def make_shared_lgb_datasets(
    X_train: Any,
    y_train: Any,
    X_valid: Optional[Any] = None,
    y_valid: Optional[Any] = None,
    *,
    sample_weight: Optional[Any] = None,
    valid_sample_weight: Optional[Any] = None,
    **bin_params: Any,
) -> Tuple[lgb.Dataset, Optional[lgb.Dataset]]:
    """Construct pre-binned `lgb.Dataset` objects with `free_raw_data=False` and `reference=train_ds` for validation."""
    params_dict = dict(bin_params) if bin_params else None
    train_ds = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sample_weight,
        params=params_dict,
        free_raw_data=False,
    )
    train_ds.construct()
    valid_ds: Optional[lgb.Dataset] = None
    if X_valid is not None:
        valid_ds = lgb.Dataset(
            X_valid,
            label=y_valid,
            weight=valid_sample_weight,
            reference=train_ds,
            params=params_dict,
            free_raw_data=False,
        )
        valid_ds.construct()
    return train_ds, valid_ds


class _BoosterModelAdapter:
    """Lightweight adapter exposing `LGBMRegressor`-compatible `.booster_`, `.predict()`, and `.feature_importances_`."""

    def __init__(self, booster: lgb.Booster, importance_type: str = "gain"):
        self.booster_ = booster
        self.importance_type = str(importance_type or "gain").strip().lower()

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self.booster_.feature_importance(importance_type=self.importance_type), dtype=np.float64)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.booster_.predict(X), dtype=np.float64)


def _lgb_dataset_fingerprint(
    X_df: pd.DataFrame,
    y_arr: np.ndarray,
    sample_weights: Optional[np.ndarray],
    bin_params: Dict[str, Any],
) -> str:
    n_rows, n_cols = X_df.shape
    y_bytes = np.ascontiguousarray(y_arr, dtype=np.float32).tobytes()
    y_hash = hashlib.blake2b(y_bytes, digest_size=16).hexdigest()
    w_hash = (
        hashlib.blake2b(np.ascontiguousarray(sample_weights, dtype=np.float64).tobytes(), digest_size=16).hexdigest()
        if sample_weights is not None
        else "none"
    )
    # Sample first, middle, and last rows + column names/dtypes for fast collision-resistant fingerprinting
    sample_idx = np.unique(np.linspace(0, max(0, n_rows - 1), num=min(n_rows, 32), dtype=np.int64))
    sample_bytes = np.ascontiguousarray(X_df.to_numpy(copy=False)[sample_idx]).tobytes()
    x_hash = hashlib.blake2b(sample_bytes, digest_size=16).hexdigest()
    payload = json.dumps(
        {
            "shape": [int(n_rows), int(n_cols)],
            "cols": [str(c) for c in X_df.columns],
            "dtypes": [str(dt) for dt in X_df.dtypes],
            "bin_params": {k: repr(v) for k, v in sorted(bin_params.items())},
            "x_hash": x_hash,
            "y_hash": y_hash,
            "w_hash": w_hash,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=24).hexdigest()


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
        era_stride: int = 1,
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
        reuse_binned_dataset: bool = False,
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
            era_stride=era_stride,
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
        self.reuse_binned_dataset = reuse_binned_dataset

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
        if bool(self.reuse_binned_dataset):
            raw_params = model.get_params()
            num_boost_round = int(raw_params.pop("n_estimators", 200))
            importance_type = str(raw_params.pop("importance_type", "gain") or "gain").strip().lower()
            raw_params.pop("class_weight", None)
            seed_val = raw_params.pop("random_state", None)
            if seed_val is not None:
                raw_params["seed"] = int(seed_val)
            n_jobs_val = raw_params.pop("n_jobs", -1)
            if n_jobs_val is not None:
                raw_params["n_jobs"] = int(n_jobs_val)
            bin_params = {
                "max_bin": int(raw_params.get("max_bin", 255)),
                "min_data_in_bin": int(raw_params.get("min_data_in_bin", 3)),
                "subsample_for_bin": int(raw_params.get("subsample_for_bin", 200000)),
                "feature_pre_filter": bool(raw_params.get("feature_pre_filter", True)),
            }
            ds_key = _lgb_dataset_fingerprint(selected_df, target_values, sample_weights, bin_params)
            if ds_key in _LGB_DATASET_CACHE:
                _LGB_DATASET_CACHE.move_to_end(ds_key)
                train_ds = _LGB_DATASET_CACHE[ds_key]
            else:
                train_ds, _ = make_shared_lgb_datasets(
                    selected_df,
                    target_values,
                    sample_weight=sample_weights,
                    **bin_params,
                )
                _LGB_DATASET_CACHE[ds_key] = train_ds
                _LGB_DATASET_CACHE.move_to_end(ds_key)
                while len(_LGB_DATASET_CACHE) > _LGB_DATASET_CACHE_MAX:
                    _LGB_DATASET_CACHE.popitem(last=False)
            booster = lgb.train(
                params=raw_params,
                train_set=train_ds,
                num_boost_round=num_boost_round,
            )
            self.model_ = _BoosterModelAdapter(booster, importance_type=importance_type)
        else:
            fit_kwargs: Dict[str, Any] = {}
            if sample_weights is not None:
                fit_kwargs["sample_weight"] = sample_weights

            model.fit(selected_df, target_values, **fit_kwargs)
            self.model_ = model

        booster = self.model_.booster_
        gain_imp = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
        split_imp = np.asarray(booster.feature_importance(importance_type="split"), dtype=np.float64)
        self.feature_importances_ = np.asarray(self.model_.feature_importances_, dtype=np.float64)
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
