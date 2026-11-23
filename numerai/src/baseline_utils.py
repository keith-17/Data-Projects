"""baseline_utils.py — BaseNumeraiEstimator parent class and BinSnappingEstimator / LagReversalEstimator child classes.

Shared feature extraction (`NumeraiExtractions`), correlation calculation (`compute_feature_target_correlations`),
search space helpers, and Numerai evaluation functions live in `utils.py`.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import detrend as scipy_detrend
from sklearn.base import BaseEstimator, RegressorMixin, clone
from statsmodels.tsa.stattools import pacf as statsmodels_pacf

from utils import (
    FORBIDDEN_FEATURE_COLS,
    NumeraiExtractions,
    _all_numeric_dtypes,
    _decode_structured,
    _safe_autocorr,
    compute_feature_target_correlations,
)

_EXTRACTION_CACHE: "OrderedDict[str, Tuple[NumeraiExtractions, pd.DataFrame]]" = OrderedDict()
_EXTRACTION_CACHE_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "max_entries": 4,
}


def configure_extraction_cache(enabled: bool = True, max_entries: int = 4) -> Dict[str, Any]:
    """Configure the per-process fold-safe LRU extraction cache used by `BaseNumeraiEstimator`."""
    _EXTRACTION_CACHE_CONFIG["enabled"] = bool(enabled)
    _EXTRACTION_CACHE_CONFIG["max_entries"] = max(0, int(max_entries))
    if not _EXTRACTION_CACHE_CONFIG["enabled"] or _EXTRACTION_CACHE_CONFIG["max_entries"] == 0:
        _EXTRACTION_CACHE.clear()
    else:
        while len(_EXTRACTION_CACHE) > _EXTRACTION_CACHE_CONFIG["max_entries"]:
            _EXTRACTION_CACHE.popitem(last=False)
    return dict(_EXTRACTION_CACHE_CONFIG)


def clear_extraction_cache() -> None:
    """Clear all entries from the per-process extraction cache."""
    _EXTRACTION_CACHE.clear()


def _build_extraction_cache_key(
    extractor: NumeraiExtractions,
    X_df: pd.DataFrame,
    era_values: Optional[np.ndarray],
    target_values: np.ndarray,
) -> str:
    """Compute a fold-safe cache key from canonicalized extractor params, era bytes, and target bytes."""
    canon_params = sorted(
        (str(key), repr(_decode_structured(val)))
        for key, val in extractor.get_params(deep=True).items()
    )
    if era_values is not None and len(era_values) > 0:
        era_contig = np.ascontiguousarray(era_values)
        era_min = int(np.min(era_contig))
        era_max = int(np.max(era_contig))
        era_hash = hashlib.blake2b(era_contig.tobytes(), digest_size=16).hexdigest()
    else:
        era_min = None
        era_max = None
        era_hash = "none"
    target_contig = np.ascontiguousarray(target_values, dtype=np.float32)
    target_hash = hashlib.blake2b(target_contig.tobytes(), digest_size=16).hexdigest()
    payload = json.dumps(
        {
            "params": canon_params,
            "n_rows": int(len(X_df)),
            "cols": [str(c) for c in X_df.columns],
            "era_min": era_min,
            "era_max": era_max,
            "era_hash": era_hash,
            "target_hash": target_hash,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=24).hexdigest()


class BaseNumeraiEstimator(RegressorMixin, BaseEstimator):
    """Shared Parent Class for `BinSnappingEstimator` (Baseline A), `LagReversalEstimator` (Baseline E), and `PersistenceEstimator`."""

    def __init__(
        self,
        extractor: Optional[NumeraiExtractions] = None,
        top_k: Optional[int] = 20,
        correlation_method: Optional[str] = "pandas_mean",
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
    ):
        self.extractor = extractor
        self.top_k = top_k
        self.correlation_method = correlation_method
        self.abs_corr = abs_corr
        self.era_weighting = era_weighting
        self.recency_decay = recency_decay
        self.prediction_transform = prediction_transform
        self.bias_correction = bias_correction
        self.walk_forward_step_eras = walk_forward_step_eras
        self.walk_forward_window = walk_forward_window
        self.target_horizon_eras = target_horizon_eras
        self.era_stride = era_stride
        self.era_col = era_col
        self.target_col = target_col
        self.verbose = verbose
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = super().get_params(deep=deep)
        if deep and self.extractor is None:
            params["extractor"] = None
            for key, val in NumeraiExtractions().get_params(deep=True).items():
                params[f"extractor__{key}"] = val
        return params

    def set_params(self, **params: Any) -> "BaseNumeraiEstimator":
        clean = {key: _decode_structured(val) for key, val in params.items()}
        if any(key.startswith("extractor__") for key in clean) and clean.get("extractor", self.extractor) is None:
            clean["extractor"] = NumeraiExtractions()
        return super().set_params(**clean)

    def _compute_era_weights(
        self,
        era_values: Optional[np.ndarray],
        explicit_sample_weight: Optional[Any],
        n_samples: int,
    ) -> Optional[np.ndarray]:
        if explicit_sample_weight is not None:
            weights = np.asarray(explicit_sample_weight, dtype=np.float64).ravel()
            if len(weights) != n_samples:
                raise ValueError(f"sample_weight length ({len(weights)}) must match X rows ({n_samples}).")
            return weights
        if era_values is None or self.era_weighting in (None, "none", "None", ""):
            return None
        mode = str(self.era_weighting).strip().lower()
        if mode == "equal_era":
            era_series = pd.Series(era_values)
            counts = era_series.map(era_series.value_counts()).to_numpy(dtype=np.float64)
            weights = 1.0 / np.maximum(counts, 1.0)
            return weights / np.mean(weights)
        if mode == "recency_linear":
            first_era = float(np.min(era_values))
            span = max(1.0, float(np.max(era_values) - first_era))
            weights = 0.5 + (era_values.astype(np.float64) - first_era) / span
            return weights / np.mean(weights)
        if mode == "recency_exp":
            last_era = float(np.max(era_values))
            weights = np.exp(-float(self.recency_decay or 0.005) * (last_era - era_values.astype(np.float64)))
            return weights / np.mean(weights)
        return None

    def _prepare_fit_inputs(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
        eras: Optional[Any] = None,
        return_matrix: bool = True,
        matrix_dtype: Any = np.float32,
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], np.ndarray, List[str], Optional[np.ndarray], Optional[np.ndarray]]:
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            arr = np.asarray(X, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            X_df = pd.DataFrame(arr, columns=[f"f{idx}" for idx in range(arr.shape[1])])

        if self.target_col and self.target_col in X_df.columns:
            target_values = pd.to_numeric(X_df[self.target_col], errors="coerce").fillna(0.5).to_numpy(dtype=np.float32)
        else:
            target_values = np.asarray(y, dtype=np.float32).ravel()

        n_samples = len(X_df)
        if len(target_values) != n_samples:
            raise ValueError(f"X rows ({n_samples}) must match y length ({len(target_values)}).")

        if eras is not None:
            era_values: Optional[np.ndarray] = np.asarray(eras).ravel()
        elif self.era_col in X_df.columns:
            era_values = pd.to_numeric(X_df[self.era_col], errors="coerce").fillna(1).to_numpy(dtype=np.int32)
        else:
            era_values = None

        # Optional in-fit training era subsampling (never applied during predict())
        stride_val = int(self.era_stride) if self.era_stride not in (None, False, "None", "none", "null", "") else 1
        if stride_val > 1 and era_values is not None and len(era_values) > 0:
            kept_fit_eras = np.unique(era_values)[::stride_val]
            if len(kept_fit_eras) < len(np.unique(era_values)):
                stride_mask = np.isin(era_values, kept_fit_eras)
                X_df = X_df.loc[stride_mask]
                target_values = target_values[stride_mask]
                era_values = era_values[stride_mask]
                if sample_weight is not None:
                    sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()[stride_mask]
                n_samples = len(X_df)

        if self.extractor is not None:
            cache_enabled = bool(_EXTRACTION_CACHE_CONFIG["enabled"]) and int(_EXTRACTION_CACHE_CONFIG["max_entries"]) > 0
            if cache_enabled:
                cache_key = _build_extraction_cache_key(self.extractor, X_df, era_values, target_values)
                if cache_key in _EXTRACTION_CACHE:
                    _EXTRACTION_CACHE.move_to_end(cache_key)
                    cached_ext, cached_X_proc = _EXTRACTION_CACHE[cache_key]
                    self.extractor_ = cached_ext
                    X_proc = cached_X_proc
                else:
                    fitted_ext = clone(self.extractor)
                    fitted_ext.fit(X_df, target_values)
                    X_proc = fitted_ext.transform(X_df)
                    _EXTRACTION_CACHE[cache_key] = (fitted_ext, X_proc)
                    _EXTRACTION_CACHE.move_to_end(cache_key)
                    while len(_EXTRACTION_CACHE) > int(_EXTRACTION_CACHE_CONFIG["max_entries"]):
                        _EXTRACTION_CACHE.popitem(last=False)
                    self.extractor_ = fitted_ext
            else:
                self.extractor_ = clone(self.extractor)
                self.extractor_.fit(X_df, target_values)
                X_proc = self.extractor_.transform(X_df)
        else:
            self.extractor_ = None
            X_proc = X_df

        t_col = str(self.target_col or "target")
        feature_names = [
            str(col)
            for col in X_proc.columns
            if str(col) not in FORBIDDEN_FEATURE_COLS
            and str(col) != t_col
            and not str(col).startswith("target")
        ]
        if not feature_names:
            raise ValueError("No valid feature columns remain after removing forbidden metadata columns.")

        raw_features_df = X_proc[feature_names]
        if _all_numeric_dtypes(raw_features_df):
            features_df = raw_features_df
        else:
            features_df = raw_features_df.copy()
            for col in feature_names:
                if not pd.api.types.is_numeric_dtype(features_df[col].dtype):
                    features_df[col] = pd.to_numeric(features_df[col], errors="coerce").astype(np.float32)

        feature_matrix: Optional[np.ndarray] = None
        if return_matrix:
            if matrix_dtype is None:
                feature_matrix = features_df.to_numpy(copy=False)
            else:
                feature_matrix = features_df.to_numpy(dtype=matrix_dtype, copy=False)

        self.n_features_in_ = len(feature_names)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)

        if self.correlation_method in (None, "none", "None", ""):
            self.feature_corrs_ = np.ones(len(feature_names), dtype=np.float64)
            ranking_scores = np.arange(len(feature_names), 0, -1, dtype=np.float64)
        else:
            can_reuse_corrs = (
                self.extractor_ is not None
                and getattr(self.extractor_, "pca_", None) is None
                and _decode_structured(getattr(self.extractor_, "year_horizon_filter", None))
                in (None, False, "None", "none", "null", "")
                and hasattr(self.extractor_, "feature_corrs_series_")
                and str(self.correlation_method).strip().lower()
                == str(getattr(self.extractor_, "correlation_method", None) or "").strip().lower()
            )
            if can_reuse_corrs:
                cached_series = self.extractor_.feature_corrs_series_
                missing_cols = [col for col in feature_names if col not in cached_series.index]
                if missing_cols:
                    extra_series = compute_feature_target_correlations(
                        features_df[missing_cols],
                        pd.Series(target_values, index=features_df.index),
                        correlation_method=self.correlation_method,
                        era_series=pd.Series(era_values, index=features_df.index) if era_values is not None else None,
                    )
                    combined_series = pd.concat([cached_series, extra_series])
                    combined_series = combined_series.loc[~combined_series.index.duplicated(keep="first")]
                    corr_series = combined_series.reindex(feature_names).fillna(0.0)
                else:
                    corr_series = cached_series.reindex(feature_names).fillna(0.0)
            else:
                corr_series = compute_feature_target_correlations(
                    features_df,
                    pd.Series(target_values, index=features_df.index),
                    correlation_method=self.correlation_method,
                    era_series=pd.Series(era_values, index=features_df.index) if era_values is not None else None,
                )
            self.feature_corrs_ = corr_series.to_numpy(dtype=np.float64)
            ranking_scores = np.abs(self.feature_corrs_) if bool(self.abs_corr) else self.feature_corrs_

        if self.top_k is not None and str(self.top_k) not in ("None", "none", ""):
            n_selected = min(max(1, int(self.top_k)), len(feature_names))
        else:
            n_selected = len(feature_names)

        selected_indices = np.argsort(-ranking_scores, kind="stable")[:n_selected]
        self.selected_indices_ = selected_indices
        self.selected_corrs_ = self.feature_corrs_[selected_indices]
        self.selected_feature_names_ = [feature_names[idx] for idx in selected_indices]

        if self.walk_forward_step_eras not in (None, False, "None", "none", "null", "", 0):
            self._wf_train_X_ = X_df
            self._wf_train_y_ = target_values
            self._wf_train_eras_ = era_values
        else:
            self._wf_train_X_ = None
            self._wf_train_y_ = None
            self._wf_train_eras_ = None

        weights = self._compute_era_weights(era_values, sample_weight, n_samples)
        return features_df, feature_matrix, target_values, feature_names, weights, era_values

    def _prepare_predict_df(self, X: Any) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            arr = np.asarray(X, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            X_df = pd.DataFrame(arr, columns=list(self.feature_names_in_))

        era_values = (
            pd.to_numeric(X_df[self.era_col], errors="coerce").fillna(1).to_numpy(dtype=np.int32)
            if self.era_col in X_df.columns
            else None
        )
        X_proc = self.extractor_.transform(X_df) if getattr(self, "extractor_", None) is not None else X_df
        feature_names = list(self.feature_names_in_)
        raw_df = X_proc[feature_names]
        if _all_numeric_dtypes(raw_df):
            return raw_df, era_values
        clean_df = raw_df.copy()
        for col in feature_names:
            if not pd.api.types.is_numeric_dtype(clean_df[col].dtype):
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").astype(np.float32)
        return clean_df, era_values

    def _prepare_predict_matrix(self, X: Any, matrix_dtype: Any = np.float32) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        features_df, era_values = self._prepare_predict_df(X)
        if matrix_dtype is None:
            matrix = features_df.to_numpy(copy=False)
        else:
            matrix = features_df.to_numpy(dtype=matrix_dtype, copy=False)
        return matrix, era_values

    def _transform_predictions_only(self, raw_pred: np.ndarray, era_values: Optional[np.ndarray]) -> np.ndarray:
        if era_values is None or self.prediction_transform in (None, "none", "None", "", "raw"):
            return raw_pred.astype(np.float64, copy=False)
        mode = str(self.prediction_transform).strip().lower()
        pred_series = pd.Series(raw_pred)
        eras = pd.Series(era_values)
        if mode == "era_zscore":
            means = pred_series.groupby(eras).transform("mean")
            stds = pred_series.groupby(eras).transform("std").replace(0.0, 1.0).fillna(1.0)
            return ((pred_series - means) / stds).to_numpy(dtype=np.float64, copy=False)
        if mode == "era_rank":
            return pred_series.groupby(eras).rank(pct=True).to_numpy(dtype=np.float64, copy=False)
        return raw_pred.astype(np.float64, copy=False)

    def _fit_bias_correction(
        self,
        target_values: np.ndarray,
        raw_train_pred: np.ndarray,
        era_values: Optional[np.ndarray],
    ) -> None:
        if self.bias_correction in (None, False, "None", "none", ""):
            self.learned_bias_offset_ = 0.0
            return
        mode = str(self.bias_correction).strip().lower()
        if mode == "mean_residual":
            transformed = self._transform_predictions_only(raw_train_pred, era_values)
            self.learned_bias_offset_ = float(np.mean(target_values - transformed)) if len(target_values) else 0.0
        else:
            raise ValueError(f"Unsupported bias_correction '{self.bias_correction}'; use None or 'mean_residual'.")

    def _apply_prediction_transform(self, raw_pred: np.ndarray, era_values: Optional[np.ndarray]) -> np.ndarray:
        transformed = self._transform_predictions_only(raw_pred, era_values)
        bias_offset = float(getattr(self, "learned_bias_offset_", 0.0))
        return (transformed + bias_offset) if abs(bias_offset) > 1e-15 else transformed


class BinSnappingEstimator(BaseNumeraiEstimator):
    """Child Class — Baseline A: Deterministic Bin-Snapping ("Map Matching") lookup-table estimator."""

    def __init__(
        self,
        extractor: Optional[NumeraiExtractions] = None,
        top_k: Optional[int] = 20,
        n_bins: int = 5,
        correlation_method: Optional[str] = "pandas_mean",
        abs_corr: bool = True,
        smoothing: Optional[float] = 0.0,
        agg_mode: str = "mean",
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
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.agg_mode = agg_mode

    def _compute_raw_snapped_pred(self, selected_bins: np.ndarray) -> np.ndarray:
        snapped_targets = self.bin_table_[np.arange(len(self.selected_indices_)), selected_bins]
        if self.agg_mode == "corr_weighted":
            abs_corrs = np.abs(self.selected_corrs_)
            corr_weight_sum = float(abs_corrs.sum())
            if corr_weight_sum > 1e-12:
                return snapped_targets @ (abs_corrs / corr_weight_sum)
            return snapped_targets.mean(axis=1)
        if self.agg_mode in (None, "none", "None", "mean"):
            return snapped_targets.mean(axis=1)
        raise ValueError(f"Unsupported agg_mode '{self.agg_mode}'; use 'mean' or 'corr_weighted'.")

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
        eras: Optional[Any] = None,
    ) -> "BinSnappingEstimator":
        _, feature_matrix, target_values, _, sample_weights, era_values = self._prepare_fit_inputs(
            X, y, sample_weight=sample_weight, eras=eras
        )
        n_bins = max(2, int(self.n_bins if self.n_bins is not None else 5))
        middle_bin = float(n_bins // 2)
        clean_features = np.nan_to_num(feature_matrix, nan=middle_bin, posinf=float(n_bins - 1), neginf=0.0)

        if sample_weights is not None and float(sample_weights.sum()) > 0.0:
            self.global_mean_ = float(np.average(target_values, weights=sample_weights))
        else:
            self.global_mean_ = float(np.mean(target_values))

        n_selected = len(self.selected_indices_)
        bin_table = np.full((n_selected, n_bins), self.global_mean_, dtype=np.float64)
        selected_bins = np.clip(np.rint(clean_features[:, self.selected_indices_]), 0, n_bins - 1).astype(np.int64)
        smoothing_weight = max(0.0, float(self.smoothing or 0.0))

        for feature_pos in range(n_selected):
            column_bins = selected_bins[:, feature_pos]
            for bin_index in range(n_bins):
                in_bin_mask = column_bins == bin_index
                if not np.any(in_bin_mask):
                    continue
                if sample_weights is not None:
                    weight_sum = float(sample_weights[in_bin_mask].sum())
                    target_sum = float(np.dot(sample_weights[in_bin_mask], target_values[in_bin_mask]))
                else:
                    weight_sum = float(np.sum(in_bin_mask))
                    target_sum = float(target_values[in_bin_mask].sum())
                if weight_sum > 0.0:
                    bin_table[feature_pos, bin_index] = (
                        target_sum + smoothing_weight * self.global_mean_
                    ) / (weight_sum + smoothing_weight)

        self.bin_table_ = bin_table
        self.lookup_table_: Dict[str, Dict[int, float]] = {
            self.selected_feature_names_[feature_pos]: {
                bin_index: float(bin_table[feature_pos, bin_index]) for bin_index in range(n_bins)
            }
            for feature_pos in range(n_selected)
        }
        raw_train_pred = self._compute_raw_snapped_pred(selected_bins)
        self._fit_bias_correction(target_values, raw_train_pred, era_values)
        del clean_features, selected_bins, feature_matrix
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "bin_table_"):
            raise ValueError("BinSnappingEstimator must be fitted before calling predict.")
        feature_matrix, era_values = self._prepare_predict_matrix(X)
        n_bins = self.bin_table_.shape[1]
        middle_bin = float(n_bins // 2)
        selected_features = np.nan_to_num(
            feature_matrix[:, self.selected_indices_],
            nan=middle_bin,
            posinf=float(n_bins - 1),
            neginf=0.0,
        )
        bin_indices = np.clip(np.rint(selected_features), 0, n_bins - 1).astype(np.int64)
        raw_pred = self._compute_raw_snapped_pred(bin_indices)
        return self._apply_prediction_transform(raw_pred, era_values)


class LagReversalEstimator(BaseNumeraiEstimator):
    """Child Class — Baseline E: Era lag-autocorrelation/PACF sign-inversion & shrinkage weighted feature sum estimator."""

    def __init__(
        self,
        extractor: Optional[NumeraiExtractions] = None,
        lag: int = 1,
        acf_mode: str = "feature_mean",
        acf_metric: str = "acf",
        acf_detrend: Optional[str] = "linear",
        acf_threshold: Optional[float] = 0.0,
        sign_mode: str = "hard",
        top_k: Optional[int] = None,
        correlation_method: Optional[str] = "pandas_mean",
        abs_corr: bool = True,
        center_features: bool = True,
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
        self.lag = lag
        self.acf_mode = acf_mode
        self.acf_metric = acf_metric
        self.acf_detrend = acf_detrend
        self.acf_threshold = acf_threshold
        self.sign_mode = sign_mode
        self.center_features = center_features

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
        eras: Optional[Any] = None,
    ) -> "LagReversalEstimator":
        _, raw_features, target_values, feature_names, _, era_values = self._prepare_fit_inputs(
            X, y, sample_weight=sample_weight, eras=eras
        )
        n_features = len(feature_names)
        finite_mask = np.isfinite(raw_features)
        observed_counts = finite_mask.sum(axis=0)
        column_sums = np.where(finite_mask, raw_features, 0.0).sum(axis=0)
        self.feature_means_ = np.divide(
            column_sums,
            observed_counts,
            out=np.zeros(n_features, dtype=np.float64),
            where=observed_counts > 0,
        )
        clean_features = np.where(finite_mask, raw_features, self.feature_means_)

        feature_acfs = np.zeros(n_features, dtype=np.float64)
        if era_values is not None and self.lag not in (None, "none", "None", 0):
            if self.acf_mode in (None, "none", "None", "feature_mean"):
                era_series_df = pd.DataFrame(clean_features, columns=feature_names).groupby(era_values, sort=True).mean()
            elif self.acf_mode == "feature_corr":
                per_era_corrs: Dict[Any, np.ndarray] = {}
                grouped_df = pd.DataFrame(clean_features, columns=feature_names)
                grouped_df["_era_"] = era_values
                grouped_df["_target_"] = target_values
                for era_key, era_df in grouped_df.groupby("_era_", sort=True):
                    era_x = era_df[feature_names].to_numpy(dtype=np.float64, copy=False)
                    era_y = era_df["_target_"].to_numpy(dtype=np.float64, copy=False)
                    centered_x = era_x - era_x.mean(axis=0)
                    centered_y = era_y - float(era_y.mean())
                    denominator = np.sqrt(np.sum(centered_x ** 2, axis=0) * float(np.sum(centered_y ** 2)))
                    per_era_corrs[era_key] = np.divide(
                        centered_x.T @ centered_y,
                        denominator,
                        out=np.zeros(n_features, dtype=np.float64),
                        where=denominator > 1e-12,
                    )
                era_series_df = pd.DataFrame.from_dict(per_era_corrs, orient="index", columns=feature_names).sort_index()
            else:
                raise ValueError(f"Unsupported acf_mode '{self.acf_mode}'; use 'feature_mean' or 'feature_corr'.")

            detrend_mode = "none" if self.acf_detrend is None else str(self.acf_detrend).lower()
            if detrend_mode in ("linear", "constant") and len(era_series_df) > (2 if detrend_mode == "linear" else 1):
                detrended_vals = scipy_detrend(
                    era_series_df.to_numpy(dtype=np.float64, copy=False),
                    axis=0,
                    type=detrend_mode,
                )
                era_series_df = pd.DataFrame(
                    detrended_vals,
                    index=era_series_df.index,
                    columns=feature_names,
                )

            lag_steps = max(1, int(self.lag))
            metric_mode = str(self.acf_metric or "acf").strip().lower()
            if len(era_series_df) > lag_steps + 1:
                if metric_mode == "pacf":
                    max_valid_lag = max(1, (len(era_series_df) // 2) - 1)
                    eff_lag = min(lag_steps, max_valid_lag)
                    pacf_list: List[float] = []
                    for column in feature_names:
                        col_arr = era_series_df[column].to_numpy(dtype=np.float64, copy=False)
                        if float(np.std(col_arr)) <= 1e-12 or len(col_arr) < 4:
                            pacf_list.append(0.0)
                        else:
                            try:
                                pacf_vals = statsmodels_pacf(col_arr, nlags=eff_lag, method="yw")
                                pacf_list.append(float(pacf_vals[eff_lag]))
                            except Exception:
                                pacf_list.append(_safe_autocorr(col_arr, lag=eff_lag))
                    feature_acfs = np.array(pacf_list, dtype=np.float64)
                elif metric_mode in ("acf", "none", ""):
                    feature_acfs = np.array(
                        [_safe_autocorr(era_series_df[column].to_numpy(dtype=np.float64, copy=False), lag=lag_steps) for column in feature_names],
                        dtype=np.float64,
                    )
                else:
                    raise ValueError(f"Unsupported acf_metric '{self.acf_metric}'; use 'acf' or 'pacf'.")
                feature_acfs = np.nan_to_num(feature_acfs, nan=0.0, posinf=0.0, neginf=0.0)

        self.feature_acfs_ = feature_acfs
        thresh = float(self.acf_threshold if self.acf_threshold is not None else 0.0)
        delta = self.feature_acfs_ - thresh
        s_mode = str(self.sign_mode or "hard").strip().lower()
        if s_mode in ("hard", "none", ""):
            self.inversion_signs_ = np.where(delta < 0.0, -1.0, 1.0)
        elif s_mode == "soft_threshold":
            raw_soft = np.sign(delta) * np.maximum(0.0, np.abs(self.feature_acfs_) - abs(thresh))
            max_abs = float(np.max(np.abs(raw_soft))) if len(raw_soft) else 0.0
            self.inversion_signs_ = (raw_soft / max_abs) if max_abs > 1e-12 else np.where(delta < 0.0, -1.0, 1.0)
        elif s_mode == "tanh":
            self.inversion_signs_ = np.tanh(delta / 0.05)
        else:
            raise ValueError(f"Unsupported sign_mode '{self.sign_mode}'; use 'hard', 'soft_threshold', or 'tanh'.")

        keep_mask = np.zeros(n_features, dtype=bool)
        keep_mask[self.selected_indices_] = True
        self.weights_ = np.where(keep_mask, self.feature_corrs_ * self.inversion_signs_, 0.0)

        adjusted_train = (clean_features - self.feature_means_) if self.center_features else clean_features
        raw_train_pred = adjusted_train @ self.weights_
        self._fit_bias_correction(target_values, raw_train_pred, era_values)
        del clean_features, raw_features, adjusted_train

        self.rule_summary_ = (
            pd.DataFrame({
                "feature": feature_names,
                "target_corr": self.feature_corrs_,
                "lag_acf": self.feature_acfs_,
                "sign_rule": self.inversion_signs_,
                "effective_weight": self.weights_,
                "_abs_weight": np.abs(self.weights_),
            })
            .sort_values("_abs_weight", ascending=False, kind="stable")
            .drop(columns=["_abs_weight"])
            .reset_index(drop=True)
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise ValueError("LagReversalEstimator must be fitted before calling predict.")
        raw_features, era_values = self._prepare_predict_matrix(X)
        clean_features = np.where(np.isfinite(raw_features), raw_features, self.feature_means_)
        adjusted_features = (clean_features - self.feature_means_) if self.center_features else clean_features
        raw_pred = adjusted_features @ self.weights_
        return self._apply_prediction_transform(raw_pred, era_values)


class PersistenceEstimator(BaseNumeraiEstimator):
    """Child Class — Brownlee Persistence Baseline: predicts each ticker's last settled target, trailing rolling target mean, expanding mean, or Zero-Rule global mean.

    Decoupled from `NumeraiExtractions` when `extractor=None` so CV search does not waste trials on feature extraction parameters.
    """

    def __init__(
        self,
        extractor: Optional[NumeraiExtractions] = None,
        ticker_col: str = "numerai_ticker",
        persistence_mode: Optional[str] = "last",
        persistence_lag: Optional[int] = 4,
        persistence_window: Optional[int] = 4,
        smoothing: Optional[float] = 0.0,
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
        self.ticker_col = ticker_col
        self.persistence_mode = persistence_mode
        self.persistence_lag = persistence_lag
        self.persistence_window = persistence_window
        self.smoothing = smoothing

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return BaseEstimator.get_params(self, deep=deep)

    def _raw_persistence_lookup(self, X_df: pd.DataFrame) -> np.ndarray:
        n_rows = len(X_df)
        if not self.ticker_lookup_ or self.ticker_col not in X_df.columns:
            return np.full(n_rows, self.global_mean_, dtype=np.float64)
        tickers = X_df[self.ticker_col].astype(str)
        return tickers.map(self.ticker_lookup_).fillna(self.global_mean_).to_numpy(dtype=np.float64)

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
        eras: Optional[Any] = None,
    ) -> "PersistenceEstimator":
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.target_col and self.target_col in X_df.columns:
            target_values = pd.to_numeric(X_df[self.target_col], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64)
        else:
            target_values = np.asarray(y, dtype=np.float64).ravel()

        if eras is not None:
            era_values: Optional[np.ndarray] = np.asarray(eras).ravel()
        elif self.era_col in X_df.columns:
            era_values = pd.to_numeric(X_df[self.era_col], errors="coerce").fillna(1).to_numpy(dtype=np.int32)
        else:
            era_values = None

        # Optional in-fit training era subsampling (never applied during predict())
        stride_val = int(self.era_stride) if self.era_stride not in (None, False, "None", "none", "null", "") else 1
        if stride_val > 1 and era_values is not None and len(era_values) > 0:
            kept_fit_eras = np.unique(era_values)[::stride_val]
            if len(kept_fit_eras) < len(np.unique(era_values)):
                stride_mask = np.isin(era_values, kept_fit_eras)
                X_df = X_df.loc[stride_mask]
                target_values = target_values[stride_mask]
                era_values = era_values[stride_mask]
                if sample_weight is not None:
                    sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()[stride_mask]
                n_samples = len(X_df)

        sample_weights = self._compute_era_weights(era_values, sample_weight, len(X_df))
        if self.walk_forward_step_eras not in (None, False, "None", "none", "null", "", 0):
            self._wf_train_X_ = X_df
            self._wf_train_y_ = target_values
            self._wf_train_eras_ = era_values
        else:
            self._wf_train_X_ = None
            self._wf_train_y_ = None
            self._wf_train_eras_ = None
        if sample_weights is not None and float(sample_weights.sum()) > 0.0:
            self.global_mean_ = float(np.average(target_values, weights=sample_weights))
        else:
            self.global_mean_ = float(np.mean(target_values)) if len(target_values) else 0.5

        self.ticker_lookup_: Dict[str, float] = {}
        mode = "none" if self.persistence_mode is None else str(self.persistence_mode).strip().lower()
        if mode in ("none", "", "zero_rule") or self.ticker_col not in X_df.columns:
            self.persistence_table_ = pd.DataFrame(columns=["ticker", "persisted_target", "obs_count", "last_era"])
            raw_train_pred = np.full(len(X_df), self.global_mean_, dtype=np.float64)
            self._fit_bias_correction(target_values, raw_train_pred, era_values)
            return self

        era_arr = (
            era_values
            if era_values is not None
            else np.ones(len(target_values), dtype=np.int32)
        )
        weights_arr = (
            sample_weights
            if sample_weights is not None
            else np.ones(len(target_values), dtype=np.float64)
        )
        hist_df = pd.DataFrame({
            "ticker": X_df[self.ticker_col].astype(str).to_numpy(),
            "era": era_arr,
            "target": target_values,
            "weight": weights_arr,
        })

        if self.persistence_lag not in (None, "none", "None", "", 0) and len(hist_df):
            lag_eras = max(0, int(self.persistence_lag))
            cutoff_era = int(hist_df["era"].max()) - lag_eras
            settled_df = hist_df.loc[hist_df["era"] <= cutoff_era]
            if len(settled_df):
                hist_df = settled_df

        hist_df = hist_df.sort_values(["ticker", "era"], kind="stable")
        smoothing_weight = max(0.0, float(self.smoothing or 0.0))

        if len(hist_df) == 0:
            self.persistence_table_ = pd.DataFrame(columns=["ticker", "persisted_target", "obs_count", "last_era"])
        elif mode == "last":
            grouped = hist_df.groupby("ticker", sort=False)
            last_rows = grouped.last()
            counts = grouped["target"].size()
            w_arr = np.maximum(last_rows["weight"].to_numpy(dtype=np.float64), 1e-12)
            y_arr = last_rows["target"].to_numpy(dtype=np.float64)
            persisted_arr = (w_arr * y_arr + smoothing_weight * self.global_mean_) / (w_arr + smoothing_weight)
            tickers_str = last_rows.index.astype(str)
            self.ticker_lookup_ = dict(zip(tickers_str.tolist(), persisted_arr.tolist()))
            self.persistence_table_ = (
                pd.DataFrame({
                    "ticker": tickers_str.to_numpy(),
                    "persisted_target": persisted_arr,
                    "obs_count": counts.reindex(last_rows.index).to_numpy(dtype=np.int64),
                    "last_era": last_rows["era"].to_numpy(dtype=np.int64),
                })
                .sort_values("obs_count", ascending=False, kind="stable")
                .reset_index(drop=True)
            )
        elif mode == "rolling_mean":
            win_eras = max(1, int(self.persistence_window if self.persistence_window not in (None, "none", "None", "", 0) else 4))
            tail_df = hist_df.groupby("ticker", sort=False).tail(win_eras).copy()
            tail_df["weighted_target"] = tail_df["target"] * tail_df["weight"]
            agg_df = tail_df.groupby("ticker", sort=False).agg(
                w_sum=("weight", "sum"),
                wt_sum=("weighted_target", "sum"),
                obs_count=("target", "size"),
                last_era=("era", "max"),
            )
            w_arr = np.maximum(agg_df["w_sum"].to_numpy(dtype=np.float64), 1e-12)
            wt_arr = agg_df["wt_sum"].to_numpy(dtype=np.float64)
            persisted_arr = (wt_arr + smoothing_weight * self.global_mean_) / (w_arr + smoothing_weight)
            tickers_str = agg_df.index.astype(str)
            self.ticker_lookup_ = dict(zip(tickers_str.tolist(), persisted_arr.tolist()))
            self.persistence_table_ = (
                pd.DataFrame({
                    "ticker": tickers_str.to_numpy(),
                    "persisted_target": persisted_arr,
                    "obs_count": agg_df["obs_count"].to_numpy(dtype=np.int64),
                    "last_era": agg_df["last_era"].to_numpy(dtype=np.int64),
                })
                .sort_values("obs_count", ascending=False, kind="stable")
                .reset_index(drop=True)
            )
        elif mode == "expanding_mean":
            hist_df["weighted_target"] = hist_df["target"] * hist_df["weight"]
            agg_df = hist_df.groupby("ticker", sort=False).agg(
                w_sum=("weight", "sum"),
                wt_sum=("weighted_target", "sum"),
                obs_count=("target", "size"),
                last_era=("era", "max"),
            )
            w_arr = np.maximum(agg_df["w_sum"].to_numpy(dtype=np.float64), 1e-12)
            wt_arr = agg_df["wt_sum"].to_numpy(dtype=np.float64)
            persisted_arr = (wt_arr + smoothing_weight * self.global_mean_) / (w_arr + smoothing_weight)
            tickers_str = agg_df.index.astype(str)
            self.ticker_lookup_ = dict(zip(tickers_str.tolist(), persisted_arr.tolist()))
            self.persistence_table_ = (
                pd.DataFrame({
                    "ticker": tickers_str.to_numpy(),
                    "persisted_target": persisted_arr,
                    "obs_count": agg_df["obs_count"].to_numpy(dtype=np.int64),
                    "last_era": agg_df["last_era"].to_numpy(dtype=np.int64),
                })
                .sort_values("obs_count", ascending=False, kind="stable")
                .reset_index(drop=True)
            )
        else:
            raise ValueError(
                f"Unsupported persistence_mode '{self.persistence_mode}'; use None, 'last', 'rolling_mean', or 'expanding_mean'."
            )
        raw_train_pred = self._raw_persistence_lookup(X_df)
        self._fit_bias_correction(target_values, raw_train_pred, era_values)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "global_mean_"):
            raise ValueError("PersistenceEstimator must be fitted before calling predict.")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        era_values = (
            pd.to_numeric(X_df[self.era_col], errors="coerce").fillna(1).to_numpy(dtype=np.int32)
            if self.era_col in X_df.columns
            else None
        )
        raw_pred = self._raw_persistence_lookup(X_df)
        return self._apply_prediction_transform(raw_pred, era_values)


BaseNumeraiBaseline = BaseNumeraiEstimator
BinSnappingBaseline = BinSnappingEstimator
LagReversalBaseline = LagReversalEstimator
PersistenceBaseline = PersistenceEstimator


def run_baseline(
    baseline_name: str,
    pipeline: Any,
    param_space: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_eras: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_eras: pd.Series,
    feature_columns: List[str],
    *,
    search_mode: str = "bayes",
    eval_mode: str = "both",
    cv_object: Any = None,
    n_iter: int = 37,
    random_state: int = 42,
    verbose: int = 3,
    error_score: Any = -1.0,
    n_jobs: int = 1,
    walk_forward_step_eras: int = 20,
    walk_forward_window: str = "expanding",
    target_horizon_eras: int = 4,
    era_stride: int = 4,
    error_bin_config: Optional[Dict[str, Any]] = None,
    extraction_cache_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Dict[str, Any]], pd.DataFrame, Dict[str, Any], Any]:
    """Execute one baseline's default holdout, optional walk-forward holdout, and CV search in a picklable worker.

    Configures error binning and resets/collects the process-local `_CV_FOLD_METRIC_LOG` inside the worker so
    parallel execution under `joblib` (`loky`) preserves all extended CV metrics without loss.
    """
    from sklearn.model_selection import GridSearchCV
    from utils import (
        build_conditional_bayesian_spaces,
        build_quick_results_df,
        candidate_score,
        configure_error_binning,
        deduplicate_param_grid,
        evaluate_numerai_predictions,
        reset_cv_metric_log,
        walk_forward_predict,
    )

    try:
        from skopt import BayesSearchCV
    except ImportError:
        BayesSearchCV = None

    if error_bin_config is not None:
        configure_error_binning(**error_bin_config)
    if extraction_cache_config is not None:
        configure_extraction_cache(**extraction_cache_config)

    results_entries: Dict[str, Dict[str, Any]] = {}
    default_model = clone(pipeline).fit(X_train, y_train)

    train_preds = default_model.predict(X_train)
    val_preds = default_model.predict(X_validation)

    train_eval = evaluate_numerai_predictions(
        y_train,
        train_preds,
        eras=train_eras,
        features_df=X_train,
        feature_columns=feature_columns,
        label=f"{baseline_name} — Final Training Scores",
        verbose=True,
    )
    val_eval = evaluate_numerai_predictions(
        y_validation,
        val_preds,
        eras=validation_eras,
        features_df=X_validation,
        feature_columns=feature_columns,
        label=f"{baseline_name} — Final Validation Scores",
        verbose=True,
    )

    default_variant_name = f"{baseline_name} [Default]"
    if eval_mode in ("static", "both"):
        results_entries[default_variant_name] = {
            "variant": default_variant_name,
            "train_corr": train_eval["numerai_corr"],
            "validation_corr": val_eval["numerai_corr"],
            "validation_corr_pct": val_eval["numerai_corr_pct"],
            "positive_era_pct": val_eval["positive_era_pct"],
            "directional_accuracy_pct": val_eval["directional_accuracy_pct"],
            "validation_spearman": val_eval["spearman"],
            "validation_payout": val_eval["payout"],
            "validation_sharpe": val_eval["numerai_sharpe"],
            "validation_spearman_sharpe": val_eval["spearman_sharpe"],
            "validation_mae": val_eval["mae"],
            "forecast_bias": val_eval["forecast_bias"],
            "directional_hit_rate": val_eval["directional_hit_rate"],
            "era_corr_lag1_acf": val_eval["era_corr_lag1_acf"],
            "ljung_box_pvalue": val_eval["ljung_box_pvalue"],
            "arch_lm_pvalue": val_eval["arch_lm_pvalue"],
            "adf_pvalue": val_eval["adf_pvalue"],
            "era_corr_skew": val_eval["era_corr_skew"],
            "era_corr_kurtosis": val_eval["era_corr_kurtosis"],
            "jarque_bera_pvalue": val_eval["jarque_bera_pvalue"],
            "feature_exposure": val_eval["feature_exposure"],
            "max_drawdown": val_eval["max_drawdown"],
            "corr_gap": train_eval["numerai_corr"] - val_eval["numerai_corr"],
            "sanity_status": val_eval["sanity_status"],
            "validation_preds": val_preds,
            "validation_era_corrs": val_eval["numerai_era_corrs"],
            "params": default_model.named_steps["estimator"].get_params(),
        }

    if eval_mode in ("walk_forward", "both"):
        wf_preds = walk_forward_predict(
            pipeline,
            X_train,
            y_train,
            train_eras,
            X_validation,
            y_validation,
            validation_eras,
            step_eras=walk_forward_step_eras,
            window_mode=walk_forward_window,
            target_horizon_eras=target_horizon_eras,
            era_stride=era_stride,
        )
        wf_eval = evaluate_numerai_predictions(
            y_validation,
            wf_preds,
            eras=validation_eras,
            features_df=X_validation,
            feature_columns=feature_columns,
            label=f"{baseline_name} — Walk-Forward Validation Scores",
            verbose=True,
        )
        wf_variant_name = f"{baseline_name} [Walk-Forward]"
        results_entries[wf_variant_name] = {
            "variant": wf_variant_name,
            "train_corr": train_eval["numerai_corr"],
            "validation_corr": wf_eval["numerai_corr"],
            "validation_corr_pct": wf_eval["numerai_corr_pct"],
            "positive_era_pct": wf_eval["positive_era_pct"],
            "directional_accuracy_pct": wf_eval["directional_accuracy_pct"],
            "validation_spearman": wf_eval["spearman"],
            "validation_payout": wf_eval["payout"],
            "validation_sharpe": wf_eval["numerai_sharpe"],
            "validation_spearman_sharpe": wf_eval["spearman_sharpe"],
            "validation_mae": wf_eval["mae"],
            "forecast_bias": wf_eval["forecast_bias"],
            "directional_hit_rate": wf_eval["directional_hit_rate"],
            "era_corr_lag1_acf": wf_eval["era_corr_lag1_acf"],
            "ljung_box_pvalue": wf_eval["ljung_box_pvalue"],
            "arch_lm_pvalue": wf_eval["arch_lm_pvalue"],
            "adf_pvalue": wf_eval["adf_pvalue"],
            "era_corr_skew": wf_eval["era_corr_skew"],
            "era_corr_kurtosis": wf_eval["era_corr_kurtosis"],
            "jarque_bera_pvalue": wf_eval["jarque_bera_pvalue"],
            "feature_exposure": wf_eval["feature_exposure"],
            "max_drawdown": wf_eval["max_drawdown"],
            "corr_gap": train_eval["numerai_corr"] - wf_eval["numerai_corr"],
            "sanity_status": wf_eval["sanity_status"],
            "validation_preds": wf_preds,
            "validation_era_corrs": wf_eval["numerai_era_corrs"],
            "params": default_model.named_steps["estimator"].get_params(),
        }

    if search_mode == "bayes":
        if BayesSearchCV is None:
            raise ImportError("scikit-optimize is required for Bayesian search")
        search_spaces = (
            param_space
            if isinstance(param_space, list)
            else build_conditional_bayesian_spaces(param_space, total_n_iter=n_iter)
        )
        search = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_spaces,
            n_iter=n_iter,
            optimizer_kwargs={
                "base_estimator": "ET",
                "acq_func": "EI",
                "n_initial_points": min(4, max(1, n_iter // 3)),
            },
            scoring=candidate_score,
            cv=cv_object,
            refit=False,
            random_state=random_state,
            verbose=verbose,
            error_score=error_score,
            n_jobs=n_jobs,
        )
    else:
        deduped_grid = deduplicate_param_grid(param_space, base_params=pipeline.get_params(deep=True))
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=deduped_grid,
            scoring=candidate_score,
            cv=cv_object,
            refit=False,
            verbose=verbose,
            error_score=error_score,
            n_jobs=n_jobs,
        )

    reset_cv_metric_log()
    search.fit(X_train, y_train, groups=train_eras)
    cv_results_df = build_quick_results_df(search)
    search_summary_row = {
        "baseline": baseline_name,
        "search_mode": search_mode,
        "best_cv_corr": float(search.best_score_),
        "best_cv_corr_pct": round(float(search.best_score_) * 100.0, 2),
        "best_params": dict(search.best_params_),
    }
    return baseline_name, results_entries, cv_results_df, search_summary_row, default_model
