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
    # Deterministic strided feature-value fingerprint so distinct feature matrices with identical eras/targets never collide
    n_rows = int(len(X_df))
    if n_rows > 0 and X_df.shape[1] > 0:
        sample_idx = np.unique(np.linspace(0, n_rows - 1, num=min(n_rows, 32), dtype=np.int64))
        num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c].dtype)]
        if num_cols:
            sample_vals = np.ascontiguousarray(X_df[num_cols].to_numpy(copy=False)[sample_idx])
            feat_hash = hashlib.blake2b(sample_vals.tobytes(), digest_size=16).hexdigest()
        else:
            feat_hash = "no_num"
    else:
        feat_hash = "empty"
    payload = json.dumps(
        {
            "params": canon_params,
            "n_rows": n_rows,
            "cols": [str(c) for c in X_df.columns],
            "era_min": era_min,
            "era_max": era_max,
            "era_hash": era_hash,
            "target_hash": target_hash,
            "feat_hash": feat_hash,
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
        if mode in ("uniqueness", "uniqueness_decay"):
            era_int = np.asarray(era_values, dtype=np.int64).ravel()
            u_eras = np.unique(era_int)
            horizon = max(1, int(getattr(self, "target_horizon_eras", 4) or 4))
            min_e = int(u_eras[0])
            span = int(u_eras[-1] - min_e) + horizon + 1
            diff = np.zeros(span + 1, dtype=np.int64)
            starts = (u_eras - min_e).astype(np.int64)
            np.add.at(diff, starts, 1)
            np.add.at(diff, starts + horizon, -1)
            concurrency = np.cumsum(diff[:-1])
            inv_c = np.where(concurrency > 0, 1.0 / np.maximum(concurrency, 1), 0.0)
            csum_inv = np.concatenate(([0.0], np.cumsum(inv_c)))
            era_u = (csum_inv[starts + horizon] - csum_inv[starts]) / float(horizon)
            if mode == "uniqueness_decay":
                # AFML Ch. 4.7: time-decay applied over cumulative uniqueness rather than raw calendar index
                cum_u = np.cumsum(era_u)
                max_cum_u = float(cum_u[-1]) if len(cum_u) else 1.0
                decay_rate = float(self.recency_decay or 0.005)
                era_u = era_u * np.exp(-decay_rate * (max_cum_u - cum_u))
            u_map = dict(zip(u_eras.tolist(), era_u.tolist()))
            era_series = pd.Series(era_int)
            row_u = era_series.map(u_map).to_numpy(dtype=np.float64)
            counts = era_series.map(era_series.value_counts()).to_numpy(dtype=np.float64)
            weights = row_u / np.maximum(counts, 1.0)
            mean_w = float(np.mean(weights))
            return (weights / mean_w) if mean_w > 1e-12 else np.ones(n_samples, dtype=np.float64)
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

    @staticmethod
    def _snap_features_to_bins(arr: np.ndarray, n_bins: int, rescale_from_5: bool = False) -> np.ndarray:
        """Snap Numerai features into integer bins `0..(n_bins - 1)`, symmetrically rescaling `[0, 4]` when `n_bins != 5`."""
        middle_bin = float(n_bins // 2)
        clean = np.nan_to_num(arr.astype(np.float64, copy=False), nan=2.0 if rescale_from_5 else middle_bin, posinf=4.0 if rescale_from_5 else float(n_bins - 1), neginf=0.0)
        if rescale_from_5 and n_bins != 5:
            clean = clean * ((float(n_bins) - 1.0) / 4.0)
        return np.clip(np.rint(clean), 0, n_bins - 1).astype(np.int64)

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
        selected_raw = feature_matrix[:, self.selected_indices_]
        finite_sel = selected_raw[np.isfinite(selected_raw)]
        # Detect standard Numerai [0, 4] quantized input so n_bins != 5 rescales symmetrically instead of clipping bins 2,3,4
        self.rescale_from_5_ = bool(
            n_bins != 5
            and len(finite_sel) > 0
            and float(np.min(finite_sel)) >= -0.01
            and float(np.max(finite_sel)) <= 4.01
        )

        if sample_weights is not None and float(sample_weights.sum()) > 0.0:
            self.global_mean_ = float(np.average(target_values, weights=sample_weights))
        else:
            self.global_mean_ = float(np.mean(target_values))

        n_selected = len(self.selected_indices_)
        bin_table = np.full((n_selected, n_bins), self.global_mean_, dtype=np.float64)
        selected_bins = self._snap_features_to_bins(selected_raw, n_bins=n_bins, rescale_from_5=self.rescale_from_5_)
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
        del selected_bins, feature_matrix
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "bin_table_"):
            raise ValueError("BinSnappingEstimator must be fitted before calling predict.")
        feature_matrix, era_values = self._prepare_predict_matrix(X)
        n_bins = self.bin_table_.shape[1]
        bin_indices = self._snap_features_to_bins(
            feature_matrix[:, self.selected_indices_],
            n_bins=n_bins,
            rescale_from_5=bool(getattr(self, "rescale_from_5_", False)),
        )
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
        _, raw_features, target_values, feature_names, sample_weights, era_values = self._prepare_fit_inputs(
            X, y, sample_weight=sample_weight, eras=eras
        )
        n_features = len(feature_names)
        finite_mask = np.isfinite(raw_features)
        if sample_weights is not None:
            row_w = np.where(np.isfinite(sample_weights) & (sample_weights > 0.0), sample_weights, 0.0).astype(np.float64)
            w_obs = np.where(finite_mask, row_w[:, None], 0.0)
            w_sums = w_obs.sum(axis=0)
            weighted_sums = (np.where(finite_mask, raw_features, 0.0) * w_obs).sum(axis=0)
            self.feature_means_ = np.divide(
                weighted_sums,
                w_sums,
                out=np.zeros(n_features, dtype=np.float64),
                where=w_sums > 1e-12,
            )
        else:
            row_w = None
            observed_counts = finite_mask.sum(axis=0)
            column_sums = np.where(finite_mask, raw_features, 0.0).sum(axis=0)
            self.feature_means_ = np.divide(
                column_sums,
                observed_counts,
                out=np.zeros(n_features, dtype=np.float64),
                where=observed_counts > 0,
            )
        clean_features = np.where(finite_mask, raw_features, self.feature_means_)

        # If era_weighting / sample_weights are supplied, re-estimate feature_corrs_ using weighted per-era correlations
        if row_w is not None and era_values is not None and self.correlation_method not in (None, "none", "None", ""):
            era_corr_rows: List[np.ndarray] = []
            era_weight_rows: List[float] = []
            w_frame = pd.DataFrame(clean_features, columns=feature_names)
            w_frame["_era_"] = era_values
            w_frame["_target_"] = target_values
            w_frame["_w_"] = row_w
            for _, era_df in w_frame.groupby("_era_", sort=True):
                ew = era_df["_w_"].to_numpy(dtype=np.float64, copy=False)
                ew_sum = float(ew.sum())
                if ew_sum <= 1e-12 or len(era_df) < 3:
                    continue
                ex = era_df[feature_names].to_numpy(dtype=np.float64, copy=False)
                ey = era_df["_target_"].to_numpy(dtype=np.float64, copy=False)
                mx = (ew @ ex) / ew_sum
                my = float(np.dot(ew, ey) / ew_sum)
                cx = ex - mx
                cy = ey - my
                denom = np.sqrt(np.maximum(np.sum(ew[:, None] * (cx ** 2), axis=0) * float(np.dot(ew, cy ** 2)), 0.0))
                ec = np.divide(cx.T @ (ew * cy), denom, out=np.zeros(n_features, dtype=np.float64), where=denom > 1e-12)
                era_corr_rows.append(ec)
                era_weight_rows.append(ew_sum)
            if era_corr_rows:
                ew_arr = np.asarray(era_weight_rows, dtype=np.float64)
                self.feature_corrs_ = np.average(np.vstack(era_corr_rows), axis=0, weights=ew_arr)
                ranking_scores = np.abs(self.feature_corrs_) if bool(self.abs_corr) else self.feature_corrs_
                n_sel = len(self.selected_indices_)
                self.selected_indices_ = np.argsort(-ranking_scores, kind="stable")[:n_sel]
                self.selected_corrs_ = self.feature_corrs_[self.selected_indices_]
                self.selected_feature_names_ = [feature_names[idx] for idx in self.selected_indices_]

        feature_acfs = np.zeros(n_features, dtype=np.float64)
        if era_values is not None and self.lag not in (None, "none", "None", 0):
            if self.acf_mode in (None, "none", "None", "feature_mean"):
                era_series_df = pd.DataFrame(clean_features, columns=feature_names).groupby(era_values, sort=True).mean()
            elif self.acf_mode == "feature_corr":
                per_era_corrs: Dict[Any, np.ndarray] = {}
                grouped_df = pd.DataFrame(clean_features, columns=feature_names)
                grouped_df["_era_"] = era_values
                grouped_df["_target_"] = target_values
                if row_w is not None:
                    grouped_df["_w_"] = row_w
                for era_key, era_df in grouped_df.groupby("_era_", sort=True):
                    era_x = era_df[feature_names].to_numpy(dtype=np.float64, copy=False)
                    era_y = era_df["_target_"].to_numpy(dtype=np.float64, copy=False)
                    if row_w is not None:
                        ew = era_df["_w_"].to_numpy(dtype=np.float64, copy=False)
                        ew_sum = float(ew.sum())
                        if ew_sum <= 1e-12:
                            per_era_corrs[era_key] = np.zeros(n_features, dtype=np.float64)
                            continue
                        centered_x = era_x - (ew @ era_x) / ew_sum
                        centered_y = era_y - float(np.dot(ew, era_y) / ew_sum)
                        denominator = np.sqrt(np.maximum(np.sum(ew[:, None] * (centered_x ** 2), axis=0) * float(np.dot(ew, centered_y ** 2)), 0.0))
                        per_era_corrs[era_key] = np.divide(
                            centered_x.T @ (ew * centered_y),
                            denominator,
                            out=np.zeros(n_features, dtype=np.float64),
                            where=denominator > 1e-12,
                        )
                    else:
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

    def _causal_persistence_lookup(
        self,
        X_df: pd.DataFrame,
        era_values: Optional[np.ndarray],
    ) -> np.ndarray:
        """Strictly causal per-ticker lookup: row at `(ticker, era)` only sees history with `era_hist <= era - eff_lag`."""
        n_rows = len(X_df)
        mode = getattr(self, "active_persistence_mode_", "none")
        if mode in ("none", "", "zero_rule") or not getattr(self, "ticker_history_", None) or self.ticker_col not in X_df.columns:
            return np.full(n_rows, self.global_mean_, dtype=np.float64)

        if era_values is None:
            # No era index supplied: fall back to settled end-of-training lookup table
            tickers = X_df[self.ticker_col].astype(str)
            return tickers.map(self.ticker_lookup_).fillna(self.global_mean_).to_numpy(dtype=np.float64)

        eff_lag = int(getattr(self, "effective_lag_eras_", 4))
        smoothing_weight = float(getattr(self, "smoothing_weight_", 0.0))
        win_eras = int(getattr(self, "effective_window_eras_", 4))
        global_mean = float(self.global_mean_)
        history_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = self.ticker_history_

        tickers_arr = X_df[self.ticker_col].astype(str).to_numpy()
        eras_arr = np.asarray(era_values, dtype=np.int64).ravel()
        out = np.full(n_rows, global_mean, dtype=np.float64)

        # Group query indices by ticker via stable argsort (zero Python per-row loop)
        order = np.argsort(tickers_arr, kind="stable")
        sorted_tickers = tickers_arr[order]
        change = np.empty(n_rows, dtype=bool)
        change[0] = True
        if n_rows > 1:
            change[1:] = sorted_tickers[1:] != sorted_tickers[:-1]
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], n_rows)

        for s, e in zip(starts, ends):
            t_key = str(sorted_tickers[s])
            hist_entry = history_map.get(t_key)
            if hist_entry is None:
                continue
            h_eras, h_targets, h_cum_w, h_cum_wt = hist_entry
            q_positions = order[s:e]
            q_cutoffs = eras_arr[q_positions] - eff_lag
            idx = np.searchsorted(h_eras, q_cutoffs, side="right") - 1
            valid = idx >= 0
            if not np.any(valid):
                continue
            v_pos = q_positions[valid]
            v_idx = idx[valid]

            if mode == "last":
                w_last = h_cum_w[v_idx] - np.where(v_idx > 0, h_cum_w[np.maximum(0, v_idx - 1)], 0.0)
                w_last = np.maximum(w_last, 1e-12)
                y_last = h_targets[v_idx]
                out[v_pos] = (w_last * y_last + smoothing_weight * global_mean) / (w_last + smoothing_weight)
            elif mode == "expanding_mean":
                w_sum = np.maximum(h_cum_w[v_idx], 1e-12)
                wt_sum = h_cum_wt[v_idx]
                out[v_pos] = (wt_sum + smoothing_weight * global_mean) / (w_sum + smoothing_weight)
            elif mode == "rolling_mean":
                left_idx = v_idx - win_eras
                w_left = np.where(left_idx >= 0, h_cum_w[np.maximum(0, left_idx)], 0.0)
                wt_left = np.where(left_idx >= 0, h_cum_wt[np.maximum(0, left_idx)], 0.0)
                w_sum = np.maximum(h_cum_w[v_idx] - w_left, 1e-12)
                wt_sum = h_cum_wt[v_idx] - wt_left
                out[v_pos] = (wt_sum + smoothing_weight * global_mean) / (w_sum + smoothing_weight)

        return out

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
        self.ticker_history_: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        mode = "none" if self.persistence_mode is None else str(self.persistence_mode).strip().lower()
        self.active_persistence_mode_ = mode
        raw_lag = int(self.persistence_lag) if self.persistence_lag not in (None, "none", "None", "", 0) else 0
        horizon_lag = max(1, int(self.target_horizon_eras or 4))
        self.effective_lag_eras_ = max(horizon_lag, raw_lag)
        self.effective_window_eras_ = max(
            1, int(self.persistence_window if self.persistence_window not in (None, "none", "None", "", 0) else 4)
        )
        self.smoothing_weight_ = max(0.0, float(self.smoothing or 0.0))

        if mode in ("none", "", "zero_rule") or self.ticker_col not in X_df.columns:
            self.persistence_table_ = pd.DataFrame(columns=["ticker", "persisted_target", "obs_count", "last_era"])
            raw_train_pred = np.full(len(X_df), self.global_mean_, dtype=np.float64)
            self._fit_bias_correction(target_values, raw_train_pred, era_values)
            return self
        if mode not in ("last", "rolling_mean", "expanding_mean"):
            raise ValueError(
                f"Unsupported persistence_mode '{self.persistence_mode}'; use None, 'last', 'rolling_mean', or 'expanding_mean'."
            )

        era_arr = (
            np.asarray(era_values, dtype=np.int64)
            if era_values is not None
            else np.ones(len(target_values), dtype=np.int64)
        )
        weights_arr = (
            sample_weights
            if sample_weights is not None
            else np.ones(len(target_values), dtype=np.float64)
        )
        tickers_arr = X_df[self.ticker_col].astype(str).to_numpy()

        # Sort by (ticker, era) once to build compact per-ticker causal prefix arrays
        order = np.lexsort((era_arr, tickers_arr))
        s_tickers = tickers_arr[order]
        s_eras = era_arr[order]
        s_targets = target_values[order]
        s_weights = np.maximum(weights_arr[order], 1e-12)

        n_obs = len(s_tickers)
        if n_obs > 0:
            change = np.empty(n_obs, dtype=bool)
            change[0] = True
            if n_obs > 1:
                change[1:] = s_tickers[1:] != s_tickers[:-1]
            starts = np.flatnonzero(change)
            ends = np.append(starts[1:], n_obs)

            table_rows: List[Dict[str, Any]] = []
            sm = self.smoothing_weight_
            gm = self.global_mean_
            win_k = self.effective_window_eras_
            for s, e in zip(starts, ends):
                t_key = str(s_tickers[s])
                te = s_eras[s:e]
                ty = s_targets[s:e]
                tw = s_weights[s:e]
                cw = np.cumsum(tw)
                cwt = np.cumsum(tw * ty)
                self.ticker_history_[t_key] = (te, ty, cw, cwt)

                if mode == "last":
                    w_val = float(tw[-1])
                    p_val = (w_val * float(ty[-1]) + sm * gm) / (w_val + sm)
                elif mode == "expanding_mean":
                    p_val = (float(cwt[-1]) + sm * gm) / (float(cw[-1]) + sm)
                else:
                    left_i = len(te) - win_k - 1
                    w_sub = float(cw[-1] - (cw[left_i] if left_i >= 0 else 0.0))
                    wt_sub = float(cwt[-1] - (cwt[left_i] if left_i >= 0 else 0.0))
                    p_val = (wt_sub + sm * gm) / (max(w_sub, 1e-12) + sm)

                self.ticker_lookup_[t_key] = float(p_val)
                table_rows.append({
                    "ticker": t_key,
                    "persisted_target": float(p_val),
                    "obs_count": int(e - s),
                    "last_era": int(te[-1]),
                })

            self.persistence_table_ = (
                pd.DataFrame(table_rows)
                .sort_values("obs_count", ascending=False, kind="stable")
                .reset_index(drop=True)
            )
        else:
            self.persistence_table_ = pd.DataFrame(columns=["ticker", "persisted_target", "obs_count", "last_era"])

        raw_train_pred = self._causal_persistence_lookup(X_df, era_values)
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
        raw_pred = self._causal_persistence_lookup(X_df, era_values)
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
    score_version: str = "v0_default",
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
        compute_deflated_sharpe_ratio,
        compute_pbo_cscv,
        configure_error_binning,
        deduplicate_param_grid,
        evaluate_numerai_predictions,
        make_candidate_scorer,
        prepare_tuned_pipeline,
        purge_train_before_validation,
        reset_cv_metric_log,
        simulate_stake_pnl,
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

    # Extract optional validation era -> calendar date mapping so R1393+/R1399+ rules trigger on v6.0 dates
    val_era_dates: Optional[pd.Series] = None
    if isinstance(X_validation, pd.DataFrame) and "date" in X_validation.columns and "era" in X_validation.columns:
        val_era_dates = X_validation.groupby("era", sort=True)["date"].first()

    # AFML Ch. 7.4.1: Purge trailing training eras whose target horizon overlaps the start of validation for static fits
    X_train_static, y_train_static, train_eras_static, purge_info = purge_train_before_validation(
        X_train,
        y_train,
        train_eras,
        validation_eras,
        target_horizon_eras=target_horizon_eras,
    )
    if purge_info["n_eras_purged"] > 0:
        print(
            f"  [AFML Purge | {baseline_name}] Purged {purge_info['n_eras_purged']} trailing train era(s) "
            f"(cutoff <= era {purge_info['cutoff_era']}, min_val_era={purge_info['min_val_era']}, "
            f"horizon={target_horizon_eras}) before static validation fits."
        )

    results_entries: Dict[str, Dict[str, Any]] = {}
    default_model = clone(pipeline).fit(X_train_static, y_train_static)

    train_preds = default_model.predict(X_train_static)
    val_preds = default_model.predict(X_validation)

    train_eval = evaluate_numerai_predictions(
        y_train_static,
        train_preds,
        eras=train_eras_static,
        features_df=X_train_static,
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
        val_pnl = simulate_stake_pnl(val_eval["numerai_era_corrs"], era_dates=val_era_dates)
        val_dsr_info = compute_deflated_sharpe_ratio(val_eval["numerai_era_corrs"], [val_eval["numerai_sharpe"]])
        results_entries[default_variant_name] = {
            "variant": default_variant_name,
            "train_corr": train_eval["numerai_corr"],
            "validation_corr": val_eval["numerai_corr"],
            "validation_corr_pct": val_eval["numerai_corr_pct"],
            "positive_era_pct": val_eval["positive_era_pct"],
            "era_consistency_pct": val_eval["era_consistency_pct"],
            "directional_accuracy_pct": val_eval["directional_accuracy_pct"],
            "validation_spearman": val_eval["spearman"],
            "validation_payout": val_eval["payout"],
            "validation_sharpe": val_eval["numerai_sharpe"],
            "validation_lo_sharpe": val_eval["lo_sharpe"],
            "validation_psr": round(float(val_dsr_info["psr"]), 4),
            "validation_dsr": round(float(val_dsr_info["dsr"]), 4),
            "effective_n_trials_onc": round(float(val_dsr_info["effective_n_trials"]), 2),
            "validation_fnc": val_eval["feature_neutral_corr"],
            "validation_fnc_sharpe": val_eval["fnc_sharpe"],
            "tuw_max_eras": val_eval["tuw_max_eras"],
            "hhi_pos": val_eval["hhi_pos"],
            "hhi_neg": val_eval["hhi_neg"],
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
            "corr_drawdown": val_eval["corr_drawdown"],
            "start_capital": round(float(val_pnl["start_capital"]), 2),
            "end_capital": round(float(val_pnl["end_capital"]), 2),
            "total_profit": round(float(val_pnl["total_profit"]), 2),
            "total_return_pct": round(float(val_pnl["total_return_pct"]), 2),
            "money_drawdown_pct": round(float(val_pnl["money_drawdown_pct"]), 2),
            "max_drawdown_gbp": round(float(val_pnl["max_drawdown_gbp"]), 2),
            "worst_round_gbp": round(float(val_pnl["worst_round_gbp"]), 2),
            "score_version": str(score_version),
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
        wf_pnl = simulate_stake_pnl(wf_eval["numerai_era_corrs"], era_dates=val_era_dates)
        wf_dsr_info = compute_deflated_sharpe_ratio(wf_eval["numerai_era_corrs"], [wf_eval["numerai_sharpe"]])
        wf_variant_name = f"{baseline_name} [Walk-Forward]"
        results_entries[wf_variant_name] = {
            "variant": wf_variant_name,
            "train_corr": train_eval["numerai_corr"],
            "validation_corr": wf_eval["numerai_corr"],
            "validation_corr_pct": wf_eval["numerai_corr_pct"],
            "positive_era_pct": wf_eval["positive_era_pct"],
            "era_consistency_pct": wf_eval["era_consistency_pct"],
            "directional_accuracy_pct": wf_eval["directional_accuracy_pct"],
            "validation_spearman": wf_eval["spearman"],
            "validation_payout": wf_eval["payout"],
            "validation_sharpe": wf_eval["numerai_sharpe"],
            "validation_lo_sharpe": wf_eval["lo_sharpe"],
            "validation_psr": round(float(wf_dsr_info["psr"]), 4),
            "validation_dsr": round(float(wf_dsr_info["dsr"]), 4),
            "effective_n_trials_onc": round(float(wf_dsr_info["effective_n_trials"]), 2),
            "validation_fnc": wf_eval["feature_neutral_corr"],
            "validation_fnc_sharpe": wf_eval["fnc_sharpe"],
            "tuw_max_eras": wf_eval["tuw_max_eras"],
            "hhi_pos": wf_eval["hhi_pos"],
            "hhi_neg": wf_eval["hhi_neg"],
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
            "corr_drawdown": wf_eval["corr_drawdown"],
            "start_capital": round(float(wf_pnl["start_capital"]), 2),
            "end_capital": round(float(wf_pnl["end_capital"]), 2),
            "total_profit": round(float(wf_pnl["total_profit"]), 2),
            "total_return_pct": round(float(wf_pnl["total_return_pct"]), 2),
            "money_drawdown_pct": round(float(wf_pnl["money_drawdown_pct"]), 2),
            "max_drawdown_gbp": round(float(wf_pnl["max_drawdown_gbp"]), 2),
            "worst_round_gbp": round(float(wf_pnl["worst_round_gbp"]), 2),
            "score_version": str(score_version),
            "corr_gap": train_eval["numerai_corr"] - wf_eval["numerai_corr"],
            "sanity_status": wf_eval["sanity_status"],
            "validation_preds": wf_preds,
            "validation_era_corrs": wf_eval["numerai_era_corrs"],
            "params": default_model.named_steps["estimator"].get_params(),
        }

    cv_scorer = make_candidate_scorer(score_version=score_version)
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
            scoring=cv_scorer,
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
            scoring=cv_scorer,
            cv=cv_object,
            refit=False,
            verbose=verbose,
            error_score=error_score,
            n_jobs=n_jobs,
        )

    reset_cv_metric_log()
    search.fit(X_train, y_train, groups=train_eras)
    cv_results_df = build_quick_results_df(search)
    best_params = dict(search.best_params_)
    effective_ver = (
        str(cv_results_df.loc[0, "score_version"])
        if ("score_version" in cv_results_df.columns and len(cv_results_df) > 0)
        else str(score_version)
    )
    search_summary_row = {
        "baseline": baseline_name,
        "search_mode": search_mode,
        "score_version": effective_ver,
        "best_cv_score": float(search.best_score_),
        "best_cv_corr": float(cv_results_df.loc[0, "mean_numerai_corr"])
        if ("mean_numerai_corr" in cv_results_df.columns and len(cv_results_df) > 0)
        else float(search.best_score_),
        "best_cv_score_pct": round(float(search.best_score_) * 100.0, 2),
        "best_params": best_params,
    }

    # Post-search refit of the best CV candidate on horizon-purged X_train -> evaluate on held-out X_validation
    tuned_pipeline = prepare_tuned_pipeline(pipeline, best_params)
    tuned_model = clone(tuned_pipeline).fit(X_train_static, y_train_static)
    tuned_train_preds = tuned_model.predict(X_train_static)
    tuned_val_preds = tuned_model.predict(X_validation)

    tuned_train_eval = evaluate_numerai_predictions(
        y_train_static,
        tuned_train_preds,
        eras=train_eras_static,
        features_df=X_train_static,
        feature_columns=feature_columns,
        label=f"{baseline_name} [Tuned] — Final Training Scores",
        verbose=True,
    )
    tuned_val_eval = evaluate_numerai_predictions(
        y_validation,
        tuned_val_preds,
        eras=validation_eras,
        features_df=X_validation,
        feature_columns=feature_columns,
        label=f"{baseline_name} [Tuned] — Final Validation Scores",
        verbose=True,
    )

    trial_srs = (
        cv_results_df["mean_sharpe"].dropna().to_numpy(dtype=np.float64)
        if ("mean_sharpe" in cv_results_df.columns and len(cv_results_df) > 0)
        else np.array([tuned_val_eval["numerai_sharpe"]], dtype=np.float64)
    )
    trials_matrix = cv_results_df.attrs.get("trials_era_corrs_df")
    pbo_info = compute_pbo_cscv(trials_matrix, random_state=random_state) if trials_matrix is not None else {"pbo": float(np.nan)}
    search_summary_row["pbo_cscv"] = round(float(pbo_info["pbo"]), 4) if np.isfinite(pbo_info["pbo"]) else float(np.nan)

    if eval_mode in ("static", "both"):
        tuned_val_pnl = simulate_stake_pnl(tuned_val_eval["numerai_era_corrs"], era_dates=val_era_dates)
        tuned_dsr_info = compute_deflated_sharpe_ratio(
            tuned_val_eval["numerai_era_corrs"],
            trial_srs,
            trials_matrix=trials_matrix,
            random_state=random_state,
        )
        tuned_variant_name = f"{baseline_name} [Tuned]"
        results_entries[tuned_variant_name] = {
            "variant": tuned_variant_name,
            "train_corr": tuned_train_eval["numerai_corr"],
            "validation_corr": tuned_val_eval["numerai_corr"],
            "validation_corr_pct": tuned_val_eval["numerai_corr_pct"],
            "positive_era_pct": tuned_val_eval["positive_era_pct"],
            "era_consistency_pct": tuned_val_eval["era_consistency_pct"],
            "directional_accuracy_pct": tuned_val_eval["directional_accuracy_pct"],
            "validation_spearman": tuned_val_eval["spearman"],
            "validation_payout": tuned_val_eval["payout"],
            "validation_sharpe": tuned_val_eval["numerai_sharpe"],
            "validation_lo_sharpe": tuned_val_eval["lo_sharpe"],
            "validation_psr": round(float(tuned_dsr_info["psr"]), 4),
            "validation_dsr": round(float(tuned_dsr_info["dsr"]), 4),
            "effective_n_trials_onc": round(float(tuned_dsr_info["effective_n_trials"]), 2),
            "validation_fnc": tuned_val_eval["feature_neutral_corr"],
            "validation_fnc_sharpe": tuned_val_eval["fnc_sharpe"],
            "tuw_max_eras": tuned_val_eval["tuw_max_eras"],
            "hhi_pos": tuned_val_eval["hhi_pos"],
            "hhi_neg": tuned_val_eval["hhi_neg"],
            "validation_spearman_sharpe": tuned_val_eval["spearman_sharpe"],
            "validation_mae": tuned_val_eval["mae"],
            "forecast_bias": tuned_val_eval["forecast_bias"],
            "directional_hit_rate": tuned_val_eval["directional_hit_rate"],
            "era_corr_lag1_acf": tuned_val_eval["era_corr_lag1_acf"],
            "ljung_box_pvalue": tuned_val_eval["ljung_box_pvalue"],
            "arch_lm_pvalue": tuned_val_eval["arch_lm_pvalue"],
            "adf_pvalue": tuned_val_eval["adf_pvalue"],
            "era_corr_skew": tuned_val_eval["era_corr_skew"],
            "era_corr_kurtosis": tuned_val_eval["era_corr_kurtosis"],
            "jarque_bera_pvalue": tuned_val_eval["jarque_bera_pvalue"],
            "feature_exposure": tuned_val_eval["feature_exposure"],
            "corr_drawdown": tuned_val_eval["corr_drawdown"],
            "start_capital": round(float(tuned_val_pnl["start_capital"]), 2),
            "end_capital": round(float(tuned_val_pnl["end_capital"]), 2),
            "total_profit": round(float(tuned_val_pnl["total_profit"]), 2),
            "total_return_pct": round(float(tuned_val_pnl["total_return_pct"]), 2),
            "money_drawdown_pct": round(float(tuned_val_pnl["money_drawdown_pct"]), 2),
            "max_drawdown_gbp": round(float(tuned_val_pnl["max_drawdown_gbp"]), 2),
            "worst_round_gbp": round(float(tuned_val_pnl["worst_round_gbp"]), 2),
            "score_version": effective_ver,
            "corr_gap": tuned_train_eval["numerai_corr"] - tuned_val_eval["numerai_corr"],
            "sanity_status": tuned_val_eval["sanity_status"],
            "validation_preds": tuned_val_preds,
            "validation_era_corrs": tuned_val_eval["numerai_era_corrs"],
            "params": tuned_model.named_steps["estimator"].get_params(),
        }

    if eval_mode in ("walk_forward", "both"):
        tuned_wf_preds = walk_forward_predict(
            tuned_pipeline,
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
        tuned_wf_eval = evaluate_numerai_predictions(
            y_validation,
            tuned_wf_preds,
            eras=validation_eras,
            features_df=X_validation,
            feature_columns=feature_columns,
            label=f"{baseline_name} [Tuned] — Walk-Forward Validation Scores",
            verbose=True,
        )
        tuned_wf_pnl = simulate_stake_pnl(tuned_wf_eval["numerai_era_corrs"], era_dates=val_era_dates)
        tuned_wf_dsr_info = compute_deflated_sharpe_ratio(
            tuned_wf_eval["numerai_era_corrs"],
            trial_srs,
            trials_matrix=trials_matrix,
            random_state=random_state,
        )
        tuned_wf_variant_name = f"{baseline_name} [Tuned Walk-Forward]"
        results_entries[tuned_wf_variant_name] = {
            "variant": tuned_wf_variant_name,
            "train_corr": tuned_train_eval["numerai_corr"],
            "validation_corr": tuned_wf_eval["numerai_corr"],
            "validation_corr_pct": tuned_wf_eval["numerai_corr_pct"],
            "positive_era_pct": tuned_wf_eval["positive_era_pct"],
            "era_consistency_pct": tuned_wf_eval["era_consistency_pct"],
            "directional_accuracy_pct": tuned_wf_eval["directional_accuracy_pct"],
            "validation_spearman": tuned_wf_eval["spearman"],
            "validation_payout": tuned_wf_eval["payout"],
            "validation_sharpe": tuned_wf_eval["numerai_sharpe"],
            "validation_lo_sharpe": tuned_wf_eval["lo_sharpe"],
            "validation_psr": round(float(tuned_wf_dsr_info["psr"]), 4),
            "validation_dsr": round(float(tuned_wf_dsr_info["dsr"]), 4),
            "effective_n_trials_onc": round(float(tuned_wf_dsr_info["effective_n_trials"]), 2),
            "validation_fnc": tuned_wf_eval["feature_neutral_corr"],
            "validation_fnc_sharpe": tuned_wf_eval["fnc_sharpe"],
            "tuw_max_eras": tuned_wf_eval["tuw_max_eras"],
            "hhi_pos": tuned_wf_eval["hhi_pos"],
            "hhi_neg": tuned_wf_eval["hhi_neg"],
            "validation_spearman_sharpe": tuned_wf_eval["spearman_sharpe"],
            "validation_mae": tuned_wf_eval["mae"],
            "forecast_bias": tuned_wf_eval["forecast_bias"],
            "directional_hit_rate": tuned_wf_eval["directional_hit_rate"],
            "era_corr_lag1_acf": tuned_wf_eval["era_corr_lag1_acf"],
            "ljung_box_pvalue": tuned_wf_eval["ljung_box_pvalue"],
            "arch_lm_pvalue": tuned_wf_eval["arch_lm_pvalue"],
            "adf_pvalue": tuned_wf_eval["adf_pvalue"],
            "era_corr_skew": tuned_wf_eval["era_corr_skew"],
            "era_corr_kurtosis": tuned_wf_eval["era_corr_kurtosis"],
            "jarque_bera_pvalue": tuned_wf_eval["jarque_bera_pvalue"],
            "feature_exposure": tuned_wf_eval["feature_exposure"],
            "corr_drawdown": tuned_wf_eval["corr_drawdown"],
            "start_capital": round(float(tuned_wf_pnl["start_capital"]), 2),
            "end_capital": round(float(tuned_wf_pnl["end_capital"]), 2),
            "total_profit": round(float(tuned_wf_pnl["total_profit"]), 2),
            "total_return_pct": round(float(tuned_wf_pnl["total_return_pct"]), 2),
            "money_drawdown_pct": round(float(tuned_wf_pnl["money_drawdown_pct"]), 2),
            "max_drawdown_gbp": round(float(tuned_wf_pnl["max_drawdown_gbp"]), 2),
            "worst_round_gbp": round(float(tuned_wf_pnl["worst_round_gbp"]), 2),
            "score_version": effective_ver,
            "corr_gap": tuned_train_eval["numerai_corr"] - tuned_wf_eval["numerai_corr"],
            "sanity_status": tuned_wf_eval["sanity_status"],
            "validation_preds": tuned_wf_preds,
            "validation_era_corrs": tuned_wf_eval["numerai_era_corrs"],
            "params": tuned_model.named_steps["estimator"].get_params(),
        }

    return baseline_name, results_entries, cv_results_df, search_summary_row, tuned_model
