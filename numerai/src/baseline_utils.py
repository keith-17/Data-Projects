"""baseline_utils.py — BaseNumeraiEstimator parent class and BinSnappingEstimator / LagReversalEstimator child classes.

Shared feature extraction (`NumeraiExtractions`), correlation calculation (`compute_feature_target_correlations`),
search space helpers, and Numerai evaluation functions live in `utils.py`.
"""

from __future__ import annotations

import gc
import hashlib
import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
    _reclaim_cached_memory,
    _safe_autocorr,
    check_memory_budget,
    compute_feature_target_correlations,
)

_EXTRACTION_CACHE: "OrderedDict[str, Tuple[NumeraiExtractions, pd.DataFrame]]" = OrderedDict()
_EXTRACTION_CACHE_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "max_entries": 4,
}
_EXTRACTION_CACHE_SUSPENDED: bool = False


def set_extraction_cache_suspended(suspended: bool) -> bool:
    """Temporarily suspend reading/writing `_EXTRACTION_CACHE` (e.g. inside `walk_forward_predict`)."""
    global _EXTRACTION_CACHE_SUSPENDED
    _EXTRACTION_CACHE_SUSPENDED = bool(suspended)
    return _EXTRACTION_CACHE_SUSPENDED


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


def clear_extraction_cache() -> int:
    """Clear all entries from the per-process extraction cache, run GC, and return cleared count."""
    cleared = len(_EXTRACTION_CACHE)
    _EXTRACTION_CACHE.clear()
    gc.collect()
    return cleared


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
            cache_enabled = (
                (not _EXTRACTION_CACHE_SUSPENDED)
                and bool(_EXTRACTION_CACHE_CONFIG["enabled"])
                and int(_EXTRACTION_CACHE_CONFIG["max_entries"]) > 0
            )
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
            item_b = np.dtype(matrix_dtype).itemsize if matrix_dtype is not None else 4
            check_memory_budget(
                len(features_df) * len(feature_names) * item_b,
                context=f"{type(self).__name__}._prepare_fit_inputs",
            )
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

    def _mark_aborted_fit(
        self,
        exc: BaseException,
        y: Any = None,
        *,
        context: str = "",
    ) -> "BaseNumeraiEstimator":
        """Record a failed-safe aborted fit so ``predict`` returns a constant array penalized to ``-1.0``."""
        fallback = 0.5
        if y is not None:
            try:
                y_arr = np.asarray(y, dtype=np.float64).ravel()
                finite_y = y_arr[np.isfinite(y_arr)]
                if len(finite_y) > 0:
                    fallback = float(np.mean(finite_y))
            except Exception:
                fallback = 0.5
        self._aborted_reason_ = str(exc) or type(exc).__name__
        self._aborted_fallback_value_ = fallback
        _reclaim_cached_memory()
        gc.collect()
        ctx = context or f"{type(self).__name__}.fit"
        warnings.warn(
            f"[FAIL-SAFE ABORT | {ctx}] {self._aborted_reason_}",
            RuntimeWarning,
            stacklevel=2,
        )
        print(
            f"  [FAIL-SAFE ABORT | {ctx}] Returning constant fallback={fallback:.4f} "
            f"so CV assigns -1.0 without crashing: {self._aborted_reason_}"
        )
        return self

    def _aborted_predict(self, X: Any) -> Optional[np.ndarray]:
        """Return a constant prediction array if ``fit`` aborted under a time/memory fail-safe."""
        if getattr(self, "_aborted_reason_", None) is None:
            return None
        n_rows = len(X)
        fallback = float(getattr(self, "_aborted_fallback_value_", 0.5))
        return np.full(n_rows, fallback, dtype=np.float64)

    def _fit_with_fail_safe(
        self,
        fit_impl: Any,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
        eras: Optional[Any] = None,
        *,
        context: str = "",
    ) -> "BaseNumeraiEstimator":
        """Run ``fit_impl``, converting ``TimeoutError``/``MemoryError`` into a constant-prediction aborted fit.

        Inside ``BayesSearchCV`` (``n_points=1``), raising out of ``fit()`` aborts the entire search with
        ``ValueError: All the 1 fits failed`` even when ``error_score=-1.0``. Returning ``self`` in an
        aborted state lets ``predict()`` emit a constant array that ``candidate_score`` penalizes to ``-1.0``.
        """
        import time
        from utils import (
            candidate_param_key,
            is_candidate_aborted,
            is_cv_time_guard_active,
            mark_candidate_aborted,
            set_active_fold_start,
        )

        self._aborted_reason_ = None
        self._aborted_fallback_value_ = 0.5
        cv_active = is_cv_time_guard_active()
        cand_key = candidate_param_key(self) if cv_active else ""
        if cv_active and is_candidate_aborted(cand_key):
            return self._mark_aborted_fit(
                TimeoutError("Candidate already aborted on an earlier CV fold."),
                y=y,
                context=context or f"{type(self).__name__}.fit",
            )
        set_active_fold_start(time.perf_counter() if cv_active else None)
        try:
            return fit_impl(X, y, sample_weight=sample_weight, eras=eras)
        except (TimeoutError, MemoryError) as exc:
            if cv_active:
                mark_candidate_aborted(cand_key)
            return self._mark_aborted_fit(exc, y=y, context=context or f"{type(self).__name__}.fit")


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
        if hasattr(self, "weighted_bin_table_"):
            w_tbl = self.weighted_bin_table_
            col_idx = np.arange(w_tbl.shape[0], dtype=np.intp)
            return np.asarray(w_tbl[col_idx, selected_bins].sum(axis=1), dtype=np.float64)
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
        clean = np.nan_to_num(
            arr.astype(np.float32, copy=False),
            nan=2.0 if rescale_from_5 else middle_bin,
            posinf=4.0 if rescale_from_5 else float(n_bins - 1),
            neginf=0.0,
        )
        if rescale_from_5 and n_bins != 5:
            clean = clean * np.float32((float(n_bins) - 1.0) / 4.0)
        return np.clip(np.rint(clean), 0.0, float(n_bins - 1)).astype(np.uint8)

    @staticmethod
    def _accumulate_bins_chunk_numpy(
        chunk_bins: np.ndarray,
        w_chunk: np.ndarray,
        wy_chunk: np.ndarray,
        n_bins: int,
        w_sum: np.ndarray,
        wy_sum: np.ndarray,
    ) -> None:
        """Accumulate weighted bin counts & target sums via 5-GEMM multi-threaded BLAS (`mask_b.T @ [w, w*y]`)."""
        wy_2 = np.column_stack((w_chunk, wy_chunk)).astype(np.float32, copy=False)
        for bin_idx in range(n_bins):
            mask_b = (chunk_bins == bin_idx).astype(np.float32)
            acc = mask_b.T @ wy_2
            w_sum[:, bin_idx] += acc[:, 0]
            wy_sum[:, bin_idx] += acc[:, 1]
            del mask_b, acc
        del wy_2

    @staticmethod
    def _accumulate_bins_chunk_tf(
        tf_mod: Any,
        chunk_bins: np.ndarray,
        w_chunk: np.ndarray,
        wy_chunk: np.ndarray,
        n_bins: int,
        w_sum: np.ndarray,
        wy_sum: np.ndarray,
    ) -> None:
        """Accumulate weighted bin counts & target sums via TensorFlow 5-GEMM oneDNN/cuBLAS kernels."""
        bins_t = tf_mod.constant(chunk_bins, dtype=tf_mod.int32)
        wy_t = tf_mod.constant(np.column_stack((w_chunk, wy_chunk)), dtype=tf_mod.float32)
        for bin_idx in range(n_bins):
            mask_b = tf_mod.cast(tf_mod.equal(bins_t, bin_idx), tf_mod.float32)
            acc = tf_mod.linalg.matmul(mask_b, wy_t, transpose_a=True).numpy()
            w_sum[:, bin_idx] += acc[:, 0]
            wy_sum[:, bin_idx] += acc[:, 1]
            del mask_b, acc
        del bins_t, wy_t

    def _build_weighted_bin_table(self) -> np.ndarray:
        """Pre-scale `bin_table_` by feature aggregation weights once in `fit()` for fast `predict()`."""
        n_selected = self.bin_table_.shape[0]
        if self.agg_mode == "corr_weighted":
            abs_corrs = np.abs(self.selected_corrs_)
            corr_weight_sum = float(abs_corrs.sum())
            if corr_weight_sum > 1e-12:
                feat_w = (abs_corrs / corr_weight_sum).astype(np.float64)
            else:
                feat_w = np.full(n_selected, 1.0 / float(max(1, n_selected)), dtype=np.float64)
        elif self.agg_mode in (None, "none", "None", "mean"):
            feat_w = np.full(n_selected, 1.0 / float(max(1, n_selected)), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported agg_mode '{self.agg_mode}'; use 'mean' or 'corr_weighted'.")
        return (self.bin_table_ * feat_w[:, None]).astype(np.float64, copy=False)

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
        eras: Optional[Any] = None,
    ) -> "BinSnappingEstimator":
        return self._fit_with_fail_safe(
            self._fit_impl,
            X,
            y,
            sample_weight=sample_weight,
            eras=eras,
            context="BinSnappingEstimator.fit",
        )

    def _fit_impl(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[Any] = None,
        eras: Optional[Any] = None,
    ) -> "BinSnappingEstimator":
        features_df, _, target_values, _, sample_weights, era_values = self._prepare_fit_inputs(
            X, y, sample_weight=sample_weight, eras=eras, return_matrix=False
        )
        n_bins = max(2, int(self.n_bins if self.n_bins is not None else 5))
        selected_cols = list(self.selected_feature_names_)
        sub_df = features_df[selected_cols]
        n_rows = len(sub_df)
        n_selected = len(selected_cols)

        # Detect standard Numerai [0, 4] quantized input from a fast head/stride sample without full-matrix copy
        self.rescale_from_5_ = False
        if n_bins != 5 and n_rows > 0 and n_selected > 0:
            probe_df = sub_df[: min(n_rows, 4096)]
            probe_arr = probe_df.to_numpy(dtype=np.float32, copy=False)
            finite_probe = probe_arr[np.isfinite(probe_arr)]
            self.rescale_from_5_ = bool(
                len(finite_probe) > 0
                and float(np.min(finite_probe)) >= -0.01
                and float(np.max(finite_probe)) <= 4.01
            )
            del probe_arr, finite_probe

        if sample_weights is not None and float(sample_weights.sum()) > 0.0:
            w_arr = np.asarray(sample_weights, dtype=np.float64).ravel()
            self.global_mean_ = float(np.average(target_values, weights=w_arr))
        else:
            w_arr = np.ones(n_rows, dtype=np.float64)
            self.global_mean_ = float(np.mean(target_values)) if n_rows > 0 else 0.5
        wy_arr = w_arr * np.asarray(target_values, dtype=np.float64).ravel()

        # Use TensorFlow cuBLAS 5-GEMM only when a GPU is available and problem size justifies dispatch; else NumPy BLAS
        tf_mod = None
        if n_rows * n_selected >= 500_000:
            try:
                from utils import get_tf_backend

                _cand_tf = get_tf_backend()
                if _cand_tf is not None and _cand_tf.config.list_physical_devices("GPU"):
                    tf_mod = _cand_tf
            except Exception:
                tf_mod = None

        w_sum = np.zeros((n_selected, n_bins), dtype=np.float64)
        wy_sum = np.zeros((n_selected, n_bins), dtype=np.float64)
        batch_rows = 32_768
        for start_r in range(0, n_rows, batch_rows):
            end_r = min(n_rows, start_r + batch_rows)
            chunk_vals = sub_df[start_r:end_r].to_numpy(dtype=np.float32, copy=False)
            chunk_bins = self._snap_features_to_bins(
                chunk_vals,
                n_bins=n_bins,
                rescale_from_5=self.rescale_from_5_,
            )
            if tf_mod is not None:
                self._accumulate_bins_chunk_tf(
                    tf_mod, chunk_bins, w_arr[start_r:end_r], wy_arr[start_r:end_r], n_bins, w_sum, wy_sum
                )
            else:
                self._accumulate_bins_chunk_numpy(
                    chunk_bins, w_arr[start_r:end_r], wy_arr[start_r:end_r], n_bins, w_sum, wy_sum
                )
            del chunk_vals, chunk_bins

        smoothing_weight = max(0.0, float(self.smoothing or 0.0))
        bin_table = np.where(
            w_sum > 0.0,
            (wy_sum + smoothing_weight * self.global_mean_) / np.where(w_sum > 0.0, w_sum + smoothing_weight, 1.0),
            self.global_mean_,
        )

        self.bin_table_ = bin_table
        self.weighted_bin_table_ = self._build_weighted_bin_table()
        self._lookup_table_cache_: Optional[Dict[str, Dict[int, float]]] = None
        if self.bias_correction not in (None, False, "None", "none", ""):
            raw_train_pred = np.empty(n_rows, dtype=np.float64)
            for start_r in range(0, n_rows, batch_rows):
                end_r = min(n_rows, start_r + batch_rows)
                chunk_vals = sub_df[start_r:end_r].to_numpy(dtype=np.float32, copy=False)
                chunk_bins = self._snap_features_to_bins(
                    chunk_vals, n_bins=n_bins, rescale_from_5=self.rescale_from_5_
                )
                raw_train_pred[start_r:end_r] = self._compute_raw_snapped_pred(chunk_bins)
                del chunk_vals, chunk_bins
            self._fit_bias_correction(target_values, raw_train_pred, era_values)
            del raw_train_pred
        else:
            self.learned_bias_offset_ = 0.0

        del w_arr, wy_arr, w_sum, wy_sum, sub_df, features_df
        gc.collect()
        return self

    @property
    def lookup_table_(self) -> Dict[str, Dict[int, float]]:
        """Lazily build the per-feature bin lookup dictionary on demand (avoids CV fold dict overhead)."""
        cached = getattr(self, "_lookup_table_cache_", None)
        if cached is not None:
            return cached
        bin_table = getattr(self, "bin_table_", None)
        names = getattr(self, "selected_feature_names_", None)
        if bin_table is None or names is None:
            raise AttributeError("BinSnappingEstimator must be fitted before accessing lookup_table_.")
        n_bins = int(bin_table.shape[1])
        built = {
            names[feature_pos]: {bin_index: float(bin_table[feature_pos, bin_index]) for bin_index in range(n_bins)}
            for feature_pos in range(len(names))
        }
        self._lookup_table_cache_ = built
        return built

    def predict(self, X: Any) -> np.ndarray:
        aborted = self._aborted_predict(X)
        if aborted is not None:
            return aborted
        if not hasattr(self, "bin_table_"):
            raise ValueError("BinSnappingEstimator must be fitted before calling predict.")
        features_df, era_values = self._prepare_predict_df(X)
        selected_cols = list(self.selected_feature_names_)
        sub_df = features_df[selected_cols]
        n_rows = len(sub_df)
        n_bins = self.bin_table_.shape[1]
        rescale = bool(getattr(self, "rescale_from_5_", False))
        raw_pred = np.empty(n_rows, dtype=np.float64)
        batch_rows = 32_768
        for start_r in range(0, n_rows, batch_rows):
            end_r = min(n_rows, start_r + batch_rows)
            chunk_vals = sub_df[start_r:end_r].to_numpy(dtype=np.float32, copy=False)
            bin_indices = self._snap_features_to_bins(
                chunk_vals,
                n_bins=n_bins,
                rescale_from_5=rescale,
            )
            raw_pred[start_r:end_r] = self._compute_raw_snapped_pred(bin_indices)
            del chunk_vals, bin_indices
        del sub_df, features_df
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
        return self._fit_with_fail_safe(
            self._fit_impl,
            X,
            y,
            sample_weight=sample_weight,
            eras=eras,
            context="LagReversalEstimator.fit",
        )

    def _fit_impl(
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
        check_memory_budget(
            len(raw_features) * n_features * 8,
            context="LagReversalEstimator.fit",
        )
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
        aborted = self._aborted_predict(X)
        if aborted is not None:
            return aborted
        if not hasattr(self, "weights_"):
            raise ValueError("LagReversalEstimator must be fitted before calling predict.")
        features_df, era_values = self._prepare_predict_df(X)
        n_rows = len(features_df)
        active_idx = np.flatnonzero(np.abs(self.weights_) > 0.0)
        if len(active_idx) == 0 or n_rows == 0:
            del features_df
            return self._apply_prediction_transform(np.zeros(n_rows, dtype=np.float64), era_values)

        active_cols = [str(self.feature_names_in_[i]) for i in active_idx]
        active_w = self.weights_[active_idx].astype(np.float64, copy=False)
        active_mu = self.feature_means_[active_idx].astype(np.float64, copy=False)
        center_offset = float(np.dot(active_mu, active_w)) if self.center_features else 0.0
        sub_df = features_df[active_cols]
        is_int_sub = all(
            isinstance(dt, np.dtype) and np.issubdtype(dt, np.integer)
            for dt in sub_df.dtypes
        )

        # Row-batched exact float64 dot product using (X - mu) @ w = X @ w - mu . w (avoids ~55 GB full-matrix copies)
        raw_pred = np.empty(n_rows, dtype=np.float64)
        batch_rows = 32_768
        for start_r in range(0, n_rows, batch_rows):
            end_r = min(n_rows, start_r + batch_rows)
            chunk_f64 = sub_df[start_r:end_r].to_numpy(dtype=np.float64, copy=True)
            if not is_int_sub:
                bad_mask = ~np.isfinite(chunk_f64)
                if np.any(bad_mask):
                    bad_rows, bad_cols = np.nonzero(bad_mask)
                    chunk_f64[bad_rows, bad_cols] = active_mu[bad_cols]
            raw_pred[start_r:end_r] = (chunk_f64 @ active_w) - center_offset
            del chunk_f64
        del sub_df, features_df
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
        return self._fit_with_fail_safe(
            self._fit_impl,
            X,
            y,
            sample_weight=sample_weight,
            eras=eras,
            context="PersistenceEstimator.fit",
        )

    def _fit_impl(
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
        aborted = self._aborted_predict(X)
        if aborted is not None:
            return aborted
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


def _build_baseline_result_entry(
    *,
    variant_name: str,
    train_eval: Dict[str, Any],
    val_eval: Dict[str, Any],
    val_preds: np.ndarray,
    val_pnl: Dict[str, Any],
    val_dsr_info: Dict[str, Any],
    score_version: str,
    estimator_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the standardized comparison dictionary for one evaluated baseline variant."""
    return {
        "variant": variant_name,
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
        "validation_preds": np.asarray(val_preds, dtype=np.float32),
        "validation_era_corrs": val_eval["numerai_era_corrs"],
        "params": estimator_params,
    }


def _build_baseline_cv_search(
    pipeline: Any,
    param_space: Any,
    *,
    search_mode: str,
    cv_scorer: Any,
    cv_object: Any,
    n_iter: int,
    random_state: int,
    verbose: int,
    error_score: Any,
    n_jobs: int,
) -> Any:
    """Instantiate a configured `BayesSearchCV` or deduplicated `GridSearchCV` search object."""
    from sklearn.model_selection import GridSearchCV
    from utils import build_conditional_bayesian_spaces, deduplicate_param_grid

    if search_mode == "bayes":
        try:
            from skopt import BayesSearchCV
        except ImportError as exc:
            raise ImportError("scikit-optimize is required for Bayesian search") from exc
        search_spaces = (
            param_space
            if isinstance(param_space, list)
            else build_conditional_bayesian_spaces(param_space, total_n_iter=n_iter)
        )
        return BayesSearchCV(
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

    deduped_grid = deduplicate_param_grid(param_space, base_params=pipeline.get_params(deep=True))
    return GridSearchCV(
        estimator=pipeline,
        param_grid=deduped_grid,
        scoring=cv_scorer,
        cv=cv_object,
        refit=False,
        verbose=verbose,
        error_score=error_score,
        n_jobs=n_jobs,
    )


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
    walk_forward_step_eras: int = 52,
    walk_forward_window: str = "expanding",
    target_horizon_eras: int = 4,
    era_stride: int = 4,
    error_bin_config: Optional[Dict[str, Any]] = None,
    extraction_cache_config: Optional[Dict[str, Any]] = None,
    memory_guard_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Dict[str, Any]], pd.DataFrame, Dict[str, Any], Any]:
    """Execute one baseline's static default holdout, CV search, and optional tuned walk-forward holdout in a picklable worker.

    Configures error binning and resets/collects the process-local `_CV_FOLD_METRIC_LOG` inside the worker so
    parallel execution under `joblib` (`loky`) preserves all extended CV metrics without loss.
    """
    from utils import (
        build_quick_results_df,
        compute_deflated_sharpe_ratio,
        compute_pbo_cscv,
        configure_error_binning,
        configure_memory_guard,
        evaluate_numerai_predictions,
        make_candidate_scorer,
        prepare_tuned_pipeline,
        purge_train_before_validation,
        reset_cv_metric_log,
        simulate_stake_pnl,
        walk_forward_predict,
    )

    if error_bin_config is not None:
        configure_error_binning(**error_bin_config)
    if extraction_cache_config is not None:
        configure_extraction_cache(**extraction_cache_config)
    if memory_guard_config is not None:
        configure_memory_guard(**memory_guard_config)

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
    default_params = default_model.named_steps["estimator"].get_params()

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
    del default_model, train_preds

    if eval_mode in ("static", "both"):
        default_variant_name = f"{baseline_name} [Default]"
        val_pnl = simulate_stake_pnl(val_eval["numerai_era_corrs"], era_dates=val_era_dates)
        val_dsr_info = compute_deflated_sharpe_ratio(val_eval["numerai_era_corrs"], [val_eval["numerai_sharpe"]])
        results_entries[default_variant_name] = _build_baseline_result_entry(
            variant_name=default_variant_name,
            train_eval=train_eval,
            val_eval=val_eval,
            val_preds=val_preds,
            val_pnl=val_pnl,
            val_dsr_info=val_dsr_info,
            score_version=str(score_version),
            estimator_params=default_params,
        )

    clear_extraction_cache()
    cv_scorer = make_candidate_scorer(score_version=score_version)
    search = _build_baseline_cv_search(
        pipeline,
        param_space,
        search_mode=search_mode,
        cv_scorer=cv_scorer,
        cv_object=cv_object,
        n_iter=n_iter,
        random_state=random_state,
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
    del search
    clear_extraction_cache()

    tuned_pipeline = prepare_tuned_pipeline(pipeline, best_params)
    tuned_model = clone(tuned_pipeline).fit(X_train_static, y_train_static)
    tuned_params = tuned_model.named_steps["estimator"].get_params()
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
        results_entries[tuned_variant_name] = _build_baseline_result_entry(
            variant_name=tuned_variant_name,
            train_eval=tuned_train_eval,
            val_eval=tuned_val_eval,
            val_preds=tuned_val_preds,
            val_pnl=tuned_val_pnl,
            val_dsr_info=tuned_dsr_info,
            score_version=effective_ver,
            estimator_params=tuned_params,
        )

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
        results_entries[tuned_wf_variant_name] = _build_baseline_result_entry(
            variant_name=tuned_wf_variant_name,
            train_eval=tuned_train_eval,
            val_eval=tuned_wf_eval,
            val_preds=tuned_wf_preds,
            val_pnl=tuned_wf_pnl,
            val_dsr_info=tuned_wf_dsr_info,
            score_version=effective_ver,
            estimator_params=tuned_params,
        )

    # Strip any retained training buffers from `tuned_model` and clear extraction cache before returning
    tuned_est = tuned_model.named_steps["estimator"] if hasattr(tuned_model, "named_steps") else tuned_model
    for _buf_attr in ("_wf_train_X_", "_wf_train_y_", "_wf_train_eras_"):
        if hasattr(tuned_est, _buf_attr):
            setattr(tuned_est, _buf_attr, None)
    tuned_ext = getattr(tuned_est, "extractor_", None)
    if tuned_ext is not None:
        for _ext_attr in ("ticker_tail_df_", "era_means_tail_"):
            if hasattr(tuned_ext, _ext_attr):
                setattr(tuned_ext, _ext_attr, None)

    del X_train_static, y_train_static, train_eras_static, tuned_train_preds
    clear_extraction_cache()
    gc.collect()
    return baseline_name, results_entries, cv_results_df, search_summary_row, tuned_model


def _resolve_baseline_artifact_key(container: Mapping[str, Any], candidates: Tuple[str, ...]) -> Optional[str]:
    """Return the first matching baseline key present in `container` (supports both new and legacy display names)."""
    for key in candidates:
        if key in container:
            return key
    return None


def extract_baseline_interpretability(
    fitted_models: Mapping[str, Any],
    *,
    top_n: int = 20,
    feature_groups: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract compact interpretability tables, % share concentration, and group distribution summaries."""
    from utils import build_feature_importance_share_and_group_summary

    artifacts: Dict[str, Any] = {}
    cap = max(1, int(top_n))

    key_bin = _resolve_baseline_artifact_key(fitted_models, ("Bin-Snapping", "Baseline A (Bin-Snapping)"))
    if key_bin is not None:
        est_a = fitted_models[key_bin].named_steps["estimator"]
        lookup_df = pd.DataFrame(est_a.lookup_table_).T
        lookup_df.columns = [f"bin_{bin_index}" for bin_index in lookup_df.columns]
        lookup_df["train_corr"] = est_a.selected_corrs_
        corr_share_in = pd.DataFrame({
            "feature": lookup_df.index.astype(str).tolist(),
            "train_corr": lookup_df["train_corr"].to_numpy(dtype=np.float64),
        })
        artifacts[key_bin] = {
            "total_features": int(len(lookup_df)),
            "lookup_table_df": lookup_df.head(cap).copy(),
            "share_summary": build_feature_importance_share_and_group_summary(
                corr_share_in,
                feature_col="feature",
                importance_col="train_corr",
                feature_groups=feature_groups,
                top_n=cap,
            ),
        }

    key_lag = _resolve_baseline_artifact_key(fitted_models, ("Lag-Reversal", "Baseline E (Lag-Reversal)"))
    if key_lag is not None:
        est_e = fitted_models[key_lag].named_steps["estimator"]
        artifacts[key_lag] = {
            "n_kept_signals": int((est_e.inversion_signs_ > 0).sum()),
            "n_inverted_signals": int((est_e.inversion_signs_ < 0).sum()),
            "rule_summary_df": est_e.rule_summary_.head(cap).copy(),
            "share_summary": build_feature_importance_share_and_group_summary(
                est_e.rule_summary_,
                feature_col="feature",
                importance_col="effective_weight",
                feature_groups=feature_groups,
                top_n=cap,
            ),
        }

    key_pers = _resolve_baseline_artifact_key(fitted_models, ("Persistence", "Baseline P (Persistence)"))
    if key_pers is not None:
        est_p = fitted_models[key_pers].named_steps["estimator"]
        artifacts[key_pers] = {
            "persistence_mode": est_p.persistence_mode,
            "persistence_lag": est_p.persistence_lag,
            "global_mean": float(est_p.global_mean_),
            "tracked_tickers": int(len(est_p.ticker_lookup_)),
            "persistence_table_df": est_p.persistence_table_.head(min(15, cap)).copy(),
        }

    return artifacts


def display_baseline_interpretability(
    interpretability: Mapping[str, Any],
    *,
    results_dir: Optional[Any] = None,
    timestamp: str = "",
    experiment_name: str = "",
    save_results: bool = False,
) -> None:
    """Display and plot compact baseline interpretability artifacts extracted prior to model cleanup."""
    from utils import display_compact_dataframe

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    key_bin = _resolve_baseline_artifact_key(interpretability, ("Bin-Snapping", "Baseline A (Bin-Snapping)"))
    if key_bin is not None:
        info_a = interpretability[key_bin]
        lookup_table_df = info_a["lookup_table_df"]
        display_compact_dataframe(
            lookup_table_df.head(15),
            title=f"Bin-Snapping Learned Lookup Table (top {info_a['total_features']} features):",
        )

    key_lag = _resolve_baseline_artifact_key(interpretability, ("Lag-Reversal", "Baseline E (Lag-Reversal)"))
    if key_lag is not None:
        info_e = interpretability[key_lag]
        print(
            f"Lag-Reversal Signal Rules | kept (+1)={info_e['n_kept_signals']} | "
            f"inverted (-1)={info_e['n_inverted_signals']}"
        )
        display_compact_dataframe(info_e["rule_summary_df"].head(20))

    key_pers = _resolve_baseline_artifact_key(interpretability, ("Persistence", "Baseline P (Persistence)"))
    if key_pers is not None:
        info_p = interpretability[key_pers]
        print(
            f"Persistence | mode={info_p['persistence_mode']} | "
            f"lag={info_p['persistence_lag']} | global_mean={info_p['global_mean']:.4f} | "
            f"tracked_tickers={info_p['tracked_tickers']}"
        )
        display_compact_dataframe(info_p["persistence_table_df"].head(15))

    if plt is not None:
        out_dir = Path(results_dir) if results_dir is not None else None
        if key_lag is not None:
            top_rule_df = interpretability[key_lag]["rule_summary_df"].head(20).copy()
            top_rule_df["abs_weight"] = top_rule_df["effective_weight"].abs()
            plot_rule_df = top_rule_df.sort_values("abs_weight", ascending=True)
            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            bar_colors = ["#10b981" if w >= 0 else "#ef4444" for w in plot_rule_df["effective_weight"]]
            ax_imp.barh(plot_rule_df["feature"], plot_rule_df["effective_weight"], color=bar_colors, alpha=0.85)
            ax_imp.axvline(0.0, color="#64748b", linewidth=0.9)
            ax_imp.set_xlabel("Effective Signed Weight")
            ax_imp.set_title("Top 20 Lag-Reversal Features by Effective Weight")
            fig_imp.tight_layout()
            if save_results and out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                fig_imp.savefig(out_dir / f"{timestamp}_{experiment_name}_lag_reversal_weights.png", dpi=140)
            plt.show()
            plt.close(fig_imp)
            del fig_imp
            gc.collect()
        elif key_bin is not None:
            top_bin_df = interpretability[key_bin]["lookup_table_df"].head(20).copy()
            top_bin_df["abs_corr"] = top_bin_df["train_corr"].abs()
            plot_bin_df = top_bin_df.sort_values("abs_corr", ascending=True)
            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            bar_colors = ["#2563eb" if c >= 0 else "#ef4444" for c in plot_bin_df["train_corr"]]
            ax_imp.barh(plot_bin_df.index, plot_bin_df["train_corr"], color=bar_colors, alpha=0.85)
            ax_imp.axvline(0.0, color="#64748b", linewidth=0.9)
            ax_imp.set_xlabel("Training Target Correlation")
            ax_imp.set_title("Top 20 Bin-Snapping Features by Training Correlation")
            fig_imp.tight_layout()
            if save_results and out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                fig_imp.savefig(out_dir / f"{timestamp}_{experiment_name}_bin_snapping_corrs.png", dpi=140)
            plt.show()
            plt.close(fig_imp)
            del fig_imp
            gc.collect()

    from utils import plot_interpretability_share_and_groups

    if key_lag is not None and interpretability[key_lag].get("share_summary"):
        plot_interpretability_share_and_groups(
            interpretability[key_lag]["share_summary"],
            model_label="Lag-Reversal",
            metric_label="Abs Effective Weight",
            results_dir=results_dir,
            timestamp=timestamp,
            experiment_name=experiment_name,
            save_results=save_results,
        )
    elif key_bin is not None and interpretability[key_bin].get("share_summary"):
        plot_interpretability_share_and_groups(
            interpretability[key_bin]["share_summary"],
            model_label="Bin-Snapping",
            metric_label="Abs Train CORR",
            results_dir=results_dir,
            timestamp=timestamp,
            experiment_name=experiment_name,
            save_results=save_results,
        )


def _baseline_name_to_slug(baseline_name: Any) -> str:
    """Map a baseline display name (new or legacy) to a clean filename slug."""
    low = str(baseline_name).strip().lower()
    if "bin-snapping" in low or "bin_snapping" in low or "baseline a" in low:
        return "bin_snapping"
    if "lag-reversal" in low or "lag_reversal" in low or "baseline e" in low:
        return "lag_reversal"
    return "persistence"


def extract_cv_analysis_artifacts(
    cv_results_by_baseline: Mapping[str, pd.DataFrame],
    *,
    top_n: int = 10,
    results_dir: Optional[Any] = None,
    timestamp: str = "",
    experiment_name: str = "",
    search_mode: str = "bayes",
    save_results: bool = False,
    include_model_slug: bool = True,
) -> Dict[str, Any]:
    """Export full CV CSVs (if requested) and retain only lightweight top-N tables + trial stats for AFML."""
    top_candidates: Dict[str, pd.DataFrame] = {}
    trial_stats: Dict[str, Dict[str, Any]] = {}
    timing_records: Dict[str, pd.DataFrame] = {}
    saved_csv_paths: Dict[str, Path] = {}
    out_dir = Path(results_dir) if results_dir is not None else None
    cap = max(1, int(top_n))

    for baseline_name, cv_df in cv_results_by_baseline.items():
        top_df = cv_df.head(cap).copy()
        top_df.attrs = {}
        top_candidates[str(baseline_name)] = top_df
        timing_cols = [
            c
            for c in ("candidate_id", "rank_test_score", "mean_fit_time", "mean_score_time", "mean_test_score")
            if c in cv_df.columns
        ]
        if timing_cols:
            t_df = cv_df[timing_cols].copy()
            t_df.attrs = {}
            fit_arr = (
                pd.to_numeric(t_df["mean_fit_time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                if "mean_fit_time" in t_df.columns
                else np.zeros(len(t_df), dtype=np.float64)
            )
            score_arr = (
                pd.to_numeric(t_df["mean_score_time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                if "mean_score_time" in t_df.columns
                else np.zeros(len(t_df), dtype=np.float64)
            )
            t_df["total_time_s"] = fit_arr + score_arr
            timing_records[str(baseline_name)] = t_df.reset_index(drop=True)
        trial_srs = (
            cv_df["mean_sharpe"].dropna().to_numpy(dtype=np.float64)
            if ("mean_sharpe" in cv_df.columns and len(cv_df) > 0)
            else np.empty(0, dtype=np.float64)
        )
        trial_stats[str(baseline_name)] = {
            "trial_srs": trial_srs,
            "trials_era_corrs_df": cv_df.attrs.get("trials_era_corrs_df"),
        }
        if save_results and out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            if include_model_slug:
                baseline_slug = _baseline_name_to_slug(baseline_name)
                cv_filename = f"{timestamp}_{experiment_name}_{baseline_slug}_{search_mode}_cv.csv"
            else:
                cv_filename = f"{timestamp}_{experiment_name}_{search_mode}_cv.csv"
            cv_results_path = out_dir / cv_filename
            cv_df.to_csv(cv_results_path, index=False)
            saved_csv_paths[str(baseline_name)] = cv_results_path

    return {
        "top_candidates": top_candidates,
        "trial_stats": trial_stats,
        "timing_records": timing_records,
        "saved_csv_paths": saved_csv_paths,
    }


def display_top_cv_candidates(cv_analysis_artifacts: Mapping[str, Any]) -> None:
    """Display pre-extracted top CV candidates per baseline and print saved CSV paths."""
    from utils import display_compact_dataframe

    top_candidates = cv_analysis_artifacts.get("top_candidates", {})
    saved_csv_paths = cv_analysis_artifacts.get("saved_csv_paths", {})
    for baseline_name, top_df in top_candidates.items():
        display_compact_dataframe(top_df, title=f"Top CV candidates for {baseline_name}:")
        csv_path = saved_csv_paths.get(baseline_name)
        if csv_path is not None:
            print("Saved CV results CSV:", csv_path)
