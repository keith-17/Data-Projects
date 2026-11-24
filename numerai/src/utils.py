"""utils.py — Shared NumeraiExtractions, per-ticker time-series extraction, diagnostics Parquet export, and CORR sanity-checked evaluation suite."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numerai_tools.scoring import correlation_contribution, numerai_corr
from scipy import stats
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, r_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_lm
from statsmodels.tsa.stattools import adfuller

_ADFULLER_HAS_RESULT_OBJECT = "result_object" in inspect.signature(adfuller).parameters
_ACORR_LM_HAS_RESULT_OBJECT = "result_object" in inspect.signature(acorr_lm).parameters


def _run_adfuller(series_vals: np.ndarray, maxlag: int) -> Tuple[float, float]:
    """Run `adfuller` using `result_object=True` when supported, falling back to tuple indexing on older statsmodels."""
    if _ADFULLER_HAS_RESULT_OBJECT:
        adf_res = adfuller(series_vals, maxlag=maxlag, autolag=None, result_object=True)
        return float(adf_res.statistic), float(adf_res.pvalue)
    adf_res = adfuller(series_vals, maxlag=maxlag, autolag=None)
    return float(adf_res[0]), float(adf_res[1])


def _run_arch_lm_pvalue(corrs_arr: np.ndarray, nlags: int) -> float:
    """Compute Engle's ARCH-LM p-value via `acorr_lm(resid ** 2, ..., result_object=True)` with older-statsmodels fallback."""
    resid = corrs_arr - float(np.mean(corrs_arr))
    resid_sq = resid ** 2
    if _ACORR_LM_HAS_RESULT_OBJECT:
        lm_res = acorr_lm(resid_sq, nlags=nlags, result_object=True)
        return float(lm_res.lm_pvalue)
    lm_res = acorr_lm(resid_sq, nlags=nlags)
    return float(lm_res[1])

try:
    from skopt.space import Categorical as SkCategorical
    SKOPT_AVAILABLE = True
except ImportError:
    SkCategorical = None
    SKOPT_AVAILABLE = False

FORBIDDEN_FEATURE_COLS = frozenset({
    "era",
    "date",
    "data_type",
    "id",
    "numerai_ticker",
    "week_no",
    "year_horizon",
    "target",
})

CORR_SANITY_LOW_THRESHOLD = 0.005
CORR_SANITY_HIGH_THRESHOLD = 0.040


def _decode_structured(value: Any) -> Any:
    """Decode JSON-encoded tuples/lists/dicts used inside `skopt.space.Categorical`."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in ("None", "none", "null"):
            return None
        if (stripped.startswith("[") and stripped.endswith("]")) or (
            stripped.startswith("{") and stripped.endswith("}")
        ):
            try:
                parsed = json.loads(stripped)
                return tuple(parsed) if isinstance(parsed, list) else parsed
            except (ValueError, TypeError):
                return value
    return value


def _encode_skopt_item(val: Any) -> Any:
    if isinstance(val, (tuple, list, dict)):
        return json.dumps(val, sort_keys=True)
    return val


def _quantize_1d_array(
    arr: np.ndarray,
    strategy: str = "zscore",
    fitted_scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    """Quantize a 1D float array into `0..4` (`np.int8`) using frozen `fitted_scaler`, `KBinsDiscretizer`, or sample z-score."""
    if len(arr) == 0:
        return np.empty(0, dtype=np.int8)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(len(arr), 2, dtype=np.int8)
    clean = np.where(finite, arr, np.nanmedian(arr[finite]))
    if fitted_scaler is not None:
        scale_val = float(fitted_scaler.scale_[0]) if hasattr(fitted_scaler, "scale_") and fitted_scaler.scale_ is not None else 1.0
        if scale_val <= 1e-12:
            return np.full(len(arr), 2, dtype=np.int8)
        z_scores = fitted_scaler.transform(clean.reshape(-1, 1)).ravel()
        binned = np.rint(np.clip(z_scores, -2.0, 2.0) + 2.0)
        return np.clip(binned, 0, 4).astype(np.int8)
    std = float(np.std(clean))
    if std <= 1e-12:
        return np.full(len(arr), 2, dtype=np.int8)
    if strategy == "quantile" and len(np.unique(clean)) >= 5:
        raw_edges = np.quantile(clean, (0.2, 0.4, 0.6, 0.8))
        # Deduplicate collapsed quantile edges (ties) cleanly without KBinsDiscretizer UserWarnings
        unique_edges = raw_edges[np.concatenate(([True], np.diff(raw_edges) > 1e-8))]
        if len(unique_edges) > 0:
            binned = np.searchsorted(unique_edges, clean, side="right")
            return np.clip(binned, 0, 4).astype(np.int8)
    z_scores = (clean - float(np.mean(clean))) / std
    binned = np.rint(np.clip(z_scores, -2.0, 2.0) + 2.0)
    return np.clip(binned, 0, 4).astype(np.int8)


def _safe_autocorr(values: Any, lag: int = 1) -> float:
    """Zero-division-safe lag-k autocorrelation (avoids numpy `c /= stddev` RuntimeWarning on constant slices)."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    k = max(1, int(lag))
    if len(arr) <= k + 1:
        return 0.0
    x0 = arr[:-k]
    x1 = arr[k:]
    c0 = x0 - float(np.mean(x0))
    c1 = x1 - float(np.mean(x1))
    denom = float(np.sqrt(np.dot(c0, c0) * np.dot(c1, c1)))
    if not np.isfinite(denom) or denom <= 1e-12:
        return 0.0
    val = float(np.dot(c0, c1) / denom)
    return val if np.isfinite(val) else 0.0


def quantize_to_5bins(
    values: np.ndarray,
    strategy: str = "zscore",
    *,
    eras: Optional[Any] = None,
    quantize_scope: str = "era",
    fitted_scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    """Centralized 5-bin (`0..4` `np.int8`) quantizer supporting `'era'` (per-era cross-sectional), `'train_fit'` (frozen `StandardScaler`), and `'global'` scopes."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if len(arr) == 0:
        return np.empty(0, dtype=np.int8)
    scope = str(quantize_scope or "era").strip().lower()
    if scope == "train_fit" and fitted_scaler is not None:
        return _quantize_1d_array(arr, strategy=strategy, fitted_scaler=fitted_scaler)
    if scope == "era" and eras is not None:
        era_arr = np.asarray(eras).ravel()
        if len(era_arr) == len(arr):
            s = pd.Series(arr)
            out = s.groupby(era_arr, sort=False).transform(
                lambda grp: _quantize_1d_array(grp.to_numpy(dtype=np.float64, copy=False), strategy=strategy)
            )
            return out.to_numpy(dtype=np.int8)
    return _quantize_1d_array(arr, strategy=strategy)


_quantize_to_5bins = quantize_to_5bins


def _parse_pipe_tokens(option: Any, valid_tokens: Sequence[str]) -> set:
    """Parse a `'|'`-delimited string option (or boolean `True` for all valid tokens) into a set of active tokens."""
    if option in (None, False, "None", "none", "null", ""):
        return set()
    if option is True:
        return set(valid_tokens)
    tokens: set = set()
    for part in str(option).lower().replace(",", "|").split("|"):
        part = part.strip()
        if part in valid_tokens:
            tokens.add(part)
        elif part in ("all", "both", "true"):
            tokens.update(valid_tokens)
    return tokens


def _all_integer_dtypes(df: pd.DataFrame) -> bool:
    """True when every column in `df` has a non-nullable NumPy integer dtype (guaranteed no NaNs)."""
    return all(
        isinstance(dt, np.dtype) and np.issubdtype(dt, np.integer)
        for dt in df.dtypes
    )


def _all_numeric_dtypes(df: pd.DataFrame) -> bool:
    """True when every column in `df` is already numeric (`int8`, `int16`, `int32`, `float32`, etc.)."""
    return all(pd.api.types.is_numeric_dtype(dt) for dt in df.dtypes)


def _fill_missing_features(
    df: pd.DataFrame,
    missing_fill_value: Any = 2.0,
    era_series: Optional[pd.Series] = None,
    learned_fill_map: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Impute missing values using `sklearn.impute.SimpleImputer` for constant/mean/median and `groupby(era)` for `'era_mean'`/`'era_median'`.

    Fast-paths already-numeric DataFrames (`int8`/`int16`/`int32`/`float32`) with no missing values without
    copying or upcasting to `float64`. When imputation is required, operates in `np.float32`.
    """
    cols = df.columns
    if len(cols) == 0:
        return df, pd.Series(dtype=np.float32)

    all_numeric = _all_numeric_dtypes(df)
    if all_numeric:
        numeric_df = df
    else:
        numeric_df = df.copy()
        for col in cols:
            if not pd.api.types.is_numeric_dtype(numeric_df[col].dtype):
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce").astype(np.float32)

    has_missing = False if _all_integer_dtypes(numeric_df) else bool(numeric_df.isna().any().any())

    mode = "2.0" if missing_fill_value is None else str(missing_fill_value).strip().lower()
    const_val: Optional[float] = None
    if missing_fill_value is None or mode in ("middle", "2", "2.0"):
        const_val = 2.0
    elif mode in ("zero", "min", "0", "0.0"):
        const_val = 0.0
    elif mode in ("max", "4", "4.0"):
        const_val = 4.0
    elif isinstance(missing_fill_value, (int, float, np.number)):
        const_val = float(missing_fill_value)

    if const_val is not None:
        col_defaults = pd.Series(np.float32(const_val), index=cols, dtype=np.float32)
        if not has_missing:
            return numeric_df, col_defaults
        imputer = SimpleImputer(strategy="constant", fill_value=np.float32(const_val), keep_empty_features=True)
        imputed_arr = imputer.fit_transform(numeric_df.to_numpy(dtype=np.float32, copy=False))
        return pd.DataFrame(np.asarray(imputed_arr, dtype=np.float32), index=numeric_df.index, columns=cols), col_defaults

    if learned_fill_map is not None:
        col_defaults = learned_fill_map.reindex(cols).fillna(2.0).astype(np.float32)
    else:
        if not has_missing:
            stats_series = (
                numeric_df.median(numeric_only=True)
                if mode in ("median", "era_median")
                else numeric_df.mean(numeric_only=True)
            )
            col_defaults = stats_series.reindex(cols).fillna(2.0).astype(np.float32)
        else:
            base_strategy = "median" if mode in ("median", "era_median") else "mean"
            imputer = SimpleImputer(strategy=base_strategy, keep_empty_features=True)
            imputer.fit(numeric_df.to_numpy(dtype=np.float32, copy=False))
            stats_arr = np.where(np.isfinite(imputer.statistics_), imputer.statistics_, 2.0).astype(np.float32)
            col_defaults = pd.Series(stats_arr, index=cols, dtype=np.float32)

    if not has_missing:
        return numeric_df, col_defaults

    if mode in ("era_mean", "era_median") and era_series is not None:
        agg_fn = "mean" if mode == "era_mean" else "median"
        era_filled = numeric_df.astype(np.float32, copy=False).groupby(era_series).transform(
            lambda s: s.fillna(s.mean() if agg_fn == "mean" else s.median())
        )
        return era_filled.fillna(col_defaults).fillna(np.float32(2.0)).astype(np.float32, copy=False), col_defaults

    return numeric_df.astype(np.float32, copy=False).fillna(col_defaults).fillna(np.float32(2.0)), col_defaults


def drop_constant_dims(space: Mapping[str, Any], pipeline: Any) -> Dict[str, Any]:
    """Pin any single-category search dimension onto `pipeline` and remove it from the active Bayesian search space."""
    active: Dict[str, Any] = {}
    pinned: Dict[str, Any] = {}
    for key, dim in space.items():
        if hasattr(dim, "categories") and len(dim.categories) == 1:
            pinned[key] = _decode_structured(dim.categories[0])
        elif isinstance(dim, (list, tuple)) and len(dim) == 1:
            pinned[key] = _decode_structured(dim[0])
        else:
            active[key] = dim
    if pinned and pipeline is not None:
        pipeline.set_params(**pinned)
    return active


def prepare_bayesian_space(space: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a Bayesian search dictionary so lists/tuples/dicts are skopt-compatible."""
    if not SKOPT_AVAILABLE or SkCategorical is None:
        return dict(space)
    out: Dict[str, Any] = {}
    for key, val in space.items():
        if isinstance(val, (list, tuple)):
            out[key] = SkCategorical([_encode_skopt_item(item) for item in val])
        elif hasattr(val, "categories"):
            out[key] = SkCategorical([_encode_skopt_item(item) for item in val.categories])
        else:
            out[key] = val
    return out


def _is_inactive(val: Any) -> bool:
    return val in (None, False, 0, "None", "none", "null", "")


def _canonicalize_candidate_params(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Canonicalize inactive conditional parameters to a sentinel so `deduplicate_param_grid` prunes no-op combinations."""
    canon = dict(candidate)
    if _is_inactive(canon.get("estimator__extractor__sparse_selector")):
        if "estimator__extractor__sparse_alpha" in canon:
            canon["estimator__extractor__sparse_alpha"] = "__INACTIVE__"
    if not bool(canon.get("estimator__extractor__include_rolling_era", False)):
        if "estimator__extractor__rolling_era_windows" in canon:
            canon["estimator__extractor__rolling_era_windows"] = "__INACTIVE__"
    if not bool(canon.get("estimator__extractor__include_lag_diff", False)):
        if "estimator__extractor__era_lag_scales" in canon:
            canon["estimator__extractor__era_lag_scales"] = "__INACTIVE__"
    if _is_inactive(canon.get("estimator__extractor__fourier_order")):
        if "estimator__extractor__fourier_period_eras" in canon:
            canon["estimator__extractor__fourier_period_eras"] = "__INACTIVE__"
    tk_feat = canon.get("estimator__extractor__ticker_features")
    if _is_inactive(tk_feat):
        if "estimator__extractor__ticker_windows" in canon:
            canon["estimator__extractor__ticker_windows"] = "__INACTIVE__"
        if "estimator__extractor__ticker_lag_steps" in canon:
            canon["estimator__extractor__ticker_lag_steps"] = "__INACTIVE__"
    elif tk_feat is not True and "raw_lags" not in str(tk_feat).lower() and "all" not in str(tk_feat).lower():
        if "estimator__extractor__ticker_lag_steps" in canon:
            canon["estimator__extractor__ticker_lag_steps"] = "__INACTIVE__"
    if _is_inactive(canon.get("estimator__extractor__correlation_method")):
        if "estimator__extractor__abs_corr" in canon:
            canon["estimator__extractor__abs_corr"] = "__INACTIVE__"
    if str(canon.get("estimator__era_weighting") or "").strip().lower() != "recency_exp":
        if "estimator__recency_decay" in canon:
            canon["estimator__recency_decay"] = "__INACTIVE__"
    if _is_inactive(canon.get("estimator__correlation_method")):
        if "estimator__abs_corr" in canon:
            canon["estimator__abs_corr"] = "__INACTIVE__"
    if _is_inactive(canon.get("estimator__walk_forward_step_eras")):
        if "estimator__walk_forward_window" in canon:
            canon["estimator__walk_forward_window"] = "__INACTIVE__"
    p_mode = canon.get("estimator__persistence_mode")
    if str(p_mode or "").strip().lower() != "rolling_mean":
        if "estimator__persistence_window" in canon:
            canon["estimator__persistence_window"] = "__INACTIVE__"
    if _is_inactive(p_mode) or str(p_mode).strip().lower() == "zero_rule":
        for p_key in ("estimator__persistence_lag", "estimator__persistence_window", "estimator__smoothing"):
            if p_key in canon:
                canon[p_key] = "__INACTIVE__"
    return canon


def build_conditional_bayesian_spaces(
    space: Mapping[str, Any],
    total_n_iter: int = 12,
) -> Any:
    """Split a Bayesian search space into conditional `(subspace, n_iter)` tuples so inactive child parameters are not wasted."""
    prepared = prepare_bayesian_space(space)
    if not SKOPT_AVAILABLE or SkCategorical is None or int(total_n_iter) < 4:
        return prepared

    def _cats(dim: Any) -> List[Any]:
        return list(dim.categories) if hasattr(dim, "categories") else []

    # 1. PersistenceEstimator conditional split (`None` vs `last`/`expanding_mean` vs `rolling_mean`)
    if "estimator__persistence_mode" in prepared:
        modes = _cats(prepared["estimator__persistence_mode"])
        if len(modes) >= 2:
            subspaces: List[Tuple[Dict[str, Any], int]] = []
            zero_modes = [m for m in modes if _is_inactive(m) or str(m).strip().lower() == "zero_rule"]
            non_roll_modes = [m for m in modes if str(m).strip().lower() in ("last", "expanding_mean")]
            roll_modes = [m for m in modes if str(m).strip().lower() == "rolling_mean"]
            remaining_iter = int(total_n_iter)
            if zero_modes:
                sub = dict(prepared)
                sub["estimator__persistence_mode"] = SkCategorical(zero_modes)
                if "estimator__persistence_lag" in sub:
                    sub["estimator__persistence_lag"] = SkCategorical([None])
                if "estimator__persistence_window" in sub:
                    sub["estimator__persistence_window"] = SkCategorical([None])
                if "estimator__smoothing" in sub:
                    sub["estimator__smoothing"] = SkCategorical([0.0])
                budget = 1
                subspaces.append((sub, budget))
                remaining_iter = max(1, remaining_iter - budget)
            if non_roll_modes and roll_modes:
                half = max(1, remaining_iter // 2)
                sub_nr = dict(prepared)
                sub_nr["estimator__persistence_mode"] = SkCategorical(non_roll_modes)
                if "estimator__persistence_window" in sub_nr:
                    sub_nr["estimator__persistence_window"] = SkCategorical([None])
                subspaces.append((sub_nr, half))
                sub_r = dict(prepared)
                sub_r["estimator__persistence_mode"] = SkCategorical(roll_modes)
                subspaces.append((sub_r, max(1, remaining_iter - half)))
            elif non_roll_modes:
                sub_nr = dict(prepared)
                sub_nr["estimator__persistence_mode"] = SkCategorical(non_roll_modes)
                if "estimator__persistence_window" in sub_nr:
                    sub_nr["estimator__persistence_window"] = SkCategorical([None])
                subspaces.append((sub_nr, remaining_iter))
            elif roll_modes:
                sub_r = dict(prepared)
                sub_r["estimator__persistence_mode"] = SkCategorical(roll_modes)
                subspaces.append((sub_r, remaining_iter))
            if len(subspaces) >= 2:
                return subspaces

    # 2. Extractor-backed estimators: gate `sparse_alpha` on `sparse_selector` so continuous alpha trials aren't wasted when off
    if "estimator__extractor__sparse_selector" in prepared and "estimator__extractor__sparse_alpha" in prepared:
        sel_cats = _cats(prepared["estimator__extractor__sparse_selector"])
        off_cats = [c for c in sel_cats if _is_inactive(c)]
        on_cats = [c for c in sel_cats if not _is_inactive(c)]
        if off_cats and on_cats:
            half = max(2, int(total_n_iter) // 2)
            sub_off = dict(prepared)
            sub_off["estimator__extractor__sparse_selector"] = SkCategorical(off_cats)
            sub_off["estimator__extractor__sparse_alpha"] = SkCategorical([0.01])
            sub_on = dict(prepared)
            sub_on["estimator__extractor__sparse_selector"] = SkCategorical(on_cats)
            return [(sub_off, half), (sub_on, max(2, int(total_n_iter) - half))]

    return prepared


def deduplicate_param_grid(
    param_grid: Mapping[str, Sequence[Any]],
    *,
    base_params: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Deduplicate a `GridSearchCV` parameter dictionary, pruning conditionally inactive no-op combinations."""
    del base_params
    grid = list(ParameterGrid(dict(param_grid)))
    seen: set = set()
    kept: List[Dict[str, Any]] = []
    for candidate in grid:
        canonical = _canonicalize_candidate_params(candidate)
        sig = hashlib.md5(
            json.dumps({k: repr(v) for k, v in sorted(canonical.items())}, sort_keys=True).encode()
        ).hexdigest()
        if sig not in seen:
            seen.add(sig)
            kept.append(candidate)
    return [{key: [val] for key, val in candidate.items()} for candidate in kept]


def _pandas_mean_corr_vectorized(
    X: Any,
    y: Any,
    era_values: Any,
) -> Optional[np.ndarray]:
    """Vectorized per-era Pearson correlation mean using stable argsort + `np.add.reduceat` in float64.

    Replicates legacy `groupby(eras)` + `corrwith` semantics: skips eras with `< 3` rows or target
    sample std (`ddof=1`) `<= 1e-12`, zeroes correlations where per-era denominator `<= 1e-12`, and
    averages over valid eras only. Returns `None` when no valid eras exist.
    """
    era_arr = np.asarray(era_values).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_rows = len(y_arr)
    if n_rows < 3 or len(era_arr) != n_rows:
        return None

    is_df = isinstance(X, pd.DataFrame)
    n_cols = X.shape[1]
    if n_cols == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(era_arr, kind="mergesort")
    sorted_eras = era_arr[order]
    y_sorted = y_arr[order]

    era_change = np.empty(n_rows, dtype=bool)
    era_change[0] = True
    era_change[1:] = sorted_eras[1:] != sorted_eras[:-1]
    starts = np.flatnonzero(era_change)
    counts = np.diff(np.append(starts, n_rows))

    y_sum = np.add.reduceat(y_sorted, starts)
    y_mean = y_sum / counts
    y_centered = y_sorted - np.repeat(y_mean, counts)
    y_ss = np.add.reduceat(y_centered * y_centered, starts)
    y_std = np.sqrt(np.where(counts > 1, y_ss / np.maximum(counts - 1, 1), 0.0))

    valid_era_mask = (counts >= 3) & (y_std > 1e-12)
    if not np.any(valid_era_mask):
        return None

    result = np.zeros(n_cols, dtype=np.float64)
    col_names = list(X.columns) if is_df else None
    chunk_size = 256
    for c0 in range(0, n_cols, chunk_size):
        c1 = min(n_cols, c0 + chunk_size)
        if is_df and col_names is not None:
            x_chunk = X[col_names[c0:c1]].to_numpy(dtype=np.float64, copy=False)
        else:
            x_chunk = np.asarray(X[:, c0:c1], dtype=np.float64)
        x_sorted = x_chunk[order]
        x_sum = np.add.reduceat(x_sorted, starts, axis=0)
        x_mean = x_sum / counts[:, None]
        x_centered = x_sorted - np.repeat(x_mean, counts, axis=0)
        x_ss = np.add.reduceat(x_centered * x_centered, starts, axis=0)
        xy_cov = np.add.reduceat(x_centered * y_centered[:, None], starts, axis=0)
        denom = np.sqrt(x_ss * y_ss[:, None])
        era_corrs = np.divide(xy_cov, denom, out=np.zeros_like(xy_cov), where=denom > 1e-12)
        era_corrs = np.nan_to_num(era_corrs, nan=0.0, posinf=0.0, neginf=0.0)
        result[c0:c1] = era_corrs[valid_era_mask].mean(axis=0)
    return result


def _pooled_pearson_corr(
    df: pd.DataFrame,
    target_values: np.ndarray,
    *,
    rank_columns: bool = False,
    chunk_size: int = 64,
) -> np.ndarray:
    """Cancellation-free, memory-bounded pooled Pearson/Spearman correlation in float64 chunks (never calls `sqrt` on negative variance)."""
    cols = list(df.columns)
    n_features = len(cols)
    if n_features == 0:
        return np.empty(0, dtype=np.float64)

    y_f64 = np.asarray(target_values, dtype=np.float64).ravel()
    if rank_columns:
        y_f64 = stats.rankdata(y_f64).astype(np.float64, copy=False)
    y_centered = y_f64 - float(np.mean(y_f64))
    y_ss = float(np.dot(y_centered, y_centered))
    if y_ss <= 1e-12:
        return np.zeros(n_features, dtype=np.float64)

    corrs = np.zeros(n_features, dtype=np.float64)
    for start in range(0, n_features, chunk_size):
        chunk_cols = cols[start : start + chunk_size]
        x_chunk = df[chunk_cols].to_numpy(dtype=np.float64, copy=False)
        if rank_columns:
            x_chunk = np.apply_along_axis(stats.rankdata, 0, x_chunk).astype(np.float64, copy=False)
        x_centered = x_chunk - np.mean(x_chunk, axis=0)
        x_ss = np.sum(x_centered * x_centered, axis=0)
        denom = np.sqrt(np.maximum(x_ss * y_ss, 0.0))
        xy_cov = x_centered.T @ y_centered
        chunk_corrs = np.divide(xy_cov, denom, out=np.zeros(len(chunk_cols), dtype=np.float64), where=denom > 1e-12)
        corrs[start : start + len(chunk_cols)] = np.nan_to_num(chunk_corrs, nan=0.0, posinf=0.0, neginf=0.0)
    return corrs


def _is_constant_1d(arr: np.ndarray) -> bool:
    """True when a 1D numeric array has fewer than 2 finite elements or zero variance."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return True
    if bool(np.all(a == a[0])):
        return True
    return float(np.nanstd(a)) <= 1e-12


def _safe_numerai_corr_frame(features_df: pd.DataFrame, target_series: pd.Series) -> pd.Series:
    """Compute `numerai_corr` while masking out zero-variance columns so `c /= stddev` never divides by zero."""
    cols = list(features_df.columns)
    out = pd.Series(0.0, index=cols, dtype=np.float64)
    if len(cols) == 0 or _is_constant_1d(target_series.to_numpy(dtype=np.float64, copy=False)):
        return out
    mat = features_df.to_numpy(dtype=np.float32, copy=False)
    if mat.shape[0] < 3:
        return out
    col_stds = np.nanstd(mat, axis=0)
    nonconst_mask = np.isfinite(col_stds) & (col_stds > 1e-12)
    if not np.any(nonconst_mask):
        return out
    active_cols = [col for col, keep in zip(cols, nonconst_mask) if keep]
    active_scores = numerai_corr(features_df[active_cols].astype(np.float32, copy=False), target_series).fillna(0.0)
    out.loc[active_cols] = active_scores.reindex(active_cols).fillna(0.0).to_numpy(dtype=np.float64)
    return out


def compute_feature_target_correlations(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    correlation_method: Optional[str] = "pandas_mean",
    era_series: Optional[pd.Series] = None,
) -> pd.Series:
    """Compute feature-target correlations with `None`, `'pandas_mean'`, `'numerai_mean'`, `'pearson'`, or `'spearman'`."""
    feature_cols = list(features_df.columns)
    n_features = len(feature_cols)
    if n_features == 0:
        return pd.Series(dtype=np.float64)

    method = "none" if correlation_method is None else str(correlation_method).strip().lower()
    if method in ("none", ""):
        return pd.Series(np.ones(n_features, dtype=np.float64), index=feature_cols)

    clean_df, _ = _fill_missing_features(features_df, missing_fill_value=2.0)
    clean_target = pd.Series(np.asarray(target_series, dtype=np.float32).ravel(), index=clean_df.index, name="target")

    if method == "pandas_mean":
        if era_series is not None and len(era_series) == len(clean_df):
            vec_corrs = _pandas_mean_corr_vectorized(clean_df, clean_target.to_numpy(dtype=np.float64), era_series)
            if vec_corrs is not None:
                return pd.Series(vec_corrs, index=feature_cols, dtype=np.float64)
        corrs = _pooled_pearson_corr(clean_df, clean_target.to_numpy(dtype=np.float64), rank_columns=False)
        return pd.Series(corrs, index=feature_cols, dtype=np.float64)

    if method == "numerai_mean":
        if era_series is not None and len(era_series) == len(clean_df):
            eras = pd.Series(np.asarray(era_series).ravel(), index=clean_df.index)
            per_era_num_corrs: List[pd.Series] = []
            for _, era_features in clean_df.groupby(eras, sort=True):
                era_target = clean_target.loc[era_features.index]
                if len(era_features) < 3 or _is_constant_1d(era_target.to_numpy(dtype=np.float64, copy=False)):
                    continue
                per_era_num_corrs.append(_safe_numerai_corr_frame(era_features, era_target))
            if per_era_num_corrs:
                return pd.concat(per_era_num_corrs, axis=1).mean(axis=1).reindex(feature_cols).fillna(0.0)
        return _safe_numerai_corr_frame(clean_df, clean_target).reindex(feature_cols).fillna(0.0)

    target_values = clean_target.to_numpy(dtype=np.float64, copy=False)
    if _is_constant_1d(target_values):
        return pd.Series(0.0, index=feature_cols, dtype=np.float64)
    if method == "spearman":
        corrs = _pooled_pearson_corr(clean_df, target_values, rank_columns=True)
        return pd.Series(corrs, index=feature_cols, dtype=np.float64)

    corrs = _pooled_pearson_corr(clean_df, target_values, rank_columns=False)
    return pd.Series(corrs, index=feature_cols, dtype=np.float64)


class NumeraiExtractions(TransformerMixin, BaseEstimator):
    """Fold-safe Numerai tabular/era/ticker feature extractor (`estimator__extractor__*` or `extractor__*`)."""

    def __init__(
        self,
        era_col: str = "era",
        target_col: str = "target",
        ticker_col: str = "numerai_ticker",
        year_horizon_filter: Optional[Any] = None,
        variance_threshold: Optional[float] = None,
        correlation_method: Optional[str] = "pandas_mean",
        abs_corr: bool = True,
        top_k: Optional[int] = None,
        feature_selection_percentile: Optional[float] = None,
        sparse_selector: Optional[str] = None,
        sparse_alpha: float = 0.01,
        pca_n_components: Optional[Any] = None,
        quantize_scope: str = "era",
        quantize_strategy: str = "zscore",
        sequence_top_n: int = 4,
        feature_groups: Optional[Any] = None,
        basic_group_stats: Optional[str] = None,
        fourier_period_eras: int = 52,
        fourier_order: Optional[int] = None,
        rolling_era_windows: Any = (4, 12),
        include_rolling_era: bool = False,
        era_lag_scales: Any = (1, 2, 4),
        include_lag_diff: bool = False,
        ticker_features: Optional[str] = None,
        ticker_windows: Any = (4, 12),
        ticker_lag_steps: Optional[Any] = None,
        era_dispersion_features: Optional[str] = None,
        missing_fill_value: Any = 2.0,
    ):
        self.era_col = era_col
        self.target_col = target_col
        self.ticker_col = ticker_col
        self.year_horizon_filter = year_horizon_filter
        self.variance_threshold = variance_threshold
        self.correlation_method = correlation_method
        self.abs_corr = abs_corr
        self.top_k = top_k
        self.feature_selection_percentile = feature_selection_percentile
        self.sparse_selector = sparse_selector
        self.sparse_alpha = sparse_alpha
        self.pca_n_components = pca_n_components
        self.quantize_scope = quantize_scope
        self.quantize_strategy = quantize_strategy
        self.sequence_top_n = sequence_top_n
        self.feature_groups = feature_groups
        self.basic_group_stats = basic_group_stats
        self.fourier_period_eras = fourier_period_eras
        self.fourier_order = fourier_order
        self.rolling_era_windows = rolling_era_windows
        self.include_rolling_era = include_rolling_era
        self.era_lag_scales = era_lag_scales
        self.include_lag_diff = include_lag_diff
        self.ticker_features = ticker_features
        self.ticker_windows = ticker_windows
        self.ticker_lag_steps = ticker_lag_steps
        self.era_dispersion_features = era_dispersion_features
        self.missing_fill_value = missing_fill_value

    def set_params(self, **params: Any) -> "NumeraiExtractions":
        clean = {key: _decode_structured(val) for key, val in params.items()}
        if "lag_scales" in clean and "era_lag_scales" not in clean:
            clean["era_lag_scales"] = clean.pop("lag_scales")
        return super().set_params(**clean)

    def _candidate_feature_cols(self, df: pd.DataFrame) -> List[str]:
        t_col = str(self.target_col or "target")
        tk_col = str(self.ticker_col or "numerai_ticker")
        return [
            str(col)
            for col in df.columns
            if str(col) not in FORBIDDEN_FEATURE_COLS
            and str(col) != t_col
            and str(col) != tk_col
            and not str(col).startswith("target")
        ]

    def _quantize_signal(self, values: np.ndarray, era_vals: Optional[pd.Series], signal_key: str) -> np.ndarray:
        """Quantize a 1D signal according to `self.quantize_scope` (`'era'`, `'train_fit'`, or `'global'`) and `self.quantize_strategy` (`'zscore'` or `'quantile'`)."""
        scope = str(self.quantize_scope or "era").strip().lower()
        q_strat = str(self.quantize_strategy or "zscore").strip().lower()
        arr = np.asarray(values, dtype=np.float64).ravel()
        if scope == "train_fit":
            scalers: Dict[str, StandardScaler] = getattr(self, "signal_scalers_", {})
            scaler = scalers.get(signal_key)
            if scaler is None and getattr(self, "_is_fitting_", False):
                finite = np.isfinite(arr)
                clean = np.where(finite, arr, np.nanmedian(arr[finite]) if finite.any() else 0.0)
                scaler = StandardScaler().fit(clean.reshape(-1, 1))
                scalers[signal_key] = scaler
                self.signal_scalers_ = scalers
            if scaler is not None:
                return quantize_to_5bins(arr, strategy=q_strat, quantize_scope="train_fit", fitted_scaler=scaler)
        return quantize_to_5bins(
            arr,
            strategy=q_strat,
            eras=era_vals.to_numpy() if era_vals is not None else None,
            quantize_scope=scope,
        )

    def fit(self, X: pd.DataFrame, y: Any = None) -> "NumeraiExtractions":
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        raw_cols = self._candidate_feature_cols(X_df)
        if not raw_cols:
            raise ValueError("NumeraiExtractions found no feature columns in X.")

        era_series = (
            pd.to_numeric(X_df[self.era_col], errors="coerce").fillna(1).astype(np.int32)
            if self.era_col in X_df.columns
            else None
        )
        clean_df, self.learned_fill_map_ = _fill_missing_features(
            X_df[raw_cols],
            missing_fill_value=self.missing_fill_value,
            era_series=era_series,
        )

        # Optional `year_horizon_filter` during fit (training-era filter only — never added as a feature)
        use_filter = False
        fit_mask = pd.Series(True, index=clean_df.index)
        decoded_horizons = _decode_structured(self.year_horizon_filter)
        if era_series is not None and decoded_horizons not in (None, "none", "None", ""):
            if isinstance(decoded_horizons, (int, float)):
                horizons = [int(decoded_horizons)]
            else:
                horizons = [int(h) for h in decoded_horizons]
            candidate_mask = ((era_series - 1) // 52 + 1).isin(horizons)
            if bool(candidate_mask.any()) and not bool(candidate_mask.all()):
                fit_mask = candidate_mask
                use_filter = True

        fit_clean_df = clean_df.loc[fit_mask] if use_filter else clean_df
        fit_era_series = (era_series.loc[fit_mask] if use_filter else era_series) if era_series is not None else None

        var_thresh = max(0.0, float(self.variance_threshold or 0.0))
        if var_thresh > 0.0:
            vt_selector = VarianceThreshold(threshold=var_thresh)
            try:
                vt_selector.fit(fit_clean_df.to_numpy(dtype=np.float32, copy=False))
                support_mask = vt_selector.get_support()
                kept_cols = [col for col, keep in zip(raw_cols, support_mask) if keep] or raw_cols
            except ValueError:
                kept_cols = raw_cols
        else:
            kept_cols = raw_cols

        fit_clean_df = fit_clean_df if kept_cols == raw_cols else fit_clean_df[kept_cols]

        target_vec: Optional[pd.Series] = None
        if self.target_col and self.target_col in X_df.columns:
            target_vec = pd.to_numeric(X_df[self.target_col], errors="coerce").fillna(0.5).astype(np.float32)
        elif y is not None:
            target_vec = pd.Series(np.asarray(y, dtype=np.float32).ravel(), index=clean_df.index)

        if target_vec is not None and self.correlation_method not in (None, "none", "None", ""):
            target_series = target_vec.loc[fit_mask] if use_filter else target_vec
            corr_series = compute_feature_target_correlations(
                fit_clean_df,
                target_series,
                correlation_method=self.correlation_method,
                era_series=fit_era_series,
            )
            ranking = corr_series.abs() if bool(self.abs_corr) else corr_series
            ordered_cols = list(ranking.sort_values(ascending=False, kind="stable").index)
        else:
            corr_series = pd.Series(np.zeros(len(kept_cols), dtype=np.float64), index=kept_cols)
            ordered_cols = list(kept_cols)

        if self.feature_selection_percentile is not None and str(self.feature_selection_percentile) not in ("None", "none", ""):
            pct = float(self.feature_selection_percentile)
            n_keep = max(1, min(len(ordered_cols), int(round(len(ordered_cols) * pct / 100.0))))
            selected_cols = ordered_cols[:n_keep]
        elif self.top_k is not None and str(self.top_k) not in ("None", "none", ""):
            k = max(1, min(len(ordered_cols), int(self.top_k)))
            selected_cols = ordered_cols[:k]
        else:
            selected_cols = ordered_cols

        # Optional sparse multivariate selector (`SelectFromModel` with `Lasso` or `ElasticNet`)
        self.sparse_selector_: Optional[SelectFromModel] = None
        if (
            self.sparse_selector not in (None, False, "None", "none", "null", "")
            and target_vec is not None
            and len(selected_cols) >= 2
            and len(fit_clean_df) >= 3
        ):
            sel_mode = str(self.sparse_selector).strip().lower()
            alpha_val = max(1e-6, float(self.sparse_alpha if self.sparse_alpha is not None else 0.01))
            if sel_mode == "lasso":
                base_sparse = Lasso(alpha=alpha_val, random_state=42, max_iter=500)
            elif sel_mode == "elasticnet":
                base_sparse = ElasticNet(alpha=alpha_val, l1_ratio=0.5, random_state=42, max_iter=500)
            else:
                raise ValueError(f"Unsupported sparse_selector '{self.sparse_selector}'; use None, 'lasso', or 'elasticnet'.")
            sfm = SelectFromModel(base_sparse)
            fit_target_arr = (target_vec.loc[fit_mask] if use_filter else target_vec).to_numpy(dtype=np.float32, copy=False)
            sfm.fit(
                fit_clean_df[selected_cols].to_numpy(dtype=np.float32, copy=False),
                fit_target_arr,
            )
            sparse_support = sfm.get_support()
            sparse_kept = [col for col, keep in zip(selected_cols, sparse_support) if keep]
            if sparse_kept:
                selected_cols = sparse_kept
            self.sparse_selector_ = sfm

        self.pca_: Optional[PCA] = None
        if (
            self.pca_n_components not in (None, False, "None", "none", "null", "", 0)
            and len(selected_cols) >= 2
            and len(fit_clean_df) >= 2
        ):
            raw_pca = self.pca_n_components
            if isinstance(raw_pca, float) and 0.0 < raw_pca < 1.0:
                n_comp: Any = float(raw_pca)
            else:
                n_comp = min(max(1, int(raw_pca)), len(selected_cols), len(fit_clean_df))
            self.pca_ = PCA(n_components=n_comp, random_state=42)
            self.pca_.fit(fit_clean_df[selected_cols].to_numpy(dtype=np.float32, copy=False))

        self.feature_names_in_ = np.asarray(raw_cols, dtype=object)
        self.n_features_in_ = len(raw_cols)
        self.feature_corrs_series_ = corr_series
        self.selected_feature_names_ = selected_cols
        self.selected_corrs_ = corr_series.reindex(selected_cols).fillna(0.0).to_numpy(dtype=np.float64)
        self.signal_scalers_ = {}
        if str(self.quantize_scope or "era").strip().lower() == "train_fit":
            self._is_fitting_ = True
            try:
                _ = self.transform(X_df.loc[fit_mask])
            finally:
                self._is_fitting_ = False
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        from data_utils import compute_per_ticker_timeseries_features

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=list(self.feature_names_in_))
        era_vals = (
            pd.to_numeric(X_df[self.era_col], errors="coerce").fillna(1).astype(np.int32)
            if self.era_col in X_df.columns
            else None
        )
        base_features, _ = _fill_missing_features(
            X_df[self.selected_feature_names_],
            missing_fill_value=self.missing_fill_value,
            era_series=era_vals,
            learned_fill_map=getattr(self, "learned_fill_map_", None),
        )
        extra_cols: Dict[str, Any] = {}
        scope = str(self.quantize_scope or "era").strip().lower()

        group_stat_tokens = _parse_pipe_tokens(self.basic_group_stats, ("mean", "std", "skew"))
        if group_stat_tokens:
            selected_set = set(self.selected_feature_names_)
            decoded_groups = _decode_structured(self.feature_groups)
            if isinstance(decoded_groups, (list, tuple)):
                try:
                    decoded_groups = {str(k): list(v) for k, v in decoded_groups}
                except (TypeError, ValueError):
                    decoded_groups = None
            groups_map: Dict[str, List[str]] = {}
            if isinstance(decoded_groups, Mapping):
                groups_map = {
                    str(grp): [str(c) for c in cols if str(c) in selected_set]
                    for grp, cols in decoded_groups.items()
                    if isinstance(cols, (list, tuple, set))
                }
                groups_map = {grp: cols for grp, cols in groups_map.items() if len(cols) >= 2}
            if not groups_map:
                prefix_groups: Dict[str, List[str]] = {}
                for col in self.selected_feature_names_:
                    parts = col.split("_")
                    grp_key = parts[1] if len(parts) > 2 and parts[0] == "feature" else parts[0]
                    grp_key = "".join(ch for ch in grp_key if not ch.isdigit()) or "group"
                    prefix_groups.setdefault(grp_key, []).append(col)
                groups_map = {grp: cols for grp, cols in prefix_groups.items() if len(cols) >= 2}
            for grp, grp_cols in groups_map.items():
                sub_mat = base_features[grp_cols]
                if "mean" in group_stat_tokens:
                    extra_cols[f"feature_{grp}_mean"] = np.clip(np.rint(sub_mat.mean(axis=1)), 0, 4).astype(np.int8)
                if "std" in group_stat_tokens:
                    extra_cols[f"feature_{grp}_std"] = self._quantize_signal(
                        sub_mat.std(axis=1).to_numpy(), era_vals, f"feature_{grp}_std"
                    )
                if "skew" in group_stat_tokens:
                    extra_cols[f"feature_{grp}_skew"] = self._quantize_signal(
                        sub_mat.skew(axis=1).fillna(0.0).to_numpy(), era_vals, f"feature_{grp}_skew"
                    )

        dispersion_tokens = _parse_pipe_tokens(self.era_dispersion_features, ("mean", "std", "extreme_frac"))
        if dispersion_tokens:
            mat = base_features.to_numpy(dtype=np.float32, copy=False)
            if "mean" in dispersion_tokens:
                extra_cols["row_feat_mean_bin"] = np.clip(np.rint(mat.mean(axis=1)), 0, 4).astype(np.int8)
            if "std" in dispersion_tokens:
                extra_cols["row_feat_std_bin"] = self._quantize_signal(
                    mat.std(axis=1), era_vals, "row_feat_std_bin"
                )
            if "extreme_frac" in dispersion_tokens:
                extra_cols["row_extreme_frac_bin"] = np.clip(
                    np.rint(((mat == 0) | (mat == 4)).mean(axis=1) * 4.0), 0, 4
                ).astype(np.int8)
            del mat

        seq_top_n = max(1, int(self.sequence_top_n or 4))

        # Causal per-ticker time-series features (`rolling_mean`, `momentum_diff`, `weeks_since_high`, `rolling_rank`, `weeks_since_change`, `expanding_stats`, `raw_lags`)
        if (
            self.ticker_features not in (None, False, "None", "none", "null", "")
            and era_vals is not None
            and self.ticker_col in X_df.columns
        ):
            raw_tw = _decode_structured(self.ticker_windows)
            t_windows = [int(w) for w in raw_tw] if isinstance(raw_tw, (list, tuple)) else [4, 12]
            raw_tlags = _decode_structured(self.ticker_lag_steps)
            t_lags = [int(k) for k in raw_tlags] if isinstance(raw_tlags, (list, tuple)) else None
            top_ticker_cols = [c for c in self.selected_feature_names_ if "_ticker_" not in c][:seq_top_n]
            if top_ticker_cols:
                already_precomputed = all(
                    any(f"{col}_ticker_" in existing for existing in self.selected_feature_names_)
                    for col in top_ticker_cols[:1]
                )
                if not already_precomputed:
                    temp_ticker_in = base_features[top_ticker_cols].copy()
                    temp_ticker_in[self.era_col] = era_vals
                    temp_ticker_in[self.ticker_col] = X_df[self.ticker_col].astype(str)
                    temp_ticker_out, added_ticker_cols = compute_per_ticker_timeseries_features(
                        temp_ticker_in,
                        top_ticker_cols,
                        ticker_col=self.ticker_col,
                        era_col=self.era_col,
                        ticker_features=self.ticker_features,
                        ticker_windows=t_windows,
                        ticker_lag_steps=t_lags,
                        top_n_cols=len(top_ticker_cols),
                        quantize_scope="era" if scope == "era" else "global",
                    )
                    for col_name in added_ticker_cols:
                        extra_cols[col_name] = temp_ticker_out[col_name].to_numpy(dtype=np.int8)

        if era_vals is not None:
            extra_cols[self.era_col] = era_vals

            if self.fourier_order not in (None, "none", "None", "", 0, False) and int(self.fourier_order) > 0:
                period = max(2.0, float(self.fourier_period_eras or 52.0))
                era_float = era_vals.to_numpy(dtype=np.float64)
                for harmonic in range(1, int(self.fourier_order) + 1):
                    angle = 2.0 * np.pi * float(harmonic) * era_float / period
                    extra_cols[f"fourier_sin_k{harmonic}"] = np.round(np.sin(angle) * 2.0 + 2.0).clip(0, 4).astype(np.int8)
                    extra_cols[f"fourier_cos_k{harmonic}"] = np.round(np.cos(angle) * 2.0 + 2.0).clip(0, 4).astype(np.int8)

            raw_windows = _decode_structured(self.rolling_era_windows)
            rolling_active = bool(self.include_rolling_era) and (raw_windows not in (None, "none", "None", "", False))
            raw_lags = _decode_structured(self.era_lag_scales)
            lag_active = bool(self.include_lag_diff) and (raw_lags not in (None, "none", "None", "", False))

            top_cols = self.selected_feature_names_[: min(seq_top_n, len(self.selected_feature_names_))]
            need_era_means = rolling_active or lag_active
            era_means = base_features[top_cols].groupby(era_vals, sort=True).mean() if (need_era_means and top_cols) else None

            if rolling_active and era_means is not None:
                windows = [int(w) for w in raw_windows] if isinstance(raw_windows, (list, tuple)) else [4, 12]
                for window in windows:
                    if window <= 1:
                        continue
                    rolling_means = era_means.rolling(window=window, min_periods=1).mean()
                    for col in top_cols:
                        mapped = era_vals.map(rolling_means[col]).to_numpy(dtype=np.float64)
                        # Era-level scalar series is constant within a single era, so use train_fit or global across eras
                        era_scope = "train_fit" if scope == "train_fit" else "global"
                        extra_cols[f"{col}_era_roll{window}"] = (
                            self._quantize_signal(mapped, None, f"{col}_era_roll{window}")
                            if era_scope == "train_fit"
                            else quantize_to_5bins(mapped, quantize_scope="global")
                        )

            if lag_active and era_means is not None:
                lags = [int(l_val) for l_val in raw_lags] if isinstance(raw_lags, (list, tuple)) else [1, 2, 4]
                for lag_step in lags:
                    if lag_step <= 0:
                        continue
                    diff_means = era_means.diff(periods=lag_step).fillna(0.0)
                    for col in top_cols:
                        mapped = era_vals.map(diff_means[col]).to_numpy(dtype=np.float64)
                        era_scope = "train_fit" if scope == "train_fit" else "global"
                        extra_cols[f"{col}_era_diff{lag_step}"] = (
                            self._quantize_signal(mapped, None, f"{col}_era_diff{lag_step}")
                            if era_scope == "train_fit"
                            else quantize_to_5bins(mapped, quantize_scope="global")
                        )

        if getattr(self, "pca_", None) is not None:
            pca_scores = self.pca_.transform(base_features.to_numpy(dtype=np.float32, copy=False))
            pca_cols = {
                f"pca_comp_{comp_idx}": self._quantize_signal(
                    pca_scores[:, comp_idx], era_vals, f"pca_comp_{comp_idx}"
                )
                for comp_idx in range(pca_scores.shape[1])
            }
            base_features = pd.DataFrame(pca_cols, index=base_features.index)
            del pca_scores

        if extra_cols:
            return pd.concat([base_features, pd.DataFrame(extra_cols, index=base_features.index)], axis=1)
        return base_features


NumeraiFeatureExtractor = NumeraiExtractions
NumeraiSequenceExtractor = NumeraiExtractions


# ---------------------------------------------------------------------------
# Numerai Evaluation Helpers (`numerai_corr`, Spearman, Payout, Sharpe, MAE, Feature Exposure, Sanity Tripwire, Diagnostics Export)
# ---------------------------------------------------------------------------

def sharpe_ratio(corrs: pd.Series) -> float:
    """Calculate the Numerai Sharpe ratio across per-era correlations (`corrs.mean() / corrs.std()`)."""
    if len(corrs) == 0:
        return 0.0
    std_val = float(corrs.std())
    if not np.isfinite(std_val) or std_val <= 1e-12:
        std_val = float(corrs.std(ddof=0))
    return float(corrs.mean() / std_val) if std_val > 1e-12 else 0.0


def compute_per_era_spearman_corr(y_true: Any, y_pred: Any, eras: Optional[Any] = None) -> pd.Series:
    """Compute per-era Spearman rank correlation (`spearmanr(target, prediction)[0]`) matching the Numerai starter notebook."""
    targets = np.asarray(y_true, dtype=np.float64).ravel()
    preds = np.asarray(y_pred, dtype=np.float64).ravel()
    era_arr = np.ones(len(targets), dtype=np.int32) if eras is None else np.asarray(eras).ravel()

    frame = pd.DataFrame({"era": era_arr, "target": targets, "prediction": preds})
    era_scores: Dict[Any, float] = {}
    for era_key, era_df in frame.groupby("era", sort=True):
        valid_df = era_df.dropna(subset=["target", "prediction"])
        if len(valid_df) < 3:
            era_scores[era_key] = 0.0
            continue
        t_arr = valid_df["target"].to_numpy(dtype=np.float64, copy=False)
        p_arr = valid_df["prediction"].to_numpy(dtype=np.float64, copy=False)
        if _is_constant_1d(p_arr) or _is_constant_1d(t_arr):
            era_scores[era_key] = 0.0
        else:
            corr_val = float(spearmanr(t_arr, p_arr)[0])
            era_scores[era_key] = corr_val if np.isfinite(corr_val) else 0.0
    return pd.Series(era_scores, dtype=np.float64)


def compute_per_era_numerai_corr(y_true: Any, y_pred: Any, eras: Optional[Any] = None) -> pd.Series:
    """Compute official `numerai_tools.scoring.numerai_corr` per era and return the era Series."""
    targets = np.asarray(y_true, dtype=np.float64).ravel()
    preds = np.asarray(y_pred, dtype=np.float64).ravel()
    era_arr = np.ones(len(targets), dtype=np.int32) if eras is None else np.asarray(eras).ravel()

    frame = pd.DataFrame({"era": era_arr, "target": targets, "prediction": preds})
    era_scores: Dict[Any, float] = {}
    for era_key, era_df in frame.groupby("era", sort=True):
        valid_df = era_df.dropna(subset=["target", "prediction"])
        if len(valid_df) < 3:
            era_scores[era_key] = 0.0
            continue
        t_arr = valid_df["target"].to_numpy(dtype=np.float64, copy=False)
        p_arr = valid_df["prediction"].to_numpy(dtype=np.float64, copy=False)
        if _is_constant_1d(p_arr) or _is_constant_1d(t_arr):
            era_scores[era_key] = 0.0
        else:
            score_val = float(numerai_corr(valid_df[["prediction"]], valid_df["target"])["prediction"])
            era_scores[era_key] = score_val if np.isfinite(score_val) else 0.0
    return pd.Series(era_scores, dtype=np.float64)


def compute_feature_exposure(
    features_df: pd.DataFrame,
    predictions: Any,
    feature_columns: Optional[Sequence[str]] = None,
) -> float:
    """Compute Feature Exposure (`np.std([spearmanr(prediction, f)[0] for f in all_features])`) matching the Numerai starter notebook."""
    if feature_columns is not None and len(feature_columns) == 0:
        return float(np.nan)
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    if len(preds) < 3 or _is_constant_1d(preds):
        return 0.0
    cols = (
        list(feature_columns)
        if feature_columns is not None
        else [str(col) for col in features_df.columns if str(col) not in FORBIDDEN_FEATURE_COLS]
    )
    valid_cols = [col for col in cols if col in features_df.columns]
    if not valid_cols:
        return float(np.nan)
    feature_corrs = _pooled_pearson_corr(features_df[valid_cols], preds, rank_columns=True)
    return float(np.round(np.std(feature_corrs), 4)) if len(feature_corrs) else 0.0


def evaluate_numerai_predictions(
    y_true: Any,
    y_pred: Any,
    eras: Optional[Any] = None,
    *,
    features_df: Optional[pd.DataFrame] = None,
    feature_columns: Optional[Sequence[str]] = None,
    label: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate predictions using official `numerai_corr`, tutorial metrics (Spearman, Payout, Sharpe, MAE, Feature Exposure), and the `[0.005, 0.04]` CORR sanity tripwire."""
    targets = np.asarray(y_true, dtype=np.float64).ravel()
    preds = np.asarray(y_pred, dtype=np.float64).ravel()

    spearman_era_corrs = compute_per_era_spearman_corr(targets, preds, eras=eras)
    numerai_era_corrs = compute_per_era_numerai_corr(targets, preds, eras=eras)

    payout_raw = (spearman_era_corrs / 0.2).clip(-1.0, 1.0)
    spearman_mean = float(np.round(spearman_era_corrs.mean(), 4)) if len(spearman_era_corrs) else 0.0
    numerai_corr_mean = float(np.round(numerai_era_corrs.mean(), 4)) if len(numerai_era_corrs) else 0.0
    payout_mean = float(np.round(payout_raw.mean(), 4)) if len(payout_raw) else 0.0
    spearman_sharpe = float(np.round(sharpe_ratio(spearman_era_corrs), 4))
    numerai_sharpe = float(np.round(sharpe_ratio(numerai_era_corrs), 4))
    mae_val = float(np.round(mean_absolute_error(targets, preds), 4)) if len(targets) else 0.0
    forecast_bias = float(np.round(np.mean(targets - preds), 4)) if len(targets) else 0.0
    raw_lag1_acf = _safe_autocorr(numerai_era_corrs.to_numpy(dtype=np.float64), lag=1) if len(numerai_era_corrs) >= 3 else 0.0
    era_corr_lag1_acf = float(np.round(raw_lag1_acf, 4)) if np.isfinite(raw_lag1_acf) else 0.0

    # Expanded Financial Time-Series Diagnostics (Ljung-Box, ARCH-LM, ADF Stationarity, Skew, Kurtosis, Jarque-Bera, Directional Hit Rate)
    corrs_arr = numerai_era_corrs.to_numpy(dtype=np.float64)
    corrs_std = float(np.std(corrs_arr)) if len(corrs_arr) else 0.0
    lb_lag = min(10, max(1, int(np.log(max(2, len(corrs_arr))))))
    if len(corrs_arr) > lb_lag + 1 and corrs_std > 1e-12:
        try:
            lb_df = acorr_ljungbox(corrs_arr, lags=[lb_lag], return_df=True)
            raw_lb_p = float(lb_df["lb_pvalue"].to_numpy()[0])
            ljung_box_pvalue = float(np.round(raw_lb_p, 4)) if np.isfinite(raw_lb_p) else 1.0
        except Exception:
            ljung_box_pvalue = 1.0
    else:
        ljung_box_pvalue = 1.0

    if len(corrs_arr) >= 6 and corrs_std > 1e-12:
        try:
            nlags_arch = min(4, max(1, (len(corrs_arr) - 2) // 3))
            raw_arch_p = _run_arch_lm_pvalue(corrs_arr, nlags=nlags_arch)
            arch_lm_pvalue = float(np.round(raw_arch_p, 4)) if np.isfinite(raw_arch_p) else 1.0
        except Exception:
            arch_lm_pvalue = 1.0
    else:
        arch_lm_pvalue = 1.0

    if len(corrs_arr) >= 6 and corrs_std > 1e-12:
        try:
            _, raw_adf_p = _run_adfuller(corrs_arr, maxlag=min(4, max(0, len(corrs_arr) // 4 - 1)))
            adf_pvalue = float(np.round(raw_adf_p, 4)) if np.isfinite(raw_adf_p) else 1.0
        except Exception:
            adf_pvalue = 1.0
    else:
        adf_pvalue = 1.0

    if len(corrs_arr) >= 3 and corrs_std > 1e-12:
        raw_skew = float(stats.skew(corrs_arr, bias=False))
        era_corr_skew = float(np.round(raw_skew, 4)) if np.isfinite(raw_skew) else 0.0
    else:
        era_corr_skew = 0.0

    if len(corrs_arr) >= 4 and corrs_std > 1e-12:
        raw_kurt = float(stats.kurtosis(corrs_arr, fisher=True, bias=False))
        era_corr_kurtosis = float(np.round(raw_kurt, 4)) if np.isfinite(raw_kurt) else 0.0
        raw_jb_p = float(stats.jarque_bera(corrs_arr).pvalue)
        jarque_bera_pvalue = float(np.round(raw_jb_p, 4)) if np.isfinite(raw_jb_p) else 1.0
    else:
        era_corr_kurtosis = 0.0
        jarque_bera_pvalue = 1.0

    if len(targets):
        pred_center = 0.5 if (0.15 <= float(np.mean(preds)) <= 0.85) else float(np.mean(preds))
        non_neutral_mask = np.abs(targets - 0.5) > 1e-9
        if np.any(non_neutral_mask):
            directional_hit_rate = float(
                np.round(np.mean(np.sign(targets[non_neutral_mask] - 0.5) == np.sign(preds[non_neutral_mask] - pred_center)), 4)
            )
        else:
            directional_hit_rate = 0.5
    else:
        directional_hit_rate = 0.0

    numerai_corr_pct = float(np.round(numerai_corr_mean * 100.0, 2))
    positive_era_pct = float(np.round(float(np.mean(corrs_arr > 0.0)) * 100.0, 2)) if len(corrs_arr) else 0.0
    directional_accuracy_pct = float(np.round(directional_hit_rate * 100.0, 2))

    cumulative_corr = numerai_era_corrs.cumsum()
    if len(cumulative_corr):
        running_peak = pd.Series(
            np.maximum.accumulate(np.insert(cumulative_corr.to_numpy(), 0, 0.0))[1:],
            index=cumulative_corr.index,
        )
        max_drawdown = float(np.round((running_peak - cumulative_corr).max(), 4))
    else:
        max_drawdown = 0.0

    feature_exposure = (
        compute_feature_exposure(features_df, preds, feature_columns=feature_columns)
        if (features_df is not None and (feature_columns is None or len(feature_columns) > 0))
        else np.nan
    )

    if numerai_corr_mean > CORR_SANITY_HIGH_THRESHOLD:
        sanity_status = "LEAKAGE_WARNING"
        sanity_msg = (
            f"[SANITY WARNING — LIKELY LEAKAGE] numerai_corr={numerai_corr_mean:.4f} > {CORR_SANITY_HIGH_THRESHOLD:.3f} ceiling "
            "(check for future era/target leakage or in-sample overfitting)."
        )
    elif numerai_corr_mean < CORR_SANITY_LOW_THRESHOLD:
        sanity_status = "LOW_SIGNAL_WARNING"
        sanity_msg = (
            f"[SANITY WARNING — WEAK/BROKEN SIGNAL] numerai_corr={numerai_corr_mean:.4f} < {CORR_SANITY_LOW_THRESHOLD:.3f} floor "
            "(check alignment, target, or over-regularization)."
        )
    else:
        sanity_status = "OK"
        sanity_msg = (
            f"[SANITY CHECK PASSED] numerai_corr={numerai_corr_mean:.4f} is within healthy "
            f"[{CORR_SANITY_LOW_THRESHOLD:.3f}, {CORR_SANITY_HIGH_THRESHOLD:.3f}] Numerai range."
        )

    if verbose:
        if label:
            print(f"--- {label} ---")
        print(
            f"Percentage Summary | Numerai CORR: {numerai_corr_pct:+.2f}% | "
            f"Positive Eras: {positive_era_pct:.1f}% | Directional Accuracy: {directional_accuracy_pct:.1f}%"
        )
        print(f"Numerai CORR: {numerai_corr_mean:.4f}")
        print(f"Spearman Correlation: {spearman_mean:.4f}")
        print(f"Average Payout: {payout_mean:.4f}")
        print(f"Sharpe Ratio (Numerai / Spearman): {numerai_sharpe:.4f} / {spearman_sharpe:.4f}")
        print(f"Mean Absolute Error (MAE): {mae_val:.4f}")
        print(f"Forecast Bias (mean(y - y_hat)): {forecast_bias:.4f}")
        print(f"Directional Hit Rate (vs 0.5): {directional_hit_rate:.4f} ({directional_accuracy_pct:.1f}%)")
        print(
            f"Era CORR White-Noise & Regime Tests | Lag-1 ACF: {era_corr_lag1_acf:.4f} | "
            f"Ljung-Box p: {ljung_box_pvalue:.4f} | ARCH-LM p: {arch_lm_pvalue:.4f} | ADF p: {adf_pvalue:.4f}"
        )
        print(
            f"Era CORR Tail Risk | Skew: {era_corr_skew:.4f} | Excess Kurtosis: {era_corr_kurtosis:.4f} | "
            f"Jarque-Bera p: {jarque_bera_pvalue:.4f}"
        )
        if np.isfinite(feature_exposure):
            print(f"Feature Exposure: {feature_exposure:.4f}")
        print(sanity_msg)

    return {
        "numerai_corr": numerai_corr_mean,
        "numerai_corr_pct": numerai_corr_pct,
        "positive_era_pct": positive_era_pct,
        "directional_accuracy_pct": directional_accuracy_pct,
        "spearman": spearman_mean,
        "payout": payout_mean,
        "numerai_sharpe": numerai_sharpe,
        "spearman_sharpe": spearman_sharpe,
        "mae": mae_val,
        "forecast_bias": forecast_bias,
        "directional_hit_rate": directional_hit_rate,
        "era_corr_lag1_acf": era_corr_lag1_acf,
        "ljung_box_pvalue": ljung_box_pvalue,
        "arch_lm_pvalue": arch_lm_pvalue,
        "adf_pvalue": adf_pvalue,
        "era_corr_skew": era_corr_skew,
        "era_corr_kurtosis": era_corr_kurtosis,
        "jarque_bera_pvalue": jarque_bera_pvalue,
        "max_drawdown": max_drawdown,
        "feature_exposure": feature_exposure,
        "sanity_status": sanity_status,
        "numerai_era_corrs": numerai_era_corrs,
        "spearman_era_corrs": spearman_era_corrs,
    }


def compute_eda_era_stationarity(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    era_col: str = "era",
) -> pd.DataFrame:
    """Run Augmented Dickey-Fuller (`adfuller`) stationarity check on per-era means of top correlated features."""
    if era_col not in df.columns or not feature_cols:
        return pd.DataFrame(columns=["feature", "adf_stat", "adf_pvalue", "stationary_at_5pct"])
    valid_cols = [str(c) for c in feature_cols if c in df.columns]
    era_means = df[valid_cols].apply(pd.to_numeric, errors="coerce").groupby(df[era_col], sort=True).mean()
    rows: List[Dict[str, Any]] = []
    for col in valid_cols:
        series_vals = era_means[col].dropna().to_numpy(dtype=np.float64)
        if len(series_vals) >= 6 and float(np.std(series_vals)) > 1e-12:
            try:
                raw_stat, raw_pval = _run_adfuller(
                    series_vals, maxlag=min(4, max(0, len(series_vals) // 4 - 1))
                )
                stat_val = float(np.round(raw_stat, 4))
                pval = float(np.round(raw_pval, 4)) if np.isfinite(raw_pval) else 1.0
            except Exception:
                stat_val, pval = 0.0, 1.0
        else:
            stat_val, pval = 0.0, 1.0
        rows.append({
            "feature": col,
            "adf_stat": stat_val,
            "adf_pvalue": pval,
            "stationary_at_5pct": bool(pval < 0.05),
        })
    return pd.DataFrame(rows)


def walk_forward_predict(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_eras: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_eras: pd.Series,
    *,
    step_eras: int = 20,
    window_mode: str = "expanding",
    target_horizon_eras: int = 4,
    era_stride: int = 1,
) -> np.ndarray:
    """Chronological walk-forward holdout evaluation respecting `target_horizon_eras` settlement lag and `era_stride` subsampling of settled validation eras."""
    train_era_arr = np.asarray(train_eras, dtype=np.int32).ravel()
    val_era_arr = np.asarray(validation_eras, dtype=np.int32).ravel()
    unique_val_eras = np.array(sorted(np.unique(val_era_arr)), dtype=np.int32)
    step = max(1, int(step_eras))
    horizon = max(1, int(target_horizon_eras))
    stride = max(1, int(era_stride))
    mode = str(window_mode or "expanding").strip().lower()
    train_era_span = int(np.max(train_era_arr) - np.min(train_era_arr)) if len(train_era_arr) else 52

    y_train_s = pd.Series(np.asarray(y_train, dtype=np.float32).ravel(), index=X_train.index)
    y_val_s = pd.Series(np.asarray(y_validation, dtype=np.float32).ravel(), index=X_validation.index)

    wf_preds = np.full(len(X_validation), float(np.mean(y_train_s)) if len(y_train_s) else 0.5, dtype=np.float64)
    for start_idx in range(0, len(unique_val_eras), step):
        chunk_eras = unique_val_eras[start_idx : start_idx + step]
        forecast_origin_era = int(chunk_eras[0])
        settled_cutoff_era = forecast_origin_era - horizon
        min_sliding_era = (settled_cutoff_era - max(1, train_era_span)) if mode == "sliding" else None

        train_mask = train_era_arr <= settled_cutoff_era
        if min_sliding_era is not None:
            train_mask = train_mask & (train_era_arr >= min_sliding_era)

        settled_val_eras = unique_val_eras[unique_val_eras <= settled_cutoff_era]
        if min_sliding_era is not None:
            settled_val_eras = settled_val_eras[settled_val_eras >= min_sliding_era]
        if stride > 1 and len(settled_val_eras) > 0:
            settled_val_eras = settled_val_eras[::stride]

        has_train = bool(np.any(train_mask))
        has_val_settled = len(settled_val_eras) > 0

        if not has_train and not has_val_settled:
            # No settled historical eras exist prior to `settled_cutoff_era` — never leak unsettled `X_train`
            continue
        elif has_train and not has_val_settled:
            if bool(np.all(train_mask)):
                X_fit = X_train
                y_fit = y_train_s
            else:
                X_fit = X_train.loc[train_mask]
                y_fit = y_train_s.loc[train_mask]
        elif not has_train and has_val_settled:
            val_fit_mask = np.isin(val_era_arr, settled_val_eras)
            X_fit = X_validation.loc[val_fit_mask]
            y_fit = y_val_s.loc[val_fit_mask]
        else:
            val_fit_mask = np.isin(val_era_arr, settled_val_eras)
            X_tr_part = X_train if bool(np.all(train_mask)) else X_train.loc[train_mask]
            y_tr_part = y_train_s if bool(np.all(train_mask)) else y_train_s.loc[train_mask]
            X_fit = pd.concat([X_tr_part, X_validation.loc[val_fit_mask]], axis=0, ignore_index=True)
            y_fit = pd.concat([y_tr_part.reset_index(drop=True), y_val_s.loc[val_fit_mask].reset_index(drop=True)], axis=0, ignore_index=True)
            del X_tr_part, y_tr_part

        if isinstance(X_fit, pd.DataFrame) and "era" in X_fit.columns and len(X_fit) > 0:
            max_fit_era = int(pd.to_numeric(X_fit["era"], errors="coerce").max())
            if max_fit_era > settled_cutoff_era:
                raise ValueError(
                    f"Walk-forward leakage detected: max fit era ({max_fit_era}) > settled_cutoff_era ({settled_cutoff_era})."
                )

        step_model = clone(estimator)
        chunk_params = step_model.get_params(deep=True)
        if "estimator__walk_forward_step_eras" in chunk_params:
            step_model.set_params(estimator__walk_forward_step_eras=None)
        elif "walk_forward_step_eras" in chunk_params:
            step_model.set_params(walk_forward_step_eras=None)
        step_model.fit(X_fit, y_fit)

        val_chunk_mask = np.isin(val_era_arr, chunk_eras)
        wf_preds[val_chunk_mask] = step_model.predict(X_validation.loc[val_chunk_mask])

        del X_fit, y_fit, step_model
    return wf_preds


def export_diagnostics_predictions(
    validation_df: pd.DataFrame,
    preds: Any,
    output_path: Any,
    era_col: str = "era",
) -> Path:
    """Rank-normalize predictions per era to `(0, 1)` and write `validation_predictions.parquet` for Numerai Diagnostics (MMC & FNC)."""
    pred_arr = np.asarray(preds, dtype=np.float64).ravel()
    if len(pred_arr) != len(validation_df):
        raise ValueError(f"preds length ({len(pred_arr)}) must match validation_df rows ({len(validation_df)}).")

    pred_series = pd.Series(pred_arr, index=validation_df.index)
    if era_col in validation_df.columns:
        ranked_preds = pred_series.groupby(validation_df[era_col], sort=False).rank(pct=True).astype(np.float32)
    else:
        ranked_preds = pred_series.rank(pct=True).astype(np.float32)

    out_cols: Dict[str, Any] = {}
    if "id" in validation_df.columns:
        out_cols["id"] = validation_df["id"].to_numpy()
    else:
        if "numerai_ticker" in validation_df.columns:
            out_cols["numerai_ticker"] = validation_df["numerai_ticker"].to_numpy()
        if "date" in validation_df.columns:
            out_cols["date"] = validation_df["date"].to_numpy()
        if era_col in validation_df.columns and "date" not in out_cols:
            out_cols[era_col] = validation_df[era_col].to_numpy()

    out_cols["prediction"] = ranked_preds.to_numpy(dtype=np.float32)
    diag_df = pd.DataFrame(out_cols, index=validation_df.index)
    if "id" in diag_df.columns:
        diag_df = diag_df.reset_index(drop=True)

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(target_path, index=False)
    return target_path


_CV_FOLD_METRIC_LOG: List[Dict[str, Any]] = []

ERROR_BIN_LABELS: Tuple[str, ...] = (
    "Extreme Underforecast",
    "Underforecast",
    "Accurate",
    "Overforecast",
    "Extreme Overforecast",
)

_ERROR_BIN_CONFIG: Dict[str, Any] = {
    "mode": "pct",
    "pct_bins": (5.0, 10.0),
    "zero_tols": (0.01, 0.10),
    "abs_bins": (0.125, 0.375),
}


def infer_target_horizon_eras(target_col: str) -> int:
    """Infer target settlement horizon in weekly eras from a Numerai Unity v6.0 target name (`target` / `*_60` -> 12, `*_4` -> 1, `*_20` / other -> 4)."""
    name = str(target_col or "target").strip().lower()
    if name == "target" or name.endswith("_60"):
        return 12
    if name.endswith("_4"):
        return 1
    return 4


VALID_DOWNSAMPLE_MODES = (None, "within_era", "recent", "earliest", "uniform", "random_eras")


def build_experiment_setup(
    *,
    target_col: str = "target",
    feature_set: str = "small",
    era_stride: int = 4,
    model_tag: str = "baselines_ae",
    baseline_mode: Optional[str] = None,
    search_mode: str = "bayes",
    eval_mode: str = "both",
    n_splits: int = 1,
    cv_split: float = 0.8,
    downsample_mode: Optional[str] = None,
    downsample_frac: float = 1.0,
    n_iter: int = 37,
    n_jobs: int = -1,
    save_results: bool = True,
    workspace_root: Any = ".",
    data_version: str = "v6.0",
    era_tag: Optional[str] = None,
    era_col: str = "era",
    ticker_col: str = "numerai_ticker",
    ticker_timeseries_features: Optional[str] = None,
    ticker_rolling_windows: Tuple[int, ...] = (4, 12),
    ticker_lag_steps: Optional[Tuple[int, ...]] = None,
    data_folder: str = "data_folder",
    kaggle_dataset: str = "svendaj/numerai-data-v6-0-unity",
    cv_strategy: str = "timeseries",
    target_horizon_eras: Optional[int] = None,
    walk_forward_step_eras: int = 20,
    walk_forward_window: str = "expanding",
    era_weighting: Optional[str] = None,
    recency_decay: float = 0.005,
    random_state: int = 42,
    verbose: int = 3,
    error_score: float = -1.0,
    export_diagnostics: bool = False,
    error_bin_mode: str = "pct",
    error_pct_bins: Tuple[float, float] = (5.0, 10.0),
    error_zero_tols: Tuple[float, float] = (0.01, 0.10),
    error_abs_bins: Tuple[float, float] = (0.125, 0.375),
    extraction_cache_enabled: bool = True,
    extraction_cache_max_entries: int = 4,
    force_reload_data_cache: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "numerai-baselines",
    lgbm_n_jobs: int = -1,
    lgbm_verbosity: int = -1,
) -> Dict[str, Any]:
    """Validate experiment knobs, auto-infer horizon/embargo, configure binning & caches, and return full Setup namespace."""
    clean_search = str(search_mode).strip().lower()
    clean_cv = str(cv_strategy).strip().lower()
    clean_eval = str(eval_mode).strip().lower()
    clean_wf_win = str(walk_forward_window).strip().lower()
    clean_weight = str(era_weighting).strip().lower() if era_weighting is not None else None
    clean_baseline = str(baseline_mode).strip().lower() if baseline_mode is not None else None
    clean_downsample_mode = (
        None
        if downsample_mode in (None, False, "None", "none", "null", "")
        else str(downsample_mode).strip().lower()
    )
    cv_split_val = float(cv_split)
    downsample_frac_val = float(downsample_frac)

    if clean_baseline is not None and clean_baseline not in ("baseline_a", "baseline_e", "persistence", "both", "all"):
        raise ValueError("baseline_mode must be 'baseline_a', 'baseline_e', 'persistence', 'both', or 'all'")
    if clean_search not in ("grid", "bayes"):
        raise ValueError("search_mode must be 'grid' or 'bayes'")
    if clean_cv != "timeseries":
        raise ValueError("cv_strategy must be 'timeseries' (non-chronological CV strategies removed to prevent regime leakage)")
    if not (0.0 < cv_split_val < 1.0):
        raise ValueError(f"cv_split must be strictly in (0.0, 1.0), got {cv_split!r}")
    if clean_downsample_mode not in VALID_DOWNSAMPLE_MODES:
        raise ValueError(
            f"downsample_mode must be one of {VALID_DOWNSAMPLE_MODES}, got {downsample_mode!r}"
        )
    if not (0.0 < downsample_frac_val <= 1.0):
        raise ValueError(f"downsample_frac must be in (0.0, 1.0], got {downsample_frac!r}")
    if clean_eval not in ("static", "walk_forward", "both"):
        raise ValueError("eval_mode must be 'static', 'walk_forward', or 'both'")
    if clean_wf_win not in ("expanding", "sliding"):
        raise ValueError("walk_forward_window must be 'expanding' or 'sliding'")
    if clean_weight not in (None, "equal_era", "recency_linear", "recency_exp"):
        raise ValueError("era_weighting must be None, 'equal_era', 'recency_linear', or 'recency_exp'")

    horizon = int(target_horizon_eras) if target_horizon_eras is not None else infer_target_horizon_eras(target_col)
    embargo = int(np.ceil(horizon / max(1, int(era_stride))))
    resolved_era_tag = era_tag if era_tag is not None else f"all_eras_stride{int(era_stride)}"
    experiment_name = f"numerai_{data_version}_{resolved_era_tag}_{model_tag}"
    results_dir = Path(workspace_root) / f"results_{experiment_name}_{target_col}_{clean_search}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    configure_error_binning(
        mode=error_bin_mode,
        pct_bins=error_pct_bins,
        zero_tols=error_zero_tols,
        abs_bins=error_abs_bins,
    )
    try:
        from baseline_utils import configure_extraction_cache

        configure_extraction_cache(enabled=extraction_cache_enabled, max_entries=extraction_cache_max_entries)
    except ImportError:
        pass

    validation_param_space = {
        "validation_mode": [clean_cv],
        "eval_mode": [clean_eval],
        "group_column": [era_col],
        "target_column": [target_col],
        "target_horizon_eras": [horizon],
        "era_stride": [int(era_stride)],
        "embargo_eras": [embargo],
        "cv_split": [cv_split_val],
        "downsample_mode": [clean_downsample_mode],
        "downsample_frac": [downsample_frac_val],
    }

    mode_str = f"baseline_mode={clean_baseline}" if clean_baseline is not None else f"model={model_tag}"
    ds_str = (
        f" | downsample={clean_downsample_mode}@{downsample_frac_val:.2f}"
        if clean_downsample_mode is not None and downsample_frac_val < 1.0
        else ""
    )
    print(
        f"Experiment: {experiment_name} | {mode_str} | search_mode={clean_search} | "
        f"cv_strategy={clean_cv} (cv_split={cv_split_val}) | ERA_COL={era_col} | TARGET_COL={target_col} | "
        f"horizon={horizon} | embargo={embargo}{ds_str}"
    )

    cfg: Dict[str, Any] = {
        "data_version": data_version,
        "era_tag": resolved_era_tag,
        "model_tag": model_tag,
        "experiment_name": experiment_name,
        "ERA_COL": era_col,
        "TICKER_COL": ticker_col,
        "TARGET_COL": target_col,
        "feature_set": feature_set,
        "era_stride": int(era_stride),
        "ticker_timeseries_features": ticker_timeseries_features,
        "ticker_rolling_windows": ticker_rolling_windows,
        "ticker_lag_steps": ticker_lag_steps,
        "data_folder": data_folder,
        "kaggle_dataset": kaggle_dataset,
        "search_mode": clean_search,
        "cv_strategy": clean_cv,
        "n_splits": int(n_splits),
        "cv_split": cv_split_val,
        "downsample_mode": clean_downsample_mode,
        "downsample_frac": downsample_frac_val,
        "target_horizon_eras": horizon,
        "embargo_eras": embargo,
        "eval_mode": clean_eval,
        "walk_forward_step_eras": int(walk_forward_step_eras),
        "walk_forward_window": clean_wf_win,
        "era_weighting": clean_weight,
        "recency_decay": float(recency_decay),
        "random_state": int(random_state),
        "n_iter": int(n_iter),
        "verbose": int(verbose),
        "error_score": float(error_score),
        "n_jobs": int(n_jobs),
        "save_results": bool(save_results),
        "export_diagnostics": bool(export_diagnostics),
        "error_bin_mode": str(error_bin_mode).strip().lower(),
        "error_pct_bins": error_pct_bins,
        "error_zero_tols": error_zero_tols,
        "error_abs_bins": error_abs_bins,
        "extraction_cache_enabled": bool(extraction_cache_enabled),
        "extraction_cache_max_entries": int(extraction_cache_max_entries),
        "force_reload_data_cache": bool(force_reload_data_cache),
        "use_wandb": bool(use_wandb),
        "wandb_project": wandb_project,
        "lgbm_n_jobs": int(lgbm_n_jobs),
        "lgbm_verbosity": int(lgbm_verbosity),
        "results_dir": results_dir,
        "timestamp": timestamp,
        "VALIDATION_PARAM_SPACE": validation_param_space,
    }
    if clean_baseline is not None:
        cfg["baseline_mode"] = clean_baseline
    return cfg


def configure_error_binning(
    mode: str = "pct",
    pct_bins: Tuple[float, float] = (5.0, 10.0),
    zero_tols: Tuple[float, float] = (0.01, 0.10),
    abs_bins: Tuple[float, float] = (0.125, 0.375),
) -> Dict[str, Any]:
    """Set module-wide error-binning parameters used by `candidate_score` and waterfall analysis."""
    clean_mode = str(mode or "pct").strip().lower()
    if clean_mode not in ("pct", "abs"):
        raise ValueError("error_bin_mode must be 'pct' or 'abs'.")
    p_acc, p_ext = sorted((abs(float(pct_bins[0])), abs(float(pct_bins[1]))))
    z_acc, z_ext = sorted((abs(float(zero_tols[0])), abs(float(zero_tols[1]))))
    a_acc, a_ext = sorted((abs(float(abs_bins[0])), abs(float(abs_bins[1]))))
    _ERROR_BIN_CONFIG["mode"] = clean_mode
    _ERROR_BIN_CONFIG["pct_bins"] = (p_acc, p_ext)
    _ERROR_BIN_CONFIG["zero_tols"] = (z_acc, z_ext)
    _ERROR_BIN_CONFIG["abs_bins"] = (a_acc, a_ext)
    return dict(_ERROR_BIN_CONFIG)


def classify_error_bins(
    y_true: Any,
    y_pred: Any,
    *,
    mode: Optional[str] = None,
    pct_bins: Optional[Tuple[float, float]] = None,
    zero_tols: Optional[Tuple[float, float]] = None,
    abs_bins: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized classification of every data point into 5 error bins (`0..4` `int8`).

    Sign convention: `residual = y_pred - y_true` (negative = Underforecast, positive = Overforecast).
    Returns `(bin_codes_int8, residual_float32, finite_mask_bool)` where invalid rows receive code `-1`.
    """
    targets = np.asarray(y_true, dtype=np.float32).ravel()
    preds = np.asarray(y_pred, dtype=np.float32).ravel()
    if len(targets) != len(preds):
        raise ValueError(f"Length mismatch: y_true={len(targets)}, y_pred={len(preds)}.")

    cfg_mode = str(mode if mode is not None else _ERROR_BIN_CONFIG["mode"]).strip().lower()
    p_acc, p_ext = pct_bins if pct_bins is not None else _ERROR_BIN_CONFIG["pct_bins"]
    z_acc, z_ext = zero_tols if zero_tols is not None else _ERROR_BIN_CONFIG["zero_tols"]
    a_acc, a_ext = abs_bins if abs_bins is not None else _ERROR_BIN_CONFIG["abs_bins"]

    finite_mask = np.isfinite(targets) & np.isfinite(preds)
    residual = np.where(finite_mask, preds - targets, np.float32(0.0)).astype(np.float32, copy=False)
    codes = np.full(len(targets), -1, dtype=np.int8)

    if not np.any(finite_mask):
        return codes, residual, finite_mask

    if cfg_mode == "abs":
        r = residual[finite_mask]
        c = np.full(len(r), 2, dtype=np.int8)
        c[r <= -a_ext] = 0
        c[(r > -a_ext) & (r <= -a_acc)] = 1
        c[(r >= a_acc) & (r < a_ext)] = 3
        c[r >= a_ext] = 4
        codes[finite_mask] = c
        return codes, residual, finite_mask

    # Default 'pct' mode (matching legacy/blind_attempt.ipynb with consistent `pred - target` sign)
    t_f = targets[finite_mask]
    p_f = preds[finite_mask]
    r_f = residual[finite_mask]
    c = np.full(len(r_f), 2, dtype=np.int8)

    zero_mask = np.abs(t_f) <= 1e-12
    nonzero_mask = ~zero_mask

    if np.any(nonzero_mask):
        pct_err = (r_f[nonzero_mask] / t_f[nonzero_mask]) * np.float32(100.0)
        cnz = np.full(len(pct_err), 2, dtype=np.int8)
        cnz[pct_err <= -p_ext] = 0
        cnz[(pct_err > -p_ext) & (pct_err <= -p_acc)] = 1
        cnz[(pct_err >= p_acc) & (pct_err < p_ext)] = 3
        cnz[pct_err >= p_ext] = 4
        c[nonzero_mask] = cnz

    if np.any(zero_mask):
        pz = p_f[zero_mask]
        cz = np.full(len(pz), 2, dtype=np.int8)
        cz[pz < -z_ext] = 0
        cz[(pz >= -z_ext) & (pz < -z_acc)] = 1
        cz[(pz > z_acc) & (pz <= z_ext)] = 3
        cz[pz > z_ext] = 4
        c[zero_mask] = cz

    codes[finite_mask] = c
    return codes, residual, finite_mask


def compute_error_bin_shares(
    y_true: Any,
    y_pred: Any,
    *,
    mode: Optional[str] = None,
    pct_bins: Optional[Tuple[float, float]] = None,
    zero_tols: Optional[Tuple[float, float]] = None,
    abs_bins: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Vectorized percentage share of each of the 5 error bins (`0..100` scale)."""
    codes, _, finite_mask = classify_error_bins(
        y_true, y_pred, mode=mode, pct_bins=pct_bins, zero_tols=zero_tols, abs_bins=abs_bins
    )
    valid_codes = codes[finite_mask]
    n_valid = len(valid_codes)
    if n_valid == 0:
        return {
            "extreme_underforecast_pct": 0.0,
            "underforecast_pct": 0.0,
            "accurate_pct": 0.0,
            "overforecast_pct": 0.0,
            "extreme_overforecast_pct": 0.0,
        }
    counts = np.bincount(valid_codes, minlength=5)
    pcts = counts.astype(np.float64) * (100.0 / float(n_valid))
    return {
        "extreme_underforecast_pct": float(pcts[0]),
        "underforecast_pct": float(pcts[1]),
        "accurate_pct": float(pcts[2]),
        "overforecast_pct": float(pcts[3]),
        "extreme_overforecast_pct": float(pcts[4]),
    }


def prepare_tuned_pipeline(pipeline: Any, best_params: Mapping[str, Any]) -> Any:
    """Clone `pipeline`, apply `best_params`, and clear `walk_forward_step_eras` so full-train refit stays RAM-lean."""
    tuned = clone(pipeline)
    if best_params:
        tuned.set_params(**dict(best_params))
    tuned_params = tuned.get_params(deep=True)
    if "estimator__walk_forward_step_eras" in tuned_params:
        tuned.set_params(estimator__walk_forward_step_eras=None)
    elif "walk_forward_step_eras" in tuned_params:
        tuned.set_params(walk_forward_step_eras=None)
    return tuned


def reset_cv_metric_log() -> None:
    """Clear the per-fold CV metric buffer before starting a new `GridSearchCV` / `BayesSearchCV` run."""
    _CV_FOLD_METRIC_LOG.clear()


def candidate_score(estimator: Any, X: pd.DataFrame, y: Any) -> float:
    """Official per-era `numerai_corr` scorer for `GridSearchCV` / `BayesSearchCV`, logging extended fold metrics for `build_quick_results_df`.

    When `estimator__walk_forward_step_eras` is set to a positive integer, evaluates the CV fold using
    leak-free `walk_forward_predict` respecting `target_horizon_eras` settlement lag; otherwise scores statically.
    """
    targets = np.asarray(y, dtype=np.float64).ravel()
    eras = X["era"].to_numpy() if isinstance(X, pd.DataFrame) and "era" in X.columns else None

    inner_est = (
        estimator.named_steps["estimator"]
        if hasattr(estimator, "named_steps") and "estimator" in estimator.named_steps
        else estimator
    )
    wf_step = getattr(inner_est, "walk_forward_step_eras", None)
    wf_train_X = getattr(inner_est, "_wf_train_X_", None)
    wf_train_y = getattr(inner_est, "_wf_train_y_", None)
    wf_train_eras = getattr(inner_est, "_wf_train_eras_", None)

    if (
        wf_step not in (None, False, "None", "none", "null", "", 0)
        and int(wf_step) > 0
        and wf_train_X is not None
        and wf_train_y is not None
        and wf_train_eras is not None
        and eras is not None
    ):
        preds = np.asarray(
            walk_forward_predict(
                estimator,
                wf_train_X,
                pd.Series(wf_train_y),
                pd.Series(wf_train_eras),
                X,
                pd.Series(targets),
                pd.Series(eras),
                step_eras=int(wf_step),
                window_mode=str(getattr(inner_est, "walk_forward_window", "expanding") or "expanding"),
                target_horizon_eras=int(getattr(inner_est, "target_horizon_eras", 4) or 4),
                era_stride=int(getattr(inner_est, "era_stride", 1) or 1),
            ),
            dtype=np.float64,
        ).ravel()
    else:
        preds = np.asarray(estimator.predict(X), dtype=np.float64).ravel()
    era_corrs = compute_per_era_numerai_corr(targets, preds, eras=eras)
    spearman_era_corrs = compute_per_era_spearman_corr(targets, preds, eras=eras)

    mean_corr = float(era_corrs.mean()) if len(era_corrs) else 0.0
    mean_corr = mean_corr if np.isfinite(mean_corr) else 0.0
    spearman_mean = float(spearman_era_corrs.mean()) if len(spearman_era_corrs) else 0.0
    spearman_mean = spearman_mean if np.isfinite(spearman_mean) else 0.0
    sharpe_val = float(sharpe_ratio(era_corrs))
    pos_era_pct = float(np.mean(era_corrs.to_numpy(dtype=np.float64) > 0.0) * 100.0) if len(era_corrs) else 0.0

    if len(targets):
        mae_val = float(mean_absolute_error(targets, preds))
        bias_val = float(np.mean(targets - preds))
        pred_center = 0.5 if (0.15 <= float(np.mean(preds)) <= 0.85) else float(np.mean(preds))
        non_neutral = np.abs(targets - 0.5) > 1e-9
        dir_acc_pct = (
            float(np.mean(np.sign(targets[non_neutral] - 0.5) == np.sign(preds[non_neutral] - pred_center)) * 100.0)
            if np.any(non_neutral)
            else 50.0
        )
    else:
        mae_val = 0.0
        bias_val = 0.0
        dir_acc_pct = 0.0

    cumulative_corr = era_corrs.cumsum()
    if len(cumulative_corr):
        running_peak = pd.Series(
            np.maximum.accumulate(np.insert(cumulative_corr.to_numpy(), 0, 0.0))[1:],
            index=cumulative_corr.index,
        )
        max_dd = float((running_peak - cumulative_corr).max())
    else:
        max_dd = 0.0

    bin_shares = compute_error_bin_shares(targets, preds)

    _CV_FOLD_METRIC_LOG.append({
        "numerai_corr": mean_corr,
        "positive_era_pct": pos_era_pct,
        "directional_accuracy_pct": dir_acc_pct,
        "sharpe": sharpe_val,
        "spearman": spearman_mean,
        "mae": mae_val,
        "forecast_bias": bias_val,
        "max_drawdown": max_dd,
        "era_corrs": era_corrs,
        **bin_shares,
    })

    print(
        f"  [Fold Score] Numerai CORR: {mean_corr * 100.0:+.2f}% ({mean_corr:.4f}) | "
        f"Sharpe: {sharpe_val:.2f} | Positive Eras: {pos_era_pct:.1f}% | "
        f"Dir Acc: {dir_acc_pct:.1f}% | Accurate Bin: {bin_shares['accurate_pct']:.1f}% | MAE: {mae_val:.4f}"
    )
    return mean_corr


def _infer_non_overlapping_stride(era_index: pd.Index, default_stride: int = 4) -> int:
    """Infer stride so non-overlapping 4-week target eras are not double-strided on pre-strided CV folds."""
    step = max(1, int(default_stride))
    if len(era_index) < 2:
        return 1
    numeric_idx = pd.to_numeric(pd.Index(era_index), errors="coerce")
    if numeric_idx.isna().any():
        return step
    diffs = np.diff(numeric_idx.to_numpy(dtype=np.float64))
    pos_diffs = diffs[diffs > 0.0]
    if len(pos_diffs) == 0:
        return step
    median_step = float(np.median(pos_diffs))
    return max(1, int(round(float(step) / median_step)))


def compute_per_era_mmc(
    validation_df: pd.DataFrame,
    preds: Union[pd.Series, np.ndarray],
    meta_model_path: Optional[Union[str, Path]] = None,
    era_col: str = "era",
    target_col: str = "target",
) -> Optional[pd.Series]:
    """Compute per-era Meta Model Contribution (MMC) via `numerai_tools.scoring.correlation_contribution`.

    Merges `meta_model.parquet` onto `validation_df` by `id` (or index) and computes per-era MMC
    with zero-variance guards. Returns `None` gracefully when `meta_model_path` is unavailable.
    """
    if meta_model_path is None:
        return None
    meta_path = Path(meta_model_path)
    if not meta_path.is_file():
        return None
    if era_col not in validation_df.columns or target_col not in validation_df.columns:
        return None

    pred_arr = np.asarray(preds, dtype=np.float64).ravel()
    if len(pred_arr) != len(validation_df):
        raise ValueError(f"preds length ({len(pred_arr)}) must match validation_df rows ({len(validation_df)}).")

    meta_df = pd.read_parquet(meta_path)
    if meta_df.empty:
        return None

    meta_col = (
        "numerai_meta_model"
        if "numerai_meta_model" in meta_df.columns
        else next(
            (
                str(c)
                for c in meta_df.columns
                if str(c) not in ("id", era_col, "date", "data_type")
                and pd.api.types.is_numeric_dtype(meta_df[c])
            ),
            None,
        )
    )
    if meta_col is None:
        return None

    work_cols = [era_col, target_col] + (["id"] if "id" in validation_df.columns else [])
    work_df = validation_df[work_cols].copy()
    work_df["prediction"] = pred_arr

    if "id" in work_df.columns and "id" in meta_df.columns:
        merged = work_df.merge(
            meta_df[["id", meta_col]].rename(columns={meta_col: "numerai_meta_model"}),
            on="id",
            how="inner",
        )
    elif "id" in work_df.columns and meta_df.index.name == "id":
        merged = work_df.merge(
            meta_df[[meta_col]].rename(columns={meta_col: "numerai_meta_model"}),
            left_on="id",
            right_index=True,
            how="inner",
        )
    else:
        meta_aligned = meta_df[meta_col].reindex(work_df.index)
        merged = work_df.copy()
        merged["numerai_meta_model"] = meta_aligned.to_numpy(dtype=np.float64)

    merged = merged.dropna(subset=["prediction", "numerai_meta_model", target_col])
    if merged.empty:
        return None

    def _safe_era_mmc(grp: pd.DataFrame) -> float:
        p = grp["prediction"].to_numpy(dtype=np.float64, copy=False)
        m = grp["numerai_meta_model"].to_numpy(dtype=np.float64, copy=False)
        t = grp[target_col].to_numpy(dtype=np.float64, copy=False)
        if len(t) < 2 or _is_constant_1d(p) or _is_constant_1d(m) or _is_constant_1d(t):
            return 0.0
        val = float(
            correlation_contribution(
                grp[["prediction"]],
                grp["numerai_meta_model"].rename("numerai_meta_model"),
                grp[target_col],
            ).to_numpy(dtype=np.float64)[0]
        )
        return val if np.isfinite(val) else 0.0

    mmc_series = merged.groupby(era_col, sort=True).apply(_safe_era_mmc, include_groups=False)
    return mmc_series.astype(np.float64).rename("mmc")


def _compute_clipped_era_returns(
    per_era_corr: pd.Series,
    *,
    per_era_mmc: Optional[pd.Series] = None,
    corr_weight: float = 8.0,
    mmc_weight: float = 32.0,
    payout_factor: Union[float, pd.Series] = 1.0,
    clip_min: float = -0.25,
    clip_max: float = 0.25,
    churn_penalty: Optional[Union[float, pd.Series]] = None,
    stride: int = 4,
    corr_multiplier: Optional[float] = None,
) -> Tuple[pd.Series, bool]:
    """Compute chronologically sorted Numerai Round 1393+ payout returns per non-overlapping round.

    Official Classic R1393+ formula:
        weighted_score_t = corr_weight * NCORR_t + mmc_weight * NMMC_t   (8 * NCORR + 32 * NMMC)
        return_t = clip(payout_factor_t * weighted_score_t, clip_min=-1.0, clip_max=+2.0)
    Optional R1399+ neutral-churn penalty reduces positive payouts only after clipping.
    """
    corr_series = pd.Series(per_era_corr, dtype=np.float64).dropna().sort_index()
    step = max(1, int(stride))
    if step > 1 and len(corr_series) > 0:
        corr_series = corr_series.loc[corr_series.index[::step]]
    if len(corr_series) == 0:
        return pd.Series(dtype=np.float64), False

    eff_corr_w = float(corr_multiplier) if corr_multiplier is not None else float(corr_weight)
    has_mmc = per_era_mmc is not None and len(per_era_mmc) > 0
    if has_mmc:
        mmc_series = (
            pd.Series(per_era_mmc, dtype=np.float64)
            .dropna()
            .sort_index()
            .reindex(corr_series.index)
            .fillna(0.0)
        )
        raw_score = eff_corr_w * corr_series + float(mmc_weight) * mmc_series
    else:
        raw_score = eff_corr_w * corr_series

    if isinstance(payout_factor, pd.Series):
        pf_aligned = (
            pd.Series(payout_factor, dtype=np.float64)
            .dropna()
            .sort_index()
            .reindex(corr_series.index)
            .ffill()
            .bfill()
            .fillna(1.0)
        )
        era_returns = (raw_score * pf_aligned).clip(float(clip_min), float(clip_max))
    else:
        era_returns = (raw_score * float(payout_factor)).clip(float(clip_min), float(clip_max))

    if churn_penalty is not None:
        if isinstance(churn_penalty, pd.Series):
            cp = (
                pd.Series(churn_penalty, dtype=np.float64)
                .dropna()
                .sort_index()
                .reindex(era_returns.index)
                .ffill()
                .bfill()
                .fillna(0.0)
                .clip(0.0, 1.0)
            )
            era_returns = pd.Series(
                np.where(era_returns > 0.0, era_returns * (1.0 - cp), era_returns),
                index=era_returns.index,
                dtype=np.float64,
            )
        else:
            cp_val = float(np.clip(float(churn_penalty), 0.0, 1.0))
            era_returns = pd.Series(
                np.where(era_returns > 0.0, era_returns * (1.0 - cp_val), era_returns),
                index=era_returns.index,
                dtype=np.float64,
            )

    return era_returns.astype(np.float64), has_mmc


def _simulate_single_window_path(
    ret_arr: np.ndarray,
    initial_stake: float,
    staking_mode: str,
) -> Tuple[np.ndarray, float, float, float]:
    """Simulate a single independent Atomic-staking window starting at `initial_stake` (zero cross-window compounding)."""
    init_stake = float(initial_stake)
    if len(ret_arr) == 0:
        return np.empty(0, dtype=np.float64), 0.0, 0.0, 0.0

    if staking_mode == "constant":
        # Atomic Constant staking: fixed `initial_stake` staked per settled round, excess profits swept idle,
        # downside bounded at 0 (cannot lose more than available capital).
        cum_pnl = np.cumsum(init_stake * ret_arr)
        eq = np.maximum(0.0, init_stake + cum_pnl)
        # Once wiped out (eq == 0), stay at 0
        wiped = np.maximum.accumulate(eq <= 0.0)
        eq = np.where(wiped, 0.0, eq)
    else:
        # Atomic Compound staking: settled round balance rolls into next round
        eq = init_stake * np.cumprod(np.maximum(0.0, 1.0 + ret_arr))

    running_peak = np.maximum.accumulate(np.maximum(init_stake, eq))
    dd_gbp = eq - running_peak
    dd_pct = np.where(running_peak > 0.0, (dd_gbp / running_peak) * 100.0, 0.0)
    prev_eq = np.concatenate(([init_stake], eq[:-1]))
    round_pnl_gbp = eq - prev_eq
    return eq, float(dd_pct.min()), float(dd_gbp.min()), float(round_pnl_gbp.min())


def _partition_window_slices(n_rounds: int, window_rounds: int) -> List[Tuple[int, int]]:
    """Split `n_rounds` non-overlapping rounds into independent chronological windows."""
    if n_rounds <= 0:
        return []
    w_len = max(1, int(window_rounds))
    if n_rounds <= w_len:
        return [(0, n_rounds)]
    n_windows = max(1, n_rounds // w_len)
    edges = np.linspace(0, n_rounds, num=n_windows + 1, dtype=np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_windows) if edges[i + 1] > edges[i]]


def simulate_stake_pnl(
    per_era_corr: pd.Series,
    per_era_mmc: Optional[pd.Series] = None,
    initial_stake: float = 1000.0,
    corr_weight: float = 8.0,
    mmc_weight: float = 32.0,
    payout_factor: Union[float, pd.Series] = 1.0,
    clip_min: float = -0.25,
    clip_max: float = 0.25,
    churn_penalty: Optional[Union[float, pd.Series]] = None,
    staking_mode: str = "compound",
    stride: int = 4,
    window_rounds: int = 13,
    score_basis: str = "corr_mmc_proxy",
    nmr_gbp_prices: Optional[pd.Series] = None,
    corr_multiplier: Optional[float] = None,
) -> Dict[str, Any]:
    """Numerai Classic Round 1393+ Atomic Staking P&L simulator (£) across independent validation windows.

    Uses the official R1393+ payout formula (`8 * NCORR + 32 * NMMC`, clipped to `[-100%, +200%]`,
    `payout_factor=1.0`, optional R1399+ neutral-churn penalty, NO arbitrary live-decay haircut).
    Partitions non-overlapping settled rounds (`stride=4` -> 13 rounds ≈ 1 year per window) into
    financially **independent** validation windows each starting fresh at `initial_stake` (£1,000),
    eliminating artificial compounding across unrelated historical regimes.
    """
    init_stake = float(initial_stake)
    clean_mode = str(staking_mode or "compound").strip().lower()
    if clean_mode not in ("compound", "constant"):
        raise ValueError(f"staking_mode must be 'compound' or 'constant', got {staking_mode!r}")

    era_returns, has_mmc = _compute_clipped_era_returns(
        per_era_corr,
        per_era_mmc=per_era_mmc,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
        payout_factor=payout_factor,
        clip_min=clip_min,
        clip_max=clip_max,
        churn_penalty=churn_penalty,
        stride=stride,
        corr_multiplier=corr_multiplier,
    )
    if len(era_returns) == 0:
        empty_series = pd.Series(dtype=np.float64)
        return {
            "mean_ending_capital": init_stake,
            "median_ending_capital": init_stake,
            "mean_profit": 0.0,
            "median_profit": 0.0,
            "mean_return_pct": 0.0,
            "std_profit": 0.0,
            "profitable_windows_pct": 0.0,
            "worst_window_ending_capital": init_stake,
            "worst_window_profit": 0.0,
            "best_window_ending_capital": init_stake,
            "best_window_profit": 0.0,
            "final_stake": init_stake,
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_gbp": 0.0,
            "worst_era_gbp": 0.0,
            "n_windows": 0,
            "score_basis": str(score_basis),
            "has_mmc": False,
            "window_outcomes_df": pd.DataFrame(),
            "round_returns": empty_series,
            "equity_curve": empty_series,
            "drawdown_pct_series": empty_series,
        }

    ret_arr = era_returns.to_numpy(dtype=np.float64)
    idx = era_returns.index
    fx_arr: Optional[np.ndarray] = None
    if nmr_gbp_prices is not None and len(nmr_gbp_prices) > 0:
        fx_aligned = (
            pd.Series(nmr_gbp_prices, dtype=np.float64)
            .dropna()
            .sort_index()
            .reindex(idx)
            .ffill()
            .bfill()
        )
        if len(fx_aligned) == len(idx) and not fx_aligned.isna().any():
            fx_arr = fx_aligned.to_numpy(dtype=np.float64)

    slices = _partition_window_slices(len(ret_arr), window_rounds=window_rounds)
    window_rows: List[Dict[str, Any]] = []
    window_curves: Dict[str, pd.Series] = {}

    for w_i, (s, e) in enumerate(slices, start=1):
        w_ret = ret_arr[s:e]
        w_idx = idx[s:e]
        w_eq, w_dd_pct, w_dd_gbp, w_worst_round = _simulate_single_window_path(w_ret, init_stake, clean_mode)
        if fx_arr is not None and fx_arr[s] > 0.0:
            w_eq = w_eq * (fx_arr[s:e] / fx_arr[s])
            w_peak = np.maximum.accumulate(np.maximum(init_stake, w_eq))
            w_dd_gbp_arr = w_eq - w_peak
            w_dd_pct = float(np.where(w_peak > 0.0, (w_dd_gbp_arr / w_peak) * 100.0, 0.0).min())
            w_dd_gbp = float(w_dd_gbp_arr.min())
            w_prev = np.concatenate(([init_stake], w_eq[:-1]))
            w_worst_round = float((w_eq - w_prev).min())

        end_cap = float(w_eq[-1])
        profit = end_cap - init_stake
        ret_pct = (profit / init_stake) * 100.0 if abs(init_stake) > 1e-12 else 0.0
        w_label = f"W{w_i} ({w_idx[0]}–{w_idx[-1]})"
        window_curves[w_label] = pd.Series(w_eq, index=w_idx, dtype=np.float64, name=w_label)
        window_rows.append({
            "window": w_label,
            "start_era": w_idx[0],
            "end_era": w_idx[-1],
            "n_rounds": int(len(w_ret)),
            "ending_capital": round(end_cap, 2),
            "profit_gbp": round(profit, 2),
            "return_pct": round(ret_pct, 2),
            "max_drawdown_pct": round(w_dd_pct, 2),
            "max_drawdown_gbp": round(w_dd_gbp, 2),
            "worst_round_gbp": round(w_worst_round, 2),
            "profitable": bool(profit > 0.0),
        })

    win_df = pd.DataFrame(window_rows)
    end_caps = win_df["ending_capital"].to_numpy(dtype=np.float64)
    profits = win_df["profit_gbp"].to_numpy(dtype=np.float64)
    ret_pcts = win_df["return_pct"].to_numpy(dtype=np.float64)

    mean_end = float(np.mean(end_caps))
    median_end = float(np.median(end_caps))
    mean_prof = float(np.mean(profits))
    median_prof = float(np.median(profits))
    mean_ret = float(np.mean(ret_pcts))
    std_prof = float(np.std(profits, ddof=0))
    prof_pct = float(np.mean(profits > 0.0) * 100.0)
    worst_end = float(np.min(end_caps))
    worst_prof = float(np.min(profits))
    best_end = float(np.max(end_caps))
    best_prof = float(np.max(profits))
    worst_dd_pct = float(win_df["max_drawdown_pct"].min())
    worst_dd_gbp = float(win_df["max_drawdown_gbp"].min())
    worst_era_gbp = float(win_df["worst_round_gbp"].min())

    # Independent per-round (£1,000 per single round) equity & drawdown series
    single_round_eq = pd.Series(init_stake * (1.0 + ret_arr), index=idx, dtype=np.float64, name="independent_round_gbp")
    single_round_dd_pct = pd.Series(np.minimum(0.0, ret_arr) * 100.0, index=idx, dtype=np.float64, name="round_burn_pct")

    return {
        "mean_ending_capital": mean_end,
        "median_ending_capital": median_end,
        "mean_profit": mean_prof,
        "median_profit": median_prof,
        "mean_return_pct": mean_ret,
        "std_profit": std_prof,
        "profitable_windows_pct": prof_pct,
        "worst_window_ending_capital": worst_end,
        "worst_window_profit": worst_prof,
        "best_window_ending_capital": best_end,
        "best_window_profit": best_prof,
        # Scalar summary keys aligned with CV/comparison tables (mean across independent windows)
        "final_stake": mean_end,
        "net_profit": mean_prof,
        "total_return_pct": mean_ret,
        "max_drawdown_pct": worst_dd_pct,
        "max_drawdown_gbp": worst_dd_gbp,
        "worst_era_gbp": worst_era_gbp,
        "n_windows": int(len(win_df)),
        "score_basis": str(score_basis),
        "has_mmc": bool(has_mmc),
        "window_outcomes_df": win_df,
        "window_curves": window_curves,
        "round_returns": era_returns,
        "equity_curve": single_round_eq,
        "drawdown_pct_series": single_round_dd_pct,
    }


def simulate_stake_pnl_monte_carlo(
    per_era_corr: pd.Series,
    per_era_mmc: Optional[pd.Series] = None,
    initial_stake: float = 1000.0,
    corr_weight: float = 8.0,
    mmc_weight: float = 32.0,
    payout_factor: Union[float, pd.Series] = 1.0,
    clip_min: float = -0.25,
    clip_max: float = 0.25,
    churn_penalty: Optional[Union[float, pd.Series]] = None,
    staking_mode: str = "compound",
    stride: int = 4,
    window_rounds: int = 13,
    n_sims: int = 1000,
    block_size: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Within-window block-bootstrap Monte Carlo simulator under Numerai Classic R1393+ Atomic Staking.

    Resamples actual per-round returns *inside each independent validation window* (each starting at
    `initial_stake`), then aggregates the terminal capital and max-drawdown distributions across all windows.
    """
    init_stake = float(initial_stake)
    clean_mode = str(staking_mode or "compound").strip().lower()
    if clean_mode not in ("compound", "constant"):
        raise ValueError(f"staking_mode must be 'compound' or 'constant', got {staking_mode!r}")

    era_returns, _ = _compute_clipped_era_returns(
        per_era_corr,
        per_era_mmc=per_era_mmc,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
        payout_factor=payout_factor,
        clip_min=clip_min,
        clip_max=clip_max,
        churn_penalty=churn_penalty,
        stride=stride,
    )
    if len(era_returns) == 0:
        return {
            "mc_p05_final_stake": init_stake,
            "mc_median_final_stake": init_stake,
            "mc_mean_final_stake": init_stake,
            "mc_p95_final_stake": init_stake,
            "mc_prob_profit_pct": 0.0,
            "mc_p95_max_drawdown_pct": 0.0,
            "window_mc_df": pd.DataFrame(),
            "final_stakes": np.empty(0, dtype=np.float64),
            "max_drawdowns_pct": np.empty(0, dtype=np.float64),
        }

    ret_arr = era_returns.to_numpy(dtype=np.float64)
    idx = era_returns.index
    slices = _partition_window_slices(len(ret_arr), window_rounds=window_rounds)
    n_s = max(10, int(n_sims))
    rng = np.random.default_rng(int(random_state))

    all_final_stakes: List[np.ndarray] = []
    all_max_dds: List[np.ndarray] = []
    win_mc_rows: List[Dict[str, Any]] = []

    for w_i, (s, e) in enumerate(slices, start=1):
        w_ret = ret_arr[s:e]
        w_idx = idx[s:e]
        n_t = len(w_ret)
        b_size = min(max(1, int(block_size)), n_t)
        n_blocks = int(np.ceil(float(n_t) / float(b_size)))

        starts = rng.integers(0, max(1, n_t - b_size + 1), size=(n_s, n_blocks))
        offsets = np.arange(b_size, dtype=np.int64)[None, None, :]
        boot_idx = (starts[:, :, None] + offsets).reshape(n_s, -1)[:, :n_t]
        sim_ret = w_ret[boot_idx]  # (n_sims, n_t)

        if clean_mode == "constant":
            sim_eq = np.maximum(0.0, init_stake + np.cumsum(init_stake * sim_ret, axis=1))
            wiped = np.maximum.accumulate(sim_eq <= 0.0, axis=1)
            sim_eq = np.where(wiped, 0.0, sim_eq)
        else:
            sim_eq = init_stake * np.cumprod(np.maximum(0.0, 1.0 + sim_ret), axis=1)

        running_peak = np.maximum.accumulate(np.maximum(init_stake, sim_eq), axis=1)
        sim_dd_pct = np.where(running_peak > 0.0, (sim_eq - running_peak) / running_peak, 0.0) * 100.0
        w_max_dd = sim_dd_pct.min(axis=1)
        w_finals = sim_eq[:, -1]

        hist_eq, _, _, _ = _simulate_single_window_path(w_ret, init_stake, clean_mode)
        all_final_stakes.append(w_finals)
        all_max_dds.append(w_max_dd)
        win_mc_rows.append({
            "window": f"W{w_i} ({w_idx[0]}–{w_idx[-1]})",
            "actual_ending_capital": float(hist_eq[-1]),
            "p05": float(np.percentile(w_finals, 5.0)),
            "p25": float(np.percentile(w_finals, 25.0)),
            "p50": float(np.percentile(w_finals, 50.0)),
            "p75": float(np.percentile(w_finals, 75.0)),
            "p95": float(np.percentile(w_finals, 95.0)),
            "prob_profit_pct": float(np.mean(w_finals > init_stake) * 100.0),
            "p95_worst_dd_pct": float(np.percentile(w_max_dd, 5.0)),
        })

    agg_finals = np.concatenate(all_final_stakes)
    agg_dds = np.concatenate(all_max_dds)

    return {
        "mc_p05_final_stake": float(np.percentile(agg_finals, 5.0)),
        "mc_median_final_stake": float(np.percentile(agg_finals, 50.0)),
        "mc_mean_final_stake": float(np.mean(agg_finals)),
        "mc_p95_final_stake": float(np.percentile(agg_finals, 95.0)),
        "mc_prob_profit_pct": float(np.mean(agg_finals > init_stake) * 100.0),
        "mc_p95_max_drawdown_pct": float(np.percentile(agg_dds, 5.0)),
        "window_mc_df": pd.DataFrame(win_mc_rows),
        "final_stakes": agg_finals,
        "max_drawdowns_pct": agg_dds,
    }


def plot_stake_pnl_dashboard(
    per_era_corr: pd.Series,
    *,
    per_era_mmc: Optional[pd.Series] = None,
    title: str = "Best Model",
    initial_stake: float = 1000.0,
    corr_weight: float = 8.0,
    mmc_weight: float = 32.0,
    payout_factor: Union[float, pd.Series] = 1.0,
    clip_min: float = -0.25,
    clip_max: float = 0.25,
    churn_penalty: Optional[Union[float, pd.Series]] = None,
    stride: int = 4,
    window_rounds: int = 13,
    score_basis: str = "corr_mmc_proxy",
    nmr_gbp_prices: Optional[pd.Series] = None,
    save_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Plot a 2-panel Independent-Window Numerai R1393+ Atomic Staking Dashboard (Constant vs Compound + Downside Stress)."""
    import matplotlib.pyplot as plt

    comp_pnl = simulate_stake_pnl(
        per_era_corr,
        per_era_mmc=per_era_mmc,
        initial_stake=initial_stake,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
        payout_factor=payout_factor,
        clip_min=clip_min,
        clip_max=clip_max,
        churn_penalty=churn_penalty,
        staking_mode="compound",
        stride=stride,
        window_rounds=window_rounds,
        score_basis=score_basis,
        nmr_gbp_prices=nmr_gbp_prices,
    )
    const_pnl = simulate_stake_pnl(
        per_era_corr,
        per_era_mmc=per_era_mmc,
        initial_stake=initial_stake,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
        payout_factor=payout_factor,
        clip_min=clip_min,
        clip_max=clip_max,
        churn_penalty=churn_penalty,
        staking_mode="constant",
        stride=stride,
        window_rounds=window_rounds,
        score_basis=score_basis,
        nmr_gbp_prices=nmr_gbp_prices,
    )
    comp_df = comp_pnl["window_outcomes_df"]
    const_df = const_pnl["window_outcomes_df"]
    if not isinstance(comp_df, pd.DataFrame) or comp_df.empty:
        return comp_pnl

    x_pos = np.arange(len(comp_df))
    w_labels = comp_df["window"].astype(str).tolist()
    bar_w = 0.36

    fig, (ax_cap, ax_dd) = plt.subplots(
        2, 1, figsize=(15.5, 8.0), sharex=True, gridspec_kw={"height_ratios": [1.85, 1.0]}
    )
    fig.patch.set_facecolor("white")

    # --- Top Subplot: Independent Window Ending Capital (Constant vs Compound) ---
    comp_caps = comp_df["ending_capital"].to_numpy(dtype=np.float64)
    const_caps = const_df["ending_capital"].to_numpy(dtype=np.float64)
    ax_cap.bar(x_pos - bar_w / 2.0, const_caps, width=bar_w, color="#7c3aed", alpha=0.84, label=f"Constant (Mean £{const_pnl['mean_ending_capital']:,.0f} | Med £{const_pnl['median_ending_capital']:,.0f})")
    ax_cap.bar(x_pos + bar_w / 2.0, comp_caps, width=bar_w, color="#059669", alpha=0.86, label=f"Compound (Mean £{comp_pnl['mean_ending_capital']:,.0f} | Med £{comp_pnl['median_ending_capital']:,.0f})")
    ax_cap.axhline(float(initial_stake), color="#475569", linestyle="--", linewidth=1.3, label=f"Initial Capital (£{initial_stake:,.0f})")
    ax_cap.axhline(float(comp_pnl["mean_ending_capital"]), color="#059669", linestyle=":", linewidth=1.2, alpha=0.8)
    ax_cap.axhline(float(const_pnl["mean_ending_capital"]), color="#7c3aed", linestyle=":", linewidth=1.2, alpha=0.8)
    ax_cap.set_ylabel("Independent Window Ending Capital (£)")
    ax_cap.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax_cap.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.8)

    summary_box = (
        f"Compound — Mean: £{comp_pnl['mean_ending_capital']:,.0f} | Median: £{comp_pnl['median_ending_capital']:,.0f}\n"
        f"  Mean P&L: £{comp_pnl['mean_profit']:+,.0f} (±£{comp_pnl['std_profit']:,.0f}) | Profitable: {comp_pnl['profitable_windows_pct']:.0f}%\n"
        f"  Worst Window: £{comp_pnl['worst_window_ending_capital']:,.0f} | Best: £{comp_pnl['best_window_ending_capital']:,.0f}\n"
        f"Constant — Mean: £{const_pnl['mean_ending_capital']:,.0f} | Worst: £{const_pnl['worst_window_ending_capital']:,.0f}"
    )
    ax_cap.text(
        0.99,
        0.95,
        summary_box,
        transform=ax_cap.transAxes,
        ha="right",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )

    # --- Bottom Subplot: Within-Window Max Drawdown (%) ---
    comp_dds = comp_df["max_drawdown_pct"].to_numpy(dtype=np.float64)
    const_dds = const_df["max_drawdown_pct"].to_numpy(dtype=np.float64)
    ax_dd.bar(x_pos - bar_w / 2.0, const_dds, width=bar_w, color="#f97316", alpha=0.80, label="Constant Max DD (%)")
    ax_dd.bar(x_pos + bar_w / 2.0, comp_dds, width=bar_w, color="#dc2626", alpha=0.82, label="Compound Max DD (%)")
    ax_dd.axhline(0.0, color="#475569", linewidth=1.0)
    min_dd = min(-5.0, float(min(comp_dds.min(), const_dds.min())) * 1.18)
    ax_dd.set_ylim(min_dd, 2.0)
    ax_dd.set_ylabel("Within-Window Max DD (%)")
    ax_dd.set_xlabel("Independent Chronological Validation Windows (Each Starting Fresh at £1,000)")
    ax_dd.set_xticks(x_pos)
    ax_dd.set_xticklabels(w_labels, rotation=32, ha="right", fontsize=8.5)
    ax_dd.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax_dd.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.8)

    basis_tag = (
        "Official NCORR/NMMC"
        if str(score_basis).strip().lower() == "ncorr_nmmc"
        else "PROXY (CORR/MMC on R1393+ Formula)"
    )
    mmc_tag = f"{corr_weight:g}×CORR + {mmc_weight:g}×MMC" if comp_pnl["has_mmc"] else f"{corr_weight:g}×CORR only (no meta_model.parquet)"
    pf_label = "Series" if isinstance(payout_factor, pd.Series) else f"{float(payout_factor):.2f}"
    fig.suptitle(f"Numerai Classic R1393+ Independent-Window Staking P&L — {title}", fontsize=13.2, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.938,
        f"{basis_tag} | {mmc_tag} | clip=[{clip_min * 100:.0f}%, +{clip_max * 100:.0f}%] | PF={pf_label} | stride={stride} | no cross-window compounding",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#475569",
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=140)
    plt.show()
    comp_pnl["constant_summary"] = {
        "mean_ending_capital": const_pnl["mean_ending_capital"],
        "median_ending_capital": const_pnl["median_ending_capital"],
        "mean_profit": const_pnl["mean_profit"],
        "worst_window_ending_capital": const_pnl["worst_window_ending_capital"],
        "profitable_windows_pct": const_pnl["profitable_windows_pct"],
    }
    return comp_pnl


def plot_stake_pnl_monte_carlo(
    per_era_corr: pd.Series,
    *,
    per_era_mmc: Optional[pd.Series] = None,
    title: str = "Best Model",
    initial_stake: float = 1000.0,
    corr_weight: float = 8.0,
    mmc_weight: float = 32.0,
    payout_factor: Union[float, pd.Series] = 1.0,
    clip_min: float = -0.25,
    clip_max: float = 0.25,
    churn_penalty: Optional[Union[float, pd.Series]] = None,
    stride: int = 4,
    window_rounds: int = 13,
    score_basis: str = "corr_mmc_proxy",
    n_sims: int = 1000,
    block_size: int = 3,
    random_state: int = 42,
    save_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Plot a separate 2-panel Within-Window Block-Bootstrap Monte Carlo Dashboard (Aggregated Distribution + Per-Window Uncertainty)."""
    import matplotlib.pyplot as plt

    mc_comp = simulate_stake_pnl_monte_carlo(
        per_era_corr,
        per_era_mmc=per_era_mmc,
        initial_stake=initial_stake,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
        payout_factor=payout_factor,
        clip_min=clip_min,
        clip_max=clip_max,
        churn_penalty=churn_penalty,
        staking_mode="compound",
        stride=stride,
        window_rounds=window_rounds,
        n_sims=n_sims,
        block_size=block_size,
        random_state=random_state,
    )
    mc_const = simulate_stake_pnl_monte_carlo(
        per_era_corr,
        per_era_mmc=per_era_mmc,
        initial_stake=initial_stake,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
        payout_factor=payout_factor,
        clip_min=clip_min,
        clip_max=clip_max,
        churn_penalty=churn_penalty,
        staking_mode="constant",
        stride=stride,
        window_rounds=window_rounds,
        n_sims=n_sims,
        block_size=block_size,
        random_state=random_state,
    )
    win_mc_df = mc_comp["window_mc_df"]
    if not isinstance(win_mc_df, pd.DataFrame) or win_mc_df.empty:
        return mc_comp

    fig, (ax_dist, ax_win) = plt.subplots(
        1, 2, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.15, 1.35]}
    )
    fig.patch.set_facecolor("white")

    # --- Left Panel: Aggregated Terminal Capital Distribution (Constant vs Compound) ---
    comp_f = np.asarray(mc_comp["final_stakes"], dtype=np.float64)
    const_f = np.asarray(mc_const["final_stakes"], dtype=np.float64)
    p99 = float(np.percentile(np.concatenate([comp_f, const_f]), 99.0))
    comp_plot = np.clip(comp_f, 0.0, max(initial_stake * 1.5, p99))
    const_plot = np.clip(const_f, 0.0, max(initial_stake * 1.5, p99))

    ax_dist.hist(const_plot, bins=35, color="#7c3aed", alpha=0.45, edgecolor="white", label=f"Constant (Med £{mc_const['mc_median_final_stake']:,.0f})")
    ax_dist.hist(comp_plot, bins=35, color="#059669", alpha=0.50, edgecolor="white", label=f"Compound (Med £{mc_comp['mc_median_final_stake']:,.0f})")
    ax_dist.axvline(float(initial_stake), color="#475569", linestyle="--", linewidth=1.4, label=f"Initial (£{initial_stake:,.0f})")
    ax_dist.axvline(float(mc_comp["mc_p05_final_stake"]), color="#dc2626", linestyle=":", linewidth=1.6, label=f"Comp 5th Pct (£{mc_comp['mc_p05_final_stake']:,.0f})")
    ax_dist.set_xlabel("Terminal Window Capital (£, clipped at 99th pct for display)")
    ax_dist.set_ylabel("Simulated Paths (Aggregated Across Windows)")
    ax_dist.set_title("Aggregated Within-Window Terminal Capital", fontsize=11.8, fontweight="bold")
    ax_dist.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax_dist.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

    risk_box = (
        f"Compound MC — P(Profit): {mc_comp['mc_prob_profit_pct']:.1f}%\n"
        f"  5th / 50th / 95th: £{mc_comp['mc_p05_final_stake']:,.0f} / £{mc_comp['mc_median_final_stake']:,.0f} / £{mc_comp['mc_p95_final_stake']:,.0f}\n"
        f"  95% Worst DD: {mc_comp['mc_p95_max_drawdown_pct']:.1f}%\n"
        f"Constant MC — P(Profit): {mc_const['mc_prob_profit_pct']:.1f}% | 5th: £{mc_const['mc_p05_final_stake']:,.0f}"
    )
    ax_dist.text(
        0.03,
        0.95,
        risk_box,
        transform=ax_dist.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )

    # --- Right Panel: Per-Window Monte Carlo Uncertainty Intervals (Compound) ---
    wx = np.arange(len(win_mc_df))
    p05 = win_mc_df["p05"].to_numpy(dtype=np.float64)
    p25 = win_mc_df["p25"].to_numpy(dtype=np.float64)
    p50 = win_mc_df["p50"].to_numpy(dtype=np.float64)
    p75 = win_mc_df["p75"].to_numpy(dtype=np.float64)
    p95 = win_mc_df["p95"].to_numpy(dtype=np.float64)
    actual = win_mc_df["actual_ending_capital"].to_numpy(dtype=np.float64)

    ax_win.vlines(wx, p05, p95, color="#3b82f6", linewidth=2.0, alpha=0.55, label="5th–95th Pct Range")
    ax_win.vlines(wx, p25, p75, color="#1d4ed8", linewidth=6.5, alpha=0.75, label="25th–75th IQR")
    ax_win.scatter(wx, p50, color="#1e3a8a", s=38, zorder=5, label="MC Median")
    ax_win.scatter(wx, actual, color="#059669", marker="D", s=42, zorder=6, label="Actual Window Outcome")
    ax_win.axhline(float(initial_stake), color="#475569", linestyle="--", linewidth=1.3, label=f"Initial (£{initial_stake:,.0f})")
    ax_win.set_xticks(wx)
    ax_win.set_xticklabels(win_mc_df["window"].astype(str).tolist(), rotation=32, ha="right", fontsize=8.5)
    ax_win.set_ylabel("Window Ending Capital (£)")
    ax_win.set_xlabel("Independent Validation Window")
    ax_win.set_title("Per-Window Monte Carlo Uncertainty (Compound)", fontsize=11.8, fontweight="bold")
    ax_win.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax_win.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

    basis_tag = (
        "Official NCORR/NMMC"
        if str(score_basis).strip().lower() == "ncorr_nmmc"
        else "PROXY (CORR/MMC on R1393+ Formula)"
    )
    fig.suptitle(f"Within-Window Monte Carlo Block-Bootstrap Risk — {title}", fontsize=13.2, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.925,
        f"{basis_tag} | n_sims={n_sims:,}/window | block_size={block_size} rounds | each window starts fresh at £{initial_stake:,.0f}",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#475569",
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=140)
    plt.show()
    mc_comp["constant_mc"] = {
        "mc_p05_final_stake": mc_const["mc_p05_final_stake"],
        "mc_median_final_stake": mc_const["mc_median_final_stake"],
        "mc_prob_profit_pct": mc_const["mc_prob_profit_pct"],
    }
    return mc_comp


def build_quick_results_df(search_or_results: Any) -> pd.DataFrame:
    """Build a flat candidate results table with percentage score, extended CV metrics, OOF staking P&L scalars, and every hyperparameter expanded into its own column."""
    cv_res = getattr(search_or_results, "cv_results_", search_or_results)
    if not cv_res:
        _CV_FOLD_METRIC_LOG.clear()
        return pd.DataFrame()
    df = pd.DataFrame(cv_res)
    if "mean_test_score" in df.columns:
        df["mean_test_corr_pct"] = (df["mean_test_score"].astype(float) * 100.0).round(2)

    # Attach per-candidate extended metrics & OOF staking P&L scalars logged by `candidate_score`
    split_cols = sorted(
        [col for col in df.columns if col.startswith("split") and col.endswith("_test_score")],
        key=lambda c: int(c[len("split") : c.index("_test_score")]),
    )
    if _CV_FOLD_METRIC_LOG and split_cols:
        metric_specs = (
            ("mean_positive_era_pct", "positive_era_pct", 2),
            ("mean_directional_accuracy_pct", "directional_accuracy_pct", 2),
            ("mean_accurate_pct", "accurate_pct", 2),
            ("mean_underforecast_pct", "underforecast_pct", 2),
            ("mean_extreme_underforecast_pct", "extreme_underforecast_pct", 2),
            ("mean_overforecast_pct", "overforecast_pct", 2),
            ("mean_extreme_overforecast_pct", "extreme_overforecast_pct", 2),
            ("mean_sharpe", "sharpe", 4),
            ("mean_spearman", "spearman", 4),
            ("mean_mae", "mae", 4),
            ("mean_forecast_bias", "forecast_bias", 4),
            ("mean_max_drawdown", "max_drawdown", 4),
        )
        pnl_scalar_keys = (
            "final_stake",
            "net_profit",
            "total_return_pct",
            "max_drawdown_pct",
            "max_drawdown_gbp",
            "worst_era_gbp",
        )
        used_mask = [False] * len(_CV_FOLD_METRIC_LOG)
        agg_cols: Dict[str, List[float]] = {out_col: [] for out_col, _, _ in metric_specs}
        pnl_cols: Dict[str, List[float]] = {k: [] for k in pnl_scalar_keys}
        for row_idx in range(len(df)):
            matched_records: List[Dict[str, Any]] = []
            for s_col in split_cols:
                score_val = float(df.loc[row_idx, s_col])
                if not np.isfinite(score_val):
                    continue
                for log_idx, rec in enumerate(_CV_FOLD_METRIC_LOG):
                    if not used_mask[log_idx] and abs(float(rec["numerai_corr"]) - score_val) <= 1e-12:
                        used_mask[log_idx] = True
                        matched_records.append(rec)
                        break
            for out_col, src_key, decimals in metric_specs:
                if matched_records:
                    val = float(np.mean([float(rec[src_key]) for rec in matched_records]))
                    agg_cols[out_col].append(round(val, decimals))
                else:
                    agg_cols[out_col].append(np.nan)
            fold_era_corrs = [
                rec["era_corrs"]
                for rec in matched_records
                if isinstance(rec.get("era_corrs"), pd.Series) and len(rec["era_corrs"]) > 0
            ]
            if fold_era_corrs:
                stitched_oof = pd.concat(fold_era_corrs).sort_index()
                stitched_oof = stitched_oof.loc[~stitched_oof.index.duplicated(keep="last")]
                oof_stride = _infer_non_overlapping_stride(stitched_oof.index, default_stride=4)
                oof_pnl = simulate_stake_pnl(stitched_oof, stride=oof_stride)
                for k in pnl_scalar_keys:
                    pnl_cols[k].append(round(float(oof_pnl[k]), 2))
            else:
                for k in pnl_scalar_keys:
                    pnl_cols[k].append(np.nan)
        for out_col, vals in agg_cols.items():
            df[out_col] = vals
        for k, vals in pnl_cols.items():
            df[k] = vals
    _CV_FOLD_METRIC_LOG.clear()

    param_prefixed = [col for col in df.columns if col.startswith("param_")]
    if "params" in df.columns:
        params_df = pd.DataFrame(list(df["params"])).rename(
            columns=lambda col: col[len("estimator__"):] if str(col).startswith("estimator__") else str(col)
        )
        drop_cols = ["params"] + param_prefixed
        df = pd.concat([df.drop(columns=drop_cols, errors="ignore"), params_df], axis=1)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.insert(0, "candidate_id", np.arange(len(df), dtype=int))
    if "rank_test_score" in df.columns:
        df = df.sort_values("rank_test_score", kind="stable").reset_index(drop=True)
    return df


def compute_era_error_envelope(
    y_true: Any,
    y_pred: Any,
    eras: Any,
) -> pd.DataFrame:
    """Compute per-era data-point counts and directional forecast error shares (`over_share`, `under_share`) for envelope analysis."""
    targets = np.asarray(y_true, dtype=np.float64).ravel()
    preds = np.asarray(y_pred, dtype=np.float64).ravel()
    era_arr = np.asarray(eras, dtype=np.int32).ravel()
    if not (len(targets) == len(preds) == len(era_arr)):
        raise ValueError(
            f"Length mismatch: y_true={len(targets)}, y_pred={len(preds)}, eras={len(era_arr)}."
        )

    finite_mask = np.isfinite(targets) & np.isfinite(preds)
    n_excluded = int((~finite_mask).sum())
    if n_excluded > 0:
        print(f"Excluded {n_excluded} non-finite predictions from envelope analysis.")
    if not np.any(finite_mask):
        raise ValueError("No finite predictions available for era error envelope analysis.")

    residual = preds[finite_mask] - targets[finite_mask]
    frame = pd.DataFrame({
        "era": era_arr[finite_mask],
        "under": residual < 0.0,
        "over": residual > 0.0,
        "abs_error": np.abs(residual),
        "residual": residual,
    })
    measured = (
        frame.groupby("era", sort=True)
        .agg(
            data_points=("residual", "size"),
            under_share=("under", "mean"),
            over_share=("over", "mean"),
            mae=("abs_error", "mean"),
            mean_residual=("residual", "mean"),
        )
        .sort_index()
    )

    min_era = int(measured.index.min())
    max_era = int(measured.index.max())
    full_eras = pd.Index(np.arange(min_era, max_era + 1, dtype=np.int32), name="era")
    envelope_df = measured.reindex(full_eras)
    envelope_df["is_interpolated"] = envelope_df["data_points"].isna()
    for col in ("data_points", "under_share", "over_share", "mae", "mean_residual"):
        envelope_df[col] = envelope_df[col].interpolate(method="index", limit_area="inside")
    return envelope_df


def plot_era_error_envelope(
    y_true: Any,
    y_pred: Any,
    eras: Any,
    *,
    title: str = "Best Model",
    band_size: float = 0.15,
    is_best: bool = False,
    save_path: Optional[Any] = None,
) -> pd.DataFrame:
    """Plot the flight-deck directional error envelope across Numerai eras (x = eras, y = data points per era, ribbon = over/under forecast share)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    PURPLE = "#7446A5"
    BLUE = "#5C9ED6"
    ORANGE = "#EEA04B"

    envelope_df = compute_era_error_envelope(y_true, y_pred, eras)
    era_index = envelope_df.index.to_numpy(dtype=np.int32)
    centre = envelope_df["data_points"].to_numpy(dtype=np.float64)
    under_share = envelope_df["under_share"].to_numpy(dtype=np.float64)
    over_share = envelope_df["over_share"].to_numpy(dtype=np.float64)
    is_interpolated = envelope_df["is_interpolated"].to_numpy(dtype=bool)
    is_measured = ~is_interpolated

    value_span = float(np.nanmax(centre) - np.nanmin(centre))
    fallback_span = max(float(np.nanmax(centre)) * 0.25, 1.0)
    scale = float(band_size) * (value_span if value_span > 1e-9 else fallback_span)
    lower = centre - scale * under_share
    upper = centre + scale * over_share

    fig, ax = plt.subplots(figsize=(17, 6), facecolor="white")
    ax.set_facecolor("white")
    ax.fill_between(era_index, centre, upper, color=BLUE, alpha=0.45, linewidth=0)
    ax.fill_between(era_index, lower, centre, color=ORANGE, alpha=0.45, linewidth=0)
    ax.plot(era_index, upper, color=BLUE, linewidth=1.3)
    ax.plot(era_index, lower, color=ORANGE, linewidth=1.3)
    ax.plot(era_index, centre, color=PURPLE, linestyle=":", linewidth=2.3, zorder=3)

    ax.scatter(era_index[is_measured], centre[is_measured], color=PURPLE, s=22, zorder=4)
    if np.any(is_interpolated):
        ax.scatter(
            era_index[is_interpolated],
            centre[is_interpolated],
            facecolors="white",
            edgecolors=PURPLE,
            linewidths=1.5,
            s=45,
            zorder=5,
        )

    n_measured = int(is_measured.sum())
    min_era, max_era = int(era_index[0]), int(era_index[-1])
    title_prefix = "[BEST] " if is_best else ""
    title_color = PURPLE if is_best else "#111111"
    fig.suptitle(
        f"{title_prefix}{title}  \u00b7  {n_measured} validation eras  \u00b7  Era {min_era}\u2013{max_era}",
        x=0.06,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=title_color,
    )
    if is_best:
        fig.patch.set_edgecolor(PURPLE)
        fig.patch.set_linewidth(2.5)
    ax.set_xlabel("Validation Era")
    ax.set_ylabel("Data points per era")
    ax.set_xlim(min_era, max_era)
    if float(np.nanmax(upper)) >= 1000.0:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000.0:g}k"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(round(value)):,}"))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Era-block / annual (52-era) section dividers matching flight-deck quarterly markers
    era_span = max_era - min_era + 1
    block_step = 52 if era_span >= 104 else max(1, era_span // 4)
    block_starts = list(range(min_era, max_era + 1, block_step))
    if len(block_starts) >= 2:
        for block_idx, b_start in enumerate(block_starts, start=1):
            b_end = block_starts[block_idx] if block_idx < len(block_starts) else max_era + 1
            if b_end - b_start < max(2, block_step // 4):
                continue
            midpoint = 0.5 * (b_start + min(b_end - 1, max_era))
            block_label = f"Y{block_idx}" if block_step == 52 else f"B{block_idx}"
            ax.text(
                midpoint,
                1.015,
                block_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#666666",
                clip_on=False,
            )
            if block_idx > 1:
                ax.axvline(b_start, color="#BBBBBB", linewidth=0.8, linestyle="--", zorder=0)

    legend_handles = [
        Line2D([], [], color=PURPLE, linestyle=":", linewidth=2.3, label="Data points per era"),
        Patch(facecolor=BLUE, alpha=0.45, label="Overforecast share"),
        Patch(facecolor=ORANGE, alpha=0.45, label="Underforecast share"),
        Line2D([], [], color=PURPLE, marker="o", linestyle="None", markersize=5, label="Evaluated era"),
    ]
    if np.any(is_interpolated):
        legend_handles.append(
            Line2D(
                [],
                [],
                color=PURPLE,
                marker="o",
                markerfacecolor="white",
                linestyle="None",
                markersize=6,
                label="Interpolated era",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=2,
        columnspacing=1.6,
        handletextpad=0.6,
        labelspacing=1.1,
        fontsize=10,
    )

    mean_over_pct = float(np.nanmean(over_share[is_measured]) * 100.0)
    mean_under_pct = float(np.nanmean(under_share[is_measured]) * 100.0)
    fig.text(
        0.06,
        0.035,
        f"Bands show the share of held-out predictions above/below target (mean overforecast {mean_over_pct:.1f}%, "
        f"underforecast {mean_under_pct:.1f}%), not error magnitude. Ribbon thickness scales to {int(round(band_size * 100))}% of era data-point range.",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.06, right=0.70, top=0.84, bottom=0.17)
    if save_path is not None:
        target_path = Path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, dpi=140)
    plt.show()
    return envelope_df


def compute_error_bin_waterfall(
    y_true: Any,
    y_pred: Any,
    eras: Any,
    *,
    dates: Optional[Any] = None,
    mode: Optional[str] = None,
    pct_bins: Optional[Tuple[float, float]] = None,
    zero_tols: Optional[Tuple[float, float]] = None,
    abs_bins: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Fully vectorized 5-bin error waterfall summary across Overall, Quarterly Avg, Seasonal Q1-Q4 Avg, and each Financial Quarter."""
    codes, residual, finite_mask = classify_error_bins(
        y_true, y_pred, mode=mode, pct_bins=pct_bins, zero_tols=zero_tols, abs_bins=abs_bins
    )
    era_arr = np.asarray(eras, dtype=np.int32).ravel()
    if len(era_arr) != len(codes):
        raise ValueError(f"Length mismatch: eras={len(era_arr)}, y_true={len(codes)}.")
    if not np.any(finite_mask):
        raise ValueError("No finite predictions available for error-bin waterfall analysis.")

    valid_codes = codes[finite_mask]
    valid_res = residual[finite_mask]
    valid_abs = np.abs(valid_res)
    valid_eras = era_arr[finite_mask]

    # Derive financial quarter keys vectorially (from `date` when available, else 13-era quarters)
    quarter_keys: Optional[np.ndarray] = None
    season_nums: Optional[np.ndarray] = None
    if dates is not None:
        dt_series = pd.to_datetime(pd.Series(np.asarray(dates).ravel()), errors="coerce")
        valid_dt = dt_series.loc[finite_mask]
        if bool(valid_dt.notna().all()):
            years = valid_dt.dt.year.to_numpy(dtype=np.int32)
            q_num = valid_dt.dt.quarter.to_numpy(dtype=np.int32)
            quarter_keys = years * 10 + q_num
            season_nums = q_num
            quarter_is_calendar = True
        else:
            quarter_is_calendar = False
    else:
        quarter_is_calendar = False

    if quarter_keys is None or season_nums is None:
        min_era = int(valid_eras.min())
        rel_era = valid_eras - min_era
        y_idx = (rel_era // 52) + 1
        q_num = ((rel_era % 52) // 13) + 1
        quarter_keys = y_idx * 10 + q_num
        season_nums = q_num

    unique_q_keys, q_inv = np.unique(quarter_keys, return_inverse=True)
    n_quarters = len(unique_q_keys)
    q_counts = np.bincount(q_inv, minlength=n_quarters).astype(np.int64)
    q_bin_counts = np.bincount(q_inv * 5 + valid_codes.astype(np.int64), minlength=n_quarters * 5).reshape(n_quarters, 5)
    q_abs_sums = np.bincount(q_inv, weights=valid_abs.astype(np.float64), minlength=n_quarters)

    # Era count per quarter via vectorized unique pairs
    era_q_pairs = np.unique(np.column_stack((q_inv, valid_eras)), axis=0)
    q_era_counts = np.bincount(era_q_pairs[:, 0], minlength=n_quarters)

    # Overall row (vectorized bincount)
    overall_n = int(len(valid_codes))
    overall_eras = int(len(np.unique(valid_eras)))
    overall_bins = np.bincount(valid_codes, minlength=5).astype(np.float64) * (100.0 / float(overall_n))
    overall_mae = float(np.mean(valid_abs))

    def _make_row(section: str, period: str, n_pts: int, n_eras_val: int, pcts_5: np.ndarray, mae_val: float) -> Dict[str, Any]:
        under_tot = float(pcts_5[0] + pcts_5[1])
        over_tot = float(pcts_5[3] + pcts_5[4])
        extreme_tot = float(pcts_5[0] + pcts_5[4])
        return {
            "section": section,
            "period": period,
            "eras": int(n_eras_val),
            "data_points": int(n_pts),
            "extreme_underforecast_pct": round(float(pcts_5[0]), 2),
            "underforecast_pct": round(float(pcts_5[1]), 2),
            "accurate_pct": round(float(pcts_5[2]), 2),
            "overforecast_pct": round(float(pcts_5[3]), 2),
            "extreme_overforecast_pct": round(float(pcts_5[4]), 2),
            "net_bias_pct": round(over_tot - under_tot, 2),
            "extreme_total_pct": round(extreme_tot, 2),
            "mae": round(float(mae_val), 4),
        }

    q_pcts = (q_bin_counts.astype(np.float64) / np.maximum(q_counts[:, None], 1)) * 100.0
    q_maes = q_abs_sums / np.maximum(q_counts, 1)

    rows: List[Dict[str, Any]] = [
        _make_row("Overall", "Overall (All Validation)", overall_n, overall_eras, overall_bins, overall_mae),
        _make_row(
            "Quarterly Avg",
            f"Quarterly Avg ({n_quarters} Qtrs)",
            int(round(float(np.mean(q_counts)))),
            int(round(float(np.mean(q_era_counts)))),
            np.mean(q_pcts, axis=0),
            float(np.mean(q_maes)),
        ),
    ]

    # Seasonal Q1..Q4 averages across years
    q_season_ids = unique_q_keys % 10
    for q_idx in (1, 2, 3, 4):
        s_mask = q_season_ids == q_idx
        if np.any(s_mask):
            rows.append(
                _make_row(
                    "Seasonal Avg",
                    f"Q{q_idx} Avg ({int(s_mask.sum())} yrs)",
                    int(round(float(np.mean(q_counts[s_mask])))),
                    int(round(float(np.mean(q_era_counts[s_mask])))),
                    np.mean(q_pcts[s_mask], axis=0),
                    float(np.mean(q_maes[s_mask])),
                )
            )

    # Individual chronological financial quarters
    for idx in range(n_quarters):
        qk = int(unique_q_keys[idx])
        yr, qt = qk // 10, qk % 10
        label = f"{yr}-Q{qt}" if quarter_is_calendar else f"Y{yr}-Q{qt}"
        rows.append(
            _make_row(
                "Financial Quarter",
                label,
                int(q_counts[idx]),
                int(q_era_counts[idx]),
                q_pcts[idx],
                float(q_maes[idx]),
            )
        )

    return pd.DataFrame(rows)


def compute_model_error_bin_comparison(
    results_dict: Mapping[str, Mapping[str, Any]],
    y_true: Any,
    *,
    best_variant_name: Optional[str] = None,
    mode: Optional[str] = None,
    pct_bins: Optional[Tuple[float, float]] = None,
    zero_tols: Optional[Tuple[float, float]] = None,
    abs_bins: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Vectorized side-by-side 5-bin error summary across all evaluated models/variants."""
    rows: List[Dict[str, Any]] = []
    for variant_name, res in results_dict.items():
        preds = res.get("validation_preds")
        if preds is None:
            continue
        codes, residual, finite_mask = classify_error_bins(
            y_true, preds, mode=mode, pct_bins=pct_bins, zero_tols=zero_tols, abs_bins=abs_bins
        )
        valid_codes = codes[finite_mask]
        n_valid = max(1, len(valid_codes))
        pcts = np.bincount(valid_codes, minlength=5).astype(np.float64) * (100.0 / float(n_valid))
        mae_val = float(np.mean(np.abs(residual[finite_mask]))) if np.any(finite_mask) else 0.0
        rows.append({
            "variant": f"[BEST] {variant_name}" if variant_name == best_variant_name else str(variant_name),
            "data_points": int(len(valid_codes)),
            "extreme_underforecast_pct": round(float(pcts[0]), 2),
            "underforecast_pct": round(float(pcts[1]), 2),
            "accurate_pct": round(float(pcts[2]), 2),
            "overforecast_pct": round(float(pcts[3]), 2),
            "extreme_overforecast_pct": round(float(pcts[4]), 2),
            "net_bias_pct": round(float((pcts[3] + pcts[4]) - (pcts[0] + pcts[1])), 2),
            "extreme_total_pct": round(float(pcts[0] + pcts[4]), 2),
            "mae": round(mae_val, 4),
        })
    return pd.DataFrame(rows)


def plot_error_bin_waterfall(
    waterfall_df: pd.DataFrame,
    *,
    title: str = "Best Model",
    save_path: Optional[Any] = None,
) -> None:
    """Plot a 2-panel flight-deck Error-Bin Waterfall (Overall step-down + Financial Quarter diverging stacked bars)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    PURPLE = "#7446A5"
    GREEN = "#10b981"
    BLUE = "#5C9ED6"
    DEEP_BLUE = "#1d4ed8"
    ORANGE = "#EEA04B"
    DEEP_ORANGE = "#c2410c"
    SLATE = "#64748b"

    overall_rows = waterfall_df.loc[waterfall_df["section"] == "Overall"]
    q_avg_rows = waterfall_df.loc[waterfall_df["section"] == "Quarterly Avg"]
    quarter_rows = waterfall_df.loc[waterfall_df["section"] == "Financial Quarter"].reset_index(drop=True)
    if overall_rows.empty:
        raise ValueError("waterfall_df must contain an 'Overall' row.")

    ov = overall_rows.reset_index(drop=True).loc[0]
    total_pts = int(ov["data_points"])

    fig, (ax_wf, ax_q) = plt.subplots(
        1, 2, figsize=(19, 6.2), facecolor="white", gridspec_kw={"width_ratios": [1.0, 1.55]}
    )
    for ax in (ax_wf, ax_q):
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Left Panel: Overall Error-Bin Waterfall (100% step-down decomposition) ---
    wf_labels = [
        "Total\n(100%)",
        "Accurate",
        "Under-\nforecast",
        "Extreme\nUnder",
        "Over-\nforecast",
        "Extreme\nOver",
    ]
    wf_vals = [
        100.0,
        float(ov["accurate_pct"]),
        float(ov["underforecast_pct"]),
        float(ov["extreme_underforecast_pct"]),
        float(ov["overforecast_pct"]),
        float(ov["extreme_overforecast_pct"]),
    ]
    wf_colors = [SLATE, GREEN, ORANGE, DEEP_ORANGE, BLUE, DEEP_BLUE]
    x_pos = np.arange(len(wf_labels))

    bottoms = [0.0]
    running = 100.0
    for val in wf_vals[1:]:
        running = max(0.0, running - val)
        bottoms.append(running)

    ax_wf.bar(x_pos, wf_vals, bottom=bottoms, color=wf_colors, width=0.58, alpha=0.88, edgecolor="white", linewidth=1.0)
    for idx in range(len(wf_labels) - 1):
        connector_y = bottoms[idx] if idx > 0 else 100.0
        if idx == 0:
            connector_y = 100.0
        ax_wf.plot(
            [x_pos[idx] + 0.29, x_pos[idx + 1] - 0.29],
            [connector_y, connector_y],
            color="#94a3b8",
            linestyle="--",
            linewidth=0.9,
        )

    for idx, (val, bot) in enumerate(zip(wf_vals, bottoms)):
        cnt = int(round(total_pts * val / 100.0))
        top_y = bot + val
        cnt_str = f"{cnt / 1e6:.2f}M" if cnt >= 1_000_000 else (f"{cnt / 1e3:.1f}k" if cnt >= 1000 else f"{cnt:,}")
        ax_wf.text(
            x_pos[idx],
            top_y + 1.5,
            f"{val:.1f}%\n({cnt_str})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
        )

    ax_wf.set_xticks(x_pos)
    ax_wf.set_xticklabels(wf_labels, fontsize=9)
    ax_wf.set_ylabel("Share of Validation Data Points (%)")
    ax_wf.set_ylim(0, 118)
    ax_wf.set_title("Overall Error-Bin Waterfall", fontsize=12.5, fontweight="bold", pad=10)

    # --- Right Panel: Financial Quarter Diverging Stacked Bars + Accurate % Trend ---
    if not quarter_rows.empty:
        q_x = np.arange(len(quarter_rows))
        q_labels = quarter_rows["period"].astype(str).tolist()
        over_v = quarter_rows["overforecast_pct"].to_numpy(dtype=np.float64)
        ext_over_v = quarter_rows["extreme_overforecast_pct"].to_numpy(dtype=np.float64)
        under_v = quarter_rows["underforecast_pct"].to_numpy(dtype=np.float64)
        ext_under_v = quarter_rows["extreme_underforecast_pct"].to_numpy(dtype=np.float64)
        acc_v = quarter_rows["accurate_pct"].to_numpy(dtype=np.float64)

        ax_q.bar(q_x, over_v, color=BLUE, alpha=0.82, width=0.68, label="Overforecast %")
        ax_q.bar(q_x, ext_over_v, bottom=over_v, color=DEEP_BLUE, alpha=0.88, width=0.68, label="Extreme Over %")
        ax_q.bar(q_x, -under_v, color=ORANGE, alpha=0.82, width=0.68, label="Underforecast %")
        ax_q.bar(q_x, -ext_under_v, bottom=-under_v, color=DEEP_ORANGE, alpha=0.88, width=0.68, label="Extreme Under %")
        ax_q.axhline(0.0, color="#475569", linewidth=1.0)

        ax_q.plot(q_x, acc_v, color=PURPLE, linestyle=":", marker="o", markersize=4.5, linewidth=2.2, zorder=5, label="Accurate %")
        if not q_avg_rows.empty:
            avg_acc = float(q_avg_rows.reset_index(drop=True).loc[0, "accurate_pct"])
            ax_q.axhline(avg_acc, color=PURPLE, linestyle="--", linewidth=1.1, alpha=0.65, label=f"Qtr Avg Accurate ({avg_acc:.1f}%)")

        step = max(1, len(q_labels) // 12)
        ax_q.set_xticks(q_x[::step])
        ax_q.set_xticklabels(q_labels[::step], rotation=40, ha="right", fontsize=8.5)
        ax_q.set_ylabel("Error Share by Direction (% Over ↑ / % Under ↓)")
        ax_q.set_title("Financial Quarter Error Breakdown", fontsize=12.5, fontweight="bold", pad=10)
        ax_q.legend(
            handles=[
                Patch(facecolor=DEEP_BLUE, alpha=0.88, label="Extreme Over"),
                Patch(facecolor=BLUE, alpha=0.82, label="Overforecast"),
                Patch(facecolor=ORANGE, alpha=0.82, label="Underforecast"),
                Patch(facecolor=DEEP_ORANGE, alpha=0.88, label="Extreme Under"),
                Line2D([], [], color=PURPLE, linestyle=":", marker="o", markersize=4.5, linewidth=2.2, label="Accurate %"),
            ],
            loc="upper right",
            frameon=False,
            ncol=3,
            fontsize=8.5,
        )

    cfg_mode = _ERROR_BIN_CONFIG["mode"]
    if cfg_mode == "pct":
        p_acc, p_ext = _ERROR_BIN_CONFIG["pct_bins"]
        z_acc, z_ext = _ERROR_BIN_CONFIG["zero_tols"]
        rule_str = f"mode='pct' (±{p_acc:g}% / ±{p_ext:g}%, zero-target ±{z_acc:g} / ±{z_ext:g})"
    else:
        a_acc, a_ext = _ERROR_BIN_CONFIG["abs_bins"]
        rule_str = f"mode='abs' (±{a_acc:g} / ±{a_ext:g})"

    fig.suptitle(
        f"[BEST] {title}  \u00b7  Error-Bin Waterfall & Quarterly Breakdown",
        x=0.04,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=PURPLE,
    )
    fig.text(
        0.04,
        0.02,
        f"Binning rule: {rule_str} | Total validation points: {total_pts:,} | "
        f"Overall Accurate: {ov['accurate_pct']:.1f}% | Net Bias (Over% - Under%): {ov['net_bias_pct']:+.1f}% | Extreme Total: {ov['extreme_total_pct']:.1f}%",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.84, bottom=0.18, wspace=0.20)
    if save_path is not None:
        target_path = Path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, dpi=140)
    plt.show()
