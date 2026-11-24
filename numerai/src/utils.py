"""utils.py — Shared NumeraiExtractions, per-ticker time-series extraction, diagnostics Parquet export, and CORR sanity-checked evaluation suite."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numerai_tools.scoring import (
    correlation_contribution,
    numerai_corr,
    sharpe_ratio as numerai_sharpe_ratio,
    spearman_correlation as numerai_spearman_correlation,
)

try:
    from numereval import numereval as _numereval_mod
    NUMEREVAL_AVAILABLE = True
except ImportError:
    _numereval_mod = None
    NUMEREVAL_AVAILABLE = False
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
    """Calculate the Numerai Sharpe ratio across per-era correlations via `numerai_tools.scoring.sharpe_ratio` (with zero-variance guard)."""
    clean = pd.Series(corrs, dtype=np.float64).dropna()
    if len(clean) == 0 or float(np.std(clean)) <= 1e-12:
        return 0.0
    val = float(numerai_sharpe_ratio(clean))
    return val if np.isfinite(val) else 0.0


def compute_per_era_spearman_corr(y_true: Any, y_pred: Any, eras: Optional[Any] = None) -> pd.Series:
    """Compute per-era Spearman rank correlation via `numerai_tools.scoring.spearman_correlation`."""
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
            corr_val = float(numerai_spearman_correlation(valid_df["target"], valid_df["prediction"]))
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
        corr_drawdown = float(np.round((running_peak - cumulative_corr).max(), 4))
    else:
        corr_drawdown = 0.0

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
        print(f"Cumulative CORR Drawdown (CORR pts): {corr_drawdown:.4f}")
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
        "corr_drawdown": corr_drawdown,
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
VALID_SCORE_VERSIONS = ("v0_default", "v1_skepta", "v2_skepta_mmc")


def build_experiment_setup(
    *,
    target_col: str = "target",
    feature_set: str = "small",
    era_stride: int = 4,
    model_tag: str = "baselines_ae",
    baseline_mode: Optional[str] = None,
    search_mode: str = "bayes",
    score_version: str = "v0_default",
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
    clean_score_version = str(score_version or "v0_default").strip().lower()
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
    if clean_score_version not in VALID_SCORE_VERSIONS:
        raise ValueError(f"score_version must be one of {VALID_SCORE_VERSIONS}, got {score_version!r}")
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
        "score_version": clean_score_version,
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


def compute_versioned_candidate_score(
    era_corrs: Union[pd.Series, np.ndarray, Sequence[float]],
    era_mmcs: Optional[Union[pd.Series, np.ndarray, Sequence[float]]] = None,
    score_version: str = "v0_default",
) -> Tuple[float, str, bool]:
    """Compute a versioned CV selection score from per-era CORR and optional per-era MMC.

    Versions:
    - `v0_default`: `mean(era_corrs)` (legacy behavior)
    - `v1_skepta`: `mean(era_corrs) - 0.5 * std(era_corrs)` (rewards mean + stability)
    - `v2_skepta_mmc`: `mean(era_corrs) + 2.0 * mean(era_mmcs) - 0.5 * std(era_corrs)`
      Only used when real `era_mmcs` is available; if MMC is missing, falls back to `v1_skepta`
      and sets `mmc_missing_fallback=True`.

    Returns `(score_value, effective_score_version, mmc_missing_fallback)`.
    """
    req_version = str(score_version or "v0_default").strip().lower()
    if req_version not in VALID_SCORE_VERSIONS:
        raise ValueError(f"Unsupported score_version {score_version!r}; expected one of {VALID_SCORE_VERSIONS}.")

    corr_s = pd.Series(era_corrs, dtype=np.float64).dropna()
    if len(corr_s) == 0:
        eff_ver = "v1_skepta" if req_version == "v2_skepta_mmc" and era_mmcs is None else req_version
        return 0.0, eff_ver, bool(req_version == "v2_skepta_mmc" and era_mmcs is None)

    mean_corr = float(corr_s.mean())
    if not np.isfinite(mean_corr):
        mean_corr = 0.0
    std_corr = float(corr_s.std(ddof=1)) if len(corr_s) >= 2 else 0.0
    if not np.isfinite(std_corr):
        std_corr = 0.0

    if req_version == "v0_default":
        return mean_corr, "v0_default", False

    if req_version == "v1_skepta":
        return float(mean_corr - 0.5 * std_corr), "v1_skepta", False

    # v2_skepta_mmc
    mmc_s = pd.Series(era_mmcs, dtype=np.float64).dropna() if era_mmcs is not None else pd.Series(dtype=np.float64)
    if len(mmc_s) == 0:
        print("  [Score Warning] score_version='v2_skepta_mmc' requested but MMC is missing; falling back to 'v1_skepta'.")
        return float(mean_corr - 0.5 * std_corr), "v1_skepta", True

    mean_mmc = float(mmc_s.mean())
    if not np.isfinite(mean_mmc):
        print("  [Score Warning] score_version='v2_skepta_mmc' requested but MMC is non-finite; falling back to 'v1_skepta'.")
        return float(mean_corr - 0.5 * std_corr), "v1_skepta", True

    return float(mean_corr + 2.0 * mean_mmc - 0.5 * std_corr), "v2_skepta_mmc", False


def candidate_score(
    estimator: Any,
    X: pd.DataFrame,
    y: Any,
    *,
    score_version: str = "v0_default",
    era_mmcs: Optional[pd.Series] = None,
) -> float:
    """Versioned per-era scorer for `GridSearchCV` / `BayesSearchCV`, logging extended fold metrics for `build_quick_results_df`.

    Defaults to `score_version='v0_default'` (`mean(era_corrs)`) for backwards compatibility.
    Supports `'v1_skepta'` (`mean_corr - 0.5 * std_corr`) and `'v2_skepta_mmc'`
    (`mean_corr + 2 * mean_mmc - 0.5 * std_corr`, falling back to `'v1_skepta'` when MMC is missing).
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

    fold_mmcs: Optional[pd.Series] = None
    if era_mmcs is not None and len(era_mmcs) > 0:
        fold_mmcs = pd.Series(era_mmcs, dtype=np.float64).reindex(era_corrs.index).dropna()
        if len(fold_mmcs) == 0:
            fold_mmcs = None

    score_val, effective_version, mmc_fallback = compute_versioned_candidate_score(
        era_corrs,
        era_mmcs=fold_mmcs,
        score_version=score_version,
    )

    mean_corr = float(era_corrs.mean()) if len(era_corrs) else 0.0
    mean_corr = mean_corr if np.isfinite(mean_corr) else 0.0
    std_corr = float(era_corrs.std(ddof=1)) if len(era_corrs) >= 2 else 0.0
    std_corr = std_corr if np.isfinite(std_corr) else 0.0
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
        corr_dd = float((running_peak - cumulative_corr).max())
    else:
        corr_dd = 0.0

    bin_shares = compute_error_bin_shares(targets, preds)

    _CV_FOLD_METRIC_LOG.append({
        "score_version": effective_version,
        "requested_score_version": str(score_version or "v0_default").strip().lower(),
        "mmc_missing_fallback": bool(mmc_fallback),
        "score_value": float(score_val),
        "numerai_corr": mean_corr,
        "era_corr_std": std_corr,
        "positive_era_pct": pos_era_pct,
        "directional_accuracy_pct": dir_acc_pct,
        "sharpe": sharpe_val,
        "spearman": spearman_mean,
        "mae": mae_val,
        "forecast_bias": bias_val,
        "corr_drawdown": corr_dd,
        "era_corrs": era_corrs,
        **bin_shares,
    })

    print(
        f"  [Fold Score | {effective_version}] Score: {score_val:.4f} | "
        f"Numerai CORR: {mean_corr * 100.0:+.2f}% ({mean_corr:.4f}, std={std_corr:.4f}) | "
        f"Sharpe: {sharpe_val:.2f} | CORR DD: {corr_dd:.4f} | Positive Eras: {pos_era_pct:.1f}% | "
        f"Dir Acc: {dir_acc_pct:.1f}% | MAE: {mae_val:.4f}"
    )
    return float(score_val)


def make_candidate_scorer(
    score_version: str = "v0_default",
    *,
    era_mmcs: Optional[pd.Series] = None,
) -> Any:
    """Create a picklable `GridSearchCV`/`BayesSearchCV` scorer bound to `score_version`."""
    clean_ver = str(score_version or "v0_default").strip().lower()
    if clean_ver not in VALID_SCORE_VERSIONS:
        raise ValueError(f"Unsupported score_version {score_version!r}; expected one of {VALID_SCORE_VERSIONS}.")
    if clean_ver == "v0_default" and era_mmcs is None:
        return candidate_score
    return partial(candidate_score, score_version=clean_ver, era_mmcs=era_mmcs)


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
    *,
    benchmark_path: Optional[Union[str, Path]] = None,
    is_benchmark_proxy: bool = False,
) -> Optional[pd.Series]:
    """Compute per-era MMC (against true `meta_model.parquet`) or BMC proxy (against v6 benchmark/example preds).

    Supports joining by `id` (classic v4/v5) OR by `('date', 'numerai_ticker')` (v6.0 Unity Parquets, which
    use `numerai_ticker` + `date` + `signal` without an `id` column). Returns `None` gracefully when no
    reference file is available. Never relabels a BMC proxy series as true MMC.
    """
    active_path_raw = meta_model_path if meta_model_path is not None else benchmark_path
    use_proxy = bool(is_benchmark_proxy or (meta_model_path is None and benchmark_path is not None))
    if active_path_raw is None:
        return None
    meta_path = Path(active_path_raw)
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

    meta_ignore = {"id", era_col, "date", "data_type", "numerai_ticker"}
    numeric_candidates = [
        str(c)
        for c in meta_df.columns
        if str(c) not in meta_ignore and pd.api.types.is_numeric_dtype(meta_df[c])
    ]
    if "numerai_meta_model" in meta_df.columns:
        ref_series = pd.to_numeric(meta_df["numerai_meta_model"], errors="coerce")
    elif "signal" in meta_df.columns:
        ref_series = pd.to_numeric(meta_df["signal"], errors="coerce")
    elif "prediction" in meta_df.columns:
        ref_series = pd.to_numeric(meta_df["prediction"], errors="coerce")
    elif len(numeric_candidates) == 1:
        ref_series = pd.to_numeric(meta_df[numeric_candidates[0]], errors="coerce")
    elif len(numeric_candidates) > 1:
        # Multi-column benchmark Parquet (e.g. validation_benchmark_models.parquet): equal-weight mean across benchmarks
        ref_series = meta_df[numeric_candidates].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        return None

    ref_col_name = "_ref_model_signal_"
    meta_join_cols = [c for c in ("id", "date", "numerai_ticker") if c in meta_df.columns]
    meta_slim = meta_df[meta_join_cols].copy()
    meta_slim[ref_col_name] = ref_series.to_numpy(dtype=np.float64)

    work_cols = [era_col, target_col] + [c for c in ("id", "date", "numerai_ticker") if c in validation_df.columns]
    work_df = validation_df[work_cols].copy()
    work_df["prediction"] = pred_arr

    if "id" in work_df.columns and "id" in meta_slim.columns:
        merged = work_df.merge(meta_slim[["id", ref_col_name]], on="id", how="inner")
    elif "id" in work_df.columns and meta_df.index.name == "id":
        meta_idx_df = pd.DataFrame({ref_col_name: ref_series.to_numpy(dtype=np.float64)}, index=meta_df.index)
        merged = work_df.merge(meta_idx_df, left_on="id", right_index=True, how="inner")
    elif (
        "date" in work_df.columns
        and "numerai_ticker" in work_df.columns
        and "date" in meta_slim.columns
        and "numerai_ticker" in meta_slim.columns
    ):
        work_df["_join_date_"] = pd.to_datetime(work_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        work_df["_join_ticker_"] = work_df["numerai_ticker"].astype(str)
        meta_slim["_join_date_"] = pd.to_datetime(meta_slim["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        meta_slim["_join_ticker_"] = meta_slim["numerai_ticker"].astype(str)
        meta_dedup = meta_slim[["_join_date_", "_join_ticker_", ref_col_name]].drop_duplicates(
            subset=["_join_date_", "_join_ticker_"], keep="last"
        )
        merged = work_df.merge(meta_dedup, on=["_join_date_", "_join_ticker_"], how="inner")
    elif ("date" in meta_slim.columns and "numerai_ticker" in meta_slim.columns) or ("id" in meta_slim.columns):
        # Refuse silent RangeIndex alignment across mismatched schemas (e.g. v5.0 validation_df vs v6.0 benchmark Parquet)
        return None
    else:
        meta_aligned = ref_series.reindex(work_df.index)
        merged = work_df.copy()
        merged[ref_col_name] = meta_aligned.to_numpy(dtype=np.float64)

    merged = merged.dropna(subset=["prediction", ref_col_name, target_col])
    if merged.empty:
        return None

    def _safe_era_contrib(grp: pd.DataFrame) -> float:
        p = grp["prediction"].to_numpy(dtype=np.float64, copy=False)
        m = grp[ref_col_name].to_numpy(dtype=np.float64, copy=False)
        t = grp[target_col].to_numpy(dtype=np.float64, copy=False)
        if len(t) < 2 or _is_constant_1d(p) or _is_constant_1d(m) or _is_constant_1d(t):
            return 0.0
        val = float(
            correlation_contribution(
                grp[["prediction"]],
                grp[ref_col_name].rename("numerai_meta_model"),
                grp[target_col],
            ).to_numpy(dtype=np.float64)[0]
        )
        return val if np.isfinite(val) else 0.0

    contrib_series = merged.groupby(era_col, sort=True).apply(_safe_era_contrib, include_groups=False)
    return contrib_series.astype(np.float64).rename("bmc_proxy" if use_proxy else "mmc")


R1393_START_ERA = 1393
R1399_START_ERA = 1399
CLASSIC_CONCURRENT_ROUNDS = 64
CLASSIC_SETTLEMENT_ROUNDS = 12


def _parse_era_numbers(era_index: pd.Index) -> np.ndarray:
    """Parse era labels (integers or `'era1234'` strings) into float64 era numbers (NaN if unparseable)."""
    raw = pd.Index(era_index)
    numeric = pd.to_numeric(raw, errors="coerce")
    if not numeric.isna().any():
        return numeric.to_numpy(dtype=np.float64)
    cleaned = [str(v).strip().lower().removeprefix("era") for v in raw]
    return pd.to_numeric(pd.Index(cleaned), errors="coerce").to_numpy(dtype=np.float64)


def compute_r1399_churn_retention(
    mean_neutral_churn: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Official Numerai R1399+ neutral-churn retained payout fraction.

    Formula: `min(1.0, 2.0 / (1.0 + exp(10.0 * (mean_neutral_churn - 0.1))))`.
    Applied only to positive payouts after multipliers and clipping, starting at round 1399.
    """
    if isinstance(mean_neutral_churn, pd.Series):
        vals = pd.to_numeric(mean_neutral_churn, errors="coerce").to_numpy(dtype=np.float64)
        clipped_exp = np.exp(np.clip(10.0 * (vals - 0.1), -60.0, 60.0))
        ret = np.minimum(1.0, 2.0 / (1.0 + clipped_exp))
        ret = np.where(np.isfinite(vals), ret, 1.0)
        return pd.Series(ret, index=mean_neutral_churn.index, dtype=np.float64)
    if isinstance(mean_neutral_churn, np.ndarray):
        vals = np.asarray(mean_neutral_churn, dtype=np.float64)
        clipped_exp = np.exp(np.clip(10.0 * (vals - 0.1), -60.0, 60.0))
        ret = np.minimum(1.0, 2.0 / (1.0 + clipped_exp))
        return np.where(np.isfinite(vals), ret, 1.0)
    val = float(mean_neutral_churn)
    if not np.isfinite(val):
        return 1.0
    return float(min(1.0, 2.0 / (1.0 + np.exp(np.clip(10.0 * (val - 0.1), -60.0, 60.0)))))


def _compute_clipped_era_returns(
    per_era_corr: pd.Series,
    *,
    per_era_mmc: Optional[pd.Series] = None,
    per_era_ncorr: Optional[pd.Series] = None,
    per_era_nmmc: Optional[pd.Series] = None,
    mmc_is_proxy: bool = False,
    corr_weight: Optional[float] = None,
    mmc_weight: Optional[float] = None,
    payout_factor: Union[float, pd.Series] = 1.0,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    neutral_churn: Optional[Union[float, pd.Series]] = None,
    churn_penalty: Optional[Union[float, pd.Series]] = None,
    payout_rule: str = "auto",
    stride: int = 1,
    corr_multiplier: Optional[float] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Compute chronologically sorted per-round Numerai Classic payout returns using era-accurate rules.

    Official rules (`payout_rule='auto'`):
    - Pre-round 1393 (`era < 1393`): `clip(3 * CORR60 + 15 * MMC60, -1.0, +1.0)`.
    - Round 1393+ (`era >= 1393`): `clip(payout_factor * (8 * NCORR + 32 * NMMC), -1.0, +2.0)` with `payout_factor=1.0`.
    - Round 1399+ (`era >= 1399`): if `neutral_churn` is provided, positive payouts are multiplied by
      `min(1, 2 / (1 + exp(10 * (mean_neutral_churn - 0.1))))` after multipliers and clipping; burns stay unchanged.
      Rounds 1393–1398 have no churn penalty.

    Honesty rule: never pretends CORR is NCORR, never pretends missing MMC is zero, and never labels
    BMC proxy (computed against benchmark/example predictions) as true MMC.
    """
    use_diagnostics = per_era_ncorr is not None and len(per_era_ncorr) > 0
    primary_corr = per_era_ncorr if use_diagnostics else per_era_corr
    primary_mmc = per_era_nmmc if use_diagnostics else per_era_mmc
    is_proxy = bool(mmc_is_proxy) or (
        isinstance(primary_mmc, pd.Series) and str(primary_mmc.name or "").strip().lower() == "bmc_proxy"
    )

    corr_series = pd.Series(primary_corr, dtype=np.float64).dropna().sort_index()
    step = max(1, int(stride))
    if step > 1 and len(corr_series) > 0:
        corr_series = corr_series.loc[corr_series.index[::step]]
    if len(corr_series) == 0:
        return pd.Series(dtype=np.float64), {
            "has_mmc": False,
            "mmc_is_proxy": False,
            "mmc_status": "missing",
            "basis_short": "CORR only",
            "score_basis": "CORR only (no True MMC / NMMC)",
            "payout_rule_summary": "none",
            "churn_status": "missing (not applied)",
            "used_diagnostics_ncorr": bool(use_diagnostics),
        }

    idx = corr_series.index
    n_eras = len(idx)
    era_nums = _parse_era_numbers(idx)
    clean_rule = str(payout_rule or "auto").strip().lower()
    if clean_rule not in ("auto", "pre_1393", "r1393"):
        raise ValueError(f"payout_rule must be 'auto', 'pre_1393', or 'r1393', got {payout_rule!r}")

    if clean_rule == "pre_1393":
        is_r1393_mask = np.zeros(n_eras, dtype=bool)
    elif clean_rule == "r1393":
        is_r1393_mask = np.ones(n_eras, dtype=bool)
    else:
        is_r1393_mask = np.isfinite(era_nums) & (era_nums >= float(R1393_START_ERA))

    is_r1399_mask = np.isfinite(era_nums) & (era_nums >= float(R1399_START_ERA))

    # Multipliers and official clips per era
    eff_corr_override = float(corr_multiplier) if corr_multiplier is not None else (
        float(corr_weight) if corr_weight is not None else None
    )
    eff_mmc_override = float(mmc_weight) if mmc_weight is not None else None

    corr_w_arr = (
        np.full(n_eras, eff_corr_override, dtype=np.float64)
        if eff_corr_override is not None
        else np.where(is_r1393_mask, 8.0, 3.0)
    )
    mmc_w_arr = (
        np.full(n_eras, eff_mmc_override, dtype=np.float64)
        if eff_mmc_override is not None
        else np.where(is_r1393_mask, 32.0, 15.0)
    )
    c_min_arr = (
        np.full(n_eras, float(clip_min), dtype=np.float64)
        if clip_min is not None
        else np.full(n_eras, -1.0, dtype=np.float64)
    )
    c_max_arr = (
        np.full(n_eras, float(clip_max), dtype=np.float64)
        if clip_max is not None
        else np.where(is_r1393_mask, 2.0, 1.0)
    )

    has_mmc = primary_mmc is not None and len(pd.Series(primary_mmc).dropna()) > 0
    corr_arr = corr_series.to_numpy(dtype=np.float64)
    if has_mmc:
        mmc_aligned = (
            pd.Series(primary_mmc, dtype=np.float64)
            .dropna()
            .sort_index()
            .reindex(idx)
        )
        missing_mmc_eras = int(mmc_aligned.isna().sum())
        mmc_arr = mmc_aligned.fillna(0.0).to_numpy(dtype=np.float64)
        raw_score = corr_w_arr * corr_arr + mmc_w_arr * mmc_arr
        if is_proxy:
            mmc_status = (
                "bmc_proxy (benchmark, NOT true MMC)"
                if missing_mmc_eras == 0
                else f"bmc_proxy partial ({missing_mmc_eras}/{n_eras} missing)"
            )
        else:
            mmc_status = "present" if missing_mmc_eras == 0 else f"partial ({missing_mmc_eras}/{n_eras} missing)"
    else:
        raw_score = corr_w_arr * corr_arr
        mmc_status = "missing"

    if isinstance(payout_factor, pd.Series):
        pf_arr = (
            pd.Series(payout_factor, dtype=np.float64)
            .dropna()
            .sort_index()
            .reindex(idx)
            .ffill()
            .bfill()
            .fillna(1.0)
            .to_numpy(dtype=np.float64)
        )
    else:
        pf_arr = np.full(n_eras, float(payout_factor), dtype=np.float64)

    # Apply payout_factor and official per-era clips
    scaled_score = raw_score * pf_arr
    clipped_arr = np.clip(scaled_score, c_min_arr, c_max_arr)

    # R1399+ neutral churn penalty: positive payouts only, after multipliers and clipping; R1393-1398 unaffected
    churn_input = neutral_churn if neutral_churn is not None else churn_penalty
    if churn_input is not None:
        if np.any(is_r1399_mask):
            if isinstance(churn_input, pd.Series):
                ch_aligned = (
                    pd.Series(churn_input, dtype=np.float64)
                    .dropna()
                    .sort_index()
                    .reindex(idx)
                )
                ch_vals = ch_aligned.to_numpy(dtype=np.float64)
                retention = compute_r1399_churn_retention(ch_vals)
            else:
                retention = np.full(n_eras, float(compute_r1399_churn_retention(float(churn_input))), dtype=np.float64)
            apply_churn = is_r1399_mask & (clipped_arr > 0.0)
            clipped_arr = np.where(apply_churn, clipped_arr * retention, clipped_arr)
            churn_status = f"applied on {int(is_r1399_mask.sum())} R1399+ era(s) (positive payouts only)"
        else:
            churn_status = "not applicable (all eras < 1399)"
    else:
        churn_status = "missing (not applied)" if np.any(is_r1399_mask) else "not applicable (all eras < 1399)"

    # Honest score basis and payout rule summary (never label BMC as MMC)
    if use_diagnostics:
        if has_mmc and not is_proxy:
            basis_short = "True MMC / NMMC"
            score_basis = "True MMC / NMMC (Diagnostics NCORR + NMMC)"
        elif has_mmc and is_proxy:
            basis_short = "BMC proxy"
            score_basis = "BMC proxy (benchmark, NOT true MMC)"
        else:
            basis_short = "CORR only"
            score_basis = "CORR only (Diagnostics NCORR only — missing True NMMC)"
    else:
        if has_mmc and is_proxy:
            basis_short = "BMC proxy"
            score_basis = "BMC proxy (benchmark, NOT true MMC)"
        elif has_mmc:
            basis_short = "True MMC / NMMC"
            score_basis = "True MMC / NMMC (CORR + MMC)"
        else:
            basis_short = "CORR only"
            score_basis = "CORR only (no True MMC / NMMC)"

    contrib_tag = "BMC_proxy" if is_proxy else ("NMMC" if use_diagnostics else "MMC")
    n_r1393 = int(is_r1393_mask.sum())
    n_pre = n_eras - n_r1393
    if n_r1393 == 0:
        payout_rule_summary = (
            f"Pre-1393: {corr_w_arr[0]:g}xCORR + {mmc_w_arr[0]:g}x{contrib_tag}, clip [{c_min_arr[0]*100:.0f}%, +{c_max_arr[0]*100:.0f}%]"
            if has_mmc
            else f"Pre-1393 (CORR only): {corr_w_arr[0]:g}xCORR, clip [{c_min_arr[0]*100:.0f}%, +{c_max_arr[0]*100:.0f}%]"
        )
    elif n_pre == 0:
        payout_rule_summary = (
            f"R1393+: {corr_w_arr[0]:g}x{'NCORR' if use_diagnostics else 'CORR'} + {mmc_w_arr[0]:g}x{contrib_tag}, clip [{c_min_arr[0]*100:.0f}%, +{c_max_arr[0]*100:.0f}%]"
            if has_mmc
            else f"R1393+ (CORR only): {corr_w_arr[0]:g}x{'NCORR' if use_diagnostics else 'CORR'}, clip [{c_min_arr[0]*100:.0f}%, +{c_max_arr[0]*100:.0f}%]"
        )
    else:
        payout_rule_summary = (
            f"Era-aware ({n_pre} pre-1393 [3x/15x, -100%..+100%] + {n_r1393} R1393+ [8x/32x, -100%..+200%])"
        )

    meta = {
        "has_mmc": bool(has_mmc),
        "mmc_is_proxy": bool(has_mmc and is_proxy),
        "mmc_status": mmc_status,
        "basis_short": basis_short,
        "score_basis": score_basis,
        "payout_rule_summary": payout_rule_summary,
        "churn_status": churn_status,
        "used_diagnostics_ncorr": bool(use_diagnostics),
        "n_pre_1393_eras": int(n_pre),
        "n_r1393_eras": int(n_r1393),
        "n_r1399_eras": int(is_r1399_mask.sum()),
    }
    return pd.Series(clipped_arr, index=idx, dtype=np.float64), meta


def _simulate_single_wallet_path(
    ret_arr: np.ndarray,
    start_capital: float,
    staking_mode: str = "compound",
    *,
    concurrent_rounds: int = CLASSIC_CONCURRENT_ROUNDS,
    settlement_rounds: int = CLASSIC_SETTLEMENT_ROUNDS,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, bool]:
    """Simulate ONE continuous wallet with overlapping rounds (no resets, no margin, equity >= 0).

    Per-round stake = `TVL / R` (`R = 64` for Classic). Each opened round stays locked for
    `settlement_rounds` weeks until it settles; locked capital cannot be reused. Once equity
    hits zero, the wallet stays at zero permanently.

    Returns `(equity_curve, round_pnls, money_drawdown_pct, max_drawdown_val, worst_round, wiped_out)`.
    """
    init_cap = max(0.0, float(start_capital))
    n = len(ret_arr)
    if n == 0 or init_cap <= 0.0:
        empty = np.zeros(n, dtype=np.float64)
        return empty, empty, 0.0, 0.0, 0.0, bool(init_cap <= 0.0)

    R = max(1, int(concurrent_rounds))
    settle_lag = max(1, int(settlement_rounds))
    clean_mode = str(staking_mode or "compound").strip().lower()
    if clean_mode not in ("compound", "constant"):
        raise ValueError(f"staking_mode must be 'compound' or 'constant', got {staking_mode!r}")

    free_cash = init_cap
    # Each entry: (settle_step, open_step, stake_amount, round_return)
    open_positions: List[Tuple[int, int, float, float]] = []
    equity_path = np.zeros(n, dtype=np.float64)
    round_pnls = np.zeros(n, dtype=np.float64)
    wiped_out = False

    for t in range(n):
        # 1. Settle any positions maturing at week t
        if open_positions:
            still_open: List[Tuple[int, int, float, float]] = []
            for s_step, o_step, pos_stake, pos_ret in open_positions:
                if s_step <= t:
                    payout = max(0.0, pos_stake * (1.0 + pos_ret))
                    free_cash += payout
                    round_pnls[o_step] = payout - pos_stake
                else:
                    still_open.append((s_step, o_step, pos_stake, pos_ret))
            open_positions = still_open

        locked_cap = sum(pos[2] for pos in open_positions)
        curr_eq = max(0.0, free_cash + locked_cap)
        if curr_eq <= 1e-12:
            wiped_out = True
            free_cash = 0.0
            open_positions.clear()
            curr_eq = 0.0

        # 2. Open new position for week t using only unlocked free_cash
        if not wiped_out:
            target_stake = (init_cap / float(R)) if clean_mode == "constant" else (curr_eq / float(R))
            actual_stake = min(max(0.0, target_stake), max(0.0, free_cash))
            if actual_stake > 1e-15:
                free_cash -= actual_stake
                open_positions.append((t + settle_lag, t, actual_stake, float(ret_arr[t])))

        locked_after = sum(pos[2] for pos in open_positions)
        equity_path[t] = 0.0 if wiped_out else max(0.0, free_cash + locked_after)

    # 3. Settle all remaining open positions chronologically so final balance reflects all opened rounds
    if not wiped_out and open_positions:
        open_positions.sort(key=lambda item: item[0])
        for _, o_step, pos_stake, pos_ret in open_positions:
            payout = max(0.0, pos_stake * (1.0 + pos_ret))
            free_cash += payout
            round_pnls[o_step] = payout - pos_stake
        open_positions.clear()
        final_eq = max(0.0, free_cash)
        if final_eq <= 1e-12:
            wiped_out = True
            final_eq = 0.0
        equity_path[-1] = final_eq
    elif wiped_out:
        equity_path[-1] = 0.0

    # Enforce strict non-negativity and permanent zero after ruin
    equity_path = np.maximum(0.0, equity_path)
    ruin_mask = np.maximum.accumulate(equity_path <= 1e-12)
    equity_path = np.where(ruin_mask, 0.0, equity_path)
    wiped_out = bool(wiped_out or bool(ruin_mask[-1]))

    # Money drawdown: (lowest - peak) / peak * 100, always in [-100, 0]
    running_peak = np.maximum.accumulate(np.maximum(init_cap, equity_path))
    dd_val = equity_path - running_peak
    dd_pct = np.where(running_peak > 0.0, (dd_val / running_peak) * 100.0, 0.0)
    dd_pct = np.clip(dd_pct, -100.0, 0.0)

    money_dd_pct = float(dd_pct.min())
    max_dd_val = float(dd_val.min())
    worst_round = float(round_pnls.min())
    return equity_path, round_pnls, money_dd_pct, max_dd_val, worst_round, wiped_out


def _extract_path_drawdown_markers(eq_path: np.ndarray, start_cap: float) -> Dict[str, Any]:
    """Extract peak, worst-trough, drawdown %, and final balance markers for a single simulated wallet path."""
    arr = np.asarray(eq_path, dtype=np.float64).ravel()
    if len(arr) == 0:
        return {
            "equity_path": arr,
            "peak_idx": 0,
            "peak_val": float(start_cap),
            "trough_idx": 0,
            "trough_val": float(start_cap),
            "max_drawdown_pct": 0.0,
            "final_val": float(start_cap),
        }
    running_peak = np.maximum.accumulate(np.maximum(float(start_cap), arr))
    dd_pct = np.where(running_peak > 0.0, ((arr - running_peak) / running_peak) * 100.0, 0.0)
    dd_pct = np.clip(dd_pct, -100.0, 0.0)
    trough_idx = int(np.argmin(dd_pct))
    trough_val = float(arr[trough_idx])
    peak_slice = arr[: trough_idx + 1]
    if len(peak_slice) > 0 and float(np.max(peak_slice)) >= float(start_cap):
        peak_idx = int(np.argmax(peak_slice))
        peak_val = float(arr[peak_idx])
    else:
        peak_idx = 0
        peak_val = float(start_cap)
    return {
        "equity_path": arr,
        "peak_idx": peak_idx,
        "peak_val": peak_val,
        "trough_idx": trough_idx,
        "trough_val": trough_val,
        "max_drawdown_pct": float(dd_pct[trough_idx]),
        "final_val": float(arr[-1]),
    }


def _build_mc_plain_english_report(
    *,
    title: str,
    start_cap: float,
    median_final: float,
    p05_final: float,
    p95_final: float,
    prob_profit_pct: float,
    prob_ruin_pct: float,
    p95_dd_pct: float,
    n_sims: int,
    staking_mode: str,
    score_basis: str,
) -> Dict[str, str]:
    """Build plain-English verdict, drawdown explainer, and verdict-first summary lines for Monte Carlo output."""
    med_ret_pct = ((median_final - start_cap) / start_cap) * 100.0 if start_cap > 1e-12 else 0.0
    prob_loss_pct = max(0.0, 100.0 - prob_profit_pct)
    mode_label = "Compounding" if str(staking_mode).lower() == "compound" else "Constant Stake"
    is_flat = abs(median_final - start_cap) <= 1e-6 and prob_profit_pct <= 1e-6 and abs(p95_final - start_cap) <= 1e-6

    if is_flat:
        verdict_status = "flat"
        verdict_text = (
            f"VERDICT: Flat / NO active signal. Median finish £{median_final:,.2f} from £{start_cap:,.0f} start (+0.0%). "
            f"0.0% of {n_sims:,} simulated paths make money (model outputs a constant prediction)."
        )
    elif median_final < start_cap or prob_profit_pct < 50.0:
        verdict_status = "loss"
        verdict_text = (
            f"VERDICT: Median path LOSES money. Median finish £{median_final:,.2f} from £{start_cap:,.0f} start ({med_ret_pct:+.1f}%). "
            f"Only {prob_profit_pct:.1f}% of {n_sims:,} simulated paths finish above £{start_cap:,.0f} "
            f"({prob_loss_pct:.1f}% lose money). This model is NOT profitable on average in simulation."
        )
    else:
        verdict_status = "profit"
        verdict_text = (
            f"VERDICT: Median path MAKES money. Median finish £{median_final:,.2f} from £{start_cap:,.0f} start ({med_ret_pct:+.1f}%). "
            f"{prob_profit_pct:.1f}% of {n_sims:,} simulated paths finish above £{start_cap:,.0f} "
            f"({prob_loss_pct:.1f}% lose money)."
        )

    abs_dd = abs(p95_dd_pct)
    drawdown_line = (
        f"What drawdown means: 95% Worst Dip {p95_dd_pct:.2f}% means in 95% of paths the worst fall from peak was "
        f"{abs_dd:.2f}% or less — this is NOT the final loss."
    )
    summary_line = (
        f"{title} [One-Wallet MC | {score_basis} | {mode_label}] | "
        f"P(Profit): {prob_profit_pct:.1f}% ({prob_loss_pct:.1f}% lose) | "
        f"Median (50th): £{median_final:,.2f} ({med_ret_pct:+.1f}%) | "
        f"5th Pct: £{p05_final:,.2f} | 95th Pct: £{p95_final:,.2f} | "
        f"95% Worst Dip: {p95_dd_pct:.2f}% | P(Ruin): {prob_ruin_pct:.1f}%"
    )
    explainer_text = (
        "How to read Drawdown vs Final Balance (Plain English):\n"
        "  • Drawdown is the biggest peak-to-trough dip DURING the journey — NOT your final profit or loss.\n"
        "  • Worked example (Winning path with a drawdown): Start £1,000 -> rises to £1,200 peak -> dips to £1,100 "
        "(-8.3% drawdown from peak) -> recovers to £1,300. You still finish +£300 ABOVE £1,000 despite the dip.\n"
        "  • Worked example (Losing path with a small drawdown): Start £1,000 -> drifts down to £955 (-4.5% dip) "
        "and ends at £984.81. A small drawdown still finishes BELOW £1,000 if it never recovers.\n"
        "  • Median (50th) vs 95th percentile are DIFFERENT paths: Median = typical path; 95th = lucky top 5% of paths.\n"
        "  • One-Wallet MC: Simulates ONE continuous wallet from start to finish over the full validation period with no resets.\n"
        "  • Within-Window MC: Simulates separate, reset windows (e.g. rolling sub-period pots) — a different test that is never mixed with One-Wallet MC.\n"
        f"  • Sizing rule in this One-Wallet run: {mode_label}. "
        + (
            "Compounding resizes each round's stake as your wallet grows, so a path can climb very high, suffer a huge "
            "peak-to-trough drawdown (e.g. -91.8% from a high peak), and STILL finish above £1,000."
            if str(staking_mode).lower() == "compound"
            else "Constant Stake keeps each round's stake fixed at Start / R (no compounding)."
        )
    )
    return {
        "mc_verdict_status": verdict_status,
        "mc_verdict_text": verdict_text,
        "mc_drawdown_explanation_line": drawdown_line,
        "mc_summary_line": summary_line,
        "mc_explainer_text": explainer_text,
    }


def simulate_stake_pnl(
    per_era_corr: pd.Series,
    per_era_mmc: Optional[pd.Series] = None,
    *,
    per_era_ncorr: Optional[pd.Series] = None,
    per_era_nmmc: Optional[pd.Series] = None,
    mmc_is_proxy: bool = False,
    total_capital: Optional[float] = None,
    initial_stake: float = 1000.0,
    concurrent_rounds: int = CLASSIC_CONCURRENT_ROUNDS,
    settlement_rounds: int = CLASSIC_SETTLEMENT_ROUNDS,
    corr_weight: Optional[float] = None,
    mmc_weight: Optional[float] = None,
    payout_factor: Union[float, pd.Series] = 1.0,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    neutral_churn: Optional[Union[float, pd.Series]] = None,
    churn_penalty: Optional[Union[float, pd.Series]] = None,
    payout_rule: str = "auto",
    staking_mode: str = "compound",
    stride: int = 1,
    nmr_gbp_rate: Optional[float] = None,
    nmr_gbp_prices: Optional[pd.Series] = None,
    corr_multiplier: Optional[float] = None,
) -> Dict[str, Any]:
    """One-wallet Numerai Classic overlapping-rounds staking simulator (no window resets, no margin).

    - Starts with one wallet at `total_capital` (default £1,000, or `initial_stake` alias).
    - Sizes each weekly round at `TVL / R` (`R = 64` concurrent rounds for Classic); locked capital
      stays locked for `settlement_rounds` (~60 trading days = 12 weekly eras) and cannot be reused.
    - Uses era-accurate Numerai Classic rules (`payout_rule='auto'`):
        * Pre-1393: `3 * CORR60 + 15 * MMC60`, clipped `[-100%, +100%]`.
        * R1393+: `8 * NCORR + 32 * NMMC`, clipped `[-100%, +200%]`, `payout_factor=1.0`.
        * R1399+: neutral churn retention `min(1, 2 / (1 + exp(10 * (churn - 0.1))))` on positive payouts only.
    - Keeps NMR and GBP separate: simulates staking in NMR, then converts to GBP via `nmr_gbp_prices`
      or `nmr_gbp_rate` (or labels constant-FX GBP-equivalent if no FX series/rate is supplied).
    - Equity can never go below 0; `money_drawdown_pct` is always in `[-100, 0]`.
    """
    start_cap_gbp = max(0.0, float(total_capital if total_capital is not None else initial_stake))
    clean_mode = str(staking_mode or "compound").strip().lower()
    if clean_mode not in ("compound", "constant"):
        raise ValueError(f"staking_mode must be 'compound' or 'constant', got {staking_mode!r}")

    era_returns, meta = _compute_clipped_era_returns(
        per_era_corr,
        per_era_mmc=per_era_mmc,
        per_era_ncorr=per_era_ncorr,
        per_era_nmmc=per_era_nmmc,
        mmc_is_proxy=mmc_is_proxy,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
        payout_factor=payout_factor,
        clip_min=clip_min,
        clip_max=clip_max,
        neutral_churn=neutral_churn,
        churn_penalty=churn_penalty,
        payout_rule=payout_rule,
        stride=stride,
        corr_multiplier=corr_multiplier,
    )

    if len(era_returns) == 0:
        empty_series = pd.Series(dtype=np.float64)
        return {
            "start_capital": start_cap_gbp,
            "end_capital": start_cap_gbp,
            "total_profit": 0.0,
            "total_return_pct": 0.0,
            "worst_round": 0.0,
            "worst_round_gbp": 0.0,
            "money_drawdown_pct": 0.0,
            "max_drawdown_gbp": 0.0,
            "start_nmr": start_cap_gbp,
            "end_nmr": start_cap_gbp,
            "profit_nmr": 0.0,
            "max_drawdown_nmr": 0.0,
            "worst_round_nmr": 0.0,
            "fx_status": "no eras",
            "final_stake": start_cap_gbp,
            "net_profit": 0.0,
            "worst_era_gbp": 0.0,
            "committed_capital": start_cap_gbp,
            "concurrent_rounds": int(concurrent_rounds),
            "settlement_rounds": int(settlement_rounds),
            "per_round_stake_initial": start_cap_gbp / float(max(1, int(concurrent_rounds))),
            "staking_mode": clean_mode,
            "wiped_out": False,
            **meta,
            "round_returns": empty_series,
            "round_pnls_gbp": empty_series,
            "equity_curve": empty_series,
            "equity_curve_nmr": empty_series,
            "drawdown_pct_series": empty_series,
        }

    ret_arr = era_returns.to_numpy(dtype=np.float64)
    idx = era_returns.index

    # Separate NMR staking from GBP conversion
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
        if len(fx_aligned) == len(idx) and not fx_aligned.isna().any() and float(fx_aligned.to_numpy()[0]) > 0.0:
            fx_arr = fx_aligned.to_numpy(dtype=np.float64)

    if fx_arr is not None:
        start_fx = float(fx_arr[0])
        start_nmr = start_cap_gbp / start_fx
        fx_status = f"live NMR/GBP series (start £{start_fx:.2f}/NMR, end £{float(fx_arr[-1]):.2f}/NMR)"
    elif nmr_gbp_rate is not None and float(nmr_gbp_rate) > 0.0:
        fixed_fx = float(nmr_gbp_rate)
        start_nmr = start_cap_gbp / fixed_fx
        fx_arr = np.full(len(idx), fixed_fx, dtype=np.float64)
        fx_status = f"fixed FX (£{fixed_fx:.2f}/NMR; converted at end)"
    else:
        start_nmr = start_cap_gbp
        fx_arr = np.ones(len(idx), dtype=np.float64)
        fx_status = "no NMR/GBP FX series provided (constant-FX GBP-equivalent)"

    eq_nmr, round_pnls_nmr, _, max_dd_nmr, worst_round_nmr, wiped_out = _simulate_single_wallet_path(
        ret_arr,
        start_nmr,
        staking_mode=clean_mode,
        concurrent_rounds=concurrent_rounds,
        settlement_rounds=settlement_rounds,
    )

    eq_gbp = np.maximum(0.0, eq_nmr * fx_arr)
    round_pnls_gbp = round_pnls_nmr * fx_arr

    # Wallet drawdown in GBP (always in [-100, 0])
    running_peak_gbp = np.maximum.accumulate(np.maximum(start_cap_gbp, eq_gbp))
    dd_gbp_arr = eq_gbp - running_peak_gbp
    dd_pct_arr = np.where(running_peak_gbp > 0.0, (dd_gbp_arr / running_peak_gbp) * 100.0, 0.0)
    dd_pct_arr = np.clip(dd_pct_arr, -100.0, 0.0)

    end_cap_gbp = float(eq_gbp[-1])
    total_profit_gbp = end_cap_gbp - start_cap_gbp
    total_return_pct = (total_profit_gbp / start_cap_gbp) * 100.0 if start_cap_gbp > 1e-12 else 0.0
    money_dd_pct = float(dd_pct_arr.min())
    max_dd_gbp = float(dd_gbp_arr.min())
    worst_round_gbp = float(round_pnls_gbp.min())

    end_nmr = float(eq_nmr[-1])
    profit_nmr = end_nmr - start_nmr

    return {
        "start_capital": start_cap_gbp,
        "end_capital": end_cap_gbp,
        "total_profit": total_profit_gbp,
        "total_return_pct": total_return_pct,
        "worst_round": worst_round_gbp,
        "worst_round_gbp": worst_round_gbp,
        "money_drawdown_pct": money_dd_pct,
        "max_drawdown_gbp": max_dd_gbp,
        "start_nmr": start_nmr,
        "end_nmr": end_nmr,
        "profit_nmr": profit_nmr,
        "max_drawdown_nmr": float(max_dd_nmr),
        "worst_round_nmr": float(worst_round_nmr),
        "fx_status": fx_status,
        # Backwards-compatible scalar keys (strictly one-wallet values)
        "final_stake": end_cap_gbp,
        "net_profit": total_profit_gbp,
        "worst_era_gbp": worst_round_gbp,
        "committed_capital": start_cap_gbp,
        "concurrent_rounds": int(concurrent_rounds),
        "settlement_rounds": int(settlement_rounds),
        "per_round_stake_initial": start_cap_gbp / float(max(1, int(concurrent_rounds))),
        "staking_mode": clean_mode,
        "wiped_out": bool(wiped_out),
        **meta,
        "round_returns": era_returns,
        "round_pnls_gbp": pd.Series(round_pnls_gbp, index=idx, dtype=np.float64, name="round_pnl_gbp"),
        "equity_curve": pd.Series(eq_gbp, index=idx, dtype=np.float64, name="wallet_gbp"),
        "equity_curve_nmr": pd.Series(eq_nmr, index=idx, dtype=np.float64, name="wallet_nmr"),
        "drawdown_pct_series": pd.Series(dd_pct_arr, index=idx, dtype=np.float64, name="money_drawdown_pct"),
    }


def simulate_stake_pnl_monte_carlo(
    per_era_corr: pd.Series,
    per_era_mmc: Optional[pd.Series] = None,
    *,
    per_era_ncorr: Optional[pd.Series] = None,
    per_era_nmmc: Optional[pd.Series] = None,
    mmc_is_proxy: bool = False,
    total_capital: Optional[float] = None,
    initial_stake: float = 1000.0,
    concurrent_rounds: int = CLASSIC_CONCURRENT_ROUNDS,
    settlement_rounds: int = CLASSIC_SETTLEMENT_ROUNDS,
    corr_weight: Optional[float] = None,
    mmc_weight: Optional[float] = None,
    payout_factor: Union[float, pd.Series] = 1.0,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    neutral_churn: Optional[Union[float, pd.Series]] = None,
    churn_penalty: Optional[Union[float, pd.Series]] = None,
    payout_rule: str = "auto",
    staking_mode: str = "compound",
    stride: int = 1,
    n_sims: int = 500,
    block_size: int = 4,
    random_state: int = 42,
    title: str = "Model",
) -> Dict[str, Any]:
    """One-wallet block-bootstrap Monte Carlo simulator under Numerai Classic overlapping-rounds staking.

    Resamples blocks of per-round returns over the full validation horizon and simulates one continuous
    wallet per path (TVL / 64 per round, no resets, no margin, equity >= 0). Also captures one example
    losing path and one example winning path (with peak, trough, drawdown, and final markers) plus
    plain-English verdict and drawdown explanations.
    """
    start_cap = max(0.0, float(total_capital if total_capital is not None else initial_stake))
    clean_mode = str(staking_mode or "compound").strip().lower()
    if clean_mode not in ("compound", "constant"):
        raise ValueError(f"staking_mode must be 'compound' or 'constant', got {staking_mode!r}")

    era_returns, meta = _compute_clipped_era_returns(
        per_era_corr,
        per_era_mmc=per_era_mmc,
        per_era_ncorr=per_era_ncorr,
        per_era_nmmc=per_era_nmmc,
        mmc_is_proxy=mmc_is_proxy,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
        payout_factor=payout_factor,
        clip_min=clip_min,
        clip_max=clip_max,
        neutral_churn=neutral_churn,
        churn_penalty=churn_penalty,
        payout_rule=payout_rule,
        stride=stride,
    )
    if len(era_returns) == 0:
        report = _build_mc_plain_english_report(
            title=title,
            start_cap=start_cap,
            median_final=start_cap,
            p05_final=start_cap,
            p95_final=start_cap,
            prob_profit_pct=0.0,
            prob_ruin_pct=0.0,
            p95_dd_pct=0.0,
            n_sims=int(n_sims),
            staking_mode=clean_mode,
            score_basis=str(meta["score_basis"]),
        )
        return {
            "start_capital": start_cap,
            "actual_end_capital": start_cap,
            "mc_p05_final_stake": start_cap,
            "mc_median_final_stake": start_cap,
            "mc_mean_final_stake": start_cap,
            "mc_p95_final_stake": start_cap,
            "mc_prob_profit_pct": 0.0,
            "mc_prob_ruin_pct": 0.0,
            "mc_p95_money_drawdown_pct": 0.0,
            "final_stakes": np.empty(0, dtype=np.float64),
            "money_drawdowns_pct": np.empty(0, dtype=np.float64),
            "staking_mode": clean_mode,
            "example_losing_path": None,
            "example_winning_path": None,
            **meta,
            **report,
        }

    ret_arr = era_returns.to_numpy(dtype=np.float64)
    n_t = len(ret_arr)
    n_s = max(10, int(n_sims))
    b_size = min(max(1, int(block_size)), n_t)
    n_blocks = int(np.ceil(float(n_t) / float(b_size)))
    rng = np.random.default_rng(int(random_state))

    starts = rng.integers(0, max(1, n_t - b_size + 1), size=(n_s, n_blocks))
    offsets = np.arange(b_size, dtype=np.int64)[None, None, :]
    boot_idx = (starts[:, :, None] + offsets).reshape(n_s, -1)[:, :n_t]
    sim_rets = ret_arr[boot_idx]

    final_wallets = np.zeros(n_s, dtype=np.float64)
    worst_dds_pct = np.zeros(n_s, dtype=np.float64)
    ruin_flags = np.zeros(n_s, dtype=bool)
    sim_paths = np.zeros((n_s, n_t), dtype=np.float64)

    for s_idx in range(n_s):
        eq_path, _, dd_pct, _, _, wiped = _simulate_single_wallet_path(
            sim_rets[s_idx],
            start_cap,
            staking_mode=clean_mode,
            concurrent_rounds=concurrent_rounds,
            settlement_rounds=settlement_rounds,
        )
        sim_paths[s_idx] = eq_path
        final_wallets[s_idx] = float(eq_path[-1])
        worst_dds_pct[s_idx] = float(dd_pct)
        ruin_flags[s_idx] = bool(wiped)

    actual_eq, _, actual_dd_pct, _, _, _ = _simulate_single_wallet_path(
        ret_arr,
        start_cap,
        staking_mode=clean_mode,
        concurrent_rounds=concurrent_rounds,
        settlement_rounds=settlement_rounds,
    )

    # Pick one representative losing path and one representative winning path for visual drawdown vs final balance explainer
    losing_indices = np.flatnonzero(final_wallets < start_cap - 1e-9)
    winning_indices = np.flatnonzero(final_wallets > start_cap + 1e-9)
    example_losing_path: Optional[Dict[str, Any]] = None
    example_winning_path: Optional[Dict[str, Any]] = None
    if len(losing_indices) > 0:
        med_loss_target = float(np.median(final_wallets[losing_indices]))
        pick_loss = int(losing_indices[int(np.argmin(np.abs(final_wallets[losing_indices] - med_loss_target)))])
        example_losing_path = _extract_path_drawdown_markers(sim_paths[pick_loss], start_cap)
    if len(winning_indices) > 0:
        med_win_target = float(np.median(final_wallets[winning_indices]))
        pick_win = int(winning_indices[int(np.argmin(np.abs(final_wallets[winning_indices] - med_win_target)))])
        example_winning_path = _extract_path_drawdown_markers(sim_paths[pick_win], start_cap)

    p05_val = float(np.percentile(final_wallets, 5.0))
    med_val = float(np.percentile(final_wallets, 50.0))
    mean_val = float(np.mean(final_wallets))
    p95_val = float(np.percentile(final_wallets, 95.0))
    prob_profit = float(np.mean(final_wallets > start_cap) * 100.0)
    prob_ruin = float(np.mean(ruin_flags) * 100.0)
    p95_dd = float(np.percentile(worst_dds_pct, 5.0))

    report = _build_mc_plain_english_report(
        title=title,
        start_cap=start_cap,
        median_final=med_val,
        p05_final=p05_val,
        p95_final=p95_val,
        prob_profit_pct=prob_profit,
        prob_ruin_pct=prob_ruin,
        p95_dd_pct=p95_dd,
        n_sims=n_s,
        staking_mode=clean_mode,
        score_basis=str(meta["score_basis"]),
    )

    return {
        "start_capital": start_cap,
        "actual_end_capital": float(actual_eq[-1]),
        "actual_money_drawdown_pct": float(actual_dd_pct),
        "mc_p05_final_stake": p05_val,
        "mc_median_final_stake": med_val,
        "mc_mean_final_stake": mean_val,
        "mc_p95_final_stake": p95_val,
        "mc_prob_profit_pct": prob_profit,
        "mc_prob_ruin_pct": prob_ruin,
        "mc_p95_money_drawdown_pct": p95_dd,
        "final_stakes": final_wallets,
        "money_drawdowns_pct": worst_dds_pct,
        "n_sims": n_s,
        "block_size": b_size,
        "concurrent_rounds": int(concurrent_rounds),
        "settlement_rounds": int(settlement_rounds),
        "staking_mode": clean_mode,
        "example_losing_path": example_losing_path,
        "example_winning_path": example_winning_path,
        **meta,
        **report,
    }


def plot_stake_pnl_dashboard(
    per_era_corr: pd.Series,
    *,
    per_era_mmc: Optional[pd.Series] = None,
    per_era_ncorr: Optional[pd.Series] = None,
    per_era_nmmc: Optional[pd.Series] = None,
    mmc_is_proxy: bool = False,
    title: str = "Model",
    total_capital: Optional[float] = None,
    initial_stake: float = 1000.0,
    concurrent_rounds: int = CLASSIC_CONCURRENT_ROUNDS,
    settlement_rounds: int = CLASSIC_SETTLEMENT_ROUNDS,
    corr_weight: Optional[float] = None,
    mmc_weight: Optional[float] = None,
    payout_factor: Union[float, pd.Series] = 1.0,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    neutral_churn: Optional[Union[float, pd.Series]] = None,
    churn_penalty: Optional[Union[float, pd.Series]] = None,
    payout_rule: str = "auto",
    staking_mode: str = "compound",
    stride: int = 1,
    nmr_gbp_rate: Optional[float] = None,
    nmr_gbp_prices: Optional[pd.Series] = None,
    save_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Plot a clean ONE-WALLET staking equity curve over the full validation period (clearly labeled by basis & compounding mode)."""
    import matplotlib.pyplot as plt

    pnl = simulate_stake_pnl(
        per_era_corr,
        per_era_mmc=per_era_mmc,
        per_era_ncorr=per_era_ncorr,
        per_era_nmmc=per_era_nmmc,
        mmc_is_proxy=mmc_is_proxy,
        total_capital=total_capital,
        initial_stake=initial_stake,
        concurrent_rounds=concurrent_rounds,
        settlement_rounds=settlement_rounds,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
        payout_factor=payout_factor,
        clip_min=clip_min,
        clip_max=clip_max,
        neutral_churn=neutral_churn,
        churn_penalty=churn_penalty,
        payout_rule=payout_rule,
        staking_mode=staking_mode,
        stride=stride,
        nmr_gbp_rate=nmr_gbp_rate,
        nmr_gbp_prices=nmr_gbp_prices,
    )
    eq_series = pnl["equity_curve"]
    if not isinstance(eq_series, pd.Series) or len(eq_series) == 0:
        return pnl

    start_cap = float(pnl["start_capital"])
    end_cap = float(pnl["end_capital"])
    dd_pct = float(pnl["money_drawdown_pct"])
    dd_gbp = float(pnl["max_drawdown_gbp"])
    worst_round = float(pnl["worst_round_gbp"])
    mode_label = "Compounding" if str(pnl["staking_mode"]).lower() == "compound" else "Constant Stake"

    fig, ax = plt.subplots(figsize=(14.5, 5.8), facecolor="white")
    ax.set_facecolor("white")

    x_vals = eq_series.index.to_numpy()
    y_vals = eq_series.to_numpy(dtype=np.float64)
    line_color = "#059669" if end_cap >= start_cap else "#dc2626"

    ax.plot(x_vals, y_vals, color=line_color, linewidth=2.2, label=f"One Wallet ({mode_label}, £)")
    ax.axhline(start_cap, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.85, label=f"Start (£{start_cap:,.0f})")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Time (Validation Era / Week)")
    ax.set_ylabel("Wallet Balance (£)")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=9.2)

    headline_text = (
        f"Start: £{start_cap:,.0f}     |     "
        f"End: £{end_cap:,.2f} ({pnl['total_return_pct']:+.2f}%)     |     "
        f"Worst peak-to-trough dip: {dd_pct:.2f}% (£{dd_gbp:,.2f})"
    )
    fig.suptitle(
        f"One-Wallet Staking ({mode_label}) — {title}  [{pnl['score_basis']}]",
        fontsize=13.0,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        headline_text,
        ha="center",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    fig.text(
        0.5,
        0.02,
        f"Basis: {pnl['score_basis']} | Worst single round: £{worst_round:+,.2f} | "
        f"Stake: TVL/{concurrent_rounds} (£{pnl['per_round_stake_initial']:.2f} initial, {settlement_rounds}-era lock, {mode_label}) | "
        f"{pnl['payout_rule_summary']} | FX: {pnl['fx_status']}",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#475569",
    )
    plt.tight_layout(rect=(0.0, 0.055, 1.0, 0.865))
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=140)
    plt.show()
    return pnl


def plot_stake_pnl_monte_carlo(
    per_era_corr: pd.Series,
    *,
    per_era_mmc: Optional[pd.Series] = None,
    per_era_ncorr: Optional[pd.Series] = None,
    per_era_nmmc: Optional[pd.Series] = None,
    mmc_is_proxy: bool = False,
    title: str = "Model",
    total_capital: Optional[float] = None,
    initial_stake: float = 1000.0,
    concurrent_rounds: int = CLASSIC_CONCURRENT_ROUNDS,
    settlement_rounds: int = CLASSIC_SETTLEMENT_ROUNDS,
    corr_weight: Optional[float] = None,
    mmc_weight: Optional[float] = None,
    payout_factor: Union[float, pd.Series] = 1.0,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    neutral_churn: Optional[Union[float, pd.Series]] = None,
    churn_penalty: Optional[Union[float, pd.Series]] = None,
    payout_rule: str = "auto",
    staking_mode: str = "compound",
    stride: int = 1,
    n_sims: int = 500,
    block_size: int = 4,
    random_state: int = 42,
    save_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Plot a plain-English 2-panel ONE-WALLET Monte Carlo risk chart with a verdict-first banner and example paths."""
    import matplotlib.pyplot as plt

    mc = simulate_stake_pnl_monte_carlo(
        per_era_corr,
        per_era_mmc=per_era_mmc,
        per_era_ncorr=per_era_ncorr,
        per_era_nmmc=per_era_nmmc,
        mmc_is_proxy=mmc_is_proxy,
        total_capital=total_capital,
        initial_stake=initial_stake,
        concurrent_rounds=concurrent_rounds,
        settlement_rounds=settlement_rounds,
        corr_weight=corr_weight,
        mmc_weight=mmc_weight,
        payout_factor=payout_factor,
        clip_min=clip_min,
        clip_max=clip_max,
        neutral_churn=neutral_churn,
        churn_penalty=churn_penalty,
        payout_rule=payout_rule,
        staking_mode=staking_mode,
        stride=stride,
        n_sims=n_sims,
        block_size=block_size,
        random_state=random_state,
        title=title,
    )
    finals = np.asarray(mc["final_stakes"], dtype=np.float64)
    if len(finals) == 0:
        return mc

    start_cap = float(mc["start_capital"])
    actual_end = float(mc["actual_end_capital"])
    med_val = float(mc["mc_median_final_stake"])
    p05_val = float(mc["mc_p05_final_stake"])
    p95_val = float(mc["mc_p95_final_stake"])
    prob_profit = float(mc["mc_prob_profit_pct"])
    prob_loss = max(0.0, 100.0 - prob_profit)
    mode_label = "Compounding" if str(mc["staking_mode"]).lower() == "compound" else "Constant Stake"

    verdict_status = str(mc.get("mc_verdict_status", "flat"))
    if verdict_status == "loss":
        banner_bg, banner_edge, banner_fg = "#fef2f2", "#dc2626", "#991b1b"
    elif verdict_status == "profit":
        banner_bg, banner_edge, banner_fg = "#ecfdf5", "#059669", "#065f46"
    else:
        banner_bg, banner_edge, banner_fg = "#f8fafc", "#64748b", "#334155"

    fig, (ax_hist, ax_paths) = plt.subplots(
        1, 2, figsize=(16.5, 6.8), facecolor="white", gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    for ax in (ax_hist, ax_paths):
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- Left Panel: Final Balance Distribution with Red (Loss < Start) vs Green (Profit > Start) shading ---
    f_min = float(np.min(finals))
    f_max = float(np.max(finals))
    span = max(f_max - f_min, start_cap * 0.04)
    x_lo = max(0.0, min(f_min, start_cap) - 0.08 * span)
    x_hi = max(f_max, start_cap) + 0.08 * span

    ax_hist.axvspan(x_lo, start_cap, color="#fee2e2", alpha=0.35, zorder=0, label=f"Losing zone (< £{start_cap:,.0f}: {prob_loss:.1f}%)")
    ax_hist.axvspan(start_cap, x_hi, color="#dcfce7", alpha=0.35, zorder=0, label=f"Winning zone (> £{start_cap:,.0f}: {prob_profit:.1f}%)")

    if f_max - f_min <= 1e-6:
        ax_hist.bar([start_cap], [len(finals)], width=max(10.0, start_cap * 0.02), color="#64748b", alpha=0.8, edgecolor="white")
    else:
        counts, bin_edges = np.histogram(finals, bins=32)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)
        bar_colors = ["#dc2626" if c < start_cap else "#059669" for c in bin_centers]
        ax_hist.bar(bin_centers, counts, width=bin_widths * 0.92, color=bar_colors, alpha=0.78, edgecolor="white", zorder=2)

    ax_hist.axvline(start_cap, color="#0f172a", linestyle="--", linewidth=1.8, zorder=4, label=f"Start (£{start_cap:,.0f})")
    ax_hist.axvline(med_val, color="#b91c1c" if med_val < start_cap else "#047857", linestyle="-", linewidth=2.3, zorder=5, label=f"Median / 50th (£{med_val:,.2f})")
    ax_hist.axvline(p05_val, color="#991b1b", linestyle=":", linewidth=1.6, zorder=4, label=f"5th Pct (£{p05_val:,.2f})")
    ax_hist.axvline(p95_val, color="#15803d", linestyle=":", linewidth=1.6, zorder=4, label=f"95th Pct (£{p95_val:,.2f})")
    ax_hist.axvline(actual_end, color="#1d4ed8", linestyle="-.", linewidth=1.5, zorder=4, label=f"Actual Path End (£{actual_end:,.2f})")

    ax_hist.set_xlim(x_lo, x_hi)
    ax_hist.set_xlabel("Final One-Wallet Balance (£)")
    ax_hist.set_ylabel(f"Number of Simulated Paths (out of {mc['n_sims']:,})")
    ax_hist.set_title(f"Final Wallet Distribution — {prob_profit:.1f}% Win / {prob_loss:.1f}% Lose", fontsize=11.5, fontweight="bold", pad=8)
    ax_hist.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.3)

    # --- Right Panel: Example Losing Path vs Example Winning Path (Peak, Drawdown Dip, Final) ---
    lose_ex = mc.get("example_losing_path")
    win_ex = mc.get("example_winning_path")
    ax_paths.axhline(start_cap, color="#0f172a", linestyle="--", linewidth=1.4, label=f"Start (£{start_cap:,.0f})")

    def _plot_example_path(ex: Dict[str, Any], color: str, tag: str) -> None:
        arr = np.asarray(ex["equity_path"], dtype=np.float64)
        steps = np.arange(len(arr))
        pk_i, pk_v = int(ex["peak_idx"]), float(ex["peak_val"])
        tr_i, tr_v = int(ex["trough_idx"]), float(ex["trough_val"])
        fn_v = float(ex["final_val"])
        dd_v = float(ex["max_drawdown_pct"])
        ax_paths.plot(steps, arr, color=color, linewidth=1.9, alpha=0.9, label=f"{tag}: End £{fn_v:,.0f} (Max Dip {dd_v:.1f}%)")
        ax_paths.scatter([pk_i], [pk_v], color=color, marker="^", s=65, zorder=5)
        ax_paths.scatter([tr_i], [tr_v], color=color, marker="v", s=65, zorder=5)
        ax_paths.scatter([len(arr) - 1], [fn_v], color=color, marker="o", s=55, zorder=5)
        if tr_i > pk_i and pk_v > tr_v:
            ax_paths.plot([pk_i, tr_i], [pk_v, tr_v], color=color, linestyle=":", linewidth=1.3, alpha=0.85)
        ax_paths.annotate(
            f"Peak £{pk_v:,.0f}",
            xy=(pk_i, pk_v),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.8,
            fontweight="bold",
            color=color,
        )
        ax_paths.annotate(
            f"Dip {dd_v:.1f}%\n(£{tr_v:,.0f})",
            xy=(tr_i, tr_v),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            fontsize=7.6,
            color=color,
        )
        ax_paths.annotate(
            f"Final £{fn_v:,.0f}",
            xy=(len(arr) - 1, fn_v),
            xytext=(-6, 6),
            textcoords="offset points",
            ha="right",
            fontsize=7.8,
            fontweight="bold",
            color=color,
        )

    if lose_ex is not None:
        _plot_example_path(lose_ex, "#dc2626", "Example Losing Path")
    if win_ex is not None:
        _plot_example_path(win_ex, "#059669", "Example Winning Path")
    if lose_ex is None and win_ex is None:
        ax_paths.text(
            0.5,
            0.5,
            f"All {mc['n_sims']:,} simulated paths stayed flat at £{start_cap:,.2f}\n(constant prediction — zero active signal).",
            transform=ax_paths.transAxes,
            ha="center",
            va="center",
            fontsize=10.0,
            color="#475569",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
        )

    ax_paths.set_xlabel("Simulated Week (Era Step)")
    ax_paths.set_ylabel("One-Wallet Balance (£)")
    ax_paths.set_title("Drawdown (Mid-Path Dip) vs Final Balance — Example Paths", fontsize=11.5, fontweight="bold", pad=8)
    ax_paths.legend(loc="best", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.3)

    # --- Top Suptitle, Plain-English Verdict Banner, & Bottom Drawdown Footnote ---
    import textwrap

    wrapped_verdict = textwrap.fill(str(mc["mc_verdict_text"]), width=110)
    fig.suptitle(
        f"One-Wallet Monte Carlo — {title}  [{mc['score_basis']}]",
        fontsize=13.0,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.935,
        wrapped_verdict,
        ha="center",
        va="top",
        fontsize=9.6,
        fontweight="bold",
        color=banner_fg,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": banner_bg, "edgecolor": banner_edge, "linewidth": 1.6},
    )
    fig.text(
        0.5,
        0.015,
        f"{mc['mc_drawdown_explanation_line']}  |  "
        f"One-Wallet MC (continuous wallet, no resets; {mode_label} sizing TVL/{concurrent_rounds}, {settlement_rounds}-era lock, block_size={mc['block_size']})  |  "
        f"Basis: {mc['score_basis']}",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#334155",
    )
    plt.tight_layout(rect=(0.0, 0.055, 1.0, 0.835))
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=140)
    plt.show()
    return mc


def build_quick_results_df(search_or_results: Any) -> pd.DataFrame:
    """Build a flat candidate results table with score_version, extended CV metrics, one-wallet OOF staking P&L scalars, and hyperparameters."""
    cv_res = getattr(search_or_results, "cv_results_", search_or_results)
    if not cv_res:
        _CV_FOLD_METRIC_LOG.clear()
        return pd.DataFrame()
    df = pd.DataFrame(cv_res)
    if "mean_test_score" in df.columns:
        df["mean_test_score_pct"] = (df["mean_test_score"].astype(float) * 100.0).round(2)

    # Attach per-candidate extended metrics & one-wallet OOF staking P&L scalars logged by `candidate_score`
    split_cols = sorted(
        [col for col in df.columns if col.startswith("split") and col.endswith("_test_score")],
        key=lambda c: int(c[len("split") : c.index("_test_score")]),
    )
    if _CV_FOLD_METRIC_LOG and split_cols:
        metric_specs = (
            ("mean_numerai_corr", "numerai_corr", 4),
            ("mean_era_corr_std", "era_corr_std", 4),
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
            ("mean_corr_drawdown", "corr_drawdown", 4),
        )
        pnl_scalar_keys = (
            "start_capital",
            "end_capital",
            "total_profit",
            "total_return_pct",
            "money_drawdown_pct",
            "max_drawdown_gbp",
            "worst_round_gbp",
        )
        used_mask = [False] * len(_CV_FOLD_METRIC_LOG)
        score_versions_col: List[str] = []
        agg_cols: Dict[str, List[float]] = {out_col: [] for out_col, _, _ in metric_specs}
        pnl_cols: Dict[str, List[float]] = {k: [] for k in pnl_scalar_keys}
        for row_idx in range(len(df)):
            matched_records: List[Dict[str, Any]] = []
            for s_col in split_cols:
                score_val = float(df.loc[row_idx, s_col])
                if not np.isfinite(score_val):
                    continue
                for log_idx, rec in enumerate(_CV_FOLD_METRIC_LOG):
                    logged_score = float(rec.get("score_value", rec["numerai_corr"]))
                    if not used_mask[log_idx] and abs(logged_score - score_val) <= 1e-12:
                        used_mask[log_idx] = True
                        matched_records.append(rec)
                        break
            if matched_records:
                score_versions_col.append(str(matched_records[0].get("score_version", "v0_default")))
            else:
                score_versions_col.append("unknown")
            for out_col, src_key, decimals in metric_specs:
                if matched_records:
                    val = float(np.mean([float(rec.get(src_key, np.nan)) for rec in matched_records]))
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
                oof_pnl = simulate_stake_pnl(stitched_oof)
                for k in pnl_scalar_keys:
                    pnl_cols[k].append(round(float(oof_pnl[k]), 2))
            else:
                for k in pnl_scalar_keys:
                    pnl_cols[k].append(np.nan)
        df["score_version"] = score_versions_col
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


def run_numereval_diagnostics(
    y_true: Any,
    y_pred: Any,
    eras: Any,
    *,
    variant_name: str = "Model",
    plot: bool = False,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """Optional side-by-side community `numereval` (Council of Elders) diagnostics companion.

    Note: `numereval` uses classic Spearman rank correlation and is NOT the official v6.0 Unity
    `numerai_corr` / `NMMC` / staking metric. It runs alongside `numerai_tools` when installed
    and returns `None` gracefully when `numereval` is unavailable (e.g. default Kaggle runtime).
    """
    if not NUMEREVAL_AVAILABLE or _numereval_mod is None:
        if verbose:
            print(
                "[numereval companion] Not installed in this environment (optional). "
                "Install with: pip install --no-deps git+https://github.com/councilofelders/numereval.git"
            )
        return None

    eval_df = pd.DataFrame({
        "era": np.asarray(eras).ravel(),
        "target": np.asarray(y_true, dtype=np.float64).ravel(),
        "prediction": np.asarray(y_pred, dtype=np.float64).ravel(),
    }).dropna(subset=["era", "target", "prediction"])
    if eval_df.empty:
        return None

    metrics_df = _numereval_mod.evaluate(eval_df, plot=plot, feature_exposure=False)
    if verbose:
        print(
            f"[numereval companion — {variant_name}] Community Spearman diagnostics "
            "(informational alongside official numerai_tools v6.0 metrics; NOT official v6.0 NCORR/NMMC):"
        )
    return metrics_df
