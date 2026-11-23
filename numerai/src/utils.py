"""utils.py — Shared NumeraiExtractions, per-ticker time-series extraction, diagnostics Parquet export, and CORR sanity-checked evaluation suite."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numerai_tools.scoring import numerai_corr
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
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller

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
        discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile", subsample=None)
        binned = discretizer.fit_transform(clean.reshape(-1, 1)).ravel()
        return np.clip(np.rint(binned), 0, 4).astype(np.int8)
    z_scores = (clean - float(np.mean(clean))) / std
    binned = np.rint(np.clip(z_scores, -2.0, 2.0) + 2.0)
    return np.clip(binned, 0, 4).astype(np.int8)


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
            eras = pd.Series(np.asarray(era_series).ravel(), index=clean_df.index)
            corr_sum = np.zeros(n_features, dtype=np.float64)
            n_valid_eras = 0
            for _, era_features in clean_df.groupby(eras, sort=True):
                era_target = clean_target.loc[era_features.index]
                if len(era_features) < 3 or float(era_target.std()) <= 1e-12:
                    continue
                era_x = era_features.to_numpy(dtype=np.float32, copy=False)
                era_y = era_target.to_numpy(dtype=np.float32, copy=False)
                era_c = np.nan_to_num(r_regression(era_x, era_y), nan=0.0, posinf=0.0, neginf=0.0)
                corr_sum += era_c
                n_valid_eras += 1
            if n_valid_eras > 0:
                return pd.Series(corr_sum / float(n_valid_eras), index=feature_cols, dtype=np.float64)
        feature_matrix = clean_df.to_numpy(dtype=np.float32, copy=False)
        target_values = clean_target.to_numpy(dtype=np.float32, copy=False)
        corrs = np.nan_to_num(r_regression(feature_matrix, target_values), nan=0.0, posinf=0.0, neginf=0.0)
        return pd.Series(corrs, index=feature_cols, dtype=np.float64)

    if method == "numerai_mean":
        if era_series is not None and len(era_series) == len(clean_df):
            eras = pd.Series(np.asarray(era_series).ravel(), index=clean_df.index)
            per_era_num_corrs: List[pd.Series] = []
            for _, era_features in clean_df.groupby(eras, sort=True):
                era_target = clean_target.loc[era_features.index]
                if len(era_features) < 3 or float(era_target.std()) <= 1e-12:
                    continue
                era_f32 = era_features.astype(np.float32, copy=False)
                per_era_num_corrs.append(numerai_corr(era_f32, era_target).fillna(0.0))
            if per_era_num_corrs:
                return pd.concat(per_era_num_corrs, axis=1).mean(axis=1).reindex(feature_cols).fillna(0.0)
        return numerai_corr(clean_df.astype(np.float32, copy=False), clean_target).reindex(feature_cols).fillna(0.0)

    target_values = clean_target.to_numpy(dtype=np.float32, copy=False)
    if method == "spearman":
        ranked_target = stats.rankdata(target_values).astype(np.float32, copy=False)
        corrs = np.zeros(n_features, dtype=np.float64)
        chunk_size = 128
        for start in range(0, n_features, chunk_size):
            chunk_cols = feature_cols[start : start + chunk_size]
            chunk_mat = clean_df[chunk_cols].to_numpy(dtype=np.float32, copy=False)
            ranked_chunk = np.apply_along_axis(stats.rankdata, 0, chunk_mat).astype(np.float32, copy=False)
            corrs[start : start + len(chunk_cols)] = np.nan_to_num(
                r_regression(ranked_chunk, ranked_target), nan=0.0, posinf=0.0, neginf=0.0
            )
        return pd.Series(corrs, index=feature_cols, dtype=np.float64)

    feature_matrix = clean_df.to_numpy(dtype=np.float32, copy=False)
    corrs = np.nan_to_num(r_regression(feature_matrix, target_values), nan=0.0, posinf=0.0, neginf=0.0)
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
            gc.collect()

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
            gc.collect()

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
        if len(valid_df) < 3 or float(valid_df["prediction"].std()) <= 1e-12 or float(valid_df["target"].std()) <= 1e-12:
            era_scores[era_key] = 0.0
        else:
            corr_val = float(spearmanr(valid_df["target"], valid_df["prediction"])[0])
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
        if len(valid_df) < 3 or float(valid_df["prediction"].std()) <= 1e-12 or float(valid_df["target"].std()) <= 1e-12:
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
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    if len(preds) < 3 or float(np.std(preds)) <= 1e-12:
        return 0.0
    cols = (
        list(feature_columns)
        if feature_columns is not None
        else [str(col) for col in features_df.columns if str(col) not in FORBIDDEN_FEATURE_COLS]
    )
    feature_corrs: List[float] = []
    for col in cols:
        if col not in features_df.columns:
            continue
        col_series = features_df[col]
        if isinstance(col_series.dtype, np.dtype) and np.issubdtype(col_series.dtype, np.integer):
            f_vals = col_series.to_numpy(dtype=np.float32, copy=False)
        elif pd.api.types.is_numeric_dtype(col_series.dtype):
            f_vals = col_series.fillna(2.0).to_numpy(dtype=np.float32, copy=False)
        else:
            f_vals = pd.to_numeric(col_series, errors="coerce").fillna(2.0).to_numpy(dtype=np.float32, copy=False)
        if float(np.std(f_vals)) <= 1e-12:
            feature_corrs.append(0.0)
        else:
            corr_val = float(spearmanr(preds, f_vals)[0])
            feature_corrs.append(corr_val if np.isfinite(corr_val) else 0.0)
    return float(np.round(np.std(feature_corrs), 4)) if feature_corrs else 0.0


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
    raw_lag1_acf = float(numerai_era_corrs.autocorr(lag=1)) if len(numerai_era_corrs) >= 3 else 0.0
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
            _, raw_arch_p, _, _ = het_arch(corrs_arr - float(np.mean(corrs_arr)), nlags=nlags_arch)
            arch_lm_pvalue = float(np.round(float(raw_arch_p), 4)) if np.isfinite(raw_arch_p) else 1.0
        except Exception:
            arch_lm_pvalue = 1.0
    else:
        arch_lm_pvalue = 1.0

    if len(corrs_arr) >= 6 and corrs_std > 1e-12:
        try:
            adf_stat_res = adfuller(corrs_arr, maxlag=min(4, max(0, len(corrs_arr) // 4 - 1)), autolag=None)
            raw_adf_p = float(adf_stat_res[1])
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
        if features_df is not None
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
                adf_res = adfuller(series_vals, maxlag=min(4, max(0, len(series_vals) // 4 - 1)), autolag=None)
                stat_val = float(np.round(float(adf_res[0]), 4))
                pval = float(np.round(float(adf_res[1]), 4)) if np.isfinite(adf_res[1]) else 1.0
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
) -> np.ndarray:
    """Chronological walk-forward holdout evaluation respecting `target_horizon_eras` settlement lag."""
    val_era_arr = np.asarray(validation_eras, dtype=np.int32).ravel()
    unique_val_eras = np.array(sorted(np.unique(val_era_arr)), dtype=np.int32)
    step = max(1, int(step_eras))
    horizon = max(1, int(target_horizon_eras))
    mode = str(window_mode or "expanding").strip().lower()
    train_era_span = int(np.max(train_eras) - np.min(train_eras)) if len(train_eras) else 52

    all_X = pd.concat([X_train, X_validation], axis=0, ignore_index=True)
    all_y = pd.concat([pd.Series(y_train).reset_index(drop=True), pd.Series(y_validation).reset_index(drop=True)], axis=0, ignore_index=True)
    all_eras = np.concatenate([np.asarray(train_eras, dtype=np.int32).ravel(), val_era_arr])
    is_val_row = np.concatenate([np.zeros(len(X_train), dtype=bool), np.ones(len(X_validation), dtype=bool)])

    wf_preds = np.full(len(X_validation), float(np.mean(y_train)) if len(y_train) else 0.5, dtype=np.float64)
    for start_idx in range(0, len(unique_val_eras), step):
        chunk_eras = unique_val_eras[start_idx : start_idx + step]
        forecast_origin_era = int(chunk_eras[0])
        settled_cutoff_era = forecast_origin_era - horizon
        fit_mask = all_eras <= settled_cutoff_era
        if mode == "sliding":
            min_sliding_era = settled_cutoff_era - max(1, train_era_span)
            fit_mask = fit_mask & (all_eras >= min_sliding_era)
        if not np.any(fit_mask):
            fit_mask = ~is_val_row

        chunk_model = clone(estimator)
        chunk_params = chunk_model.get_params(deep=True)
        if "estimator__walk_forward_step_eras" in chunk_params:
            chunk_model.set_params(estimator__walk_forward_step_eras=None)
        elif "walk_forward_step_eras" in chunk_params:
            chunk_model.set_params(walk_forward_step_eras=None)
        chunk_model.fit(all_X.loc[fit_mask], all_y.loc[fit_mask])
        val_chunk_mask = np.isin(val_era_arr, chunk_eras)
        wf_preds[val_chunk_mask] = chunk_model.predict(X_validation.loc[val_chunk_mask])
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


_CV_FOLD_METRIC_LOG: List[Dict[str, float]] = []

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
        **bin_shares,
    })

    print(
        f"  [Fold Score] Numerai CORR: {mean_corr * 100.0:+.2f}% ({mean_corr:.4f}) | "
        f"Sharpe: {sharpe_val:.2f} | Positive Eras: {pos_era_pct:.1f}% | "
        f"Dir Acc: {dir_acc_pct:.1f}% | Accurate Bin: {bin_shares['accurate_pct']:.1f}% | MAE: {mae_val:.4f}"
    )
    return mean_corr


def build_quick_results_df(search_or_results: Any) -> pd.DataFrame:
    """Build a flat candidate results table with percentage score, extended CV metrics, and every hyperparameter expanded into its own column."""
    cv_res = getattr(search_or_results, "cv_results_", search_or_results)
    if not cv_res:
        _CV_FOLD_METRIC_LOG.clear()
        return pd.DataFrame()
    df = pd.DataFrame(cv_res)
    if "mean_test_score" in df.columns:
        df["mean_test_corr_pct"] = (df["mean_test_score"].astype(float) * 100.0).round(2)

    # Attach per-candidate extended metrics logged by `candidate_score` during CV folds
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
        used_mask = [False] * len(_CV_FOLD_METRIC_LOG)
        agg_cols: Dict[str, List[float]] = {out_col: [] for out_col, _, _ in metric_specs}
        for row_idx in range(len(df)):
            matched_records: List[Dict[str, float]] = []
            for s_col in split_cols:
                score_val = float(df.loc[row_idx, s_col])
                if not np.isfinite(score_val):
                    continue
                for log_idx, rec in enumerate(_CV_FOLD_METRIC_LOG):
                    if not used_mask[log_idx] and abs(rec["numerai_corr"] - score_val) <= 1e-12:
                        used_mask[log_idx] = True
                        matched_records.append(rec)
                        break
            for out_col, src_key, decimals in metric_specs:
                if matched_records:
                    val = float(np.mean([rec[src_key] for rec in matched_records]))
                    agg_cols[out_col].append(round(val, decimals))
                else:
                    agg_cols[out_col].append(np.nan)
        for out_col, vals in agg_cols.items():
            df[out_col] = vals
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
