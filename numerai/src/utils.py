"""utils.py — Shared NumeraiExtractions, per-ticker time-series extraction, diagnostics Parquet export, and CORR sanity-checked evaluation suite."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import inspect
import json
import os
import sys
import time
from datetime import datetime
from functools import partial
from operator import itemgetter
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

from afml_utils import (
    CombinatorialPurgedEraCV,
    PurgedEraKFold,
    build_compact_afml_reference_table,
    build_era_cv_splits,
    cluster_features_onc,
    compute_clustered_mda,
    compute_deflated_sharpe_ratio,
    compute_drawdown_and_tuw,
    compute_hhi_concentration,
    compute_lo_adjusted_sharpe,
    compute_min_track_record_length,
    compute_onc_clusters,
    compute_pbo_cscv,
    compute_per_era_fnc,
    compute_probabilistic_sharpe_ratio,
    compute_strategy_risk,
    compute_variation_of_information_matrix,
    neutralize_predictions,
    plot_afml_backtest_story,
    purge_train_before_validation,
    summarize_afml_backtest_diagnostics,
)

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
SHARPE_OFFICIAL_CEILING = 2.50

_MEMORY_GUARD_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "min_free_gb": 3.0,
    "max_fraction": 0.80,
}

_TIME_GUARD_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "max_candidate_seconds": 3600.0,
}

_TF_BACKEND_CACHE: Dict[str, Any] = {
    "checked": False,
    "module": None,
}


def get_tf_backend() -> Optional[Any]:
    """Lazily import TensorFlow once per process for accelerated linear algebra / tensor ops.

    Enables GPU memory growth when GPUs are present so TensorFlow does not pre-allocate all VRAM.
    Returns `None` cleanly if TensorFlow is not installed.
    """
    if _TF_BACKEND_CACHE["checked"]:
        return _TF_BACKEND_CACHE["module"]
    _TF_BACKEND_CACHE["checked"] = True
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf_mod

        try:
            for gpu in tf_mod.config.list_physical_devices("GPU"):
                tf_mod.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
        _TF_BACKEND_CACHE["module"] = tf_mod
    except ImportError:
        _TF_BACKEND_CACHE["module"] = None
    return _TF_BACKEND_CACHE["module"]


def configure_memory_guard(
    enabled: bool = True,
    min_free_gb: float = 3.0,
    max_fraction: float = 0.80,
) -> Dict[str, Any]:
    """Configure the explicit RAM fail-safe guard (`enabled=False` turns it off completely)."""
    _MEMORY_GUARD_CONFIG["enabled"] = bool(enabled)
    _MEMORY_GUARD_CONFIG["min_free_gb"] = max(0.0, float(min_free_gb))
    _MEMORY_GUARD_CONFIG["max_fraction"] = float(np.clip(float(max_fraction), 0.10, 0.99))
    return dict(_MEMORY_GUARD_CONFIG)


def configure_time_guard(
    enabled: bool = True,
    max_candidate_seconds: float = 3600.0,
) -> Dict[str, Any]:
    """Configure the explicit per-candidate wall-clock time fail-safe (`enabled=False` turns it off completely)."""
    _TIME_GUARD_CONFIG["enabled"] = bool(enabled)
    _TIME_GUARD_CONFIG["max_candidate_seconds"] = max(0.0, float(max_candidate_seconds))
    return dict(_TIME_GUARD_CONFIG)


def check_time_budget(
    start_time: Optional[float],
    *,
    context: str = "",
    max_seconds: Optional[float] = None,
) -> bool:
    """Fail-safe wall-clock check: raises `TimeoutError` if elapsed time exceeds `max_candidate_seconds`.

    When `_TIME_GUARD_CONFIG['enabled']` is `False` (or `start_time` is None / limit <= 0), returns `True`.
    Inside `BayesSearchCV` / `GridSearchCV` (`error_score=-1.0`), raising `TimeoutError` aborts only the
    offending slow candidate cleanly instead of stalling the entire hyperparameter search.
    """
    if not bool(_TIME_GUARD_CONFIG.get("enabled", True)) or start_time is None:
        return True
    limit_s = float(_TIME_GUARD_CONFIG["max_candidate_seconds"] if max_seconds is None else max_seconds)
    if limit_s <= 0.0:
        return True
    elapsed_s = float(time.perf_counter() - float(start_time))
    if elapsed_s > limit_s:
        ctx = f"[{context}] " if context else ""
        msg = (
            f"[TIME FAIL-SAFE] {ctx}Aborting candidate: elapsed {elapsed_s:.1f}s "
            f"exceeds max_candidate_seconds={limit_s:.1f}s."
        )
        print(msg)
        _reclaim_cached_memory()
        raise TimeoutError(msg)
    return True


def _read_cgroup_available_bytes() -> Optional[int]:
    """Read remaining container RAM from Linux cgroup v2 or v1 (used on Kaggle CPU/GPU VMs)."""
    try:
        max_path = Path("/sys/fs/cgroup/memory.max")
        cur_path = Path("/sys/fs/cgroup/memory.current")
        if max_path.is_file() and cur_path.is_file():
            raw_max = max_path.read_text(encoding="utf-8").strip()
            if raw_max.isdigit():
                limit_b = int(raw_max)
                used_b = int(cur_path.read_text(encoding="utf-8").strip())
                if limit_b > 0 and limit_b < (1 << 60):
                    return max(0, limit_b - used_b)
    except OSError:
        pass
    try:
        lim_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        use_path = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if lim_path.is_file() and use_path.is_file():
            limit_b = int(lim_path.read_text(encoding="utf-8").strip())
            used_b = int(use_path.read_text(encoding="utf-8").strip())
            if limit_b > 0 and limit_b < (1 << 60):
                return max(0, limit_b - used_b)
    except (OSError, ValueError):
        pass
    return None


class _WinMemoryStatusEx(ctypes.Structure):
    """Top-level Win32 `MEMORYSTATUSEX` struct for querying available physical RAM without psutil."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _read_host_available_bytes() -> Optional[int]:
    """Read host available RAM cross-platform via `psutil`, `/proc/meminfo`, or Windows `GlobalMemoryStatusEx`."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except (OSError, ValueError):
        pass
    if os.name == "nt":
        try:
            stat = _WinMemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_WinMemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            pass
    return None


def get_available_memory_bytes() -> Optional[int]:
    """Return conservative available RAM in bytes (minimum of container cgroup headroom and host available RAM)."""
    candidates = [v for v in (_read_cgroup_available_bytes(), _read_host_available_bytes()) if v is not None]
    return min(candidates) if candidates else None


def _reclaim_cached_memory() -> None:
    """Clear per-process extraction/LightGBM caches (if loaded) and run garbage collection."""
    bu_mod = sys.modules.get("baseline_utils")
    if bu_mod is not None and hasattr(bu_mod, "clear_extraction_cache"):
        bu_mod.clear_extraction_cache()
    lgb_mod = sys.modules.get("lgbm_utils")
    if lgb_mod is not None and hasattr(lgb_mod, "clear_lgb_dataset_cache"):
        lgb_mod.clear_lgb_dataset_cache()
    gc.collect()


def check_memory_budget(
    estimated_bytes: int,
    *,
    context: str = "",
    min_free_gb: Optional[float] = None,
    max_fraction: Optional[float] = None,
    min_check_mb: float = 64.0,
) -> bool:
    """Fail-safe RAM preflight check: raises `MemoryError` if a candidate allocation would breach the safety floor.

    When `_MEMORY_GUARD_CONFIG['enabled']` is `False`, this is a zero-overhead no-op returning `True`.
    Small allocations below `min_check_mb` (default 64 MB) are skipped so tiny slices/unit tests never false-trigger
    on machines with low ambient free RAM. Inside `BayesSearchCV` / `GridSearchCV` (`error_score=-1.0`),
    raising `MemoryError` fails only the offending candidate instead of triggering an OS OOM kill (`DeadKernelError`).
    """
    if not bool(_MEMORY_GUARD_CONFIG.get("enabled", True)):
        return True
    est_b = max(0, int(estimated_bytes))
    min_check_b = int(max(0.0, float(min_check_mb)) * (1024 ** 2))
    if est_b == 0 or est_b < min_check_b:
        return True

    floor_gb = float(_MEMORY_GUARD_CONFIG["min_free_gb"] if min_free_gb is None else min_free_gb)
    frac_lim = float(_MEMORY_GUARD_CONFIG["max_fraction"] if max_fraction is None else max_fraction)
    floor_b = int(max(0.0, floor_gb) * (1024 ** 3))

    avail_b = get_available_memory_bytes()
    if avail_b is None:
        return True

    over_floor = (avail_b - est_b) < floor_b
    over_frac = est_b > int(avail_b * frac_lim)
    if over_floor or over_frac:
        # Try reclaiming caches + GC once before failing the candidate
        _reclaim_cached_memory()
        avail_b = get_available_memory_bytes()
        if avail_b is None:
            return True
        over_floor = (avail_b - est_b) < floor_b
        over_frac = est_b > int(avail_b * frac_lim)
        if over_floor or over_frac:
            est_gb = est_b / (1024 ** 3)
            avail_gb = avail_b / (1024 ** 3)
            ctx_label = f" [{context}]" if context else ""
            msg = (
                f"[MEMORY FAIL-SAFE]{ctx_label} Aborting candidate: estimated allocation "
                f"{est_gb:.2f} GB exceeds safe budget (available={avail_gb:.2f} GB, "
                f"min_free_gb={floor_gb:.2f} GB, max_fraction={frac_lim:.0%})."
            )
            print(msg)
            raise MemoryError(msg)
    return True


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


def _quantize_era_series_group(group: pd.Series, *, strategy: str) -> np.ndarray:
    """Quantize one era's numeric Series slice into `0..4` `np.int8` bins."""
    return _quantize_1d_array(group.to_numpy(dtype=np.float64, copy=False), strategy=strategy)


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
                partial(_quantize_era_series_group, strategy=strategy)
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


def _impute_era_series_group(series: pd.Series, *, agg_fn: str) -> pd.Series:
    """Impute missing values in one era group using its within-era mean or median."""
    fill_value = series.mean() if agg_fn == "mean" else series.median()
    return series.fillna(fill_value)


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
            partial(_impute_era_series_group, agg_fn=agg_fn)
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


def _extract_skopt_dimension_categories(dim: Any) -> List[Any]:
    """Return the category list from a skopt categorical dimension, or an empty list."""
    return list(dim.categories) if hasattr(dim, "categories") else []


def build_conditional_bayesian_spaces(
    space: Mapping[str, Any],
    total_n_iter: int = 12,
) -> Any:
    """Split a Bayesian search space into conditional `(subspace, n_iter)` tuples so inactive child parameters are not wasted."""
    prepared = prepare_bayesian_space(space)
    if not SKOPT_AVAILABLE or SkCategorical is None or int(total_n_iter) < 4:
        return prepared

    # 1. PersistenceEstimator conditional split (`None` vs `last`/`expanding_mean` vs `rolling_mean`)
    if "estimator__persistence_mode" in prepared:
        modes = _extract_skopt_dimension_categories(prepared["estimator__persistence_mode"])
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
        sel_cats = _extract_skopt_dimension_categories(prepared["estimator__extractor__sparse_selector"])
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
    tf_mod = get_tf_backend() if (n_rows * n_cols >= 100_000) else None

    if tf_mod is not None:
        n_eras = len(starts)
        seg_ids = np.repeat(np.arange(n_eras, dtype=np.int32), counts)
        seg_ids_t = tf_mod.constant(seg_ids, dtype=tf_mod.int32)
        counts_t = tf_mod.constant(counts.astype(np.float64), dtype=tf_mod.float64)[:, None]
        y_c_t = tf_mod.constant(y_centered, dtype=tf_mod.float64)[:, None]
        y_ss_t = tf_mod.constant(y_ss, dtype=tf_mod.float64)[:, None]
        chunk_size = 512
        for c0 in range(0, n_cols, chunk_size):
            c1 = min(n_cols, c0 + chunk_size)
            if is_df and col_names is not None:
                x_chunk = X[col_names[c0:c1]].to_numpy(dtype=np.float64, copy=False)
            else:
                x_chunk = np.asarray(X[:, c0:c1], dtype=np.float64)
            x_sorted_t = tf_mod.constant(x_chunk[order], dtype=tf_mod.float64)
            x_sum_t = tf_mod.math.unsorted_segment_sum(x_sorted_t, seg_ids_t, n_eras)
            x_mean_t = x_sum_t / counts_t
            x_centered_t = x_sorted_t - tf_mod.gather(x_mean_t, seg_ids_t)
            x_ss_t = tf_mod.math.unsorted_segment_sum(x_centered_t * x_centered_t, seg_ids_t, n_eras)
            xy_cov_t = tf_mod.math.unsorted_segment_sum(x_centered_t * y_c_t, seg_ids_t, n_eras)
            denom_t = tf_mod.sqrt(tf_mod.maximum(x_ss_t * y_ss_t, 0.0))
            era_corrs = tf_mod.where(
                denom_t > 1e-12,
                tf_mod.math.divide_no_nan(xy_cov_t, denom_t),
                tf_mod.zeros_like(xy_cov_t),
            ).numpy()
            era_corrs = np.nan_to_num(era_corrs, nan=0.0, posinf=0.0, neginf=0.0)
            result[c0:c1] = era_corrs[valid_era_mask].mean(axis=0)
            del x_sorted_t, x_sum_t, x_mean_t, x_centered_t, x_ss_t, xy_cov_t, denom_t, era_corrs
        del seg_ids_t, counts_t, y_c_t, y_ss_t
        return result

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
    n_rows = len(y_f64)
    tf_mod = get_tf_backend() if (n_rows * n_features >= 100_000) else None
    if tf_mod is not None:
        y_c_t = tf_mod.constant(y_centered, dtype=tf_mod.float64)
        y_ss_t = tf_mod.constant(y_ss, dtype=tf_mod.float64)
        tf_chunk = max(int(chunk_size), 256)
        for start in range(0, n_features, tf_chunk):
            chunk_cols = cols[start : start + tf_chunk]
            x_chunk = df[chunk_cols].to_numpy(dtype=np.float64, copy=False)
            if rank_columns:
                x_chunk = np.apply_along_axis(stats.rankdata, 0, x_chunk).astype(np.float64, copy=False)
            x_t = tf_mod.constant(x_chunk, dtype=tf_mod.float64)
            x_c_t = x_t - tf_mod.reduce_mean(x_t, axis=0, keepdims=True)
            x_ss_t = tf_mod.reduce_sum(x_c_t * x_c_t, axis=0)
            denom_t = tf_mod.sqrt(tf_mod.maximum(x_ss_t * y_ss_t, 0.0))
            xy_cov_t = tf_mod.linalg.matvec(x_c_t, y_c_t, transpose_a=True)
            chunk_corrs = tf_mod.where(
                denom_t > 1e-12,
                tf_mod.math.divide_no_nan(xy_cov_t, denom_t),
                tf_mod.zeros_like(xy_cov_t),
            ).numpy()
            corrs[start : start + len(chunk_cols)] = np.nan_to_num(chunk_corrs, nan=0.0, posinf=0.0, neginf=0.0)
            del x_t, x_c_t, x_ss_t, denom_t, xy_cov_t, chunk_corrs
        del y_c_t, y_ss_t
        return corrs

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
            return self._quantize_signal_frozen(arr, signal_key)
        return quantize_to_5bins(
            arr,
            strategy=q_strat,
            eras=era_vals.to_numpy() if era_vals is not None else None,
            quantize_scope=scope,
        )

    def _quantize_signal_frozen(self, values: np.ndarray, signal_key: str) -> np.ndarray:
        """Leak-free quantization using a StandardScaler frozen strictly during `fit()` (never looks ahead across val eras)."""
        q_strat = str(self.quantize_strategy or "zscore").strip().lower()
        arr = np.asarray(values, dtype=np.float64).ravel()
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
        # Fallback if transform was called without fitting this signal: causal constant-center zscore without refitting
        finite = np.isfinite(arr)
        if not finite.any():
            return np.full(len(arr), 2, dtype=np.int8)
        return quantize_to_5bins(arr, strategy=q_strat, quantize_scope="global")

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
        self.sparse_scaler_: Optional[StandardScaler] = None
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
            check_memory_budget(
                len(fit_clean_df) * len(selected_cols) * 4 * 2,
                context=f"NumeraiExtractions.sparse_selector({sel_mode})",
            )
            fit_target_arr = (target_vec.loc[fit_mask] if use_filter else target_vec).to_numpy(dtype=np.float32, copy=False)
            sparse_raw = fit_clean_df[selected_cols].to_numpy(dtype=np.float32, copy=False)
            self.sparse_scaler_ = StandardScaler().fit(sparse_raw)
            sfm.fit(
                self.sparse_scaler_.transform(sparse_raw),
                fit_target_arr,
            )
            sparse_support = sfm.get_support()
            sparse_kept = [col for col, keep in zip(selected_cols, sparse_support) if keep]
            if sparse_kept:
                selected_cols = sparse_kept
            self.sparse_selector_ = sfm

        self.pca_: Optional[PCA] = None
        self.pca_scaler_: Optional[StandardScaler] = None
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
            check_memory_budget(
                len(fit_clean_df) * len(selected_cols) * 4 * 2,
                context=f"NumeraiExtractions.pca(n_components={n_comp})",
            )
            pca_raw = fit_clean_df[selected_cols].to_numpy(dtype=np.float32, copy=False)
            self.pca_scaler_ = StandardScaler().fit(pca_raw)
            self.pca_ = PCA(n_components=n_comp, random_state=42)
            self.pca_.fit(self.pca_scaler_.transform(pca_raw))

        self.feature_names_in_ = np.asarray(raw_cols, dtype=object)
        self.n_features_in_ = len(raw_cols)
        self.feature_corrs_series_ = corr_series
        self.selected_feature_names_ = selected_cols
        self.selected_corrs_ = corr_series.reindex(selected_cols).fillna(0.0).to_numpy(dtype=np.float64)
        self.signal_scalers_ = {}
        self.era_means_tail_: Optional[pd.DataFrame] = None
        self.ticker_tail_df_: Optional[pd.DataFrame] = None

        # Always run a strictly training-only fitting pass whenever frozen scalers or causal tail buffers are needed
        need_fit_pass = (
            str(self.quantize_scope or "era").strip().lower() == "train_fit"
            or bool(self.include_rolling_era)
            or bool(self.include_lag_diff)
            or (self.ticker_features not in (None, False, "None", "none", "null", ""))
        )
        if need_fit_pass:
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
            n_rows_disp = len(base_features)
            disp_batch = 65_536
            row_mean_out = np.empty(n_rows_disp, dtype=np.float32) if "mean" in dispersion_tokens else None
            row_std_out = np.empty(n_rows_disp, dtype=np.float32) if "std" in dispersion_tokens else None
            row_ext_out = np.empty(n_rows_disp, dtype=np.float32) if "extreme_frac" in dispersion_tokens else None
            for start_r in range(0, n_rows_disp, disp_batch):
                end_r = min(n_rows_disp, start_r + disp_batch)
                chunk_mat = base_features[start_r:end_r].to_numpy(dtype=np.float32, copy=False)
                if row_mean_out is not None:
                    row_mean_out[start_r:end_r] = chunk_mat.mean(axis=1)
                if row_std_out is not None:
                    row_std_out[start_r:end_r] = chunk_mat.std(axis=1)
                if row_ext_out is not None:
                    row_ext_out[start_r:end_r] = ((chunk_mat == 0) | (chunk_mat == 4)).mean(axis=1)
                del chunk_mat
            if row_mean_out is not None:
                extra_cols["row_feat_mean_bin"] = np.clip(np.rint(row_mean_out), 0, 4).astype(np.int8)
            if row_std_out is not None:
                extra_cols["row_feat_std_bin"] = self._quantize_signal(
                    row_std_out, era_vals, "row_feat_std_bin"
                )
            if row_ext_out is not None:
                extra_cols["row_extreme_frac_bin"] = np.clip(
                    np.rint(row_ext_out * 4.0), 0, 4
                ).astype(np.int8)
            del row_mean_out, row_std_out, row_ext_out

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
                    is_fit_call = bool(getattr(self, "_is_fitting_", False))
                    max_t_lookback = max(t_windows + (t_lags if t_lags else []) + [12])

                    if is_fit_call:
                        # Save strictly training trailing eras as causal warm-up tail buffer for unseen validation/test eras
                        uniq_fit_eras = sorted(era_vals.unique())
                        keep_tail_eras = set(uniq_fit_eras[-max_t_lookback:])
                        self.ticker_tail_df_ = temp_ticker_in.loc[era_vals.isin(keep_tail_eras)].copy()
                        ticker_work_in = temp_ticker_in
                        used_tail = False
                    else:
                        saved_tail = getattr(self, "ticker_tail_df_", None)
                        if saved_tail is not None and len(saved_tail) > 0 and len(era_vals) > 0:
                            min_cur_era = int(era_vals.min())
                            causal_tail = saved_tail.loc[saved_tail[self.era_col] < min_cur_era]
                            if len(causal_tail) > 0:
                                tail_part = causal_tail.copy()
                                tail_part["_is_query_row_"] = False
                                query_part = temp_ticker_in.copy()
                                query_part["_is_query_row_"] = True
                                ticker_work_in = pd.concat([tail_part, query_part], axis=0, ignore_index=True)
                                used_tail = True
                            else:
                                ticker_work_in = temp_ticker_in
                                used_tail = False
                        else:
                            ticker_work_in = temp_ticker_in
                            used_tail = False

                    # Never silently fall back to look-ahead 'global' when scope == 'train_fit'
                    eff_ticker_qscope = "global" if scope == "global" else "era"
                    temp_ticker_out, added_ticker_cols = compute_per_ticker_timeseries_features(
                        ticker_work_in,
                        top_ticker_cols,
                        ticker_col=self.ticker_col,
                        era_col=self.era_col,
                        ticker_features=self.ticker_features,
                        ticker_windows=t_windows,
                        ticker_lag_steps=t_lags,
                        top_n_cols=len(top_ticker_cols),
                        quantize_scope=eff_ticker_qscope,
                    )
                    if used_tail:
                        temp_ticker_out = temp_ticker_out.loc[temp_ticker_out["_is_query_row_"]]
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

            if need_era_means and era_means is not None:
                windows = [int(w) for w in raw_windows] if isinstance(raw_windows, (list, tuple)) else [4, 12]
                lags = [int(l_val) for l_val in raw_lags] if isinstance(raw_lags, (list, tuple)) else [1, 2, 4]
                max_era_hist = max(windows + lags + [1])
                is_fit_call = bool(getattr(self, "_is_fitting_", False))
                if is_fit_call:
                    self.era_means_tail_ = era_means.tail(max_era_hist).copy()
                    combined_era_means = era_means
                else:
                    saved_em = getattr(self, "era_means_tail_", None)
                    if saved_em is not None and len(saved_em) > 0 and len(era_means) > 0:
                        min_cur_e = era_means.index.min()
                        causal_em = saved_em.loc[saved_em.index < min_cur_e]
                        combined_era_means = (
                            pd.concat([causal_em, era_means], axis=0).sort_index()
                            if len(causal_em) > 0
                            else era_means
                        )
                    else:
                        combined_era_means = era_means

                if rolling_active:
                    for window in windows:
                        if window <= 1:
                            continue
                        rolling_means = combined_era_means.rolling(window=window, min_periods=1).mean()
                        for col in top_cols:
                            mapped = era_vals.map(rolling_means[col]).to_numpy(dtype=np.float64)
                            # Era-level scalar series is constant within an era: ALWAYS use frozen train-fit scaler
                            # (unless scope=='global' was explicitly requested), eliminating val/test look-ahead bias!
                            extra_cols[f"{col}_era_roll{window}"] = (
                                quantize_to_5bins(mapped, quantize_scope="global")
                                if scope == "global"
                                else self._quantize_signal_frozen(mapped, f"{col}_era_roll{window}")
                            )

                if lag_active:
                    for lag_step in lags:
                        if lag_step <= 0:
                            continue
                        diff_means = combined_era_means.diff(periods=lag_step).fillna(0.0)
                        for col in top_cols:
                            mapped = era_vals.map(diff_means[col]).to_numpy(dtype=np.float64)
                            extra_cols[f"{col}_era_diff{lag_step}"] = (
                                quantize_to_5bins(mapped, quantize_scope="global")
                                if scope == "global"
                                else self._quantize_signal_frozen(mapped, f"{col}_era_diff{lag_step}")
                            )

        if getattr(self, "pca_", None) is not None:
            n_rows_pca = len(base_features)
            n_comp_out = int(getattr(self.pca_, "n_components_", self.pca_.components_.shape[0]))
            pca_scores = np.empty((n_rows_pca, n_comp_out), dtype=np.float32)
            pca_batch = 65_536
            pca_scaler = getattr(self, "pca_scaler_", None)
            for start_r in range(0, n_rows_pca, pca_batch):
                end_r = min(n_rows_pca, start_r + pca_batch)
                pca_chunk = base_features[start_r:end_r].to_numpy(dtype=np.float32, copy=False)
                if pca_scaler is not None:
                    pca_chunk = pca_scaler.transform(pca_chunk)
                pca_scores[start_r:end_r] = self.pca_.transform(pca_chunk)
                del pca_chunk
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
    *,
    eras: Optional[Any] = None,
    return_per_era: bool = False,
) -> Union[float, Tuple[float, pd.Series]]:
    """Compute Feature Exposure (`std([spearman(pred, f)])`): per-era mean when `eras` is supplied, pooled otherwise."""
    if feature_columns is not None and len(feature_columns) == 0:
        return (float(np.nan), pd.Series(dtype=np.float64)) if return_per_era else float(np.nan)
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    if len(preds) < 3 or _is_constant_1d(preds):
        return (0.0, pd.Series(dtype=np.float64)) if return_per_era else 0.0
    cols = (
        list(feature_columns)
        if feature_columns is not None
        else [str(col) for col in features_df.columns if str(col) not in FORBIDDEN_FEATURE_COLS]
    )
    valid_cols = [col for col in cols if col in features_df.columns]
    if not valid_cols:
        return (float(np.nan), pd.Series(dtype=np.float64)) if return_per_era else float(np.nan)

    if eras is not None:
        era_arr = np.asarray(eras).ravel()
        if len(era_arr) == len(preds) and len(np.unique(era_arr)) >= 1:
            era_exposures: Dict[Any, float] = {}
            sub_df = features_df[valid_cols]
            order = np.argsort(era_arr, kind="stable")
            sorted_eras = era_arr[order]
            change = np.empty(len(era_arr), dtype=bool)
            change[0] = True
            if len(era_arr) > 1:
                change[1:] = sorted_eras[1:] != sorted_eras[:-1]
            starts = np.flatnonzero(change)
            ends = np.append(starts[1:], len(era_arr))
            row_labels = sub_df.index.to_numpy()
            for s, e in zip(starts, ends):
                idx = order[s:e]
                era_key = sorted_eras[s]
                p_e = preds[idx]
                if len(idx) < 3 or _is_constant_1d(p_e):
                    era_exposures[era_key] = 0.0
                    continue
                fc = _pooled_pearson_corr(sub_df.loc[row_labels[idx]], p_e, rank_columns=True)
                era_exposures[era_key] = float(np.std(fc)) if len(fc) else 0.0
            per_era_s = pd.Series(era_exposures, dtype=np.float64, name="feature_exposure")
            mean_fe = float(np.round(per_era_s.mean(), 4)) if len(per_era_s) else 0.0
            return (mean_fe, per_era_s) if return_per_era else mean_fe

    feature_corrs = _pooled_pearson_corr(features_df[valid_cols], preds, rank_columns=True)
    pooled_fe = float(np.round(np.std(feature_corrs), 4)) if len(feature_corrs) else 0.0
    return (pooled_fe, pd.Series([pooled_fe], dtype=np.float64)) if return_per_era else pooled_fe


def evaluate_numerai_predictions(
    y_true: Any,
    y_pred: Any,
    eras: Optional[Any] = None,
    *,
    features_df: Optional[pd.DataFrame] = None,
    feature_columns: Optional[Sequence[str]] = None,
    compute_fnc: Optional[bool] = None,
    label: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate predictions using official `numerai_corr`, Numerai Forum #1454 diagnostics (Era Consistency > 0.002, FNC, per-era Feature Exposure), AFML risk stats (PSR, Lo-Sharpe, MinTRL, TuW, HHI), and the `[0.005, 0.04]` CORR sanity tripwire."""
    targets = np.asarray(y_true, dtype=np.float64).ravel()
    preds = np.asarray(y_pred, dtype=np.float64).ravel()

    spearman_era_corrs = compute_per_era_spearman_corr(targets, preds, eras=eras)
    numerai_era_corrs = compute_per_era_numerai_corr(targets, preds, eras=eras)

    # Legacy tutorial Spearman payout proxy (not official v6.0 NCORR/NMMC staking payout; see simulate_stake_pnl)
    payout_raw = (spearman_era_corrs / 0.2).clip(-1.0, 1.0)
    spearman_mean = float(np.round(spearman_era_corrs.mean(), 4)) if len(spearman_era_corrs) else 0.0
    numerai_corr_mean = float(np.round(numerai_era_corrs.mean(), 4)) if len(numerai_era_corrs) else 0.0
    payout_mean = float(np.round(payout_raw.mean(), 4)) if len(payout_raw) else 0.0
    spearman_sharpe = float(np.round(sharpe_ratio(spearman_era_corrs), 4))
    numerai_sharpe = float(np.round(sharpe_ratio(numerai_era_corrs), 4))
    validation_psr = float(np.round(compute_probabilistic_sharpe_ratio(numerai_era_corrs, benchmark_sr=0.0), 4))
    lo_info = compute_lo_adjusted_sharpe(numerai_era_corrs)
    lo_sharpe = float(np.round(lo_info["sr_lo"], 4))
    min_trl = compute_min_track_record_length(numerai_era_corrs, benchmark_sr=0.0, confidence=0.95)
    min_trl_out = float(np.round(min_trl, 1)) if np.isfinite(min_trl) else float(np.inf)
    hhi_info = compute_hhi_concentration(numerai_era_corrs)
    tuw_info = compute_drawdown_and_tuw(numerai_era_corrs, is_returns=True)
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
        in_unit_scale = (float(np.nanmin(preds)) >= -0.05) and (float(np.nanmax(preds)) <= 1.05) and (0.15 <= float(np.mean(preds)) <= 0.85)
        pred_center = 0.5 if in_unit_scale else float(np.nanmedian(preds))
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
    # Numerai Forum #1454 (Posts #5-#6): Era Consistency (% of eras with CORR > 0.002)
    era_consistency_pct = float(np.round(float(np.mean(corrs_arr > 0.002)) * 100.0, 2)) if len(corrs_arr) else 0.0
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

    max_feature_exposure = float(np.nan)
    if features_df is not None and (feature_columns is None or len(feature_columns) > 0):
        fe_mean, fe_per_era = compute_feature_exposure(
            features_df, preds, feature_columns=feature_columns, eras=eras, return_per_era=True
        )
        feature_exposure = float(fe_mean)
        if len(fe_per_era) > 0:
            max_feature_exposure = float(np.round(float(fe_per_era.max()), 4))
    else:
        feature_exposure = float(np.nan)

    # Feature Neutral Correlation (FNC): automatic on small/medium feature slices or whenever compute_fnc=True
    fnc_mean = float(np.nan)
    fnc_sharpe = float(np.nan)
    fnc_era_corrs = pd.Series(dtype=np.float64)
    if features_df is not None and (feature_columns is None or len(feature_columns) > 0):
        n_cand_cols = len(feature_columns) if feature_columns is not None else len(features_df.columns)
        auto_run_fnc = (compute_fnc is True) or (compute_fnc is None and len(preds) <= 250_000 and n_cand_cols <= 100)
        if auto_run_fnc:
            fnc_res = compute_per_era_fnc(
                targets,
                preds,
                features_df,
                eras=eras,
                feature_columns=feature_columns,
                proportion=1.0,
                max_neutralize_features=100,
            )
            fnc_mean = float(fnc_res["fnc_mean"])
            fnc_sharpe = float(fnc_res["fnc_sharpe"])
            fnc_era_corrs = fnc_res["fnc_era_corrs"]

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
            f"Positive Eras: {positive_era_pct:.1f}% | Era Consistency (>0.002): {era_consistency_pct:.1f}% | "
            f"Directional Accuracy: {directional_accuracy_pct:.1f}%"
        )
        print(f"Numerai CORR: {numerai_corr_mean:.4f}")
        print(f"Spearman Correlation: {spearman_mean:.4f}")
        print(f"Average Payout (Legacy Spearman Proxy): {payout_mean:.4f}")
        print(
            f"Sharpe Ratio (Numerai / Lo-Adj / Spearman): {numerai_sharpe:.4f} / {lo_sharpe:.4f} / {spearman_sharpe:.4f} | "
            f"PSR (SR > 0): {validation_psr:.4f} | MinTRL: {min_trl_out} eras"
        )
        print(
            f"Cumulative CORR Drawdown: {corr_drawdown:.4f} pts | Max TuW: {tuw_info['tuw_max_eras']} eras "
            f"({tuw_info['tuw_max_years']:.2f} yr) | HHI (+ / -): {hhi_info['hhi_pos']:.4f} / {hhi_info['hhi_neg']:.4f}"
        )
        print(f"Mean Absolute Error (MAE): {mae_val:.4f}")
        print(f"Forecast Bias (mean(y - y_hat)): {forecast_bias:.4f}")
        print(f"Directional Hit Rate: {directional_hit_rate:.4f} ({directional_accuracy_pct:.1f}%)")
        print(
            f"Era CORR White-Noise & Regime Tests | Lag-1 ACF: {era_corr_lag1_acf:.4f} | "
            f"Ljung-Box p: {ljung_box_pvalue:.4f} | ARCH-LM p: {arch_lm_pvalue:.4f} | ADF p: {adf_pvalue:.4f}"
        )
        print(
            f"Era CORR Tail Risk | Skew: {era_corr_skew:.4f} | Excess Kurtosis: {era_corr_kurtosis:.4f} | "
            f"Jarque-Bera p: {jarque_bera_pvalue:.4f}"
        )
        if np.isfinite(feature_exposure):
            fe_max_str = f" (max era: {max_feature_exposure:.4f})" if np.isfinite(max_feature_exposure) else ""
            print(f"Feature Exposure (Per-Era Mean): {feature_exposure:.4f}{fe_max_str}")
        if np.isfinite(fnc_mean):
            print(f"Feature Neutral Correlation (FNC): {fnc_mean:.4f} (FNC Sharpe: {fnc_sharpe:.4f})")
        print(sanity_msg)

    return {
        "numerai_corr": numerai_corr_mean,
        "numerai_corr_pct": numerai_corr_pct,
        "positive_era_pct": positive_era_pct,
        "era_consistency_pct": era_consistency_pct,
        "directional_accuracy_pct": directional_accuracy_pct,
        "spearman": spearman_mean,
        "payout": payout_mean,
        "legacy_spearman_payout": payout_mean,
        "numerai_sharpe": numerai_sharpe,
        "lo_sharpe": lo_sharpe,
        "spearman_sharpe": spearman_sharpe,
        "validation_psr": validation_psr,
        "min_trl": min_trl_out,
        "tuw_max_eras": int(tuw_info["tuw_max_eras"]),
        "hhi_pos": float(hhi_info["hhi_pos"]),
        "hhi_neg": float(hhi_info["hhi_neg"]),
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
        "max_feature_exposure": max_feature_exposure,
        "feature_neutral_corr": fnc_mean,
        "fnc_sharpe": fnc_sharpe,
        "fnc_era_corrs": fnc_era_corrs,
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
    settlement_buffer_eras: int = 0,
    balance_era_density: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """Chronological walk-forward holdout evaluation respecting `target_horizon_eras` settlement lag,
    optional `settlement_buffer_eras`, era-density equalization, double-stride prevention, and causal
    feature-only gap context for rolling/sequence predictors."""
    train_era_arr = np.asarray(train_eras, dtype=np.int32).ravel()
    val_era_arr = np.asarray(validation_eras, dtype=np.int32).ravel()
    unique_val_eras = np.array(sorted(np.unique(val_era_arr)), dtype=np.int32)
    step = max(1, int(step_eras))
    horizon = max(1, int(target_horizon_eras))
    buf = max(0, int(settlement_buffer_eras))
    stride = max(1, int(era_stride))
    mode = str(window_mode or "expanding").strip().lower()
    train_era_span = int(np.max(train_era_arr) - np.min(train_era_arr)) if len(train_era_arr) else 52
    rng = np.random.default_rng(int(random_state))

    # Median rows per era in X_train to detect downsampling imbalance vs 100%-density validation eras
    if len(train_era_arr) > 0:
        _, tr_counts = np.unique(train_era_arr, return_counts=True)
        median_train_era_rows = int(np.median(tr_counts))
    else:
        median_train_era_rows = 0

    y_train_s = pd.Series(np.asarray(y_train, dtype=np.float32).ravel(), index=X_train.index)
    y_val_s = pd.Series(np.asarray(y_validation, dtype=np.float32).ravel(), index=X_validation.index)

    # Suspend per-process extraction/LGB caches during walk-forward steps (each step has a unique cutoff -> 0% cache hit rate)
    import sys

    _bu_mod = sys.modules.get("baseline_utils")
    _lgb_mod = sys.modules.get("lgbm_utils")
    _prev_ext_suspended = bool(getattr(_bu_mod, "_EXTRACTION_CACHE_SUSPENDED", False)) if _bu_mod is not None else False
    _prev_lgb_suspended = bool(getattr(_lgb_mod, "_LGB_DATASET_CACHE_SUSPENDED", False)) if _lgb_mod is not None else False
    if _bu_mod is not None and hasattr(_bu_mod, "set_extraction_cache_suspended"):
        _bu_mod.set_extraction_cache_suspended(True)
    if _lgb_mod is not None and hasattr(_lgb_mod, "set_lgb_dataset_cache_suspended"):
        _lgb_mod.set_lgb_dataset_cache_suspended(True)

    wf_preds = np.full(len(X_validation), float(np.mean(y_train_s)) if len(y_train_s) else 0.5, dtype=np.float64)
    wf_start_t = time.perf_counter()
    try:
        for start_idx in range(0, len(unique_val_eras), step):
            check_time_budget(wf_start_t, context="walk_forward_predict")
            chunk_eras = unique_val_eras[start_idx : start_idx + step]
            forecast_origin_era = int(chunk_eras[0])
            settled_cutoff_era = forecast_origin_era - horizon - buf
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

            val_fit_mask = np.isin(val_era_arr, settled_val_eras) if has_val_settled else np.zeros(len(val_era_arr), dtype=bool)
            # Prevent 4x-100x era-density imbalance when X_train was downsampled while X_validation is 100% density
            if balance_era_density and has_train and has_val_settled and median_train_era_rows > 0:
                val_fit_indices = np.flatnonzero(val_fit_mask)
                val_fit_eras_sub = val_era_arr[val_fit_indices]
                _, vf_counts = np.unique(val_fit_eras_sub, return_counts=True)
                if len(vf_counts) > 0 and float(np.median(vf_counts)) > 1.5 * float(median_train_era_rows):
                    balanced_mask = np.zeros(len(val_era_arr), dtype=bool)
                    for e_id in settled_val_eras:
                        e_pos = np.flatnonzero(val_era_arr == e_id)
                        if len(e_pos) > median_train_era_rows:
                            pick = rng.choice(e_pos, size=median_train_era_rows, replace=False)
                            balanced_mask[pick] = True
                        else:
                            balanced_mask[e_pos] = True
                    val_fit_mask = balanced_mask

            if has_train and not has_val_settled:
                if bool(np.all(train_mask)):
                    X_fit = X_train
                    y_fit = y_train_s
                else:
                    X_fit = X_train.loc[train_mask]
                    y_fit = y_train_s.loc[train_mask]
            elif not has_train and has_val_settled:
                X_fit = X_validation.loc[val_fit_mask]
                y_fit = y_val_s.loc[val_fit_mask]
            else:
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
            set_overrides: Dict[str, Any] = {}
            if "estimator__walk_forward_step_eras" in chunk_params:
                set_overrides["estimator__walk_forward_step_eras"] = None
            elif "walk_forward_step_eras" in chunk_params:
                set_overrides["walk_forward_step_eras"] = None
            # Prevent double-striding inside step_model when walk_forward_predict already strided settled eras
            if stride > 1:
                if "estimator__era_stride" in chunk_params:
                    set_overrides["estimator__era_stride"] = 1
                elif "era_stride" in chunk_params:
                    set_overrides["era_stride"] = 1
            if set_overrides:
                step_model.set_params(**set_overrides)
            step_model.fit(X_fit, y_fit)

            val_chunk_mask = np.isin(val_era_arr, chunk_eras)
            # Provide strictly causal feature-only warm-up context for eras in (settled_cutoff_era, forecast_origin_era)
            # so rolling / MiniRocket sequence models don't suffer a cold-start across the settlement gap
            gap_mask = (val_era_arr > settled_cutoff_era) & (val_era_arr < forecast_origin_era)
            if bool(np.any(gap_mask)) and isinstance(X_validation, pd.DataFrame):
                # Strip any target column from gap+chunk to guarantee zero target access during predict()
                non_target_cols = [c for c in X_validation.columns if not str(c).startswith("target")]
                X_gap = X_validation.loc[gap_mask, non_target_cols]
                X_chunk = X_validation.loc[val_chunk_mask, non_target_cols]
                X_pred_in = pd.concat([X_gap, X_chunk], axis=0, ignore_index=True)
                chunk_preds_full = np.asarray(step_model.predict(X_pred_in), dtype=np.float64).ravel()
                wf_preds[val_chunk_mask] = chunk_preds_full[len(X_gap) :]
                del X_gap, X_chunk, X_pred_in, chunk_preds_full
            else:
                wf_preds[val_chunk_mask] = step_model.predict(X_validation.loc[val_chunk_mask])

            del X_fit, y_fit, step_model
            gc.collect()
    finally:
        if _bu_mod is not None and hasattr(_bu_mod, "set_extraction_cache_suspended"):
            _bu_mod.set_extraction_cache_suspended(_prev_ext_suspended)
        if _lgb_mod is not None and hasattr(_lgb_mod, "set_lgb_dataset_cache_suspended"):
            _lgb_mod.set_lgb_dataset_cache_suspended(_prev_lgb_suspended)
        gc.collect()
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


VALID_DOWNSAMPLE_MODES = (None, "recent", "earliest", "uniform")
VALID_VAL_DOWNSAMPLE_MODES = VALID_DOWNSAMPLE_MODES + ("same",)
VALID_SCORE_VERSIONS = ("v0_default", "v1_skepta", "v2_skepta_mmc")
CANONICAL_BASELINES: Tuple[str, ...] = ("bin_snapping", "lag_reversal", "persistence")
BASELINE_DISPLAY_NAMES: Dict[str, str] = {
    "bin_snapping": "Bin-Snapping",
    "lag_reversal": "Lag-Reversal",
    "persistence": "Persistence",
}
_BASELINE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "bin_snapping": ("bin_snapping",),
    "bin-snapping": ("bin_snapping",),
    "binsnapping": ("bin_snapping",),
    "baseline_a": ("bin_snapping",),
    "lag_reversal": ("lag_reversal",),
    "lag-reversal": ("lag_reversal",),
    "lagreversal": ("lag_reversal",),
    "baseline_e": ("lag_reversal",),
    "persistence": ("persistence",),
    "baseline_p": ("persistence",),
    "all": CANONICAL_BASELINES,
}


def normalize_baseline_mode(baseline_mode: Any) -> Optional[Tuple[str, ...]]:
    """Normalize `baseline_mode` into a deduplicated tuple of canonical names (`'bin_snapping'`, `'lag_reversal'`, `'persistence'`).

    Accepts a single name (`'bin_snapping'`, `'lag_reversal'`, `'persistence'`, `'all'`), a comma/pipe-separated
    string (`'bin_snapping, lag_reversal'`), a list/tuple/set of names (`['bin_snapping', 'persistence']`), or
    legacy aliases (`'baseline_a'`, `'baseline_e'`, `'baseline_p'`, `'both'`).
    """
    if baseline_mode is None:
        return None
    if isinstance(baseline_mode, str):
        raw_tokens = [tok.strip().lower() for tok in baseline_mode.replace("|", ",").split(",") if tok.strip()]
    elif isinstance(baseline_mode, (list, tuple, set)):
        raw_tokens = []
        for item in baseline_mode:
            for tok in str(item).replace("|", ",").split(","):
                if tok.strip():
                    raw_tokens.append(tok.strip().lower())
    else:
        raise ValueError(
            "baseline_mode must be a baseline name, comma-separated string, or list/tuple of "
            "['bin_snapping', 'lag_reversal', 'persistence'] (or 'all')."
        )
    if not raw_tokens:
        raise ValueError(
            "baseline_mode cannot be empty; choose from ['bin_snapping', 'lag_reversal', 'persistence'] or 'all'."
        )
    resolved: List[str] = []
    for tok in raw_tokens:
        if tok not in _BASELINE_ALIASES:
            raise ValueError(
                f"Unknown baseline {tok!r}; choose from 'bin_snapping', 'lag_reversal', 'persistence', 'all' "
                "(as a string, comma-separated string, or list)."
            )
        for canon in _BASELINE_ALIASES[tok]:
            if canon not in resolved:
                resolved.append(canon)
    return tuple(resolved)


def build_experiment_setup(
    *,
    target_col: str = "target",
    feature_set: str = "small",
    era_stride: Union[str, int, None] = "auto",
    model_tag: str = "baselines",
    baseline_mode: Optional[Any] = None,
    search_mode: str = "bayes",
    score_version: str = "v0_default",
    eval_mode: str = "both",
    n_splits: int = 1,
    cv_split: float = 0.8,
    downsample_mode: Optional[str] = None,
    downsample_frac: float = 1.0,
    val_downsample_mode: Optional[str] = None,
    val_downsample_frac: Optional[float] = None,
    n_iter: int = 37,
    n_jobs: int = -1,
    save_results: bool = True,
    save_plots: bool = False,
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
    memory_guard_enabled: bool = True,
    memory_guard_min_free_gb: float = 3.0,
    time_guard_enabled: bool = True,
    max_candidate_seconds: float = 3600.0,
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
    resolved_baselines = normalize_baseline_mode(baseline_mode)
    if resolved_baselines is None:
        clean_baseline: Any = None
    elif resolved_baselines == CANONICAL_BASELINES and isinstance(baseline_mode, str) and baseline_mode.strip().lower() == "all":
        clean_baseline = "all"
    elif len(resolved_baselines) == 1 and isinstance(baseline_mode, str):
        clean_baseline = resolved_baselines[0]
    else:
        clean_baseline = list(resolved_baselines)
    clean_downsample_mode = (
        None
        if downsample_mode in (None, False, "None", "none", "null", "")
        else str(downsample_mode).strip().lower()
    )
    clean_val_downsample_mode = (
        None
        if val_downsample_mode in (None, False, "None", "none", "null", "")
        else str(val_downsample_mode).strip().lower()
    )
    cv_split_val = float(cv_split)
    downsample_frac_val = float(downsample_frac)
    val_downsample_frac_raw = None if val_downsample_frac is None else float(val_downsample_frac)

    if clean_search not in ("grid", "bayes"):
        raise ValueError("search_mode must be 'grid' or 'bayes'")
    if clean_score_version not in VALID_SCORE_VERSIONS:
        raise ValueError(f"score_version must be one of {VALID_SCORE_VERSIONS}, got {score_version!r}")
    if clean_cv not in ("timeseries", "purged_kfold", "cpcv"):
        raise ValueError(
            "cv_strategy must be 'timeseries', 'purged_kfold', or 'cpcv' "
            "(unpurged/non-temporal CV strategies rejected to prevent leakage)"
        )
    if not (0.0 < cv_split_val < 1.0):
        raise ValueError(f"cv_split must be strictly in (0.0, 1.0), got {cv_split!r}")
    if clean_downsample_mode not in VALID_DOWNSAMPLE_MODES:
        raise ValueError(
            f"downsample_mode must be a whole-era mode in {VALID_DOWNSAMPLE_MODES} "
            f"(row-level/random modes like 'within_era' and 'random_eras' are unsupported so cross-sectional "
            f"numerai_corr, per-era FNC, and ticker/time-series continuity remain valid), got {downsample_mode!r}"
        )
    if not (0.0 < downsample_frac_val <= 1.0):
        raise ValueError(f"downsample_frac must be in (0.0, 1.0], got {downsample_frac!r}")
    if clean_val_downsample_mode not in VALID_VAL_DOWNSAMPLE_MODES:
        raise ValueError(
            f"val_downsample_mode must be a whole-era mode in {VALID_VAL_DOWNSAMPLE_MODES} "
            f"(row-level/random modes like 'within_era' and 'random_eras' are unsupported so cross-sectional "
            f"numerai_corr, per-era FNC, and ticker/time-series continuity remain valid), got {val_downsample_mode!r}"
        )
    if val_downsample_frac_raw is not None and not (0.0 < val_downsample_frac_raw <= 1.0):
        raise ValueError(f"val_downsample_frac must be in (0.0, 1.0], got {val_downsample_frac!r}")
    if clean_val_downsample_mode is None:
        resolved_val_downsample_mode = None
        resolved_val_downsample_frac = 1.0 if val_downsample_frac_raw is None else val_downsample_frac_raw
    else:
        resolved_val_downsample_mode = (
            clean_downsample_mode if clean_val_downsample_mode == "same" else clean_val_downsample_mode
        )
        resolved_val_downsample_frac = (
            downsample_frac_val if val_downsample_frac_raw is None else val_downsample_frac_raw
        )
    if clean_eval not in ("static", "walk_forward", "both"):
        raise ValueError("eval_mode must be 'static', 'walk_forward', or 'both'")
    if clean_wf_win not in ("expanding", "sliding"):
        raise ValueError("walk_forward_window must be 'expanding' or 'sliding'")
    if clean_weight not in (None, "equal_era", "recency_linear", "recency_exp", "uniqueness", "uniqueness_decay"):
        raise ValueError(
            "era_weighting must be None, 'equal_era', 'recency_linear', 'recency_exp', 'uniqueness', or 'uniqueness_decay'"
        )

    horizon = int(target_horizon_eras) if target_horizon_eras is not None else infer_target_horizon_eras(target_col)
    if era_stride is None or (isinstance(era_stride, str) and era_stride.strip().lower() in ("auto", "none", "")):
        resolved_era_stride = max(1, int(horizon))
        era_stride_auto = True
    else:
        if isinstance(era_stride, bool):
            raise ValueError(f"era_stride must be 'auto' or an integer >= 1, got {era_stride!r}")
        try:
            resolved_era_stride = int(era_stride)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"era_stride must be 'auto' or an integer >= 1, got {era_stride!r}") from exc
        if resolved_era_stride < 1:
            raise ValueError(f"era_stride must be >= 1, got {era_stride!r}")
        era_stride_auto = False

    embargo = max(1, int(np.ceil(horizon / resolved_era_stride)))
    resolved_era_tag = era_tag if era_tag is not None else f"all_eras_stride{resolved_era_stride}"
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
        "era_stride": [resolved_era_stride],
        "embargo_eras": [embargo],
        "cv_split": [cv_split_val],
        "downsample_mode": [clean_downsample_mode],
        "downsample_frac": [downsample_frac_val],
        "val_downsample_mode": [resolved_val_downsample_mode],
        "val_downsample_frac": [resolved_val_downsample_frac],
    }

    mode_str = f"baseline_mode={clean_baseline}" if clean_baseline is not None else f"model={model_tag}"
    ds_parts = []
    if clean_downsample_mode is not None and downsample_frac_val < 1.0:
        ds_parts.append(f"train={clean_downsample_mode}@{downsample_frac_val:.2f}")
    if resolved_val_downsample_mode is not None and resolved_val_downsample_frac < 1.0:
        ds_parts.append(f"val={resolved_val_downsample_mode}@{resolved_val_downsample_frac:.2f}")
    ds_str = f" | downsample({', '.join(ds_parts)})" if ds_parts else ""
    stride_str = (
        f"era_stride={resolved_era_stride} (auto-matched to horizon={horizon})"
        if era_stride_auto
        else f"era_stride={resolved_era_stride}"
    )
    print(
        f"Experiment: {experiment_name} | {mode_str} | search_mode={clean_search} | "
        f"cv_strategy={clean_cv} (cv_split={cv_split_val}) | ERA_COL={era_col} | TARGET_COL={target_col} | "
        f"{stride_str} | horizon={horizon} | embargo={embargo}{ds_str}"
    )
    if resolved_era_stride < horizon:
        print(
            f"[Overlap Notice] TARGET_COL='{target_col}' has horizon={horizon} eras while "
            f"era_stride={resolved_era_stride} (~{horizon / resolved_era_stride:.1f}x label overlap). "
            f"Consider era_stride={horizon} or era_weighting='uniqueness'."
        )

    configure_memory_guard(
        enabled=bool(memory_guard_enabled),
        min_free_gb=float(memory_guard_min_free_gb),
    )
    configure_time_guard(
        enabled=bool(time_guard_enabled),
        max_candidate_seconds=float(max_candidate_seconds),
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
        "era_stride": resolved_era_stride,
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
        "val_downsample_mode": resolved_val_downsample_mode,
        "val_downsample_frac": resolved_val_downsample_frac,
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
        "save_plots": bool(save_plots),
        "export_diagnostics": bool(export_diagnostics),
        "error_bin_mode": str(error_bin_mode).strip().lower(),
        "error_pct_bins": error_pct_bins,
        "error_zero_tols": error_zero_tols,
        "error_abs_bins": error_abs_bins,
        "extraction_cache_enabled": bool(extraction_cache_enabled),
        "extraction_cache_max_entries": int(extraction_cache_max_entries),
        "memory_guard_enabled": bool(memory_guard_enabled),
        "memory_guard_min_free_gb": float(memory_guard_min_free_gb),
        "time_guard_enabled": bool(time_guard_enabled),
        "max_candidate_seconds": float(max_candidate_seconds),
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
        cfg["selected_baselines"] = list(resolved_baselines) if resolved_baselines is not None else []
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

    era_corrs_arr = era_corrs.to_numpy(dtype=np.float64) if len(era_corrs) else np.empty(0, dtype=np.float64)
    all_zero_era_corrs = bool(len(era_corrs_arr) == 0 or float(np.max(np.abs(era_corrs_arr))) <= 1e-12)
    is_const_pred = bool(_is_constant_1d(preds) or all_zero_era_corrs)
    if is_const_pred:
        _, effective_version, mmc_fallback = compute_versioned_candidate_score(
            era_corrs,
            era_mmcs=fold_mmcs,
            score_version=score_version,
        )
        score_val = -1.0
        mean_corr = 0.0
        std_corr = 0.0
        spearman_mean = 0.0
        sharpe_val = 0.0
        pos_era_pct = 0.0
        era_consistency_pct = 0.0
    else:
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
        era_consistency_pct = float(np.mean(era_corrs.to_numpy(dtype=np.float64) > 0.002) * 100.0) if len(era_corrs) else 0.0

    if len(targets):
        mae_val = float(mean_absolute_error(targets, preds))
        bias_val = float(np.mean(targets - preds))
        in_unit_scale = (float(np.nanmin(preds)) >= -0.05) and (float(np.nanmax(preds)) <= 1.05) and (0.15 <= float(np.mean(preds)) <= 0.85)
        pred_center = 0.5 if in_unit_scale else float(np.nanmedian(preds))
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
        "is_constant_pred": bool(is_const_pred),
        "score_value": float(score_val),
        "numerai_corr": mean_corr,
        "era_corr_std": std_corr,
        "positive_era_pct": pos_era_pct,
        "era_consistency_pct": era_consistency_pct,
        "directional_accuracy_pct": dir_acc_pct,
        "sharpe": sharpe_val,
        "spearman": spearman_mean,
        "mae": mae_val,
        "forecast_bias": bias_val,
        "corr_drawdown": corr_dd,
        "era_corrs": era_corrs,
        **bin_shares,
    })

    fold_pnl = (
        simulate_stake_pnl(era_corrs)
        if len(era_corrs) > 0
        else {"end_capital": 1000.0, "total_return_pct": 0.0, "money_drawdown_pct": 0.0}
    )
    acc_pct = float(bin_shares.get("accurate_pct", np.nan))
    acc_tag = f", Acc {acc_pct:.1f}%" if np.isfinite(acc_pct) else ""
    const_tag = " [CONSTANT PRED PENALIZED -> -1.0000]" if is_const_pred else ""
    print(
        f"  [Fold Score | {effective_version}] Score: {score_val:.4f}{const_tag} | "
        f"Science: CORR {mean_corr * 100.0:+.2f}% (std={std_corr:.4f}), Sharpe {sharpe_val:.2f}, "
        f"CORR DD {corr_dd:.4f}, MAE {mae_val:.4f} | "
        f"Business: £{fold_pnl['end_capital']:,.2f} ({fold_pnl['total_return_pct']:+.2f}%, "
        f"Dip {fold_pnl['money_drawdown_pct']:.2f}%), Pos Eras {pos_era_pct:.1f}%, "
        f"Dir Acc {dir_acc_pct:.1f}%{acc_tag}"
    )
    # Per-iteration / per-fold teardown: release heavy CV fold state on the scored trial clone and run GC
    for _attr in ("_wf_train_X_", "_wf_train_y_", "_wf_train_eras_"):
        if hasattr(inner_est, _attr):
            setattr(inner_est, _attr, None)
    del preds, targets
    gc.collect()
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

    contrib_series = merged.groupby(era_col, sort=True).apply(
        partial(_compute_single_era_contribution, ref_col_name=ref_col_name, target_col=target_col),
        include_groups=False,
    )
    return contrib_series.astype(np.float64).rename("bmc_proxy" if use_proxy else "mmc")


def _compute_single_era_contribution(
    era_frame: pd.DataFrame,
    *,
    ref_col_name: str,
    target_col: str,
) -> float:
    """Compute finite Numerai correlation contribution (MMC/BMC proxy) for a single era group."""
    p = era_frame["prediction"].to_numpy(dtype=np.float64, copy=False)
    m = era_frame[ref_col_name].to_numpy(dtype=np.float64, copy=False)
    t = era_frame[target_col].to_numpy(dtype=np.float64, copy=False)
    if len(t) < 2 or _is_constant_1d(p) or _is_constant_1d(m) or _is_constant_1d(t):
        return 0.0
    val = float(
        correlation_contribution(
            era_frame[["prediction"]],
            era_frame[ref_col_name].rename("numerai_meta_model"),
            era_frame[target_col],
        ).to_numpy(dtype=np.float64)[0]
    )
    return val if np.isfinite(val) else 0.0


R1393_START_ERA = 1393
R1399_START_ERA = 1399
R1393_START_DATE = pd.Timestamp("2026-11-06")
R1399_START_DATE = pd.Timestamp("2026-11-14")
CLASSIC_CONCURRENT_ROUNDS = 64
CLASSIC_SETTLEMENT_ROUNDS = 12


def _resolve_era_timestamps(
    idx: pd.Index,
    era_dates: Optional[Union[pd.Series, Mapping[Any, Any]]] = None,
) -> Optional[pd.DatetimeIndex]:
    """Align optional calendar dates (Mapping/Series `era -> date`, or DatetimeIndex) onto `idx`."""
    if era_dates is not None:
        if isinstance(era_dates, Mapping):
            mapped = pd.Index([era_dates.get(k, pd.NaT) for k in idx])
            ts = pd.to_datetime(mapped, errors="coerce")
        else:
            s = pd.Series(era_dates)
            if len(s) == len(idx) and s.index.equals(idx):
                ts = pd.to_datetime(s, errors="coerce")
            else:
                aligned = s.reindex(idx)
                ts = pd.to_datetime(aligned, errors="coerce")
        if bool(pd.notna(ts).any()):
            return pd.DatetimeIndex(ts)
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    # Detect ISO date-string indices (never coerce small integers as epoch nanoseconds!)
    if not pd.api.types.is_numeric_dtype(idx):
        parsed = pd.to_datetime(idx, errors="coerce")
        if bool(pd.notna(parsed).any()) and int(parsed.dropna().year.min()) >= 1990:
            return pd.DatetimeIndex(parsed)
    return None


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
    era_dates: Optional[Union[pd.Series, Mapping[Any, Any]]] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Compute chronologically sorted per-round Numerai Classic payout returns using era- and date-accurate rules.

    Official rules (`payout_rule='auto'`):
    - Pre-round 1393 (`era < 1393` and `date < 2026-11-06`): `clip(3 * CORR60 + 15 * MMC60, -1.0, +1.0)`.
    - Round 1393+ (`era >= 1393` OR `date >= 2026-11-06`): `clip(payout_factor * (8 * NCORR + 32 * NMMC), -1.0, +2.0)`.
    - Round 1399+ (`era >= 1399` OR `date >= 2026-11-14`): if `neutral_churn` is provided, positive payouts are
      multiplied by `min(1, 2 / (1 + exp(10 * (mean_neutral_churn - 0.1))))` after multipliers and clipping;
      burns stay unchanged. Rounds 1393–1398 have no churn penalty.

    Supports both official round indices (e.g. 1393+) AND remapped v6.0 sequential eras (`1..N`) when
    `era_dates` (era -> calendar date mapping/Series) is provided.
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
    resolved_ts = _resolve_era_timestamps(idx, era_dates=era_dates)
    clean_rule = str(payout_rule or "auto").strip().lower()
    if clean_rule not in ("auto", "pre_1393", "r1393"):
        raise ValueError(f"payout_rule must be 'auto', 'pre_1393', or 'r1393', got {payout_rule!r}")

    date_is_r1393 = (
        np.asarray(pd.notna(resolved_ts) & (resolved_ts >= R1393_START_DATE), dtype=bool)
        if resolved_ts is not None
        else np.zeros(n_eras, dtype=bool)
    )
    date_is_r1399 = (
        np.asarray(pd.notna(resolved_ts) & (resolved_ts >= R1399_START_DATE), dtype=bool)
        if resolved_ts is not None
        else np.zeros(n_eras, dtype=bool)
    )

    if clean_rule == "pre_1393":
        is_r1393_mask = np.zeros(n_eras, dtype=bool)
    elif clean_rule == "r1393":
        is_r1393_mask = np.ones(n_eras, dtype=bool)
    else:
        is_r1393_mask = (np.isfinite(era_nums) & (era_nums >= float(R1393_START_ERA))) | date_is_r1393

    is_r1399_mask = (np.isfinite(era_nums) & (era_nums >= float(R1399_START_ERA))) | date_is_r1399

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

    # 3. Settle remaining open positions step-by-step chronologically so intermediate unwind drawdowns
    #    are faithfully tracked (no artificial lump-sum masking at the end of the validation period)
    unwind_equities: List[float] = []
    if not wiped_out and open_positions:
        open_positions.sort(key=itemgetter(0))
        remaining_locked = sum(pos[2] for pos in open_positions)
        for _, o_step, pos_stake, pos_ret in open_positions:
            payout = max(0.0, pos_stake * (1.0 + pos_ret))
            free_cash += payout
            remaining_locked -= pos_stake
            round_pnls[o_step] = payout - pos_stake
            step_eq = max(0.0, free_cash + max(0.0, remaining_locked))
            if step_eq <= 1e-12:
                wiped_out = True
                free_cash = 0.0
                step_eq = 0.0
            unwind_equities.append(step_eq)
        open_positions.clear()
        final_eq = 0.0 if wiped_out else max(0.0, free_cash)
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

    # Track true money drawdown including intermediate terminal unwind steps (always in [-100, 0])
    full_eval_path = (
        np.concatenate([equity_path[:-1], np.asarray(unwind_equities, dtype=np.float64)])
        if unwind_equities and not bool(ruin_mask[-1])
        else equity_path
    )
    full_eval_path = np.maximum(0.0, full_eval_path)
    full_ruin = np.maximum.accumulate(full_eval_path <= 1e-12)
    full_eval_path = np.where(full_ruin, 0.0, full_eval_path)
    running_peak = np.maximum.accumulate(np.maximum(init_cap, full_eval_path))
    dd_val = full_eval_path - running_peak
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
    era_dates: Optional[Union[pd.Series, Mapping[Any, Any]]] = None,
) -> Dict[str, Any]:
    """One-wallet Numerai Classic overlapping-rounds staking simulator (no window resets, no margin).

    - Starts with one wallet at `total_capital` (default £1,000, or `initial_stake` alias).
    - Sizes each weekly round at `TVL / R` (`R = 64` concurrent rounds for Classic, or `R = 12` for
      100% weekly capital utilization via `simulate_live_staking_scenarios`); locked capital stays
      locked for `settlement_rounds` (~60 trading days = 12 weekly eras) and cannot be reused.
    - Uses era- and date-accurate Numerai Classic rules (`payout_rule='auto'`, passing `era_dates` for v6.0):
        * Pre-1393 (`< 2026-11-06`): `3 * CORR60 + 15 * MMC60`, clipped `[-100%, +100%]`.
        * R1393+ (`>= 2026-11-06`): `8 * NCORR + 32 * NMMC`, clipped `[-100%, +200%]`.
        * R1399+ (`>= 2026-11-14`): neutral churn retention `min(1, 2 / (1 + exp(10 * (churn - 0.1))))` on positive payouts only.
    - Keeps NMR and GBP separate: simulates staking in NMR, marks equity to market via `nmr_gbp_prices`
      and converts realized round PnL using settlement-week FX (`t + settlement_rounds`).
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
        era_dates=era_dates,
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

    eq_nmr, round_pnls_nmr, nmr_dd_pct, max_dd_nmr, worst_round_nmr, wiped_out = _simulate_single_wallet_path(
        ret_arr,
        start_nmr,
        staking_mode=clean_mode,
        concurrent_rounds=concurrent_rounds,
        settlement_rounds=settlement_rounds,
    )

    eq_gbp = np.maximum(0.0, eq_nmr * fx_arr)
    # Convert realized round PnL at settlement-week FX (t + settlement_rounds) rather than open-week FX
    settle_lag = max(1, int(settlement_rounds))
    settle_fx_idx = np.minimum(len(fx_arr) - 1, np.arange(len(fx_arr), dtype=np.int64) + settle_lag)
    round_pnls_gbp = round_pnls_nmr * fx_arr[settle_fx_idx]

    # Wallet drawdown in GBP (always in [-100, 0]; incorporate unwind drawdown when FX is constant)
    running_peak_gbp = np.maximum.accumulate(np.maximum(start_cap_gbp, eq_gbp))
    dd_gbp_arr = eq_gbp - running_peak_gbp
    dd_pct_arr = np.where(running_peak_gbp > 0.0, (dd_gbp_arr / running_peak_gbp) * 100.0, 0.0)
    dd_pct_arr = np.clip(dd_pct_arr, -100.0, 0.0)

    end_cap_gbp = float(eq_gbp[-1])
    total_profit_gbp = end_cap_gbp - start_cap_gbp
    total_return_pct = (total_profit_gbp / start_cap_gbp) * 100.0 if start_cap_gbp > 1e-12 else 0.0
    const_fx = bool(np.allclose(fx_arr, fx_arr[0]))
    money_dd_pct = float(min(float(dd_pct_arr.min()), float(nmr_dd_pct))) if const_fx else float(dd_pct_arr.min())
    max_dd_gbp = float(min(float(dd_gbp_arr.min()), float(max_dd_nmr * fx_arr[0]))) if const_fx else float(dd_gbp_arr.min())
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
    block_size: int = 12,
    random_state: int = 42,
    title: str = "Model",
    era_dates: Optional[Union[pd.Series, Mapping[Any, Any]]] = None,
    retain_paths: bool = True,
) -> Dict[str, Any]:
    """One-wallet circular block-bootstrap Monte Carlo simulator under Numerai Classic overlapping-rounds staking.

    Uses circular block resampling (`block_size=12` default matching 60D target overlap so boundary eras
    are sampled with equal probability and autocorrelation structure is preserved) and simulates one continuous
    wallet per path (no resets, no margin, equity >= 0).
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
        era_dates=era_dates,
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
            "sim_equity_paths": np.empty((0, 0), dtype=np.float64),
            "actual_equity_path": np.empty(0, dtype=np.float64),
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

    # Circular block bootstrap so first/last eras have exact uniform sampling probability 1/n_t
    starts = rng.integers(0, n_t, size=(n_s, n_blocks))
    offsets = np.arange(b_size, dtype=np.int64)[None, None, :]
    boot_idx = ((starts[:, :, None] + offsets) % n_t).reshape(n_s, -1)[:, :n_t]
    sim_rets = ret_arr[boot_idx]

    final_wallets = np.zeros(n_s, dtype=np.float64)
    worst_dds_pct = np.zeros(n_s, dtype=np.float64)
    ruin_flags = np.zeros(n_s, dtype=bool)
    sim_paths = np.zeros((n_s, n_t), dtype=np.float64) if retain_paths else np.empty((0, 0), dtype=np.float64)

    for s_idx in range(n_s):
        eq_path, _, dd_pct, _, _, wiped = _simulate_single_wallet_path(
            sim_rets[s_idx],
            start_cap,
            staking_mode=clean_mode,
            concurrent_rounds=concurrent_rounds,
            settlement_rounds=settlement_rounds,
        )
        if retain_paths:
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
        if retain_paths:
            loss_eq = sim_paths[pick_loss]
        else:
            loss_eq, _, _, _, _, _ = _simulate_single_wallet_path(
                sim_rets[pick_loss],
                start_cap,
                staking_mode=clean_mode,
                concurrent_rounds=concurrent_rounds,
                settlement_rounds=settlement_rounds,
            )
        example_losing_path = _extract_path_drawdown_markers(loss_eq, start_cap)
    if len(winning_indices) > 0:
        med_win_target = float(np.median(final_wallets[winning_indices]))
        pick_win = int(winning_indices[int(np.argmin(np.abs(final_wallets[winning_indices] - med_win_target)))])
        if retain_paths:
            win_eq = sim_paths[pick_win]
        else:
            win_eq, _, _, _, _, _ = _simulate_single_wallet_path(
                sim_rets[pick_win],
                start_cap,
                staking_mode=clean_mode,
                concurrent_rounds=concurrent_rounds,
                settlement_rounds=settlement_rounds,
            )
        example_winning_path = _extract_path_drawdown_markers(win_eq, start_cap)

    del starts, offsets, boot_idx, sim_rets
    gc.collect()

    p05_val = float(np.percentile(final_wallets, 5.0))
    med_val = float(np.percentile(final_wallets, 50.0))
    mean_val = float(np.mean(final_wallets))
    p95_val = float(np.percentile(final_wallets, 95.0))
    prob_profit = float(np.mean(final_wallets > start_cap + 1e-9) * 100.0)
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
        "sim_equity_paths": sim_paths,
        "actual_equity_path": actual_eq,
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
    conservative_pnl: Optional[Dict[str, Any]] = None,
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
    era_dates: Optional[Union[pd.Series, Mapping[Any, Any]]] = None,
    save_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Plot a clean ONE-WALLET staking equity curve over the full validation period (optionally overlaying Conservative Floor)."""
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
        era_dates=era_dates,
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

    has_cons = (
        isinstance(conservative_pnl, dict)
        and isinstance(conservative_pnl.get("equity_curve"), pd.Series)
        and len(conservative_pnl["equity_curve"]) > 0
    )
    primary_label = f"Realistic Live ({mode_label}, £)" if has_cons else f"One Wallet ({mode_label}, £)"
    ax.plot(x_vals, y_vals, color=line_color, linewidth=2.3, label=primary_label)
    if has_cons and conservative_pnl is not None:
        cons_eq = conservative_pnl["equity_curve"]
        ax.plot(
            cons_eq.index.to_numpy(),
            cons_eq.to_numpy(dtype=np.float64),
            color="#2563eb",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.9,
            label="Conservative Floor (CORR only, £)",
        )
    ax.axhline(start_cap, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.85, label=f"Start (£{start_cap:,.0f})")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Time (Validation Era / Week)")
    ax.set_ylabel("Wallet Balance (£)")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=9.2)

    if has_cons and conservative_pnl is not None:
        c_end = float(conservative_pnl["end_capital"])
        c_ret = float(conservative_pnl["total_return_pct"])
        c_dd = float(conservative_pnl["money_drawdown_pct"])
        headline_text = (
            f"Realistic Live: £{start_cap:,.0f} -> £{end_cap:,.2f} ({pnl['total_return_pct']:+.2f}%) | Dip: {dd_pct:.2f}% (£{dd_gbp:,.2f})\n"
            f"Conservative Floor (CORR only): £{start_cap:,.0f} -> £{c_end:,.2f} ({c_ret:+.2f}%) | Dip: {c_dd:.2f}%"
        )
        headline_fs = 10.5
        rect_top = 0.835
    else:
        headline_text = (
            f"Start: £{start_cap:,.0f}     |     "
            f"End: £{end_cap:,.2f} ({pnl['total_return_pct']:+.2f}%)     |     "
            f"Worst peak-to-trough dip: {dd_pct:.2f}% (£{dd_gbp:,.2f})"
        )
        headline_fs = 12.0
        rect_top = 0.865

    fig.suptitle(
        f"One-Wallet Staking ({mode_label}) — {title}  [{pnl['score_basis']}]",
        fontsize=13.0,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.93,
        headline_text,
        ha="center",
        va="top",
        fontsize=headline_fs,
        fontweight="bold",
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    pf_label = f"{float(payout_factor):g}" if not isinstance(payout_factor, pd.Series) else "series"
    fig.text(
        0.5,
        0.02,
        f"Basis: {pnl['score_basis']} | PF={pf_label} | Worst single round: £{worst_round:+,.2f} | "
        f"Stake: TVL/{concurrent_rounds} (£{pnl['per_round_stake_initial']:.2f} initial, {settlement_rounds}-era lock, {mode_label}) | "
        f"{pnl['payout_rule_summary']} | FX: {pnl['fx_status']}",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#475569",
    )
    plt.tight_layout(rect=(0.0, 0.055, 1.0, rect_top))
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=140)
    plt.show()
    plt.close(fig)
    del fig
    gc.collect()
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
    block_size: int = 12,
    random_state: int = 42,
    era_dates: Optional[Union[pd.Series, Mapping[Any, Any]]] = None,
    save_path: Optional[Any] = None,
    retain_paths: bool = True,
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
        era_dates=era_dates,
        retain_paths=True,
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

    fig = plt.figure(figsize=(16.5, 11.4), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.08], hspace=0.34, wspace=0.18)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_paths = fig.add_subplot(gs[0, 1])
    ax_fan = fig.add_subplot(gs[1, :])
    for ax in (ax_hist, ax_paths, ax_fan):
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

    if lose_ex is not None:
        _plot_monte_carlo_example_path(ax_paths, lose_ex, "#dc2626", "Example Losing Path")
    if win_ex is not None:
        _plot_monte_carlo_example_path(ax_paths, win_ex, "#059669", "Example Winning Path")
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

    # --- Bottom Full-Width Panel: Cleaner Monte Carlo Simulation Graph (Cumulative P&L £, can go negative) ---
    from matplotlib.ticker import FuncFormatter

    sim_eq_paths = np.asarray(mc.get("sim_equity_paths", np.empty((0, 0))), dtype=np.float64)
    actual_eq_path = np.asarray(mc.get("actual_equity_path", np.empty(0)), dtype=np.float64)
    if sim_eq_paths.ndim == 2 and sim_eq_paths.shape[0] > 0 and sim_eq_paths.shape[1] > 0:
        # Convert wallet equity to Cumulative P&L (£) relative to start_cap so losses go negative (< £0)
        pnl_paths = np.hstack([np.zeros((sim_eq_paths.shape[0], 1), dtype=np.float64), sim_eq_paths - start_cap])
        steps_fan = np.arange(pnl_paths.shape[1], dtype=np.int64)

        # Faded neutral background paths to show full range without visual clutter
        ax_fan.plot(steps_fan, pnl_paths.T, color="#94a3b8", alpha=0.14, linewidth=0.75, zorder=1)
        ax_fan.plot([], [], color="#94a3b8", alpha=0.55, linewidth=1.2, label=f"All {mc['n_sims']:,} Simulated Paths")

        p25_curve = np.percentile(pnl_paths, 25.0, axis=0)
        p50_curve = np.percentile(pnl_paths, 50.0, axis=0)
        p75_curve = np.percentile(pnl_paths, 75.0, axis=0)

        p25_end, p50_end, p75_end = float(p25_curve[-1]), float(p50_curve[-1]), float(p75_curve[-1])

        ax_fan.axhline(0.0, color="#0f172a", linestyle="--", linewidth=1.3, zorder=3, label="Break-even (£0 P&L)")
        if len(actual_eq_path) == sim_eq_paths.shape[1]:
            actual_pnl_curve = np.concatenate(([0.0], actual_eq_path - start_cap))
            actual_pnl_end = float(actual_pnl_curve[-1])
            ax_fan.plot(
                steps_fan,
                actual_pnl_curve,
                color="#7c3aed",
                linestyle="-.",
                linewidth=1.5,
                alpha=0.85,
                zorder=4,
                label=f"Actual Path (£{actual_pnl_end:+,.2f})",
            )

        ax_fan.plot(steps_fan, p75_curve, color="#059669", linewidth=2.2, zorder=5, label=f"75th Percentile (£{p75_end:+,.2f})")
        ax_fan.plot(steps_fan, p50_curve, color="#1d4ed8", linewidth=2.6, zorder=6, label=f"Median / 50th Pct (£{p50_end:+,.2f})")
        ax_fan.plot(steps_fan, p25_curve, color="#dc2626", linewidth=2.2, zorder=5, label=f"25th Percentile (£{p25_end:+,.2f})")

        y_min = float(np.min(pnl_paths))
        y_max = float(np.max(pnl_paths))
        y_span = max(y_max - y_min, start_cap * 0.05)
        y_lo = min(y_min, 0.0) - 0.08 * y_span
        y_hi = max(y_max, 0.0) + 0.08 * y_span
        if y_lo < 0.0:
            ax_fan.axhspan(y_lo, 0.0, color="#fee2e2", alpha=0.22, zorder=0)
        ax_fan.set_ylim(y_lo, y_hi)
        ax_fan.set_xlim(0, max(1, int(steps_fan[-1])) * 1.07)

        for y_end, color, tag in (
            (p75_end, "#059669", "75th"),
            (p50_end, "#1d4ed8", "Median"),
            (p25_end, "#dc2626", "25th"),
        ):
            ax_fan.annotate(
                f"{tag}: £{y_end:+,.0f}",
                xy=(steps_fan[-1], y_end),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8.2,
                fontweight="bold",
                color=color,
            )

        ax_fan.yaxis.set_major_formatter(FuncFormatter(_format_signed_gbp_tick))
        ax_fan.set_xlabel("Simulated Week (Era Step)")
        ax_fan.set_ylabel("Cumulative P&L (£)")
        ax_fan.set_title(
            f"Cleaner Monte Carlo Simulation Paths — Cumulative P&L (£) Across {mc['n_sims']:,} Runs (25th / Median / 75th)",
            fontsize=11.5,
            fontweight="bold",
            pad=8,
        )
        ax_fan.legend(loc="upper left", ncol=3, frameon=True, facecolor="white", framealpha=0.92, fontsize=8.3)
        del pnl_paths

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
    fig.subplots_adjust(left=0.06, right=0.95, top=0.835, bottom=0.075)
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=140)
    plt.show()
    plt.close(fig)
    del fig
    gc.collect()
    if not retain_paths:
        mc["sim_equity_paths"] = np.empty((0, 0), dtype=np.float64)
    del sim_eq_paths
    gc.collect()
    return mc


def _plot_monte_carlo_example_path(
    ax_paths: Any,
    example_path: Dict[str, Any],
    color: str,
    tag: str,
) -> None:
    """Plot an annotated Monte Carlo example equity path showing peak, trough dip, and final balance."""
    arr = np.asarray(example_path["equity_path"], dtype=np.float64)
    steps = np.arange(len(arr))
    pk_i, pk_v = int(example_path["peak_idx"]), float(example_path["peak_val"])
    tr_i, tr_v = int(example_path["trough_idx"]), float(example_path["trough_val"])
    fn_v = float(example_path["final_val"])
    dd_v = float(example_path["max_drawdown_pct"])
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


def _format_signed_gbp_tick(value: float, _tick_pos: Any) -> str:
    """Format a signed GBP currency axis tick allowing negative cumulative P&L values."""
    if value < -1e-6:
        return f"-£{abs(value):,.0f}"
    if value > 1e-6:
        return f"+£{value:,.0f}"
    return "£0"


def _extract_cv_split_index(col_name: str) -> int:
    """Extract the integer fold index from a scikit-learn `split<N>_test_score` column name."""
    return int(col_name[len("split") : col_name.index("_test_score")])


def _strip_estimator_param_prefix(col_name: Any) -> str:
    """Strip a leading `estimator__` prefix from a CV parameter column name."""
    name_str = str(col_name)
    return name_str[len("estimator__") :] if name_str.startswith("estimator__") else name_str


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
        key=_extract_cv_split_index,
    )
    trials_era_corrs_map: Dict[int, pd.Series] = {}
    if _CV_FOLD_METRIC_LOG and split_cols:
        expected_entries = len(df) * len(split_cols)
        if len(_CV_FOLD_METRIC_LOG) < expected_entries:
            print(
                f"  [CV Metric Log Notice] Captured {len(_CV_FOLD_METRIC_LOG)}/{expected_entries} fold records "
                "(multi-process n_jobs != 1 does not share in-memory _CV_FOLD_METRIC_LOG across workers; "
                "use n_jobs=1 for complete per-trial OOF matrices)."
            )
        metric_specs = (
            ("mean_numerai_corr", "numerai_corr", 4),
            ("mean_era_corr_std", "era_corr_std", 4),
            ("mean_positive_era_pct", "positive_era_pct", 2),
            ("mean_era_consistency_pct", "era_consistency_pct", 2),
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
                trials_era_corrs_map[row_idx] = stitched_oof
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
        params_df = pd.DataFrame(list(df["params"])).rename(columns=_strip_estimator_param_prefix)
        drop_cols = ["params"] + param_prefixed
        df = pd.concat([df.drop(columns=drop_cols, errors="ignore"), params_df], axis=1)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.insert(0, "candidate_id", np.arange(len(df), dtype=int))
    if "rank_test_score" in df.columns:
        df = df.sort_values("rank_test_score", kind="stable").reset_index(drop=True)
    # Surface Business (P&L + accuracy) and Science (CORR + risk) columns first for display()
    front_cols = [
        c
        for c in (
            "candidate_id",
            "rank_test_score",
            "mean_test_score",
            "score_version",
            "end_capital",
            "total_profit",
            "total_return_pct",
            "money_drawdown_pct",
            "mean_directional_accuracy_pct",
            "mean_accurate_pct",
            "mean_numerai_corr",
            "mean_sharpe",
            "mean_positive_era_pct",
            "mean_era_consistency_pct",
            "mean_corr_drawdown",
            "mean_mae",
        )
        if c in df.columns
    ]
    remaining_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + remaining_cols]
    if trials_era_corrs_map:
        trials_df = pd.DataFrame({f"trial_{cid}": s for cid, s in sorted(trials_era_corrs_map.items())}).sort_index()
        # Store as plain nested dict so pandas DataFrame.attrs == attrs in to_string()/concat() never raises ValueError
        df.attrs["trials_era_corrs_df"] = trials_df.to_dict(orient="dict")
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
            under_count=("under", "sum"),
            over_count=("over", "sum"),
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
    for col in ("data_points", "under_share", "over_share", "under_count", "over_count", "mae", "mean_residual"):
        envelope_df[col] = envelope_df[col].interpolate(method="index", limit_area="inside")
    return envelope_df


def _find_envelope_peak_era(
    era_index: np.ndarray,
    total_counts: np.ndarray,
    shares: np.ndarray,
    peak_counts: np.ndarray,
    is_measured: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Find the single measured validation era with the highest forecast error share (tie-broken by count)."""
    measured_pos = np.flatnonzero(is_measured)
    if len(measured_pos) == 0:
        return None
    m_shares = np.nan_to_num(shares[measured_pos], nan=-1.0)
    m_counts = np.nan_to_num(peak_counts[measured_pos], nan=-1.0)
    max_share = float(np.max(m_shares))
    tied_mask = np.abs(m_shares - max_share) <= 1e-12
    tied_local = np.flatnonzero(tied_mask)
    best_local = int(tied_local[int(np.argmax(m_counts[tied_local]))])
    pos = int(measured_pos[best_local])
    return {
        "pos": pos,
        "era": int(era_index[pos]),
        "share_pct": float(shares[pos] * 100.0),
        "count": int(round(float(peak_counts[pos]))),
        "total": int(round(float(total_counts[pos]))),
    }


def _annotate_envelope_peak(
    ax: Any,
    *,
    era_id: int,
    y_val: float,
    label_text: str,
    marker_color: str,
    bg_color: str,
    text_color: str,
    is_upper: bool,
    era_min: int,
    era_max: int,
) -> None:
    """Plot a bold peak marker and non-overlapping callout box on the error-direction envelope."""
    span = max(1, era_max - era_min)
    rel_x = float(era_id - era_min) / float(span)
    if rel_x < 0.28:
        ha_align, x_off = "left", 10
    elif rel_x > 0.72:
        ha_align, x_off = "right", -10
    else:
        ha_align, x_off = "center", 0
    y_off = 24 if is_upper else -28
    va_align = "bottom" if is_upper else "top"
    marker_shape = "^" if is_upper else "v"

    ax.scatter(
        [era_id],
        [y_val],
        color=marker_color,
        edgecolors="white",
        linewidths=1.4,
        marker=marker_shape,
        s=105,
        zorder=7,
    )
    ax.annotate(
        label_text,
        xy=(era_id, y_val),
        xytext=(x_off, y_off),
        textcoords="offset points",
        ha=ha_align,
        va=va_align,
        fontsize=8.6,
        fontweight="bold",
        color=text_color,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": bg_color, "edgecolor": marker_color, "linewidth": 1.2, "alpha": 0.96},
        arrowprops={"arrowstyle": "->", "color": marker_color, "linewidth": 1.3},
        zorder=8,
    )


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
    """Plot the directional error envelope across Numerai eras with max overforecast/underforecast peak callouts."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    PURPLE = "#7446A5"
    BLUE = "#5C9ED6"
    DEEP_BLUE = "#1d4ed8"
    ORANGE = "#EEA04B"
    DEEP_ORANGE = "#c2410c"

    envelope_df = compute_era_error_envelope(y_true, y_pred, eras)
    era_index = envelope_df.index.to_numpy(dtype=np.int32)
    centre = envelope_df["data_points"].to_numpy(dtype=np.float64)
    under_share = envelope_df["under_share"].to_numpy(dtype=np.float64)
    over_share = envelope_df["over_share"].to_numpy(dtype=np.float64)
    under_count = envelope_df["under_count"].to_numpy(dtype=np.float64)
    over_count = envelope_df["over_count"].to_numpy(dtype=np.float64)
    is_interpolated = envelope_df["is_interpolated"].to_numpy(dtype=bool)
    is_measured = ~is_interpolated

    value_span = float(np.nanmax(centre) - np.nanmin(centre))
    fallback_span = max(float(np.nanmax(centre)) * 0.25, 1.0)
    scale = float(band_size) * (value_span if value_span > 1e-9 else fallback_span)
    lower = centre - scale * under_share
    upper = centre + scale * over_share
    theo_upper = centre + scale
    theo_lower = centre - scale

    fig, ax = plt.subplots(figsize=(17.5, 6.8), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(era_index, theo_upper, color="#94a3b8", linestyle=":", linewidth=0.9, alpha=0.65, zorder=1)
    ax.plot(era_index, theo_lower, color="#94a3b8", linestyle=":", linewidth=0.9, alpha=0.65, zorder=1)
    ax.fill_between(era_index, centre, upper, color=BLUE, alpha=0.45, linewidth=0, zorder=2)
    ax.fill_between(era_index, lower, centre, color=ORANGE, alpha=0.45, linewidth=0, zorder=2)
    ax.plot(era_index, upper, color=BLUE, linewidth=1.3, zorder=3)
    ax.plot(era_index, lower, color=ORANGE, linewidth=1.3, zorder=3)
    ax.plot(era_index, centre, color=PURPLE, linestyle=":", linewidth=2.3, zorder=4)

    ax.scatter(era_index[is_measured], centre[is_measured], color=PURPLE, s=22, zorder=5)
    if np.any(is_interpolated):
        ax.scatter(
            era_index[is_interpolated],
            centre[is_interpolated],
            facecolors="white",
            edgecolors=PURPLE,
            linewidths=1.5,
            s=45,
            zorder=6,
        )

    n_measured = int(is_measured.sum())
    min_era, max_era = int(era_index[0]), int(era_index[-1])

    max_over_peak = _find_envelope_peak_era(era_index, centre, over_share, over_count, is_measured)
    max_under_peak = _find_envelope_peak_era(era_index, centre, under_share, under_count, is_measured)

    if max_over_peak is not None:
        ov_pos = max_over_peak["pos"]
        _annotate_envelope_peak(
            ax,
            era_id=max_over_peak["era"],
            y_val=float(upper[ov_pos]),
            label_text=(
                f"Max Over: {max_over_peak['share_pct']:.1f}% "
                f"({max_over_peak['count']:,}/{max_over_peak['total']:,}) @ Era {max_over_peak['era']}"
            ),
            marker_color=DEEP_BLUE,
            bg_color="#eff6ff",
            text_color="#1e3a8a",
            is_upper=True,
            era_min=min_era,
            era_max=max_era,
        )

    if max_under_peak is not None:
        un_pos = max_under_peak["pos"]
        _annotate_envelope_peak(
            ax,
            era_id=max_under_peak["era"],
            y_val=float(lower[un_pos]),
            label_text=(
                f"Max Under: {max_under_peak['share_pct']:.1f}% "
                f"({max_under_peak['count']:,}/{max_under_peak['total']:,}) @ Era {max_under_peak['era']}"
            ),
            marker_color=DEEP_ORANGE,
            bg_color="#fff7ed",
            text_color="#9a3412",
            is_upper=False,
            era_min=min_era,
            era_max=max_era,
        )

    env_ymin = float(np.nanmin(theo_lower))
    env_ymax = float(np.nanmax(theo_upper))
    env_span = max(env_ymax - env_ymin, 1.0)
    ax.set_ylim(env_ymin - 0.24 * env_span, env_ymax + 0.26 * env_span)

    title_prefix = "[BEST] " if is_best else ""
    title_color = PURPLE if is_best else "#111111"
    fig.suptitle(
        f"{title_prefix}{title}  \u00b7  {n_measured} validation eras  \u00b7  Era {min_era}\u2013{max_era}",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=17.5,
        fontweight="bold",
        color=title_color,
    )
    if max_over_peak is not None and max_under_peak is not None:
        badge_line = (
            f"Theoretical Max Volume: 100% of era rows  |  "
            f"Observed Peak Over: {max_over_peak['share_pct']:.1f}% ({max_over_peak['count']:,}/{max_over_peak['total']:,}) @ Era {max_over_peak['era']}  |  "
            f"Observed Peak Under: {max_under_peak['share_pct']:.1f}% ({max_under_peak['count']:,}/{max_under_peak['total']:,}) @ Era {max_under_peak['era']}"
        )
        fig.text(
            0.06,
            0.895,
            badge_line,
            ha="left",
            va="top",
            fontsize=8.9,
            fontweight="bold",
            color="#334155",
            bbox={"boxstyle": "round,pad=0.30", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "linewidth": 1.0},
        )
    if is_best:
        fig.patch.set_edgecolor(PURPLE)
        fig.patch.set_linewidth(2.5)
    ax.set_xlabel("Validation Era")
    ax.set_ylabel("Data points per era")
    ax.set_xlim(min_era, max_era)
    if float(np.nanmax(theo_upper)) >= 1000.0:
        ax.yaxis.set_major_formatter(FuncFormatter(_format_thousands_count_tick))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(_format_integer_count_tick))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Era-block / annual (52-era) section dividers
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
                1.01,
                block_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9.5,
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
        Line2D([], [], color=DEEP_BLUE, marker="^", linestyle="None", markersize=7, label="Max overforecast era"),
        Line2D([], [], color=DEEP_ORANGE, marker="v", linestyle="None", markersize=7, label="Max underforecast era"),
        Line2D([], [], color="#94a3b8", linestyle=":", linewidth=1.2, label="100% max volume ref"),
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
        bbox_to_anchor=(1.015, 0.5),
        frameon=False,
        ncol=1,
        handletextpad=0.6,
        labelspacing=0.95,
        fontsize=9.5,
    )

    mean_over_pct = float(np.nanmean(over_share[is_measured]) * 100.0)
    mean_under_pct = float(np.nanmean(under_share[is_measured]) * 100.0)
    fig.text(
        0.06,
        0.025,
        f"Bands show the share of held-out predictions above/below target (mean overforecast {mean_over_pct:.1f}%, "
        f"underforecast {mean_under_pct:.1f}%), not error magnitude. Dotted bounds mark 100% theoretical era-row volume "
        f"(ribbon scaled to {int(round(band_size * 100))}% of era data-point range).",
        fontsize=8.8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.06, right=0.80, top=0.80, bottom=0.15)
    if save_path is not None:
        target_path = Path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, dpi=140)
    plt.show()
    plt.close(fig)
    del fig
    gc.collect()
    return envelope_df


def _format_thousands_count_tick(value: float, _tick_pos: Any) -> str:
    """Format a count axis tick in compact thousands (`k`) notation."""
    return f"{value / 1000.0:g}k"


def _format_integer_count_tick(value: float, _tick_pos: Any) -> str:
    """Format a count axis tick as a comma-separated integer."""
    return f"{int(round(value)):,}"


def _build_error_waterfall_summary_row(
    section: str,
    period: str,
    n_pts: int,
    n_eras_val: int,
    pcts_5: np.ndarray,
    mae_val: float,
) -> Dict[str, Any]:
    """Build one formatted 5-bin error waterfall summary row dictionary."""
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

    q_pcts = (q_bin_counts.astype(np.float64) / np.maximum(q_counts[:, None], 1)) * 100.0
    q_maes = q_abs_sums / np.maximum(q_counts, 1)

    rows: List[Dict[str, Any]] = [
        _build_error_waterfall_summary_row("Overall", "Overall (All Validation)", overall_n, overall_eras, overall_bins, overall_mae),
        _build_error_waterfall_summary_row(
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
                _build_error_waterfall_summary_row(
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
            _build_error_waterfall_summary_row(
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
    """Plot a 2-panel Error-Bin Waterfall (Overall step-down + Financial Quarter diverging stacked bars)."""
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
    plt.close(fig)
    del fig
    gc.collect()


def _format_pct_tick(value: float, _tick_pos: Any) -> str:
    """Format a percentage axis tick as `{value:g}%`."""
    return f"{value:g}%"


def _apply_clean_residual_axes_style(ax: Any) -> None:
    """Apply CAVU flight-deck horizontal grey gridlines and clean top/right spine removal."""
    ax.yaxis.grid(True, color="grey", alpha=0.25)
    ax.set_axisbelow(True)
    try:
        import seaborn as sns

        sns.despine(ax=ax)
    except ImportError:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def _order_residual_variants(
    predictions_by_model: Mapping[str, Any],
    *,
    best_variant_name: Optional[str] = None,
    variant_order: Optional[Sequence[str]] = None,
    max_models: int = 4,
) -> List[str]:
    """Order model variants with `best_variant_name` first, then `variant_order`, then insertion order."""
    ordered: List[str] = []
    if best_variant_name is not None and best_variant_name in predictions_by_model:
        ordered.append(str(best_variant_name))
    candidates = list(variant_order) if variant_order is not None else list(predictions_by_model.keys())
    for name in candidates:
        name_str = str(name)
        if name_str in predictions_by_model and name_str not in ordered:
            ordered.append(name_str)
    for name in predictions_by_model:
        name_str = str(name)
        if name_str not in ordered:
            ordered.append(name_str)
    return ordered[: max(1, int(max_models))]


def _subsample_scatter_pair(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    max_scatter_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically subsample paired 1D arrays only when row count exceeds `max_scatter_points`."""
    cap = max(1, int(max_scatter_points))
    if len(x_vals) > cap:
        pick = rng.choice(len(x_vals), size=cap, replace=False)
        return x_vals[pick], y_vals[pick]
    return x_vals, y_vals


def _normalize_eras_for_grouping(eras: Any) -> Tuple[np.ndarray, str]:
    """Normalize era labels (int, float, `'era1234'`, ISO date/Timestamp, or string) for safe grouping and plotting."""
    raw_series = pd.Series(np.asarray(eras).ravel())
    if len(raw_series) == 0:
        return np.empty(0, dtype=np.int64), "int"
    if pd.api.types.is_datetime64_any_dtype(raw_series.dtype):
        return pd.to_datetime(raw_series).to_numpy(), "datetime"
    if raw_series.dtype == object and isinstance(raw_series.dropna().head(1).to_numpy().tolist()[0], (pd.Timestamp, datetime)) if raw_series.notna().any() else False:
        return pd.to_datetime(raw_series).to_numpy(), "datetime"

    parsed_nums = _parse_era_numbers(pd.Index(raw_series))
    if np.all(np.isfinite(parsed_nums)):
        if np.all(np.abs(parsed_nums - np.rint(parsed_nums)) <= 1e-9):
            return np.rint(parsed_nums).astype(np.int64), "int"
        return parsed_nums.astype(np.float64, copy=False), "float"

    parsed_dates = pd.to_datetime(raw_series, errors="coerce")
    if not parsed_dates.isna().any():
        return parsed_dates.to_numpy(), "datetime"

    return raw_series.astype(str).to_numpy(), "str"


def _coerce_era_window_bound(bound: Any, *, era_kind: str) -> Any:
    """Coerce one `era_window` boundary value to match `era_kind` (`'int'`, `'float'`, `'datetime'`, `'str'`)."""
    if bound is None:
        return None
    if era_kind in ("int", "float"):
        parsed = _parse_era_numbers(pd.Index([bound]))
        if len(parsed) == 1 and np.isfinite(parsed[0]):
            return int(round(float(parsed[0]))) if era_kind == "int" else float(parsed[0])
        return None
    if era_kind == "datetime":
        ts = pd.to_datetime(bound, errors="coerce")
        return None if pd.isna(ts) else ts
    return str(bound)


def _slice_era_window(
    ts_df: pd.DataFrame,
    era_window: Optional[Tuple[Any, Any]],
    *,
    era_kind: str,
) -> pd.DataFrame:
    """Safely slice a sorted era-indexed DataFrame using boolean bounds across numeric, datetime, or string eras."""
    if ts_df.empty or era_window is None or len(era_window) != 2:
        return ts_df
    lo = _coerce_era_window_bound(era_window[0], era_kind=era_kind)
    hi = _coerce_era_window_bound(era_window[1], era_kind=era_kind)
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo

    idx = ts_df.index
    mask = np.ones(len(ts_df), dtype=bool)
    if lo is not None:
        mask &= np.asarray(idx >= lo, dtype=bool)
    if hi is not None:
        mask &= np.asarray(idx <= hi, dtype=bool)

    sliced = ts_df.loc[mask]
    if sliced.empty:
        print(
            f"[plot_residual_diagnostics] era_window={era_window!r} matched 0 eras; "
            f"showing all {len(ts_df)} available era(s)."
        )
        return ts_df
    return sliced


def _align_predictions_to_target_scale(
    preds: np.ndarray,
    actuals: np.ndarray,
    eras: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool]:
    """Align out-of-range predictions (e.g. `era_zscore` in `[-4, +4]`) onto `[0, 1]` via per-era percentile rank.

    Returns `(aligned_preds, was_rank_aligned)`. Predictions already on the `[0, 1]` target scale are preserved.
    """
    p_arr = np.asarray(preds, dtype=np.float64).ravel()
    a_arr = np.asarray(actuals, dtype=np.float64).ravel()
    finite_p = np.isfinite(p_arr)
    finite_a = np.isfinite(a_arr)
    if not np.any(finite_p) or not np.any(finite_a):
        return p_arr, False

    a_min, a_max = float(np.min(a_arr[finite_a])), float(np.max(a_arr[finite_a]))
    p_min, p_max = float(np.min(p_arr[finite_p])), float(np.max(p_arr[finite_p]))
    target_in_unit = (a_min >= -0.05) and (a_max <= 1.05)
    pred_out_of_unit = (p_min < -0.10) or (p_max > 1.10)
    if not (target_in_unit and pred_out_of_unit):
        return p_arr, False

    p_series = pd.Series(p_arr)
    if eras is not None and len(eras) == len(p_arr):
        aligned = p_series.groupby(eras, sort=False).rank(pct=True).to_numpy(dtype=np.float64)
    else:
        aligned = p_series.rank(pct=True).to_numpy(dtype=np.float64)
    return aligned, True


def plot_residual_diagnostics(
    y_true: Any,
    predictions_by_model: Mapping[str, Any],
    eras: Any,
    *,
    best_variant_name: Optional[str] = None,
    variant_order: Optional[Sequence[str]] = None,
    max_models: int = 4,
    max_scatter_points: int = 5000,
    jitter_discrete: bool = False,
    era_window: Optional[Tuple[Any, Any]] = None,
    random_state: int = 42,
    save_prefix: Optional[Any] = None,
) -> pd.DataFrame:
    """Plot clean Numerai-native residual diagnostics (replacing vertical-stripe raw scatter, flat era-mean, and % error charts).

    Generates two clean, scale-aligned diagnostic figures:
      1. Residual Distribution by Actual Target Bin (`Aligned Pred - Actual`, boxplot + bin means per target bin).
      2. Prediction Decile Calibration & Per-Era MAE Trend (monotonic calibration across 10 predicted deciles
         stacked above per-era mean absolute residual across validation eras).

    Automatically rank-normalizes out-of-range predictions (e.g. `era_zscore` in `[-4, +4]`) per era to `[0, 1]`
    so residual scales remain directly comparable across variants.
    """
    import matplotlib.pyplot as plt

    del max_scatter_points, jitter_discrete, random_state  # Retained for call-site compatibility

    actual_all = np.asarray(y_true, dtype=np.float64).ravel()
    era_all, era_kind = _normalize_eras_for_grouping(eras)
    if len(actual_all) == 0 or len(actual_all) != len(era_all):
        raise ValueError(
            f"y_true ({len(actual_all)}) and eras ({len(era_all)}) must be non-empty and equal length."
        )
    if not predictions_by_model:
        raise ValueError("predictions_by_model must contain at least one model entry.")

    ordered_names = _order_residual_variants(
        predictions_by_model,
        best_variant_name=best_variant_name,
        variant_order=variant_order,
        max_models=max_models,
    )

    extracted_preds: Dict[str, np.ndarray] = {}
    rank_aligned_flags: Dict[str, bool] = {}
    for name in ordered_names:
        pred_arr = _extract_variant_predictions(predictions_by_model[name])
        if len(pred_arr) != len(actual_all):
            raise ValueError(
                f"Prediction length mismatch for {name!r}: expected {len(actual_all)}, got {len(pred_arr)}."
            )
        aligned_arr, was_aligned = _align_predictions_to_target_scale(pred_arr, actual_all, eras=era_all)
        extracted_preds[name] = aligned_arr
        rank_aligned_flags[name] = was_aligned

    palette = ["#2e8b57", "#e07b00", "#2563eb", "#7446A5", "#dc2626", "#0d9488"]
    model_colors = {name: palette[idx % len(palette)] for idx, name in enumerate(extracted_preds)}
    display_labels = {
        name: (f"[BEST] {name}" if best_variant_name is not None and name == best_variant_name else name)
        for name in extracted_preds
    }

    finite_actual = np.isfinite(actual_all)
    valid_masks: Dict[str, np.ndarray] = {}
    summary_rows: List[Dict[str, Any]] = []
    for name, pred_arr in extracted_preds.items():
        mask = finite_actual & np.isfinite(pred_arr)
        if not np.any(mask):
            raise ValueError(f"No finite prediction/target pairs available for {name!r}.")
        valid_masks[name] = mask
        a_v = actual_all[mask]
        p_v = pred_arr[mask]
        r_v = p_v - a_v
        summary_rows.append({
            "variant": display_labels[name],
            "n_valid": int(len(r_v)),
            "mean_residual": round(float(np.mean(r_v)), 6),
            "mae": round(float(np.mean(np.abs(r_v))), 4),
            "rmse": round(float(np.sqrt(np.mean(r_v * r_v))), 4),
            "under_predict_pct": round(float(np.mean(r_v < 0.0) * 100.0), 2),
            "over_predict_pct": round(float(np.mean(r_v > 0.0) * 100.0), 2),
        })

    n_models = len(extracted_preds)
    unique_actuals = np.sort(np.unique(actual_all[finite_actual]))
    is_discrete_target = 2 <= len(unique_actuals) <= 10

    prefix_path = Path(save_prefix) if save_prefix is not None else None
    if prefix_path is not None:
        prefix_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Residual Distribution by Actual Target Bin (Boxplot + Bin Mean)
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(
        1,
        n_models,
        figsize=(6.8 * n_models, 4.8),
        sharey=True,
        squeeze=False,
    )
    flat_axes1 = axes1.ravel()
    for idx, (name, pred_arr) in enumerate(extracted_preds.items()):
        ax = flat_axes1[idx]
        mask = valid_masks[name]
        a_v = actual_all[mask]
        r_v = pred_arr[mask] - a_v
        color = model_colors[name]

        if is_discrete_target:
            bin_labels = [f"{u:.2f}" for u in unique_actuals]
            grouped_res = [r_v[a_v == u] for u in unique_actuals]
        else:
            q_edges = np.unique(np.quantile(a_v, np.linspace(0.0, 1.0, 6)))
            if len(q_edges) >= 3:
                bin_idx = np.clip(np.searchsorted(q_edges[1:-1], a_v, side="right"), 0, len(q_edges) - 2)
                bin_labels = [f"[{q_edges[i]:.2f}, {q_edges[i + 1]:.2f}]" for i in range(len(q_edges) - 1)]
                grouped_res = [r_v[bin_idx == i] for i in range(len(q_edges) - 1)]
            else:
                bin_labels = ["All"]
                grouped_res = [r_v]

        non_empty = [(lbl, arr) for lbl, arr in zip(bin_labels, grouped_res) if len(arr) > 0]
        positions = np.arange(1, len(non_empty) + 1)
        bp = ax.boxplot(
            [arr for _, arr in non_empty],
            positions=positions,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#0f172a", "linewidth": 1.6},
            whiskerprops={"color": color, "linewidth": 1.2},
            capprops={"color": color, "linewidth": 1.2},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.32)
            patch.set_edgecolor(color)

        means = [float(np.mean(arr)) for _, arr in non_empty]
        ax.plot(positions, means, color=color, marker="D", markersize=5.5, linewidth=1.6, label="Bin mean residual", zorder=4)
        ax.axhline(0.0, color="#0f172a", linestyle="--", linewidth=1.1, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([lbl for lbl, _ in non_empty], rotation=0 if is_discrete_target else 20)
        align_note = " (era-rank aligned)" if rank_aligned_flags[name] else ""
        ax.set_title(f"{display_labels[name]}{align_note}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Actual Target Bin")
        if idx == 0:
            ax.set_ylabel("Residual: Predicted - Actual")
        ax.legend(loc="upper right", frameon=False, fontsize=8.5)
        _apply_clean_residual_axes_style(ax)

    fig1.suptitle("3C-i. Residual Distribution by Target Bin (below 0 = under-predicting)", fontsize=13, fontweight="bold")
    fig1.tight_layout()
    if prefix_path is not None:
        fig1.savefig(f"{prefix_path}_residuals_by_target_bin.png", dpi=140)
    plt.show()
    plt.close(fig1)
    del fig1
    gc.collect()

    # ------------------------------------------------------------------
    # 2. Decile Calibration Curve & Per-Era MAE Trend
    # ------------------------------------------------------------------
    fig2, (ax_cal, ax_mae) = plt.subplots(2, 1, figsize=(13.5, 8.2))
    overall_actual_mean = float(np.mean(actual_all[finite_actual])) if np.any(finite_actual) else 0.5
    ax_cal.axhline(overall_actual_mean, color="#64748b", linestyle=":", linewidth=1.2, label=f"Target mean ({overall_actual_mean:.3f})")

    era_mae_cols: Dict[str, Any] = {"era": era_all}
    for name, pred_arr in extracted_preds.items():
        mask = valid_masks[name]
        a_v = actual_all[mask]
        p_v = pred_arr[mask]
        color = model_colors[name]

        # Decile calibration: mean actual target across 10 predicted deciles
        p_ranks = pd.Series(p_v).rank(method="first", pct=True).to_numpy(dtype=np.float64)
        deciles = np.clip(np.ceil(p_ranks * 10.0).astype(np.int64), 1, 10)
        dec_x = []
        dec_y = []
        for d in range(1, 11):
            d_mask = deciles == d
            if np.any(d_mask):
                dec_x.append(d)
                dec_y.append(float(np.mean(a_v[d_mask])))
        ax_cal.plot(dec_x, dec_y, color=color, marker="o", linewidth=2.0, markersize=5, label=display_labels[name])

        abs_err = np.full(len(actual_all), np.nan, dtype=np.float64)
        abs_err[mask] = np.abs(p_v - a_v)
        era_mae_cols[name] = abs_err

    ax_cal.set_xticks(range(1, 11))
    ax_cal.set_xlabel("Predicted Decile (1 = Lowest Predicted, 10 = Highest Predicted)")
    ax_cal.set_ylabel("Mean Actual Target")
    ax_cal.set_title("3C-ii. Prediction Decile Calibration (Upward Slope = Positive Cross-Sectional Signal)", fontsize=11, fontweight="bold")
    ax_cal.legend(frameon=False, fontsize=9.0)
    _apply_clean_residual_axes_style(ax_cal)

    mae_df = pd.DataFrame(era_mae_cols).groupby("era", sort=True).mean()
    mae_df = _slice_era_window(mae_df, era_window, era_kind=era_kind)
    if not mae_df.empty:
        n_eras_plotted = len(mae_df)
        if era_kind == "str":
            x_era = np.arange(n_eras_plotted, dtype=np.int64)
            era_tick_labels = mae_df.index.astype(str).to_numpy()
        else:
            x_era = mae_df.index.to_numpy()
            era_tick_labels = None

        roll_win = min(12, max(1, n_eras_plotted // 4))
        for name in extracted_preds:
            s_mae = mae_df[name].to_numpy(dtype=np.float64)
            s_roll = pd.Series(s_mae).rolling(window=roll_win, min_periods=1).mean().to_numpy(dtype=np.float64)
            ax_mae.plot(x_era, s_mae, color=model_colors[name], alpha=0.28, linewidth=1.0)
            ax_mae.plot(x_era, s_roll, color=model_colors[name], linewidth=2.1, label=f"{display_labels[name]} ({roll_win}-era MA)")
        if era_tick_labels is not None:
            tick_step = max(1, n_eras_plotted // 12)
            ax_mae.set_xticks(x_era[::tick_step])
            ax_mae.set_xticklabels(era_tick_labels[::tick_step], rotation=35, ha="right")

    ax_mae.set_xlabel("Era")
    ax_mae.set_ylabel("Per-Era Mean Absolute Residual (MAE)")
    ax_mae.set_title("Per-Era Forecast Error (MAE) Across Validation Eras", fontsize=11, fontweight="bold")
    ax_mae.legend(frameon=False, fontsize=9.0)
    _apply_clean_residual_axes_style(ax_mae)

    fig2.tight_layout()
    if prefix_path is not None:
        fig2.savefig(f"{prefix_path}_calibration_and_era_mae.png", dpi=140)
    plt.show()
    plt.close(fig2)
    del fig2
    gc.collect()

    del extracted_preds, valid_masks, mae_df
    return pd.DataFrame(summary_rows)


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


def simulate_live_staking_scenarios(
    per_era_corr: pd.Series,
    *,
    per_era_mmc: Optional[pd.Series] = None,
    mmc_is_proxy: bool = True,
    total_capital: float = 1000.0,
    settlement_rounds: int = 12,
    weekly_concurrent_rounds: int = 12,
    conservative_payout_factor: float = 0.5,
    live_payout_factor: float = 0.5,
    bmc_proxy_haircut: float = 0.5,
    n_sims: int = 500,
    block_size: int = 12,
    random_state: int = 42,
    title: str = "Model",
    era_dates: Optional[Union[pd.Series, Mapping[Any, Any]]] = None,
    retain_paths: bool = False,
) -> Dict[str, Any]:
    """Run dual-scenario realistic live staking simulations (Conservative Floor vs. Realistic Live Estimate).

    1. `conservative_floor`: CORR only (`per_era_mmc=None`), `payout_factor=conservative_payout_factor` (default 0.5),
       `concurrent_rounds=weekly_concurrent_rounds` (default 12 for 100% weekly capital utilization).
    2. `realistic_live`: CORR + haircutted contribution (`per_era_mmc * bmc_proxy_haircut` when `mmc_is_proxy=True`,
       or `1.0x` if true Meta Model), `payout_factor=live_payout_factor` (default 0.5), `concurrent_rounds=weekly_concurrent_rounds`.
    3. `optimistic_raw_proxy`: Un-haircutted reference at `payout_factor=1.0` and `concurrent_rounds=64`.
    """
    eff_haircut = float(bmc_proxy_haircut) if bool(mmc_is_proxy) else 1.0
    haircutted_mmc: Optional[pd.Series] = None
    if per_era_mmc is not None and len(pd.Series(per_era_mmc).dropna()) > 0:
        haircutted_mmc = pd.Series(per_era_mmc, dtype=np.float64) * eff_haircut
        haircutted_mmc.name = "bmc_proxy" if bool(mmc_is_proxy) else "mmc"

    cons_pnl = simulate_stake_pnl(
        per_era_corr,
        per_era_mmc=None,
        mmc_is_proxy=False,
        total_capital=total_capital,
        concurrent_rounds=weekly_concurrent_rounds,
        settlement_rounds=settlement_rounds,
        payout_factor=conservative_payout_factor,
        payout_rule="auto",
        staking_mode="compound",
        era_dates=era_dates,
    )
    cons_mc = simulate_stake_pnl_monte_carlo(
        per_era_corr,
        per_era_mmc=None,
        mmc_is_proxy=False,
        total_capital=total_capital,
        concurrent_rounds=weekly_concurrent_rounds,
        settlement_rounds=settlement_rounds,
        payout_factor=conservative_payout_factor,
        payout_rule="auto",
        staking_mode="compound",
        n_sims=n_sims,
        block_size=block_size,
        random_state=random_state,
        title=f"{title} [Conservative Floor]",
        era_dates=era_dates,
        retain_paths=retain_paths,
    )

    live_pnl = simulate_stake_pnl(
        per_era_corr,
        per_era_mmc=haircutted_mmc,
        mmc_is_proxy=bool(mmc_is_proxy),
        total_capital=total_capital,
        concurrent_rounds=weekly_concurrent_rounds,
        settlement_rounds=settlement_rounds,
        payout_factor=live_payout_factor,
        payout_rule="auto",
        staking_mode="compound",
        era_dates=era_dates,
    )
    live_mc = simulate_stake_pnl_monte_carlo(
        per_era_corr,
        per_era_mmc=haircutted_mmc,
        mmc_is_proxy=bool(mmc_is_proxy),
        total_capital=total_capital,
        concurrent_rounds=weekly_concurrent_rounds,
        settlement_rounds=settlement_rounds,
        payout_factor=live_payout_factor,
        payout_rule="auto",
        staking_mode="compound",
        n_sims=n_sims,
        block_size=block_size,
        random_state=random_state,
        title=f"{title} [Realistic Live]",
        era_dates=era_dates,
        retain_paths=retain_paths,
    )

    opt_pnl = simulate_stake_pnl(
        per_era_corr,
        per_era_mmc=per_era_mmc,
        mmc_is_proxy=bool(mmc_is_proxy),
        total_capital=total_capital,
        concurrent_rounds=CLASSIC_CONCURRENT_ROUNDS,
        settlement_rounds=settlement_rounds,
        payout_factor=1.0,
        payout_rule="auto",
        staking_mode="compound",
        era_dates=era_dates,
    )

    return {
        "title": title,
        "conservative_floor": {"pnl": cons_pnl, "mc": cons_mc},
        "realistic_live": {"pnl": live_pnl, "mc": live_mc},
        "optimistic_raw_proxy": {"pnl": opt_pnl},
        "meta": {
            "payout_factor": float(live_payout_factor),
            "conservative_payout_factor": float(conservative_payout_factor),
            "bmc_haircut": float(eff_haircut),
            "concurrent_rounds": int(weekly_concurrent_rounds),
            "settlement_rounds": int(settlement_rounds),
            "mmc_is_proxy": bool(mmc_is_proxy),
        },
    }


def build_live_staking_scenarios_df(
    scenarios_by_variant: Mapping[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Build a side-by-side Conservative Floor vs. Realistic Live Estimate staking comparison table."""
    rows: List[Dict[str, Any]] = []
    for variant_name, bundle in scenarios_by_variant.items():
        c_pnl = bundle["conservative_floor"]["pnl"]
        c_mc = bundle["conservative_floor"]["mc"]
        l_pnl = bundle["realistic_live"]["pnl"]
        l_mc = bundle["realistic_live"]["mc"]
        m = bundle["meta"]
        rows.append({
            "variant": str(variant_name),
            "conservative_end_gbp": round(float(c_pnl["end_capital"]), 2),
            "conservative_return_pct": round(float(c_pnl["total_return_pct"]), 2),
            "conservative_worst_dip_pct": round(float(c_pnl["money_drawdown_pct"]), 2),
            "conservative_mc_p_profit_pct": round(float(c_mc["mc_prob_profit_pct"]), 1),
            "realistic_live_end_gbp": round(float(l_pnl["end_capital"]), 2),
            "realistic_live_return_pct": round(float(l_pnl["total_return_pct"]), 2),
            "realistic_live_worst_dip_pct": round(float(l_pnl["money_drawdown_pct"]), 2),
            "realistic_live_mc_p_profit_pct": round(float(l_mc["mc_prob_profit_pct"]), 1),
            "realistic_live_mc_median_gbp": round(float(l_mc["mc_median_final_stake"]), 2),
            "payout_factor": round(float(m["payout_factor"]), 2),
            "bmc_haircut": round(float(m["bmc_haircut"]), 2),
            "concurrent_rounds": int(m["concurrent_rounds"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4-Layer Analysis Suite: Executive Scorecards, JSON Export, CORR & Residuals
# ---------------------------------------------------------------------------

def _json_safe_scalar(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats into strict RFC-8259 JSON-compatible values."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        return fval if np.isfinite(fval) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def format_tuw_years(
    tuw_eras: Any,
    periods_per_year: float = 52.0,
) -> str:
    """Format Time-under-Water era count cleanly in years (e.g. `1.25 yr (65 eras)`)."""
    if tuw_eras is None:
        return "N/A"
    try:
        val = float(tuw_eras)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(val):
        return "N/A"
    eras_int = max(0, int(round(val)))
    yrs = float(eras_int) / max(1.0, float(periods_per_year))
    return f"{yrs:.2f} yr ({eras_int}e)"


def _prioritize_and_mark_winner(
    df: pd.DataFrame,
    best_variant_name: Optional[str] = None,
) -> pd.DataFrame:
    """Return a copy of `df` with `best_variant_name` promoted to row 0 and prefixed with `[WINNER]`."""
    out = df.copy().reset_index(drop=True)
    if best_variant_name is None or "variant" not in out.columns or out.empty:
        return out
    is_winner = out["variant"].astype(str) == str(best_variant_name)
    if bool(is_winner.any()):
        out = pd.concat([out.loc[is_winner], out.loc[~is_winner]], axis=0, ignore_index=True)
        out["variant"] = np.where(
            out["variant"].astype(str) == str(best_variant_name),
            "[WINNER] " + out["variant"].astype(str),
            out["variant"].astype(str),
        )
    return out


def compute_science_score_pct(
    validation_corr: Any,
    validation_sharpe: Any,
    *,
    corr_ceiling: float = CORR_SANITY_HIGH_THRESHOLD,
    sharpe_ceiling: float = SHARPE_OFFICIAL_CEILING,
) -> float:
    """Compute a 0–100% Science Score normalizing validation CORR and Sharpe against Numerai's official ceilings.

    Uses `corr_ceiling = 0.040` (4.0% CORR) and `sharpe_ceiling = 2.50` (validation Sharpe), clamping each
    component to `[0%, 100%]` and averaging them equally (50% CORR + 50% Sharpe).
    """
    c_ceil = max(1e-9, float(corr_ceiling))
    s_ceil = max(1e-9, float(sharpe_ceiling))
    c_val = float(pd.to_numeric(validation_corr, errors="coerce"))
    s_val = float(pd.to_numeric(validation_sharpe, errors="coerce"))
    parts: List[float] = []
    if np.isfinite(c_val):
        parts.append(float(np.clip(c_val / c_ceil, 0.0, 1.0)) * 100.0)
    if np.isfinite(s_val):
        parts.append(float(np.clip(s_val / s_ceil, 0.0, 1.0)) * 100.0)
    if not parts:
        return float(np.nan)
    return round(float(np.mean(parts)), 1)


def _compute_live_round_metrics_for_row(
    row: Mapping[str, Any],
    *,
    live_round_eras: int = CLASSIC_SETTLEMENT_ROUNDS,
) -> Dict[str, Any]:
    """Compute per-live-Numerai-round (default 12-era / 60-day) Money (1A) and Science (1B) metrics for one variant row."""
    round_len = max(1, int(live_round_eras))
    raw_corrs = row.get("validation_era_corrs", None)
    era_s = pd.Series(dtype=np.float64)
    if isinstance(raw_corrs, pd.Series):
        era_s = pd.to_numeric(raw_corrs, errors="coerce").dropna().astype(np.float64)
    elif isinstance(raw_corrs, Mapping) and raw_corrs:
        era_s = pd.to_numeric(pd.Series(raw_corrs), errors="coerce").dropna().astype(np.float64)
    elif isinstance(raw_corrs, (list, tuple, np.ndarray)) and len(raw_corrs) > 0:
        era_s = pd.to_numeric(pd.Series(raw_corrs), errors="coerce").dropna().astype(np.float64)

    n_eras = len(era_s)
    if n_eras == 0:
        for count_key in ("n_validation_eras", "n_eras", "validation_eras_count"):
            raw_n = row.get(count_key, None)
            if raw_n is not None:
                n_num = float(pd.to_numeric(raw_n, errors="coerce"))
                if np.isfinite(n_num) and n_num > 0:
                    n_eras = int(round(n_num))
                    break

    n_live_rounds = (float(n_eras) / float(round_len)) if n_eras > 0 else float(np.nan)
    live_rounds_span = f"{n_live_rounds:.1f} rds ({n_eras}e)" if n_eras > 0 else "N/A"

    start_cap = float(pd.to_numeric(row.get("start_capital", 1000.0), errors="coerce"))
    if not np.isfinite(start_cap) or start_cap <= 0.0:
        start_cap = 1000.0
    end_cap = float(pd.to_numeric(row.get("end_capital", np.nan), errors="coerce"))
    tot_profit = float(pd.to_numeric(row.get("total_profit", np.nan), errors="coerce"))
    if not np.isfinite(tot_profit) and np.isfinite(end_cap):
        tot_profit = end_cap - start_cap

    live_round_return_pct = float(np.nan)
    live_round_profit_gbp = float(np.nan)
    if np.isfinite(n_live_rounds) and n_live_rounds > 0.0:
        if np.isfinite(end_cap):
            if end_cap <= 0.0:
                live_round_return_pct = -100.0
            else:
                live_round_return_pct = round(
                    float(((end_cap / start_cap) ** (1.0 / n_live_rounds) - 1.0) * 100.0), 2
                )
        if np.isfinite(tot_profit):
            live_round_profit_gbp = round(float(tot_profit / n_live_rounds), 2)

    val_corr = float(pd.to_numeric(row.get("validation_corr", np.nan), errors="coerce"))
    live_round_12e_corr = float(np.nan)
    live_round_12e_pos_pct = float(np.nan)
    if len(era_s) >= round_len:
        rolling_round_sums = era_s.rolling(window=round_len, min_periods=round_len).sum().dropna()
        if len(rolling_round_sums) > 0:
            live_round_12e_corr = round(float(rolling_round_sums.mean()), 4)
            live_round_12e_pos_pct = round(float((rolling_round_sums > 0.0).mean() * 100.0), 1)
    elif np.isfinite(val_corr):
        live_round_12e_corr = round(float(val_corr * float(round_len)), 4)

    return {
        "live_round_return_pct": live_round_return_pct,
        "live_round_profit_gbp": live_round_profit_gbp,
        "live_rounds_span": live_rounds_span,
        "live_round_12e_corr": live_round_12e_corr,
        "live_round_12e_pos_pct": live_round_12e_pos_pct,
    }


def build_executive_scorecard_tables(
    comparison_df: pd.DataFrame,
    *,
    best_variant_name: Optional[str] = None,
    periods_per_year: float = 52.0,
    corr_ceiling: float = CORR_SANITY_HIGH_THRESHOLD,
    sharpe_ceiling: float = SHARPE_OFFICIAL_CEILING,
    live_round_eras: int = CLASSIC_SETTLEMENT_ROUNDS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the wide comparison DataFrame into two compact, untruncated Executive Scorecard tables.

    Returns `(money_df, science_df)`:
      - `money_df` (Table 1A, 10 cols): `variant`, `end_capital`, `total_return_pct`,
        `live_round_return_pct`, `live_round_profit_gbp`, `money_drawdown_pct`,
        `max_drawdown_gbp`, `worst_round_gbp`, `tuw_max_eras` (formatted in years), `live_rounds_span`.
      - `science_df` (Table 1B, 11 cols): `variant`, `train_corr`, `validation_corr`, `corr_gap`,
        `validation_sharpe`, `live_round_12e_corr`, `live_round_12e_pos_pct`, `validation_psr`,
        `validation_dsr`, `science_score_pct`, `sanity_status`.
    """
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = _prioritize_and_mark_winner(comparison_df, best_variant_name=best_variant_name)
    live_round_rows = [
        _compute_live_round_metrics_for_row(base.loc[idx], live_round_eras=live_round_eras)
        for idx in base.index
    ]

    money_cols = [
        "variant",
        "end_capital",
        "total_return_pct",
        "live_round_return_pct",
        "live_round_profit_gbp",
        "money_drawdown_pct",
        "max_drawdown_gbp",
        "worst_round_gbp",
        "tuw_max_eras",
        "live_rounds_span",
    ]
    money_data: Dict[str, Any] = {}
    for col in money_cols:
        if col in ("live_round_return_pct", "live_round_profit_gbp", "live_rounds_span"):
            money_data[col] = [r[col] for r in live_round_rows]
        elif col not in base.columns:
            money_data[col] = [np.nan] * len(base)
        elif col == "variant":
            money_data[col] = base[col].astype(str).tolist()
        elif col == "tuw_max_eras":
            money_data[col] = [
                format_tuw_years(v, periods_per_year=periods_per_year)
                for v in base[col].tolist()
            ]
        else:
            money_data[col] = pd.to_numeric(base[col], errors="coerce").round(2).tolist()
    money_df = pd.DataFrame(money_data)

    science_cols = [
        "variant",
        "train_corr",
        "validation_corr",
        "corr_gap",
        "validation_sharpe",
        "live_round_12e_corr",
        "live_round_12e_pos_pct",
        "validation_psr",
        "validation_dsr",
        "science_score_pct",
        "sanity_status",
    ]
    val_corr_vals = base["validation_corr"].tolist() if "validation_corr" in base.columns else [np.nan] * len(base)
    val_sharpe_vals = base["validation_sharpe"].tolist() if "validation_sharpe" in base.columns else [np.nan] * len(base)
    science_scores = [
        compute_science_score_pct(
            c_v,
            s_v,
            corr_ceiling=corr_ceiling,
            sharpe_ceiling=sharpe_ceiling,
        )
        for c_v, s_v in zip(val_corr_vals, val_sharpe_vals)
    ]

    science_data: Dict[str, Any] = {}
    for col in science_cols:
        if col == "science_score_pct":
            science_data[col] = science_scores
        elif col in ("live_round_12e_corr", "live_round_12e_pos_pct"):
            science_data[col] = [r[col] for r in live_round_rows]
        elif col not in base.columns:
            science_data[col] = [np.nan] * len(base)
        elif col in ("variant", "sanity_status"):
            science_data[col] = base[col].astype(str).tolist()
        else:
            science_data[col] = pd.to_numeric(base[col], errors="coerce").round(4).tolist()
    science_df = pd.DataFrame(science_data)

    return money_df, science_df


def build_diagnostics_summary_table(
    comparison_df: pd.DataFrame,
    *,
    best_variant_name: Optional[str] = None,
) -> pd.DataFrame:
    """Build the compact 7-column Layer 3 Model Diagnostics table (no horizontal truncation)."""
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame()

    base = _prioritize_and_mark_winner(comparison_df, best_variant_name=best_variant_name)
    diag_cols = [
        "variant",
        "feature_exposure",
        "validation_fnc",
        "validation_fnc_sharpe",
        "validation_mae",
        "forecast_bias",
        "directional_hit_rate",
    ]
    out_data: Dict[str, Any] = {}
    for col in diag_cols:
        if col not in base.columns:
            out_data[col] = [np.nan] * len(base)
        elif col == "variant":
            out_data[col] = base[col].astype(str).tolist()
        else:
            out_data[col] = pd.to_numeric(base[col], errors="coerce").round(4).tolist()
    return pd.DataFrame(out_data)


def display_compact_dataframe(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
) -> None:
    """Display a compact DataFrame without horizontal `...` column truncation."""
    if title:
        print(title)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.max_colwidth",
        64,
        "display.expand_frame_repr",
        False,
    ):
        try:
            from IPython.display import display as ipython_display

            ipython_display(df)
        except Exception:
            print(df.to_string(index=False))


def export_winner_scorecard_json(
    comparison_df: pd.DataFrame,
    *,
    best_variant_name: str,
    model_tag: str,
    output_dir: Union[str, Path],
    experiment_meta: Optional[Mapping[str, Any]] = None,
    periods_per_year: float = 52.0,
) -> Path:
    """Export the Train-CV-Locked Winner's scorecard row to `summary_<model_tag>.json` for cross-notebook comparison."""
    if comparison_df is None or comparison_df.empty:
        raise ValueError("comparison_df must be non-empty to export winner scorecard JSON.")

    matched = comparison_df.loc[comparison_df["variant"].astype(str) == str(best_variant_name)]
    winner_row = matched.reset_index(drop=True).loc[0] if not matched.empty else comparison_df.reset_index(drop=True).loc[0]

    tuw_eras_raw = winner_row.get("tuw_max_eras", np.nan)
    tuw_eras_num = _json_safe_scalar(tuw_eras_raw)
    tuw_years_val = (
        round(float(tuw_eras_num) / max(1.0, float(periods_per_year)), 3)
        if isinstance(tuw_eras_num, (int, float))
        else None
    )

    money_keys = (
        "end_capital",
        "total_return_pct",
        "money_drawdown_pct",
        "max_drawdown_gbp",
        "worst_round_gbp",
        "tuw_max_eras",
    )
    science_keys = (
        "train_corr",
        "validation_corr",
        "corr_gap",
        "validation_sharpe",
        "validation_lo_sharpe",
        "validation_psr",
        "validation_dsr",
        "sanity_status",
    )
    diagnostics_keys = (
        "feature_exposure",
        "validation_fnc",
        "validation_fnc_sharpe",
        "validation_mae",
        "forecast_bias",
        "directional_hit_rate",
        "adf_pvalue",
        "ljung_box_pvalue",
        "arch_lm_pvalue",
        "hhi_pos",
        "hhi_neg",
    )

    live_round_info = _compute_live_round_metrics_for_row(winner_row)
    money_metrics = {k: _json_safe_scalar(winner_row.get(k, None)) for k in money_keys}
    money_metrics["tuw_max_years"] = tuw_years_val
    money_metrics["tuw_formatted"] = format_tuw_years(tuw_eras_raw, periods_per_year=periods_per_year)
    money_metrics["live_round_return_pct"] = _json_safe_scalar(live_round_info["live_round_return_pct"])
    money_metrics["live_round_profit_gbp"] = _json_safe_scalar(live_round_info["live_round_profit_gbp"])
    money_metrics["live_rounds_span"] = live_round_info["live_rounds_span"]
    science_metrics = {k: _json_safe_scalar(winner_row.get(k, None)) for k in science_keys}
    science_metrics["live_round_12e_corr"] = _json_safe_scalar(live_round_info["live_round_12e_corr"])
    science_metrics["live_round_12e_pos_pct"] = _json_safe_scalar(live_round_info["live_round_12e_pos_pct"])
    science_metrics["science_score_pct"] = _json_safe_scalar(
        compute_science_score_pct(
            winner_row.get("validation_corr", np.nan),
            winner_row.get("validation_sharpe", np.nan),
        )
    )
    diagnostics_metrics = {k: _json_safe_scalar(winner_row.get(k, None)) for k in diagnostics_keys}

    safe_tag = (
        str(model_tag or "model")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    payload: Dict[str, Any] = {
        "model_tag": safe_tag,
        "variant": str(winner_row.get("variant", best_variant_name)),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_meta": {
            str(k): _json_safe_scalar(v) for k, v in (experiment_meta or {}).items()
        },
        **money_metrics,
        **science_metrics,
        **diagnostics_metrics,
        "money_table": money_metrics,
        "science_table": science_metrics,
        "diagnostics_table": diagnostics_metrics,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{safe_tag}.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return out_path


def _extract_variant_era_corrs(entry: Any) -> pd.Series:
    """Extract a clean, chronologically sorted per-era validation CORR Series from a variant entry."""
    raw = entry["validation_era_corrs"] if isinstance(entry, Mapping) and "validation_era_corrs" in entry else entry
    return pd.Series(raw, dtype=np.float64).dropna().sort_index()


def plot_cumulative_validation_corr(
    results_by_variant: Mapping[str, Mapping[str, Any]],
    *,
    best_variant_name: Optional[str] = None,
    title: str = "Cumulative Validation CORR Across Variants",
    save_path: Optional[Any] = None,
) -> pd.DataFrame:
    """Plot Cumulative Validation CORR across all variants with the Train-CV-Locked Winner highlighted."""
    import matplotlib.pyplot as plt

    if not results_by_variant:
        raise ValueError("results_by_variant must contain at least one variant.")

    cum_cols: Dict[str, pd.Series] = {}
    sharpe_map: Dict[str, float] = {}
    for variant_name, entry in results_by_variant.items():
        era_s = _extract_variant_era_corrs(entry)
        if len(era_s) == 0:
            continue
        cum_cols[str(variant_name)] = era_s.cumsum()
        if isinstance(entry, Mapping) and "validation_sharpe" in entry and np.isfinite(float(entry["validation_sharpe"])):
            sharpe_map[str(variant_name)] = float(entry["validation_sharpe"])
        else:
            sharpe_map[str(variant_name)] = float(sharpe_ratio(era_s))

    if not cum_cols:
        return pd.DataFrame()

    cum_df = pd.DataFrame(cum_cols).sort_index()
    x_pos = np.arange(len(cum_df), dtype=np.int64)
    era_labels = cum_df.index.astype(str).to_numpy()
    tick_step = max(1, len(x_pos) // 10)
    palette = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626", "#0891b2"]

    fig, ax = plt.subplots(figsize=(14.5, 5.2), facecolor="white")
    ax.set_facecolor("white")
    ax.axhline(0.0, color="#0f172a", linestyle="--", linewidth=1.2, alpha=0.85, zorder=2)

    for idx, variant_name in enumerate(cum_df.columns):
        is_winner = best_variant_name is not None and str(variant_name) == str(best_variant_name)
        color = "#1d4ed8" if is_winner else palette[(idx + 1) % len(palette)]
        y_vals = cum_df[variant_name].to_numpy(dtype=np.float64)
        final_val = float(y_vals[-1]) if len(y_vals) else 0.0
        lbl_prefix = "[WINNER] " if is_winner else ""
        ax.plot(
            x_pos,
            y_vals,
            color=color,
            linewidth=2.6 if is_winner else 1.8,
            alpha=1.0 if is_winner else 0.82,
            zorder=5 if is_winner else 3,
            label=f"{lbl_prefix}{variant_name} (SR {sharpe_map[variant_name]:.2f}, End {final_val:+.2f})",
        )
        if is_winner and len(x_pos) > 0:
            ax.scatter([x_pos[-1]], [final_val], color=color, edgecolors="white", s=60, zorder=6)

    ax.set_xticks(x_pos[::tick_step])
    ax.set_xticklabels(era_labels[::tick_step], rotation=40, ha="right", fontsize=8.5)
    ax.set_xlabel("Validation Era")
    ax.set_ylabel("Cumulative Numerai CORR")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.6)

    fig.tight_layout()
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
    plt.show()
    plt.close(fig)
    del fig
    gc.collect()
    return cum_df


def plot_rolling_validation_corr(
    per_era_corr: pd.Series,
    *,
    window: int = 20,
    comparison_corrs: Optional[Mapping[str, pd.Series]] = None,
    title: str = "Rolling 20-Era Validation CORR",
    save_path: Optional[Any] = None,
) -> pd.Series:
    """Plot a Rolling 20-Era Validation CORR chart with `y=0` reference and positive/negative regime shading."""
    import matplotlib.pyplot as plt

    corr_s = pd.Series(per_era_corr, dtype=np.float64).dropna().sort_index()
    if len(corr_s) == 0:
        return pd.Series(dtype=np.float64)

    req_window = max(2, int(window))
    eff_window = min(req_window, len(corr_s))
    min_p = eff_window if len(corr_s) >= req_window else max(1, eff_window // 2)
    rolling_s = corr_s.rolling(window=eff_window, min_periods=min_p).mean().dropna()
    if len(rolling_s) == 0:
        rolling_s = corr_s.expanding(min_periods=1).mean()

    x_pos = np.arange(len(rolling_s), dtype=np.int64)
    y_vals = rolling_s.to_numpy(dtype=np.float64)
    era_labels = rolling_s.index.astype(str).to_numpy()
    tick_step = max(1, len(x_pos) // 10)
    overall_mean = float(corr_s.mean())
    pos_window_pct = float(np.mean(y_vals > 0.0) * 100.0)
    latest_val = float(y_vals[-1])
    min_idx = int(np.argmin(y_vals))
    min_val = float(y_vals[min_idx])
    min_era = era_labels[min_idx]

    fig, ax = plt.subplots(figsize=(14.5, 5.4), facecolor="white")
    ax.set_facecolor("white")

    # Optional subtle overlays of non-winner variants
    all_plotted_vals: List[np.ndarray] = [y_vals]
    if comparison_corrs:
        sec_colors = ["#64748b", "#8b5cf6", "#0891b2", "#a16207"]
        for c_idx, (v_name, v_series) in enumerate(comparison_corrs.items()):
            v_clean = pd.Series(v_series, dtype=np.float64).dropna().sort_index()
            if len(v_clean) == 0:
                continue
            v_roll = v_clean.rolling(window=eff_window, min_periods=min_p).mean().reindex(rolling_s.index)
            if v_roll.notna().any():
                v_arr = v_roll.to_numpy(dtype=np.float64)
                all_plotted_vals.append(v_arr[np.isfinite(v_arr)])
                ax.plot(
                    x_pos,
                    v_arr,
                    color=sec_colors[c_idx % len(sec_colors)],
                    linestyle="--",
                    linewidth=1.25,
                    alpha=0.58,
                    zorder=2,
                    label=f"{v_name} ({eff_window}-era roll)",
                )

    ax.axhline(0.0, color="#0f172a", linestyle="--", linewidth=1.5, zorder=3, label="Zero Edge (0.0)")
    ax.axhline(
        overall_mean,
        color="#475569",
        linestyle=":",
        linewidth=1.3,
        zorder=3,
        label=f"Full-Period Mean ({overall_mean:+.4f})",
    )
    ax.fill_between(x_pos, 0.0, y_vals, where=(y_vals >= 0.0), interpolate=True, color="#10b981", alpha=0.28, zorder=2)
    ax.fill_between(x_pos, 0.0, y_vals, where=(y_vals < 0.0), interpolate=True, color="#ef4444", alpha=0.28, zorder=2)
    ax.plot(x_pos, y_vals, color="#1d4ed8", linewidth=2.2, zorder=4, label=f"Winner Rolling {eff_window}-Era CORR")

    # Highlight lowest rolling regime trough
    ax.scatter([min_idx], [min_val], color="#b91c1c", edgecolors="white", s=65, zorder=6)
    x_off = 10 if min_idx < 0.70 * len(x_pos) else -10
    ha_min = "left" if x_off > 0 else "right"
    ax.annotate(
        f"Trough: {min_val:+.4f} @ Era {min_era}",
        xy=(min_idx, min_val),
        xytext=(x_off, -18),
        textcoords="offset points",
        ha=ha_min,
        va="top",
        fontsize=8.3,
        fontweight="bold",
        color="#991b1b",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fef2f2", "edgecolor": "#dc2626", "alpha": 0.94},
        arrowprops={"arrowstyle": "->", "color": "#b91c1c", "linewidth": 1.1},
        zorder=7,
    )

    combined_vals = np.concatenate([a for a in all_plotted_vals if len(a) > 0])
    y_min_all = min(float(np.min(combined_vals)), 0.0)
    y_max_all = max(float(np.max(combined_vals)), 0.0)
    y_span = max(y_max_all - y_min_all, 0.02)
    ax.set_ylim(y_min_all - 0.25 * y_span, y_max_all + 0.18 * y_span)

    ax.set_xticks(x_pos[::tick_step])
    ax.set_xticklabels(era_labels[::tick_step], rotation=40, ha="right", fontsize=8.5)
    ax.set_xlabel("Validation Era")
    ax.set_ylabel(f"Rolling {eff_window}-Era Mean CORR")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.4)

    fig.suptitle(title, fontsize=13.0, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.925,
        f"Window: {eff_window} eras  |  Positive Windows: {pos_window_pct:.1f}%  |  "
        f"Overall Mean: {overall_mean:+.4f}  |  Latest {eff_window}-Era: {latest_val:+.4f}  |  Worst Trough: {min_val:+.4f} (@ Era {min_era})",
        ha="center",
        va="top",
        fontsize=8.9,
        fontweight="bold",
        color="#334155",
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.17)
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
    plt.show()
    plt.close(fig)
    del fig
    gc.collect()
    return rolling_s


def _extract_variant_predictions(entry: Any) -> np.ndarray:
    """Extract a 1D float64 prediction array from either a result dict or raw array-like."""
    raw = entry["validation_preds"] if isinstance(entry, Mapping) and "validation_preds" in entry else entry
    return np.asarray(raw, dtype=np.float64).ravel()


def _select_top2_variants(
    predictions_by_model: Mapping[str, Any],
    *,
    best_variant_name: Optional[str] = None,
    variant_order: Optional[Sequence[str]] = None,
) -> List[str]:
    """Select the top 2 variants for side-by-side residual comparison (locked winner first, then top runner-up)."""
    selected: List[str] = []
    if best_variant_name is not None and best_variant_name in predictions_by_model:
        selected.append(str(best_variant_name))
    candidates = list(variant_order) if variant_order is not None else list(predictions_by_model.keys())
    for name in candidates:
        name_str = str(name)
        if name_str in predictions_by_model and name_str not in selected:
            selected.append(name_str)
        if len(selected) >= 2:
            break
    if len(selected) < 2:
        for name in predictions_by_model:
            name_str = str(name)
            if name_str not in selected:
                selected.append(name_str)
            if len(selected) >= 2:
                break
    return selected[:2]


def plot_top2_residuals_vs_actual(
    y_true: Any,
    predictions_by_model: Mapping[str, Any],
    eras: Optional[Any] = None,
    *,
    best_variant_name: Optional[str] = None,
    variant_order: Optional[Sequence[str]] = None,
    max_scatter_points: int = 5000,
    jitter_strength: float = 0.0,
    random_state: int = 42,
    save_path: Optional[Any] = None,
) -> pd.DataFrame:
    """Plot a clean CAVU flight-deck style side-by-side Residuals vs. Actual Target comparison for the top 2 variants."""
    import matplotlib.pyplot as plt

    del eras  # Accepted for signature parity with plot_residual_diagnostics
    actual_all = np.asarray(y_true, dtype=np.float64).ravel()
    if len(actual_all) == 0:
        raise ValueError("y_true must be non-empty.")
    if not predictions_by_model:
        raise ValueError("predictions_by_model must contain at least one variant.")

    top_names = _select_top2_variants(
        predictions_by_model,
        best_variant_name=best_variant_name,
        variant_order=variant_order,
    )
    extracted: Dict[str, np.ndarray] = {}
    for name in top_names:
        pred_arr = _extract_variant_predictions(predictions_by_model[name])
        if len(pred_arr) != len(actual_all):
            raise ValueError(
                f"Prediction length mismatch for {name!r}: expected {len(actual_all)}, got {len(pred_arr)}."
            )
        extracted[name] = pred_arr

    finite_actual = np.isfinite(actual_all)
    unique_actuals = np.sort(np.unique(actual_all[finite_actual]))
    is_discrete = 2 <= len(unique_actuals) <= 10
    min_step = float(np.min(np.diff(unique_actuals))) if len(unique_actuals) >= 2 else 0.25
    jitter_half = min(abs(float(jitter_strength)), 0.16 * min_step) if (is_discrete and jitter_strength) else 0.0

    n_panels = len(top_names)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(7 * n_panels, 5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    panel_colors = ("#2e8b57", "#e07b00")
    rng = np.random.default_rng(int(random_state))
    cap_pts = max(1, int(max_scatter_points))
    summary_rows: List[Dict[str, Any]] = []

    for idx, name in enumerate(top_names):
        ax = flat_axes[idx]
        pred_arr = extracted[name]
        mask = finite_actual & np.isfinite(pred_arr)
        if not np.any(mask):
            raise ValueError(f"No finite prediction/target pairs available for {name!r}.")

        a_v = actual_all[mask]
        p_v = pred_arr[mask]
        r_v = p_v - a_v
        mae_v = float(np.mean(np.abs(r_v)))
        rmse_v = float(np.sqrt(np.mean(r_v * r_v)))
        mean_r = float(np.mean(r_v))
        under_pct = float(np.mean(r_v < 0.0) * 100.0)
        over_pct = float(np.mean(r_v > 0.0) * 100.0)
        is_winner = best_variant_name is not None and name == best_variant_name
        display_name = f"[WINNER] {name}" if is_winner else name

        summary_rows.append({
            "variant": display_name,
            "n_valid": int(len(r_v)),
            "mean_residual": round(mean_r, 6),
            "mae": round(mae_v, 4),
            "rmse": round(rmse_v, 4),
            "under_predict_pct": round(under_pct, 2),
            "over_predict_pct": round(over_pct, 2),
        })

        a_s, p_s = _subsample_scatter_pair(a_v, p_v, max_scatter_points=cap_pts, rng=rng)
        x_plot = a_s + rng.uniform(-jitter_half, jitter_half, size=len(a_s)) if jitter_half > 0.0 else a_s
        base_color = panel_colors[idx % len(panel_colors)]

        ax.scatter(x_plot, p_s - a_s, alpha=0.55, s=24, color=base_color)
        ax.axhline(0.0, color="black", linestyle="--")
        ax.set_title(f"{display_name}: Residuals vs Actual")
        ax.set_xlabel("Actual Target")
        ax.set_ylabel("Residual: Predicted - Actual")
        _apply_clean_residual_axes_style(ax)

    fig.suptitle("Residuals vs Actual Target (below 0 = under-predicting)", fontsize=13)
    fig.tight_layout()
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
    plt.show()
    plt.close(fig)
    del fig
    gc.collect()
    return pd.DataFrame(summary_rows)


def infer_feature_group_name(
    feature_name: Any,
    feature_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> str:
    """Map a feature column name to its Numerai / engineered feature group."""
    col = str(feature_name).strip()
    if feature_groups:
        for grp_name, cols in feature_groups.items():
            if col in cols:
                return str(grp_name)
    low = col.lower()
    if low.startswith("pca_comp_"):
        return "pca"
    if low.startswith("fourier_"):
        return "fourier"
    if "_ticker_" in low:
        return "ticker_ts"
    if "_era_roll" in low or "_era_diff" in low:
        return "era_ts"
    if low.startswith("row_"):
        return "era_dispersion"
    if low.endswith("_country_ranknorm"):
        return "signals_ranknorm"
    if low.endswith("_factor"):
        return "signals_factor"
    parts = [p for p in low.split("_") if p]
    if len(parts) >= 2 and parts[0] == "feature":
        token = "".join(ch for ch in parts[1] if not ch.isdigit())
        return token or parts[1]
    return parts[0] if parts else "other"


def build_feature_importance_share_and_group_summary(
    importance_df: pd.DataFrame,
    *,
    feature_col: str = "feature",
    importance_col: str = "gain_importance",
    split_col: Optional[str] = None,
    feature_groups: Optional[Mapping[str, Sequence[str]]] = None,
    top_n: int = 20,
) -> Dict[str, Any]:
    """Compute % share of total importance per feature, Pareto concentration curve, and group distribution table."""
    if importance_df is None or importance_df.empty or feature_col not in importance_df.columns or importance_col not in importance_df.columns:
        return {
            "n_total_features": 0,
            "n_nonzero_features": 0,
            "total_importance": 0.0,
            "pct_features_for_50pct": float(np.nan),
            "pct_features_for_80pct": float(np.nan),
            "lorenz_pct_features": np.empty(0, dtype=np.float64),
            "lorenz_cum_importance_pct": np.empty(0, dtype=np.float64),
            "top_share_df": pd.DataFrame(),
            "group_summary_df": pd.DataFrame(),
        }

    work = pd.DataFrame({
        "feature": importance_df[feature_col].astype(str).tolist(),
        "abs_importance": np.abs(pd.to_numeric(importance_df[importance_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)),
    })
    has_split = bool(split_col and split_col in importance_df.columns)
    if has_split and split_col is not None:
        work["split_importance"] = np.abs(
            pd.to_numeric(importance_df[split_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        )

    work = work.sort_values("abs_importance", ascending=False, kind="stable").reset_index(drop=True)
    n_total = int(len(work))
    imp_arr = work["abs_importance"].to_numpy(dtype=np.float64)
    total_imp = float(imp_arr.sum())
    n_nonzero = int(np.sum(imp_arr > 1e-15))

    if total_imp > 1e-15:
        share_pct = (imp_arr / total_imp) * 100.0
    else:
        share_pct = np.zeros(n_total, dtype=np.float64)
    cum_pct = np.cumsum(share_pct)
    feat_pct = (np.arange(1, n_total + 1, dtype=np.float64) / float(max(1, n_total))) * 100.0

    work["importance_share_pct"] = np.round(share_pct, 3)
    work["cum_importance_pct"] = np.round(cum_pct, 2)
    work["feature_group"] = [infer_feature_group_name(f, feature_groups=feature_groups) for f in work["feature"]]

    pct_for_50 = float(feat_pct[int(np.searchsorted(cum_pct, 50.0))]) if total_imp > 1e-15 else float(np.nan)
    pct_for_80 = float(feat_pct[int(np.searchsorted(cum_pct, 80.0))]) if total_imp > 1e-15 else float(np.nan)

    # Downsample concentration curve to at most 101 points (0%..100%) for compact storage
    curve_x = np.concatenate(([0.0], feat_pct))
    curve_y = np.concatenate(([0.0], cum_pct))
    if len(curve_x) > 101:
        pick_idx = np.unique(np.linspace(0, len(curve_x) - 1, num=101, dtype=np.int64))
        curve_x = curve_x[pick_idx]
        curve_y = curve_y[pick_idx]

    total_split = float(work["split_importance"].sum()) if has_split else 0.0
    grp_rows: List[Dict[str, Any]] = []
    for grp_name, gdf in work.groupby("feature_group", sort=True):
        g_cnt = int(len(gdf))
        g_feat_pct = (float(g_cnt) / float(max(1, n_total))) * 100.0
        g_imp_sum = float(gdf["abs_importance"].sum())
        g_imp_pct = (g_imp_sum / total_imp) * 100.0 if total_imp > 1e-15 else 0.0
        g_lift = (g_imp_pct / g_feat_pct) if g_feat_pct > 1e-12 else 0.0
        row_entry: Dict[str, Any] = {
            "feature_group": str(grp_name),
            "n_features": g_cnt,
            "feature_share_pct": round(g_feat_pct, 2),
            "importance_share_pct": round(g_imp_pct, 2),
            "importance_lift": round(g_lift, 2),
        }
        if has_split:
            g_split_sum = float(gdf["split_importance"].sum())
            g_split_pct = (g_split_sum / total_split) * 100.0 if total_split > 1e-15 else 0.0
            row_entry["split_share_pct"] = round(g_split_pct, 2)
        grp_rows.append(row_entry)

    group_summary_df = (
        pd.DataFrame(grp_rows)
        .sort_values(["importance_share_pct", "n_features"], ascending=[False, False], kind="stable")
        .reset_index(drop=True)
    )
    cap = max(1, int(top_n))
    return {
        "n_total_features": n_total,
        "n_nonzero_features": n_nonzero,
        "total_importance": total_imp,
        "pct_features_for_50pct": round(pct_for_50, 1) if np.isfinite(pct_for_50) else float(np.nan),
        "pct_features_for_80pct": round(pct_for_80, 1) if np.isfinite(pct_for_80) else float(np.nan),
        "lorenz_pct_features": curve_x,
        "lorenz_cum_importance_pct": curve_y,
        "top_share_df": work.head(cap).copy(),
        "group_summary_df": group_summary_df,
    }


def plot_interpretability_share_and_groups(
    share_summary: Mapping[str, Any],
    *,
    model_label: str = "Model",
    metric_label: str = "Importance",
    results_dir: Optional[Any] = None,
    timestamp: str = "",
    experiment_name: str = "",
    save_results: bool = False,
) -> None:
    """Plot 3D-ii (% of Features Cumulative Importance) and 3D-iii (Feature Group Distribution by Gain/Importance)."""
    if not share_summary or int(share_summary.get("n_total_features", 0)) == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    group_df = share_summary.get("group_summary_df")
    if isinstance(group_df, pd.DataFrame) and not group_df.empty:
        display_compact_dataframe(
            group_df.head(15),
            title=f"3D-iii. {model_label} Feature Group Distribution by {metric_label} (Lift = Importance % / Feature %):",
        )

    if plt is None:
        return

    curve_x = np.asarray(share_summary.get("lorenz_pct_features", np.empty(0)), dtype=np.float64)
    curve_y = np.asarray(share_summary.get("lorenz_cum_importance_pct", np.empty(0)), dtype=np.float64)
    pct_50 = float(share_summary.get("pct_features_for_50pct", np.nan))
    pct_80 = float(share_summary.get("pct_features_for_80pct", np.nan))
    n_tot = int(share_summary.get("n_total_features", 0))
    n_nz = int(share_summary.get("n_nonzero_features", 0))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 5.8))

    # Left: 3D-ii Cumulative % Importance vs % of Features
    if len(curve_x) >= 2:
        ax_l.plot(curve_x, curve_y, color="#2563eb", linewidth=2.4, label=f"Cumulative {metric_label} %")
        ax_l.fill_between(curve_x, 0.0, curve_y, color="#2563eb", alpha=0.12)
        ax_l.plot([0.0, 100.0], [0.0, 100.0], color="#94a3b8", linestyle="--", linewidth=1.2, label="Uniform (equal share)")
        if np.isfinite(pct_50):
            ax_l.axhline(50.0, color="#059669", linestyle=":", linewidth=1.1)
            ax_l.axvline(pct_50, color="#059669", linestyle=":", linewidth=1.1)
            ax_l.scatter([pct_50], [50.0], color="#059669", s=45, zorder=5, label=f"50% {metric_label} in top {pct_50:.1f}% features")
        if np.isfinite(pct_80):
            ax_l.axhline(80.0, color="#d97706", linestyle=":", linewidth=1.1)
            ax_l.axvline(pct_80, color="#d97706", linestyle=":", linewidth=1.1)
            ax_l.scatter([pct_80], [80.0], color="#d97706", s=45, zorder=5, label=f"80% {metric_label} in top {pct_80:.1f}% features")
    ax_l.set_xlim(0.0, 100.0)
    ax_l.set_ylim(0.0, 103.0)
    ax_l.set_xlabel("% of Features (Ranked by Importance)")
    ax_l.set_ylabel(f"Cumulative % of Total {metric_label}")
    ax_l.set_title(f"3D-ii. {model_label} Concentration ({n_nz}/{n_tot} Active Features)", fontsize=11, fontweight="bold")
    ax_l.legend(loc="lower right", frameon=False, fontsize=8.5)
    _apply_clean_residual_axes_style(ax_l)

    # Right: 3D-iii Feature Group Distribution (% Importance vs % Feature Count)
    if isinstance(group_df, pd.DataFrame) and not group_df.empty:
        top_g = group_df.head(12).sort_values("importance_share_pct", ascending=True, kind="stable")
        y_pos = np.arange(len(top_g), dtype=np.float64)
        bar_h = 0.36
        imp_vals = top_g["importance_share_pct"].to_numpy(dtype=np.float64)
        cnt_vals = top_g["feature_share_pct"].to_numpy(dtype=np.float64)
        lifts = top_g["importance_lift"].to_numpy(dtype=np.float64)
        labels = top_g["feature_group"].astype(str).tolist()

        ax_r.barh(y_pos + bar_h / 2.0, imp_vals, height=bar_h, color="#10b981", alpha=0.88, label=f"% of Total {metric_label}")
        ax_r.barh(y_pos - bar_h / 2.0, cnt_vals, height=bar_h, color="#94a3b8", alpha=0.65, label="% of Features")
        max_x = max(float(np.max(imp_vals)) if len(imp_vals) else 10.0, float(np.max(cnt_vals)) if len(cnt_vals) else 10.0, 10.0)
        for yi, iv, lf in zip(y_pos, imp_vals, lifts):
            ax_r.text(iv + max_x * 0.015, yi + bar_h / 2.0, f"{iv:.1f}% ({lf:.2f}x)", va="center", fontsize=8.2, color="#0f172a")
        ax_r.set_xlim(0.0, max_x * 1.24)
        ax_r.set_yticks(y_pos)
        ax_r.set_yticklabels(labels)
    ax_r.set_xlabel("Share (%)")
    ax_r.set_title(f"3D-iii. {model_label} Feature Group Share vs. Feature Count", fontsize=11, fontweight="bold")
    ax_r.legend(loc="lower right", frameon=False, fontsize=8.5)
    _apply_clean_residual_axes_style(ax_r)

    fig.tight_layout()
    if save_results and results_dir is not None:
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = model_label.lower().replace(" ", "_").replace("-", "_")
        fig.savefig(out_dir / f"{timestamp}_{experiment_name}_{slug}_importance_share_and_groups.png", dpi=140)
    plt.show()
    plt.close(fig)
    del curve_x, curve_y, fig
    gc.collect()


def strip_validation_predictions(results_by_variant: Mapping[str, Dict[str, Any]]) -> int:
    """Remove large `validation_preds` arrays from variant result dicts and reclaim memory."""
    removed = 0
    for entry in results_by_variant.values():
        if isinstance(entry, dict) and entry.pop("validation_preds", None) is not None:
            removed += 1
    gc.collect()
    return removed


def build_afml_diagnostics_df(
    results_by_variant: Mapping[str, Mapping[str, Any]],
    *,
    cv_trial_stats_by_baseline: Optional[Mapping[str, Mapping[str, Any]]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build AFML backtest diagnostics strictly from per-era CORRs, precomputed FNC scalars, and lightweight CV trial stats.

    Avoids requiring `X_validation`, `y_validation`, `validation_eras`, raw `validation_preds`, or full CV DataFrames.
    """
    trial_stats_map = cv_trial_stats_by_baseline or {}
    rows: List[Dict[str, Any]] = []
    for variant_name, res_entry in results_by_variant.items():
        base_key = str(variant_name).split(" [")[0]
        is_tuned = "[Tuned" in str(variant_name)
        stats_ref = trial_stats_map.get(base_key, {})
        raw_trial_srs = stats_ref.get("trial_srs") if is_tuned else None
        v_trial_srs = (
            raw_trial_srs
            if (raw_trial_srs is not None and len(raw_trial_srs) > 0)
            else [res_entry["validation_sharpe"]]
        )
        v_trials_mat = stats_ref.get("trials_era_corrs_df") if is_tuned else None
        row = summarize_afml_backtest_diagnostics(
            res_entry["validation_era_corrs"],
            trial_srs=v_trial_srs,
            trials_matrix=v_trials_mat,
            label=str(variant_name),
            random_state=random_state,
        )
        if "validation_fnc" in res_entry:
            fnc_val = float(res_entry["validation_fnc"])
            row["fnc_mean"] = round(fnc_val, 4) if np.isfinite(fnc_val) else float(np.nan)
        if "validation_fnc_sharpe" in res_entry:
            fnc_sr = float(res_entry["validation_fnc_sharpe"])
            row["fnc_sharpe"] = round(fnc_sr, 4) if np.isfinite(fnc_sr) else float(np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _estimate_continuous_mode(values: np.ndarray) -> float:
    """Estimate the modal value of a 1D continuous sample via histogram peak bin center."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(np.nan)
    if len(arr) <= 2 or float(np.max(arr) - np.min(arr)) <= 1e-12:
        return float(np.median(arr))
    n_bins = min(20, max(4, int(np.ceil(np.sqrt(len(arr))))))
    counts, edges = np.histogram(arr, bins=n_bins)
    peak_idx = int(np.argmax(counts))
    return float(0.5 * (edges[peak_idx] + edges[peak_idx + 1]))


def build_candidate_timing_summary(
    timing_records_by_model: Mapping[str, pd.DataFrame],
    *,
    max_candidate_seconds: Optional[float] = None,
) -> pd.DataFrame:
    """Summarize per-candidate CV fit & score timings (mean, median, mode, quartiles, P90/P95, failures, suggested cap)."""
    rows: List[Dict[str, Any]] = []
    for model_name, t_df in timing_records_by_model.items():
        if not isinstance(t_df, pd.DataFrame) or t_df.empty:
            continue
        fit_s = (
            pd.to_numeric(t_df["mean_fit_time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            if "mean_fit_time" in t_df.columns
            else np.zeros(len(t_df), dtype=np.float64)
        )
        score_s = (
            pd.to_numeric(t_df["mean_score_time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            if "mean_score_time" in t_df.columns
            else np.zeros(len(t_df), dtype=np.float64)
        )
        if "total_time_s" in t_df.columns:
            raw_tot = pd.to_numeric(t_df["total_time_s"], errors="coerce").to_numpy(dtype=np.float64)
            total_s = np.where(np.isfinite(raw_tot), raw_tot, fit_s + score_s)
        else:
            total_s = fit_s + score_s
        valid_tot = total_s[np.isfinite(total_s) & (total_s >= 0.0)]
        n_cand = int(len(t_df))
        if "mean_test_score" in t_df.columns:
            scores = pd.to_numeric(t_df["mean_test_score"], errors="coerce").to_numpy(dtype=np.float64)
            n_failed = int(np.sum((~np.isfinite(scores)) | (scores <= -0.9999)))
        else:
            n_failed = 0
        fail_pct = (100.0 * float(n_failed) / float(max(1, n_cand)))

        if len(valid_tot) == 0:
            continue
        q25 = float(np.percentile(valid_tot, 25))
        med = float(np.median(valid_tot))
        q75 = float(np.percentile(valid_tot, 75))
        p90 = float(np.percentile(valid_tot, 90))
        p95 = float(np.percentile(valid_tot, 95))
        iqr = float(q75 - q25)
        mean_v = float(np.mean(valid_tot))
        std_v = float(np.std(valid_tot, ddof=1)) if len(valid_tot) >= 2 else 0.0
        mode_v = _estimate_continuous_mode(valid_tot)
        suggested_cap = float(max(60.0, np.ceil(max(q75 + 3.0 * iqr, 2.0 * p95))))

        row_dict: Dict[str, Any] = {
            "model": str(model_name),
            "n_candidates": n_cand,
            "n_failed": n_failed,
            "fail_rate_pct": round(fail_pct, 1),
            "min_s": round(float(np.min(valid_tot)), 2),
            "q25_s": round(q25, 2),
            "median_s": round(med, 2),
            "mean_s": round(mean_v, 2),
            "mode_s": round(mode_v, 2),
            "q75_s": round(q75, 2),
            "iqr_s": round(iqr, 2),
            "p90_s": round(p90, 2),
            "p95_s": round(p95, 2),
            "max_s": round(float(np.max(valid_tot)), 2),
            "std_s": round(std_v, 2),
            "median_fit_s": round(float(np.median(fit_s)), 2),
            "median_score_s": round(float(np.median(score_s)), 2),
            "total_search_min": round(float(np.sum(valid_tot)) / 60.0, 2),
            "suggested_timeout_s": int(suggested_cap),
        }
        if max_candidate_seconds is not None and float(max_candidate_seconds) > 0.0:
            row_dict["configured_timeout_s"] = int(round(float(max_candidate_seconds)))
        rows.append(row_dict)
    return pd.DataFrame(rows)


def plot_candidate_timing_analysis(
    cv_analysis_artifacts: Mapping[str, Any],
    *,
    max_candidate_seconds: Optional[float] = None,
    results_dir: Optional[Any] = None,
    timestamp: str = "",
    experiment_name: str = "",
    save_results: bool = False,
) -> pd.DataFrame:
    """Display 5B Candidate Scoring & Fit Timing summary table (mean/median/mode/quartiles/P95) and 3-panel compute chart."""
    timing_records = cv_analysis_artifacts.get("timing_records", {}) if isinstance(cv_analysis_artifacts, Mapping) else {}
    if not timing_records:
        # Graceful fallback: build timing records from `top_candidates` if `timing_records` was not populated
        top_cands = cv_analysis_artifacts.get("top_candidates", {}) if isinstance(cv_analysis_artifacts, Mapping) else {}
        timing_records = {
            k: v
            for k, v in top_cands.items()
            if isinstance(v, pd.DataFrame) and ("mean_fit_time" in v.columns or "mean_score_time" in v.columns)
        }
    summary_df = build_candidate_timing_summary(
        timing_records,
        max_candidate_seconds=max_candidate_seconds,
    )
    if summary_df.empty:
        print("5B. Candidate Scoring Timing Analysis: no CV timing columns available.")
        return summary_df

    display_compact_dataframe(
        summary_df,
        title="5B. Candidate Scoring & Fit Timing Summary (seconds; Suggested Timeout = max(60s, Q3 + 3*IQR, 2*P95)):",
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return summary_df

    for model_name, t_df in timing_records.items():
        if not isinstance(t_df, pd.DataFrame) or t_df.empty:
            continue
        fit_s = (
            pd.to_numeric(t_df["mean_fit_time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            if "mean_fit_time" in t_df.columns
            else np.zeros(len(t_df), dtype=np.float64)
        )
        score_s = (
            pd.to_numeric(t_df["mean_score_time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            if "mean_score_time" in t_df.columns
            else np.zeros(len(t_df), dtype=np.float64)
        )
        if "total_time_s" in t_df.columns:
            raw_tot = pd.to_numeric(t_df["total_time_s"], errors="coerce").to_numpy(dtype=np.float64)
            total_s = np.where(np.isfinite(raw_tot), raw_tot, fit_s + score_s)
        else:
            total_s = fit_s + score_s
        if "candidate_id" in t_df.columns:
            raw_cids = pd.to_numeric(t_df["candidate_id"], errors="coerce").to_numpy(dtype=np.float64)
            cand_ids = np.where(np.isfinite(raw_cids), raw_cids, np.arange(len(t_df), dtype=np.float64)).astype(np.int64)
        else:
            cand_ids = np.arange(len(t_df), dtype=np.int64)
        order = np.argsort(cand_ids, kind="stable")
        cand_ids = cand_ids[order]
        fit_s = fit_s[order]
        score_s = score_s[order]
        total_s = total_s[order]
        scores = (
            pd.to_numeric(t_df["mean_test_score"], errors="coerce").to_numpy(dtype=np.float64)[order]
            if "mean_test_score" in t_df.columns
            else np.full(len(t_df), np.nan, dtype=np.float64)
        )
        failed_mask = (~np.isfinite(scores)) | (scores <= -0.9999)

        med_v = float(np.median(total_s))
        mean_v = float(np.mean(total_s))
        mode_v = _estimate_continuous_mode(total_s)
        q75_v = float(np.percentile(total_s, 75))
        p95_v = float(np.percentile(total_s, 95))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.8))

        # 5B-i: Distribution Histogram with Mean, Median, Mode, Q3, P95
        n_bins = min(20, max(4, int(np.ceil(np.sqrt(len(total_s))))))
        ax1.hist(total_s, bins=n_bins, color="#3b82f6", alpha=0.68, edgecolor="#ffffff")
        ax1.axvline(med_v, color="#059669", linestyle="-", linewidth=2.0, label=f"Median: {med_v:.1f}s")
        ax1.axvline(mean_v, color="#1d4ed8", linestyle="--", linewidth=1.8, label=f"Mean: {mean_v:.1f}s")
        if np.isfinite(mode_v):
            ax1.axvline(mode_v, color="#7c3aed", linestyle=":", linewidth=1.8, label=f"Mode: {mode_v:.1f}s")
        ax1.axvline(q75_v, color="#d97706", linestyle="-.", linewidth=1.8, label=f"Q3 (75%): {q75_v:.1f}s")
        ax1.axvline(p95_v, color="#dc2626", linestyle="--", linewidth=1.5, label=f"P95: {p95_v:.1f}s")
        if (
            max_candidate_seconds is not None
            and float(max_candidate_seconds) > 0.0
            and float(max_candidate_seconds) <= max(float(np.max(total_s)) * 1.5, 1.0)
        ):
            ax1.axvline(float(max_candidate_seconds), color="#991b1b", linestyle="-", linewidth=2.0, label=f"Cap: {float(max_candidate_seconds):.0f}s")
        ax1.set_xlabel("Candidate Total Time (Fit + Score, s)")
        ax1.set_ylabel("Candidates")
        ax1.set_title(f"5B-i. {model_name} Time Distribution", fontsize=10.5, fontweight="bold")
        ax1.legend(loc="upper right", frameon=False, fontsize=8.0)
        _apply_clean_residual_axes_style(ax1)

        # 5B-ii: Per-Candidate Fit vs Score Timeline
        x_pos = np.arange(len(cand_ids), dtype=np.float64)
        ax2.bar(x_pos, fit_s, color="#2563eb", alpha=0.82, label="Fit Time (s)")
        ax2.bar(x_pos, score_s, bottom=fit_s, color="#10b981", alpha=0.82, label="Score Time (s)")
        if np.any(failed_mask):
            ax2.scatter(
                x_pos[failed_mask],
                total_s[failed_mask],
                color="#dc2626",
                marker="x",
                s=55,
                zorder=5,
                label="Failed / Timed-out",
            )
        ax2.axhline(q75_v, color="#d97706", linestyle="-.", linewidth=1.3, label=f"Q3 ({q75_v:.1f}s)")
        ax2.set_xlabel("Candidate Trial Order")
        ax2.set_ylabel("Elapsed Seconds (s)")
        ax2.set_title(f"5B-ii. {model_name} Fit vs. Score Timeline", fontsize=10.5, fontweight="bold")
        ax2.legend(loc="upper right", frameon=False, fontsize=8.0)
        _apply_clean_residual_axes_style(ax2)

        # 5B-iii: Compute Time vs CV Score
        ok_mask = (~failed_mask) & np.isfinite(scores)
        if np.any(ok_mask):
            ax3.scatter(total_s[ok_mask], scores[ok_mask], color="#059669", alpha=0.8, s=42, label="Valid Candidates")
        if np.any(failed_mask):
            fail_y = np.where(np.isfinite(scores[failed_mask]), scores[failed_mask], -1.0)
            ax3.scatter(total_s[failed_mask], fail_y, color="#dc2626", marker="x", s=55, label="Failed / Penalized")
        ax3.axvline(q75_v, color="#d97706", linestyle="-.", linewidth=1.3, label=f"Q3 ({q75_v:.1f}s)")
        ax3.set_xlabel("Candidate Total Time (s)")
        ax3.set_ylabel("CV Mean Test Score")
        ax3.set_title(f"5B-iii. {model_name} Compute vs. CV Score", fontsize=10.5, fontweight="bold")
        ax3.legend(loc="best", frameon=False, fontsize=8.0)
        _apply_clean_residual_axes_style(ax3)

        fig.tight_layout()
        if save_results and results_dir is not None:
            out_dir = Path(results_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            slug = str(model_name).lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            fig.savefig(out_dir / f"{timestamp}_{experiment_name}_{slug}_candidate_timing_analysis.png", dpi=140)
        plt.show()
        plt.close(fig)
        del fit_s, score_s, total_s, cand_ids, scores, failed_mask, fig
        gc.collect()

    return summary_df
