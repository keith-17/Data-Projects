"""data_utils.py — Kaggle (`kagglehub` + `KaggleApi`) and `numerapi` data loading for Numerai Classic v6.0 ("Unity")."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view
from requests.exceptions import HTTPError, RequestException

from utils import quantize_to_5bins

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore[assignment]
    POLARS_AVAILABLE = False

try:
    import kagglehub
    from kagglehub.exceptions import (
        BackendError,
        ColabEnvironmentError,
        ColabHTTPError,
        CredentialError,
        DataCorruptionError,
        KaggleApiHTTPError,
        KaggleEnvironmentError,
        NotFoundError,
        UnauthenticatedError,
        UserCancelledError,
    )
    KAGGLEHUB_EXCEPTIONS: Tuple[type[BaseException], ...] = (
        KaggleApiHTTPError,
        ColabHTTPError,
        UnauthenticatedError,
        CredentialError,
        NotFoundError,
        DataCorruptionError,
        BackendError,
        KaggleEnvironmentError,
        ColabEnvironmentError,
        UserCancelledError,
    )
except ImportError:
    kagglehub = None
    KaggleApiHTTPError = ColabHTTPError = HTTPError  # type: ignore[misc,assignment]
    KAGGLEHUB_EXCEPTIONS = ()

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except (ImportError, OSError, IOError, ValueError):
    # `kaggle/__init__.py` calls `api.authenticate()` at package import time, which raises
    # `OSError`/`IOError` when `~/.kaggle/kaggle.json` is absent or `ValueError` when a key is missing.
    KaggleApi = None  # type: ignore[misc,assignment]

try:
    from kaggle.rest import ApiException as KaggleApiException  # type: ignore[import-not-found]
except (ImportError, OSError, ValueError):
    # kaggle >= 1.7 moved its transport to kagglesdk (raising requests.HTTPError); older versions use kaggle.rest.ApiException.
    KaggleApiException = HTTPError  # type: ignore[misc,assignment]

try:
    from numerapi import NumerAPI
except ImportError:
    NumerAPI = None  # type: ignore[misc,assignment]

KAGGLE_DATASET_SLUG = "svendaj/numerai-data-v6-0-unity"
FORBIDDEN_FEATURE_COLS = frozenset({
    "era",
    "date",
    "data_type",
    "id",
    "numerai_ticker",
    "week_no",
    "year_horizon",
})
METADATA_COLS = ("era", "date", "data_type", "id", "numerai_ticker")
NUMERAPI_DATASET_FILES = ("features.json", "train.parquet", "validation.parquet")
HTTP_STATUS_GUIDANCE = {
    400: "400 Bad Request — check dataset handle or parameters.",
    401: "401 Unauthorized — check Kaggle API credentials (kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).",
    403: "403 Forbidden — check dataset permissions or accept competition/dataset rules.",
    404: "404 Not Found — check dataset slug or version.",
    429: "429 Too Many Requests — Kaggle API rate limit reached.",
    500: "500 Internal Server Error — Kaggle backend error.",
}


def _extract_http_status_code(error: BaseException) -> Optional[int]:
    """Return the HTTP status code attached to a `KaggleApiHTTPError`, `HTTPError`, or `kaggle.rest.ApiException`."""
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None) or getattr(error, "status", None)
    return int(status_code) if isinstance(status_code, int) else None


def _format_http_error(prefix: str, error: BaseException) -> str:
    """Format an API error with its HTTP status code guidance (400, 401, 403, 404, 429, 500) when available."""
    status_code = _extract_http_status_code(error)
    hint = HTTP_STATUS_GUIDANCE.get(status_code, f"HTTP {status_code}") if status_code is not None else ""
    return f"{prefix} ({hint}): {error}" if hint else f"{prefix}: {error}"


def _resolve_numerai_paths(data_root: Path) -> Tuple[Path, Path, Path]:
    """Return `(train_path, validation_path, features_path)`, preferring the int8 Parquets."""
    train_int8_path = data_root / "train_int8.parquet"
    validation_int8_path = data_root / "validation_int8.parquet"
    train_path = train_int8_path if train_int8_path.is_file() else data_root / "train.parquet"
    validation_path = validation_int8_path if validation_int8_path.is_file() else data_root / "validation.parquet"
    return train_path, validation_path, data_root / "features.json"


def _has_complete_dataset(directory: Path) -> bool:
    """True when `directory` exists and contains `features.json` plus train and validation Parquets."""
    if not directory.is_dir():
        return False
    train_path, validation_path, features_path = _resolve_numerai_paths(directory)
    return features_path.is_file() and train_path.is_file() and validation_path.is_file()


def _find_dataset_root(base_dir: Path) -> Optional[Path]:
    """Return `base_dir` or the first nested directory containing `features.json` plus train/validation Parquets."""
    if not base_dir.is_dir():
        return None
    if _has_complete_dataset(base_dir):
        return base_dir
    for features_file in sorted(base_dir.rglob("features.json")):
        if _has_complete_dataset(features_file.parent):
            return features_file.parent
    return None


def _compute_weeks_since_high_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Vectorized causal count of observations since `arr` last hit its rolling maximum over `window` steps."""
    arr_1d = np.asarray(arr, dtype=np.float64).ravel()
    n = len(arr_1d)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    w = max(2, int(window))
    clean = np.where(np.isfinite(arr_1d), arr_1d, -np.inf)
    padded = np.pad(clean, (w - 1, 0), mode="constant", constant_values=-np.inf)
    windows_reversed = sliding_window_view(padded, w)[:, ::-1]
    return np.argmax(windows_reversed, axis=1).astype(np.float64)


def _compute_weeks_since_change_1d(arr: np.ndarray) -> np.ndarray:
    """Vectorized causal count of consecutive observations since `arr` last changed value."""
    arr_1d = np.asarray(arr, dtype=np.float64).ravel()
    n = len(arr_1d)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    idx = np.arange(n, dtype=np.int64)
    changed = np.ones(n, dtype=bool)
    if n > 1:
        changed[1:] = np.abs(arr_1d[1:] - arr_1d[:-1]) > 1e-12
    last_change_idx = np.maximum.accumulate(np.where(changed, idx, 0))
    return (idx - last_change_idx).astype(np.float64)


def _compute_weeks_since_high_grouped(
    sorted_vals: np.ndarray,
    pos_in_group: np.ndarray,
    window: int,
) -> np.ndarray:
    """Vectorized O(N * window) causal weeks-since-high across all contiguous ticker groups without per-group Python loops."""
    n = len(sorted_vals)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    w = max(2, int(window))
    clean = np.where(np.isfinite(sorted_vals), sorted_vals, -np.inf)
    best_val = clean.copy()
    best_lag = np.zeros(n, dtype=np.int32)
    for lag in range(1, min(w, n)):
        valid = pos_in_group[lag:] >= lag
        shifted = clean[:-lag]
        better = valid & (shifted > best_val[lag:])
        best_val[lag:] = np.where(better, shifted, best_val[lag:])
        best_lag[lag:] = np.where(better, lag, best_lag[lag:])
    return best_lag.astype(np.float64)


def _compute_weeks_since_change_grouped(
    sorted_vals: np.ndarray,
    pos_in_group: np.ndarray,
) -> np.ndarray:
    """Vectorized O(N) causal weeks-since-change across all contiguous ticker groups without per-group Python loops."""
    n = len(sorted_vals)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    idx = np.arange(n, dtype=np.int64)
    changed = np.ones(n, dtype=bool)
    if n > 1:
        changed[1:] = (pos_in_group[1:] == 0) | (np.abs(sorted_vals[1:] - sorted_vals[:-1]) > 1e-12)
    last_change_idx = np.maximum.accumulate(np.where(changed, idx, 0))
    return (idx - last_change_idx).astype(np.float64)


def _pl_impute_nonfinite_expr(
    col_name: str,
    era_col: str,
    quantize_scope: str,
) -> Any:
    """Polars expression imputing non-finite values with era or global median before quantization."""
    scope = str(quantize_scope or "era").strip().lower()
    raw = pl.col(col_name).cast(pl.Float64)
    finite = raw.is_finite()
    med = raw.filter(finite).median().over(era_col) if scope == "era" else raw.filter(finite).median()
    return pl.when(finite).then(raw).otherwise(med).fill_null(0.0).alias(col_name)


def _pl_quantize_to_5bins_expr(
    col_name: str,
    era_col: str,
    quantize_scope: str,
) -> Any:
    """Polars expression for 5-bin (`0..4` `Int8`) z-score quantization (`'era'` or `'global'`) on finite Float64 columns."""
    scope = str(quantize_scope or "era").strip().lower()
    clean = pl.col(col_name).cast(pl.Float64)
    if scope == "era":
        mean_e = clean.mean().over(era_col)
        std_e = clean.std(ddof=0).over(era_col)
    else:
        mean_e = clean.mean()
        std_e = clean.std(ddof=0)

    valid_std = std_e.is_not_null() & (std_e > 1e-12)
    safe_std = pl.when(valid_std).then(std_e).otherwise(1.0)
    z = (clean - mean_e) / safe_std
    binned = (z.clip(-2.0, 2.0) + 2.0).round(0).clip(0.0, 4.0).fill_nan(2.0).fill_null(2.0).cast(pl.Int8)
    return pl.when(valid_std).then(binned).otherwise(pl.lit(2, dtype=pl.Int8)).alias(col_name)


def _compute_per_ticker_timeseries_polars(
    df: pd.DataFrame,
    chosen_cols: List[str],
    *,
    ticker_col: str,
    era_col: str,
    active_modes: set,
    windows: List[int],
    lag_steps: List[int],
    quantize_scope: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Multi-threaded Polars + vectorized NumPy implementation of `compute_per_ticker_timeseries_features`."""
    n_rows = len(df)
    if n_rows == 0:
        return df, []

    sub_data: Dict[str, Any] = {
        "_orig_row_idx_": np.arange(n_rows, dtype=np.int64),
        ticker_col: df[ticker_col].astype(str).to_numpy(),
        era_col: df[era_col].to_numpy(),
    }
    for col in chosen_cols:
        sub_data[col] = pd.to_numeric(df[col], errors="coerce").fillna(2.0).to_numpy(dtype=np.float64, copy=False)

    pl_df = pl.DataFrame(sub_data).sort([ticker_col, era_col], maintain_order=True)

    # Contiguous per-ticker 0-based position for vectorized weeks_since_high / weeks_since_change
    need_pos = ("weeks_since_high" in active_modes) or ("weeks_since_change" in active_modes)
    pos_in_group = (
        pl_df.select(pl.int_range(0, pl.len()).over(ticker_col).alias("_pos_")).get_column("_pos_").to_numpy()
        if need_pos
        else None
    )

    quantize_cols: List[str] = []
    direct_int8_cols: List[str] = []
    exprs: List[Any] = []
    extra_numpy_series: Dict[str, np.ndarray] = {}

    for col in chosen_cols:
        c_expr = pl.col(col)
        sorted_vals = pl_df.get_column(col).to_numpy() if need_pos else None

        if "weeks_since_change" in active_modes and sorted_vals is not None and pos_in_group is not None:
            cname = f"{col}_ticker_wks_change"
            extra_numpy_series[cname] = _compute_weeks_since_change_grouped(sorted_vals, pos_in_group)
            quantize_cols.append(cname)

        if "expanding_stats" in active_modes:
            exp_mean_name = f"{col}_ticker_exp_mean"
            exp_min_name = f"{col}_ticker_exp_min"
            exp_max_name = f"{col}_ticker_exp_max"
            exprs.append((c_expr.cum_sum().over(ticker_col) / c_expr.cum_count().over(ticker_col)).alias(exp_mean_name))
            exprs.append(c_expr.cum_min().over(ticker_col).round(0).clip(0.0, 4.0).cast(pl.Int8).alias(exp_min_name))
            exprs.append(c_expr.cum_max().over(ticker_col).round(0).clip(0.0, 4.0).cast(pl.Int8).alias(exp_max_name))
            quantize_cols.append(exp_mean_name)
            direct_int8_cols.extend([exp_min_name, exp_max_name])

        if "raw_lags" in active_modes:
            for lag_k in lag_steps:
                cname = f"{col}_ticker_lag{lag_k}"
                exprs.append(
                    c_expr.shift(lag_k).over(ticker_col).fill_null(2.0).round(0).clip(0.0, 4.0).cast(pl.Int8).alias(cname)
                )
                direct_int8_cols.append(cname)

        for w in windows:
            need_roll_mean = ("rolling_mean" in active_modes) or ("momentum_diff" in active_modes)
            roll_mean_expr = c_expr.rolling_mean(window_size=w, min_samples=1).over(ticker_col) if need_roll_mean else None
            if "rolling_mean" in active_modes and roll_mean_expr is not None:
                cname = f"{col}_ticker_roll_mean{w}"
                exprs.append(roll_mean_expr.alias(cname))
                quantize_cols.append(cname)
            if "momentum_diff" in active_modes and roll_mean_expr is not None:
                cname = f"{col}_ticker_mom_diff{w}"
                exprs.append((c_expr - roll_mean_expr).alias(cname))
                quantize_cols.append(cname)
            if "weeks_since_high" in active_modes and sorted_vals is not None and pos_in_group is not None:
                cname = f"{col}_ticker_wks_high{w}"
                wsh_arr = _compute_weeks_since_high_grouped(sorted_vals, pos_in_group, w)
                scaled_wsh = np.clip(np.rint(wsh_arr * 4.0 / max(1.0, float(w - 1))), 0, 4).astype(np.int8)
                extra_numpy_series[cname] = scaled_wsh
                direct_int8_cols.append(cname)
            if "rolling_rank" in active_modes:
                cname = f"{col}_ticker_roll_rank{w}"
                denom = pl.min_horizontal(c_expr.cum_count().over(ticker_col), pl.lit(w)).cast(pl.Float64)
                rr_expr = (
                    c_expr.rolling_rank(window_size=w, min_samples=1, method="average").over(ticker_col) / denom
                ).fill_null(0.5)
                exprs.append(rr_expr.alias(cname))
                quantize_cols.append(cname)

    if not exprs and not extra_numpy_series:
        return df, []

    if extra_numpy_series:
        pl_df = pl_df.with_columns([pl.Series(name, arr) for name, arr in extra_numpy_series.items()])
    if exprs:
        pl_df = pl_df.with_columns(exprs)
    if quantize_cols:
        # Two-pass expressions avoid nesting .over() median inside .over() mean/std
        pl_df = pl_df.with_columns([_pl_impute_nonfinite_expr(c, era_col, quantize_scope) for c in quantize_cols])
        pl_df = pl_df.with_columns([_pl_quantize_to_5bins_expr(c, era_col, quantize_scope) for c in quantize_cols])

    # Preserve exact deterministic column ordering matching the legacy Pandas path
    added_col_names: List[str] = []
    for col in chosen_cols:
        if "weeks_since_change" in active_modes:
            added_col_names.append(f"{col}_ticker_wks_change")
        if "expanding_stats" in active_modes:
            added_col_names.extend([f"{col}_ticker_exp_mean", f"{col}_ticker_exp_min", f"{col}_ticker_exp_max"])
        if "raw_lags" in active_modes:
            added_col_names.extend([f"{col}_ticker_lag{k}" for k in lag_steps])
        for w in windows:
            if "rolling_mean" in active_modes:
                added_col_names.append(f"{col}_ticker_roll_mean{w}")
            if "momentum_diff" in active_modes:
                added_col_names.append(f"{col}_ticker_mom_diff{w}")
            if "weeks_since_high" in active_modes:
                added_col_names.append(f"{col}_ticker_wks_high{w}")
            if "rolling_rank" in active_modes:
                added_col_names.append(f"{col}_ticker_roll_rank{w}")

    restored = pl_df.sort("_orig_row_idx_").select(added_col_names)
    added_df = restored.to_pandas()
    added_df.index = df.index
    for cname in added_col_names:
        if added_df[cname].dtype != np.int8:
            added_df[cname] = added_df[cname].to_numpy(dtype=np.int8, copy=False)
    out_df = pd.concat([df, added_df], axis=1)
    return out_df, added_col_names


def compute_per_ticker_timeseries_features(
    df: pd.DataFrame,
    base_feature_cols: Sequence[str],
    *,
    ticker_col: str = "numerai_ticker",
    era_col: str = "era",
    ticker_features: Optional[str] = "rolling_mean|momentum_diff|weeks_since_high|rolling_rank|weeks_since_change|expanding_stats|raw_lags",
    ticker_windows: Sequence[int] = (4, 12),
    ticker_lag_steps: Optional[Sequence[int]] = None,
    top_n_cols: int = 4,
    quantize_scope: str = "era",
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute causal per-ticker time-series features (`rolling_mean`, `momentum_diff`, `weeks_since_high`, `rolling_rank`, `weeks_since_change`, `expanding_stats`, `raw_lags`) sorted by `era` (`0..4` `int8` bins)."""
    if (
        ticker_features in (None, False, "None", "none", "null", "")
        or ticker_col not in df.columns
        or era_col not in df.columns
    ):
        return df, []

    valid_modes = (
        "rolling_mean",
        "momentum_diff",
        "weeks_since_high",
        "rolling_rank",
        "weeks_since_change",
        "expanding_stats",
        "raw_lags",
    )
    if ticker_features is True:
        active_modes = set(valid_modes)
    else:
        active_modes = {
            part.strip()
            for part in str(ticker_features).lower().replace(",", "|").split("|")
            if part.strip() in valid_modes or part.strip() in ("all", "true")
        }
        if "all" in active_modes or "true" in active_modes:
            active_modes = set(valid_modes)
    if not active_modes:
        return df, []

    chosen_cols = [str(c) for c in list(base_feature_cols)[: max(1, int(top_n_cols))] if c in df.columns]
    if not chosen_cols:
        return df, []

    windows = (
        [int(w) for w in ticker_windows if int(w) > 1]
        if isinstance(ticker_windows, (list, tuple, set))
        else [4, 12]
    ) or [4, 12]
    lag_steps = (
        [int(k) for k in ticker_lag_steps if int(k) >= 1]
        if isinstance(ticker_lag_steps, (list, tuple, set))
        else [1, 2, 4]
    ) or [1, 2, 4]

    if POLARS_AVAILABLE and pl is not None:
        return _compute_per_ticker_timeseries_polars(
            df,
            chosen_cols,
            ticker_col=ticker_col,
            era_col=era_col,
            active_modes=active_modes,
            windows=windows,
            lag_steps=lag_steps,
            quantize_scope=quantize_scope,
        )

    sorted_df = df.sort_values([ticker_col, era_col], kind="stable")
    sorted_eras = sorted_df[era_col].to_numpy()

    new_cols_dict: Dict[str, pd.Series] = {}
    for col in chosen_cols:
        col_float = pd.to_numeric(sorted_df[col], errors="coerce").fillna(2.0).astype(np.float64)
        col_grouped = col_float.groupby(sorted_df[ticker_col], sort=False)

        if "weeks_since_change" in active_modes:
            cname = f"{col}_ticker_wks_change"
            wsc_series = col_grouped.transform(
                lambda s: _compute_weeks_since_change_1d(s.to_numpy(dtype=np.float64, copy=False))
            )
            new_cols_dict[cname] = pd.Series(
                quantize_to_5bins(wsc_series.to_numpy(dtype=np.float64), eras=sorted_eras, quantize_scope=quantize_scope),
                index=sorted_df.index,
                dtype=np.int8,
            )

        if "expanding_stats" in active_modes:
            exp_mean = col_grouped.transform(lambda s: s.expanding(min_periods=1).mean())
            exp_min = col_grouped.transform(lambda s: s.expanding(min_periods=1).min())
            exp_max = col_grouped.transform(lambda s: s.expanding(min_periods=1).max())
            new_cols_dict[f"{col}_ticker_exp_mean"] = pd.Series(
                quantize_to_5bins(exp_mean.to_numpy(dtype=np.float64), eras=sorted_eras, quantize_scope=quantize_scope),
                index=sorted_df.index,
                dtype=np.int8,
            )
            new_cols_dict[f"{col}_ticker_exp_min"] = pd.Series(
                np.clip(np.rint(exp_min.to_numpy(dtype=np.float64)), 0, 4).astype(np.int8),
                index=sorted_df.index,
                dtype=np.int8,
            )
            new_cols_dict[f"{col}_ticker_exp_max"] = pd.Series(
                np.clip(np.rint(exp_max.to_numpy(dtype=np.float64)), 0, 4).astype(np.int8),
                index=sorted_df.index,
                dtype=np.int8,
            )

        if "raw_lags" in active_modes:
            for lag_k in lag_steps:
                shifted = col_grouped.shift(lag_k).fillna(2.0)
                new_cols_dict[f"{col}_ticker_lag{lag_k}"] = pd.Series(
                    np.clip(np.rint(shifted.to_numpy(dtype=np.float64)), 0, 4).astype(np.int8),
                    index=sorted_df.index,
                    dtype=np.int8,
                )

        for w in windows:
            need_roll_mean = ("rolling_mean" in active_modes) or ("momentum_diff" in active_modes)
            roll_mean = (
                col_grouped.transform(lambda s: s.rolling(window=w, min_periods=1).mean())
                if need_roll_mean
                else None
            )
            if "rolling_mean" in active_modes and roll_mean is not None:
                cname = f"{col}_ticker_roll_mean{w}"
                new_cols_dict[cname] = pd.Series(
                    quantize_to_5bins(roll_mean.to_numpy(), eras=sorted_eras, quantize_scope=quantize_scope),
                    index=sorted_df.index,
                    dtype=np.int8,
                )
            if "momentum_diff" in active_modes and roll_mean is not None:
                cname = f"{col}_ticker_mom_diff{w}"
                diff_vals = (col_float - roll_mean).to_numpy()
                new_cols_dict[cname] = pd.Series(
                    quantize_to_5bins(diff_vals, eras=sorted_eras, quantize_scope=quantize_scope),
                    index=sorted_df.index,
                    dtype=np.int8,
                )
            if "weeks_since_high" in active_modes:
                cname = f"{col}_ticker_wks_high{w}"
                wsh_series = col_grouped.transform(
                    lambda s: _compute_weeks_since_high_1d(s.to_numpy(dtype=np.float64, copy=False), w)
                )
                scaled_wsh = np.clip(
                    np.rint(wsh_series.to_numpy(dtype=np.float64) * 4.0 / max(1.0, float(w - 1))),
                    0,
                    4,
                ).astype(np.int8)
                new_cols_dict[cname] = pd.Series(scaled_wsh, index=sorted_df.index, dtype=np.int8)
            if "rolling_rank" in active_modes:
                cname = f"{col}_ticker_roll_rank{w}"
                roll_rank = col_grouped.transform(
                    lambda s: s.rolling(window=w, min_periods=1).rank(pct=True)
                )
                new_cols_dict[cname] = pd.Series(
                    quantize_to_5bins(roll_rank.fillna(0.5).to_numpy(dtype=np.float64), eras=sorted_eras, quantize_scope=quantize_scope),
                    index=sorted_df.index,
                    dtype=np.int8,
                )

    if not new_cols_dict:
        return df, []

    added_df = pd.DataFrame(new_cols_dict, index=sorted_df.index).reindex(df.index)
    out_df = pd.concat([df, added_df], axis=1)
    return out_df, list(new_cols_dict.keys())


VALID_DOWNSAMPLE_MODES: Tuple[str, ...] = (
    "within_era",
    "recent",
    "earliest",
    "uniform",
    "random_eras",
)


def downsample_training_data(
    train_df: pd.DataFrame,
    *,
    downsample_mode: Optional[str] = None,
    downsample_frac: float = 1.0,
    era_col: str = "era",
    random_state: int = 42,
) -> pd.DataFrame:
    """Deterministically downsample `train_df` (never `validation_df`) while preserving chronological era order.

    Supported modes:
      - `None`: keep 100% of rows.
      - `'within_era'`: keep every era, randomly sampling `downsample_frac` of rows within each era.
      - `'recent'`: keep the most recent `downsample_frac` fraction of unique eras.
      - `'earliest'`: keep the earliest `downsample_frac` fraction of unique eras.
      - `'uniform'`: keep evenly spaced eras spanning the full training range.
      - `'random_eras'`: randomly sample `downsample_frac` of unique eras without replacement.
    """
    mode = (
        None
        if downsample_mode in (None, False, "None", "none", "null", "")
        else str(downsample_mode).strip().lower()
    )
    if mode is not None and mode not in VALID_DOWNSAMPLE_MODES:
        raise ValueError(
            f"downsample_mode must be None or one of {VALID_DOWNSAMPLE_MODES}, got {downsample_mode!r}"
        )
    frac = float(downsample_frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"downsample_frac must be in (0.0, 1.0], got {downsample_frac!r}")
    if mode is None or frac >= 1.0 or len(train_df) == 0:
        return train_df
    if era_col not in train_df.columns:
        raise KeyError(f"Column '{era_col}' not found in train_df.")

    rng = np.random.default_rng(int(random_state))
    era_vals = train_df[era_col].to_numpy()

    if mode == "within_era":
        order = np.argsort(era_vals, kind="stable")
        sorted_eras = era_vals[order]
        _, start_indices, counts = np.unique(sorted_eras, return_index=True, return_counts=True)
        keep_mask = np.zeros(len(train_df), dtype=bool)
        for start, count in zip(start_indices, counts):
            positions = order[start : start + count]
            n_keep = max(1, int(round(int(count) * frac)))
            if n_keep >= int(count):
                keep_mask[positions] = True
            else:
                chosen = rng.choice(positions, size=n_keep, replace=False)
                keep_mask[chosen] = True
        return train_df.loc[keep_mask].copy()

    unique_eras = np.sort(pd.unique(era_vals))
    n_eras = len(unique_eras)
    n_keep_eras = max(1, int(round(n_eras * frac)))
    if n_keep_eras >= n_eras:
        return train_df

    if mode == "recent":
        kept_eras = unique_eras[-n_keep_eras:]
    elif mode == "earliest":
        kept_eras = unique_eras[:n_keep_eras]
    elif mode == "uniform":
        idx = np.round(np.linspace(0, n_eras - 1, n_keep_eras)).astype(np.int64)
        kept_eras = unique_eras[idx]
    elif mode == "random_eras":
        kept_eras = np.sort(rng.choice(unique_eras, size=n_keep_eras, replace=False))
    else:  # pragma: no cover - already validated above
        raise ValueError(f"Unsupported downsample_mode '{mode}'.")

    return train_df.loc[train_df[era_col].isin(kept_eras)].copy()


def _polars_cast_features_to_int8(pl_df: Any, feature_columns: List[str]) -> Any:
    """Cast non-Int8 feature columns to Int8 in parallel inside Polars, preserving `[0, 1]` -> `0..4` scaling."""
    schema = pl_df.schema
    non_int8 = [col for col in feature_columns if schema.get(col) != pl.Int8]
    if not non_int8:
        return pl_df

    # Compute min/max in a single parallel pass to detect float [0, 1] unit-interval columns
    stat_exprs: List[Any] = []
    for col in non_int8:
        dt = schema.get(col)
        is_int_dt = dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        if not is_int_dt:
            c_f = pl.col(col).cast(pl.Float32, strict=False)
            c_fin = c_f.filter(c_f.is_finite())
            stat_exprs.append(c_fin.min().alias(f"__min__{col}"))
            stat_exprs.append(c_fin.max().alias(f"__max__{col}"))

    stats_row = pl_df.select(stat_exprs).row(0, named=True) if stat_exprs else {}

    cast_exprs: List[Any] = []
    for col in non_int8:
        dt = schema.get(col)
        is_int_dt = dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        c_f = pl.col(col).cast(pl.Float32, strict=False)
        is_fin = c_f.is_finite()
        if not is_int_dt:
            lo = stats_row.get(f"__min__{col}")
            hi = stats_row.get(f"__max__{col}")
            is_unit = lo is not None and hi is not None and 0.0 <= float(lo) and float(hi) <= 1.0 + 1e-6
        else:
            is_unit = False

        if is_unit:
            cast_exprs.append(
                (pl.when(is_fin).then(c_f).otherwise(0.5) * 4.0).round(0).clip(0.0, 4.0).cast(pl.Int8).alias(col)
            )
        else:
            cast_exprs.append(
                pl.when(is_fin).then(c_f).otherwise(2.0).round(0).clip(-128.0, 127.0).cast(pl.Int8).alias(col)
            )
    return pl_df.with_columns(cast_exprs)


def _load_and_clean_splits_polars(
    train_path: Path,
    validation_path: Path,
    train_cols: List[str],
    val_cols: List[str],
    feature_columns: List[str],
    target_col: str,
    *,
    apply_era_stride_early: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fast multi-threaded Polars Parquet reader with predicate pushdown, parallel Int8 casting, and era unification."""
    train_lf = pl.scan_parquet(train_path).select(train_cols).filter(pl.col(target_col).is_not_null())
    val_lf = pl.scan_parquet(validation_path).select(val_cols).filter(pl.col(target_col).is_not_null())

    train_pl = train_lf.collect()
    val_pl = val_lf.collect()

    if "data_type" in val_pl.columns:
        val_mask = val_pl.get_column("data_type").cast(pl.Utf8).str.to_lowercase() == "validation"
        if bool(val_mask.any()):
            val_pl = val_pl.filter(val_mask)

    train_pl = train_pl.with_columns(pl.col(target_col).cast(pl.Float32))
    val_pl = val_pl.with_columns(pl.col(target_col).cast(pl.Float32))

    # Unify era across classic `era` and v6.0 `date`
    if "era" in train_pl.columns and "era" in val_pl.columns:
        tr_num = train_pl.get_column("era").cast(pl.Int32, strict=False)
        va_num = val_pl.get_column("era").cast(pl.Int32, strict=False)
        if tr_num.null_count() == 0 and va_num.null_count() == 0:
            train_pl = train_pl.with_columns(tr_num.alias("era"))
            val_pl = val_pl.with_columns(va_num.alias("era"))
            tr_keys = va_keys = None
        else:
            tr_keys = train_pl.get_column("era").cast(pl.Utf8)
            va_keys = val_pl.get_column("era").cast(pl.Utf8)
    elif "date" in train_pl.columns and "date" in val_pl.columns:
        def _date_to_str_series(series: Any) -> Any:
            if series.dtype == pl.Date:
                return series.dt.strftime("%Y-%m-%d")
            if series.dtype == pl.Datetime:
                return series.dt.strftime("%Y-%m-%d")
            return pl.Series("date", pd.to_datetime(series.to_pandas()).dt.strftime("%Y-%m-%d").to_numpy())

        tr_keys = _date_to_str_series(train_pl.get_column("date"))
        va_keys = _date_to_str_series(val_pl.get_column("date"))
    else:
        raise KeyError("Expected 'era' or 'date' column in both train and validation Parquet files.")

    if tr_keys is not None and va_keys is not None:
        all_keys = sorted(set(tr_keys.unique().to_list()) | set(va_keys.unique().to_list()))
        key_to_id = {k: np.int32(pos + 1) for pos, k in enumerate(all_keys)}
        train_pl = train_pl.with_columns(tr_keys.replace_strict(key_to_id, return_dtype=pl.Int32).alias("era"))
        val_pl = val_pl.with_columns(va_keys.replace_strict(key_to_id, return_dtype=pl.Int32).alias("era"))

    train_pl = train_pl.sort("era", maintain_order=True)
    val_pl = val_pl.sort("era", maintain_order=True)

    # Apply era_stride early before Int8 casting / Pandas conversion when pre-stride ticker features are not requested
    if apply_era_stride_early is not None and int(apply_era_stride_early) > 1:
        kept_eras = sorted(train_pl.get_column("era").unique().to_list())[:: int(apply_era_stride_early)]
        train_pl = train_pl.filter(pl.col("era").is_in(kept_eras))

    train_pl = _polars_cast_features_to_int8(train_pl, feature_columns)
    val_pl = _polars_cast_features_to_int8(val_pl, feature_columns)

    return train_pl.to_pandas(), val_pl.to_pandas()


def load_unity_numerai_data(
    data_version: str = "v6.0",
    data_folder: str = "data_folder",
    kaggle_dataset: str = KAGGLE_DATASET_SLUG,
    feature_set: str = "small",
    target_col: str = "target",
    era_stride: int = 4,
    downsample_mode: Optional[str] = None,
    downsample_frac: float = 1.0,
    random_state: int = 42,
    ticker_timeseries_features: Optional[str] = None,
    ticker_rolling_windows: Sequence[int] = (4, 12),
    ticker_lag_steps: Optional[Sequence[int]] = None,
    ticker_top_n_features: int = 4,
    workspace_root: Optional[Any] = None,
    auto_download: bool = True,
    use_numerapi_fallback: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
    """Locate or download Numerai v6.0 Unity data via `kagglehub` / `KaggleApi` / `NumerAPI`.

    Preserves `numerai_ticker`, `id`, `date`, and `era` as metadata in `train_df` and `validation_df`
    (never in `feature_columns`). When `ticker_timeseries_features` is enabled, computes causal per-ticker
    time-series features on the full weekly history BEFORE `era_stride` subsampling is applied to `train_df`.
    Validation eras are never strided so all validation eras remain intact for scoring and Numerai Diagnostics.
    """
    dataset_slug_name = kaggle_dataset.split("/")[-1]
    workspace_dir = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    if workspace_dir.name.lower() == "notebooks":
        workspace_dir = workspace_dir.parent

    # 1. Check standard Kaggle mounts or local version-matched workspace folders (never recurse into a different version like v5.0!)
    candidate_dirs = (
        Path("/kaggle/input/datasets") / kaggle_dataset,
        Path("/kaggle/input/datasets") / kaggle_dataset / data_version,
        Path("/kaggle/input") / dataset_slug_name,
        Path("/kaggle/input") / f"numerai-data-{data_version.replace('.', '-')}",
        workspace_dir / data_folder / data_version,
        workspace_dir / data_folder / dataset_slug_name,
    )
    data_root: Optional[Path] = next((found for cand in candidate_dirs if (found := _find_dataset_root(cand)) is not None), None)
    if data_root is None and _has_complete_dataset(workspace_dir / data_folder):
        data_root = workspace_dir / data_folder
    if data_root is None and Path("/kaggle/input").is_dir():
        data_root = _find_dataset_root(Path("/kaggle/input"))

    # 2. Use official Kaggle clients (`kagglehub` first, then `KaggleApi`), then `NumerAPI` fallback
    if data_root is None and auto_download:
        target_download_dir = workspace_dir / data_folder / data_version

        # 2a. Modern `kagglehub` client (auto-resolves attached /kaggle/input mounts & local cache)
        if kagglehub is not None:
            try:
                resolved_dir = Path(kagglehub.dataset_download(kaggle_dataset))
                data_root = _find_dataset_root(resolved_dir)
            except (KaggleApiHTTPError, ColabHTTPError) as http_error:
                warnings.warn(_format_http_error(f"kagglehub HTTP error for '{kaggle_dataset}'", http_error))
            except KAGGLEHUB_EXCEPTIONS as hub_error:
                warnings.warn(f"kagglehub could not resolve '{kaggle_dataset}' ({type(hub_error).__name__}): {hub_error}")
            except (RequestException, OSError, ValueError) as io_error:
                warnings.warn(_format_http_error(f"kagglehub download failed for '{kaggle_dataset}'", io_error))

        # 2b. Official `KaggleApi` client (`kaggle.api.kaggle_api_extended.KaggleApi`)
        if data_root is None and KaggleApi is not None:
            target_download_dir.mkdir(parents=True, exist_ok=True)
            try:
                kaggle_client = KaggleApi()
                kaggle_client.authenticate()
                kaggle_client.dataset_download_files(
                    kaggle_dataset,
                    path=str(target_download_dir),
                    unzip=True,
                    quiet=False,
                )
                data_root = _find_dataset_root(target_download_dir)
            except (KaggleApiException, HTTPError) as api_error:
                warnings.warn(_format_http_error(f"KaggleApi HTTP error for '{kaggle_dataset}'", api_error))
            except (OSError, IOError, ValueError, RuntimeError, RequestException) as local_error:
                warnings.warn(
                    f"KaggleApi local/auth error for '{kaggle_dataset}' ({type(local_error).__name__}): {local_error}"
                )

        # 2c. Official `NumerAPI` client fallback (`numerapi.NumerAPI`)
        if data_root is None and use_numerapi_fallback and NumerAPI is not None:
            target_download_dir.mkdir(parents=True, exist_ok=True)
            numerai_client = NumerAPI()
            try:
                for file_name in NUMERAPI_DATASET_FILES:
                    numerai_client.download_dataset(
                        f"{data_version}/{file_name}",
                        str(target_download_dir / file_name),
                    )
                data_root = _find_dataset_root(target_download_dir)
            except ValueError as graphql_error:
                raise FileNotFoundError(
                    f"NumerAPI GraphQL error for '{data_version}': {graphql_error}"
                ) from graphql_error
            except (HTTPError, RequestException, OSError) as download_error:
                raise ConnectionError(
                    _format_http_error(f"NumerAPI download failed for '{data_version}'", download_error)
                ) from download_error

    if data_root is None:
        mounted_info = (
            [str(p) for p in sorted(Path("/kaggle/input").rglob("features.json"))]
            if Path("/kaggle/input").is_dir()
            else []
        )
        raise FileNotFoundError(
            f"Numerai dataset '{kaggle_dataset}' ({data_version}) not found (found features.json in /kaggle/input: {mounted_info}). "
            "Attach the Kaggle dataset, configure ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY), "
            "or enable `use_numerapi_fallback=True` with `numerapi` installed."
        )

    train_path, validation_path, features_path = _resolve_numerai_paths(data_root)

    # 3. Read metadata and required Parquet columns
    with features_path.open("r", encoding="utf-8") as features_file:
        feature_metadata = json.load(features_file)

    feature_sets = feature_metadata.get("feature_sets", {})
    if feature_set not in feature_sets:
        raise KeyError(f"Unknown feature_set '{feature_set}'; available: {sorted(feature_sets.keys())}")

    train_schema_cols = set(pq.read_schema(train_path).names)
    validation_schema_cols = set(pq.read_schema(validation_path).names)
    feature_columns = [
        str(column)
        for column in feature_sets[feature_set]
        if str(column) not in FORBIDDEN_FEATURE_COLS
        and str(column) != target_col
        and not str(column).startswith("target")
        and column in train_schema_cols
        and column in validation_schema_cols
    ]
    if not feature_columns:
        raise ValueError(f"No matching feature columns found for '{feature_set}' in both Parquets.")

    train_meta_cols = [column for column in METADATA_COLS if column in train_schema_cols]
    val_meta_cols = [column for column in METADATA_COLS if column in validation_schema_cols]
    train_req_cols = list(dict.fromkeys(train_meta_cols + feature_columns + [target_col]))
    val_req_cols = list(dict.fromkeys(val_meta_cols + feature_columns + [target_col]))
    want_ticker_ts = ticker_timeseries_features not in (None, False, "None", "none", "null", "")
    strided_early = False

    if POLARS_AVAILABLE and pl is not None:
        early_stride = int(era_stride) if (not want_ticker_ts and int(era_stride) > 1) else None
        train_df, validation_df = _load_and_clean_splits_polars(
            train_path,
            validation_path,
            train_req_cols,
            val_req_cols,
            feature_columns,
            target_col,
            apply_era_stride_early=early_stride,
        )
        strided_early = early_stride is not None
    else:
        train_df = pd.read_parquet(train_path, columns=train_req_cols)
        validation_df = pd.read_parquet(validation_path, columns=val_req_cols)

        # 4. Clean targets, unify era across classic `era` and v6.0 `date`, cast features to int8
        if "data_type" in validation_df.columns:
            is_validation_row = validation_df["data_type"].astype(str).str.lower() == "validation"
            if is_validation_row.any():
                validation_df = validation_df.loc[is_validation_row].copy()

        train_df = train_df.dropna(subset=[target_col]).copy()
        validation_df = validation_df.dropna(subset=[target_col]).copy()
        train_df[target_col] = train_df[target_col].astype(np.float32)
        validation_df[target_col] = validation_df[target_col].astype(np.float32)

        if "era" in train_df.columns and "era" in validation_df.columns:
            train_era_numeric = pd.to_numeric(train_df["era"], errors="coerce")
            validation_era_numeric = pd.to_numeric(validation_df["era"], errors="coerce")
            if train_era_numeric.notna().all() and validation_era_numeric.notna().all():
                train_df["era"] = train_era_numeric.astype(np.int32)
                validation_df["era"] = validation_era_numeric.astype(np.int32)
                train_era_keys = validation_era_keys = None
            else:
                train_era_keys = train_df["era"].astype(str)
                validation_era_keys = validation_df["era"].astype(str)
        elif "date" in train_df.columns and "date" in validation_df.columns:
            train_era_keys = pd.to_datetime(train_df["date"]).dt.strftime("%Y-%m-%d")
            validation_era_keys = pd.to_datetime(validation_df["date"]).dt.strftime("%Y-%m-%d")
        else:
            raise KeyError("Expected 'era' or 'date' column in both train and validation Parquet files.")

        if train_era_keys is not None:
            all_era_keys = sorted(set(train_era_keys.unique()) | set(validation_era_keys.unique()))
            era_key_to_id = {era_key: np.int32(position + 1) for position, era_key in enumerate(all_era_keys)}
            train_df["era"] = train_era_keys.map(era_key_to_id).astype(np.int32)
            validation_df["era"] = validation_era_keys.map(era_key_to_id).astype(np.int32)

        train_df = train_df.sort_values("era", kind="stable")
        validation_df = validation_df.sort_values("era", kind="stable")

        for split_df in (train_df, validation_df):
            for column in feature_columns:
                if split_df[column].dtype == np.int8:
                    continue
                values = pd.to_numeric(split_df[column], errors="coerce").to_numpy(dtype=np.float32, copy=False)
                missing_mask = ~np.isfinite(values)
                observed_values = values[~missing_mask]
                is_unit_interval = (
                    len(observed_values) > 0
                    and 0.0 <= float(observed_values.min())
                    and float(observed_values.max()) <= 1.0 + 1e-6
                    and not pd.api.types.is_integer_dtype(split_df[column])
                )
                if is_unit_interval:
                    binned_values = np.rint(np.where(missing_mask, 0.5, values) * 4.0)
                    split_df[column] = np.clip(binned_values, 0, 4).astype(np.int8)
                else:
                    filled_values = np.rint(np.where(missing_mask, 2.0, values))
                    split_df[column] = np.clip(filled_values, -128, 127).astype(np.int8)

    # 5. Optional causal per-ticker time-series features computed BEFORE `era_stride` subsampling
    #    Concatenate the warm-up tail of `train_df` onto `validation_df` before computing validation
    #    per-ticker rolling/lag features so early validation eras do not suffer a cold-start boundary.
    if want_ticker_ts:
        base_cols_snapshot = list(feature_columns)
        win_candidates = [int(w) for w in (ticker_rolling_windows or (4, 12))] if isinstance(ticker_rolling_windows, (list, tuple, set)) else [4, 12]
        lag_candidates = [int(k) for k in (ticker_lag_steps or (1, 2, 4))] if isinstance(ticker_lag_steps, (list, tuple, set)) else [1, 2, 4]
        warmup_eras = max(win_candidates + lag_candidates + [12])
        unique_train_eras_pre = sorted(train_df["era"].unique())
        if "expanding" in str(ticker_timeseries_features).lower() or len(unique_train_eras_pre) <= warmup_eras:
            train_warmup_df = train_df.copy()
        else:
            warmup_era_set = set(unique_train_eras_pre[-warmup_eras:])
            train_warmup_df = train_df.loc[train_df["era"].isin(warmup_era_set)].copy()

        train_df, added_train_cols = compute_per_ticker_timeseries_features(
            train_df,
            base_cols_snapshot,
            ticker_col="numerai_ticker",
            era_col="era",
            ticker_features=ticker_timeseries_features,
            ticker_windows=ticker_rolling_windows,
            ticker_lag_steps=ticker_lag_steps,
            top_n_cols=ticker_top_n_features,
        )

        orig_val_index = validation_df.index
        train_warmup_df["_is_val_split_"] = False
        val_tagged_df = validation_df.copy()
        val_tagged_df["_is_val_split_"] = True
        combined_val_in = pd.concat([train_warmup_df, val_tagged_df], axis=0, ignore_index=True)
        combined_val_out, added_val_cols = compute_per_ticker_timeseries_features(
            combined_val_in,
            base_cols_snapshot,
            ticker_col="numerai_ticker",
            era_col="era",
            ticker_features=ticker_timeseries_features,
            ticker_windows=ticker_rolling_windows,
            ticker_lag_steps=ticker_lag_steps,
            top_n_cols=ticker_top_n_features,
        )
        validation_df = (
            combined_val_out.loc[combined_val_out["_is_val_split_"]]
            .drop(columns=["_is_val_split_"])
            .copy()
        )
        validation_df.index = orig_val_index

        common_added = [c for c in added_train_cols if c in added_val_cols and c not in feature_columns]
        feature_columns = feature_columns + common_added

    # 6. Apply `era_stride` strictly once to `train_df` after pre-stride ticker history features (validation keeps every era)
    if not strided_early and int(era_stride) > 1:
        kept_train_eras = sorted(train_df["era"].unique())[:: int(era_stride)]
        train_df = train_df.loc[train_df["era"].isin(kept_train_eras)]

    # 7. Optional training-only downsampling (validation_df is never downsampled)
    train_df = downsample_training_data(
        train_df,
        downsample_mode=downsample_mode,
        downsample_frac=downsample_frac,
        era_col="era",
        random_state=random_state,
    )

    assert not any(
        column in FORBIDDEN_FEATURE_COLS or column == target_col or column.startswith("target")
        for column in feature_columns
    )
    return train_df, validation_df, feature_columns, feature_metadata


def _source_files_fingerprint(
    data_version: str,
    data_folder: str,
    kaggle_dataset: str,
    workspace_dir: Path,
) -> Optional[List[List[Any]]]:
    dataset_slug_name = kaggle_dataset.split("/")[-1]
    candidate_dirs = (
        Path("/kaggle/input/datasets") / kaggle_dataset,
        Path("/kaggle/input/datasets") / kaggle_dataset / data_version,
        Path("/kaggle/input") / dataset_slug_name,
        Path("/kaggle/input") / f"numerai-data-{data_version.replace('.', '-')}",
        workspace_dir / data_folder / data_version,
        workspace_dir / data_folder / dataset_slug_name,
    )
    data_root = next((found for cand in candidate_dirs if (found := _find_dataset_root(cand)) is not None), None)
    if data_root is None and _has_complete_dataset(workspace_dir / data_folder):
        data_root = workspace_dir / data_folder
    if data_root is None and Path("/kaggle/input").is_dir():
        data_root = _find_dataset_root(Path("/kaggle/input"))
    if data_root is None:
        return None
    train_p, val_p, feat_p = _resolve_numerai_paths(data_root)
    fp: List[List[Any]] = []
    for p in (train_p, val_p, feat_p):
        st = p.stat()
        fp.append([p.name, int(st.st_mtime_ns), int(st.st_size)])
    return fp


def load_unity_numerai_data_cached(
    data_version: str = "v6.0",
    data_folder: str = "data_folder",
    kaggle_dataset: str = KAGGLE_DATASET_SLUG,
    feature_set: str = "small",
    target_col: str = "target",
    era_stride: int = 4,
    downsample_mode: Optional[str] = None,
    downsample_frac: float = 1.0,
    random_state: int = 42,
    ticker_timeseries_features: Optional[str] = None,
    ticker_rolling_windows: Sequence[int] = (4, 12),
    ticker_lag_steps: Optional[Sequence[int]] = None,
    ticker_top_n_features: int = 4,
    workspace_root: Optional[Any] = None,
    auto_download: bool = True,
    use_numerapi_fallback: bool = True,
    force_reload: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
    """Cached wrapper around `load_unity_numerai_data` persisting processed Parquets & metadata in `<workspace>/<data_folder>/processed_cache/`.

    Caches the full strided `(train_df, validation_df)` and applies `downsample_training_data(...)`
    in-memory to `train_df` only so changing `downsample_mode` / `downsample_frac` reuses the cache.
    """
    workspace_dir = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    if workspace_dir.name.lower() == "notebooks":
        workspace_dir = workspace_dir.parent

    cache_dir = workspace_dir / data_folder / "processed_cache"
    source_fp = _source_files_fingerprint(data_version, data_folder, kaggle_dataset, workspace_dir)
    key_payload = {
        "data_version": str(data_version),
        "feature_set": str(feature_set),
        "era_stride": int(era_stride),
        "target_col": str(target_col),
        "ticker_timeseries_features": ticker_timeseries_features,
        "ticker_rolling_windows": list(ticker_rolling_windows) if ticker_rolling_windows is not None else None,
        "ticker_lag_steps": list(ticker_lag_steps) if ticker_lag_steps is not None else None,
        "ticker_top_n_features": int(ticker_top_n_features),
        "kaggle_dataset": str(kaggle_dataset),
        "source_fp": source_fp,
    }
    cache_hash = hashlib.blake2b(json.dumps(key_payload, sort_keys=True).encode("utf-8"), digest_size=12).hexdigest()
    train_cache = cache_dir / f"train_{cache_hash}.parquet"
    val_cache = cache_dir / f"validation_{cache_hash}.parquet"
    meta_cache = cache_dir / f"meta_{cache_hash}.json"

    if not force_reload and source_fp is not None and train_cache.is_file() and val_cache.is_file() and meta_cache.is_file():
        try:
            with meta_cache.open("r", encoding="utf-8") as mf:
                meta = json.load(mf)
            if meta.get("key_payload") == key_payload:
                train_df = pd.read_parquet(train_cache)
                validation_df = pd.read_parquet(val_cache)
                feature_columns = list(meta["feature_columns"])
                feature_metadata = dict(meta["feature_metadata"])
                valid_hit = (
                    len(train_df) == int(meta["n_train_rows"])
                    and len(validation_df) == int(meta["n_val_rows"])
                    and list(train_df.columns) == list(meta["train_columns"])
                    and list(validation_df.columns) == list(meta["val_columns"])
                    and all(train_df[c].dtype == np.int8 and validation_df[c].dtype == np.int8 for c in feature_columns)
                    and train_df["era"].dtype == np.int32
                    and validation_df["era"].dtype == np.int32
                    and train_df[target_col].dtype == np.float32
                    and validation_df[target_col].dtype == np.float32
                )
                if valid_hit:
                    train_df = downsample_training_data(
                        train_df,
                        downsample_mode=downsample_mode,
                        downsample_frac=downsample_frac,
                        era_col="era",
                        random_state=random_state,
                    )
                    return train_df, validation_df, feature_columns, feature_metadata
        except Exception:
            pass

    train_df, validation_df, feature_columns, feature_metadata = load_unity_numerai_data(
        data_version=data_version,
        data_folder=data_folder,
        kaggle_dataset=kaggle_dataset,
        feature_set=feature_set,
        target_col=target_col,
        era_stride=era_stride,
        downsample_mode=None,
        downsample_frac=1.0,
        random_state=random_state,
        ticker_timeseries_features=ticker_timeseries_features,
        ticker_rolling_windows=ticker_rolling_windows,
        ticker_lag_steps=ticker_lag_steps,
        ticker_top_n_features=ticker_top_n_features,
        workspace_root=workspace_root,
        auto_download=auto_download,
        use_numerapi_fallback=use_numerapi_fallback,
    )

    try:
        source_fp_after = _source_files_fingerprint(data_version, data_folder, kaggle_dataset, workspace_dir)
        key_payload["source_fp"] = source_fp_after
        cache_hash = hashlib.blake2b(json.dumps(key_payload, sort_keys=True).encode("utf-8"), digest_size=12).hexdigest()
        train_cache = cache_dir / f"train_{cache_hash}.parquet"
        val_cache = cache_dir / f"validation_{cache_hash}.parquet"
        meta_cache = cache_dir / f"meta_{cache_hash}.json"
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(train_cache, index=False)
        validation_df.to_parquet(val_cache, index=False)
        with meta_cache.open("w", encoding="utf-8") as mf:
            json.dump(
                {
                    "key_payload": key_payload,
                    "feature_columns": feature_columns,
                    "feature_metadata": feature_metadata,
                    "n_train_rows": int(len(train_df)),
                    "n_val_rows": int(len(validation_df)),
                    "train_columns": [str(c) for c in train_df.columns],
                    "val_columns": [str(c) for c in validation_df.columns],
                },
                mf,
            )
    except OSError as cache_err:
        warnings.warn(f"Could not write processed data cache ({cache_err}); continuing without cache.")

    train_df = downsample_training_data(
        train_df,
        downsample_mode=downsample_mode,
        downsample_frac=downsample_frac,
        era_col="era",
        random_state=random_state,
    )
    return train_df, validation_df, feature_columns, feature_metadata


V6_CONTRIBUTION_KEYWORDS: Tuple[str, ...] = ("meta", "benchmark", "neutralizer", "example")
V6_BENCHMARK_FALLBACK_ORDER: Tuple[str, ...] = (
    "validation_benchmark_models.parquet",
    "validation_example_preds.parquet",
)


def discover_and_fetch_v6_meta_or_benchmark(
    data_version: str = "v6.0",
    data_folder: str = "data_folder",
    workspace_root: Optional[Any] = None,
    auto_download: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """List live NumerAPI datasets for `data_version`, fetch true `meta_model.parquet` if published, or fall back honestly to BMC proxy.

    Honesty rules:
    - Never invents `meta_model.parquet` and never renames a benchmark/example file to `meta_model.parquet`.
    - If `v6.0/meta_model.parquet` exists, downloads it to `<workspace>/<data_folder>/<data_version>/meta_model.parquet`
      and returns `basis_kind='true_mmc'`, `basis_label='True MMC / NMMC'`.
    - Otherwise, prints a clear message that no true Meta Model is published yet, downloads the v6 benchmark/example
      predictions Parquet (`validation_example_preds.parquet` or `validation_benchmark_models.parquet`) for BMC proxy,
      and labels it `basis_kind='bmc_proxy'`, `basis_label='BMC proxy (benchmark, NOT true MMC)'`.
    - If neither file is available locally or downloadable, returns `basis_kind='corr_only'`, `basis_label='CORR only'`.
    """
    workspace_dir = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    if workspace_dir.name.lower() == "notebooks":
        workspace_dir = workspace_dir.parent

    save_dir = workspace_dir / data_folder / data_version
    prefix = f"{data_version}/"

    all_datasets: List[str] = []
    filtered_datasets: List[str] = []
    downloaded_files: List[Tuple[str, str]] = []
    api_error: Optional[str] = None
    napi = None

    if NumerAPI is not None:
        try:
            napi = NumerAPI()
            all_datasets = sorted([p for p in napi.list_datasets() if str(p).startswith(prefix)])
            filtered_datasets = [
                p for p in all_datasets if any(kw in p.lower() for kw in V6_CONTRIBUTION_KEYWORDS)
            ]
        except Exception as exc:
            api_error = f"{type(exc).__name__}: {exc}"
    else:
        api_error = "numerapi package is not installed"

    if verbose:
        print(f"=== NumerAPI Dataset Discovery ({data_version}) ===")
        if all_datasets:
            print(f"All {data_version} files found ({len(all_datasets)}):")
            for item in all_datasets:
                print(f"  - {item}")
            print(f"\nFiltered ({', '.join(V6_CONTRIBUTION_KEYWORDS)}) ({len(filtered_datasets)}):")
            for item in filtered_datasets:
                print(f"  - {item}")
        else:
            print(f"Could not list {data_version} datasets via NumerAPI ({api_error or 'empty response'}).")

    def _find_local_file(filename: str) -> Optional[Path]:
        candidates = [
            save_dir / filename,
            workspace_dir / data_folder / filename,
        ]
        if Path("/kaggle/input").is_dir():
            candidates.extend(sorted(Path("/kaggle/input").rglob(filename)))
        return next((p for p in candidates if p.is_file()), None)

    # 1. Check for true Meta Model (`v6.0/meta_model.parquet`)
    meta_remote = f"{prefix}meta_model.parquet"
    meta_local_path = _find_local_file("meta_model.parquet")
    if meta_local_path is None and meta_remote in all_datasets and auto_download and napi is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        dest_meta = save_dir / "meta_model.parquet"
        try:
            napi.download_dataset(meta_remote, str(dest_meta))
            if dest_meta.is_file():
                meta_local_path = dest_meta
                downloaded_files.append((meta_remote, str(dest_meta)))
        except Exception as exc:
            warnings.warn(f"Failed to download {meta_remote}: {exc}")

    if meta_local_path is not None:
        reason = f"True Meta Model found at {meta_local_path}. True MMC / NMMC can be computed."
        if verbose:
            print(f"\n[True MMC / NMMC] {reason}")
        return {
            "data_version": data_version,
            "all_datasets": all_datasets,
            "filtered_datasets": filtered_datasets,
            "basis_kind": "true_mmc",
            "basis_label": "True MMC / NMMC",
            "meta_model_path": meta_local_path,
            "benchmark_path": None,
            "contribution_path": meta_local_path,
            "mmc_is_proxy": False,
            "downloaded_files": downloaded_files,
            "reason": reason,
        }

    # True meta_model.parquet does NOT exist — never invent or rename another file!
    no_meta_msg = (
        f"No {data_version} meta_model.parquet published yet, so true MMC / NMMC cannot be computed in backtest."
    )
    if verbose:
        print(f"\n[MMC Status] {no_meta_msg}")

    # 2. Clean BMC fallback: download v6 benchmark / example predictions file if available
    benchmark_local_path: Optional[Path] = None
    chosen_remote: Optional[str] = None
    for fname in V6_BENCHMARK_FALLBACK_ORDER:
        local_hit = _find_local_file(fname)
        if local_hit is not None:
            benchmark_local_path = local_hit
            chosen_remote = f"{prefix}{fname}"
            break
        remote_name = f"{prefix}{fname}"
        if remote_name in all_datasets and auto_download and napi is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            dest_bench = save_dir / fname
            try:
                if verbose:
                    print(f"Downloading BMC proxy file: {remote_name} -> {dest_bench}")
                napi.download_dataset(remote_name, str(dest_bench))
                if dest_bench.is_file():
                    benchmark_local_path = dest_bench
                    chosen_remote = remote_name
                    downloaded_files.append((remote_name, str(dest_bench)))
                    break
            except Exception as exc:
                warnings.warn(f"Failed to download {remote_name}: {exc}")

    if benchmark_local_path is not None:
        is_actual_benchmark = "benchmark" in benchmark_local_path.name.lower()
        if is_actual_benchmark:
            basis_kind = "bmc_proxy"
            basis_label = "BMC proxy (benchmark, NOT true MMC)"
            tag = "BMC Proxy Fallback"
        else:
            basis_kind = "example_proxy"
            basis_label = "Example-model proxy (NOT true MMC / NOT true BMC)"
            tag = "Example-Model Proxy Fallback"
        reason = (
            f"{no_meta_msg} Falling back to {benchmark_local_path.name} (saved at {benchmark_local_path}) "
            f"to compute {basis_label}."
        )
        if verbose:
            print(f"[{tag}] Using {benchmark_local_path.name} saved at: {benchmark_local_path}")
            print(f"  -> Basis everywhere will be labeled: {basis_label}")
        return {
            "data_version": data_version,
            "all_datasets": all_datasets,
            "filtered_datasets": filtered_datasets,
            "basis_kind": basis_kind,
            "basis_label": basis_label,
            "meta_model_path": None,
            "benchmark_path": benchmark_local_path,
            "contribution_path": benchmark_local_path,
            "mmc_is_proxy": True,
            "downloaded_files": downloaded_files,
            "chosen_remote": chosen_remote,
            "reason": reason,
        }

    reason = (
        f"{no_meta_msg} No local or downloadable benchmark/example predictions file was found "
        f"({api_error or 'not available'}). Staking will use CORR only."
    )
    if verbose:
        print(f"[CORR Only] {reason}")
    return {
        "data_version": data_version,
        "all_datasets": all_datasets,
        "filtered_datasets": filtered_datasets,
        "basis_kind": "corr_only",
        "basis_label": "CORR only",
        "meta_model_path": None,
        "benchmark_path": None,
        "contribution_path": None,
        "mmc_is_proxy": False,
        "downloaded_files": downloaded_files,
        "reason": reason,
    }
