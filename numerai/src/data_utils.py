"""data_utils.py — Kaggle (`kagglehub` + `KaggleApi`) and `numerapi` data loading for Numerai Classic v6.0 ("Unity")."""

from __future__ import annotations

import hashlib
import json
import warnings
from functools import partial
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
VALID_TICKER_TIMESERIES_MODES: Tuple[str, ...] = (
    "rolling_mean",
    "momentum_diff",
    "weeks_since_high",
    "rolling_rank",
    "weeks_since_change",
    "expanding_stats",
    "raw_lags",
)
VALID_DOWNSAMPLE_MODES: Tuple[str, ...] = (
    "within_era",
    "recent",
    "earliest",
    "uniform",
    "random_eras",
)
V6_CONTRIBUTION_KEYWORDS: Tuple[str, ...] = ("meta", "benchmark", "neutralizer", "example")
V6_BENCHMARK_FALLBACK_ORDER: Tuple[str, ...] = (
    "validation_benchmark_models.parquet",
    "validation_example_preds.parquet",
)


# ---------------------------------------------------------------------------
# HTTP Error Formatting & Local/Remote Dataset Resolution Helpers
# ---------------------------------------------------------------------------

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


def _resolve_workspace_root(workspace_root: Optional[Any]) -> Path:
    """Normalize an explicit or implicit workspace root, stepping out of `notebooks/` if needed."""
    workspace_dir = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    if workspace_dir.name.lower() == "notebooks":
        workspace_dir = workspace_dir.parent
    return workspace_dir


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


def _locate_local_dataset_root(
    data_version: str,
    data_folder: str,
    kaggle_dataset: str,
    workspace_dir: Path,
) -> Optional[Path]:
    """Search Kaggle mounts and version-matched workspace folders without recursing into other versions."""
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
    return data_root


def _download_dataset_via_kagglehub(kaggle_dataset: str) -> Optional[Path]:
    """Attempt to resolve or download the Numerai dataset through `kagglehub`."""
    if kagglehub is None:
        return None
    try:
        resolved_dir = Path(kagglehub.dataset_download(kaggle_dataset))
        return _find_dataset_root(resolved_dir)
    except (KaggleApiHTTPError, ColabHTTPError) as http_error:
        warnings.warn(_format_http_error(f"kagglehub HTTP error for '{kaggle_dataset}'", http_error))
    except KAGGLEHUB_EXCEPTIONS as hub_error:
        warnings.warn(f"kagglehub could not resolve '{kaggle_dataset}' ({type(hub_error).__name__}): {hub_error}")
    except (RequestException, OSError, ValueError) as io_error:
        warnings.warn(_format_http_error(f"kagglehub download failed for '{kaggle_dataset}'", io_error))
    return None


def _download_dataset_via_kaggle_api(kaggle_dataset: str, target_download_dir: Path) -> Optional[Path]:
    """Attempt to download and unzip the Numerai dataset through the official `KaggleApi` client."""
    if KaggleApi is None:
        return None
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
        return _find_dataset_root(target_download_dir)
    except (KaggleApiException, HTTPError) as api_error:
        warnings.warn(_format_http_error(f"KaggleApi HTTP error for '{kaggle_dataset}'", api_error))
    except (OSError, IOError, ValueError, RuntimeError, RequestException) as local_error:
        warnings.warn(
            f"KaggleApi local/auth error for '{kaggle_dataset}' ({type(local_error).__name__}): {local_error}"
        )
    return None


def _download_dataset_via_numerapi(data_version: str, target_download_dir: Path) -> Optional[Path]:
    """Download core Numerai Parquet and metadata files through `numerapi.NumerAPI`."""
    if NumerAPI is None:
        return None
    target_download_dir.mkdir(parents=True, exist_ok=True)
    numerai_client = NumerAPI()
    try:
        for file_name in NUMERAPI_DATASET_FILES:
            numerai_client.download_dataset(
                f"{data_version}/{file_name}",
                str(target_download_dir / file_name),
            )
        return _find_dataset_root(target_download_dir)
    except ValueError as graphql_error:
        raise FileNotFoundError(
            f"NumerAPI GraphQL error for '{data_version}': {graphql_error}"
        ) from graphql_error
    except (HTTPError, RequestException, OSError) as download_error:
        raise ConnectionError(
            _format_http_error(f"NumerAPI download failed for '{data_version}'", download_error)
        ) from download_error


def _acquire_dataset_root(
    data_version: str,
    data_folder: str,
    kaggle_dataset: str,
    workspace_dir: Path,
    *,
    auto_download: bool,
    use_numerapi_fallback: bool,
) -> Path:
    """Locate the dataset locally or download it via `kagglehub` -> `KaggleApi` -> `NumerAPI`."""
    data_root = _locate_local_dataset_root(data_version, data_folder, kaggle_dataset, workspace_dir)
    if data_root is None and auto_download:
        target_download_dir = workspace_dir / data_folder / data_version
        data_root = _download_dataset_via_kagglehub(kaggle_dataset)
        if data_root is None:
            data_root = _download_dataset_via_kaggle_api(kaggle_dataset, target_download_dir)
        if data_root is None and use_numerapi_fallback:
            data_root = _download_dataset_via_numerapi(data_version, target_download_dir)

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
    return data_root


def _resolve_feature_and_request_columns(
    train_path: Path,
    validation_path: Path,
    features_path: Path,
    feature_set: str,
    target_col: str,
) -> Tuple[List[str], List[str], List[str], Dict[str, Any]]:
    """Read `features.json` and Parquet schemas to build validated feature and column-request lists."""
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
    return feature_columns, train_req_cols, val_req_cols, feature_metadata


# ---------------------------------------------------------------------------
# Causal Per-Ticker Time-Series Feature Primitives & Builders
# ---------------------------------------------------------------------------

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


def _compute_weeks_since_change_series(series: pd.Series) -> np.ndarray:
    """Compute causal weeks-since-change for a single ticker `pd.Series` group."""
    return _compute_weeks_since_change_1d(series.to_numpy(dtype=np.float64, copy=False))


def _compute_weeks_since_high_series(series: pd.Series, window: int) -> np.ndarray:
    """Compute causal weeks-since-high for a single ticker `pd.Series` group over `window` steps."""
    return _compute_weeks_since_high_1d(series.to_numpy(dtype=np.float64, copy=False), window)


def _compute_expanding_mean_series(series: pd.Series) -> pd.Series:
    """Compute causal expanding mean for a single ticker `pd.Series` group."""
    return series.expanding(min_periods=1).mean()


def _compute_expanding_min_series(series: pd.Series) -> pd.Series:
    """Compute causal expanding minimum for a single ticker `pd.Series` group."""
    return series.expanding(min_periods=1).min()


def _compute_expanding_max_series(series: pd.Series) -> pd.Series:
    """Compute causal expanding maximum for a single ticker `pd.Series` group."""
    return series.expanding(min_periods=1).max()


def _compute_rolling_mean_series(series: pd.Series, window: int) -> pd.Series:
    """Compute causal rolling mean over `window` steps for a single ticker `pd.Series` group."""
    return series.rolling(window=window, min_periods=1).mean()


def _compute_rolling_rank_series(series: pd.Series, window: int) -> pd.Series:
    """Compute causal rolling percentile rank over `window` steps for a single ticker `pd.Series` group."""
    return series.rolling(window=window, min_periods=1).rank(pct=True)


def _parse_ticker_timeseries_modes(ticker_features: Optional[Any]) -> set:
    """Parse `ticker_features` (`True`, `'all'`, or pipe/comma-separated mode names) into a validated set."""
    if ticker_features in (None, False, "None", "none", "null", ""):
        return set()
    if ticker_features is True:
        return set(VALID_TICKER_TIMESERIES_MODES)
    active_modes = {
        part.strip()
        for part in str(ticker_features).lower().replace(",", "|").split("|")
        if part.strip() in VALID_TICKER_TIMESERIES_MODES or part.strip() in ("all", "true")
    }
    if "all" in active_modes or "true" in active_modes:
        return set(VALID_TICKER_TIMESERIES_MODES)
    return active_modes


def _normalize_ticker_windows(ticker_windows: Sequence[int]) -> List[int]:
    """Validate rolling window sizes (> 1), falling back to `[4, 12]`."""
    windows = (
        [int(w) for w in ticker_windows if int(w) > 1]
        if isinstance(ticker_windows, (list, tuple, set))
        else [4, 12]
    )
    return windows or [4, 12]


def _normalize_ticker_lag_steps(ticker_lag_steps: Optional[Sequence[int]]) -> List[int]:
    """Validate positive lag offsets (>= 1), falling back to `[1, 2, 4]`."""
    lag_steps = (
        [int(k) for k in ticker_lag_steps if int(k) >= 1]
        if isinstance(ticker_lag_steps, (list, tuple, set))
        else [1, 2, 4]
    )
    return lag_steps or [1, 2, 4]


def _build_ticker_timeseries_column_order(
    chosen_cols: Sequence[str],
    active_modes: set,
    windows: Sequence[int],
    lag_steps: Sequence[int],
) -> List[str]:
    """Return the deterministic output column order shared by both Polars and Pandas implementations."""
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
    return added_col_names


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


def _build_polars_ticker_feature_expressions(
    pl_df: Any,
    chosen_cols: Sequence[str],
    *,
    ticker_col: str,
    active_modes: set,
    windows: Sequence[int],
    lag_steps: Sequence[int],
    pos_in_group: Optional[np.ndarray],
) -> Tuple[List[Any], Dict[str, np.ndarray], List[str]]:
    """Construct Polars expressions and grouped NumPy arrays for requested per-ticker time-series modes."""
    need_pos = pos_in_group is not None
    quantize_cols: List[str] = []
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

        if "raw_lags" in active_modes:
            for lag_k in lag_steps:
                cname = f"{col}_ticker_lag{lag_k}"
                exprs.append(
                    c_expr.shift(lag_k).over(ticker_col).fill_null(2.0).round(0).clip(0.0, 4.0).cast(pl.Int8).alias(cname)
                )

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
                extra_numpy_series[cname] = np.clip(np.rint(wsh_arr * 4.0 / max(1.0, float(w - 1))), 0, 4).astype(np.int8)
            if "rolling_rank" in active_modes:
                cname = f"{col}_ticker_roll_rank{w}"
                denom = pl.min_horizontal(c_expr.cum_count().over(ticker_col), pl.lit(w)).cast(pl.Float64)
                rr_expr = (
                    c_expr.rolling_rank(window_size=w, min_samples=1, method="average").over(ticker_col) / denom
                ).fill_null(0.5)
                exprs.append(rr_expr.alias(cname))
                quantize_cols.append(cname)

    return exprs, extra_numpy_series, quantize_cols


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
    need_pos = ("weeks_since_high" in active_modes) or ("weeks_since_change" in active_modes)
    pos_in_group = (
        pl_df.select(pl.int_range(0, pl.len()).over(ticker_col).alias("_pos_")).get_column("_pos_").to_numpy()
        if need_pos
        else None
    )

    exprs, extra_numpy_series, quantize_cols = _build_polars_ticker_feature_expressions(
        pl_df,
        chosen_cols,
        ticker_col=ticker_col,
        active_modes=active_modes,
        windows=windows,
        lag_steps=lag_steps,
        pos_in_group=pos_in_group,
    )
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

    added_col_names = _build_ticker_timeseries_column_order(chosen_cols, active_modes, windows, lag_steps)
    added_df = pl_df.sort("_orig_row_idx_").select(added_col_names).to_pandas()
    added_df.index = df.index
    for cname in added_col_names:
        if added_df[cname].dtype != np.int8:
            added_df[cname] = added_df[cname].to_numpy(dtype=np.int8, copy=False)
    return pd.concat([df, added_df], axis=1), added_col_names


def _compute_per_ticker_timeseries_pandas(
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
    """Pandas fallback implementation of `compute_per_ticker_timeseries_features` when Polars is unavailable."""
    sorted_df = df.sort_values([ticker_col, era_col], kind="stable")
    sorted_eras = sorted_df[era_col].to_numpy()

    new_cols_dict: Dict[str, pd.Series] = {}
    for col in chosen_cols:
        col_float = pd.to_numeric(sorted_df[col], errors="coerce").fillna(2.0).astype(np.float64)
        col_grouped = col_float.groupby(sorted_df[ticker_col], sort=False)

        if "weeks_since_change" in active_modes:
            wsc_series = col_grouped.transform(_compute_weeks_since_change_series)
            new_cols_dict[f"{col}_ticker_wks_change"] = pd.Series(
                quantize_to_5bins(wsc_series.to_numpy(dtype=np.float64), eras=sorted_eras, quantize_scope=quantize_scope),
                index=sorted_df.index,
                dtype=np.int8,
            )

        if "expanding_stats" in active_modes:
            exp_mean = col_grouped.transform(_compute_expanding_mean_series)
            exp_min = col_grouped.transform(_compute_expanding_min_series)
            exp_max = col_grouped.transform(_compute_expanding_max_series)
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
            roll_mean = col_grouped.transform(partial(_compute_rolling_mean_series, window=w)) if need_roll_mean else None
            if "rolling_mean" in active_modes and roll_mean is not None:
                new_cols_dict[f"{col}_ticker_roll_mean{w}"] = pd.Series(
                    quantize_to_5bins(roll_mean.to_numpy(), eras=sorted_eras, quantize_scope=quantize_scope),
                    index=sorted_df.index,
                    dtype=np.int8,
                )
            if "momentum_diff" in active_modes and roll_mean is not None:
                diff_vals = (col_float - roll_mean).to_numpy()
                new_cols_dict[f"{col}_ticker_mom_diff{w}"] = pd.Series(
                    quantize_to_5bins(diff_vals, eras=sorted_eras, quantize_scope=quantize_scope),
                    index=sorted_df.index,
                    dtype=np.int8,
                )
            if "weeks_since_high" in active_modes:
                wsh_series = col_grouped.transform(partial(_compute_weeks_since_high_series, window=w))
                scaled_wsh = np.clip(
                    np.rint(wsh_series.to_numpy(dtype=np.float64) * 4.0 / max(1.0, float(w - 1))),
                    0,
                    4,
                ).astype(np.int8)
                new_cols_dict[f"{col}_ticker_wks_high{w}"] = pd.Series(scaled_wsh, index=sorted_df.index, dtype=np.int8)
            if "rolling_rank" in active_modes:
                roll_rank = col_grouped.transform(partial(_compute_rolling_rank_series, window=w))
                new_cols_dict[f"{col}_ticker_roll_rank{w}"] = pd.Series(
                    quantize_to_5bins(roll_rank.fillna(0.5).to_numpy(dtype=np.float64), eras=sorted_eras, quantize_scope=quantize_scope),
                    index=sorted_df.index,
                    dtype=np.int8,
                )

    if not new_cols_dict:
        return df, []

    added_df = pd.DataFrame(new_cols_dict, index=sorted_df.index).reindex(df.index)
    return pd.concat([df, added_df], axis=1), list(new_cols_dict.keys())


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
    if ticker_col not in df.columns or era_col not in df.columns:
        return df, []
    active_modes = _parse_ticker_timeseries_modes(ticker_features)
    if not active_modes:
        return df, []

    chosen_cols = [str(c) for c in list(base_feature_cols)[: max(1, int(top_n_cols))] if c in df.columns]
    if not chosen_cols:
        return df, []

    windows = _normalize_ticker_windows(ticker_windows)
    lag_steps = _normalize_ticker_lag_steps(ticker_lag_steps)

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
    return _compute_per_ticker_timeseries_pandas(
        df,
        chosen_cols,
        ticker_col=ticker_col,
        era_col=era_col,
        active_modes=active_modes,
        windows=windows,
        lag_steps=lag_steps,
        quantize_scope=quantize_scope,
    )


# ---------------------------------------------------------------------------
# Panel Downsampling Helpers
# ---------------------------------------------------------------------------

def _normalize_downsample_mode(downsample_mode: Optional[str]) -> Optional[str]:
    """Normalize inactive downsampling sentinels to `None` and active names to lowercase."""
    if downsample_mode in (None, False, "None", "none", "null", ""):
        return None
    return str(downsample_mode).strip().lower()


def _resolve_val_downsample_config(
    *,
    downsample_mode: Optional[str],
    downsample_frac: float,
    val_downsample_mode: Optional[str],
    val_downsample_frac: Optional[float],
) -> Tuple[Optional[str], float]:
    """Validate and resolve validation downsampling knobs (`None`, `'same'`, or any `VALID_DOWNSAMPLE_MODES`)."""
    train_mode = _normalize_downsample_mode(downsample_mode)
    raw_val_mode = _normalize_downsample_mode(val_downsample_mode)
    if raw_val_mode is not None and raw_val_mode != "same" and raw_val_mode not in VALID_DOWNSAMPLE_MODES:
        raise ValueError(
            f"val_downsample_mode must be None, 'same', or one of {VALID_DOWNSAMPLE_MODES}, got {val_downsample_mode!r}"
        )
    if val_downsample_frac is not None:
        v_frac = float(val_downsample_frac)
        if not (0.0 < v_frac <= 1.0):
            raise ValueError(f"val_downsample_frac must be in (0.0, 1.0], got {val_downsample_frac!r}")
    else:
        v_frac = None

    if raw_val_mode is None:
        return None, 1.0 if v_frac is None else v_frac

    resolved_mode = train_mode if raw_val_mode == "same" else raw_val_mode
    resolved_frac = float(downsample_frac) if v_frac is None else v_frac
    if not (0.0 < resolved_frac <= 1.0):
        raise ValueError(f"val_downsample_frac must be in (0.0, 1.0], got {resolved_frac!r}")
    return resolved_mode, resolved_frac


def _build_within_era_keep_mask(
    era_vals: np.ndarray,
    frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a boolean row mask keeping `frac` of rows randomly within every era."""
    order = np.argsort(era_vals, kind="stable")
    sorted_eras = era_vals[order]
    _, start_indices, counts = np.unique(sorted_eras, return_index=True, return_counts=True)
    keep_mask = np.zeros(len(era_vals), dtype=bool)
    for start, count in zip(start_indices, counts):
        positions = order[start : start + count]
        n_keep = max(1, int(round(int(count) * frac)))
        if n_keep >= int(count):
            keep_mask[positions] = True
        else:
            chosen = rng.choice(positions, size=n_keep, replace=False)
            keep_mask[chosen] = True
    return keep_mask


def _select_downsampled_eras(
    unique_eras: np.ndarray,
    mode: str,
    n_keep_eras: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select the subset of unique eras according to `recent`, `earliest`, `uniform`, or `random_eras`."""
    n_eras = len(unique_eras)
    if mode == "recent":
        return unique_eras[-n_keep_eras:]
    if mode == "earliest":
        return unique_eras[:n_keep_eras]
    if mode == "uniform":
        idx = np.round(np.linspace(0, n_eras - 1, n_keep_eras)).astype(np.int64)
        return unique_eras[idx]
    if mode == "random_eras":
        return np.sort(rng.choice(unique_eras, size=n_keep_eras, replace=False))
    raise ValueError(f"Unsupported downsample_mode '{mode}'.")


def downsample_panel_data(
    df: pd.DataFrame,
    *,
    downsample_mode: Optional[str] = None,
    downsample_frac: float = 1.0,
    era_col: str = "era",
    random_state: int = 42,
) -> pd.DataFrame:
    """Deterministically downsample a Numerai panel DataFrame (train or validation) while preserving chronological era order.

    Supported modes:
      - `None`: keep 100% of rows.
      - `'within_era'`: keep every era, randomly sampling `downsample_frac` of rows within each era.
      - `'recent'`: keep the most recent `downsample_frac` fraction of unique eras.
      - `'earliest'`: keep the earliest `downsample_frac` fraction of unique eras.
      - `'uniform'`: keep evenly spaced eras spanning the full range.
      - `'random_eras'`: randomly sample `downsample_frac` of unique eras without replacement.
    """
    mode = _normalize_downsample_mode(downsample_mode)
    if mode is not None and mode not in VALID_DOWNSAMPLE_MODES:
        raise ValueError(
            f"downsample_mode must be None or one of {VALID_DOWNSAMPLE_MODES}, got {downsample_mode!r}"
        )
    frac = float(downsample_frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"downsample_frac must be in (0.0, 1.0], got {downsample_frac!r}")
    if mode is None or frac >= 1.0 or len(df) == 0:
        return df
    if era_col not in df.columns:
        raise KeyError(f"Column '{era_col}' not found in DataFrame.")

    rng = np.random.default_rng(int(random_state))
    era_vals = df[era_col].to_numpy()
    if mode == "within_era":
        keep_mask = _build_within_era_keep_mask(era_vals, frac, rng)
        return df.loc[keep_mask].copy()

    unique_eras = np.sort(pd.unique(era_vals))
    n_keep_eras = max(1, int(round(len(unique_eras) * frac)))
    if n_keep_eras >= len(unique_eras):
        return df
    kept_eras = _select_downsampled_eras(unique_eras, mode, n_keep_eras, rng)
    return df.loc[df[era_col].isin(kept_eras)].copy()


def downsample_training_data(
    train_df: pd.DataFrame,
    *,
    downsample_mode: Optional[str] = None,
    downsample_frac: float = 1.0,
    era_col: str = "era",
    random_state: int = 42,
) -> pd.DataFrame:
    """Backward-compatible alias for `downsample_panel_data` applied to `train_df`."""
    return downsample_panel_data(
        train_df,
        downsample_mode=downsample_mode,
        downsample_frac=downsample_frac,
        era_col=era_col,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Polars & Pandas Parquet Loading, Era Unification, and Int8 Casting Helpers
# ---------------------------------------------------------------------------

def _format_polars_date_column_iso(series: Any) -> Any:
    """Convert a Polars `Date`, `Datetime`, or string-like date column into `YYYY-MM-DD` UTF-8 strings."""
    if series.dtype in (pl.Date, pl.Datetime):
        return series.dt.strftime("%Y-%m-%d")
    return pl.Series("date", pd.to_datetime(series.to_pandas()).dt.strftime("%Y-%m-%d").to_numpy())


def _unify_polars_split_eras(train_pl: Any, val_pl: Any) -> Tuple[Any, Any]:
    """Ensure both Polars train and validation frames share a monotonic `Int32` `era` column."""
    if "era" in train_pl.columns and "era" in val_pl.columns:
        tr_num = train_pl.get_column("era").cast(pl.Int32, strict=False)
        va_num = val_pl.get_column("era").cast(pl.Int32, strict=False)
        if tr_num.null_count() == 0 and va_num.null_count() == 0:
            return (
                train_pl.with_columns(tr_num.alias("era")),
                val_pl.with_columns(va_num.alias("era")),
            )
        tr_keys = train_pl.get_column("era").cast(pl.Utf8)
        va_keys = val_pl.get_column("era").cast(pl.Utf8)
    elif "date" in train_pl.columns and "date" in val_pl.columns:
        tr_keys = _format_polars_date_column_iso(train_pl.get_column("date"))
        va_keys = _format_polars_date_column_iso(val_pl.get_column("date"))
    else:
        raise KeyError("Expected 'era' or 'date' column in both train and validation Parquet files.")

    all_keys = sorted(set(tr_keys.unique().to_list()) | set(va_keys.unique().to_list()))
    key_to_id = {k: np.int32(pos + 1) for pos, k in enumerate(all_keys)}
    return (
        train_pl.with_columns(tr_keys.replace_strict(key_to_id, return_dtype=pl.Int32).alias("era")),
        val_pl.with_columns(va_keys.replace_strict(key_to_id, return_dtype=pl.Int32).alias("era")),
    )


def _polars_cast_features_to_int8(pl_df: Any, feature_columns: List[str]) -> Any:
    """Cast non-Int8 feature columns to Int8 in parallel inside Polars, preserving `[0, 1]` -> `0..4` scaling."""
    schema = pl_df.schema
    non_int8 = [col for col in feature_columns if schema.get(col) != pl.Int8]
    if not non_int8:
        return pl_df

    int_dtypes = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    stat_exprs: List[Any] = []
    for col in non_int8:
        if schema.get(col) not in int_dtypes:
            c_f = pl.col(col).cast(pl.Float32, strict=False)
            c_fin = c_f.filter(c_f.is_finite())
            stat_exprs.append(c_fin.min().alias(f"__min__{col}"))
            stat_exprs.append(c_fin.max().alias(f"__max__{col}"))

    stats_row = pl_df.select(stat_exprs).row(0, named=True) if stat_exprs else {}

    cast_exprs: List[Any] = []
    for col in non_int8:
        is_int_dt = schema.get(col) in int_dtypes
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


def _scan_parquet_polars(path: Path, *, parallel_prefiltered: bool = False) -> Any:
    """Open a Polars lazy Parquet scan, using `parallel='prefiltered'` when supported with clean fallback."""
    if parallel_prefiltered:
        try:
            return pl.scan_parquet(path, parallel="prefiltered")
        except (TypeError, ValueError):
            return pl.scan_parquet(path)
    return pl.scan_parquet(path)


def _pl_is_in_values(col_expr: Any, values: Any, dtype: Any) -> Any:
    """Build a Polars `is_in` filter expression with exact dtype matching and zero deprecation warnings."""
    val_series = values if isinstance(values, pl.Series) else pl.Series(values, dtype=dtype)
    if val_series.dtype != dtype:
        val_series = val_series.cast(dtype)
    rhs = val_series.implode() if hasattr(val_series, "implode") else val_series
    return col_expr.is_in(rhs)


def _pl_non_null_target_mask(target_col: str) -> Any:
    """Polars predicate matching Pandas `dropna(subset=[target_col])` for both SQL nulls and float NaNs."""
    t_col = pl.col(target_col)
    return t_col.is_not_null() & t_col.cast(pl.Float32, strict=False).is_not_nan()


def _scan_split_metadata_polars(
    path: Path,
    available_cols: Sequence[str],
    target_col: str,
    *,
    filter_validation_type: bool,
) -> Tuple[Any, Optional[Any]]:
    """Pass 1: Scan only lightweight metadata/index columns (`_row_idx`, `era`/`date`, `data_type`, `target_col`).

    Returns `(filtered_meta_pl, raw_era_dtype)` where `raw_era_dtype` is the native Parquet dtype of `'era'`
    when present (used to enable native row-group statistics pushdown in Pass 2).
    """
    avail_set = set(available_cols)
    meta_cols = [c for c in ("era", "date", "data_type") if c in avail_set]
    scan_cols = list(dict.fromkeys(meta_cols + [target_col]))

    lf = _scan_parquet_polars(path, parallel_prefiltered=False)
    schema = lf.collect_schema() if hasattr(lf, "collect_schema") else lf.schema
    raw_era_dtype = schema.get("era") if "era" in avail_set else None

    meta_pl = (
        lf.with_row_index("_row_idx")
        .select(["_row_idx"] + scan_cols)
        .filter(_pl_non_null_target_mask(target_col))
        .collect()
    )
    if filter_validation_type and "data_type" in meta_pl.columns:
        val_mask = meta_pl.get_column("data_type").cast(pl.Utf8).str.to_lowercase() == "validation"
        if bool(val_mask.any()):
            meta_pl = meta_pl.filter(val_mask)

    keep_meta_cols = [c for c in ("_row_idx", "era", "date") if c in meta_pl.columns]
    return meta_pl.select(keep_meta_cols), raw_era_dtype


def _is_polars_integer_dtype(dtype: Any) -> bool:
    """True when a Polars dtype is a native signed or unsigned integer type."""
    if dtype is None or pl is None:
        return False
    return dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )


def _select_split_keep_metadata(
    meta_pl: Any,
    *,
    era_stride: int,
    downsample_mode: Optional[str],
    downsample_frac: float,
    random_state: int,
) -> Tuple[Any, Optional[List[int]], bool]:
    """Apply chronological sort, optional `era_stride`, and deterministic downsampling on Pass 1 metadata.

    Uses the exact same `np.random.default_rng(random_state)` helpers as `downsample_panel_data`
    so Pass 2 loads the bit-for-bit identical row subset as the Pandas fallback path.
    Returns `(kept_meta_pl, kept_eras, is_era_subset)`.
    """
    mode = _normalize_downsample_mode(downsample_mode)
    if mode is not None and mode not in VALID_DOWNSAMPLE_MODES:
        raise ValueError(
            f"downsample_mode must be None or one of {VALID_DOWNSAMPLE_MODES}, got {downsample_mode!r}"
        )
    frac = float(downsample_frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"downsample_frac must be in (0.0, 1.0], got {downsample_frac!r}")

    sorted_meta = meta_pl.select(["_row_idx", "era"]).sort("era", maintain_order=True)
    if len(sorted_meta) == 0:
        return sorted_meta, [], True

    all_unique_eras = sorted(sorted_meta.get_column("era").unique().to_list())
    n_total_eras = len(all_unique_eras)

    if int(era_stride) > 1:
        strided_eras = all_unique_eras[:: int(era_stride)]
        sorted_meta = sorted_meta.filter(_pl_is_in_values(pl.col("era"), strided_eras, pl.Int32))

    if mode is not None and frac < 1.0 and len(sorted_meta) > 0:
        rng = np.random.default_rng(int(random_state))
        era_vals = sorted_meta.get_column("era").to_numpy()
        if mode == "within_era":
            keep_mask = _build_within_era_keep_mask(era_vals, frac, rng)
            sorted_meta = sorted_meta.filter(pl.Series(keep_mask))
        else:
            unique_eras = np.sort(pd.unique(era_vals))
            n_keep_eras = max(1, int(round(len(unique_eras) * frac)))
            if n_keep_eras < len(unique_eras):
                chosen_eras = _select_downsampled_eras(unique_eras, mode, n_keep_eras, rng)
                sorted_meta = sorted_meta.filter(
                    _pl_is_in_values(pl.col("era"), [int(e) for e in chosen_eras], pl.Int32)
                )

    kept_eras = sorted(int(e) for e in sorted_meta.get_column("era").unique().to_list())
    is_era_subset = len(kept_eras) < n_total_eras
    return sorted_meta, kept_eras, is_era_subset


def _load_filtered_split_polars(
    path: Path,
    req_cols: List[str],
    feature_columns: List[str],
    target_col: str,
    kept_meta_pl: Any,
    *,
    kept_eras: Optional[List[int]],
    era_is_native_int: bool,
    pushdown_era_filter: bool,
    raw_era_dtype: Any,
) -> pd.DataFrame:
    """Pass 2: Push down `_row_idx` (and native `era` when applicable) before selecting `req_cols` and casting to Int8."""
    has_orig_era = "era" in req_cols
    final_cols = list(req_cols) if has_orig_era else list(req_cols) + ["era"]

    if len(kept_meta_pl) == 0:
        empty_pl = (
            _scan_parquet_polars(path, parallel_prefiltered=False)
            .select(req_cols)
            .head(0)
            .collect()
            .with_columns(pl.col(target_col).cast(pl.Float32))
        )
        if not has_orig_era:
            empty_pl = empty_pl.with_columns(pl.lit(None, dtype=pl.Int32).alias("era"))
        else:
            empty_pl = empty_pl.with_columns(pl.col("era").cast(pl.Int32))
        empty_pl = _polars_cast_features_to_int8(empty_pl, feature_columns)
        return empty_pl.select(final_cols).to_pandas()

    row_idx_series = kept_meta_pl.get_column("_row_idx").sort()
    filter_expr = _pl_is_in_values(pl.col("_row_idx"), row_idx_series, row_idx_series.dtype)
    if (
        pushdown_era_filter
        and era_is_native_int
        and kept_eras is not None
        and len(kept_eras) > 0
        and _is_polars_integer_dtype(raw_era_dtype)
    ):
        era_filter = _pl_is_in_values(pl.col("era"), kept_eras, raw_era_dtype)
        filter_expr = era_filter & filter_expr

    split_pl = (
        _scan_parquet_polars(path, parallel_prefiltered=True)
        .with_row_index("_row_idx")
        .filter(filter_expr)
        .select(["_row_idx"] + req_cols)
        .collect()
    )

    if era_is_native_int and has_orig_era:
        split_pl = split_pl.with_columns(pl.col("era").cast(pl.Int32))
    else:
        era_lookup = kept_meta_pl.select(["_row_idx", pl.col("era").cast(pl.Int32).alias("_unified_era_")])
        split_pl = split_pl.join(era_lookup, on="_row_idx", how="inner")
        if has_orig_era:
            split_pl = split_pl.with_columns(pl.col("_unified_era_").alias("era")).drop("_unified_era_")
        else:
            split_pl = split_pl.rename({"_unified_era_": "era"})

    split_pl = split_pl.with_columns(pl.col(target_col).cast(pl.Float32))
    split_pl = _polars_cast_features_to_int8(split_pl, feature_columns)
    split_pl = split_pl.sort(["era", "_row_idx"]).select(final_cols)
    return split_pl.to_pandas()


def _load_and_clean_splits_polars(
    train_path: Path,
    validation_path: Path,
    train_cols: List[str],
    val_cols: List[str],
    feature_columns: List[str],
    target_col: str,
    *,
    apply_era_stride_early: Optional[int] = None,
    two_pass: bool = True,
    downsample_mode: Optional[str] = None,
    downsample_frac: float = 1.0,
    val_downsample_mode: Optional[str] = None,
    val_downsample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fast multi-threaded Polars Parquet reader with feature-set-agnostic two-pass predicate pushdown, Int8 casting, and era unification."""
    if not two_pass:
        train_pl = (
            _scan_parquet_polars(train_path, parallel_prefiltered=False)
            .select(train_cols)
            .filter(_pl_non_null_target_mask(target_col))
            .collect()
        )
        val_pl = (
            _scan_parquet_polars(validation_path, parallel_prefiltered=False)
            .select(val_cols)
            .filter(_pl_non_null_target_mask(target_col))
            .collect()
        )

        if "data_type" in val_pl.columns:
            val_mask = val_pl.get_column("data_type").cast(pl.Utf8).str.to_lowercase() == "validation"
            if bool(val_mask.any()):
                val_pl = val_pl.filter(val_mask)

        train_pl = train_pl.with_columns(pl.col(target_col).cast(pl.Float32))
        val_pl = val_pl.with_columns(pl.col(target_col).cast(pl.Float32))
        train_pl, val_pl = _unify_polars_split_eras(train_pl, val_pl)

        train_pl = train_pl.sort("era", maintain_order=True)
        val_pl = val_pl.sort("era", maintain_order=True)

        if apply_era_stride_early is not None and int(apply_era_stride_early) > 1:
            kept_eras = sorted(train_pl.get_column("era").unique().to_list())[:: int(apply_era_stride_early)]
            train_pl = train_pl.filter(_pl_is_in_values(pl.col("era"), kept_eras, pl.Int32))

        train_pl = _polars_cast_features_to_int8(train_pl, feature_columns)
        val_pl = _polars_cast_features_to_int8(val_pl, feature_columns)
        return train_pl.to_pandas(), val_pl.to_pandas()

    # Pass 1: Metadata-only scan (independent of feature_set width)
    resolved_val_mode, resolved_val_frac = _resolve_val_downsample_config(
        downsample_mode=downsample_mode,
        downsample_frac=downsample_frac,
        val_downsample_mode=val_downsample_mode,
        val_downsample_frac=val_downsample_frac,
    )
    train_meta_pl, train_raw_era_dtype = _scan_split_metadata_polars(
        train_path, train_cols, target_col, filter_validation_type=False
    )
    val_meta_pl, val_raw_era_dtype = _scan_split_metadata_polars(
        validation_path, val_cols, target_col, filter_validation_type=True
    )
    train_meta_pl, val_meta_pl = _unify_polars_split_eras(train_meta_pl, val_meta_pl)

    train_stride = int(apply_era_stride_early) if apply_era_stride_early is not None else 1
    train_keep_meta, train_kept_eras, train_is_era_subset = _select_split_keep_metadata(
        train_meta_pl,
        era_stride=train_stride,
        downsample_mode=downsample_mode,
        downsample_frac=downsample_frac,
        random_state=random_state,
    )
    val_keep_meta, val_kept_eras, val_is_era_subset = _select_split_keep_metadata(
        val_meta_pl,
        era_stride=1,
        downsample_mode=resolved_val_mode,
        downsample_frac=resolved_val_frac,
        random_state=random_state,
    )

    # Pass 2: Prefiltered scan pulling only req_cols for the active feature_set and selected rows
    era_is_native_int = _is_polars_integer_dtype(train_raw_era_dtype) and _is_polars_integer_dtype(val_raw_era_dtype)
    train_df = _load_filtered_split_polars(
        train_path,
        train_cols,
        feature_columns,
        target_col,
        train_keep_meta,
        kept_eras=train_kept_eras,
        era_is_native_int=era_is_native_int,
        pushdown_era_filter=train_is_era_subset and era_is_native_int,
        raw_era_dtype=train_raw_era_dtype,
    )
    val_df = _load_filtered_split_polars(
        validation_path,
        val_cols,
        feature_columns,
        target_col,
        val_keep_meta,
        kept_eras=val_kept_eras,
        era_is_native_int=era_is_native_int,
        pushdown_era_filter=val_is_era_subset and era_is_native_int,
        raw_era_dtype=val_raw_era_dtype,
    )
    return train_df, val_df


def _unify_pandas_split_eras(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure both Pandas train and validation frames share a monotonic `int32` `era` column."""
    if "era" in train_df.columns and "era" in validation_df.columns:
        train_era_numeric = pd.to_numeric(train_df["era"], errors="coerce")
        validation_era_numeric = pd.to_numeric(validation_df["era"], errors="coerce")
        if train_era_numeric.notna().all() and validation_era_numeric.notna().all():
            train_df["era"] = train_era_numeric.astype(np.int32)
            validation_df["era"] = validation_era_numeric.astype(np.int32)
            return train_df, validation_df
        train_era_keys = train_df["era"].astype(str)
        validation_era_keys = validation_df["era"].astype(str)
    elif "date" in train_df.columns and "date" in validation_df.columns:
        train_era_keys = pd.to_datetime(train_df["date"]).dt.strftime("%Y-%m-%d")
        validation_era_keys = pd.to_datetime(validation_df["date"]).dt.strftime("%Y-%m-%d")
    else:
        raise KeyError("Expected 'era' or 'date' column in both train and validation Parquet files.")

    all_era_keys = sorted(set(train_era_keys.unique()) | set(validation_era_keys.unique()))
    era_key_to_id = {era_key: np.int32(position + 1) for position, era_key in enumerate(all_era_keys)}
    train_df["era"] = train_era_keys.map(era_key_to_id).astype(np.int32)
    validation_df["era"] = validation_era_keys.map(era_key_to_id).astype(np.int32)
    return train_df, validation_df


def _cast_pandas_features_to_int8(split_df: pd.DataFrame, feature_columns: Sequence[str]) -> None:
    """Cast non-int8 feature columns in-place to `np.int8`, rescaling float `[0, 1]` columns to `0..4`."""
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


def _load_and_clean_splits_pandas(
    train_path: Path,
    validation_path: Path,
    train_req_cols: List[str],
    val_req_cols: List[str],
    feature_columns: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pandas fallback Parquet reader matching `_load_and_clean_splits_polars` target cleaning, era unification, and Int8 casting."""
    train_df = pd.read_parquet(train_path, columns=train_req_cols)
    validation_df = pd.read_parquet(validation_path, columns=val_req_cols)

    if "data_type" in validation_df.columns:
        is_validation_row = validation_df["data_type"].astype(str).str.lower() == "validation"
        if is_validation_row.any():
            validation_df = validation_df.loc[is_validation_row].copy()

    train_df = train_df.dropna(subset=[target_col]).copy()
    validation_df = validation_df.dropna(subset=[target_col]).copy()
    train_df[target_col] = train_df[target_col].astype(np.float32)
    validation_df[target_col] = validation_df[target_col].astype(np.float32)

    train_df, validation_df = _unify_pandas_split_eras(train_df, validation_df)
    train_df = train_df.sort_values("era", kind="stable")
    validation_df = validation_df.sort_values("era", kind="stable")

    _cast_pandas_features_to_int8(train_df, feature_columns)
    _cast_pandas_features_to_int8(validation_df, feature_columns)
    return train_df, validation_df


def _attach_pre_stride_ticker_timeseries_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: List[str],
    *,
    ticker_timeseries_features: Optional[str],
    ticker_rolling_windows: Sequence[int],
    ticker_lag_steps: Optional[Sequence[int]],
    ticker_top_n_features: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Compute causal per-ticker time-series features on train and warm-up-prefixed validation splits before era striding."""
    base_cols_snapshot = list(feature_columns)
    win_candidates = _normalize_ticker_windows(ticker_rolling_windows)
    lag_candidates = _normalize_ticker_lag_steps(ticker_lag_steps)
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
    return train_df, validation_df, feature_columns + common_added


def load_unity_numerai_data(
    data_version: str = "v6.0",
    data_folder: str = "data_folder",
    kaggle_dataset: str = KAGGLE_DATASET_SLUG,
    feature_set: str = "small",
    target_col: str = "target",
    era_stride: int = 4,
    downsample_mode: Optional[str] = None,
    downsample_frac: float = 1.0,
    val_downsample_mode: Optional[str] = None,
    val_downsample_frac: Optional[float] = None,
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
    workspace_dir = _resolve_workspace_root(workspace_root)
    data_root = _acquire_dataset_root(
        data_version,
        data_folder,
        kaggle_dataset,
        workspace_dir,
        auto_download=auto_download,
        use_numerapi_fallback=use_numerapi_fallback,
    )
    train_path, validation_path, features_path = _resolve_numerai_paths(data_root)
    feature_columns, train_req_cols, val_req_cols, feature_metadata = _resolve_feature_and_request_columns(
        train_path, validation_path, features_path, feature_set, target_col
    )

    want_ticker_ts = ticker_timeseries_features not in (None, False, "None", "none", "null", "")
    resolved_val_mode, resolved_val_frac = _resolve_val_downsample_config(
        downsample_mode=downsample_mode,
        downsample_frac=downsample_frac,
        val_downsample_mode=val_downsample_mode,
        val_downsample_frac=val_downsample_frac,
    )
    filtered_early = False
    if POLARS_AVAILABLE and pl is not None:
        if not want_ticker_ts:
            train_df, validation_df = _load_and_clean_splits_polars(
                train_path,
                validation_path,
                train_req_cols,
                val_req_cols,
                feature_columns,
                target_col,
                apply_era_stride_early=int(era_stride),
                two_pass=True,
                downsample_mode=downsample_mode,
                downsample_frac=downsample_frac,
                val_downsample_mode=resolved_val_mode,
                val_downsample_frac=resolved_val_frac,
                random_state=random_state,
            )
            filtered_early = True
        else:
            train_df, validation_df = _load_and_clean_splits_polars(
                train_path,
                validation_path,
                train_req_cols,
                val_req_cols,
                feature_columns,
                target_col,
                apply_era_stride_early=None,
                two_pass=False,
            )
    else:
        train_df, validation_df = _load_and_clean_splits_pandas(
            train_path, validation_path, train_req_cols, val_req_cols, feature_columns, target_col
        )

    if want_ticker_ts:
        train_df, validation_df, feature_columns = _attach_pre_stride_ticker_timeseries_features(
            train_df,
            validation_df,
            feature_columns,
            ticker_timeseries_features=ticker_timeseries_features,
            ticker_rolling_windows=ticker_rolling_windows,
            ticker_lag_steps=ticker_lag_steps,
            ticker_top_n_features=ticker_top_n_features,
        )

    if not filtered_early:
        if int(era_stride) > 1:
            kept_train_eras = sorted(train_df["era"].unique())[:: int(era_stride)]
            train_df = train_df.loc[train_df["era"].isin(kept_train_eras)]

        train_df = downsample_panel_data(
            train_df,
            downsample_mode=downsample_mode,
            downsample_frac=downsample_frac,
            era_col="era",
            random_state=random_state,
        )
        validation_df = downsample_panel_data(
            validation_df,
            downsample_mode=resolved_val_mode,
            downsample_frac=resolved_val_frac,
            era_col="era",
            random_state=random_state,
        )

    assert not any(
        column in FORBIDDEN_FEATURE_COLS or column == target_col or column.startswith("target")
        for column in feature_columns
    )
    return train_df, validation_df, feature_columns, feature_metadata


# ---------------------------------------------------------------------------
# Processed Parquet Cache Helpers
# ---------------------------------------------------------------------------

def _source_files_fingerprint(
    data_version: str,
    data_folder: str,
    kaggle_dataset: str,
    workspace_dir: Path,
) -> Optional[List[List[Any]]]:
    """Return `[filename, mtime_ns, size]` triples for the resolved train, validation, and features files."""
    data_root = _locate_local_dataset_root(data_version, data_folder, kaggle_dataset, workspace_dir)
    if data_root is None:
        return None
    train_p, val_p, feat_p = _resolve_numerai_paths(data_root)
    fp: List[List[Any]] = []
    for p in (train_p, val_p, feat_p):
        st = p.stat()
        fp.append([p.name, int(st.st_mtime_ns), int(st.st_size)])
    return fp


def _build_processed_cache_key_payload(
    *,
    data_version: str,
    feature_set: str,
    era_stride: int,
    target_col: str,
    downsample_mode: Optional[str] = None,
    downsample_frac: float = 1.0,
    val_downsample_mode: Optional[str] = None,
    val_downsample_frac: Optional[float] = None,
    random_state: int = 42,
    ticker_timeseries_features: Optional[str],
    ticker_rolling_windows: Sequence[int],
    ticker_lag_steps: Optional[Sequence[int]],
    ticker_top_n_features: int,
    kaggle_dataset: str,
    source_fp: Optional[List[List[Any]]],
) -> Dict[str, Any]:
    """Assemble the canonical JSON-serializable payload defining a processed dataset cache key."""
    norm_train_mode = _normalize_downsample_mode(downsample_mode)
    norm_train_frac = 1.0 if norm_train_mode is None else round(float(downsample_frac), 12)
    resolved_val_mode, resolved_val_frac = _resolve_val_downsample_config(
        downsample_mode=downsample_mode,
        downsample_frac=downsample_frac,
        val_downsample_mode=val_downsample_mode,
        val_downsample_frac=val_downsample_frac,
    )
    norm_val_frac = 1.0 if resolved_val_mode is None else round(float(resolved_val_frac), 12)
    return {
        "data_version": str(data_version),
        "feature_set": str(feature_set),
        "era_stride": int(era_stride),
        "target_col": str(target_col),
        "downsample_mode": norm_train_mode,
        "downsample_frac": norm_train_frac,
        "val_downsample_mode": resolved_val_mode,
        "val_downsample_frac": norm_val_frac,
        "random_state": int(random_state),
        "ticker_timeseries_features": ticker_timeseries_features,
        "ticker_rolling_windows": list(ticker_rolling_windows) if ticker_rolling_windows is not None else None,
        "ticker_lag_steps": list(ticker_lag_steps) if ticker_lag_steps is not None else None,
        "ticker_top_n_features": int(ticker_top_n_features),
        "kaggle_dataset": str(kaggle_dataset),
        "source_fp": source_fp,
    }


def _build_processed_cache_paths(cache_dir: Path, key_payload: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    """Hash `key_payload` into `(train_cache_path, val_cache_path, meta_cache_path)` under `cache_dir`."""
    cache_hash = hashlib.blake2b(json.dumps(key_payload, sort_keys=True).encode("utf-8"), digest_size=12).hexdigest()
    return (
        cache_dir / f"train_{cache_hash}.parquet",
        cache_dir / f"validation_{cache_hash}.parquet",
        cache_dir / f"meta_{cache_hash}.json",
    )


def _try_load_cached_unity_splits(
    train_cache: Path,
    val_cache: Path,
    meta_cache: Path,
    key_payload: Dict[str, Any],
    target_col: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]]:
    """Read and validate cached train/validation Parquets and metadata, returning `None` on miss or schema mismatch."""
    if not (train_cache.is_file() and val_cache.is_file() and meta_cache.is_file()):
        return None
    try:
        with meta_cache.open("r", encoding="utf-8") as mf:
            meta = json.load(mf)
        if meta.get("key_payload") != key_payload:
            return None
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
        return (train_df, validation_df, feature_columns, feature_metadata) if valid_hit else None
    except Exception:
        return None


def _write_cached_unity_splits(
    cache_dir: Path,
    key_payload: Dict[str, Any],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: List[str],
    feature_metadata: Dict[str, Any],
) -> None:
    """Persist processed train/validation Parquets and JSON schema metadata to `cache_dir`."""
    try:
        train_cache, val_cache, meta_cache = _build_processed_cache_paths(cache_dir, key_payload)
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


def load_unity_numerai_data_cached(
    data_version: str = "v6.0",
    data_folder: str = "data_folder",
    kaggle_dataset: str = KAGGLE_DATASET_SLUG,
    feature_set: str = "small",
    target_col: str = "target",
    era_stride: int = 4,
    downsample_mode: Optional[str] = None,
    downsample_frac: float = 1.0,
    val_downsample_mode: Optional[str] = None,
    val_downsample_frac: Optional[float] = None,
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

    Keys the disk cache by `(feature_set, era_stride, target_col, downsample_mode, downsample_frac,
    val_downsample_mode, val_downsample_frac, random_state, ...)` and passes all downsampling knobs
    directly into `load_unity_numerai_data(...)` so two-pass Polars skips unused rows on cache miss
    and only the already-downsampled lightweight Parquets are written to disk.
    """
    resolved_val_mode, resolved_val_frac = _resolve_val_downsample_config(
        downsample_mode=downsample_mode,
        downsample_frac=downsample_frac,
        val_downsample_mode=val_downsample_mode,
        val_downsample_frac=val_downsample_frac,
    )
    workspace_dir = _resolve_workspace_root(workspace_root)
    cache_dir = workspace_dir / data_folder / "processed_cache"
    source_fp = _source_files_fingerprint(data_version, data_folder, kaggle_dataset, workspace_dir)
    key_payload = _build_processed_cache_key_payload(
        data_version=data_version,
        feature_set=feature_set,
        era_stride=era_stride,
        target_col=target_col,
        downsample_mode=downsample_mode,
        downsample_frac=downsample_frac,
        val_downsample_mode=resolved_val_mode,
        val_downsample_frac=resolved_val_frac,
        random_state=random_state,
        ticker_timeseries_features=ticker_timeseries_features,
        ticker_rolling_windows=ticker_rolling_windows,
        ticker_lag_steps=ticker_lag_steps,
        ticker_top_n_features=ticker_top_n_features,
        kaggle_dataset=kaggle_dataset,
        source_fp=source_fp,
    )

    cached_result: Optional[Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]] = None
    if not force_reload and source_fp is not None:
        train_cache, val_cache, meta_cache = _build_processed_cache_paths(cache_dir, key_payload)
        cached_result = _try_load_cached_unity_splits(train_cache, val_cache, meta_cache, key_payload, target_col)

    if cached_result is not None:
        return cached_result

    train_df, validation_df, feature_columns, feature_metadata = load_unity_numerai_data(
        data_version=data_version,
        data_folder=data_folder,
        kaggle_dataset=kaggle_dataset,
        feature_set=feature_set,
        target_col=target_col,
        era_stride=era_stride,
        downsample_mode=downsample_mode,
        downsample_frac=downsample_frac,
        val_downsample_mode=resolved_val_mode,
        val_downsample_frac=resolved_val_frac,
        random_state=random_state,
        ticker_timeseries_features=ticker_timeseries_features,
        ticker_rolling_windows=ticker_rolling_windows,
        ticker_lag_steps=ticker_lag_steps,
        ticker_top_n_features=ticker_top_n_features,
        workspace_root=workspace_root,
        auto_download=auto_download,
        use_numerapi_fallback=use_numerapi_fallback,
    )
    key_payload["source_fp"] = _source_files_fingerprint(data_version, data_folder, kaggle_dataset, workspace_dir)
    _write_cached_unity_splits(cache_dir, key_payload, train_df, validation_df, feature_columns, feature_metadata)
    return train_df, validation_df, feature_columns, feature_metadata


# ---------------------------------------------------------------------------
# Numerai v6.0 Meta-Model & Benchmark Proxy Discovery Helpers
# ---------------------------------------------------------------------------

def _list_numerapi_version_datasets(data_version: str) -> Tuple[Any, List[str], List[str], Optional[str]]:
    """Query `NumerAPI.list_datasets()` for files matching `data_version/` and contribution keywords."""
    if NumerAPI is None:
        return None, [], [], "numerapi package is not installed"
    prefix = f"{data_version}/"
    try:
        napi = NumerAPI()
        all_datasets = sorted([p for p in napi.list_datasets() if str(p).startswith(prefix)])
        filtered_datasets = [
            p for p in all_datasets if any(kw in p.lower() for kw in V6_CONTRIBUTION_KEYWORDS)
        ]
        return napi, all_datasets, filtered_datasets, None
    except Exception as exc:
        return None, [], [], f"{type(exc).__name__}: {exc}"


def _log_numerapi_discovery_results(
    data_version: str,
    all_datasets: Sequence[str],
    filtered_datasets: Sequence[str],
    api_error: Optional[str],
) -> None:
    """Print the discovered NumerAPI dataset listing and contribution-keyword matches."""
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


def _find_local_contribution_file(
    filename: str,
    save_dir: Path,
    workspace_dir: Path,
    data_folder: str,
) -> Optional[Path]:
    """Locate a local meta-model or benchmark Parquet in the version folder, data folder, or `/kaggle/input`."""
    candidates = [
        save_dir / filename,
        workspace_dir / data_folder / filename,
    ]
    if Path("/kaggle/input").is_dir():
        candidates.extend(sorted(Path("/kaggle/input").rglob(filename)))
    return next((p for p in candidates if p.is_file()), None)


def _download_numerapi_contribution_file(
    napi: Any,
    remote_name: str,
    dest_path: Path,
    *,
    verbose: bool = False,
) -> Optional[Path]:
    """Download a single meta-model or benchmark Parquet via `NumerAPI`, warning cleanly on failure."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if verbose:
            print(f"Downloading BMC proxy file: {remote_name} -> {dest_path}")
        napi.download_dataset(remote_name, str(dest_path))
        return dest_path if dest_path.is_file() else None
    except Exception as exc:
        warnings.warn(f"Failed to download {remote_name}: {exc}")
        return None


def _resolve_benchmark_fallback_file(
    *,
    data_version: str,
    save_dir: Path,
    workspace_dir: Path,
    data_folder: str,
    all_datasets: Sequence[str],
    napi: Any,
    auto_download: bool,
    verbose: bool,
) -> Tuple[Optional[Path], Optional[str], List[Tuple[str, str]]]:
    """Find or download the highest-priority benchmark/example predictions file for honest proxy scoring."""
    prefix = f"{data_version}/"
    downloaded_files: List[Tuple[str, str]] = []
    for fname in V6_BENCHMARK_FALLBACK_ORDER:
        local_hit = _find_local_contribution_file(fname, save_dir, workspace_dir, data_folder)
        if local_hit is not None:
            return local_hit, f"{prefix}{fname}", downloaded_files
        remote_name = f"{prefix}{fname}"
        if remote_name in all_datasets and auto_download and napi is not None:
            downloaded = _download_numerapi_contribution_file(
                napi, remote_name, save_dir / fname, verbose=verbose
            )
            if downloaded is not None:
                downloaded_files.append((remote_name, str(downloaded)))
                return downloaded, remote_name, downloaded_files
    return None, None, downloaded_files


def _classify_contribution_proxy_file(contribution_path: Path) -> Tuple[str, str, str]:
    """Return `(basis_kind, basis_label, log_tag)` distinguishing benchmark BMC proxy from example-model proxy."""
    if "benchmark" in contribution_path.name.lower():
        return "bmc_proxy", "BMC proxy (benchmark, NOT true MMC)", "BMC Proxy Fallback"
    return "example_proxy", "Example-model proxy (NOT true MMC / NOT true BMC)", "Example-Model Proxy Fallback"


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
    workspace_dir = _resolve_workspace_root(workspace_root)
    save_dir = workspace_dir / data_folder / data_version
    prefix = f"{data_version}/"

    napi, all_datasets, filtered_datasets, api_error = _list_numerapi_version_datasets(data_version)
    if verbose:
        _log_numerapi_discovery_results(data_version, all_datasets, filtered_datasets, api_error)

    downloaded_files: List[Tuple[str, str]] = []
    meta_remote = f"{prefix}meta_model.parquet"
    meta_local_path = _find_local_contribution_file("meta_model.parquet", save_dir, workspace_dir, data_folder)
    if meta_local_path is None and meta_remote in all_datasets and auto_download and napi is not None:
        meta_local_path = _download_numerapi_contribution_file(
            napi, meta_remote, save_dir / "meta_model.parquet", verbose=False
        )
        if meta_local_path is not None:
            downloaded_files.append((meta_remote, str(meta_local_path)))

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

    no_meta_msg = (
        f"No {data_version} meta_model.parquet published yet, so true MMC / NMMC cannot be computed in backtest."
    )
    if verbose:
        print(f"\n[MMC Status] {no_meta_msg}")

    benchmark_local_path, chosen_remote, bench_downloads = _resolve_benchmark_fallback_file(
        data_version=data_version,
        save_dir=save_dir,
        workspace_dir=workspace_dir,
        data_folder=data_folder,
        all_datasets=all_datasets,
        napi=napi,
        auto_download=auto_download,
        verbose=verbose,
    )
    downloaded_files.extend(bench_downloads)

    if benchmark_local_path is not None:
        basis_kind, basis_label, tag = _classify_contribution_proxy_file(benchmark_local_path)
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
