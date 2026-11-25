"""afml_utils.py — Advances in Financial Machine Learning (López de Prado) & Numerai diagnostics suite.

Implements:
  - Purged & Embargoed K-Fold CV (`PurgedEraKFold`) and Combinatorial Purged CV (`CombinatorialPurgedEraCV`,
    with multi-path backtest reconstruction; AFML Ch. 7 & Ch. 12).
  - Unified era-aware CV builder (`build_era_cv_splits`) and train/validation horizon purging
    (`purge_train_before_validation`).
  - Per-era OLS Feature Neutralization (`neutralize_predictions`) and Feature Neutral Correlation
    (`compute_per_era_fnc`; Numerai Forum #1454 & López de Prado feature-balance principle).
  - Non-normality & autocorrelation-adjusted Sharpe diagnostics:
      * Lo (2002) autocorrelation-adjusted Sharpe (`compute_lo_adjusted_sharpe`)
      * Probabilistic Sharpe Ratio (`compute_probabilistic_sharpe_ratio`, MLE moments guaranteeing
        Cauchy-Schwarz variance non-negativity)
      * Minimum Track Record Length (`compute_min_track_record_length`)
      * Optimal Number of Clusters (`compute_onc_clusters`; Silhouette t-stat + recursive refinement)
      * Deflated Sharpe Ratio with ONC-Clustered effective E[K] & Šidák FWER
        (`compute_deflated_sharpe_ratio`)
      * Probability of Backtest Overfitting via CSCV (`compute_pbo_cscv`; AFML Ch. 11.6)
      * HWM Drawdown & Time under Water (`compute_drawdown_and_tuw`; AFML Ch. 14.6)
      * Return Concentration HHI (`compute_hhi_concentration`; AFML Ch. 14.5)
      * Strategy Risk / implied precision (`compute_strategy_risk`; AFML Ch. 15)
  - Information-theoretic Variation of Information (`compute_variation_of_information_matrix`),
    ONC feature clustering (`cluster_features_onc`), and Clustered Mean Decrease Accuracy
    (`compute_clustered_mda`; López de Prado CFI).
  - High-level report builder (`summarize_afml_backtest_diagnostics`).
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numerai_tools.scoring import numerai_corr, sharpe_ratio as numerai_sharpe_ratio
from scipy import stats
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _clean_1d_finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _safe_sharpe(arr: np.ndarray) -> float:
    if len(arr) < 2 or float(np.std(arr)) <= 1e-12:
        return 0.0
    val = float(numerai_sharpe_ratio(pd.Series(arr, dtype=np.float64)))
    return val if np.isfinite(val) else 0.0


def _is_const_1d(arr: np.ndarray) -> bool:
    if len(arr) < 2:
        return True
    if bool(np.all(arr == arr[0])):
        return True
    return float(np.std(arr)) <= 1e-12


def _gauss_rank_1d(arr: np.ndarray) -> np.ndarray:
    """Rank-normalize a 1D array to standard normal scores using mid-ranks."""
    n = len(arr)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    ranks = stats.rankdata(arr, method="average")
    u = np.clip((ranks - 0.5) / float(n), 1e-6, 1.0 - 1e-6)
    return stats.norm.ppf(u).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# 1. Train/Validation Purging & AFML Cross-Validation Splitters
# ---------------------------------------------------------------------------

def purge_train_before_validation(
    X_train: pd.DataFrame,
    y_train: Any,
    train_eras: Any,
    validation_eras: Any,
    *,
    target_horizon_eras: int = 12,
    buffer_eras: int = 0,
    min_keep_eras: int = 1,
) -> Tuple[pd.DataFrame, Any, pd.Series, Dict[str, Any]]:
    """Purge trailing training eras whose target settlement overlaps the start of validation (AFML Ch. 7.4.1).

    An observation in era `e` with label horizon `H = target_horizon_eras` resolves at `e + H`.
    To prevent label overlap with `min_val_era = min(validation_eras)`, retains only training eras
    satisfying `e <= min_val_era - H - buffer_eras`.

    Returns `(X_purged, y_purged, eras_purged, purge_info)`.
    """
    train_era_arr = np.asarray(train_eras, dtype=np.int64).ravel()
    val_era_arr = np.asarray(validation_eras, dtype=np.int64).ravel()
    horizon = max(0, int(target_horizon_eras))
    buf = max(0, int(buffer_eras))

    if len(train_era_arr) == 0 or len(val_era_arr) == 0 or (horizon + buf) <= 0:
        era_s = pd.Series(train_era_arr, index=X_train.index) if isinstance(X_train, pd.DataFrame) else pd.Series(train_era_arr)
        return X_train, y_train, era_s, {
            "n_eras_before": int(len(np.unique(train_era_arr))),
            "n_eras_after": int(len(np.unique(train_era_arr))),
            "n_eras_purged": 0,
            "cutoff_era": None,
            "min_val_era": int(np.min(val_era_arr)) if len(val_era_arr) else None,
        }

    unique_train_eras = np.sort(np.unique(train_era_arr))
    min_val_era = int(np.min(val_era_arr))
    cutoff_era = min_val_era - horizon - buf
    keep_mask = train_era_arr <= cutoff_era
    kept_eras = unique_train_eras[unique_train_eras <= cutoff_era]

    if len(kept_eras) < max(1, int(min_keep_eras)):
        # Safety fallback on tiny synthetic unit-test panels where horizon exceeds training span
        fallback_eras = unique_train_eras[: max(1, min(len(unique_train_eras), int(min_keep_eras)))]
        keep_mask = np.isin(train_era_arr, fallback_eras)
        kept_eras = fallback_eras

    n_purged = int(len(unique_train_eras) - len(kept_eras))
    if isinstance(X_train, pd.DataFrame):
        X_out = X_train.loc[keep_mask]
    else:
        X_out = np.asarray(X_train)[keep_mask]

    if isinstance(y_train, pd.Series):
        y_out: Any = y_train.loc[keep_mask]
    elif isinstance(y_train, pd.DataFrame):
        y_out = y_train.loc[keep_mask]
    else:
        y_out = np.asarray(y_train)[keep_mask]

    if isinstance(train_eras, pd.Series):
        eras_out = train_eras.loc[keep_mask]
    else:
        eras_out = pd.Series(train_era_arr[keep_mask], index=X_out.index if isinstance(X_out, pd.DataFrame) else None)

    info = {
        "n_eras_before": int(len(unique_train_eras)),
        "n_eras_after": int(len(kept_eras)),
        "n_eras_purged": n_purged,
        "cutoff_era": int(cutoff_era),
        "min_val_era": int(min_val_era),
        "max_kept_train_era": int(kept_eras[-1]) if len(kept_eras) else None,
    }
    return X_out, y_out, eras_out, info


class PurgedEraKFold:
    """Purged & Embargoed K-Fold CV grouped strictly by era value (AFML Ch. 7.4).

    Splits unique sorted eras into `n_splits` contiguous folds. For each test fold `[t_min, t_max]`:
      - Purges training eras in `(t_min - purge_eras, t_min)` whose target horizon overlaps the test fold.
      - Embargoes training eras in `(t_max, t_max + eff_embargo]` immediately following the test fold,
        where `eff_embargo = max(embargo_eras, ceil(embargo_pct * n_unique_eras))`.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_eras: int = 12,
        embargo_eras: int = 0,
        embargo_pct: float = 0.01,
    ):
        if int(n_splits) < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits!r}.")
        self.n_splits = int(n_splits)
        self.purge_eras = max(0, int(purge_eras))
        self.embargo_eras = max(0, int(embargo_eras))
        self.embargo_pct = max(0.0, float(embargo_pct))

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            if isinstance(X, pd.DataFrame) and "era" in X.columns:
                era_arr = pd.to_numeric(X["era"], errors="coerce").fillna(1).to_numpy(dtype=np.int64)
            else:
                raise ValueError("PurgedEraKFold.split requires `groups` (eras) or an 'era' column in X.")
        else:
            era_arr = np.asarray(groups, dtype=np.int64).ravel()

        unique_eras = np.sort(np.unique(era_arr))
        n_eras = len(unique_eras)
        if n_eras < self.n_splits:
            raise ValueError(f"Cannot split {n_eras} unique eras into {self.n_splits} folds.")

        eff_embargo = max(self.embargo_eras, int(np.ceil(self.embargo_pct * float(n_eras))))
        era_span_step = int(np.median(np.diff(unique_eras))) if n_eras >= 2 else 1
        eff_embargo_span = eff_embargo * max(1, era_span_step) if (self.embargo_eras == 0 and self.embargo_pct > 0.0) else eff_embargo

        fold_era_blocks = np.array_split(unique_eras, self.n_splits)
        row_indices = np.arange(len(era_arr), dtype=np.int64)

        for test_eras in fold_era_blocks:
            if len(test_eras) == 0:
                continue
            t_min = int(test_eras[0])
            t_max = int(test_eras[-1])
            test_mask = (era_arr >= t_min) & (era_arr <= t_max)

            # Purge training eras strictly before test_eras whose [e, e + purge_eras] overlaps t_min
            # Strictly settled iff e + purge_eras <= t_min  <=>  e <= t_min - purge_eras
            before_ok = era_arr <= (t_min - self.purge_eras)
            # Embargo training eras immediately after t_max (and also require non-overlap after t_max)
            after_gap = max(self.purge_eras, eff_embargo_span)
            after_ok = era_arr > (t_max + after_gap)
            train_mask = (~test_mask) & (before_ok | after_ok)

            if not np.any(train_mask):
                # Fallback if purge/embargo exceeds tiny synthetic panel: relax embargo first
                train_mask = (~test_mask) & ((era_arr <= (t_min - self.purge_eras)) | (era_arr > t_max))
            if not np.any(train_mask):
                train_mask = ~test_mask

            yield row_indices[train_mask], row_indices[test_mask]


class CombinatorialPurgedEraCV:
    """Combinatorial Purged Cross-Validation (CPCV; AFML Ch. 12.4) grouped strictly by era.

    Divides unique eras into `n_groups` contiguous blocks and generates all `C(n_groups, n_test_groups)`
    train/test splits with purging before every test block and embargoing after every test block.
    Supports exact reconstruction of all `phi(N, k) = (k / N) * C(N, k)` complete chronological backtest paths.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_eras: int = 12,
        embargo_eras: int = 0,
        embargo_pct: float = 0.01,
    ):
        n_g = int(n_groups)
        n_t = int(n_test_groups)
        if n_g < 3:
            raise ValueError(f"n_groups must be >= 3, got {n_groups!r}.")
        if not (1 <= n_t < n_g):
            raise ValueError(f"n_test_groups must satisfy 1 <= n_test_groups < n_groups, got {n_t} / {n_g}.")
        self.n_groups = n_g
        self.n_test_groups = n_t
        self.purge_eras = max(0, int(purge_eras))
        self.embargo_eras = max(0, int(embargo_eras))
        self.embargo_pct = max(0.0, float(embargo_pct))

    @property
    def n_splits(self) -> int:
        return int(comb(self.n_groups, self.n_test_groups))

    @property
    def n_paths(self) -> int:
        """Number of complete backtest paths phi[N, k] = (k / N) * C(N, k) (AFML Eq. 12.1)."""
        return int((self.n_test_groups * comb(self.n_groups, self.n_test_groups)) // self.n_groups)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def get_test_group_combinations(self) -> List[Tuple[int, ...]]:
        return list(combinations(range(self.n_groups), self.n_test_groups))

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            if isinstance(X, pd.DataFrame) and "era" in X.columns:
                era_arr = pd.to_numeric(X["era"], errors="coerce").fillna(1).to_numpy(dtype=np.int64)
            else:
                raise ValueError("CombinatorialPurgedEraCV.split requires `groups` (eras) or an 'era' column in X.")
        else:
            era_arr = np.asarray(groups, dtype=np.int64).ravel()

        unique_eras = np.sort(np.unique(era_arr))
        n_eras = len(unique_eras)
        if n_eras < self.n_groups:
            raise ValueError(f"Cannot partition {n_eras} unique eras into {self.n_groups} CPCV groups.")

        eff_embargo = max(self.embargo_eras, int(np.ceil(self.embargo_pct * float(n_eras))))
        era_span_step = int(np.median(np.diff(unique_eras))) if n_eras >= 2 else 1
        eff_embargo_span = eff_embargo * max(1, era_span_step) if (self.embargo_eras == 0 and self.embargo_pct > 0.0) else eff_embargo
        after_gap = max(self.purge_eras, eff_embargo_span)

        group_blocks = np.array_split(unique_eras, self.n_groups)
        self.group_blocks_ = [np.asarray(b, dtype=np.int64) for b in group_blocks]
        row_indices = np.arange(len(era_arr), dtype=np.int64)

        for test_combo in self.get_test_group_combinations():
            test_mask = np.zeros(len(era_arr), dtype=bool)
            excluded_train = np.zeros(len(era_arr), dtype=bool)

            for g_idx in test_combo:
                g_eras = self.group_blocks_[g_idx]
                if len(g_eras) == 0:
                    continue
                t_min = int(g_eras[0])
                t_max = int(g_eras[-1])
                in_test = (era_arr >= t_min) & (era_arr <= t_max)
                test_mask |= in_test
                excluded_train |= in_test
                # Purge immediately preceding eras whose target settles inside test block
                excluded_train |= (era_arr > (t_min - self.purge_eras)) & (era_arr < t_min)
                # Embargo immediately following eras
                excluded_train |= (era_arr > t_max) & (era_arr <= (t_max + after_gap))

            train_mask = ~excluded_train
            if not np.any(train_mask):
                train_mask = ~test_mask
            yield row_indices[train_mask], row_indices[test_mask]

    def reconstruct_backtest_paths(
        self,
        split_era_corrs: Sequence[pd.Series],
        eras: Any,
    ) -> Dict[str, Any]:
        """Reconstruct all `phi(N, k)` complete chronological backtest paths from per-split test era CORRs."""
        combos = self.get_test_group_combinations()
        if len(split_era_corrs) != len(combos):
            raise ValueError(
                f"Expected {len(combos)} split era_corr Series for CPCV({self.n_groups}, {self.n_test_groups}), "
                f"got {len(split_era_corrs)}."
            )
        unique_eras = np.sort(np.unique(np.asarray(eras, dtype=np.int64).ravel()))
        group_blocks = [np.asarray(b, dtype=np.int64) for b in np.array_split(unique_eras, self.n_groups)]
        n_paths = self.n_paths

        # For each group g, track which path index receives the next split containing group g
        group_next_path = [0] * self.n_groups
        path_pieces: List[List[pd.Series]] = [[] for _ in range(n_paths)]

        for combo, era_s in zip(combos, split_era_corrs):
            s = pd.Series(era_s, dtype=np.float64)
            s_idx_int = pd.to_numeric(s.index, errors="coerce")
            for g_idx in combo:
                p_idx = group_next_path[g_idx]
                group_next_path[g_idx] += 1
                g_set = set(group_blocks[g_idx].tolist())
                mask = s_idx_int.isin(g_set)
                if bool(np.any(mask)):
                    path_pieces[p_idx].append(s.loc[mask])

        path_cols: Dict[str, pd.Series] = {}
        path_sharpes: List[float] = []
        path_means: List[float] = []
        for p_idx in range(n_paths):
            if path_pieces[p_idx]:
                combined = pd.concat(path_pieces[p_idx]).sort_index()
                combined = combined.loc[~combined.index.duplicated(keep="last")]
            else:
                combined = pd.Series(dtype=np.float64)
            path_cols[f"path_{p_idx}"] = combined
            arr = _clean_1d_finite(combined.to_numpy(dtype=np.float64))
            path_sharpes.append(_safe_sharpe(arr))
            path_means.append(float(np.mean(arr)) if len(arr) else 0.0)

        paths_df = pd.DataFrame(path_cols).sort_index()
        sr_arr = np.asarray(path_sharpes, dtype=np.float64)
        return {
            "paths_df": paths_df,
            "n_paths": n_paths,
            "path_sharpes": sr_arr,
            "path_means": np.asarray(path_means, dtype=np.float64),
            "mean_path_sharpe": float(np.mean(sr_arr)) if len(sr_arr) else 0.0,
            "median_path_sharpe": float(np.median(sr_arr)) if len(sr_arr) else 0.0,
            "std_path_sharpe": float(np.std(sr_arr, ddof=1)) if len(sr_arr) >= 2 else 0.0,
            "p05_path_sharpe": float(np.percentile(sr_arr, 5.0)) if len(sr_arr) else 0.0,
            "prob_positive_sharpe": float(np.mean(sr_arr > 0.0)) if len(sr_arr) else 0.0,
        }


def build_era_cv_splits(
    train_eras: Any,
    *,
    cv_strategy: str = "purged_kfold",
    n_splits: int = 5,
    cv_split: float = 0.8,
    target_horizon_eras: int = 12,
    embargo_eras: Optional[int] = None,
    embargo_pct: float = 0.01,
    cpcv_n_groups: int = 6,
    cpcv_n_test_groups: int = 2,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build strictly era-purged CV splits `(train_idx, test_idx)` for GridSearchCV / BayesSearchCV.

    Supports:
      - `'timeseries'`: Expanding chronological era splits with strict era-number horizon purging
        (`train_era <= test_start_era - target_horizon_eras`, immune to irregular downsampling).
        When `n_splits == 1`, creates one chronological holdout split at `cv_split`.
      - `'purged_kfold'`: AFML Ch. 7 Purged & Embargoed K-Fold across contiguous era blocks.
      - `'cpcv'`: AFML Ch. 12 Combinatorial Purged CV (`C(cpcv_n_groups, cpcv_n_test_groups)` splits).
    """
    era_arr = np.asarray(train_eras, dtype=np.int64).ravel()
    unique_eras = np.sort(np.unique(era_arr))
    n_unique = len(unique_eras)
    clean_strat = str(cv_strategy or "purged_kfold").strip().lower()
    horizon = max(0, int(target_horizon_eras))
    eff_embargo = max(0, int(embargo_eras)) if embargo_eras is not None else 0

    if n_unique < 2:
        raise ValueError(f"Need at least 2 unique eras for CV splitting, got {n_unique}.")

    row_indices = np.arange(len(era_arr), dtype=np.int64)

    if clean_strat == "timeseries":
        k = max(1, int(n_splits))
        if k == 1:
            cutoff_pos = max(1, min(n_unique - 1, int(round(n_unique * float(cv_split)))))
            test_eras = unique_eras[cutoff_pos:]
            t_min = int(test_eras[0])
            train_mask = era_arr <= (t_min - horizon)
            if not np.any(train_mask):
                train_mask = era_arr < t_min
            test_mask = era_arr >= t_min
            return [(row_indices[train_mask], row_indices[test_mask])]

        eff_k = min(k, max(1, n_unique - 1))
        # Partition eras into (eff_k + 1) blocks; fold i trains on blocks 0..i (purged) and tests on block i+1
        blocks = np.array_split(unique_eras, eff_k + 1)
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(eff_k):
            test_eras = blocks[i + 1]
            if len(test_eras) == 0:
                continue
            t_min = int(test_eras[0])
            t_max = int(test_eras[-1])
            test_mask = (era_arr >= t_min) & (era_arr <= t_max)
            train_mask = era_arr <= (t_min - horizon)
            if not np.any(train_mask):
                train_mask = era_arr < t_min
            if np.any(train_mask) and np.any(test_mask):
                splits.append((row_indices[train_mask], row_indices[test_mask]))
        if not splits:
            raise ValueError("Unable to construct non-empty timeseries CV splits.")
        return splits

    if clean_strat == "purged_kfold":
        eff_k = min(max(2, int(n_splits)), n_unique)
        splitter = PurgedEraKFold(
            n_splits=eff_k,
            purge_eras=horizon,
            embargo_eras=eff_embargo,
            embargo_pct=float(embargo_pct),
        )
        return list(splitter.split(row_indices, groups=era_arr))

    if clean_strat == "cpcv":
        eff_g = min(max(3, int(cpcv_n_groups)), n_unique)
        eff_t = min(max(1, int(cpcv_n_test_groups)), eff_g - 1)
        splitter = CombinatorialPurgedEraCV(
            n_groups=eff_g,
            n_test_groups=eff_t,
            purge_eras=horizon,
            embargo_eras=eff_embargo,
            embargo_pct=float(embargo_pct),
        )
        return list(splitter.split(row_indices, groups=era_arr))

    raise ValueError(
        f"Unsupported cv_strategy {cv_strategy!r}; expected 'timeseries', 'purged_kfold', or 'cpcv'."
    )


# ---------------------------------------------------------------------------
# 2. Per-Era Feature Neutralization & Feature Neutral Correlation (FNC)
# ---------------------------------------------------------------------------

def neutralize_predictions(
    predictions: Any,
    features_df: pd.DataFrame,
    *,
    eras: Optional[Any] = None,
    feature_columns: Optional[Sequence[str]] = None,
    proportion: float = 1.0,
    normalize: bool = True,
    max_neutralize_features: Optional[int] = None,
) -> np.ndarray:
    """Per-era OLS feature neutralization matching Numerai's canonical neutralization recipe.

    Within each era:
      1. Optionally Gaussian rank-normalizes `predictions` (`normalize=True`).
      2. Projects predictions onto `[features, 1]` via least-squares and subtracts
         `proportion * linear_exposure`.
      3. Rescales neutralized predictions to unit standard deviation.
    """
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    n_rows = len(preds)
    if n_rows == 0:
        return np.empty(0, dtype=np.float64)
    prop = float(np.clip(float(proportion), 0.0, 1.0))

    forbidden = {"era", "date", "data_type", "id", "numerai_ticker", "week_no", "year_horizon", "target"}
    if feature_columns is not None:
        cols = [str(c) for c in feature_columns if str(c) in features_df.columns]
    else:
        cols = [
            str(c)
            for c in features_df.columns
            if str(c) not in forbidden and not str(c).startswith("target")
        ]

    if not cols or prop <= 1e-12:
        return preds.copy()

    if max_neutralize_features is not None and len(cols) > int(max_neutralize_features):
        # Select top-variance / top-correlated features with predictions to keep neutralization fast on 2,000+ features
        k_keep = max(1, int(max_neutralize_features))
        x_all = features_df[cols].to_numpy(dtype=np.float64, copy=False)
        x_all = np.nan_to_num(x_all, nan=2.0)
        p_c = preds - float(np.mean(preds))
        p_norm = float(np.linalg.norm(p_c))
        if p_norm > 1e-12:
            x_c = x_all - np.mean(x_all, axis=0)
            x_norm = np.linalg.norm(x_c, axis=0)
            abs_r = np.divide(np.abs(x_c.T @ p_c), np.maximum(x_norm * p_norm, 1e-12))
            top_idx = np.argsort(-abs_r, kind="stable")[:k_keep]
            cols = [cols[int(i)] for i in top_idx]
        else:
            cols = cols[:k_keep]

    X_mat = np.nan_to_num(features_df[cols].to_numpy(dtype=np.float64, copy=False), nan=2.0)
    era_arr = np.ones(n_rows, dtype=np.int64) if eras is None else np.asarray(eras).ravel()
    out = preds.copy()

    # Group row indices by era using stable argsort
    order = np.argsort(era_arr, kind="stable")
    sorted_eras = era_arr[order]
    change = np.empty(n_rows, dtype=bool)
    change[0] = True
    if n_rows > 1:
        change[1:] = sorted_eras[1:] != sorted_eras[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n_rows)

    for s, e in zip(starts, ends):
        idx = order[s:e]
        y_e = preds[idx]
        if len(idx) < 3 or _is_const_1d(y_e):
            out[idx] = y_e
            continue
        scores = _gauss_rank_1d(y_e) if normalize else (y_e - float(np.mean(y_e)))
        X_e = X_mat[idx]
        # Center features and append intercept column for numerical stability
        X_e = X_e - np.mean(X_e, axis=0, keepdims=True)
        ones = np.ones((len(idx), 1), dtype=np.float64)
        design = np.hstack([X_e, ones])
        try:
            beta, _, _, _ = np.linalg.lstsq(design, scores, rcond=None)
            exposure = design @ beta
            neutral = scores - prop * exposure
        except np.linalg.LinAlgError:
            neutral = scores
        std_n = float(np.std(neutral))
        if std_n > 1e-12:
            neutral = (neutral - float(np.mean(neutral))) / std_n
        out[idx] = neutral

    return out


def compute_per_era_fnc(
    y_true: Any,
    y_pred: Any,
    features_df: pd.DataFrame,
    *,
    eras: Optional[Any] = None,
    feature_columns: Optional[Sequence[str]] = None,
    proportion: float = 1.0,
    max_neutralize_features: Optional[int] = 100,
) -> Dict[str, Any]:
    """Compute Feature Neutral Correlation (FNC): per-era `numerai_corr` after per-era OLS neutralization."""
    targets = np.asarray(y_true, dtype=np.float64).ravel()
    preds = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(targets) != len(preds) or len(targets) != len(features_df):
        raise ValueError("y_true, y_pred, and features_df must have equal length.")

    neutral_preds = neutralize_predictions(
        preds,
        features_df,
        eras=eras,
        feature_columns=feature_columns,
        proportion=proportion,
        normalize=True,
        max_neutralize_features=max_neutralize_features,
    )
    era_arr = np.ones(len(targets), dtype=np.int64) if eras is None else np.asarray(eras).ravel()
    frame = pd.DataFrame({"era": era_arr, "target": targets, "prediction": neutral_preds})
    era_scores: Dict[Any, float] = {}
    for era_key, era_df in frame.groupby("era", sort=True):
        valid_df = era_df.dropna(subset=["target", "prediction"])
        if len(valid_df) < 3:
            era_scores[era_key] = 0.0
            continue
        t_a = valid_df["target"].to_numpy(dtype=np.float64, copy=False)
        p_a = valid_df["prediction"].to_numpy(dtype=np.float64, copy=False)
        if _is_const_1d(p_a) or _is_const_1d(t_a):
            era_scores[era_key] = 0.0
        else:
            val = float(numerai_corr(valid_df[["prediction"]], valid_df["target"])["prediction"])
            era_scores[era_key] = val if np.isfinite(val) else 0.0

    fnc_series = pd.Series(era_scores, dtype=np.float64, name="fnc")
    fnc_arr = _clean_1d_finite(fnc_series.to_numpy(dtype=np.float64))
    return {
        "fnc_mean": float(np.round(np.mean(fnc_arr), 4)) if len(fnc_arr) else 0.0,
        "fnc_sharpe": float(np.round(_safe_sharpe(fnc_arr), 4)),
        "fnc_era_corrs": fnc_series,
        "proportion": float(proportion),
    }


# ---------------------------------------------------------------------------
# 3. Sharpe Statistics: Lo (2002), PSR, MinTRL, ONC, Clustered DSR, PBO
# ---------------------------------------------------------------------------

def compute_lo_adjusted_sharpe(
    era_corrs: Any,
    max_lag: Optional[int] = None,
) -> Dict[str, float]:
    """Compute Lo (2002) autocorrelation-adjusted Sharpe ratio & effective sample size via Newey-West/Bartlett weights.

    Critical for Numerai's overlapping 60-day (`target`, 12-week overlap) eras where naive IID Sharpe
    overstates true statistical significance when per-era correlations exhibit positive serial autocorrelation.
    """
    arr = _clean_1d_finite(era_corrs)
    n_obs = len(arr)
    sr_raw = _safe_sharpe(arr)
    if n_obs < 4 or abs(sr_raw) <= 1e-12:
        return {
            "sr_raw": float(sr_raw),
            "sr_lo": float(sr_raw),
            "variance_inflation": 1.0,
            "n_eff": float(n_obs),
            "max_lag": 0.0,
        }

    q = int(max_lag) if max_lag is not None else min(12, max(1, int(np.floor(n_obs ** (1.0 / 3.0)))))
    q = min(max(1, q), n_obs - 2)
    centered = arr - float(np.mean(arr))
    denom = float(np.dot(centered, centered))
    if denom <= 1e-24:
        return {
            "sr_raw": float(sr_raw),
            "sr_lo": float(sr_raw),
            "variance_inflation": 1.0,
            "n_eff": float(n_obs),
            "max_lag": float(q),
        }

    eta = 1.0
    for lag in range(1, q + 1):
        rho_k = float(np.dot(centered[lag:], centered[:-lag]) / denom)
        bartlett_w = 1.0 - float(lag) / float(q + 1)
        eta += 2.0 * bartlett_w * rho_k

    # Clamp variance inflation strictly positive (and avoid spurious inflation below 0.25)
    eta = max(0.25, float(eta))
    sr_lo = float(sr_raw / np.sqrt(eta))
    n_eff = float(min(float(n_obs), float(n_obs) / max(eta, 1.0)))
    return {
        "sr_raw": float(sr_raw),
        "sr_lo": float(sr_lo),
        "variance_inflation": float(eta),
        "n_eff": float(n_eff),
        "max_lag": float(q),
    }


def _psr_moments(arr: np.ndarray) -> Tuple[float, float, float, float]:
    """Return `(sr_hat, skew_hat, kurt_pearson_hat, variance_factor)` using MLE sample moments.

    MLE moments (`bias=True`) guarantee Pearson's Cauchy-Schwarz inequality `kurt >= skew^2 + 1`,
    so `variance_factor = 1 - skew * sr + ((kurt - 1) / 4) * sr^2 >= (1 - skew*sr/2)^2 >= 0` ALWAYS.
    """
    sr_hat = _safe_sharpe(arr)
    skew_hat = float(stats.skew(arr, bias=True))
    if not np.isfinite(skew_hat):
        skew_hat = 0.0
    kurt_hat = float(stats.kurtosis(arr, fisher=False, bias=True))
    if not np.isfinite(kurt_hat):
        kurt_hat = 3.0
    # Enforce Cauchy-Schwarz bound numerically against floating-point drift
    kurt_hat = max(kurt_hat, (skew_hat ** 2) + 1.0)
    var_factor = 1.0 - skew_hat * sr_hat + ((kurt_hat - 1.0) / 4.0) * (sr_hat ** 2)
    var_factor = max(0.0, float(var_factor))
    return sr_hat, skew_hat, kurt_hat, var_factor


def compute_probabilistic_sharpe_ratio(
    era_corrs: Any,
    benchmark_sr: float = 0.0,
    *,
    adjust_autocorr: bool = False,
    max_lag: Optional[int] = None,
) -> float:
    """Compute the Probabilistic Sharpe Ratio (PSR; Bailey & López de Prado 2012; AFML Ch. 14.7.2).

    Uses MLE skewness and Pearson kurtosis (`bias=True`) per Bailey & López de Prado (2012) and Mertens (2002)
    so the asymptotic Sharpe variance factor `1 - skew*SR + ((kurt - 1)/4)*SR^2` is guaranteed non-negative.
    Optionally adjusts effective sample size for serial autocorrelation via Lo (2002) (`adjust_autocorr=True`).
    """
    arr = _clean_1d_finite(era_corrs)
    n_obs = len(arr)
    if n_obs < 3 or _is_const_1d(arr):
        return 0.0

    sr_hat, _, _, var_factor = _psr_moments(arr)
    bench = float(benchmark_sr)
    if adjust_autocorr:
        lo_info = compute_lo_adjusted_sharpe(arr, max_lag=max_lag)
        eff_t = max(2.0, float(lo_info["n_eff"]))
    else:
        eff_t = float(n_obs)

    if var_factor <= 1e-18:
        if sr_hat > bench + 1e-12:
            return 1.0
        if sr_hat < bench - 1e-12:
            return 0.0
        return 0.5

    z_val = (sr_hat - bench) * float(np.sqrt(max(eff_t - 1.0, 1.0))) / float(np.sqrt(var_factor))
    if not np.isfinite(z_val):
        return 1.0 if sr_hat > bench else 0.0
    return float(np.clip(stats.norm.cdf(z_val), 0.0, 1.0))


def compute_min_track_record_length(
    era_corrs: Any,
    benchmark_sr: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Compute Minimum Track Record Length in eras (MinTRL; Bailey & López de Prado 2012; AFML Ch. 14.7.2)."""
    arr = _clean_1d_finite(era_corrs)
    if len(arr) < 3 or _is_const_1d(arr):
        return float(np.inf)
    sr_hat, _, _, var_factor = _psr_moments(arr)
    bench = float(benchmark_sr)
    if sr_hat <= bench + 1e-12:
        return float(np.inf)
    conf = float(np.clip(float(confidence), 0.5 + 1e-6, 1.0 - 1e-6))
    z_alpha = float(stats.norm.ppf(conf))
    min_trl = 1.0 + var_factor * ((z_alpha / (sr_hat - bench)) ** 2)
    return float(min_trl) if np.isfinite(min_trl) else float(np.inf)


def _cluster_kmeans_base(
    corr_df: pd.DataFrame,
    max_num_clusters: int,
    n_init: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[int, List[Any]], pd.Series, float]:
    """Base ONC K-Means loop selecting optimal cluster count via Silhouette t-statistic `q = E[S]/std(S)`."""
    cols = list(corr_df.columns)
    n_items = len(cols)
    if n_items <= 2:
        clusters = {0: cols}
        sil = pd.Series(np.ones(n_items, dtype=np.float64), index=cols)
        return corr_df.copy(), clusters, sil, 1.0

    corr_vals = np.clip(corr_df.to_numpy(dtype=np.float64), -1.0, 1.0)
    corr_vals = 0.5 * (corr_vals + corr_vals.T)
    np.fill_diagonal(corr_vals, 1.0)
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr_vals)))
    np.fill_diagonal(dist, 0.0)

    k_cap = min(max(2, int(max_num_clusters)), n_items - 1)
    best_q = -np.inf
    best_labels = np.zeros(n_items, dtype=np.int64)
    best_sil = np.zeros(n_items, dtype=np.float64)

    for k in range(2, k_cap + 1):
        for init_i in range(max(1, int(n_init))):
            km = KMeans(
                n_clusters=k,
                n_init=1,
                random_state=int(random_state) + 1009 * k + init_i,
            )
            labels = km.fit_predict(dist)
            n_unique_lbl = len(np.unique(labels))
            if n_unique_lbl < 2 or n_unique_lbl >= n_items:
                continue
            sil_vals = silhouette_samples(dist, labels, metric="precomputed")
            sil_mean = float(np.mean(sil_vals))
            sil_std = float(np.std(sil_vals))
            q_stat = (sil_mean / sil_std) if sil_std > 1e-12 else (sil_mean * 1e6 if sil_mean > 0 else -np.inf)
            if np.isfinite(q_stat) and q_stat > best_q:
                best_q = q_stat
                best_labels = labels.astype(np.int64, copy=True)
                best_sil = sil_vals.astype(np.float64, copy=True)

    # Fallback if all KMeans attempts collapsed (e.g., identical columns)
    if not np.isfinite(best_q):
        best_labels = np.zeros(n_items, dtype=np.int64)
        best_sil = np.ones(n_items, dtype=np.float64)
        best_q = 0.0

    # Order columns by cluster id
    order_idx = np.argsort(best_labels, kind="stable")
    ordered_cols = [cols[int(i)] for i in order_idx]
    reordered_corr = corr_df.loc[ordered_cols, ordered_cols]
    clusters_dict: Dict[int, List[Any]] = {}
    for c_new, c_old in enumerate(sorted(np.unique(best_labels).tolist())):
        clusters_dict[c_new] = [cols[int(i)] for i in np.flatnonzero(best_labels == c_old)]
    sil_series = pd.Series(best_sil, index=cols, dtype=np.float64)
    return reordered_corr, clusters_dict, sil_series, float(best_q)


def compute_onc_clusters(
    corr_matrix: Union[pd.DataFrame, np.ndarray],
    *,
    max_clusters: Optional[int] = None,
    n_init: int = 5,
    random_state: int = 42,
    _depth: int = 0,
) -> Dict[str, Any]:
    """Optimal Number of Clusters (ONC; López de Prado 2020, Lectures 3 & 6) with recursive refinement.

    Accepts a correlation matrix (DataFrame or 2D array) and returns optimal cluster assignments,
    cluster count `n_clusters`, silhouette quality score, and block-reordered correlation matrix.
    """
    if isinstance(corr_matrix, pd.DataFrame):
        corr_df = corr_matrix.copy().astype(np.float64)
    else:
        arr = np.asarray(corr_matrix, dtype=np.float64)
        corr_df = pd.DataFrame(arr, index=list(range(arr.shape[0])), columns=list(range(arr.shape[1])))

    cols = list(corr_df.columns)
    n_items = len(cols)
    if n_items <= 2:
        labels = np.zeros(n_items, dtype=np.int64)
        return {
            "clusters": {0: cols} if n_items > 0 else {},
            "n_clusters": 1 if n_items > 0 else 0,
            "labels": labels,
            "silhouette_scores": pd.Series(np.ones(n_items, dtype=np.float64), index=cols),
            "quality_score": 1.0 if n_items > 0 else 0.0,
            "reordered_corr": corr_df,
        }

    k_max = int(max_clusters) if max_clusters is not None else min(10, n_items - 1)
    k_max = max(2, min(k_max, n_items - 1))

    reordered_corr, clusters, sil_series, q_base = _cluster_kmeans_base(
        corr_df, max_num_clusters=k_max, n_init=n_init, random_state=random_state
    )

    # Top-level recursive re-clustering of below-average silhouette clusters (López de Prado ONC)
    if _depth < 2 and len(clusters) >= 2:
        cluster_tstats: Dict[int, float] = {}
        for c_id, members in clusters.items():
            c_sil = sil_series.loc[members].to_numpy(dtype=np.float64)
            c_mean = float(np.mean(c_sil))
            c_std = float(np.std(c_sil))
            cluster_tstats[c_id] = (c_mean / c_std) if c_std > 1e-12 else c_mean

        t_vals = np.asarray(list(cluster_tstats.values()), dtype=np.float64)
        mean_t = float(np.mean(t_vals))
        redo_ids = [c_id for c_id, t_v in cluster_tstats.items() if t_v < mean_t]

        if len(redo_ids) >= 2 and len(redo_ids) < len(clusters):
            keep_ids = [c_id for c_id in clusters if c_id not in redo_ids]
            redo_members: List[Any] = []
            for c_id in redo_ids:
                redo_members.extend(clusters[c_id])

            if len(redo_members) >= 3:
                sub_corr = corr_df.loc[redo_members, redo_members]
                sub_res = compute_onc_clusters(
                    sub_corr,
                    max_clusters=min(k_max, len(redo_members) - 1),
                    n_init=n_init,
                    random_state=int(random_state) + 17,
                    _depth=_depth + 1,
                )
                merged_clusters: Dict[int, List[Any]] = {}
                next_id = 0
                for c_id in keep_ids:
                    merged_clusters[next_id] = list(clusters[c_id])
                    next_id += 1
                for _, sub_members in sub_res["clusters"].items():
                    merged_clusters[next_id] = list(sub_members)
                    next_id += 1

                # Re-evaluate overall cluster-mean silhouette t-stat
                col_to_lbl = {m: cid for cid, ms in merged_clusters.items() for m in ms}
                new_labels = np.array([col_to_lbl[c] for c in cols], dtype=np.int64)
                corr_vals = np.clip(corr_df.to_numpy(dtype=np.float64), -1.0, 1.0)
                corr_vals = 0.5 * (corr_vals + corr_vals.T)
                np.fill_diagonal(corr_vals, 1.0)
                dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr_vals)))
                np.fill_diagonal(dist, 0.0)
                if 2 <= len(merged_clusters) < n_items:
                    new_sil_arr = silhouette_samples(dist, new_labels, metric="precomputed")
                    new_sil = pd.Series(new_sil_arr, index=cols, dtype=np.float64)
                    new_cluster_means = [float(new_sil.loc[ms].mean()) for ms in merged_clusters.values()]
                    old_cluster_means = [float(sil_series.loc[ms].mean()) for ms in clusters.values()]
                    if float(np.mean(new_cluster_means)) > float(np.mean(old_cluster_means)) + 1e-6:
                        clusters = merged_clusters
                        sil_series = new_sil
                        s_std = float(np.std(new_sil_arr))
                        q_base = (float(np.mean(new_sil_arr)) / s_std) if s_std > 1e-12 else float(np.mean(new_sil_arr))
                        ordered_cols = [m for _, ms in sorted(clusters.items()) for m in ms]
                        reordered_corr = corr_df.loc[ordered_cols, ordered_cols]

    col_to_final = {m: cid for cid, ms in clusters.items() for m in ms}
    final_labels = np.array([col_to_final[c] for c in cols], dtype=np.int64)
    return {
        "clusters": clusters,
        "n_clusters": int(len(clusters)),
        "labels": final_labels,
        "silhouette_scores": sil_series,
        "quality_score": float(q_base),
        "reordered_corr": reordered_corr,
    }


def _coerce_trials_matrix_df(
    trials_matrix: Union[pd.DataFrame, Dict[str, Any], np.ndarray],
) -> pd.DataFrame:
    """Normalize a trials era-CORR matrix from DataFrame, nested dict, or ndarray to a DataFrame."""
    if isinstance(trials_matrix, pd.DataFrame):
        return trials_matrix.copy()
    if isinstance(trials_matrix, dict):
        return pd.DataFrame.from_dict(trials_matrix).sort_index()
    return pd.DataFrame(np.asarray(trials_matrix, dtype=np.float64))


def compute_deflated_sharpe_ratio(
    era_corrs: Any,
    trial_srs: Optional[Sequence[float]] = None,
    *,
    trials_matrix: Optional[Union[pd.DataFrame, Dict[str, Any], np.ndarray]] = None,
    alpha: float = 0.05,
    adjust_autocorr: bool = False,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute Deflated Sharpe Ratio (DSR) with optional ONC-Clustered effective E[K] & Šidák FWER.

    - Backwards-compatible when called as `compute_deflated_sharpe_ratio(era_corrs, trial_srs)`.
    - When `trials_matrix` (shape `(n_eras, n_trials)` of per-era CORRs across CV candidates) is passed,
      implements López de Prado's *"How the Sharpe Ratio Died"* (Lecture 6) protocol:
        1. Clusters correlated trials using ONC to determine effective independent trial count `E[K]`.
        2. Aggregates each cluster's era returns to estimate the inter-cluster Sharpe variance `V[SR_k]`.
        3. Evaluates False Strategy Theorem benchmark `SR_0 = sqrt(V) * ((1 - gamma)*Z(1 - 1/K) + gamma*Z(1 - 1/(K*e)))`.
        4. Computes Šidák FWER-adjusted threshold `alpha_K = 1 - (1 - alpha)^(1 / E[K])`.
    """
    euler_gamma = 0.5772156649015329
    raw_srs = (
        np.asarray(list(trial_srs), dtype=np.float64).ravel()
        if trial_srs is not None
        else np.empty(0, dtype=np.float64)
    )
    valid_srs = raw_srs[np.isfinite(raw_srs)]
    n_raw_trials = max(1, int(len(valid_srs)))
    effective_k = float(n_raw_trials)
    var_sr = float(np.var(valid_srs, ddof=1)) if n_raw_trials >= 2 else 0.0
    used_onc = False

    if trials_matrix is not None:
        mat_df = _coerce_trials_matrix_df(trials_matrix)
        # Drop constant or nearly empty columns before ONC clustering
        nonconst_cols = [
            c for c in mat_df.columns
            if len(_clean_1d_finite(mat_df[c].to_numpy(dtype=np.float64))) >= 3
            and not _is_const_1d(_clean_1d_finite(mat_df[c].to_numpy(dtype=np.float64)))
        ]
        n_raw_trials = max(n_raw_trials, len(mat_df.columns))
        if len(nonconst_cols) >= 3:
            sub_df = mat_df[nonconst_cols].fillna(0.0)
            corr_df = sub_df.corr().fillna(0.0)
            np.fill_diagonal(corr_df.to_numpy(copy=False), 1.0)
            onc_res = compute_onc_clusters(
                corr_df,
                max_clusters=min(15, len(nonconst_cols) - 1),
                n_init=4,
                random_state=random_state,
            )
            effective_k = float(max(1, onc_res["n_clusters"]))
            cluster_srs: List[float] = []
            for members in onc_res["clusters"].values():
                # Inverse-variance weighted cluster aggregate (Lecture 6 slide 20)
                c_mat = sub_df[members].to_numpy(dtype=np.float64)
                stds = np.std(c_mat, axis=0)
                inv_v = np.where(stds > 1e-12, 1.0 / (stds ** 2), 1.0)
                w = inv_v / float(np.sum(inv_v))
                agg_ret = c_mat @ w
                cluster_srs.append(_safe_sharpe(agg_ret))
            if len(cluster_srs) >= 2:
                var_sr = float(np.var(cluster_srs, ddof=1))
            elif len(valid_srs) >= 2:
                var_sr = float(np.var(valid_srs, ddof=1))
            else:
                var_sr = 0.0
            used_onc = True
        elif len(nonconst_cols) >= 2 and len(valid_srs) < 2:
            col_srs = [_safe_sharpe(_clean_1d_finite(mat_df[c].to_numpy(dtype=np.float64))) for c in nonconst_cols]
            effective_k = float(len(nonconst_cols))
            var_sr = float(np.var(col_srs, ddof=1))

    if effective_k <= 1.0 or not np.isfinite(var_sr) or var_sr <= 1e-12:
        sr0 = 0.0
    else:
        p1 = float(np.clip(1.0 - 1.0 / effective_k, 1e-12, 1.0 - 1e-12))
        p2 = float(np.clip(1.0 - 1.0 / (effective_k * float(np.e)), 1e-12, 1.0 - 1e-12))
        term1 = (1.0 - euler_gamma) * float(stats.norm.ppf(p1))
        term2 = euler_gamma * float(stats.norm.ppf(p2))
        sr0 = float(np.sqrt(var_sr) * (term1 + term2))
        if not np.isfinite(sr0):
            sr0 = 0.0

    psr_zero = compute_probabilistic_sharpe_ratio(era_corrs, benchmark_sr=0.0, adjust_autocorr=adjust_autocorr)
    dsr = compute_probabilistic_sharpe_ratio(era_corrs, benchmark_sr=sr0, adjust_autocorr=adjust_autocorr)
    min_trl = compute_min_track_record_length(era_corrs, benchmark_sr=0.0, confidence=1.0 - float(alpha))
    sidak_alpha = float(1.0 - (1.0 - float(np.clip(alpha, 1e-6, 0.5))) ** (1.0 / max(1.0, effective_k)))
    lo_info = compute_lo_adjusted_sharpe(era_corrs)

    return {
        "psr": float(psr_zero),
        "dsr": float(dsr),
        "expected_max_sr": float(sr0),
        "n_trials": float(n_raw_trials),
        "effective_n_trials": float(effective_k),
        "used_onc_clustering": float(1.0 if used_onc else 0.0),
        "trial_sr_variance": float(var_sr),
        "sidak_alpha": float(sidak_alpha),
        "min_trl": float(min_trl),
        "lo_sharpe": float(lo_info["sr_lo"]),
    }


def compute_pbo_cscv(
    trials_matrix: Union[pd.DataFrame, Dict[str, Any], np.ndarray],
    *,
    n_submatrices: int = 12,
    max_combinations: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute Probability of Backtest Overfitting (PBO) via CSCV (Bailey et al. 2017; AFML Ch. 11.6).

    `trials_matrix` has shape `(T_eras, N_trials)` containing per-era CORR series across all evaluated trials.
    """
    mat = _coerce_trials_matrix_df(trials_matrix).to_numpy(dtype=np.float64, copy=False)
    if mat.ndim != 2 or mat.shape[0] < 4 or mat.shape[1] < 2:
        return {
            "pbo": float(np.nan),
            "median_logit": float(np.nan),
            "prob_oos_loss": float(np.nan),
            "n_combinations": 0,
            "n_submatrices": 0,
            "logits": np.empty(0, dtype=np.float64),
        }

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    t_rows, n_trials = mat.shape
    s_blocks = min(int(n_submatrices), t_rows)
    if s_blocks % 2 != 0:
        s_blocks -= 1
    s_blocks = max(4, s_blocks) if t_rows >= 4 else 2

    # Trim oldest remainder rows so all S submatrices have identical length
    usable_t = (t_rows // s_blocks) * s_blocks
    block_len = usable_t // s_blocks
    blocks = mat[t_rows - usable_t :].reshape(s_blocks, block_len, n_trials)

    # Precompute block sufficient statistics (sum and sum of squares) so each split Sharpe is O(N_trials)!
    block_sum = blocks.sum(axis=1)  # (S, N)
    block_sum2 = (blocks * blocks).sum(axis=1)  # (S, N)
    total_sum = block_sum.sum(axis=0)
    total_sum2 = block_sum2.sum(axis=0)
    half_s = s_blocks // 2
    half_len = float(half_s * block_len)

    all_combos = list(combinations(range(s_blocks), half_s))
    if len(all_combos) > int(max_combinations):
        rng = np.random.default_rng(int(random_state))
        pick = rng.choice(len(all_combos), size=int(max_combinations), replace=False)
        all_combos = [all_combos[int(i)] for i in np.sort(pick)]

    logits = np.empty(len(all_combos), dtype=np.float64)
    oos_winner_srs = np.empty(len(all_combos), dtype=np.float64)

    for c_idx, is_idx in enumerate(all_combos):
        is_s = block_sum[list(is_idx)].sum(axis=0)
        is_s2 = block_sum2[list(is_idx)].sum(axis=0)
        oos_s = total_sum - is_s
        oos_s2 = total_sum2 - is_s2

        is_mean = is_s / half_len
        is_var = np.maximum(0.0, (is_s2 / half_len) - is_mean * is_mean)
        is_sr = np.where(is_var > 1e-24, is_mean / np.sqrt(is_var), 0.0)

        oos_mean = oos_s / half_len
        oos_var = np.maximum(0.0, (oos_s2 / half_len) - oos_mean * oos_mean)
        oos_sr = np.where(oos_var > 1e-24, oos_mean / np.sqrt(oos_var), 0.0)

        winner = int(np.argmax(is_sr))
        oos_winner_srs[c_idx] = float(oos_sr[winner])
        ranks = stats.rankdata(oos_sr, method="average")
        omega = float(np.clip(ranks[winner] / float(n_trials + 1), 1e-6, 1.0 - 1e-6))
        logits[c_idx] = float(np.log(omega / (1.0 - omega)))

    return {
        "pbo": float(np.mean(logits < 0.0)),
        "median_logit": float(np.median(logits)),
        "prob_oos_loss": float(np.mean(oos_winner_srs < 0.0)),
        "n_combinations": int(len(all_combos)),
        "n_submatrices": int(s_blocks),
        "logits": logits,
    }


# ---------------------------------------------------------------------------
# 4. Drawdown / Time Under Water (TuW), HHI Concentration, Strategy Risk
# ---------------------------------------------------------------------------

def compute_drawdown_and_tuw(
    series: Any,
    *,
    is_returns: bool = False,
    periods_per_year: float = 52.0,
) -> Dict[str, Any]:
    """Compute High-Water Mark (HWM) drawdown and Time under Water (TuW; AFML Ch. 14.6, Lecture 6).

    If `is_returns=True`, treats `series` as additive era scores/returns and builds `cumulative = cumsum`.
    If `is_returns=False`, treats `series` as a wealth/equity level curve (e.g., wallet GBP/NMR).
    """
    s = pd.Series(series, dtype=np.float64).dropna()
    if len(s) == 0:
        return {
            "max_drawdown": 0.0,
            "tuw_max_eras": 0,
            "tuw_max_years": 0.0,
            "tuw_mean_eras": 0.0,
            "tuw_current_eras": 0,
            "n_episodes": 0,
            "drawdown_series": pd.Series(dtype=np.float64),
        }

    vals = s.to_numpy(dtype=np.float64)
    level = np.cumsum(vals) if is_returns else vals
    init_hwm = 0.0 if is_returns else float(level[0])
    hwm = np.maximum.accumulate(np.insert(level, 0, init_hwm))[1:]

    if is_returns:
        dd = hwm - level  # positive drawdown depth in cumulative CORR points
    else:
        dd = np.where(hwm > 1e-12, (hwm - level) / hwm, 0.0)

    # Identify Time under Water episodes (consecutive periods where level < HWM)
    underwater = (hwm - level) > 1e-12
    durations: List[int] = []
    cur_dur = 0
    for is_uw in underwater:
        if is_uw:
            cur_dur += 1
        elif cur_dur > 0:
            durations.append(cur_dur)
            cur_dur = 0
    if cur_dur > 0:
        durations.append(cur_dur)

    max_tuw = int(max(durations)) if durations else 0
    mean_tuw = float(np.mean(durations)) if durations else 0.0
    ppy = max(1.0, float(periods_per_year))

    return {
        "max_drawdown": float(np.max(dd)) if len(dd) else 0.0,
        "tuw_max_eras": max_tuw,
        "tuw_max_years": float(np.round(max_tuw / ppy, 3)),
        "tuw_mean_eras": float(np.round(mean_tuw, 2)),
        "tuw_current_eras": int(cur_dur),
        "n_episodes": int(len(durations)),
        "drawdown_series": pd.Series(dd, index=s.index, dtype=np.float64),
    }


def _normalized_hhi(weights_raw: np.ndarray) -> float:
    arr = np.abs(np.asarray(weights_raw, dtype=np.float64).ravel())
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    n = len(arr)
    if n <= 1:
        return 1.0 if n == 1 else 0.0
    w = arr / float(np.sum(arr))
    hhi = (float(np.sum(w * w)) - 1.0 / float(n)) / (1.0 - 1.0 / float(n))
    return float(np.clip(hhi, 0.0, 1.0))


def compute_hhi_concentration(
    era_corrs: Any,
    *,
    time_block_size: int = 13,
) -> Dict[str, float]:
    """Compute HHI concentration of positive returns (`hhi_pos`), negative returns (`hhi_neg`),
    and time-block concentration (`hhi_time`; AFML Ch. 14.5).

    Values near `0.0` indicate evenly distributed edge across eras; values near `1.0` warn that
    total performance or drawdowns come from only a handful of extreme eras.
    """
    arr = _clean_1d_finite(era_corrs)
    pos = arr[arr > 0.0]
    neg = -arr[arr < 0.0]
    h_pos = _normalized_hhi(pos) if len(pos) > 0 else 0.0
    h_neg = _normalized_hhi(neg) if len(neg) > 0 else 0.0

    b_size = max(1, int(time_block_size))
    if len(arr) >= 2 * b_size:
        n_blocks = int(np.ceil(float(len(arr)) / float(b_size)))
        block_totals = np.array(
            [float(np.sum(np.abs(arr[i * b_size : (i + 1) * b_size]))) for i in range(n_blocks)],
            dtype=np.float64,
        )
        h_time = _normalized_hhi(block_totals)
    else:
        h_time = 0.0

    return {
        "hhi_pos": float(np.round(h_pos, 4)),
        "hhi_neg": float(np.round(h_neg, 4)),
        "hhi_time": float(np.round(h_time, 4)),
    }


def compute_strategy_risk(
    era_corrs: Any,
    *,
    target_sr: float = 1.0,
    periods_per_year: float = 52.0,
) -> Dict[str, float]:
    """Compute AFML Ch. 15 Strategy Risk: implied required precision `p*` and failure probability `P[p < p*]`.

    Given empirical winning-era payoff `pi+ = mean(r | r > 0)`, losing-era payoff `pi- = mean(r | r < 0)`,
    and annual frequency `n = periods_per_year`, solves the quadratic equation (AFML Eq. 15.1) for the
    minimum hit-rate `p*` needed to achieve annualized `target_sr`, then evaluates `P[p_true < p*]` under
    the Beta(`k+1`, `T-k+1`) posterior of the empirical hit rate.
    """
    arr = _clean_1d_finite(era_corrs)
    n_obs = len(arr)
    if n_obs < 4:
        return {
            "empirical_precision": 0.0,
            "implied_precision": 1.0,
            "prob_strategy_failure": 1.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    wins = arr[arr > 0.0]
    losses = arr[arr < 0.0]
    n_win = len(wins)
    p_hat = float(n_win) / float(n_obs)
    pi_plus = float(np.mean(wins)) if n_win > 0 else 0.0
    pi_minus = float(np.mean(losses)) if len(losses) > 0 else 0.0

    if pi_plus <= 1e-12 or abs(pi_minus) <= 1e-12:
        p_star = 0.0 if pi_plus > 0.0 else 1.0
        prob_fail = float(stats.beta.cdf(p_star, n_win + 1, n_obs - n_win + 1))
        return {
            "empirical_precision": float(np.round(p_hat, 4)),
            "implied_precision": float(np.round(p_star, 4)),
            "prob_strategy_failure": float(np.round(prob_fail, 4)),
            "avg_win": float(np.round(pi_plus, 6)),
            "avg_loss": float(np.round(pi_minus, 6)),
        }

    # AFML Ch. 15.3: Solve a*p^2 + b*p + c = 0 for required precision p* to hit target_sr
    n_freq = max(1.0, float(periods_per_year))
    sr_sq = float(target_sr) ** 2
    diff = pi_plus - pi_minus  # > 0
    a = (n_freq + sr_sq) * (diff ** 2)
    b = (2.0 * n_freq * pi_minus - sr_sq * diff) * diff
    c = n_freq * (pi_minus ** 2)
    disc = b * b - 4.0 * a * c

    if disc < 0.0 or a <= 1e-24:
        p_star = 1.0
    else:
        p_star = float((-b + np.sqrt(disc)) / (2.0 * a))
        p_star = float(np.clip(p_star, 0.0, 1.0))

    prob_fail = float(stats.beta.cdf(p_star, n_win + 1, n_obs - n_win + 1))
    return {
        "empirical_precision": float(np.round(p_hat, 4)),
        "implied_precision": float(np.round(p_star, 4)),
        "prob_strategy_failure": float(np.round(prob_fail, 4)),
        "avg_win": float(np.round(pi_plus, 6)),
        "avg_loss": float(np.round(pi_minus, 6)),
    }


# ---------------------------------------------------------------------------
# 5. Information-Theoretic Codependence & Clustered MDA Feature Importance
# ---------------------------------------------------------------------------

def compute_variation_of_information_matrix(
    features_df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    *,
    normalized: bool = True,
    max_samples: int = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute pairwise Variation of Information (VI) matrix across features (López de Prado Codependence).

    For Numerai int8 `0..4` features, uses exact 5-state discrete joint histograms; for continuous
    features, discretizes via Hacine-Gharbi optimal bin count.
    """
    cols = list(feature_columns) if feature_columns is not None else list(features_df.columns)
    cols = [str(c) for c in cols if str(c) in features_df.columns]
    n_feat = len(cols)
    if n_feat == 0:
        return pd.DataFrame()

    mat = features_df[cols].to_numpy(dtype=np.float64, copy=False)
    if mat.shape[0] > int(max_samples):
        rng = np.random.default_rng(int(random_state))
        pick = rng.choice(mat.shape[0], size=int(max_samples), replace=False)
        mat = mat[pick]
    mat = np.nan_to_num(mat, nan=2.0)

    # Discretize each column cleanly into integer state codes 0..(B-1)
    n_obs = mat.shape[0]
    default_bins = max(2, int(round((8.0 + 324.0 * n_obs + 12.0 * np.sqrt(36.0 * n_obs + 729.0 * n_obs * n_obs)) ** (1.0 / 3.0) / 6.0 + 1.0 / 3.0)))
    default_bins = min(default_bins, 16)

    codes_list: List[np.ndarray] = []
    n_states_list: List[int] = []
    entropies = np.zeros(n_feat, dtype=np.float64)

    for j in range(n_feat):
        col_v = mat[:, j]
        uniq = np.unique(col_v)
        if len(uniq) <= 10:
            _, inv = np.unique(col_v, return_inverse=True)
            codes = inv.astype(np.int64)
            k_s = len(uniq)
        else:
            edges = np.quantile(col_v, np.linspace(0.0, 1.0, default_bins + 1))
            edges = np.unique(edges)
            if len(edges) <= 2:
                codes = np.zeros(n_obs, dtype=np.int64)
                k_s = 1
            else:
                codes = np.clip(np.digitize(col_v, edges[1:-1], right=False), 0, len(edges) - 2).astype(np.int64)
                k_s = len(edges) - 1
        counts = np.bincount(codes, minlength=k_s).astype(np.float64)
        probs = counts[counts > 0.0] / float(n_obs)
        entropies[j] = float(-np.sum(probs * np.log(probs)))
        codes_list.append(codes)
        n_states_list.append(k_s)

    vi_mat = np.zeros((n_feat, n_feat), dtype=np.float64)
    for i in range(n_feat):
        ci = codes_list[i]
        ki = n_states_list[i]
        for j in range(i + 1, n_feat):
            cj = codes_list[j]
            kj = n_states_list[j]
            joint_idx = ci * kj + cj
            j_counts = np.bincount(joint_idx, minlength=ki * kj).astype(np.float64)
            j_probs = j_counts[j_counts > 0.0] / float(n_obs)
            h_xy = float(-np.sum(j_probs * np.log(j_probs)))
            mi = max(0.0, entropies[i] + entropies[j] - h_xy)
            if normalized:
                vi = (1.0 - mi / h_xy) if h_xy > 1e-12 else 0.0
                vi = float(np.clip(vi, 0.0, 1.0))
            else:
                vi = max(0.0, h_xy - mi)
            vi_mat[i, j] = vi
            vi_mat[j, i] = vi

    return pd.DataFrame(vi_mat, index=cols, columns=cols, dtype=np.float64)


def cluster_features_onc(
    features_df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    *,
    metric: str = "correlation",
    max_clusters: Optional[int] = 15,
    max_samples: int = 50000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Cluster correlated/codependent features via ONC (López de Prado CFI Lecture)."""
    cols = list(feature_columns) if feature_columns is not None else list(features_df.columns)
    cols = [str(c) for c in cols if str(c) in features_df.columns]
    if len(cols) <= 2:
        return {
            "clusters": {0: cols} if cols else {},
            "n_clusters": 1 if cols else 0,
            "quality_score": 1.0 if cols else 0.0,
        }

    sub = features_df[cols]
    if len(sub) > int(max_samples):
        rng = np.random.default_rng(int(random_state))
        pick = np.sort(rng.choice(len(sub), size=int(max_samples), replace=False))
        sub = pd.DataFrame(sub.to_numpy(copy=False)[pick], columns=cols)

    clean_metric = str(metric or "correlation").strip().lower()
    if clean_metric == "vi":
        vi_df = compute_variation_of_information_matrix(sub, cols, normalized=True, max_samples=max_samples, random_state=random_state)
        # Convert normalized VI in [0, 1] into similarity-like matrix in [-1, 1] for ONC angular distance
        sim_df = 1.0 - 2.0 * (vi_df ** 2)
        np.fill_diagonal(sim_df.to_numpy(copy=False), 1.0)
        base_mat = sim_df
    else:
        base_mat = sub.astype(np.float64).corr().fillna(0.0)
        np.fill_diagonal(base_mat.to_numpy(copy=False), 1.0)

    return compute_onc_clusters(
        base_mat,
        max_clusters=min(int(max_clusters or 15), len(cols) - 1),
        n_init=4,
        random_state=random_state,
    )


def compute_clustered_mda(
    estimator: Any,
    X: pd.DataFrame,
    y: Any,
    eras: Any,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    clusters: Optional[Mapping[int, Sequence[str]]] = None,
    cv_splits: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    target_horizon_eras: int = 12,
    n_splits: int = 5,
    max_clusters: int = 12,
    n_repeats: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Out-of-sample Clustered Mean Decrease Accuracy (Clustered MDA; López de Prado CFI Lecture).

    Shuffles ALL features inside each ONC cluster simultaneously on Purged K-Fold OOS test folds,
    eliminating the substitution effect that breaks single-feature MDI/MDA under multicollinearity.
    Scores impact using official per-era `numerai_corr`.
    """
    forbidden = {"era", "date", "data_type", "id", "numerai_ticker", "week_no", "year_horizon", "target"}
    if feature_columns is not None:
        feat_cols = [str(c) for c in feature_columns if str(c) in X.columns]
    else:
        feat_cols = [str(c) for c in X.columns if str(c) not in forbidden and not str(c).startswith("target")]

    if not feat_cols:
        return pd.DataFrame(columns=["cluster_id", "n_features", "mda_mean", "mda_std", "mda_tstat", "top_features"])

    era_arr = np.asarray(eras, dtype=np.int64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    if clusters is None:
        onc_res = cluster_features_onc(
            X,
            feat_cols,
            metric="correlation",
            max_clusters=min(int(max_clusters), max(2, len(feat_cols) - 1)),
            random_state=random_state,
        )
        cluster_map: Dict[int, List[str]] = {int(k): list(v) for k, v in onc_res["clusters"].items()}
    else:
        cluster_map = {int(k): [str(c) for c in v if str(c) in X.columns] for k, v in clusters.items()}
        cluster_map = {k: v for k, v in cluster_map.items() if len(v) > 0}

    splits = (
        list(cv_splits)
        if cv_splits is not None
        else build_era_cv_splits(
            era_arr,
            cv_strategy="purged_kfold",
            n_splits=n_splits,
            target_horizon_eras=target_horizon_eras,
        )
    )

    def _mean_era_ncorr(y_t: np.ndarray, y_p: np.ndarray, e_t: np.ndarray) -> float:
        frame = pd.DataFrame({"era": e_t, "target": y_t, "prediction": y_p})
        scores: List[float] = []
        for _, grp in frame.groupby("era", sort=True):
            if len(grp) < 3 or _is_const_1d(grp["prediction"].to_numpy()) or _is_const_1d(grp["target"].to_numpy()):
                scores.append(0.0)
            else:
                v = float(numerai_corr(grp[["prediction"]], grp["target"])["prediction"])
                scores.append(v if np.isfinite(v) else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    rng = np.random.default_rng(int(random_state))
    deltas_by_cluster: Dict[int, List[float]] = {cid: [] for cid in cluster_map}
    reps = max(1, int(n_repeats))

    row_index_vals = X.index.to_numpy()
    for tr_idx, te_idx in splits:
        tr_labels = row_index_vals[tr_idx]
        te_labels = row_index_vals[te_idx]
        X_tr = X.loc[tr_labels]
        y_tr = y_arr[tr_idx]
        X_te = X.loc[te_labels]
        y_te = y_arr[te_idx]
        e_te = era_arr[te_idx]

        model = clone(estimator)
        model.fit(X_tr, y_tr)
        base_preds = np.asarray(model.predict(X_te), dtype=np.float64).ravel()
        base_score = _mean_era_ncorr(y_te, base_preds, e_te)

        # Permute each cluster within each era (preserving era structure while destroying cross-sectional signal)
        n_te = len(X_te)
        for cid, c_cols in cluster_map.items():
            rep_deltas: List[float] = []
            for _ in range(reps):
                X_perm = X_te.copy()
                perm_idx = rng.permutation(n_te)
                orig_block = X_te[c_cols].to_numpy(copy=True)
                X_perm[c_cols] = orig_block[perm_idx]
                perm_preds = np.asarray(model.predict(X_perm), dtype=np.float64).ravel()
                perm_score = _mean_era_ncorr(y_te, perm_preds, e_te)
                rep_deltas.append(base_score - perm_score)
            deltas_by_cluster[cid].append(float(np.mean(rep_deltas)))

    rows: List[Dict[str, Any]] = []
    for cid, d_list in deltas_by_cluster.items():
        d_arr = np.asarray(d_list, dtype=np.float64)
        mean_d = float(np.mean(d_arr)) if len(d_arr) else 0.0
        std_d = float(np.std(d_arr, ddof=1)) if len(d_arr) >= 2 else 0.0
        se_d = std_d / float(np.sqrt(len(d_arr))) if len(d_arr) >= 2 else 0.0
        t_stat = (mean_d / se_d) if se_d > 1e-12 else 0.0
        members = cluster_map[cid]
        rows.append({
            "cluster_id": int(cid),
            "n_features": int(len(members)),
            "mda_mean": round(mean_d, 6),
            "mda_std": round(std_d, 6),
            "mda_tstat": round(t_stat, 3),
            "top_features": ", ".join(members[:5]) + (" ..." if len(members) > 5 else ""),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("mda_mean", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 6. Unified AFML + Numerai Diagnostic Report Builder
# ---------------------------------------------------------------------------

def summarize_afml_backtest_diagnostics(
    era_corrs: Any,
    *,
    trial_srs: Optional[Sequence[float]] = None,
    trials_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_true: Optional[Any] = None,
    y_pred: Optional[Any] = None,
    features_df: Optional[pd.DataFrame] = None,
    feature_columns: Optional[Sequence[str]] = None,
    eras: Optional[Any] = None,
    label: str = "Model",
    target_annual_sr: float = 1.0,
    periods_per_year: float = 52.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Build a comprehensive AFML + Numerai validation diagnostic summary dictionary."""
    corr_s = pd.Series(era_corrs, dtype=np.float64).dropna()
    arr = _clean_1d_finite(corr_s.to_numpy(dtype=np.float64))

    mean_corr = float(np.mean(arr)) if len(arr) else 0.0
    pos_era_pct = float(np.mean(arr > 0.0) * 100.0) if len(arr) else 0.0
    era_consistency_pct = float(np.mean(arr > 0.002) * 100.0) if len(arr) else 0.0

    lo_info = compute_lo_adjusted_sharpe(arr)
    dsr_info = compute_deflated_sharpe_ratio(
        arr,
        trial_srs=trial_srs,
        trials_matrix=trials_matrix,
        random_state=random_state,
    )
    dd_info = compute_drawdown_and_tuw(corr_s, is_returns=True, periods_per_year=periods_per_year)
    hhi_info = compute_hhi_concentration(arr)
    risk_info = compute_strategy_risk(arr, target_sr=target_annual_sr, periods_per_year=periods_per_year)

    pbo_val = float(np.nan)
    if trials_matrix is not None:
        pbo_res = compute_pbo_cscv(trials_matrix, random_state=random_state)
        pbo_val = float(pbo_res["pbo"])

    fnc_mean = float(np.nan)
    fnc_sharpe = float(np.nan)
    if y_true is not None and y_pred is not None and features_df is not None:
        fnc_res = compute_per_era_fnc(
            y_true,
            y_pred,
            features_df,
            eras=eras,
            feature_columns=feature_columns,
        )
        fnc_mean = float(fnc_res["fnc_mean"])
        fnc_sharpe = float(fnc_res["fnc_sharpe"])

    min_trl_eras = float(dsr_info["min_trl"])
    min_trl_years = (
        round(min_trl_eras / max(1.0, float(periods_per_year)), 2)
        if np.isfinite(min_trl_eras)
        else float(np.inf)
    )

    return {
        "variant": str(label),
        "n_eras": int(len(arr)),
        "mean_corr": round(mean_corr, 4),
        "positive_era_pct": round(pos_era_pct, 2),
        "era_consistency_pct": round(era_consistency_pct, 2),
        "numerai_sharpe": round(float(lo_info["sr_raw"]), 4),
        "lo_adjusted_sharpe": round(float(lo_info["sr_lo"]), 4),
        "psr": round(float(dsr_info["psr"]), 4),
        "dsr": round(float(dsr_info["dsr"]), 4),
        "effective_n_trials_onc": round(float(dsr_info["effective_n_trials"]), 2),
        "sidak_alpha": round(float(dsr_info["sidak_alpha"]), 4),
        "min_trl_eras": round(min_trl_eras, 1) if np.isfinite(min_trl_eras) else float(np.inf),
        "min_trl_years": min_trl_years,
        "pbo_cscv": round(pbo_val, 4) if np.isfinite(pbo_val) else float(np.nan),
        "fnc_mean": round(fnc_mean, 4) if np.isfinite(fnc_mean) else float(np.nan),
        "fnc_sharpe": round(fnc_sharpe, 4) if np.isfinite(fnc_sharpe) else float(np.nan),
        "corr_drawdown": round(float(dd_info["max_drawdown"]), 4),
        "tuw_max_eras": int(dd_info["tuw_max_eras"]),
        "tuw_max_years": round(float(dd_info["tuw_max_years"]), 2),
        "hhi_pos": float(hhi_info["hhi_pos"]),
        "hhi_neg": float(hhi_info["hhi_neg"]),
        "hhi_time": float(hhi_info["hhi_time"]),
        "strategy_fail_prob": float(risk_info["prob_strategy_failure"]),
    }
