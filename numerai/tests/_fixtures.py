"""Shared deterministic fixtures and synthetic Numerai Unity datasets for unit tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def make_synthetic_numerai_panel(
    *,
    n_eras: int = 12,
    n_tickers: int = 12,
    n_features: int = 10,
    start_era: int = 1,
    seed: int = 42,
    signal_strength: float = 0.65,
    data_type: str = "train",
    include_date: bool = True,
    include_id: bool = True,
    include_ticker: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a small deterministic Numerai v6.0 Unity-like panel DataFrame.

    Returns `(df, feature_cols)` where features are `int8` in `0..4`, `era` is `int32`,
    and `target` is `float32` in `{0.0, 0.25, 0.5, 0.75, 1.0}` with controllable signal
    from `feature_alpha_0`.
    """
    rng = np.random.default_rng(seed)
    n_rows = n_eras * n_tickers

    eras = np.repeat(np.arange(start_era, start_era + n_eras, dtype=np.int32), n_tickers)
    tickers = np.tile(np.array([f"TK_{i:02d}" for i in range(n_tickers)]), n_eras)

    data: Dict[str, object] = {"era": eras}
    if include_id:
        data["id"] = [f"{data_type}_{idx:05d}" for idx in range(n_rows)]
    if include_date:
        base_date = pd.Timestamp("2020-01-03")
        era_dates = [base_date + pd.Timedelta(days=7 * int(e - 1)) for e in eras]
        data["date"] = pd.Series(era_dates).dt.strftime("%Y-%m-%d").to_numpy()
    if include_ticker:
        data["numerai_ticker"] = tickers
    data["data_type"] = np.full(n_rows, data_type, dtype=object)

    feature_cols: List[str] = []
    half = max(1, n_features // 2)
    for idx in range(n_features):
        group = "alpha" if idx < half else "beta"
        col = f"feature_{group}_{idx}"
        data[col] = rng.integers(0, 5, size=n_rows, dtype=np.int8)
        feature_cols.append(col)

    # Construct a target with controllable linear correlation to feature_alpha_0
    f0 = data[feature_cols[0]].astype(np.float64)
    latent = signal_strength * ((f0 - 2.0) / 2.0) + (1.0 - signal_strength) * rng.normal(0.0, 0.8, size=n_rows)
    bins = np.clip(np.rint((latent + 1.0) * 2.0), 0, 4)
    data["target"] = (bins * 0.25).astype(np.float32)

    df = pd.DataFrame(data)
    return df, feature_cols


def write_synthetic_unity_dataset(
    root_dir: Path,
    *,
    n_train_eras: int = 12,
    n_val_eras: int = 6,
    n_tickers: int = 10,
    n_features: int = 8,
    use_int8_filenames: bool = False,
    date_only: bool = False,
    float_unit_features: bool = False,
    include_non_validation_rows: bool = False,
    include_nan_targets: bool = False,
    seed: int = 42,
) -> Tuple[Path, List[str]]:
    """Write a minimal complete Numerai Unity dataset (`features.json`, train & val Parquets) into `root_dir`."""
    root_dir.mkdir(parents=True, exist_ok=True)
    train_df, feature_cols = make_synthetic_numerai_panel(
        n_eras=n_train_eras,
        n_tickers=n_tickers,
        n_features=n_features,
        start_era=1,
        seed=seed,
        data_type="train",
        include_date=True,
        include_id=True,
        include_ticker=True,
    )
    val_df, _ = make_synthetic_numerai_panel(
        n_eras=n_val_eras,
        n_tickers=n_tickers,
        n_features=n_features,
        start_era=n_train_eras + 1,
        seed=seed + 100,
        data_type="validation",
        include_date=True,
        include_id=True,
        include_ticker=True,
    )

    if include_non_validation_rows:
        live_rows = val_df.head(n_tickers).copy()
        live_rows["data_type"] = "live"
        live_rows["id"] = [f"live_{i:05d}" for i in range(len(live_rows))]
        val_df = pd.concat([val_df, live_rows], ignore_index=True)

    if include_nan_targets:
        train_df.loc[train_df.index[:2], "target"] = np.nan
        val_df.loc[val_df.index[:2], "target"] = np.nan

    if float_unit_features:
        for col in feature_cols:
            train_df[col] = (train_df[col].astype(np.float32) / 4.0).astype(np.float32)
            val_df[col] = (val_df[col].astype(np.float32) / 4.0).astype(np.float32)
        train_df.loc[train_df.index[0], feature_cols[0]] = np.nan

    if date_only:
        train_df = train_df.drop(columns=["era"])
        val_df = val_df.drop(columns=["era"])

    # Include a forbidden column inside features.json to verify loader strips it
    feature_metadata = {
        "feature_sets": {
            "small": feature_cols + ["era", "target", "missing_col"],
            "medium": feature_cols,
        },
        "targets": ["target"],
    }
    (root_dir / "features.json").write_text(json.dumps(feature_metadata), encoding="utf-8")

    train_name = "train_int8.parquet" if use_int8_filenames else "train.parquet"
    val_name = "validation_int8.parquet" if use_int8_filenames else "validation.parquet"
    train_df.to_parquet(root_dir / train_name, index=False)
    val_df.to_parquet(root_dir / val_name, index=False)
    return root_dir, feature_cols
