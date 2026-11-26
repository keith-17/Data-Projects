"""Unit tests for data loading, per-ticker time-series features, downsampling, and caching in src/data_utils.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from _fixtures import make_synthetic_numerai_panel, write_synthetic_unity_dataset
from afml_utils import build_era_cv_splits
from baseline_utils import PersistenceEstimator, run_baseline
import data_utils
from data_utils import (
    FORBIDDEN_FEATURE_COLS,
    POLARS_AVAILABLE,
    _compute_weeks_since_change_1d,
    _compute_weeks_since_change_grouped,
    _compute_weeks_since_high_1d,
    _compute_weeks_since_high_grouped,
    _extract_http_status_code,
    _find_dataset_root,
    _format_http_error,
    _has_complete_dataset,
    _resolve_numerai_paths,
    compute_per_ticker_timeseries_features,
    downsample_training_data,
    load_unity_numerai_data,
    load_unity_numerai_data_cached,
)


class _MockResponseStatusError(Exception):
    """Test error carrying an HTTP `response.status_code` attribute."""

    def __init__(self, code: int):
        super().__init__(f"boom {code}")
        self.response = SimpleNamespace(status_code=code)


class _MockAttributeStatusError(Exception):
    """Test error carrying a top-level HTTP `status` attribute."""

    def __init__(self, code: int):
        super().__init__(f"status {code}")
        self.status = code


def _copy_staged_numerapi_file(staged_dir: Path, _remote_path: str, dest_path: str) -> None:
    """Mock `NumerAPI.download_dataset` side effect copying a staged synthetic file into `dest_path`."""
    src_name = Path(dest_path).name
    Path(dest_path).write_bytes((staged_dir / src_name).read_bytes())


def _write_100_row_multi_feature_set_dataset(root_dir: Path, *, seed: int = 42) -> Tuple[Path, dict]:
    """Create a 100-row synthetic Unity Parquet fixture (60 train + 40 val) defining 'small', 'medium', and 'all'."""
    root_dir.mkdir(parents=True, exist_ok=True)
    train_df, all_feature_cols = make_synthetic_numerai_panel(
        n_eras=6,
        n_tickers=10,
        n_features=12,
        start_era=1,
        seed=seed,
        data_type="train",
        include_date=True,
        include_id=True,
        include_ticker=True,
    )
    val_df, _ = make_synthetic_numerai_panel(
        n_eras=4,
        n_tickers=10,
        n_features=12,
        start_era=7,
        seed=seed + 100,
        data_type="validation",
        include_date=True,
        include_id=True,
        include_ticker=True,
    )
    assert len(train_df) + len(val_df) == 100

    feature_sets = {
        "small": all_feature_cols[:4],
        "medium": all_feature_cols[:8],
        "all": all_feature_cols,
    }
    feature_metadata = {
        "feature_sets": feature_sets,
        "targets": ["target"],
    }
    (root_dir / "features.json").write_text(json.dumps(feature_metadata), encoding="utf-8")
    train_df.to_parquet(root_dir / "train.parquet", index=False)
    val_df.to_parquet(root_dir / "validation.parquet", index=False)
    return root_dir, feature_sets


def _scan_parquet_reject_parallel(orig_scan: object, path: object, **kwargs: object) -> object:
    """Simulate an older Polars version where `scan_parquet(..., parallel='auto')` raises TypeError."""
    if "parallel" in kwargs:
        raise TypeError("unexpected keyword argument 'parallel'")
    return orig_scan(path, **kwargs)


def _scan_parquet_record_parallel(
    orig_scan: object,
    recorded_modes: list,
    path: object,
    **kwargs: object,
) -> object:
    """Record every `parallel` kwarg passed to `pl.scan_parquet` and delegate to `orig_scan`."""
    recorded_modes.append(kwargs.get("parallel"))
    return orig_scan(path, **kwargs)


def _write_custom_unity_dataset(
    root_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
) -> Path:
    """Persist custom train/val DataFrames and `features.json` into `root_dir` for parity tests."""
    root_dir.mkdir(parents=True, exist_ok=True)
    feature_metadata = {"feature_sets": {"small": list(feature_cols), "all": list(feature_cols)}, "targets": ["target"]}
    (root_dir / "features.json").write_text(json.dumps(feature_metadata), encoding="utf-8")
    train_df.to_parquet(root_dir / "train.parquet", index=False)
    val_df.to_parquet(root_dir / "validation.parquet", index=False)
    return root_dir


class TestHttpAndPathHelpers(unittest.TestCase):
    """Unit tests for HTTP error formatting and local dataset path discovery."""

    def test_extract_http_status_code_and_format_error(self) -> None:
        self.assertEqual(_extract_http_status_code(_MockResponseStatusError(401)), 401)
        self.assertEqual(_extract_http_status_code(_MockAttributeStatusError(429)), 429)
        self.assertIsNone(_extract_http_status_code(RuntimeError("plain")))

        msg_401 = _format_http_error("Kaggle failed", _MockResponseStatusError(401))
        self.assertIn("401 Unauthorized", msg_401)

        msg_418 = _format_http_error("Kaggle failed", _MockResponseStatusError(418))
        self.assertIn("HTTP 418", msg_418)

        msg_plain = _format_http_error("Kaggle failed", RuntimeError("disk full"))
        self.assertEqual(msg_plain, "Kaggle failed: disk full")

    def test_resolve_paths_prefers_int8_and_find_dataset_root_nested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(_has_complete_dataset(root / "does_not_exist"))
            self.assertIsNone(_find_dataset_root(root / "does_not_exist"))

            # Standard parquet files in nested subdir
            nested = root / "outer" / "v6.0"
            write_synthetic_unity_dataset(nested, n_train_eras=4, n_val_eras=2, n_tickers=4, n_features=4)
            self.assertTrue(_has_complete_dataset(nested))
            self.assertEqual(_find_dataset_root(root), nested)

            tr_p, val_p, feat_p = _resolve_numerai_paths(nested)
            self.assertEqual(tr_p.name, "train.parquet")
            self.assertEqual(val_p.name, "validation.parquet")
            self.assertEqual(feat_p.name, "features.json")

            # Add int8 files -> resolver must prefer them
            (nested / "train_int8.parquet").write_bytes((nested / "train.parquet").read_bytes())
            (nested / "validation_int8.parquet").write_bytes((nested / "validation.parquet").read_bytes())
            tr_i8, val_i8, _ = _resolve_numerai_paths(nested)
            self.assertEqual(tr_i8.name, "train_int8.parquet")
            self.assertEqual(val_i8.name, "validation_int8.parquet")


class TestCausalTickerTimeSeriesFeatures(unittest.TestCase):
    """Unit tests for 1D causal helpers and compute_per_ticker_timeseries_features."""

    def test_weeks_since_high_1d(self) -> None:
        self.assertEqual(len(_compute_weeks_since_high_1d(np.array([]), window=4)), 0)
        # Strictly increasing -> always at rolling high -> all zeros
        np.testing.assert_array_equal(
            _compute_weeks_since_high_1d(np.array([1.0, 2.0, 3.0, 4.0]), window=4),
            np.array([0.0, 0.0, 0.0, 0.0]),
        )
        # Peak at index 1 (value 10), then declining within window=4
        np.testing.assert_array_equal(
            _compute_weeks_since_high_1d(np.array([2.0, 10.0, 5.0, 3.0, 1.0]), window=4),
            np.array([0.0, 0.0, 1.0, 2.0, 3.0]),
        )

    def test_weeks_since_change_1d(self) -> None:
        self.assertEqual(len(_compute_weeks_since_change_1d(np.array([]))), 0)
        np.testing.assert_array_equal(
            _compute_weeks_since_change_1d(np.array([2.0, 2.0, 2.0, 4.0, 4.0, 1.0])),
            np.array([0.0, 1.0, 2.0, 0.0, 1.0, 0.0]),
        )

    def test_grouped_weeks_helpers_match_1d_reference(self) -> None:
        g1 = np.array([2.0, 10.0, 5.0, 3.0, 1.0])
        g2 = np.array([4.0, 4.0, 4.0, 1.0])
        combined = np.concatenate([g1, g2])
        pos = np.concatenate([np.arange(len(g1)), np.arange(len(g2))])

        expected_wsh = np.concatenate([_compute_weeks_since_high_1d(g1, 4), _compute_weeks_since_high_1d(g2, 4)])
        np.testing.assert_array_equal(_compute_weeks_since_high_grouped(combined, pos, 4), expected_wsh)

        expected_wsc = np.concatenate([_compute_weeks_since_change_1d(g1), _compute_weeks_since_change_1d(g2)])
        np.testing.assert_array_equal(_compute_weeks_since_change_grouped(combined, pos), expected_wsc)

    def test_polars_vs_pandas_timeseries_parity(self) -> None:
        if not POLARS_AVAILABLE:
            self.skipTest("Polars not installed")
        df, feature_cols = make_synthetic_numerai_panel(n_eras=10, n_tickers=8, n_features=4, seed=71)
        shuffled = df.sample(frac=1.0, random_state=71)

        out_pl, cols_pl = compute_per_ticker_timeseries_features(
            shuffled,
            feature_cols,
            ticker_features="all",
            ticker_windows=(2, 4),
            ticker_lag_steps=(1, 2),
            top_n_cols=2,
            quantize_scope="era",
        )
        with patch.object(data_utils, "POLARS_AVAILABLE", False):
            out_pd, cols_pd = compute_per_ticker_timeseries_features(
                shuffled,
                feature_cols,
                ticker_features="all",
                ticker_windows=(2, 4),
                ticker_lag_steps=(1, 2),
                top_n_cols=2,
                quantize_scope="era",
            )

        self.assertEqual(cols_pl, cols_pd)
        self.assertEqual(list(out_pl.index), list(out_pd.index))
        for col in cols_pl:
            self.assertEqual(out_pl[col].dtype, np.int8)
            diff = np.abs(out_pl[col].to_numpy(dtype=np.int16) - out_pd[col].to_numpy(dtype=np.int16))
            # Half-away-from-zero vs half-to-even rounding can differ by at most 1 bin on exact .5 ties
            self.assertLessEqual(int(diff.max()), 1, f"Column {col} diverged beyond tie-rounding tolerance")
            self.assertGreaterEqual(float(np.mean(diff == 0)), 0.90, f"Column {col} agreement too low")

    def test_compute_per_ticker_timeseries_features_all_modes_and_dtypes(self) -> None:
        df, feature_cols = make_synthetic_numerai_panel(n_eras=8, n_tickers=6, n_features=4, seed=17)
        # Shuffle rows to verify original index/order is preserved
        shuffled = df.sample(frac=1.0, random_state=17)

        out_df, added_cols = compute_per_ticker_timeseries_features(
            shuffled,
            feature_cols,
            ticker_features="all",
            ticker_windows=(2, 4),
            ticker_lag_steps=(1, 2),
            top_n_cols=2,
            quantize_scope="era",
        )
        self.assertEqual(list(out_df.index), list(shuffled.index))
        self.assertGreater(len(added_cols), 0)
        for col in added_cols:
            self.assertEqual(out_df[col].dtype, np.int8)
            vals = out_df[col].to_numpy()
            self.assertTrue(np.all((vals >= 0) & (vals <= 4)))

        # Disabled or missing columns returns original df and empty list
        no_df, no_cols = compute_per_ticker_timeseries_features(df, feature_cols, ticker_features=None)
        self.assertIs(no_df, df)
        self.assertEqual(no_cols, [])

        no_df2, no_cols2 = compute_per_ticker_timeseries_features(df.drop(columns=["numerai_ticker"]), feature_cols)
        self.assertEqual(no_cols2, [])

    def test_compute_per_ticker_timeseries_features_strictly_causal(self) -> None:
        """Perturbing future eras (> cutoff) must NOT alter features at eras <= cutoff."""
        df, feature_cols = make_synthetic_numerai_panel(n_eras=10, n_tickers=8, n_features=4, seed=29)
        cutoff_era = 5

        out_orig, added_cols = compute_per_ticker_timeseries_features(
            df,
            feature_cols,
            ticker_features="all",
            ticker_windows=(2, 4),
            ticker_lag_steps=(1, 2),
            top_n_cols=2,
            quantize_scope="era",
        )

        df_perturbed = df.copy()
        future_mask = df_perturbed["era"] > cutoff_era
        for col in feature_cols:
            df_perturbed.loc[future_mask, col] = np.int8(4) - df_perturbed.loc[future_mask, col]

        out_pert, _ = compute_per_ticker_timeseries_features(
            df_perturbed,
            feature_cols,
            ticker_features="all",
            ticker_windows=(2, 4),
            ticker_lag_steps=(1, 2),
            top_n_cols=2,
            quantize_scope="era",
        )

        past_mask = df["era"] <= cutoff_era
        for col in added_cols:
            np.testing.assert_array_equal(
                out_orig.loc[past_mask, col].to_numpy(),
                out_pert.loc[past_mask, col].to_numpy(),
                err_msg=f"Future leakage detected in {col}",
            )


class TestDownsampleTrainingData(unittest.TestCase):
    """Unit tests for downsample_training_data across all modes and edge cases."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.train_df, _ = make_synthetic_numerai_panel(n_eras=12, n_tickers=10, n_features=4, seed=55)

    def test_noop_modes_return_original(self) -> None:
        self.assertIs(downsample_training_data(self.train_df, downsample_mode=None, downsample_frac=0.5), self.train_df)
        self.assertIs(downsample_training_data(self.train_df, downsample_mode="recent", downsample_frac=1.0), self.train_df)
        empty = self.train_df.head(0)
        self.assertIs(downsample_training_data(empty, downsample_mode="recent", downsample_frac=0.5), empty)

    def test_removed_row_and_random_modes_raise_value_error(self) -> None:
        for removed_mode in ("within_era", "random_eras"):
            with self.subTest(removed_mode=removed_mode):
                with self.assertRaises(ValueError) as ctx:
                    downsample_training_data(self.train_df, downsample_mode=removed_mode, downsample_frac=0.5)
                self.assertIn("whole-era", str(ctx.exception).lower())
                self.assertIn("numerai_corr", str(ctx.exception))
                # Also rejected even when frac=1.0
                with self.assertRaises(ValueError):
                    downsample_training_data(self.train_df, downsample_mode=removed_mode, downsample_frac=1.0)

    def test_recent_earliest_and_uniform_whole_era_modes(self) -> None:
        ds_recent = downsample_training_data(self.train_df, downsample_mode="recent", downsample_frac=0.5)
        np.testing.assert_array_equal(sorted(ds_recent["era"].unique()), [7, 8, 9, 10, 11, 12])

        ds_earliest = downsample_training_data(self.train_df, downsample_mode="earliest", downsample_frac=0.5)
        np.testing.assert_array_equal(sorted(ds_earliest["era"].unique()), [1, 2, 3, 4, 5, 6])

        ds_uniform = downsample_training_data(self.train_df, downsample_mode="uniform", downsample_frac=0.5)
        self.assertEqual(len(ds_uniform["era"].unique()), 6)
        self.assertEqual(int(ds_uniform["era"].min()), 1)
        self.assertEqual(int(ds_uniform["era"].max()), 12)
        ds_uniform_repeat = downsample_training_data(
            self.train_df, downsample_mode="uniform", downsample_frac=0.5, random_state=99
        )
        pd.testing.assert_frame_equal(ds_uniform, ds_uniform_repeat)

    def test_tiny_downsample_frac_warns_and_clamps_to_at_least_two_eras(self) -> None:
        for mode, expected_eras in (("recent", [11, 12]), ("earliest", [1, 2]), ("uniform", [1, 12])):
            with self.subTest(mode=mode):
                with self.assertWarns(UserWarning) as warn_ctx:
                    ds = downsample_training_data(self.train_df, downsample_mode=mode, downsample_frac=0.01)
                self.assertIn("clamping to 2 eras", str(warn_ctx.warning))
                np.testing.assert_array_equal(sorted(ds["era"].unique()), expected_eras)

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            downsample_training_data(self.train_df, downsample_mode="bad_mode", downsample_frac=0.5)
        with self.assertRaises(ValueError):
            downsample_training_data(self.train_df, downsample_mode="recent", downsample_frac=0.0)
        with self.assertRaises(ValueError):
            downsample_training_data(self.train_df, downsample_mode="recent", downsample_frac=1.5)
        with self.assertRaises(KeyError):
            downsample_training_data(self.train_df, downsample_mode="recent", downsample_frac=0.5, era_col="missing")


class TestLoadUnityNumeraiDataAndCache(unittest.TestCase):
    """Offline unit tests for load_unity_numerai_data and load_unity_numerai_data_cached."""

    def test_load_local_dataset_cleans_casts_strides_and_preserves_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            write_synthetic_unity_dataset(
                ws / "data_folder" / "v6.0",
                n_train_eras=12,
                n_val_eras=6,
                n_tickers=8,
                n_features=6,
                float_unit_features=True,
                include_non_validation_rows=True,
                include_nan_targets=True,
            )

            train_df, val_df, feature_cols, metadata = load_unity_numerai_data(
                data_version="v6.0",
                feature_set="small",
                era_stride=2,
                downsample_mode="recent",
                downsample_frac=0.5,
                workspace_root=ws,
                auto_download=False,
            )

            # Forbidden & missing columns stripped
            self.assertEqual(len(feature_cols), 6)
            for col in feature_cols:
                self.assertNotIn(col, FORBIDDEN_FEATURE_COLS)
                self.assertFalse(col.startswith("target"))
                self.assertEqual(train_df[col].dtype, np.int8)
                self.assertEqual(val_df[col].dtype, np.int8)

            # Targets cleaned and float32
            self.assertEqual(train_df["target"].dtype, np.float32)
            self.assertEqual(val_df["target"].dtype, np.float32)
            self.assertFalse(train_df["target"].isna().any())
            self.assertFalse(val_df["target"].isna().any())

            # Train strided (12 -> 6) then downsampled 50% (6 -> 3 eras); validation untouched (all 6 eras, only data_type='validation')
            self.assertEqual(len(train_df["era"].unique()), 3)
            self.assertEqual(len(val_df["era"].unique()), 6)
            self.assertTrue((val_df["data_type"] == "validation").all())
            self.assertIn("feature_sets", metadata)

    def test_load_date_only_schema_and_pre_stride_ticker_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            write_synthetic_unity_dataset(
                ws / "data_folder" / "v6.0",
                n_train_eras=8,
                n_val_eras=4,
                n_tickers=6,
                n_features=4,
                date_only=True,
            )

            train_df, val_df, feature_cols, _ = load_unity_numerai_data(
                data_version="v6.0",
                feature_set="small",
                era_stride=2,
                ticker_timeseries_features="rolling_mean|raw_lags",
                ticker_rolling_windows=(2,),
                ticker_lag_steps=(1,),
                ticker_top_n_features=2,
                workspace_root=ws,
                auto_download=False,
            )

            self.assertEqual(train_df["era"].dtype, np.int32)
            self.assertEqual(val_df["era"].dtype, np.int32)
            self.assertGreater(int(val_df["era"].min()), int(train_df["era"].max()))
            self.assertTrue(any("_ticker_roll_mean2" in c for c in feature_cols))
            self.assertTrue(any("_ticker_lag1" in c for c in feature_cols))

    def test_load_errors_unknown_feature_set_and_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                load_unity_numerai_data(workspace_root=ws, auto_download=False)

            write_synthetic_unity_dataset(ws / "data_folder" / "v6.0", n_train_eras=4, n_val_eras=2, n_tickers=4, n_features=4)
            with self.assertRaises(KeyError):
                load_unity_numerai_data(feature_set="nonexistent_set", workspace_root=ws, auto_download=False)

    def test_mocked_download_fallback_chain_kagglehub_to_numerapi(self) -> None:
        """Verify fallback from kagglehub -> KaggleApi -> NumerAPI without real network calls."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            staged_dir = ws / "staged_remote"
            write_synthetic_unity_dataset(staged_dir, n_train_eras=4, n_val_eras=2, n_tickers=4, n_features=4)

            mock_kagglehub = MagicMock()
            mock_kagglehub.dataset_download.side_effect = OSError("kagglehub offline")

            mock_numerapi_instance = MagicMock()
            mock_numerapi_instance.download_dataset.side_effect = partial(_copy_staged_numerapi_file, staged_dir)
            mock_numerapi_cls = MagicMock(return_value=mock_numerapi_instance)

            with (
                patch.object(data_utils, "kagglehub", mock_kagglehub),
                patch.object(data_utils, "KaggleApi", None),
                patch.object(data_utils, "NumerAPI", mock_numerapi_cls),
                self.assertWarns(UserWarning),
            ):
                train_df, val_df, feat_cols, _ = load_unity_numerai_data(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=1,
                    workspace_root=ws,
                    auto_download=True,
                    use_numerapi_fallback=True,
                )

            self.assertEqual(mock_numerapi_instance.download_dataset.call_count, 3)
            self.assertEqual(len(feat_cols), 4)
            self.assertEqual(len(train_df["era"].unique()), 4)
            self.assertEqual(len(val_df["era"].unique()), 2)

    def test_cached_loader_writes_lightweight_downsampled_parquets_and_hits_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            write_synthetic_unity_dataset(ws / "data_folder" / "v6.0", n_train_eras=8, n_val_eras=4, n_tickers=6, n_features=4)

            # First call with whole-era downsampling populates processed_cache with already-downsampled Parquets
            tr1, val1, cols1, _ = load_unity_numerai_data_cached(
                data_version="v6.0",
                feature_set="small",
                era_stride=1,
                downsample_mode="recent",
                downsample_frac=0.5,
                val_downsample_mode="recent",
                val_downsample_frac=0.5,
                random_state=42,
                workspace_root=ws,
                auto_download=False,
            )
            cache_dir = ws / "data_folder" / "processed_cache"
            self.assertTrue(cache_dir.is_dir())
            cached_train_files = list(cache_dir.glob("train_*.parquet"))
            cached_val_files = list(cache_dir.glob("validation_*.parquet"))
            self.assertEqual(len(cached_train_files), 1)
            self.assertEqual(len(cached_val_files), 1)
            # Disk cache must store ONLY the already-downsampled whole eras (4*6=24 train, 2*6=12 val), NOT full 48/24
            self.assertEqual(len(pd.read_parquet(cached_train_files[0])), len(tr1))
            self.assertEqual(len(pd.read_parquet(cached_val_files[0])), len(val1))
            self.assertEqual(len(tr1), 4 * 6)
            self.assertEqual(len(val1), 2 * 6)

            # Repeat call with identical knobs MUST hit cache (never invoke uncached loader)
            with patch.object(data_utils, "load_unity_numerai_data", side_effect=AssertionError("Should have hit cache!")):
                tr_hit, val_hit, cols_hit, _ = load_unity_numerai_data_cached(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=1,
                    downsample_mode="recent",
                    downsample_frac=0.5,
                    val_downsample_mode="recent",
                    val_downsample_frac=0.5,
                    random_state=42,
                    workspace_root=ws,
                    auto_download=False,
                )
            self.assertEqual(cols1, cols_hit)
            pd.testing.assert_frame_equal(tr1.reset_index(drop=True), tr_hit.reset_index(drop=True))
            pd.testing.assert_frame_equal(val1.reset_index(drop=True), val_hit.reset_index(drop=True))

            # Equivalent `val_downsample_mode='same'` resolves to the identical cache key -> MUST also hit cache
            with patch.object(data_utils, "load_unity_numerai_data", side_effect=AssertionError("Should have hit cache!")):
                tr_same, val_same, _, _ = load_unity_numerai_data_cached(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=1,
                    downsample_mode="recent",
                    downsample_frac=0.5,
                    val_downsample_mode="same",
                    val_downsample_frac=None,
                    random_state=42,
                    workspace_root=ws,
                    auto_download=False,
                )
            pd.testing.assert_frame_equal(tr1.reset_index(drop=True), tr_same.reset_index(drop=True))
            pd.testing.assert_frame_equal(val1.reset_index(drop=True), val_same.reset_index(drop=True))

            # Different downsampling config creates a separate lightweight cache entry
            tr_earliest, val_earliest, _, _ = load_unity_numerai_data_cached(
                data_version="v6.0",
                feature_set="small",
                era_stride=1,
                downsample_mode="earliest",
                downsample_frac=0.5,
                val_downsample_mode="uniform",
                val_downsample_frac=0.5,
                random_state=42,
                workspace_root=ws,
                auto_download=False,
            )
            self.assertEqual(len(tr_earliest["era"].unique()), 4)
            self.assertEqual(len(val_earliest["era"].unique()), 2)
            self.assertEqual(len(list(cache_dir.glob("train_*.parquet"))), 2)

            # force_reload=True bypasses cache
            with patch.object(data_utils, "load_unity_numerai_data", wraps=load_unity_numerai_data) as spy:
                load_unity_numerai_data_cached(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=1,
                    downsample_mode="recent",
                    downsample_frac=0.5,
                    val_downsample_mode="recent",
                    val_downsample_frac=0.5,
                    random_state=42,
                    workspace_root=ws,
                    auto_download=False,
                    force_reload=True,
                )
                self.assertEqual(spy.call_count, 1)

    def test_declarative_polars_100_row_feature_sets_cache_and_persistence_baseline(self) -> None:
        """Fast 100-row verification across 'small', 'medium', and 'all' feature sets: Polars vs Pandas parity, cache hits, and Persistence baseline."""
        if not POLARS_AVAILABLE:
            self.skipTest("Polars not installed")
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            _, expected_feature_sets = _write_100_row_multi_feature_set_dataset(ws / "data_folder" / "v6.0", seed=42)
            cache_dir = ws / "data_folder" / "processed_cache"

            last_split: Optional[tuple] = None
            for fset_name, expected_cols in expected_feature_sets.items():
                with self.subTest(feature_set=fset_name):
                    # 1. Cache miss via declarative whole-era Polars loader
                    tr_pl, va_pl, cols_pl, _ = load_unity_numerai_data_cached(
                        data_version="v6.0",
                        feature_set=fset_name,
                        era_stride=2,
                        downsample_mode="recent",
                        downsample_frac=0.67,
                        val_downsample_mode="earliest",
                        val_downsample_frac=0.5,
                        random_state=42,
                        workspace_root=ws,
                        auto_download=False,
                    )
                    self.assertEqual(cols_pl, expected_cols)
                    # 6 train eras strided by 2 -> [1, 3, 5] -> recent 67% keeps 2 eras x 10 rows = 20 rows
                    # 4 val eras -> earliest 50% keeps 2 eras x 10 rows = 20 rows
                    self.assertEqual(len(tr_pl), 20)
                    self.assertEqual(len(va_pl), 20)

                    # 2. Cache hit must return identical DataFrames without calling uncached loader
                    with patch.object(
                        data_utils, "load_unity_numerai_data", side_effect=AssertionError("Expected cache hit!")
                    ):
                        tr_hit, va_hit, cols_hit, _ = load_unity_numerai_data_cached(
                            data_version="v6.0",
                            feature_set=fset_name,
                            era_stride=2,
                            downsample_mode="recent",
                            downsample_frac=0.67,
                            val_downsample_mode="earliest",
                            val_downsample_frac=0.5,
                            random_state=42,
                            workspace_root=ws,
                            auto_download=False,
                        )
                    self.assertEqual(cols_hit, cols_pl)
                    pd.testing.assert_frame_equal(tr_pl.reset_index(drop=True), tr_hit.reset_index(drop=True))
                    pd.testing.assert_frame_equal(va_pl.reset_index(drop=True), va_hit.reset_index(drop=True))

                    # 3. Pandas fallback parity: exact same rows, column sets, values, and dtypes
                    with patch.object(data_utils, "POLARS_AVAILABLE", False):
                        tr_pd, va_pd, cols_pd, _ = load_unity_numerai_data(
                            data_version="v6.0",
                            feature_set=fset_name,
                            era_stride=2,
                            downsample_mode="recent",
                            downsample_frac=0.67,
                            val_downsample_mode="earliest",
                            val_downsample_frac=0.5,
                            random_state=42,
                            workspace_root=ws,
                            auto_download=False,
                        )
                    self.assertEqual(cols_pl, cols_pd)
                    self.assertEqual(list(tr_pl.columns), list(tr_pd.columns))
                    self.assertEqual(list(va_pl.columns), list(va_pd.columns))
                    self.assertEqual(tr_pl["era"].dtype, np.int32)
                    self.assertEqual(va_pl["era"].dtype, np.int32)
                    self.assertEqual(tr_pl["target"].dtype, np.float32)
                    self.assertEqual(va_pl["target"].dtype, np.float32)
                    for col in cols_pl:
                        self.assertEqual(tr_pl[col].dtype, np.int8)
                        self.assertEqual(va_pl[col].dtype, np.int8)
                    pd.testing.assert_frame_equal(tr_pl.reset_index(drop=True), tr_pd.reset_index(drop=True))
                    pd.testing.assert_frame_equal(va_pl.reset_index(drop=True), va_pd.reset_index(drop=True))
                    last_split = (tr_pl, va_pl, cols_pl)

            # All 3 feature sets stored lightweight downsampled Parquets on disk
            self.assertEqual(len(list(cache_dir.glob("train_*.parquet"))), 3)
            for p_file in cache_dir.glob("train_*.parquet"):
                self.assertEqual(len(pd.read_parquet(p_file)), 20)
            for p_file in cache_dir.glob("validation_*.parquet"):
                self.assertEqual(len(pd.read_parquet(p_file)), 20)

            # 4. Persistence baseline smoke test on the 100-row loaded split
            assert last_split is not None
            tr_pers, va_pers, _ = last_split
            persist_cols = ["era", "numerai_ticker"]
            pipe = Pipeline([("estimator", PersistenceEstimator(persistence_mode="last", persistence_lag=1, target_horizon_eras=1))])
            grid = {"estimator__persistence_mode": ["last", "expanding_mean"], "estimator__smoothing": [0.0]}
            b_name, b_entries, b_cv_df, b_summary, _ = run_baseline(
                "Baseline P (Persistence)",
                pipe,
                grid,
                tr_pers[persist_cols],
                tr_pers["target"],
                tr_pers["era"],
                va_pers[persist_cols],
                va_pers["target"],
                va_pers["era"],
                [],
                search_mode="grid",
                score_version="v1_skepta",
                eval_mode="static",
                cv_object=TimeSeriesSplit(n_splits=2),
                verbose=0,
                n_jobs=1,
            )
            self.assertEqual(b_name, "Baseline P (Persistence)")
            self.assertIn("Baseline P (Persistence) [Default]", b_entries)
            self.assertIn("Baseline P (Persistence) [Tuned]", b_entries)
            self.assertEqual(len(b_entries["Baseline P (Persistence) [Tuned]"]["validation_preds"]), len(va_pers))
            self.assertGreaterEqual(len(b_cv_df), 2)
            self.assertEqual(b_summary["score_version"], "v1_skepta")

    def test_declarative_polars_parallel_fallback_no_prefiltered_no_row_idx_and_whole_era_modes(self) -> None:
        if not POLARS_AVAILABLE:
            self.skipTest("Polars not installed")
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            _write_100_row_multi_feature_set_dataset(ws / "data_folder" / "v6.0", seed=77)

            # 1. Fallback when installed Polars rejects `parallel='auto'`
            orig_scan = data_utils.pl.scan_parquet
            with patch.object(data_utils.pl, "scan_parquet", side_effect=partial(_scan_parquet_reject_parallel, orig_scan)):
                tr_fb, va_fb, cols_fb, _ = load_unity_numerai_data(
                    data_version="v6.0",
                    feature_set="medium",
                    era_stride=2,
                    downsample_mode="recent",
                    downsample_frac=0.67,
                    val_downsample_mode="earliest",
                    val_downsample_frac=0.5,
                    random_state=42,
                    workspace_root=ws,
                    auto_download=False,
                )
            self.assertEqual(len(cols_fb), 8)
            self.assertEqual(len(tr_fb), 20)
            self.assertEqual(len(va_fb), 20)

            # 2. Verify `parallel='prefiltered'` and `LazyFrame.with_row_index` are NEVER used
            recorded_modes: list = []
            with (
                patch.object(
                    data_utils.pl,
                    "scan_parquet",
                    side_effect=partial(_scan_parquet_record_parallel, orig_scan, recorded_modes),
                ),
                patch.object(
                    data_utils.pl.LazyFrame,
                    "with_row_index",
                    side_effect=AssertionError("with_row_index must not be called by declarative Polars loader"),
                ),
            ):
                load_unity_numerai_data(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=2,
                    downsample_mode="recent",
                    downsample_frac=0.67,
                    val_downsample_mode="earliest",
                    val_downsample_frac=0.5,
                    workspace_root=ws,
                    auto_download=False,
                )
            self.assertTrue(len(recorded_modes) >= 4)
            self.assertNotIn("prefiltered", recorded_modes)
            self.assertTrue(all(mode == "auto" for mode in recorded_modes))

            # 3. Removed modes raise ValueError immediately in load_unity_numerai_data
            for bad_mode in ("within_era", "random_eras"):
                with self.subTest(bad_train_mode=bad_mode):
                    with self.assertRaises(ValueError):
                        load_unity_numerai_data(
                            data_version="v6.0",
                            feature_set="small",
                            downsample_mode=bad_mode,
                            downsample_frac=0.5,
                            workspace_root=ws,
                            auto_download=False,
                        )
                with self.subTest(bad_val_mode=bad_mode):
                    with self.assertRaises(ValueError):
                        load_unity_numerai_data(
                            data_version="v6.0",
                            feature_set="small",
                            val_downsample_mode=bad_mode,
                            val_downsample_frac=0.5,
                            workspace_root=ws,
                            auto_download=False,
                        )

            # 4. Parity across all supported whole-era modes
            for ds_mode in (None, "recent", "earliest", "uniform"):
                with self.subTest(downsample_mode=ds_mode):
                    tr_pl, va_pl, _, _ = load_unity_numerai_data(
                        data_version="v6.0",
                        feature_set="small",
                        era_stride=1,
                        downsample_mode=ds_mode,
                        downsample_frac=0.5,
                        val_downsample_mode=ds_mode,
                        val_downsample_frac=0.5,
                        random_state=19,
                        workspace_root=ws,
                        auto_download=False,
                    )
                    with patch.object(data_utils, "POLARS_AVAILABLE", False):
                        tr_pd, va_pd, _, _ = load_unity_numerai_data(
                            data_version="v6.0",
                            feature_set="small",
                            era_stride=1,
                            downsample_mode=ds_mode,
                            downsample_frac=0.5,
                            val_downsample_mode=ds_mode,
                            val_downsample_frac=0.5,
                            random_state=19,
                            workspace_root=ws,
                            auto_download=False,
                        )
                    pd.testing.assert_frame_equal(tr_pl.reset_index(drop=True), tr_pd.reset_index(drop=True))
                    pd.testing.assert_frame_equal(va_pl.reset_index(drop=True), va_pd.reset_index(drop=True))

    def test_polars_vs_pandas_parity_across_date_only_int_era_unsorted_modes_and_empty(self) -> None:
        """Exhaustive Polars vs Pandas parity for Unity v6.0 ISO date-only, integer era, unsorted Parquets, all modes, and empty split."""
        if not POLARS_AVAILABLE:
            self.skipTest("Polars not installed")

        # 1. Unity v6.0-style date: pl.String Parquet (no 'era' column) + verify pd.to_datetime is NOT invoked on ISO strings
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            write_synthetic_unity_dataset(
                ws / "data_folder" / "v6.0",
                n_train_eras=24,
                n_val_eras=10,
                n_tickers=4,
                n_features=6,
                date_only=True,
                include_non_validation_rows=True,
                include_nan_targets=True,
                seed=123,
            )
            mode_combos = [
                (1, None, 1.0, None, None),
                (12, None, 1.0, None, None),
                (1, "recent", 0.5, "earliest", 0.4),
                (1, "earliest", 0.5, "recent", 0.4),
                (1, "uniform", 0.5, "uniform", 0.5),
                (4, "recent", 0.5, "earliest", 0.5),
            ]
            for stride, ds_m, ds_f, v_m, v_f in mode_combos:
                with self.subTest(schema="date_only", stride=stride, ds_m=ds_m, v_m=v_m):
                    with patch.object(
                        data_utils.pd,
                        "to_datetime",
                        side_effect=AssertionError("pd.to_datetime must not be called for ISO YYYY-MM-DD dates"),
                    ):
                        tr_pl, va_pl, cols_pl, _ = load_unity_numerai_data(
                            data_version="v6.0",
                            feature_set="small",
                            era_stride=stride,
                            downsample_mode=ds_m,
                            downsample_frac=ds_f,
                            val_downsample_mode=v_m,
                            val_downsample_frac=v_f,
                            workspace_root=ws,
                            auto_download=False,
                        )
                    with patch.object(data_utils, "POLARS_AVAILABLE", False):
                        tr_pd, va_pd, cols_pd, _ = load_unity_numerai_data(
                            data_version="v6.0",
                            feature_set="small",
                            era_stride=stride,
                            downsample_mode=ds_m,
                            downsample_frac=ds_f,
                            val_downsample_mode=v_m,
                            val_downsample_frac=v_f,
                            workspace_root=ws,
                            auto_download=False,
                        )
                    self.assertEqual(cols_pl, cols_pd)
                    self.assertEqual(tr_pl["era"].dtype, np.int32)
                    self.assertEqual(va_pl["era"].dtype, np.int32)
                    self.assertEqual(tr_pl["target"].dtype, np.float32)
                    self.assertEqual(va_pl["target"].dtype, np.float32)
                    for col in cols_pl:
                        self.assertEqual(tr_pl[col].dtype, np.int8)
                        self.assertEqual(va_pl[col].dtype, np.int8)
                    pd.testing.assert_frame_equal(tr_pl.reset_index(drop=True), tr_pd.reset_index(drop=True))
                    pd.testing.assert_frame_equal(va_pl.reset_index(drop=True), va_pd.reset_index(drop=True))

        # 2. Classic integer `era` Parquet (without `date`)
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            tr_raw, feat_cols = make_synthetic_numerai_panel(
                n_eras=24, n_tickers=4, n_features=5, start_era=1, seed=31, data_type="train", include_date=False
            )
            va_raw, _ = make_synthetic_numerai_panel(
                n_eras=8, n_tickers=4, n_features=5, start_era=25, seed=32, data_type="validation", include_date=False
            )
            _write_custom_unity_dataset(ws / "data_folder" / "v6.0", tr_raw, va_raw, feat_cols)
            for stride, ds_m, ds_f, v_m, v_f in [
                (12, None, 1.0, None, None),
                (4, "recent", 0.5, "earliest", 0.5),
                (1, "uniform", 0.5, "recent", 0.5),
            ]:
                with self.subTest(schema="int_era", stride=stride, ds_m=ds_m, v_m=v_m):
                    tr_pl, va_pl, _, _ = load_unity_numerai_data(
                        data_version="v6.0",
                        feature_set="small",
                        era_stride=stride,
                        downsample_mode=ds_m,
                        downsample_frac=ds_f,
                        val_downsample_mode=v_m,
                        val_downsample_frac=v_f,
                        workspace_root=ws,
                        auto_download=False,
                    )
                    with patch.object(data_utils, "POLARS_AVAILABLE", False):
                        tr_pd, va_pd, _, _ = load_unity_numerai_data(
                            data_version="v6.0",
                            feature_set="small",
                            era_stride=stride,
                            downsample_mode=ds_m,
                            downsample_frac=ds_f,
                            val_downsample_mode=v_m,
                            val_downsample_frac=v_f,
                            workspace_root=ws,
                            auto_download=False,
                        )
                    pd.testing.assert_frame_equal(tr_pl.reset_index(drop=True), tr_pd.reset_index(drop=True))
                    pd.testing.assert_frame_equal(va_pl.reset_index(drop=True), va_pd.reset_index(drop=True))

        # 3. Unsorted Parquet (date-only shuffled across eras -> conditional sort MUST restore monotonic era order)
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            tr_raw, feat_cols = make_synthetic_numerai_panel(
                n_eras=12, n_tickers=4, n_features=4, start_era=1, seed=41, data_type="train", include_date=True
            )
            va_raw, _ = make_synthetic_numerai_panel(
                n_eras=6, n_tickers=4, n_features=4, start_era=13, seed=42, data_type="validation", include_date=True
            )
            tr_shuf = tr_raw.drop(columns=["era"]).sample(frac=1.0, random_state=7).reset_index(drop=True)
            va_shuf = va_raw.drop(columns=["era"]).sample(frac=1.0, random_state=9).reset_index(drop=True)
            _write_custom_unity_dataset(ws / "data_folder" / "v6.0", tr_shuf, va_shuf, feat_cols)

            tr_pl, va_pl, _, _ = load_unity_numerai_data(
                data_version="v6.0",
                feature_set="small",
                era_stride=4,
                downsample_mode="recent",
                downsample_frac=0.67,
                val_downsample_mode="earliest",
                val_downsample_frac=0.5,
                workspace_root=ws,
                auto_download=False,
            )
            with patch.object(data_utils, "POLARS_AVAILABLE", False):
                tr_pd, va_pd, _, _ = load_unity_numerai_data(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=4,
                    downsample_mode="recent",
                    downsample_frac=0.67,
                    val_downsample_mode="earliest",
                    val_downsample_frac=0.5,
                    workspace_root=ws,
                    auto_download=False,
                )
            self.assertTrue(tr_pl["era"].is_monotonic_increasing)
            self.assertTrue(va_pl["era"].is_monotonic_increasing)
            pd.testing.assert_frame_equal(tr_pl.reset_index(drop=True), tr_pd.reset_index(drop=True))
            pd.testing.assert_frame_equal(va_pl.reset_index(drop=True), va_pd.reset_index(drop=True))

        # 4. Empty-result edge case (all validation targets NaN -> 0 rows with intact schema/dtypes)
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            tr_raw, feat_cols = make_synthetic_numerai_panel(
                n_eras=4, n_tickers=3, n_features=4, start_era=1, seed=51, data_type="train", include_date=True
            )
            va_raw, _ = make_synthetic_numerai_panel(
                n_eras=2, n_tickers=3, n_features=4, start_era=5, seed=52, data_type="validation", include_date=True
            )
            tr_raw = tr_raw.drop(columns=["era"])
            va_raw = va_raw.drop(columns=["era"])
            va_raw["target"] = np.float32(np.nan)
            _write_custom_unity_dataset(ws / "data_folder" / "v6.0", tr_raw, va_raw, feat_cols)

            tr_pl, va_pl, cols_pl, _ = load_unity_numerai_data(
                data_version="v6.0",
                feature_set="small",
                era_stride=1,
                downsample_mode="recent",
                downsample_frac=0.5,
                val_downsample_mode="earliest",
                val_downsample_frac=0.5,
                workspace_root=ws,
                auto_download=False,
            )
            with patch.object(data_utils, "POLARS_AVAILABLE", False):
                tr_pd, va_pd, cols_pd, _ = load_unity_numerai_data(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=1,
                    downsample_mode="recent",
                    downsample_frac=0.5,
                    val_downsample_mode="earliest",
                    val_downsample_frac=0.5,
                    workspace_root=ws,
                    auto_download=False,
                )
            self.assertEqual(cols_pl, cols_pd)
            self.assertEqual(len(va_pl), 0)
            self.assertEqual(list(va_pl.columns), list(va_pd.columns))
            self.assertEqual(va_pl["era"].dtype, np.int32)
            self.assertEqual(va_pl["target"].dtype, np.float32)
            for col in cols_pl:
                self.assertEqual(va_pl[col].dtype, np.int8)
            pd.testing.assert_frame_equal(tr_pl.reset_index(drop=True), tr_pd.reset_index(drop=True))

        # 5. Tiny downsample_frac (e.g. 0.01 after era_stride) warns & clamps to >=2 eras so era-level CV stays valid
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            write_synthetic_unity_dataset(
                ws / "data_folder" / "v6.0",
                n_train_eras=24,
                n_val_eras=10,
                n_tickers=4,
                n_features=4,
                date_only=True,
                seed=61,
            )
            with self.assertWarns(UserWarning):
                tr_pl, va_pl, _, _ = load_unity_numerai_data(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=4,
                    downsample_mode="recent",
                    downsample_frac=0.01,
                    val_downsample_mode="recent",
                    val_downsample_frac=0.01,
                    workspace_root=ws,
                    auto_download=False,
                )
            with patch.object(data_utils, "POLARS_AVAILABLE", False), self.assertWarns(UserWarning):
                tr_pd, va_pd, _, _ = load_unity_numerai_data(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=4,
                    downsample_mode="recent",
                    downsample_frac=0.01,
                    val_downsample_mode="recent",
                    val_downsample_frac=0.01,
                    workspace_root=ws,
                    auto_download=False,
                )
            self.assertEqual(tr_pl["era"].nunique(), 2)
            self.assertEqual(va_pl["era"].nunique(), 2)
            pd.testing.assert_frame_equal(tr_pl.reset_index(drop=True), tr_pd.reset_index(drop=True))
            pd.testing.assert_frame_equal(va_pl.reset_index(drop=True), va_pd.reset_index(drop=True))

            cv_splits = build_era_cv_splits(
                tr_pl["era"].to_numpy(),
                cv_strategy="timeseries",
                n_splits=1,
                cv_split=0.8,
                target_horizon_eras=12,
            )
            self.assertEqual(len(cv_splits), 1)
            self.assertGreater(len(cv_splits[0][0]), 0)
            self.assertGreater(len(cv_splits[0][1]), 0)

    def test_polars_vs_pandas_load_unity_parity(self) -> None:
        if not POLARS_AVAILABLE:
            self.skipTest("Polars not installed")
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            write_synthetic_unity_dataset(
                ws / "data_folder" / "v6.0",
                n_train_eras=8,
                n_val_eras=4,
                n_tickers=6,
                n_features=6,
                float_unit_features=True,
                include_non_validation_rows=True,
                include_nan_targets=True,
            )
            tr_pl, va_pl, cols_pl, _ = load_unity_numerai_data(
                data_version="v6.0", feature_set="small", era_stride=2, workspace_root=ws, auto_download=False
            )
            with patch.object(data_utils, "POLARS_AVAILABLE", False):
                tr_pd, va_pd, cols_pd, _ = load_unity_numerai_data(
                    data_version="v6.0", feature_set="small", era_stride=2, workspace_root=ws, auto_download=False
                )

            self.assertEqual(cols_pl, cols_pd)
            self.assertEqual(tr_pl.shape, tr_pd.shape)
            self.assertEqual(va_pl.shape, va_pd.shape)
            np.testing.assert_array_equal(tr_pl["era"].to_numpy(), tr_pd["era"].to_numpy())
            np.testing.assert_array_equal(va_pl["era"].to_numpy(), va_pd["era"].to_numpy())
            np.testing.assert_allclose(tr_pl["target"].to_numpy(), tr_pd["target"].to_numpy())
            for col in cols_pl:
                np.testing.assert_array_equal(tr_pl[col].to_numpy(), tr_pd[col].to_numpy())
                np.testing.assert_array_equal(va_pl[col].to_numpy(), va_pd[col].to_numpy())

    def test_polars_per_ticker_timeseries_era_quantize_no_nested_window_error(self) -> None:
        if not POLARS_AVAILABLE:
            self.skipTest("Polars not installed")
        df = pd.DataFrame(
            {
                "era": [1, 1, 2, 2, 3, 3, 4, 4],
                "numerai_ticker": ["A", "B", "A", "B", "A", "B", "A", "B"],
                "f1": [0, 1, 2, 3, 4, 2, 1, 0],
                "f2": [4.0, np.nan, 1.0, 2.0, 3.0, 4.0, 0.0, 2.0],
            }
        )
        for qscope in ("era", "global"):
            with self.subTest(quantize_scope=qscope):
                out_df, added_cols = data_utils.compute_per_ticker_timeseries_features(
                    df,
                    ["f1", "f2"],
                    ticker_col="numerai_ticker",
                    era_col="era",
                    ticker_features=(
                        "rolling_mean|momentum_diff|rolling_rank|"
                        "expanding_stats|weeks_since_change|weeks_since_high|raw_lags"
                    ),
                    ticker_windows=[2, 4],
                    ticker_lag_steps=[1, 2],
                    quantize_scope=qscope,
                )
                self.assertEqual(len(added_cols), 28)
                for cname in added_cols:
                    self.assertEqual(out_df[cname].dtype, np.int8)
                    self.assertTrue(((out_df[cname] >= 0) & (out_df[cname] <= 4)).all())


if __name__ == "__main__":
    unittest.main()
