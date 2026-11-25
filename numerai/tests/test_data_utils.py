"""Unit tests for data loading, per-ticker time-series features, downsampling, and caching in src/data_utils.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from _fixtures import make_synthetic_numerai_panel, write_synthetic_unity_dataset
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


class TestHttpAndPathHelpers(unittest.TestCase):
    """Unit tests for HTTP error formatting and local dataset path discovery."""

    def test_extract_http_status_code_and_format_error(self) -> None:
        class _RespErr(Exception):
            def __init__(self, code: int):
                super().__init__(f"boom {code}")
                self.response = SimpleNamespace(status_code=code)

        class _StatusAttrErr(Exception):
            def __init__(self, code: int):
                super().__init__(f"status {code}")
                self.status = code

        self.assertEqual(_extract_http_status_code(_RespErr(401)), 401)
        self.assertEqual(_extract_http_status_code(_StatusAttrErr(429)), 429)
        self.assertIsNone(_extract_http_status_code(RuntimeError("plain")))

        msg_401 = _format_http_error("Kaggle failed", _RespErr(401))
        self.assertIn("401 Unauthorized", msg_401)

        msg_418 = _format_http_error("Kaggle failed", _RespErr(418))
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

    def test_within_era_keeps_all_eras_and_is_deterministic(self) -> None:
        ds1 = downsample_training_data(self.train_df, downsample_mode="within_era", downsample_frac=0.5, random_state=7)
        ds2 = downsample_training_data(self.train_df, downsample_mode="within_era", downsample_frac=0.5, random_state=7)
        self.assertEqual(set(ds1["era"].unique()), set(self.train_df["era"].unique()))
        self.assertEqual(len(ds1), 12 * 5)
        pd.testing.assert_frame_equal(ds1, ds2)

    def test_recent_earliest_uniform_and_random_eras(self) -> None:
        ds_recent = downsample_training_data(self.train_df, downsample_mode="recent", downsample_frac=0.5)
        np.testing.assert_array_equal(sorted(ds_recent["era"].unique()), [7, 8, 9, 10, 11, 12])

        ds_earliest = downsample_training_data(self.train_df, downsample_mode="earliest", downsample_frac=0.5)
        np.testing.assert_array_equal(sorted(ds_earliest["era"].unique()), [1, 2, 3, 4, 5, 6])

        ds_uniform = downsample_training_data(self.train_df, downsample_mode="uniform", downsample_frac=0.5)
        self.assertEqual(len(ds_uniform["era"].unique()), 6)
        self.assertEqual(int(ds_uniform["era"].min()), 1)
        self.assertEqual(int(ds_uniform["era"].max()), 12)

        ds_rand1 = downsample_training_data(self.train_df, downsample_mode="random_eras", downsample_frac=0.5, random_state=99)
        ds_rand2 = downsample_training_data(self.train_df, downsample_mode="random_eras", downsample_frac=0.5, random_state=99)
        self.assertEqual(len(ds_rand1["era"].unique()), 6)
        pd.testing.assert_frame_equal(ds_rand1, ds_rand2)

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

            def _fake_download(_remote_path: str, dest_path: str) -> None:
                src_name = Path(dest_path).name
                Path(dest_path).write_bytes((staged_dir / src_name).read_bytes())

            mock_numerapi_instance.download_dataset.side_effect = _fake_download
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

    def test_cached_loader_writes_hits_and_downsamples_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            write_synthetic_unity_dataset(ws / "data_folder" / "v6.0", n_train_eras=8, n_val_eras=4, n_tickers=6, n_features=4)

            # First call populates processed_cache
            tr1, val1, cols1, _ = load_unity_numerai_data_cached(
                data_version="v6.0",
                feature_set="small",
                era_stride=1,
                workspace_root=ws,
                auto_download=False,
            )
            cache_dir = ws / "data_folder" / "processed_cache"
            self.assertTrue(cache_dir.is_dir())
            self.assertGreaterEqual(len(list(cache_dir.glob("*.parquet"))), 2)

            # Second call with different downsample_mode MUST hit cache (never invoke uncached loader)
            with patch.object(data_utils, "load_unity_numerai_data", side_effect=AssertionError("Should have hit cache!")):
                tr2, val2, cols2, _ = load_unity_numerai_data_cached(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=1,
                    downsample_mode="recent",
                    downsample_frac=0.5,
                    workspace_root=ws,
                    auto_download=False,
                )

            self.assertEqual(cols1, cols2)
            self.assertEqual(len(tr2["era"].unique()), 4)
            self.assertEqual(len(val2), len(val1))

            # Third call downsampling BOTH train and validation MUST also hit cache
            with patch.object(data_utils, "load_unity_numerai_data", side_effect=AssertionError("Should have hit cache!")):
                tr3, val3, _, _ = load_unity_numerai_data_cached(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=1,
                    downsample_mode="within_era",
                    downsample_frac=0.5,
                    val_downsample_mode="same",
                    val_downsample_frac=0.5,
                    workspace_root=ws,
                    auto_download=False,
                )
            self.assertEqual(len(tr3), len(tr1) // 2)
            self.assertEqual(len(val3), len(val1) // 2)

            # Explicit independent validation downsampling mode (recent eras on val, within_era on train)
            with patch.object(data_utils, "load_unity_numerai_data", side_effect=AssertionError("Should have hit cache!")):
                tr4, val4, _, _ = load_unity_numerai_data_cached(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=1,
                    downsample_mode="within_era",
                    downsample_frac=0.5,
                    val_downsample_mode="recent",
                    val_downsample_frac=0.5,
                    workspace_root=ws,
                    auto_download=False,
                )
            self.assertEqual(len(tr4["era"].unique()), 8)
            self.assertEqual(len(val4["era"].unique()), 2)

            # force_reload=True bypasses cache
            with patch.object(data_utils, "load_unity_numerai_data", wraps=load_unity_numerai_data) as spy:
                load_unity_numerai_data_cached(
                    data_version="v6.0",
                    feature_set="small",
                    era_stride=1,
                    workspace_root=ws,
                    auto_download=False,
                    force_reload=True,
                )
                self.assertEqual(spy.call_count, 1)

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
