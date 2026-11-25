"""Unit tests for general helper functions, NumeraiExtractions, evaluation, and diagnostics in src/utils.py."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _fixtures import make_synthetic_numerai_panel
import utils
from utils import (
    CORR_SANITY_HIGH_THRESHOLD,
    CORR_SANITY_LOW_THRESHOLD,
    FORBIDDEN_FEATURE_COLS,
    NumeraiExtractions,
    _decode_structured,
    _encode_skopt_item,
    _fill_missing_features,
    _infer_non_overlapping_stride,
    _is_constant_1d,
    _is_inactive,
    _parse_era_numbers,
    _parse_pipe_tokens,
    _safe_autocorr,
    build_conditional_bayesian_spaces,
    build_experiment_setup,
    build_quick_results_df,
    candidate_score,
    classify_error_bins,
    compute_eda_era_stationarity,
    compute_era_error_envelope,
    compute_error_bin_shares,
    compute_error_bin_waterfall,
    compute_feature_exposure,
    compute_feature_target_correlations,
    compute_model_error_bin_comparison,
    compute_per_era_mmc,
    compute_per_era_numerai_corr,
    compute_per_era_spearman_corr,
    configure_error_binning,
    deduplicate_param_grid,
    drop_constant_dims,
    evaluate_numerai_predictions,
    export_diagnostics_predictions,
    infer_target_horizon_eras,
    plot_era_error_envelope,
    plot_error_bin_waterfall,
    prepare_bayesian_space,
    prepare_tuned_pipeline,
    quantize_to_5bins,
    reset_cv_metric_log,
    sharpe_ratio,
    walk_forward_predict,
)


class _DummyMeanEstimator(RegressorMixin, BaseEstimator):
    """Minimal deterministic sklearn estimator for testing walk_forward_predict & pipelines."""

    def __init__(self, walk_forward_step_eras=None, offset: float = 0.0):
        self.walk_forward_step_eras = walk_forward_step_eras
        self.offset = offset

    def fit(self, X, y):
        self.fit_max_era_ = int(pd.to_numeric(X["era"]).max()) if isinstance(X, pd.DataFrame) and "era" in X.columns else None
        self.mean_y_ = float(np.mean(y))
        return self

    def predict(self, X):
        n = len(X)
        if isinstance(X, pd.DataFrame) and "feature_alpha_0" in X.columns:
            return self.mean_y_ + self.offset + 0.05 * (X["feature_alpha_0"].to_numpy(dtype=np.float64) - 2.0)
        return np.full(n, self.mean_y_ + self.offset, dtype=np.float64)


class TestEncodingAndParsingHelpers(unittest.TestCase):
    """Unit tests for structured encoding, token parsing, and small numeric guards."""

    def test_decode_and_encode_structured_roundtrip(self) -> None:
        cases = [
            ("None", None),
            ("none", None),
            ("null", None),
            ("[1, 2, 4]", (1, 2, 4)),
            ('{"a": 1}', {"a": 1}),
            ("plain_string", "plain_string"),
            ("[invalid json", "[invalid json"),
            (42, 42),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(_decode_structured(raw), expected)

        self.assertEqual(_encode_skopt_item((4, 12)), "[4, 12]")
        self.assertEqual(_encode_skopt_item({"b": 2, "a": 1}), '{"a": 1, "b": 2}')
        self.assertEqual(_encode_skopt_item(5), 5)

    def test_parse_pipe_tokens_handles_aliases_and_empty(self) -> None:
        valid = ("mean", "std", "skew")
        self.assertEqual(_parse_pipe_tokens(None, valid), set())
        self.assertEqual(_parse_pipe_tokens(False, valid), set())
        self.assertEqual(_parse_pipe_tokens("none", valid), set())
        self.assertEqual(_parse_pipe_tokens(True, valid), set(valid))
        self.assertEqual(_parse_pipe_tokens("all", valid), set(valid))
        self.assertEqual(_parse_pipe_tokens("mean|skew|invalid", valid), {"mean", "skew"})
        self.assertEqual(_parse_pipe_tokens("mean, std", valid), {"mean", "std"})

    def test_is_inactive_and_is_constant_1d(self) -> None:
        for val in (None, False, 0, "None", "none", "null", ""):
            self.assertTrue(_is_inactive(val))
        for val in (True, 1, "lasso", (1, 2)):
            self.assertFalse(_is_inactive(val))

        self.assertTrue(_is_constant_1d(np.array([])))
        self.assertTrue(_is_constant_1d(np.array([1.0])))
        self.assertTrue(_is_constant_1d(np.array([3.0, 3.0, 3.0])))
        self.assertTrue(_is_constant_1d(np.array([np.nan, np.nan])))
        self.assertFalse(_is_constant_1d(np.array([1.0, 2.0, 3.0])))

    def test_parse_era_numbers_and_infer_stride(self) -> None:
        parsed_int = _parse_era_numbers(pd.Index([1, 2, 5]))
        np.testing.assert_allclose(parsed_int, [1.0, 2.0, 5.0])

        parsed_str = _parse_era_numbers(pd.Index(["era100", "era104", "era108"]))
        np.testing.assert_allclose(parsed_str, [100.0, 104.0, 108.0])

        self.assertEqual(_infer_non_overlapping_stride(pd.Index([1]), default_stride=4), 1)
        self.assertEqual(_infer_non_overlapping_stride(pd.Index([1, 2, 3, 4]), default_stride=4), 4)
        self.assertEqual(_infer_non_overlapping_stride(pd.Index([1, 5, 9, 13]), default_stride=4), 1)


class TestQuantizationAndAutocorr(unittest.TestCase):
    """Unit tests for quantize_to_5bins and _safe_autocorr."""

    def test_quantize_to_5bins_bounds_and_dtype(self) -> None:
        rng = np.random.default_rng(7)
        vals = rng.normal(0.0, 1.0, size=100)
        eras = np.repeat(np.arange(1, 6), 20)

        for scope in ("era", "global"):
            for strat in ("zscore", "quantile"):
                with self.subTest(scope=scope, strategy=strat):
                    out = quantize_to_5bins(vals, strategy=strat, eras=eras, quantize_scope=scope)
                    self.assertEqual(out.dtype, np.int8)
                    self.assertEqual(len(out), len(vals))
                    self.assertTrue(np.all((out >= 0) & (out <= 4)))

    def test_quantize_to_5bins_train_fit_uses_frozen_scaler(self) -> None:
        train_vals = np.linspace(-2.0, 2.0, 50)
        scaler = StandardScaler().fit(train_vals.reshape(-1, 1))
        test_vals = np.array([-10.0, 0.0, 10.0])

        out = quantize_to_5bins(test_vals, quantize_scope="train_fit", fitted_scaler=scaler)
        np.testing.assert_array_equal(out, np.array([0, 2, 4], dtype=np.int8))

    def test_quantize_to_5bins_edge_cases(self) -> None:
        self.assertEqual(len(quantize_to_5bins(np.array([]))), 0)
        np.testing.assert_array_equal(
            quantize_to_5bins(np.array([np.nan, np.inf, -np.inf])),
            np.array([2, 2, 2], dtype=np.int8),
        )
        np.testing.assert_array_equal(
            quantize_to_5bins(np.array([5.0, 5.0, 5.0, 5.0])),
            np.array([2, 2, 2, 2], dtype=np.int8),
        )

    def test_safe_autocorr_edge_and_signal_cases(self) -> None:
        self.assertEqual(_safe_autocorr([1.0, 1.0, 1.0, 1.0], lag=1), 0.0)
        self.assertEqual(_safe_autocorr([1.0, 2.0], lag=1), 0.0)
        alternating = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        self.assertLess(_safe_autocorr(alternating, lag=1), -0.95)
        self.assertGreater(_safe_autocorr(alternating, lag=2), 0.95)


class TestMissingFeatureImputation(unittest.TestCase):
    """Unit tests for _fill_missing_features."""

    def test_fast_path_for_clean_int8_dataframe(self) -> None:
        df = pd.DataFrame({"a": np.array([0, 2, 4], dtype=np.int8), "b": np.array([1, 3, 2], dtype=np.int8)})
        filled, defaults = _fill_missing_features(df, missing_fill_value=2.0)
        self.assertIs(filled, df)
        self.assertEqual(filled["a"].dtype, np.int8)
        self.assertTrue(np.allclose(defaults.to_numpy(), [2.0, 2.0]))

    def test_constant_mean_median_and_era_imputation(self) -> None:
        df = pd.DataFrame({
            "f0": [0.0, np.nan, 4.0, 4.0],
            "f1": [1.0, 3.0, np.nan, 3.0],
        })
        eras = pd.Series([1, 1, 2, 2])

        filled_const, _ = _fill_missing_features(df, missing_fill_value="zero")
        self.assertEqual(float(filled_const.loc[1, "f0"]), 0.0)
        self.assertFalse(filled_const.isna().any().any())

        filled_mean, learned = _fill_missing_features(df, missing_fill_value="mean")
        self.assertAlmostEqual(float(filled_mean.loc[1, "f0"]), (0.0 + 4.0 + 4.0) / 3.0, places=5)

        filled_era, _ = _fill_missing_features(df, missing_fill_value="era_mean", era_series=eras)
        self.assertAlmostEqual(float(filled_era.loc[1, "f0"]), 0.0, places=5)
        self.assertAlmostEqual(float(filled_era.loc[2, "f1"]), 3.0, places=5)

        # Applying learned_fill_map on a new DataFrame with NaNs
        test_df = pd.DataFrame({"f0": [np.nan], "f1": [np.nan]})
        filled_test, _ = _fill_missing_features(test_df, missing_fill_value="mean", learned_fill_map=learned)
        self.assertAlmostEqual(float(filled_test.loc[0, "f0"]), float(learned["f0"]), places=5)

    def test_empty_dataframe_returns_empty(self) -> None:
        empty_df = pd.DataFrame()
        out, defaults = _fill_missing_features(empty_df)
        self.assertTrue(out.empty)
        self.assertTrue(defaults.empty)


class TestSearchSpaceHelpers(unittest.TestCase):
    """Unit tests for grid deduplication and Bayesian search space helpers."""

    def test_deduplicate_param_grid_prunes_inactive_combinations(self) -> None:
        param_grid = {
            "estimator__extractor__sparse_selector": [None, "lasso"],
            "estimator__extractor__sparse_alpha": [0.001, 0.01, 0.1],
            "estimator__era_weighting": [None, "recency_exp"],
            "estimator__recency_decay": [0.005, 0.01],
        }
        # Raw Cartesian product = 2 * 3 * 2 * 2 = 24
        # When sparse_selector=None, 3 alphas collapse to 1; when era_weighting=None, 2 decays collapse to 1
        # Expected unique active combinations = (1 + 3) * (1 + 2) = 12
        deduped = deduplicate_param_grid(param_grid)
        self.assertEqual(len(deduped), 12)

    def test_drop_constant_dims_pins_single_values_on_pipeline(self) -> None:
        pipe = Pipeline([("estimator", _DummyMeanEstimator())])
        space = {
            "estimator__offset": [0.25],
            "estimator__walk_forward_step_eras": [None, 10],
        }
        active = drop_constant_dims(space, pipe)
        self.assertNotIn("estimator__offset", active)
        self.assertIn("estimator__walk_forward_step_eras", active)
        self.assertAlmostEqual(pipe.get_params()["estimator__offset"], 0.25)

    def test_prepare_and_build_conditional_bayesian_spaces(self) -> None:
        space = {
            "estimator__extractor__sparse_selector": [None, "lasso"],
            "estimator__extractor__sparse_alpha": [0.001, 0.01],
            "estimator__extractor__rolling_era_windows": [(4, 12), (4, 8)],
        }
        prepared = prepare_bayesian_space(space)
        self.assertIn("estimator__extractor__rolling_era_windows", prepared)

        subspaces = build_conditional_bayesian_spaces(space, total_n_iter=10)
        if utils.SKOPT_AVAILABLE:
            self.assertIsInstance(subspaces, list)
            self.assertEqual(len(subspaces), 2)
            self.assertEqual(sum(n for _, n in subspaces), 10)

    def test_prepare_tuned_pipeline_clears_walk_forward_step_eras(self) -> None:
        pipe = Pipeline([("estimator", _DummyMeanEstimator(walk_forward_step_eras=20, offset=0.0))])
        tuned = prepare_tuned_pipeline(pipe, {"estimator__offset": 0.5})
        self.assertAlmostEqual(tuned.get_params()["estimator__offset"], 0.5)
        self.assertIsNone(tuned.get_params()["estimator__walk_forward_step_eras"])
        # Original pipeline must not be mutated
        self.assertEqual(pipe.get_params()["estimator__walk_forward_step_eras"], 20)


class TestCorrelationsAndNumeraiExtractions(unittest.TestCase):
    """Unit tests for compute_feature_target_correlations and NumeraiExtractions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.df, cls.feature_cols = make_synthetic_numerai_panel(
            n_eras=10, n_tickers=12, n_features=10, seed=101, signal_strength=0.75
        )

    def test_compute_feature_target_correlations_all_methods(self) -> None:
        feats = self.df[self.feature_cols]
        target = self.df["target"]
        eras = self.df["era"]

        for method in (None, "pandas_mean", "numerai_mean", "pearson", "spearman"):
            with self.subTest(method=method):
                corrs = compute_feature_target_correlations(
                    feats, target, correlation_method=method, era_series=eras
                )
                self.assertEqual(list(corrs.index), self.feature_cols)
                self.assertTrue(np.all(np.isfinite(corrs.to_numpy())))
                if method is not None:
                    # feature_alpha_0 carries the injected signal and should be highest correlated
                    self.assertGreater(float(corrs["feature_alpha_0"]), 0.25)

    def test_compute_feature_target_correlations_constant_target_and_empty(self) -> None:
        feats = self.df[self.feature_cols]
        const_target = pd.Series(np.full(len(feats), 0.5), index=feats.index)
        corrs = compute_feature_target_correlations(feats, const_target, correlation_method="pearson")
        np.testing.assert_allclose(corrs.to_numpy(), 0.0)

        empty_corrs = compute_feature_target_correlations(pd.DataFrame(), const_target)
        self.assertTrue(empty_corrs.empty)

    def test_numerai_extractions_excludes_forbidden_columns_and_selects_top_k(self) -> None:
        ext = NumeraiExtractions(top_k=4, correlation_method="pandas_mean", abs_corr=True)
        out = ext.fit_transform(self.df, self.df["target"])

        self.assertEqual(len(ext.selected_feature_names_), 4)
        self.assertIn("feature_alpha_0", ext.selected_feature_names_)
        for forbidden in FORBIDDEN_FEATURE_COLS:
            self.assertNotIn(forbidden, ext.selected_feature_names_)
        self.assertEqual(len(out), len(self.df))
        self.assertIn("era", out.columns)

    def test_numerai_extractions_feature_engineering_modes(self) -> None:
        ext = NumeraiExtractions(
            top_k=6,
            basic_group_stats="mean|std|skew",
            era_dispersion_features="mean|std|extreme_frac",
            fourier_order=2,
            include_rolling_era=True,
            rolling_era_windows=(2, 4),
            include_lag_diff=True,
            era_lag_scales=(1, 2),
            ticker_features="rolling_mean|raw_lags",
            ticker_windows=(2,),
            ticker_lag_steps=(1,),
            quantize_scope="train_fit",
        )
        out = ext.fit_transform(self.df, self.df["target"])

        self.assertIn("fourier_sin_k1", out.columns)
        self.assertIn("fourier_cos_k2", out.columns)
        self.assertIn("row_feat_mean_bin", out.columns)
        self.assertIn("row_extreme_frac_bin", out.columns)
        self.assertTrue(any("_era_roll2" in c for c in out.columns))
        self.assertTrue(any("_era_diff1" in c for c in out.columns))
        self.assertTrue(any("_ticker_roll_mean2" in c for c in out.columns))
        self.assertTrue(len(ext.signal_scalers_) > 0)

        # Transforming a new batch must reuse learned scalers without error
        out_val = ext.transform(self.df.head(24))
        self.assertEqual(list(out_val.columns), list(out.columns))

    def test_numerai_extractions_sparse_selector_and_pca(self) -> None:
        ext = NumeraiExtractions(
            sparse_selector="lasso",
            sparse_alpha=0.001,
            pca_n_components=3,
        )
        out = ext.fit_transform(self.df, self.df["target"])
        self.assertIsNotNone(ext.sparse_selector_)
        self.assertIsNotNone(ext.pca_)
        self.assertIn("pca_comp_0", out.columns)
        self.assertEqual(out["pca_comp_0"].dtype, np.int8)

    def test_numerai_extractions_raises_on_no_features_or_invalid_sparse(self) -> None:
        meta_only = self.df[["era", "id", "target"]]
        with self.assertRaises(ValueError):
            NumeraiExtractions().fit(meta_only, self.df["target"])

        with self.assertRaises(ValueError):
            NumeraiExtractions(sparse_selector="invalid_sel").fit(self.df, self.df["target"])

    def test_numerai_extractions_set_params_decodes_json_and_alias(self) -> None:
        ext = NumeraiExtractions()
        ext.set_params(lag_scales="[1, 3]", rolling_era_windows="[4, 8]")
        self.assertEqual(ext.era_lag_scales, (1, 3))
        self.assertEqual(ext.rolling_era_windows, (4, 8))
        cloned = clone(ext)
        self.assertEqual(cloned.era_lag_scales, (1, 3))

    def test_numerai_extractions_strictly_causal_no_val_lookahead(self) -> None:
        # Phase 1 regression: perturbing future validation eras MUST NOT change earlier validation eras
        # when include_rolling_era=True, include_lag_diff=True (quantize_scope='era') or ticker_features (train_fit)
        train_df = self.df.loc[self.df["era"] <= 5].copy()
        val_orig = self.df.loc[self.df["era"] > 5].copy()
        val_pert = val_orig.copy()
        # Massively perturb the final validation era (era 10)
        future_mask = val_pert["era"] == val_pert["era"].max()
        for col in self.feature_cols:
            val_pert.loc[future_mask, col] = 4

        for qscope in ("era", "train_fit"):
            with self.subTest(quantize_scope=qscope):
                ext = NumeraiExtractions(
                    top_k=4,
                    include_rolling_era=True,
                    rolling_era_windows=(2,),
                    include_lag_diff=True,
                    era_lag_scales=(1,),
                    ticker_features="rolling_mean",
                    ticker_windows=(2,),
                    quantize_scope=qscope,
                ).fit(train_df, train_df["target"])
                out_orig = ext.transform(val_orig)
                out_pert = ext.transform(val_pert)
                early_mask = (val_orig["era"] < val_orig["era"].max()).to_numpy()
                pd.testing.assert_frame_equal(
                    out_orig.loc[early_mask].reset_index(drop=True),
                    out_pert.loc[early_mask].reset_index(drop=True),
                )


class TestEvaluationWalkForwardAndExport(unittest.TestCase):
    """Unit tests for evaluation metrics, walk-forward prediction, MMC, and Parquet export."""

    def test_sharpe_ratio_and_per_era_correlations(self) -> None:
        self.assertEqual(sharpe_ratio(pd.Series(dtype=np.float64)), 0.0)
        self.assertEqual(sharpe_ratio(pd.Series([0.02, 0.02, 0.02])), 0.0)
        s = pd.Series([0.01, 0.03, 0.02, 0.04])
        # Official numerai_tools.scoring.sharpe_ratio uses population std (ddof=0)
        self.assertAlmostEqual(sharpe_ratio(s), float(s.mean() / s.std(ddof=0)), places=10)

        df, _ = make_synthetic_numerai_panel(n_eras=6, n_tickers=20, n_features=4, seed=9)
        y_true = df["target"].to_numpy()
        eras = df["era"].to_numpy()
        sp = compute_per_era_spearman_corr(y_true, y_true, eras=eras)
        nc = compute_per_era_numerai_corr(y_true, y_true, eras=eras)
        self.assertEqual(len(sp), 6)
        self.assertEqual(len(nc), 6)
        self.assertTrue(np.all(sp.to_numpy() > 0.99))
        self.assertTrue(np.all(nc.to_numpy() > 0.90))

    def test_evaluate_numerai_predictions_sanity_tripwire(self) -> None:
        df, feat_cols = make_synthetic_numerai_panel(n_eras=8, n_tickers=30, n_features=6, seed=12)
        y_true = df["target"].to_numpy(dtype=np.float64)
        eras = df["era"].to_numpy()

        # Perfect predictions -> LEAKAGE_WARNING (> 0.04)
        ev_leak = evaluate_numerai_predictions(
            y_true, y_true, eras=eras, features_df=df, feature_columns=feat_cols, verbose=False
        )
        self.assertEqual(ev_leak["sanity_status"], "LEAKAGE_WARNING")
        self.assertGreater(ev_leak["numerai_corr"], CORR_SANITY_HIGH_THRESHOLD)
        self.assertTrue(np.isfinite(ev_leak["feature_exposure"]))

        # Constant predictions -> LOW_SIGNAL_WARNING (< 0.005)
        ev_low = evaluate_numerai_predictions(
            y_true, np.full_like(y_true, 0.5), eras=eras, verbose=False
        )
        self.assertEqual(ev_low["sanity_status"], "LOW_SIGNAL_WARNING")
        self.assertLess(ev_low["numerai_corr"], CORR_SANITY_LOW_THRESHOLD)

    def test_compute_feature_exposure_edge_cases(self) -> None:
        df, feat_cols = make_synthetic_numerai_panel(n_eras=2, n_tickers=10, n_features=4, seed=5)
        self.assertTrue(np.isnan(compute_feature_exposure(df, np.zeros(len(df)), feature_columns=[])))
        self.assertEqual(compute_feature_exposure(df, np.full(len(df), 0.5), feature_columns=feat_cols), 0.0)
        fe = compute_feature_exposure(df, df[feat_cols[0]].to_numpy(dtype=np.float64), feature_columns=feat_cols)
        self.assertGreater(fe, 0.0)

    def test_compute_eda_era_stationarity(self) -> None:
        df, feat_cols = make_synthetic_numerai_panel(n_eras=12, n_tickers=8, n_features=4, seed=13)
        res = compute_eda_era_stationarity(df, feat_cols[:2], era_col="era")
        self.assertEqual(list(res["feature"]), feat_cols[:2])
        self.assertSetEqual(set(res.columns), {"feature", "adf_stat", "adf_pvalue", "stationary_at_5pct"})

        empty_res = compute_eda_era_stationarity(df, feat_cols, era_col="missing_era")
        self.assertTrue(empty_res.empty)

    def test_walk_forward_predict_expanding_and_sliding_no_leakage(self) -> None:
        train_df, _ = make_synthetic_numerai_panel(n_eras=10, n_tickers=6, n_features=4, start_era=1, seed=21)
        val_df, _ = make_synthetic_numerai_panel(n_eras=10, n_tickers=6, n_features=4, start_era=11, seed=22)

        for mode in ("expanding", "sliding"):
            with self.subTest(window_mode=mode):
                preds = walk_forward_predict(
                    _DummyMeanEstimator(),
                    train_df,
                    train_df["target"],
                    train_df["era"],
                    val_df,
                    val_df["target"],
                    val_df["era"],
                    step_eras=4,
                    window_mode=mode,
                    target_horizon_eras=4,
                    era_stride=2,
                )
                self.assertEqual(len(preds), len(val_df))
                self.assertTrue(np.all(np.isfinite(preds)))

    def test_export_diagnostics_predictions_and_mmc(self) -> None:
        val_df, _ = make_synthetic_numerai_panel(n_eras=4, n_tickers=15, n_features=4, start_era=20, seed=31)
        rng = np.random.default_rng(31)
        preds = rng.normal(0.5, 0.1, size=len(val_df))

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "diag" / "validation_predictions.parquet"
            written = export_diagnostics_predictions(val_df, preds, out_path, era_col="era")
            self.assertTrue(written.is_file())
            loaded = pd.read_parquet(written)
            self.assertEqual(len(loaded), len(val_df))
            self.assertIn("id", loaded.columns)
            self.assertIn("prediction", loaded.columns)
            self.assertTrue((loaded["prediction"] > 0.0).all() and (loaded["prediction"] <= 1.0).all())

            # Length mismatch raises ValueError
            with self.assertRaises(ValueError):
                export_diagnostics_predictions(val_df, preds[:-1], out_path)

            # compute_per_era_mmc: missing file -> None
            self.assertIsNone(compute_per_era_mmc(val_df, preds, meta_model_path=None))
            self.assertIsNone(compute_per_era_mmc(val_df, preds, meta_model_path=Path(tmp) / "nope.parquet"))

            # compute_per_era_mmc: valid meta_model.parquet
            meta_path = Path(tmp) / "meta_model.parquet"
            pd.DataFrame({
                "id": val_df["id"],
                "numerai_meta_model": rng.normal(0.5, 0.1, size=len(val_df)),
            }).to_parquet(meta_path, index=False)
            mmc_s = compute_per_era_mmc(val_df, preds, meta_model_path=meta_path)
            self.assertIsNotNone(mmc_s)
            self.assertEqual(len(mmc_s), 4)
            self.assertTrue(np.all(np.isfinite(mmc_s.to_numpy())))


class TestSetupErrorBinningAndReporting(unittest.TestCase):
    """Unit tests for build_experiment_setup, error binning, envelope, waterfall, and CV result tables."""

    def tearDown(self) -> None:
        configure_error_binning(mode="pct", pct_bins=(5.0, 10.0), zero_tols=(0.01, 0.10), abs_bins=(0.125, 0.375))
        reset_cv_metric_log()
        plt.close("all")

    def test_infer_target_horizon_eras(self) -> None:
        self.assertEqual(infer_target_horizon_eras("target"), 12)
        self.assertEqual(infer_target_horizon_eras("target_cyrus_v4_60"), 12)
        self.assertEqual(infer_target_horizon_eras("target_alpha_4"), 1)
        self.assertEqual(infer_target_horizon_eras("target_victor_20"), 4)

    def test_build_experiment_setup_defaults_and_validation_errors(self) -> None:
        cfg = build_experiment_setup(
            target_col="target",
            era_stride=4,
            search_mode="bayes",
            score_version="v1_skepta",
            downsample_mode="recent",
            downsample_frac=0.5,
        )
        self.assertEqual(cfg["target_horizon_eras"], 12)
        self.assertEqual(cfg["embargo_eras"], 3)
        self.assertEqual(cfg["score_version"], "v1_skepta")
        self.assertEqual(cfg["downsample_mode"], "recent")
        self.assertIsNone(cfg["val_downsample_mode"])
        self.assertEqual(cfg["val_downsample_frac"], 1.0)

        cfg_both = build_experiment_setup(
            target_col="target",
            era_stride=4,
            downsample_mode="within_era",
            downsample_frac=0.05,
            val_downsample_mode="same",
        )
        self.assertEqual(cfg_both["val_downsample_mode"], "within_era")
        self.assertEqual(cfg_both["val_downsample_frac"], 0.05)

        # era_stride='auto' / None auto-syncs with target_horizon_eras and suppresses [Overlap Notice]
        buf_60d = io.StringIO()
        with contextlib.redirect_stdout(buf_60d):
            cfg_auto_60 = build_experiment_setup(target_col="target", era_stride="auto")
        self.assertEqual(cfg_auto_60["era_stride"], 12)
        self.assertEqual(cfg_auto_60["target_horizon_eras"], 12)
        self.assertEqual(cfg_auto_60["embargo_eras"], 1)
        self.assertEqual(cfg_auto_60["era_tag"], "all_eras_stride12")
        self.assertIn("era_stride=12 (auto-matched to horizon=12)", buf_60d.getvalue())
        self.assertNotIn("[Overlap Notice]", buf_60d.getvalue())

        buf_20d = io.StringIO()
        with contextlib.redirect_stdout(buf_20d):
            cfg_auto_20 = build_experiment_setup(target_col="target_jupiter_20", era_stride="auto")
        self.assertEqual(cfg_auto_20["era_stride"], 4)
        self.assertEqual(cfg_auto_20["target_horizon_eras"], 4)
        self.assertEqual(cfg_auto_20["embargo_eras"], 1)
        self.assertEqual(cfg_auto_20["era_tag"], "all_eras_stride4")
        self.assertNotIn("[Overlap Notice]", buf_20d.getvalue())

        cfg_none_stride = build_experiment_setup(target_col="target", era_stride=None)
        self.assertEqual(cfg_none_stride["era_stride"], 12)
        self.assertEqual(cfg_none_stride["embargo_eras"], 1)

        buf_overlap = io.StringIO()
        with contextlib.redirect_stdout(buf_overlap):
            cfg_explicit_4 = build_experiment_setup(target_col="target", era_stride=4)
        self.assertEqual(cfg_explicit_4["era_stride"], 4)
        self.assertEqual(cfg_explicit_4["target_horizon_eras"], 12)
        self.assertEqual(cfg_explicit_4["embargo_eras"], 3)
        self.assertIn("[Overlap Notice]", buf_overlap.getvalue())

        invalid_kwargs = [
            {"baseline_mode": "invalid"},
            {"search_mode": "random"},
            {"score_version": "v99"},
            {"cv_strategy": "kfold"},
            {"cv_split": 1.0},
            {"era_stride": 0},
            {"era_stride": -2},
            {"era_stride": "bad_stride"},
            {"era_stride": True},
            {"downsample_mode": "unknown"},
            {"downsample_frac": 0.0},
            {"val_downsample_mode": "unknown"},
            {"val_downsample_frac": 0.0},
            {"eval_mode": "invalid"},
            {"walk_forward_window": "invalid"},
            {"era_weighting": "invalid"},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    build_experiment_setup(**kwargs)

    def test_classify_error_bins_pct_and_abs_modes(self) -> None:
        y_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, np.nan])
        y_pred = np.array([0.85, 0.92, 1.0, 1.07, 1.15, 0.005, 0.20, 0.5])

        codes, residuals, finite = classify_error_bins(y_true, y_pred, mode="pct", pct_bins=(5.0, 10.0), zero_tols=(0.01, 0.10))
        np.testing.assert_array_equal(codes, np.array([0, 1, 2, 3, 4, 2, 4, -1], dtype=np.int8))
        self.assertEqual(int(finite.sum()), 7)
        self.assertEqual(len(residuals), 8)

        codes_abs, _, _ = classify_error_bins(
            np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
            np.array([0.0, 0.3, 0.5, 0.7, 1.0]),
            mode="abs",
            abs_bins=(0.125, 0.375),
        )
        np.testing.assert_array_equal(codes_abs, np.array([0, 1, 2, 3, 4], dtype=np.int8))

        shares = compute_error_bin_shares(y_true, y_pred, mode="pct")
        self.assertAlmostEqual(sum(shares.values()), 100.0, places=5)

        with self.assertRaises(ValueError):
            classify_error_bins([0.5, 0.5], [0.5])
        with self.assertRaises(ValueError):
            configure_error_binning(mode="invalid")

    def test_era_error_envelope_waterfall_and_comparison(self) -> None:
        # Leave a gap at era 2 to test interpolation
        eras = np.array([1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int32)
        y_true = np.array([0.5, 0.5, 0.5, 0.5] * 3, dtype=np.float64)
        y_pred = np.array([0.2, 0.4, 0.6, 0.8] * 3, dtype=np.float64)

        env_df = compute_era_error_envelope(y_true, y_pred, eras)
        self.assertIn(2, env_df.index)
        self.assertTrue(bool(env_df.loc[2, "is_interpolated"]))
        self.assertFalse(bool(env_df.loc[1, "is_interpolated"]))
        self.assertAlmostEqual(float(env_df.loc[1, "over_share"]), 0.5)
        self.assertAlmostEqual(float(env_df.loc[1, "under_share"]), 0.5)

        plotted_env = plot_era_error_envelope(y_true, y_pred, eras, title="Test Envelope", is_best=True)
        self.assertEqual(len(plotted_env), len(env_df))

        wf_df = compute_error_bin_waterfall(y_true, y_pred, eras, mode="abs")
        self.assertIn("Overall", set(wf_df["section"]))
        self.assertIn("Financial Quarter", set(wf_df["section"]))
        plot_error_bin_waterfall(wf_df, title="Test Waterfall")

        comp_df = compute_model_error_bin_comparison(
            {"ModelA": {"validation_preds": y_pred}, "ModelB": {"validation_preds": y_true}},
            y_true,
            best_variant_name="ModelB",
            mode="abs",
        )
        self.assertEqual(len(comp_df), 2)
        self.assertIn("[BEST] ModelB", list(comp_df["variant"]))

        with self.assertRaises(ValueError):
            compute_era_error_envelope(y_true, y_pred[:-1], eras)

    def test_plot_flight_residual_diagnostics_summary_and_exports(self) -> None:
        eras = np.repeat(np.arange(1, 7, dtype=np.int32), 25)
        rng = np.random.default_rng(77)
        y_true = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=len(eras))
        preds_a = np.clip(y_true * 0.2 + 0.4 + rng.normal(0.0, 0.03, size=len(eras)), 0.0, 1.0)
        preds_b = np.clip(y_true * 0.1 + 0.45 + rng.normal(0.0, 0.05, size=len(eras)), 0.0, 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "flight_diag"
            summary_df = utils.plot_flight_residual_diagnostics(
                y_true,
                {"Model A [Tuned]": {"validation_preds": preds_a}, "Model B [Tuned]": preds_b},
                eras,
                best_variant_name="Model A [Tuned]",
                max_scatter_points=60,
                era_window=(2, 5),
                random_state=77,
                save_prefix=prefix,
            )
            self.assertEqual(len(summary_df), 2)
            self.assertEqual(summary_df.loc[0, "variant"], "[BEST] Model A [Tuned]")
            self.assertSetEqual(
                set(summary_df.columns),
                {
                    "variant",
                    "n_valid",
                    "mean_residual",
                    "mae",
                    "rmse",
                    "mean_pct_error_nonzero",
                    "under_predict_pct",
                    "over_predict_pct",
                },
            )
            self.assertTrue((Path(f"{prefix}_residuals_vs_actual.png")).is_file())
            self.assertTrue((Path(f"{prefix}_actual_vs_predicted_by_era.png")).is_file())
            self.assertTrue((Path(f"{prefix}_pct_error_vs_actual.png")).is_file())

        with self.assertRaises(ValueError):
            utils.plot_flight_residual_diagnostics(y_true, {"Bad": preds_a[:-1]}, eras)

    def test_candidate_score_and_build_quick_results_df(self) -> None:
        df, _ = make_synthetic_numerai_panel(n_eras=6, n_tickers=15, n_features=4, seed=44)
        est = _DummyMeanEstimator().fit(df, df["target"])

        reset_cv_metric_log()
        score = candidate_score(est, df, df["target"], score_version="v1_skepta")
        self.assertTrue(np.isfinite(score))

        cv_results = {
            "mean_test_score": [score],
            "split0_test_score": [score],
            "rank_test_score": [1],
            "params": [{"estimator__offset": 0.0}],
            "param_estimator__offset": [0.0],
        }
        quick_df = build_quick_results_df(cv_results)
        self.assertEqual(len(quick_df), 1)
        self.assertEqual(quick_df.loc[0, "score_version"], "v1_skepta")
        self.assertIn("mean_numerai_corr", quick_df.columns)
        self.assertIn("offset", quick_df.columns)

        # Empty input returns empty DataFrame
        self.assertTrue(build_quick_results_df({}).empty)

    def test_build_quick_results_df_wide_repr_with_trials_attrs(self) -> None:
        df, _ = make_synthetic_numerai_panel(n_eras=8, n_tickers=15, n_features=4, seed=45)
        est = _DummyMeanEstimator().fit(df, df["target"])

        reset_cv_metric_log()
        score1 = candidate_score(est, df, df["target"], score_version="v1_skepta")
        score2 = candidate_score(est, df, df["target"], score_version="v1_skepta")
        wide_params = {f"estimator__p{i}": float(i) for i in range(25)}
        cv_results = {
            "mean_test_score": [score1, score2],
            "split0_test_score": [score1, score2],
            "rank_test_score": [1, 2],
            "params": [wide_params, wide_params],
        }
        quick_df = build_quick_results_df(cv_results)
        self.assertGreater(len(quick_df.columns), 20)
        self.assertIsInstance(quick_df.attrs.get("trials_era_corrs_df"), dict)
        # Wide DataFrame repr() triggers horizontal truncation + concat(); must not raise ValueError
        rendered = repr(quick_df)
        self.assertGreater(len(rendered), 0)

        pbo = utils.compute_pbo_cscv(quick_df.attrs["trials_era_corrs_df"], n_submatrices=4, random_state=42)
        self.assertIn("pbo", pbo)


if __name__ == "__main__":
    unittest.main()
