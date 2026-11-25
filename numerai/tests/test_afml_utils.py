"""Unit tests for Phase 2 AFML & Numerai diagnostic suite (src/afml_utils.py)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _fixtures import make_synthetic_numerai_panel
from afml_utils import (
    CombinatorialPurgedEraCV,
    PurgedEraKFold,
    _find_longest_underwater_span,
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
from baseline_utils import BinSnappingEstimator


class TestPurgedCVAndPurging(unittest.TestCase):
    """Tests for PurgedEraKFold, CombinatorialPurgedEraCV, build_era_cv_splits, and train/val purging."""

    def test_purge_train_before_validation_enforces_horizon_gap(self) -> None:
        train_df, _ = make_synthetic_numerai_panel(n_eras=30, n_tickers=6, n_features=4, start_era=1, seed=1)
        val_df, _ = make_synthetic_numerai_panel(n_eras=10, n_tickers=6, n_features=4, start_era=31, seed=2)

        X_p, y_p, eras_p, info = purge_train_before_validation(
            train_df,
            train_df["target"],
            train_df["era"],
            val_df["era"],
            target_horizon_eras=12,
        )
        # min_val_era = 31, horizon = 12 -> cutoff = 19, eras 20..30 purged (11 eras)
        self.assertEqual(info["cutoff_era"], 19)
        self.assertEqual(info["n_eras_purged"], 11)
        self.assertLessEqual(int(eras_p.max()), 19)
        self.assertEqual(len(X_p), len(y_p))

    def test_purged_era_kfold_purges_and_embargoes_by_era_value(self) -> None:
        # Irregular era spacing (simulating random_eras downsampling)
        era_ids = np.array([1, 2, 5, 9, 10, 14, 20, 25, 30, 35, 42, 50, 55, 62, 70, 75, 82, 90, 95, 100])
        eras = np.repeat(era_ids, 4)
        X_dummy = np.zeros((len(eras), 2), dtype=np.float32)

        splitter = PurgedEraKFold(n_splits=4, purge_eras=12, embargo_eras=6, embargo_pct=0.0)
        splits = list(splitter.split(X_dummy, groups=eras))
        self.assertEqual(len(splits), 4)

        for tr_idx, te_idx in splits:
            tr_eras = eras[tr_idx]
            te_eras = eras[te_idx]
            t_min, t_max = int(te_eras.min()), int(te_eras.max())
            # Zero overlap between train and test eras
            self.assertEqual(len(set(tr_eras.tolist()) & set(te_eras.tolist())), 0)
            # Every training era before test fold must satisfy e <= t_min - purge_eras
            before = tr_eras[tr_eras < t_min]
            if len(before) > 0:
                self.assertLessEqual(int(before.max()), t_min - 12)
            # Every training era after test fold must satisfy e > t_max + max(purge_eras, embargo_eras)
            after = tr_eras[tr_eras > t_max]
            if len(after) > 0:
                self.assertGreater(int(after.min()), t_max + 12)

    def test_combinatorial_purged_cv_splits_and_path_reconstruction(self) -> None:
        eras = np.repeat(np.arange(1, 61, dtype=np.int64), 4)
        X_dummy = np.zeros((len(eras), 2), dtype=np.float32)

        cpcv = CombinatorialPurgedEraCV(n_groups=6, n_test_groups=2, purge_eras=4, embargo_eras=2, embargo_pct=0.0)
        self.assertEqual(cpcv.n_splits, 15)
        self.assertEqual(cpcv.n_paths, 5)
        splits = list(cpcv.split(X_dummy, groups=eras))
        self.assertEqual(len(splits), 15)

        rng = np.random.default_rng(17)
        split_corrs = []
        for tr_idx, te_idx in splits:
            tr_e = set(eras[tr_idx].tolist())
            te_e = sorted(set(eras[te_idx].tolist()))
            self.assertEqual(len(tr_e & set(te_e)), 0)
            split_corrs.append(pd.Series(rng.normal(0.02, 0.01, size=len(te_e)), index=te_e))

        reconstructed = cpcv.reconstruct_backtest_paths(split_corrs, eras)
        self.assertEqual(reconstructed["n_paths"], 5)
        self.assertEqual(reconstructed["paths_df"].shape, (60, 5))
        self.assertFalse(reconstructed["paths_df"].isna().any().any())
        self.assertGreater(reconstructed["mean_path_sharpe"], 0.0)

    def test_build_era_cv_splits_all_strategies(self) -> None:
        eras = np.repeat(np.arange(1, 51, dtype=np.int64), 3)
        for strat, expected_min in (("timeseries", 3), ("purged_kfold", 4), ("cpcv", 10)):
            splits = build_era_cv_splits(
                eras,
                cv_strategy=strat,
                n_splits=4,
                target_horizon_eras=4,
                cpcv_n_groups=5,
                cpcv_n_test_groups=2,
            )
            self.assertGreaterEqual(len(splits), expected_min)
        with self.assertRaises(ValueError):
            build_era_cv_splits(eras, cv_strategy="random_kfold")


class TestNeutralizationAndSharpeDiagnostics(unittest.TestCase):
    """Tests for Feature Neutralization, FNC, Lo-Sharpe, PSR, MinTRL, ONC, Clustered DSR, PBO, TuW, HHI, Risk."""

    def test_neutralize_predictions_removes_linear_feature_exposure_per_era(self) -> None:
        df, feat_cols = make_synthetic_numerai_panel(n_eras=6, n_tickers=40, n_features=5, seed=9)
        # Pure linear exposure to feature_alpha_0
        raw_pred = 0.5 + 0.2 * (df[feat_cols[0]].to_numpy(dtype=np.float64) - 2.0) + np.random.default_rng(3).normal(0, 0.05, size=len(df))
        neutral = neutralize_predictions(
            raw_pred,
            df,
            eras=df["era"],
            feature_columns=feat_cols,
            proportion=1.0,
            normalize=False,
        )
        # Within every era, neutralized predictions must be orthogonal to all features
        for _, grp_idx in df.groupby("era").groups.items():
            pos = df.index.get_indexer(grp_idx)
            n_e = neutral[pos]
            f0_e = df.loc[grp_idx, feat_cols[0]].to_numpy(dtype=np.float64)
            corr = float(np.corrcoef(n_e, f0_e)[0, 1])
            self.assertLess(abs(corr), 1e-8)

        fnc = compute_per_era_fnc(df["target"], raw_pred, df, eras=df["era"], feature_columns=feat_cols)
        self.assertEqual(len(fnc["fnc_era_corrs"]), 6)
        self.assertTrue(np.isfinite(fnc["fnc_mean"]))
        self.assertTrue(np.isfinite(fnc["fnc_sharpe"]))

    def test_psr_small_sample_cauchy_schwarz_regression_and_mintrl(self) -> None:
        # Regression test: small 4-era sample with high positive SR and strongly negative excess kurtosis (< -2)
        # Old bias=False kurtosis violated Cauchy-Schwarz and returned PSR=0.0!
        small_high_sr = np.array([0.75, 0.90, 0.76, 0.89], dtype=np.float64)
        psr_val = compute_probabilistic_sharpe_ratio(small_high_sr, benchmark_sr=0.0)
        self.assertGreater(psr_val, 0.99)

        # Positive track record -> finite MinTRL; negative track record -> inf MinTRL
        rng = np.random.default_rng(42)
        good_corrs = rng.normal(0.03, 0.02, size=100)
        self.assertGreater(compute_probabilistic_sharpe_ratio(good_corrs, 0.0), 0.95)
        self.assertTrue(np.isfinite(compute_min_track_record_length(good_corrs, 0.0, confidence=0.95)))

        bad_corrs = rng.normal(-0.02, 0.02, size=100)
        self.assertLess(compute_probabilistic_sharpe_ratio(bad_corrs, 0.0), 0.1)
        self.assertTrue(np.isinf(compute_min_track_record_length(bad_corrs, 0.0, confidence=0.95)))

    def test_lo_adjusted_sharpe_penalizes_positive_autocorrelation(self) -> None:
        rng = np.random.default_rng(12)
        eps = rng.normal(0.0, 0.01, size=200)
        ar1 = np.zeros(200, dtype=np.float64)
        for t in range(1, 200):
            ar1[t] = 0.75 * ar1[t - 1] + eps[t]
        ar1 = ar1 + 0.02
        lo_res = compute_lo_adjusted_sharpe(ar1, max_lag=6)
        self.assertGreater(lo_res["variance_inflation"], 1.5)
        self.assertLess(lo_res["sr_lo"], lo_res["sr_raw"])
        self.assertLess(lo_res["n_eff"], 200.0)

    def test_onc_clusters_recovers_block_correlation_structure(self) -> None:
        # 12 strategies in 3 clear correlated blocks of 4
        rng = np.random.default_rng(7)
        factors = rng.normal(0.0, 1.0, size=(300, 3))
        cols = {}
        for blk in range(3):
            for j in range(4):
                cols[f"s_{blk}_{j}"] = factors[:, blk] + rng.normal(0.0, 0.25, size=300)
        corr_df = pd.DataFrame(cols).corr()
        onc = compute_onc_clusters(corr_df, max_clusters=6, n_init=5, random_state=42)
        self.assertEqual(onc["n_clusters"], 3)

    def test_onc_clustered_dsr_and_pbo_cscv(self) -> None:
        rng = np.random.default_rng(21)
        t_eras = 80
        # 9 trials formed from 3 base factors (so effective E[K] << 9)
        f = rng.normal(0.0, 0.02, size=(t_eras, 3))
        trials = {}
        for k in range(3):
            for rep in range(3):
                drift = 0.035 if k == 0 else 0.0
                trials[f"t_{k}_{rep}"] = drift + f[:, k] + rng.normal(0.0, 0.005, size=t_eras)
        trials_df = pd.DataFrame(trials)
        winner_corrs = trials_df["t_0_0"]
        trial_srs = [float(trials_df[c].mean() / trials_df[c].std()) for c in trials_df.columns]

        dsr_onc = compute_deflated_sharpe_ratio(winner_corrs, trial_srs, trials_matrix=trials_df, random_state=42)
        self.assertEqual(dsr_onc["used_onc_clustering"], 1.0)
        self.assertLess(dsr_onc["effective_n_trials"], float(len(trials_df.columns)))
        self.assertLessEqual(dsr_onc["dsr"], dsr_onc["psr"] + 1e-12)
        self.assertLess(dsr_onc["sidak_alpha"], 0.05)

        pbo_signal = compute_pbo_cscv(trials_df, n_submatrices=8, random_state=42)
        self.assertLess(pbo_signal["pbo"], 0.20)

        # Pure random noise trials should have higher PBO (near 0.5)
        noise_df = pd.DataFrame(rng.normal(0.0, 0.02, size=(t_eras, 10)))
        pbo_noise = compute_pbo_cscv(noise_df, n_submatrices=8, random_state=42)
        self.assertGreater(pbo_noise["pbo"], 0.25)

    def test_drawdown_tuw_hhi_and_strategy_risk(self) -> None:
        # Equity curve: 100 -> 120 -> 100 -> 105 -> 110 -> 125 (3 eras underwater)
        eq = pd.Series([100.0, 120.0, 100.0, 105.0, 110.0, 125.0])
        dd = compute_drawdown_and_tuw(eq, is_returns=False, periods_per_year=52.0)
        self.assertAlmostEqual(dd["max_drawdown"], 20.0 / 120.0, places=6)
        self.assertEqual(dd["tuw_max_eras"], 3)
        self.assertEqual(dd["n_episodes"], 1)

        # Even positive returns -> hhi_pos == 0.0; single spike -> hhi_pos near 1.0
        even_corrs = np.full(20, 0.02)
        self.assertAlmostEqual(compute_hhi_concentration(even_corrs)["hhi_pos"], 0.0, places=4)
        spike_corrs = np.concatenate([[1.0], np.full(19, 1e-4)])
        self.assertGreater(compute_hhi_concentration(spike_corrs)["hhi_pos"], 0.9)

        # Strong edge -> low strategy failure probability
        strong = np.concatenate([np.full(80, 0.03), np.full(20, -0.01)])
        risk = compute_strategy_risk(strong, target_sr=1.0, periods_per_year=52.0)
        self.assertLess(risk["prob_strategy_failure"], 0.05)

    def test_vi_matrix_clustered_mda_and_summary_report(self) -> None:
        df, feat_cols = make_synthetic_numerai_panel(n_eras=16, n_tickers=20, n_features=6, seed=55, signal_strength=0.8)
        vi = compute_variation_of_information_matrix(df, feat_cols, normalized=True)
        self.assertEqual(vi.shape, (6, 6))
        np.testing.assert_allclose(np.diag(vi.to_numpy()), 0.0, atol=1e-12)
        np.testing.assert_allclose(vi.to_numpy(), vi.to_numpy().T, atol=1e-12)
        self.assertTrue(((vi.to_numpy() >= 0.0) & (vi.to_numpy() <= 1.0)).all())

        onc_vi = cluster_features_onc(df, feat_cols, metric="vi", max_clusters=3, random_state=42)
        self.assertGreaterEqual(onc_vi["n_clusters"], 1)

        # Put the injected signal feature in cluster 0 and noise features in cluster 1
        clusters = {0: [feat_cols[0]], 1: feat_cols[1:]}
        est = BinSnappingEstimator(top_k=6, correlation_method="pandas_mean")
        mda_df = compute_clustered_mda(
            est,
            df[["era"] + feat_cols],
            df["target"],
            df["era"],
            feature_columns=feat_cols,
            clusters=clusters,
            target_horizon_eras=1,
            n_splits=3,
            random_state=42,
        )
        self.assertEqual(len(mda_df), 2)
        # Cluster 0 (containing true signal feature_alpha_0) must rank #1 in Clustered MDA!
        self.assertEqual(int(mda_df.loc[0, "cluster_id"]), 0)
        self.assertGreater(float(mda_df.loc[0, "mda_mean"]), float(mda_df.loc[1, "mda_mean"]))

        # Full summary report
        rng = np.random.default_rng(1)
        era_corrs = pd.Series(rng.normal(0.02, 0.015, size=30), index=np.arange(1, 31))
        summary = summarize_afml_backtest_diagnostics(
            era_corrs,
            trial_srs=[0.5, 0.8, 1.1],
            y_true=df["target"],
            y_pred=df[feat_cols[0]].to_numpy(dtype=np.float64) / 4.0,
            features_df=df,
            feature_columns=feat_cols,
            eras=df["era"],
            label="Test Model",
        )
        for key in (
            "era_consistency_pct",
            "lo_adjusted_sharpe",
            "psr",
            "dsr",
            "effective_n_trials_onc",
            "sidak_alpha",
            "min_trl_eras",
            "fnc_mean",
            "tuw_max_eras",
            "hhi_pos",
            "strategy_fail_prob",
        ):
            self.assertIn(key, summary)

    def test_plot_afml_backtest_story_and_compact_reference_table(self) -> None:
        dd_series = pd.Series([0.0, 0.02, 0.05, 0.01, 0.0, 0.03, 0.0], index=np.arange(1, 8))
        span_start, span_end, span_len = _find_longest_underwater_span(dd_series)
        self.assertEqual((span_start, span_end, span_len), (1, 3, 3))

        rng = np.random.default_rng(77)
        era_corrs = pd.Series(rng.normal(0.025, 0.018, size=60), index=np.arange(901, 961))
        summary = summarize_afml_backtest_diagnostics(
            era_corrs,
            trial_srs=[0.6, 0.9, 1.3],
            label="Model [Tuned]",
            random_state=42,
        )
        afml_df = pd.DataFrame([
            {**summary, "variant": "Model [Default]"},
            summary,
        ])
        compact_df = build_compact_afml_reference_table(afml_df, best_variant_name="Model [Tuned]")
        self.assertEqual(compact_df.loc[0, "variant"], "[WINNER] Model [Tuned]")
        self.assertLessEqual(len(compact_df.columns), 12)

        with tempfile.TemporaryDirectory() as tmp:
            story_png = Path(tmp) / "afml_story.png"
            story_meta = plot_afml_backtest_story(
                era_corrs,
                summary,
                regime_stats={
                    "adf_pvalue": 0.008,
                    "ljung_box_pvalue": 0.24,
                    "arch_lm_pvalue": 0.61,
                },
                title="Model [Tuned]",
                save_path=story_png,
            )
            self.assertTrue(story_png.is_file())
            self.assertEqual(len(story_meta["regime_badges"]), 4)
            self.assertIn(story_meta["psr_status"], ("PASS", "BORDERLINE", "FAIL"))

            # Edge case: negative-Sharpe / infinite MinTRL / zero-drawdown series
            neg_corrs = pd.Series(np.full(12, -0.01), index=np.arange(1, 13))
            neg_summary = summarize_afml_backtest_diagnostics(neg_corrs, label="Weak")
            neg_meta = plot_afml_backtest_story(neg_corrs, neg_summary, title="Weak")
            self.assertFalse(neg_meta["track_record_sufficient"])
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
