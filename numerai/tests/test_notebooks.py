"""Unit & integration tests for the Unity notebooks (baseline, LightGBM, MiniRocket) and their model utility modules."""

from __future__ import annotations

import json
import re
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from _fixtures import NOTEBOOKS_DIR, make_synthetic_numerai_panel
import baseline_utils
from baseline_utils import (
    BaseNumeraiEstimator,
    BinSnappingEstimator,
    LagReversalEstimator,
    PersistenceEstimator,
    _build_extraction_cache_key,
    clear_extraction_cache,
    configure_extraction_cache,
    run_baseline,
)
import lgbm_utils
from lgbm_utils import (
    LightGBMNumeraiEstimator,
    _lgb_dataset_fingerprint,
    clear_lgb_dataset_cache,
    make_shared_lgb_datasets,
)
import minirocket_utils
from minirocket_utils import (
    MiniRocketNumeraiEstimator,
    _NumpyMiniRocketTransformer,
    _compute_dilations_and_counts,
)
from utils import NumeraiExtractions, reset_cv_metric_log


UNITY_NOTEBOOKS = (
    "baseline_unity.ipynb",
    "lightgbm_unity.ipynb",
    "minirocket_unity.ipynb",
)


class TestNotebookStructureAndStandards(unittest.TestCase):
    """Validate structure, syntax, coding standards, and required sections across all Unity notebooks."""

    def test_all_unity_notebooks_exist_and_compile_cleanly(self) -> None:
        for nb_name in UNITY_NOTEBOOKS:
            nb_path = NOTEBOOKS_DIR / nb_name
            with self.subTest(notebook=nb_name):
                self.assertTrue(nb_path.is_file(), f"Missing notebook: {nb_path}")
                nb = json.loads(nb_path.read_text(encoding="utf-8"))
                cells = nb.get("cells", [])
                self.assertGreater(len(cells), 10)

                markdown_headings = []
                for idx, cell in enumerate(cells):
                    src = "".join(cell.get("source", []))
                    if cell["cell_type"] == "markdown":
                        markdown_headings.extend(
                            line.strip() for line in src.splitlines() if line.strip().startswith("#")
                        )
                        continue
                    if cell["cell_type"] != "code":
                        continue

                    # Repository coding standards: no forbidden indexer, no def in notebooks, imports in Cell 1 only
                    self.assertNotIn("." + "iloc", src, f"Forbidden indexer in {nb_name} cell {idx}")
                    self.assertIsNone(
                        re.search(r"^\s*(async\s+)?def\s+", src, flags=re.M),
                        f"Function definition found in {nb_name} cell {idx}",
                    )
                    if idx > 1:
                        self.assertIsNone(
                            re.search(r"^\s*(import\s+|from\s+\S+\s+import\s+)", src, flags=re.M),
                            f"Mid-notebook import found in {nb_name} cell {idx}",
                        )

                    # Every non-empty code cell must be syntactically valid Python (ignoring Jupyter magics)
                    cleaned_lines = [
                        line for line in src.splitlines() if not line.lstrip().startswith(("%", "!"))
                    ]
                    cleaned_src = "\n".join(cleaned_lines)
                    if cleaned_src.strip():
                        compile(cleaned_src, f"{nb_name}:cell_{idx}", "exec")

                # Standard workflow sections present in every Unity notebook
                for expected_section in ("# Setup", "# Prep", "# Params", "# Train", "# Analysis"):
                    self.assertIn(expected_section, markdown_headings, f"{nb_name} missing {expected_section}")

    def test_notebook_cell1_imports_expected_model_modules(self) -> None:
        expected_symbols = {
            "baseline_unity.ipynb": ("BinSnappingEstimator", "LagReversalEstimator", "PersistenceEstimator", "run_baseline"),
            "lightgbm_unity.ipynb": ("LightGBMNumeraiEstimator",),
            "minirocket_unity.ipynb": ("MiniRocketNumeraiEstimator",),
        }
        for nb_name, symbols in expected_symbols.items():
            nb = json.loads((NOTEBOOKS_DIR / nb_name).read_text(encoding="utf-8"))
            cell1_src = "".join(nb["cells"][1].get("source", []))
            for sym in symbols:
                with self.subTest(notebook=nb_name, symbol=sym):
                    self.assertIn(sym, cell1_src)


class TestBaselineNotebookUtils(unittest.TestCase):
    """Unit tests for baseline_unity.ipynb helpers and estimators in src/baseline_utils.py."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.train_df, cls.feature_cols = make_synthetic_numerai_panel(
            n_eras=12, n_tickers=10, n_features=8, start_era=1, seed=101, signal_strength=0.7
        )
        cls.val_df, _ = make_synthetic_numerai_panel(
            n_eras=6, n_tickers=10, n_features=8, start_era=13, seed=202, signal_strength=0.7
        )
        cls.input_cols = ["era", "numerai_ticker"] + cls.feature_cols

    def tearDown(self) -> None:
        clear_extraction_cache()
        configure_extraction_cache(enabled=True, max_entries=4)
        reset_cv_metric_log()

    def test_extraction_cache_config_and_fold_safe_keys(self) -> None:
        cfg = configure_extraction_cache(enabled=True, max_entries=2)
        self.assertEqual(cfg, {"enabled": True, "max_entries": 2})

        ext = NumeraiExtractions(top_k=4)
        X = self.train_df[self.input_cols]
        y = self.train_df["target"].to_numpy(dtype=np.float32)
        eras = self.train_df["era"].to_numpy()

        k1 = _build_extraction_cache_key(ext, X, eras, y)
        k2 = _build_extraction_cache_key(ext, X, eras, y)
        k_diff_y = _build_extraction_cache_key(ext, X, eras, y + 0.1)
        k_diff_ext = _build_extraction_cache_key(NumeraiExtractions(top_k=6), X, eras, y)
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k_diff_y)
        self.assertNotEqual(k1, k_diff_ext)

        configure_extraction_cache(enabled=False, max_entries=0)
        self.assertEqual(len(baseline_utils._EXTRACTION_CACHE), 0)

    def test_base_estimator_era_weights_and_prediction_transforms(self) -> None:
        eras = np.array([1, 1, 2, 2, 2, 3], dtype=np.int32)
        for mode in (None, "equal_era", "recency_linear", "recency_exp"):
            with self.subTest(era_weighting=mode):
                base = BaseNumeraiEstimator(era_weighting=mode, recency_decay=0.1)
                w = base._compute_era_weights(eras, None, len(eras))
                if mode is None:
                    self.assertIsNone(w)
                else:
                    self.assertEqual(len(w), len(eras))
                    self.assertAlmostEqual(float(np.mean(w)), 1.0, places=6)
                    if mode in ("recency_linear", "recency_exp"):
                        self.assertGreater(float(w[-1]), float(w[0]))

        with self.assertRaises(ValueError):
            BaseNumeraiEstimator()._compute_era_weights(eras, [1.0, 2.0], len(eras))

        raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ranked = BaseNumeraiEstimator(prediction_transform="era_rank")._transform_predictions_only(raw, eras)
        self.assertTrue(np.all((ranked > 0.0) & (ranked <= 1.0)))
        zscored = BaseNumeraiEstimator(prediction_transform="era_zscore")._transform_predictions_only(raw, eras)
        self.assertAlmostEqual(float(np.mean(zscored[:2])), 0.0, places=6)

    def test_bin_snapping_estimator_fit_predict_and_modes(self) -> None:
        X_tr = self.train_df[self.input_cols]
        y_tr = self.train_df["target"]
        X_va = self.val_df[self.input_cols]

        with self.assertRaises(ValueError):
            BinSnappingEstimator().predict(X_va)

        for agg_mode in ("mean", "corr_weighted"):
            with self.subTest(agg_mode=agg_mode):
                est = BinSnappingEstimator(
                    extractor=NumeraiExtractions(top_k=5),
                    top_k=4,
                    smoothing=5.0,
                    agg_mode=agg_mode,
                    bias_correction="mean_residual",
                )
                est.fit(X_tr, y_tr)
                preds = est.predict(X_va)
                self.assertEqual(len(preds), len(X_va))
                self.assertTrue(np.all(np.isfinite(preds)))
                self.assertEqual(len(est.lookup_table_), 4)

        with self.assertRaises(ValueError):
            BinSnappingEstimator(agg_mode="bad_agg").fit(X_tr, y_tr)

    def test_lag_reversal_estimator_modes_and_errors(self) -> None:
        X_tr = self.train_df[self.input_cols]
        y_tr = self.train_df["target"]
        X_va = self.val_df[self.input_cols]

        with self.assertRaises(ValueError):
            LagReversalEstimator().predict(X_va)

        configs = [
            {"acf_mode": "feature_mean", "acf_metric": "acf", "sign_mode": "hard"},
            {"acf_mode": "feature_corr", "acf_metric": "pacf", "sign_mode": "soft_threshold"},
            {"acf_mode": "feature_mean", "acf_metric": "acf", "sign_mode": "tanh", "prediction_transform": "era_rank"},
        ]
        for cfg in configs:
            with self.subTest(**cfg):
                est = LagReversalEstimator(lag=1, top_k=5, **cfg).fit(X_tr, y_tr)
                preds = est.predict(X_va)
                self.assertEqual(len(preds), len(X_va))
                self.assertTrue(np.all(np.isfinite(preds)))
                self.assertIn("effective_weight", est.rule_summary_.columns)

        with self.assertRaises(ValueError):
            LagReversalEstimator(acf_mode="invalid").fit(X_tr, y_tr)
        with self.assertRaises(ValueError):
            LagReversalEstimator(acf_metric="invalid").fit(X_tr, y_tr)
        with self.assertRaises(ValueError):
            LagReversalEstimator(sign_mode="invalid").fit(X_tr, y_tr)

    def test_persistence_estimator_modes_and_strict_causality(self) -> None:
        X_tr = self.train_df[self.input_cols]
        y_tr = self.train_df["target"]
        X_va = self.val_df[self.input_cols]

        with self.assertRaises(ValueError):
            PersistenceEstimator().predict(X_va)

        for mode in (None, "zero_rule", "last", "rolling_mean", "expanding_mean"):
            with self.subTest(persistence_mode=mode):
                est = PersistenceEstimator(
                    persistence_mode=mode,
                    persistence_lag=4,
                    persistence_window=3,
                    smoothing=1.0,
                    target_horizon_eras=4,
                ).fit(X_tr, y_tr)
                preds = est.predict(X_va)
                self.assertEqual(len(preds), len(X_va))
                self.assertTrue(np.all(np.isfinite(preds)))

        # Unseen ticker must fall back to global_mean_
        est_last = PersistenceEstimator(persistence_mode="last", persistence_lag=4, target_horizon_eras=4).fit(X_tr, y_tr)
        unseen = pd.DataFrame({"era": [20], "numerai_ticker": ["UNSEEN_TICKER"]})
        self.assertAlmostEqual(float(est_last.predict(unseen)[0]), est_last.global_mean_, places=6)

        # Strict causality: changing targets at eras > (query_era - effective_lag) must NOT affect query_era prediction
        query_era = 8
        y_pert = y_tr.copy()
        y_pert.loc[self.train_df["era"] > (query_era - 4)] = 1.0
        est_orig = PersistenceEstimator(persistence_mode="last", persistence_lag=4, target_horizon_eras=4, smoothing=0.0).fit(X_tr, y_tr)
        est_pert = PersistenceEstimator(persistence_mode="last", persistence_lag=4, target_horizon_eras=4, smoothing=0.0).fit(X_tr, y_pert)
        q_df = self.train_df.loc[self.train_df["era"] == query_era, ["era", "numerai_ticker"]]
        np.testing.assert_allclose(est_orig.predict(q_df), est_pert.predict(q_df))

        with self.assertRaises(ValueError):
            PersistenceEstimator(persistence_mode="invalid").fit(X_tr, y_tr)

    def test_run_baseline_smoke_grid_search(self) -> None:
        X_tr = self.train_df[self.input_cols]
        y_tr = self.train_df["target"]
        X_va = self.val_df[self.input_cols]
        y_va = self.val_df["target"]

        pipe = Pipeline([("estimator", BinSnappingEstimator(top_k=4))])
        grid = {"estimator__top_k": [3, 4], "estimator__smoothing": [0.0]}
        name, entries, cv_df, summary, tuned = run_baseline(
            "Baseline A (Test)",
            pipe,
            grid,
            X_tr,
            y_tr,
            self.train_df["era"],
            X_va,
            y_va,
            self.val_df["era"],
            self.feature_cols,
            search_mode="grid",
            score_version="v1_skepta",
            eval_mode="static",
            cv_object=TimeSeriesSplit(n_splits=2),
            verbose=0,
            n_jobs=1,
        )
        self.assertEqual(name, "Baseline A (Test)")
        self.assertIn("Baseline A (Test) [Default]", entries)
        self.assertIn("Baseline A (Test) [Tuned]", entries)
        self.assertGreaterEqual(len(cv_df), 2)
        self.assertEqual(summary["score_version"], "v1_skepta")
        self.assertTrue(hasattr(tuned.named_steps["estimator"], "bin_table_"))


class TestLightGBMNotebookUtils(unittest.TestCase):
    """Unit tests for lightgbm_unity.ipynb estimator & helpers in src/lgbm_utils.py."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.train_df, cls.feature_cols = make_synthetic_numerai_panel(
            n_eras=10, n_tickers=12, n_features=8, start_era=1, seed=303, signal_strength=0.7
        )
        cls.val_df, _ = make_synthetic_numerai_panel(
            n_eras=5, n_tickers=12, n_features=8, start_era=11, seed=404, signal_strength=0.7
        )
        cls.input_cols = ["era", "numerai_ticker"] + cls.feature_cols

    def tearDown(self) -> None:
        clear_lgb_dataset_cache()

    def test_make_shared_lgb_datasets_and_fingerprint(self) -> None:
        X = self.train_df[self.feature_cols]
        y = self.train_df["target"].to_numpy(dtype=np.float32)
        X_val = self.val_df[self.feature_cols]
        y_val = self.val_df["target"].to_numpy(dtype=np.float32)

        tr_ds, val_ds = make_shared_lgb_datasets(X, y, X_val, y_val, max_bin=63)
        self.assertEqual(tr_ds.num_data(), len(X))
        self.assertIsNotNone(val_ds)
        self.assertEqual(val_ds.num_data(), len(X_val))

        fp1 = _lgb_dataset_fingerprint(X, y, None, {"max_bin": 63})
        fp2 = _lgb_dataset_fingerprint(X, y, None, {"max_bin": 63})
        fp3 = _lgb_dataset_fingerprint(X, y, None, {"max_bin": 127})
        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, fp3)

    def test_build_lgbm_model_parameter_clamping_and_modes(self) -> None:
        # RF boosting auto-enables bagging
        rf_model = LightGBMNumeraiEstimator(
            boosting_type="rf", subsample=1.0, subsample_freq=0, max_depth=3, num_leaves=100,
            force_col_wise=True, force_row_wise=True,
        )._build_lgbm_model()
        params = rf_model.get_params()
        self.assertLess(params["subsample"], 1.0)
        self.assertGreaterEqual(params["subsample_freq"], 1)
        self.assertLessEqual(params["num_leaves"], 8)
        self.assertTrue(params["force_col_wise"])
        self.assertFalse(params["force_row_wise"])

        # DART + quantile / huber / fair objectives
        dart_model = LightGBMNumeraiEstimator(boosting_type="dart", objective="quantile", alpha=0.5)._build_lgbm_model()
        self.assertEqual(dart_model.get_params()["boosting_type"], "dart")
        self.assertAlmostEqual(dart_model.get_params()["alpha"], 0.5)

    def test_lgbm_estimator_standard_and_reused_binned_dataset_paths(self) -> None:
        X_tr = self.train_df[self.input_cols]
        y_tr = self.train_df["target"]
        X_va = self.val_df[self.input_cols]

        with self.assertRaises(ValueError):
            LightGBMNumeraiEstimator().predict(X_va)

        for reuse in (False, True):
            with self.subTest(reuse_binned_dataset=reuse):
                est = LightGBMNumeraiEstimator(
                    extractor=NumeraiExtractions(top_k=6),
                    n_estimators=10,
                    num_leaves=7,
                    max_depth=3,
                    colsample_bytree=0.8,
                    min_child_samples=5,
                    n_jobs=1,
                    reuse_binned_dataset=reuse,
                    bias_correction="mean_residual",
                )
                pipe = Pipeline([("estimator", est)])
                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_va)
                self.assertEqual(len(preds), len(X_va))
                self.assertTrue(np.all(np.isfinite(preds)))

                fitted_est = pipe.named_steps["estimator"]
                self.assertSetEqual(
                    set(fitted_est.feature_importance_df_.columns),
                    {"feature", "gain_importance", "split_importance", "train_corr"},
                )
                if reuse:
                    self.assertGreaterEqual(len(lgbm_utils._LGB_DATASET_CACHE), 1)


class TestMiniRocketNotebookUtils(unittest.TestCase):
    """Unit tests for minirocket_unity.ipynb estimator & helpers in src/minirocket_utils.py."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.train_df, cls.feature_cols = make_synthetic_numerai_panel(
            n_eras=12, n_tickers=8, n_features=10, start_era=1, seed=505, signal_strength=0.7
        )
        cls.val_df, _ = make_synthetic_numerai_panel(
            n_eras=5, n_tickers=8, n_features=10, start_era=13, seed=606, signal_strength=0.7
        )
        cls.input_cols = ["era", "numerai_ticker"] + cls.feature_cols

    def test_compute_dilations_and_counts_invariants(self) -> None:
        dilations, counts = _compute_dilations_and_counts(seq_len=16, num_kernels=336, max_dilations_per_kernel=8)
        self.assertTrue(np.all(dilations >= 1))
        self.assertEqual(len(dilations), len(np.unique(dilations)))
        self.assertEqual(int(counts.sum()), 336 // 84)

    def test_numpy_minirocket_transformer_shapes_bounds_and_determinism(self) -> None:
        rng = np.random.default_rng(7)
        X_3d = rng.normal(0.0, 1.0, size=(24, 3, 12)).astype(np.float32)

        t1 = _NumpyMiniRocketTransformer(num_kernels=168, max_dilations_per_kernel=4, random_state=42).fit(X_3d)
        t2 = _NumpyMiniRocketTransformer(num_kernels=168, max_dilations_per_kernel=4, random_state=42).fit(X_3d)
        Z1 = t1.transform(X_3d)
        Z2 = t2.transform(X_3d)

        self.assertEqual(Z1.shape, (24, 168))
        self.assertTrue(np.all((Z1 >= 0.0) & (Z1 <= 1.0)))
        np.testing.assert_array_equal(Z1, Z2)

        with self.assertRaises(ValueError):
            _NumpyMiniRocketTransformer(num_kernels=84).fit(np.zeros((10, 5)))

    def test_minirocket_estimator_ticker_sequence_feature_profile_and_heads(self) -> None:
        X_tr = self.train_df[self.input_cols]
        y_tr = self.train_df["target"]
        X_va = self.val_df[self.input_cols]

        with self.assertRaises(ValueError):
            MiniRocketNumeraiEstimator().predict(X_va)

        for head in ("ridge_cv", "ridge", "sgd_elasticnet", "sgd_huber"):
            with self.subTest(head_type=head):
                est = MiniRocketNumeraiEstimator(
                    input_mode="ticker_sequence",
                    seq_len=9,
                    max_channels=4,
                    num_kernels=84,
                    max_dilations_per_kernel=2,
                    fit_sample_size=32,
                    backend="numpy",
                    head_type=head,
                    alpha=10.0,
                    batch_size=32,
                ).fit(X_tr, y_tr)
                preds = est.predict(X_va)
                self.assertEqual(len(preds), len(X_va))
                self.assertTrue(np.all(np.isfinite(preds)))
                self.assertEqual(est.active_input_mode_, "ticker_sequence")
                self.assertGreater(len(est.ticker_tail_buffer_), 0)
                self.assertIn("abs_coef", est.kernel_importance_df_.columns)
                self.assertIn("kernel_weight_sum", est.channel_importance_df_.columns)

        # feature_profile fallback when ticker column is absent (and n_cols < 9 padding)
        X_no_tk = self.train_df[["era"] + self.feature_cols[:4]]
        est_fp = MiniRocketNumeraiEstimator(
            input_mode="ticker_sequence",
            num_kernels=84,
            backend="numpy",
            head_type="ridge",
        ).fit(X_no_tk, y_tr)
        self.assertEqual(est_fp.active_input_mode_, "feature_profile")
        self.assertEqual(est_fp.seq_len_in_, 9)
        preds_fp = est_fp.predict(self.val_df[["era"] + self.feature_cols[:4]])
        self.assertEqual(len(preds_fp), len(self.val_df))

        # Invalid options raise ValueError
        with self.assertRaises(ValueError):
            MiniRocketNumeraiEstimator(input_mode="invalid").fit(X_tr, y_tr)
        with self.assertRaises(ValueError):
            MiniRocketNumeraiEstimator(backend="invalid").fit(X_tr, y_tr)
        with self.assertRaises(ValueError):
            MiniRocketNumeraiEstimator(head_type="invalid", num_kernels=84).fit(X_tr, y_tr)

    def test_minirocket_sequence_tensor_strictly_causal(self) -> None:
        """Perturbing future eras (> cutoff) must NOT alter sequence windows at eras <= cutoff."""
        est = MiniRocketNumeraiEstimator(seq_len=9, pad_mode="first")
        mat = self.train_df[self.feature_cols[:3]].to_numpy(dtype=np.float32)
        eras = self.train_df["era"].to_numpy(dtype=np.int32)
        tickers = self.train_df["numerai_ticker"].to_numpy()
        cutoff = 6

        tensor_orig = est._build_sequence_tensor(mat, eras, tickers, mode="ticker_sequence", is_fit=True)

        mat_pert = mat.copy()
        mat_pert[eras > cutoff] = 4.0 - mat_pert[eras > cutoff]
        tensor_pert = est._build_sequence_tensor(mat_pert, eras, tickers, mode="ticker_sequence", is_fit=True)

        np.testing.assert_array_equal(tensor_orig[eras <= cutoff], tensor_pert[eras <= cutoff])

    def test_minirocket_pipeline_clone_and_extractor_integration(self) -> None:
        pipe = Pipeline([
            (
                "estimator",
                MiniRocketNumeraiEstimator(
                    extractor=NumeraiExtractions(top_k=6),
                    num_kernels=84,
                    seq_len=9,
                    max_channels=4,
                    backend="numpy",
                    head_type="ridge",
                ),
            )
        ])
        cloned = clone(pipe)
        cloned.fit(self.train_df[self.input_cols], self.train_df["target"])
        preds = cloned.predict(self.val_df[self.input_cols])
        self.assertEqual(len(preds), len(self.val_df))
        self.assertTrue(np.all(np.isfinite(preds)))


if __name__ == "__main__":
    unittest.main()
