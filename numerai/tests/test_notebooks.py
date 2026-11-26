"""Unit & integration tests for the Unity notebooks (baseline, LightGBM, MiniRocket) and their model utility modules."""

from __future__ import annotations

import ast
import builtins
import json
import re
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from _fixtures import NOTEBOOKS_DIR, REPO_ROOT, make_synthetic_numerai_panel
import baseline_utils
from baseline_utils import (
    BaseNumeraiEstimator,
    BinSnappingEstimator,
    LagReversalEstimator,
    PersistenceEstimator,
    _build_extraction_cache_key,
    clear_extraction_cache,
    configure_extraction_cache,
    display_baseline_interpretability,
    display_top_cv_candidates,
    extract_baseline_interpretability,
    extract_cv_analysis_artifacts,
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
from utils import NumeraiExtractions, build_experiment_setup, reset_cv_metric_log


UNITY_NOTEBOOKS = (
    "baseline_unity.ipynb",
    "lightgbm_unity.ipynb",
    "minirocket_unity.ipynb",
)


def _find_nested_callables_and_lambdas(py_path: object) -> list[str]:
    """Inspect a Python module AST for nested `def`/`class` inside functions or anonymous `lambda` expressions."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    violations: list[str] = []
    stack: list[tuple[ast.AST, bool, str]] = [(tree, False, "")]
    while stack:
        node, inside_function, owner_name = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if inside_function:
                violations.append(f"{py_path.name}:{node.lineno} nested def '{node.name}' inside '{owner_name}'")
            for child in ast.iter_child_nodes(node):
                stack.append((child, True, node.name))
        elif isinstance(node, ast.ClassDef):
            if inside_function:
                violations.append(f"{py_path.name}:{node.lineno} nested class '{node.name}' inside '{owner_name}'")
            for child in ast.iter_child_nodes(node):
                stack.append((child, inside_function, owner_name))
        elif isinstance(node, ast.Lambda):
            violations.append(f"{py_path.name}:{node.lineno} anonymous lambda inside '{owner_name or '<module>'}'")
            for child in ast.iter_child_nodes(node):
                stack.append((child, inside_function, owner_name))
        else:
            for child in ast.iter_child_nodes(node):
                stack.append((child, inside_function, owner_name))
    return violations


class TestNotebookStructureAndStandards(unittest.TestCase):
    """Validate structure, syntax, coding standards, and required sections across all Unity notebooks."""

    def test_no_nested_functions_classes_or_lambdas_in_src_or_tests(self) -> None:
        violations: list[str] = []
        for folder_name in ("src", "tests"):
            for py_path in sorted((REPO_ROOT / folder_name).glob("*.py")):
                violations.extend(_find_nested_callables_and_lambdas(py_path))
        self.assertEqual(violations, [], f"Nested functions/classes/lambdas found: {violations}")

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

                    # Repository coding standards: no forbidden indexer, no def/lambda in notebooks, imports in Cell 1 only
                    self.assertNotIn("." + "iloc", src, f"Forbidden indexer in {nb_name} cell {idx}")
                    self.assertNotIn("display(comparison_df)", src, f"Truncated 42-col comparison_df dump in {nb_name} cell {idx}")
                    self.assertIsNone(
                        re.search(r"^\s*(async\s+)?def\s+", src, flags=re.M),
                        f"Function definition found in {nb_name} cell {idx}",
                    )
                    self.assertIsNone(
                        re.search(r"\blambda\b", src),
                        f"Anonymous lambda found in {nb_name} cell {idx}",
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

                # Standard workflow + 4-layer # Analysis header hierarchy mirrored across all Unity notebooks (in order)
                expected_sections = (
                    "# Setup",
                    "# Prep",
                    "## Exploratory Data Analysis (EDA)",
                    "# Params",
                    "## Grid",
                    "## Bayesian",
                    "# Train",
                    "# Analysis",
                    "## 1. Executive Scorecard (1A–1B)",
                    "## 2. Core Performance & Money Visuals (2A–2D)",
                    "### NumerAPI v6.0 Dataset Discovery & True MMC vs BMC Proxy Check",
                    "### 2A. Simulated Numerai Staking P&L Dashboard (£)",
                    "### 2B. Monte Carlo Block-Bootstrap Risk Analysis (£)",
                    "### 2C–2D. Cumulative (2C) & Rolling 20-Era (2D) Validation CORR",
                    "## 3. Model Diagnostics & Residual Visuals (3A–3E)",
                    "### 3A. Model Diagnostics Table",
                    "### 3B. Error-Direction Envelope Analysis",
                    "### 3C. Top-2 Variants Residual Diagnostics",
                    "### 3D–3E. Model Interpretability (3D) & Top CV Candidates (3E)",
                    "## 4. AFML Backtest Visual Story (4A–4B)",
                    "### 4A. Statistical Credibility & Regime Dashboard",
                    "### 4B. AFML Backtest Credibility Reference Table",
                    "## 5. Optional Community Companion: 5A. `numereval` (Council of Elders)",
                )
                section_positions = []
                for expected_section in expected_sections:
                    self.assertIn(expected_section, markdown_headings, f"{nb_name} missing {expected_section}")
                    section_positions.append(markdown_headings.index(expected_section))
                self.assertEqual(
                    section_positions,
                    sorted(section_positions),
                    f"{nb_name} header sections are out of order: {list(zip(expected_sections, section_positions))}",
                )

    def test_notebook_cell1_imports_expected_model_modules(self) -> None:
        shared_analysis_symbols = (
            "build_compact_afml_reference_table",
            "build_diagnostics_summary_table",
            "build_executive_scorecard_tables",
            "display_compact_dataframe",
            "export_winner_scorecard_json",
            "plot_afml_backtest_story",
            "plot_cumulative_validation_corr",
            "plot_rolling_validation_corr",
            "plot_top2_residuals_vs_actual",
        )
        expected_symbols = {
            "baseline_unity.ipynb": (
                "BinSnappingEstimator",
                "LagReversalEstimator",
                "PersistenceEstimator",
                "clear_extraction_cache",
                "run_baseline",
            ) + shared_analysis_symbols,
            "lightgbm_unity.ipynb": ("LightGBMNumeraiEstimator",) + shared_analysis_symbols,
            "minirocket_unity.ipynb": ("MiniRocketNumeraiEstimator",) + shared_analysis_symbols,
        }
        for nb_name, symbols in expected_symbols.items():
            nb = json.loads((NOTEBOOKS_DIR / nb_name).read_text(encoding="utf-8"))
            cell1_src = "".join(nb["cells"][1].get("source", []))
            for sym in symbols:
                with self.subTest(notebook=nb_name, symbol=sym):
                    self.assertIn(sym, cell1_src)

    def test_minirocket_interpretability_receives_feature_group_mapping(self) -> None:
        nb = json.loads((NOTEBOOKS_DIR / "minirocket_unity.ipynb").read_text(encoding="utf-8"))
        notebook_src = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
        self.assertIn(
            "extract_minirocket_interpretability(fitted_default_models, model_name=model_name, "
            "feature_groups=unity_feature_groups_dict)",
            notebook_src,
        )


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

    def test_clear_extraction_cache_frees_populated_entries_on_100_rows(self) -> None:
        small_df, _ = make_synthetic_numerai_panel(
            n_eras=10, n_tickers=10, n_features=6, start_era=1, seed=77
        )
        self.assertEqual(len(small_df), 100)
        configure_extraction_cache(enabled=True, max_entries=4)
        clear_extraction_cache()
        self.assertEqual(len(baseline_utils._EXTRACTION_CACHE), 0)

        est = BinSnappingEstimator(extractor=NumeraiExtractions(top_k=4), top_k=3)
        est.fit(small_df, small_df["target"])
        self.assertGreaterEqual(len(baseline_utils._EXTRACTION_CACHE), 1)

        cleared_count = clear_extraction_cache()
        self.assertGreaterEqual(cleared_count, 1)
        self.assertEqual(len(baseline_utils._EXTRACTION_CACHE), 0)

    def test_persistence_only_artifact_extraction_on_100_rows(self) -> None:
        train_100, feat_cols = make_synthetic_numerai_panel(
            n_eras=10, n_tickers=10, n_features=4, start_era=1, seed=88
        )
        val_100, _ = make_synthetic_numerai_panel(
            n_eras=10, n_tickers=10, n_features=4, start_era=11, seed=89
        )
        self.assertEqual(len(train_100), 100)
        self.assertEqual(len(val_100), 100)
        persist_cols = ["era", "numerai_ticker"]

        pipe = Pipeline([("estimator", PersistenceEstimator(persistence_mode="last", persistence_lag=2, target_horizon_eras=2))])
        grid = {"estimator__persistence_mode": ["last", "expanding_mean"], "estimator__smoothing": [0.0]}
        b_name, entries, cv_df, _, tuned_model = run_baseline(
            "Baseline P (Persistence)",
            pipe,
            grid,
            train_100[persist_cols],
            train_100["target"],
            train_100["era"],
            val_100[persist_cols],
            val_100["target"],
            val_100["era"],
            [],
            search_mode="grid",
            score_version="v1_skepta",
            eval_mode="static",
            cv_object=TimeSeriesSplit(n_splits=2),
            target_horizon_eras=2,
            era_stride=1,
            verbose=0,
            n_jobs=1,
        )
        self.assertEqual(b_name, "Baseline P (Persistence)")
        self.assertIn("Baseline P (Persistence) [Tuned]", entries)

        interp = extract_baseline_interpretability({b_name: tuned_model})
        self.assertIn(b_name, interp)
        self.assertGreater(interp[b_name]["tracked_tickers"], 0)
        display_baseline_interpretability(interp)

        cv_art = extract_cv_analysis_artifacts({b_name: cv_df}, top_n=5, save_results=False)
        self.assertIn(b_name, cv_art["top_candidates"])
        self.assertIn(b_name, cv_art["trial_stats"])
        self.assertEqual(cv_art["top_candidates"][b_name].attrs, {})
        display_top_cv_candidates(cv_art)

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

    def test_numpy_minirocket_column_order_matches_feature_meta_and_no_unpadded_zero_bias(self) -> None:
        """Regression test: Z[:, j] must match feature_meta_[j] (kernel_id, dilation, bias, channels) strictly over valid positions."""
        rng = np.random.default_rng(123)
        X_arr = rng.normal(1.5, 1.0, size=(12, 3, 16)).astype(np.float32)
        tr = _NumpyMiniRocketTransformer(num_kernels=252, max_dilations_per_kernel=2, random_state=42).fit(X_arr)
        Z = tr.transform(X_arr)
        self.assertEqual(Z.shape, (12, len(tr.feature_meta_)))

        # Independently recompute PPV for every feature directly from feature_meta_ and dilation_specs_
        spec_by_dilation = {int(s["dilation"]): s for s in tr.dilation_specs_}
        for j, meta in enumerate(tr.feature_meta_):
            dil = int(meta["dilation"])
            k_id = int(meta["kernel_id"])
            bias = float(meta["bias"])
            spec = spec_by_dilation[dil]
            conv, valid_lengths = tr._convolve_batch_for_dilation(
                X_arr,
                dilation=dil,
                channel_combos=spec["channel_combos"],
                kernel_indices=spec["kernel_indices"],
                padding_flags=spec["padding_flags"],
            )
            v_len = int(valid_lengths[k_id])
            expected_ppv = (conv[:, k_id, :v_len] > bias).mean(axis=1, dtype=np.float32)
            np.testing.assert_allclose(Z[:, j], expected_ppv, atol=1e-6)

    def test_minirocket_stride4_train_to_dense_val_era_alignment(self) -> None:
        """Regression test: training on stride=4 eras and predicting on step=1 dense validation eras aligns physical lookback."""
        # Build single-ticker deterministic ramp where feature value == era number
        tr_eras = np.arange(4, 52, 4, dtype=np.int32)  # 12 strided eras: 4, 8, ..., 48
        va_eras = np.arange(49, 61, 1, dtype=np.int32)  # 12 dense eras: 49..60
        tr_mat = tr_eras[:, None].astype(np.float32)
        va_mat = va_eras[:, None].astype(np.float32)
        tr_tk = np.array(["T0"] * len(tr_eras), dtype=object)
        va_tk = np.array(["T0"] * len(va_eras), dtype=object)

        est = MiniRocketNumeraiEstimator(seq_len=9, pad_mode="first")
        _ = est._build_sequence_tensor(tr_mat, tr_eras, tr_tk, mode="ticker_sequence", is_fit=True)
        self.assertEqual(est.train_era_step_, 4)

        val_tensor = est._build_sequence_tensor(va_mat, va_eras, va_tk, mode="ticker_sequence", is_fit=False)
        # For query era 60 (last row), lookback with train_era_step_=4 must sample eras [28, 32, 36, 40, 44, 48, 52, 56, 60]!
        expected_era60 = np.array([28, 32, 36, 40, 44, 48, 52, 56, 60], dtype=np.float32)
        np.testing.assert_allclose(val_tensor[-1, 0, :], expected_era60, atol=1e-6)

        # Strict causality on dense validation: perturbing era 60 must not alter era 56's sequence
        va_mat_pert = va_mat.copy()
        va_mat_pert[-1, 0] = 999.0
        val_tensor_pert = est._build_sequence_tensor(va_mat_pert, va_eras, va_tk, mode="ticker_sequence", is_fit=False)
        np.testing.assert_array_equal(val_tensor[:-1], val_tensor_pert[:-1])


class TestPhase1ModelAndNotebookRegressions(unittest.TestCase):
    """Phase 1 regression tests: cache fingerprinting, BinSnapping symmetry, LagReversal weights, and notebook AST hygiene."""

    def test_extraction_cache_key_distinguishes_feature_values(self) -> None:
        df1, feat_cols = make_synthetic_numerai_panel(n_eras=6, n_tickers=8, n_features=4, seed=1)
        df2 = df1.copy()
        df2[feat_cols[0]] = (4 - df2[feat_cols[0]]).astype(np.int8)
        ext = NumeraiExtractions(top_k=3)
        k1 = _build_extraction_cache_key(ext, df1, df1["era"].to_numpy(), df1["target"].to_numpy())
        k2 = _build_extraction_cache_key(ext, df2, df2["era"].to_numpy(), df2["target"].to_numpy())
        self.assertNotEqual(k1, k2)

    def test_bin_snapping_n_bins_3_rescales_symmetrically(self) -> None:
        # Bins 0..4 with target increasing linearly in feature -> n_bins=3 must separate 2 from 4 symmetrically!
        f0 = np.tile(np.array([0, 1, 2, 3, 4], dtype=np.int8), 20)
        y = f0.astype(np.float32) / 4.0
        df = pd.DataFrame({"era": np.repeat(np.arange(1, 11), 10), "f0": f0})
        est = BinSnappingEstimator(n_bins=3, top_k=1, correlation_method="pandas_mean", smoothing=0.0).fit(df, y)
        preds = est.predict(pd.DataFrame({"era": [11, 11, 11], "f0": [0, 2, 4]}))
        # Bin 0 < Bin 1 < Bin 2 strictly (previously 2, 3, 4 all clipped into bin 2!)
        self.assertLess(float(preds[0]), float(preds[1]))
        self.assertLess(float(preds[1]), float(preds[2]))

    def test_lag_reversal_honors_sample_weights_and_uniqueness_decay(self) -> None:
        df, feat_cols = make_synthetic_numerai_panel(n_eras=10, n_tickers=12, n_features=5, seed=19)
        X = df[["era"] + feat_cols]
        y = df["target"]
        w_unweighted = BaseNumeraiEstimator(era_weighting="uniqueness_decay", target_horizon_eras=4)._compute_era_weights(
            df["era"].to_numpy(), None, len(df)
        )
        self.assertIsNotNone(w_unweighted)
        self.assertAlmostEqual(float(np.mean(w_unweighted)), 1.0, places=5)

        est_plain = LagReversalEstimator(lag=1, top_k=5, era_weighting=None).fit(X, y)
        est_decay = LagReversalEstimator(lag=1, top_k=5, era_weighting="uniqueness_decay", recency_decay=0.5).fit(X, y)
        self.assertFalse(np.allclose(est_plain.feature_means_, est_decay.feature_means_))

    def test_all_unity_notebooks_have_zero_undefined_names(self) -> None:
        """AST static check across all Unity notebooks to prevent NameErrors (e.g. missing Cell 1 imports)."""
        setup_globals = set(build_experiment_setup(baseline_mode="all").keys())
        builtin_names = set(dir(builtins)) | {"__name__", "__file__", "display", "get_ipython"}

        for nb_name in UNITY_NOTEBOOKS:
            with self.subTest(notebook=nb_name):
                nb = json.loads((NOTEBOOKS_DIR / nb_name).read_text(encoding="utf-8"))
                defined = set(builtin_names) | setup_globals
                undefined_refs = []

                for cell_idx, cell in enumerate(nb.get("cells", [])):
                    if cell.get("cell_type") != "code":
                        continue
                    src = "".join(cell.get("source", []))
                    if not src.strip():
                        continue
                    tree = ast.parse(src, filename=f"{nb_name}:cell_{cell_idx}")

                    # Walk top-level statements in order so `del x` removes `x` from `defined`
                    # and use-after-del NameErrors are caught across and within cells.
                    for stmt in tree.body:
                        stmt_bound = set()
                        for node in ast.walk(stmt):
                            if isinstance(node, (ast.Import, ast.ImportFrom)):
                                for alias in node.names:
                                    stmt_bound.add((alias.asname or alias.name).split(".")[0])
                            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                                stmt_bound.add(node.id)
                            elif isinstance(node, ast.ExceptHandler) and node.name:
                                stmt_bound.add(node.name)
                            elif isinstance(node, ast.arg):
                                stmt_bound.add(node.arg)
                            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                stmt_bound.add(node.name)

                        scope = defined | stmt_bound
                        for node in ast.walk(stmt):
                            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                                if node.id not in scope:
                                    undefined_refs.append((cell_idx, node.id))

                        defined |= stmt_bound
                        for node in ast.walk(stmt):
                            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
                                defined.discard(node.id)
                            elif (
                                isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Attribute)
                                and node.func.attr == "pop"
                                and isinstance(node.func.value, ast.Call)
                                and isinstance(node.func.value.func, ast.Name)
                                and node.func.value.func.id == "globals"
                                and node.args
                                and isinstance(node.args[0], ast.Constant)
                                and isinstance(node.args[0].value, str)
                            ):
                                defined.discard(node.args[0].value)

                self.assertEqual(
                    undefined_refs,
                    [],
                    f"Undefined names detected in {nb_name}: {undefined_refs}",
                )

    def test_baseline_mode_canonical_names_and_list_or_csv_selection(self) -> None:
        from utils import normalize_baseline_mode

        self.assertEqual(normalize_baseline_mode("bin_snapping"), ("bin_snapping",))
        self.assertEqual(normalize_baseline_mode("lag_reversal"), ("lag_reversal",))
        self.assertEqual(normalize_baseline_mode("persistence"), ("persistence",))
        self.assertEqual(
            normalize_baseline_mode("bin_snapping, lag_reversal"),
            ("bin_snapping", "lag_reversal"),
        )
        self.assertEqual(
            normalize_baseline_mode(["persistence", "bin_snapping", "persistence"]),
            ("persistence", "bin_snapping"),
        )
        self.assertEqual(
            normalize_baseline_mode("all"),
            ("bin_snapping", "lag_reversal", "persistence"),
        )
        # Backward-compatible aliases (without 'both')
        self.assertEqual(normalize_baseline_mode("baseline_a"), ("bin_snapping",))
        self.assertEqual(normalize_baseline_mode("baseline_e"), ("lag_reversal",))
        with self.assertRaises(ValueError):
            normalize_baseline_mode("both")

        cfg_list = build_experiment_setup(baseline_mode=["lag_reversal", "persistence"])
        self.assertEqual(cfg_list["selected_baselines"], ["lag_reversal", "persistence"])
        cfg_csv = build_experiment_setup(baseline_mode="bin_snapping, persistence")
        self.assertEqual(cfg_csv["selected_baselines"], ["bin_snapping", "persistence"])
        with self.assertRaises(ValueError):
            normalize_baseline_mode([])
        with self.assertRaises(ValueError):
            normalize_baseline_mode("bin_snapping, unknown_model")

    def test_memory_guard_explicit_toggle_and_fail_safe(self) -> None:
        from utils import check_memory_budget, configure_memory_guard

        try:
            # Explicit boolean toggle OFF bypasses even absurdly large estimates
            cfg_off = build_experiment_setup(memory_guard_enabled=False, memory_guard_min_free_gb=3.0)
            self.assertFalse(cfg_off["memory_guard_enabled"])
            self.assertTrue(check_memory_budget(10 * (1024 ** 4), context="test_disabled"))

            # Explicit boolean toggle ON raises MemoryError so BayesSearchCV/GridSearchCV assigns error_score=-1.0
            cfg_on = build_experiment_setup(memory_guard_enabled=True, memory_guard_min_free_gb=3.0)
            self.assertTrue(cfg_on["memory_guard_enabled"])
            with self.assertRaises(MemoryError):
                check_memory_budget(10 * (1024 ** 4), context="test_oom_candidate")
        finally:
            configure_memory_guard(enabled=True, min_free_gb=3.0)

    def test_lag_reversal_and_bin_snapping_batched_predict_exact_and_cache_teardown(self) -> None:
        from baseline_utils import _EXTRACTION_CACHE, clear_extraction_cache, configure_extraction_cache
        from utils import candidate_score, walk_forward_predict

        df, feat_cols = make_synthetic_numerai_panel(n_eras=10, n_tickers=12, n_features=6, seed=7)
        X = df[["era", "numerai_ticker"] + feat_cols].copy()
        y = df["target"].copy()

        # Verify LagReversalEstimator batched predict matches exact (X - mu) @ w including NaNs
        est = LagReversalEstimator(lag=1, top_k=None, center_features=True).fit(X, y)
        X_val = X.copy()
        X_val[feat_cols[0]] = X_val[feat_cols[0]].astype(np.float32)
        X_val.loc[X_val.index[0], feat_cols[0]] = np.nan
        preds = est.predict(X_val)
        raw_mat = X_val[list(est.feature_names_in_)].to_numpy(dtype=np.float64)
        clean_mat = np.where(np.isfinite(raw_mat), raw_mat, est.feature_means_)
        expected = (clean_mat - est.feature_means_) @ est.weights_
        np.testing.assert_allclose(preds, expected, atol=1e-6)

        # Verify walk_forward_predict suspends _EXTRACTION_CACHE so it does not pollute cache
        configure_extraction_cache(enabled=True, max_entries=2)
        clear_extraction_cache()
        tr_mask = df["era"] <= 6
        va_mask = df["era"] > 6
        wf_est = BinSnappingEstimator(extractor=NumeraiExtractions(top_k=4), top_k=4)
        _ = walk_forward_predict(
            wf_est,
            X.loc[tr_mask], y.loc[tr_mask], df.loc[tr_mask, "era"],
            X.loc[va_mask], y.loc[va_mask], df.loc[va_mask, "era"],
            step_eras=2, target_horizon_eras=1, era_stride=1,
        )
        self.assertEqual(len(_EXTRACTION_CACHE), 0)

        # Verify candidate_score clears _wf_train_X_ on the scored estimator
        cv_est = BinSnappingEstimator(top_k=4, walk_forward_step_eras=2, target_horizon_eras=1).fit(X.loc[tr_mask], y.loc[tr_mask])
        self.assertIsNotNone(cv_est._wf_train_X_)
        _ = candidate_score(cv_est, X.loc[va_mask], y.loc[va_mask])
        self.assertIsNone(cv_est._wf_train_X_)


if __name__ == "__main__":
    unittest.main()
