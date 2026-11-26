"""Acceptance and unit tests for one-wallet staking simulator, drawdown math, plots, and versioned candidate scoring."""

from __future__ import annotations

import ast
import json
import re
import sys
import unittest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import utils
from utils import (
    compute_r1399_churn_retention,
    compute_versioned_candidate_score,
    evaluate_numerai_predictions,
    make_candidate_scorer,
    plot_stake_pnl_dashboard,
    plot_stake_pnl_monte_carlo,
    simulate_stake_pnl,
)


def _collect_nested_callables_and_lambdas(py_file: Path) -> List[str]:
    """Return AST violation strings for any nested `def`/`class` inside a function or any `lambda` in `py_file`."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    violations: List[str] = []
    stack: List[tuple[ast.AST, bool, str]] = [(tree, False, "")]
    while stack:
        node, inside_function, owner_name = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if inside_function:
                violations.append(f"{py_file.name}:{node.lineno} nested def '{node.name}' inside '{owner_name}'")
            for child in ast.iter_child_nodes(node):
                stack.append((child, True, node.name))
        elif isinstance(node, ast.ClassDef):
            if inside_function:
                violations.append(f"{py_file.name}:{node.lineno} nested class '{node.name}' inside '{owner_name}'")
            for child in ast.iter_child_nodes(node):
                stack.append((child, inside_function, owner_name))
        elif isinstance(node, ast.Lambda):
            violations.append(f"{py_file.name}:{node.lineno} anonymous lambda inside '{owner_name or '<module>'}'")
            for child in ast.iter_child_nodes(node):
                stack.append((child, inside_function, owner_name))
        else:
            for child in ast.iter_child_nodes(node):
                stack.append((child, inside_function, owner_name))
    return violations


class TestCodingStandards(unittest.TestCase):
    """Repository-wide static checks for forbidden .iloc, nested callables, lambdas, and mid-notebook imports."""

    def test_src_and_tests_coding_standards(self) -> None:
        for py_file in sorted((REPO_ROOT / "src").glob("*.py")):
            with self.subTest(src_file=py_file.name):
                txt = py_file.read_text(encoding="utf-8")
                self.assertNotIn(".iloc", txt, f"Forbidden .iloc found in {py_file}")
                violations = _collect_nested_callables_and_lambdas(py_file)
                self.assertEqual(violations, [], f"Nested functions/classes/lambdas found in {py_file.name}")

        for py_file in sorted((REPO_ROOT / "tests").glob("*.py")):
            with self.subTest(test_file=py_file.name):
                violations = _collect_nested_callables_and_lambdas(py_file)
                self.assertEqual(violations, [], f"Nested functions/classes/lambdas found in {py_file.name}")

    def test_notebooks_coding_standards(self) -> None:
        for nb_file in sorted((REPO_ROOT / "notebooks").glob("*.ipynb")):
            nb = json.loads(nb_file.read_text(encoding="utf-8"))
            for idx, cell in enumerate(nb["cells"]):
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(notebook=nb_file.name, cell=idx):
                    src = "".join(cell.get("source", []))
                    self.assertNotIn(".iloc", src, f"Forbidden .iloc in {nb_file.name} cell {idx}")
                    self.assertIsNone(
                        re.search(r"^\s*(async\s+)?def\s+", src, flags=re.M),
                        f"Forbidden def in {nb_file.name} cell {idx}",
                    )
                    cleaned_src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("%", "!")))
                    if cleaned_src.strip():
                        cell_tree = ast.parse(cleaned_src, filename=f"{nb_file.name}:cell_{idx}")
                        lambdas = [n.lineno for n in ast.walk(cell_tree) if isinstance(n, ast.Lambda)]
                        self.assertEqual(lambdas, [], f"Forbidden lambda in {nb_file.name} cell {idx} lines {lambdas}")
                    if idx > 1:
                        self.assertIsNone(
                            re.search(r"^\s*(import\s+|from\s+\S+\s+import\s+)", src, flags=re.M),
                            f"Mid-notebook import in {nb_file.name} cell {idx}",
                        )


class TestStakingSimulatorAndChurn(unittest.TestCase):
    """Unit and boundary tests for simulate_stake_pnl, R1399 churn, FX lag, and staking plots."""

    def tearDown(self) -> None:
        plt.close("all")

    def test_equity_and_drawdown_bounds(self) -> None:
        # Arrange
        extreme_burns = pd.Series(np.full(200, -0.5), index=np.arange(1000, 1200))
        wipe_then_boom = pd.Series(
            np.concatenate([np.full(100, -0.5), np.full(100, 0.5)]),
            index=np.arange(1000, 1200),
        )
        rng = np.random.default_rng(7)
        vol_corrs = pd.Series(rng.normal(0.005, 0.04, size=300), index=np.arange(900, 1200))

        # Act & Assert
        for mode in ("compound", "constant"):
            with self.subTest(staking_mode=mode):
                res = simulate_stake_pnl(extreme_burns, total_capital=1000.0, staking_mode=mode)
                eq = res["equity_curve"].to_numpy()
                self.assertTrue(np.all(eq >= 0.0), f"Negative equity in {mode}: min={eq.min()}")
                self.assertGreaterEqual(res["money_drawdown_pct"], -100.0)
                self.assertLessEqual(res["money_drawdown_pct"], 0.0)
                self.assertLessEqual(res["max_drawdown_gbp"], 0.0)

                res_vol = simulate_stake_pnl(vol_corrs, total_capital=1000.0, staking_mode=mode)
                self.assertTrue(np.all(res_vol["equity_curve"].to_numpy() >= 0.0))
                self.assertGreaterEqual(res_vol["money_drawdown_pct"], -100.0)
                self.assertLessEqual(res_vol["money_drawdown_pct"], 0.0)

        # Wiped-out constant wallet must stay at 0 even if positive returns follow
        res_wb = simulate_stake_pnl(wipe_then_boom, total_capital=1000.0, staking_mode="constant")
        self.assertTrue(np.all(res_wb["equity_curve"].to_numpy() >= 0.0))
        self.assertGreaterEqual(res_wb["money_drawdown_pct"], -100.0)
        self.assertLessEqual(res_wb["money_drawdown_pct"], 0.0)
        self.assertEqual(res_wb["end_capital"], 0.0)

    def test_simulate_stake_pnl_invalid_mode_and_boundary_inputs(self) -> None:
        corrs = pd.Series([0.01, 0.02], index=[1000, 1001])
        with self.assertRaises(ValueError):
            simulate_stake_pnl(corrs, total_capital=1000.0, staking_mode="martingale")

        # Zero-capital boundary
        res_zero = simulate_stake_pnl(corrs, total_capital=0.0, staking_mode="compound")
        self.assertEqual(res_zero["start_capital"], 0.0)
        self.assertEqual(res_zero["end_capital"], 0.0)
        self.assertEqual(res_zero["money_drawdown_pct"], 0.0)

        # Empty series boundary
        res_empty = simulate_stake_pnl(pd.Series(dtype=np.float64), total_capital=1000.0)
        self.assertEqual(res_empty["end_capital"], 1000.0)
        self.assertEqual(res_empty["total_return_pct"], 0.0)

    def test_constant_corr_single_digit_return(self) -> None:
        const_52_pre = pd.Series(np.full(52, 0.005), index=np.arange(1000, 1052))
        res_pre = simulate_stake_pnl(const_52_pre, total_capital=1000.0, staking_mode="compound")
        self.assertGreater(res_pre["total_return_pct"], 0.5)
        self.assertLess(res_pre["total_return_pct"], 5.0)
        self.assertEqual(res_pre["score_basis"], "CORR only (no True MMC / NMMC)")
        self.assertEqual(res_pre["mmc_status"], "missing")

        const_52_r1393 = pd.Series(np.full(52, 0.005), index=np.arange(1393, 1445))
        res_r1393 = simulate_stake_pnl(const_52_r1393, total_capital=1000.0, staking_mode="compound")
        self.assertGreater(res_r1393["total_return_pct"], 1.0)
        self.assertLess(res_r1393["total_return_pct"], 6.0)

    def test_compute_r1399_churn_retention_boundaries_and_types(self) -> None:
        # Scalar boundaries: <= 0.1 saturates at 1.0; non-finite falls back to 1.0
        self.assertAlmostEqual(compute_r1399_churn_retention(0.0), 1.0, places=12)
        self.assertAlmostEqual(compute_r1399_churn_retention(0.1), 1.0, places=12)
        self.assertAlmostEqual(compute_r1399_churn_retention(0.2), 2.0 / (1.0 + np.exp(1.0)), places=12)
        self.assertLess(compute_r1399_churn_retention(1.0), 1e-3)
        self.assertEqual(compute_r1399_churn_retention(np.nan), 1.0)
        self.assertEqual(compute_r1399_churn_retention(np.inf), 1.0)

        # Array & Series equivalence partitions
        arr = np.array([0.05, 0.2, np.nan], dtype=np.float64)
        arr_out = compute_r1399_churn_retention(arr)
        self.assertEqual(arr_out.shape, (3,))
        self.assertAlmostEqual(arr_out[0], 1.0, places=12)
        self.assertAlmostEqual(arr_out[1], 2.0 / (1.0 + np.exp(1.0)), places=12)
        self.assertEqual(arr_out[2], 1.0)

        ser = pd.Series([0.05, 0.2, np.nan], index=["a", "b", "c"])
        ser_out = compute_r1399_churn_retention(ser)
        self.assertEqual(list(ser_out.index), ["a", "b", "c"])
        np.testing.assert_allclose(ser_out.to_numpy(), arr_out, rtol=1e-12, atol=1e-12)

    def test_r1399_churn_and_nmr_gbp_separation(self) -> None:
        ret_02 = compute_r1399_churn_retention(0.2)
        mixed_eras = pd.Series([0.05, -0.05, 0.05, -0.05], index=[1395, 1396, 1399, 1400])
        rets_ch, _ = utils._compute_clipped_era_returns(mixed_eras, neutral_churn=0.2, payout_rule="auto")
        self.assertAlmostEqual(rets_ch.loc[1395], 0.40, places=12)
        self.assertAlmostEqual(rets_ch.loc[1396], -0.40, places=12)
        self.assertAlmostEqual(rets_ch.loc[1399], 0.40 * ret_02, places=12)
        self.assertAlmostEqual(rets_ch.loc[1400], -0.40, places=12)

        const_52_pre = pd.Series(np.full(52, 0.005), index=np.arange(1000, 1052))
        fx_series = pd.Series(np.linspace(10.0, 20.0, 52), index=np.arange(1000, 1052))
        res_fx = simulate_stake_pnl(const_52_pre, total_capital=1000.0, nmr_gbp_prices=fx_series)
        self.assertAlmostEqual(res_fx["start_nmr"], 100.0, places=9)
        self.assertGreater(res_fx["end_nmr"], 100.0)
        self.assertAlmostEqual(res_fx["end_capital"], res_fx["end_nmr"] * 20.0, places=6)

    @patch("matplotlib.pyplot.show")
    def test_one_wallet_plots(self, mock_show: MagicMock) -> None:
        rng = np.random.default_rng(11)
        vol_corrs = pd.Series(rng.normal(0.005, 0.03, size=120), index=np.arange(1000, 1120))
        dash_res = plot_stake_pnl_dashboard(vol_corrs, total_capital=1000.0, title="Test One-Wallet")
        self.assertNotIn("mean_ending_capital", dash_res)
        self.assertNotIn("window_outcomes_df", dash_res)

        mc_res = plot_stake_pnl_monte_carlo(vol_corrs, total_capital=1000.0, n_sims=50, title="Test One-Wallet MC")
        self.assertNotIn("window_mc_df", mc_res)
        self.assertEqual(len(mc_res["final_stakes"]), 50)
        self.assertEqual(mc_res["sim_equity_paths"].shape, (50, len(vol_corrs)))
        self.assertLess(float(np.min(mc_res["sim_equity_paths"] - mc_res["start_capital"])), 0.0)
        self.assertEqual(mock_show.call_count, 2)

    def test_date_aware_r1393_unwind_drawdown_fx_lag_and_circular_mc(self) -> None:
        # 1. Date-aware R1393+ & R1399+ switching on v6.0 remapped sequential eras (< 1393!)
        remapped_corrs = pd.Series([0.05, 0.05, 0.05], index=[1230, 1245, 1247])
        era_dates = pd.Series(["2026-08-21", "2026-11-06", "2026-11-14"], index=[1230, 1245, 1247])
        rets_date, meta_date = utils._compute_clipped_era_returns(
            remapped_corrs,
            neutral_churn=0.2,
            payout_rule="auto",
            era_dates=era_dates,
        )
        self.assertEqual(meta_date["n_pre_1393_eras"], 1)
        self.assertEqual(meta_date["n_r1393_eras"], 2)
        self.assertEqual(meta_date["n_r1399_eras"], 1)
        self.assertAlmostEqual(rets_date.loc[1230], 0.15, places=12)
        self.assertAlmostEqual(rets_date.loc[1245], 0.40, places=12)
        self.assertAlmostEqual(rets_date.loc[1247], 0.40 * compute_r1399_churn_retention(0.2), places=12)

        # 2. Terminal unwind drawdown captured chronologically (burns then recovery during final 12 open rounds)
        tail_burn_then_recover = pd.Series(
            np.concatenate([np.full(12, 0.0), np.full(6, -0.33), np.full(6, 0.33)]),
            index=np.arange(1000, 1024),
        )
        res_unwind = simulate_stake_pnl(
            tail_burn_then_recover,
            total_capital=1000.0,
            concurrent_rounds=12,
            settlement_rounds=12,
            staking_mode="constant",
        )
        self.assertLess(res_unwind["money_drawdown_pct"], -15.0)

        # 3. Settlement-week FX conversion lag (round_pnls_gbp uses fx[t + settlement_rounds], not fx[t])
        const_corrs = pd.Series(np.full(20, 0.05), index=np.arange(1000, 1020))
        fx_step = pd.Series([10.0] * 12 + [30.0] * 8, index=np.arange(1000, 1020))
        res_fx_lag = simulate_stake_pnl(
            const_corrs,
            total_capital=1000.0,
            concurrent_rounds=12,
            settlement_rounds=12,
            staking_mode="constant",
            nmr_gbp_prices=fx_step,
        )
        expected_nmr_pnl_0 = (100.0 / 12.0) * (3.0 * 0.05)
        self.assertAlmostEqual(res_fx_lag["round_pnls_gbp"].loc[1000], expected_nmr_pnl_0 * 30.0, places=6)

        # 4. Circular block bootstrap default block_size=12
        mc_default = utils.simulate_stake_pnl_monte_carlo(
            pd.Series(np.random.default_rng(3).normal(0.01, 0.02, size=60), index=np.arange(1000, 1060)),
            n_sims=40,
        )
        self.assertEqual(mc_default["block_size"], 12)


class TestVersionedCandidateScoring(unittest.TestCase):
    """Unit, boundary, and mock-isolated tests for versioned candidate scoring."""

    def test_versioned_candidate_scoring_and_drawdown_split(self) -> None:
        s_a = pd.Series([0.006 - 0.02 / np.sqrt(2), 0.006 + 0.02 / np.sqrt(2)])
        s_b = pd.Series([0.005 - 0.005 / np.sqrt(2), 0.005 + 0.005 / np.sqrt(2)])
        v0_a, ver0_a, _ = compute_versioned_candidate_score(s_a, score_version="v0_default")
        v0_b, ver0_b, _ = compute_versioned_candidate_score(s_b, score_version="v0_default")
        v1_a, ver1_a, _ = compute_versioned_candidate_score(s_a, score_version="v1_skepta")
        v1_b, ver1_b, _ = compute_versioned_candidate_score(s_b, score_version="v1_skepta")
        self.assertEqual(ver0_a, "v0_default")
        self.assertEqual(ver0_b, "v0_default")
        self.assertAlmostEqual(v0_a, 0.006, places=12)
        self.assertAlmostEqual(v0_b, 0.005, places=12)
        self.assertGreater(v0_a, v0_b)
        self.assertEqual(ver1_a, "v1_skepta")
        self.assertEqual(ver1_b, "v1_skepta")
        self.assertAlmostEqual(v1_a, -0.004, places=12)
        self.assertAlmostEqual(v1_b, 0.0025, places=12)
        self.assertGreater(v1_b, v1_a)

        v2_fb, ver2_fb, fb_flag = compute_versioned_candidate_score(s_b, era_mmcs=None, score_version="v2_skepta_mmc")
        self.assertEqual(ver2_fb, "v1_skepta")
        self.assertTrue(fb_flag)
        self.assertAlmostEqual(v2_fb, v1_b, places=12)

        mmc_s = pd.Series([0.003, 0.003])
        v2_real, ver2_real, fb_flag2 = compute_versioned_candidate_score(s_b, era_mmcs=mmc_s, score_version="v2_skepta_mmc")
        self.assertEqual(ver2_real, "v2_skepta_mmc")
        self.assertFalse(fb_flag2)
        self.assertAlmostEqual(v2_real, 0.005 + 2.0 * 0.003 - 0.5 * 0.005, places=12)

        rng = np.random.default_rng(1)
        y_true = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=300)
        y_pred = y_true + rng.normal(0.0, 0.4, size=300)
        eras = np.repeat(np.arange(1000, 1010), 30)
        ev = evaluate_numerai_predictions(y_true, y_pred, eras=eras, verbose=False)
        self.assertIn("corr_drawdown", ev)
        self.assertNotIn("max_drawdown", ev)
        self.assertTrue(callable(make_candidate_scorer("v1_skepta")))

    def test_versioned_score_boundaries_and_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            compute_versioned_candidate_score([0.01, 0.02], score_version="v99_unknown")

        # Empty series boundary
        score_empty, ver_empty, fb_empty = compute_versioned_candidate_score([], score_version="v2_skepta_mmc")
        self.assertEqual(score_empty, 0.0)
        self.assertEqual(ver_empty, "v1_skepta")
        self.assertTrue(fb_empty)

        # Single-element boundary (std_corr == 0.0)
        score_one, ver_one, fb_one = compute_versioned_candidate_score([0.02], score_version="v1_skepta")
        self.assertAlmostEqual(score_one, 0.02, places=12)
        self.assertEqual(ver_one, "v1_skepta")
        self.assertFalse(fb_one)

        # Non-finite MMC falls back to v1_skepta
        score_inf, ver_inf, fb_inf = compute_versioned_candidate_score(
            [0.02, 0.02], era_mmcs=[np.nan, np.inf], score_version="v2_skepta_mmc"
        )
        self.assertAlmostEqual(score_inf, 0.02, places=12)
        self.assertEqual(ver_inf, "v1_skepta")
        self.assertTrue(fb_inf)

    @patch("utils.compute_per_era_spearman_corr")
    @patch("utils.compute_per_era_numerai_corr")
    def test_make_candidate_scorer_isolated_with_mock(
        self,
        mock_num_corr: MagicMock,
        mock_sp_corr: MagicMock,
    ) -> None:
        # Arrange
        era_series = pd.Series([0.01, 0.03], index=[1, 2], dtype=np.float64)
        mock_num_corr.return_value = era_series
        mock_sp_corr.return_value = era_series
        mock_est = MagicMock(spec=["predict"])
        mock_est.predict.return_value = np.array([0.4, 0.6, 0.5, 0.7])
        X_df = pd.DataFrame({"era": [1, 1, 2, 2], "f1": [0.1, 0.2, 0.3, 0.4]})
        y_arr = np.array([0.0, 1.0, 0.25, 0.75])

        # Act
        scorer = make_candidate_scorer("v1_skepta")
        score = scorer(mock_est, X_df, y_arr)

        # Assert
        mock_est.predict.assert_called_once()
        mock_num_corr.assert_called_once()
        mock_sp_corr.assert_called_once()
        expected = float(era_series.mean() - 0.5 * era_series.std(ddof=1))
        self.assertAlmostEqual(score, expected, places=12)


if __name__ == "__main__":
    unittest.main()
