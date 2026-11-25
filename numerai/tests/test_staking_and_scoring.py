"""Acceptance tests for one-wallet staking simulator, drawdown math, plots, and versioned candidate scoring."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

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


def test_coding_standards() -> None:
    for py_file in (REPO_ROOT / "src").glob("*.py"):
        txt = py_file.read_text(encoding="utf-8")
        assert ".iloc" not in txt, f"Forbidden .iloc found in {py_file}"
    for nb_file in (REPO_ROOT / "notebooks").glob("*.ipynb"):
        nb = json.load(open(nb_file, encoding="utf-8"))
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell.get("source", []))
            assert ".iloc" not in src, f"Forbidden .iloc in {nb_file.name} cell {idx}"
            assert not re.search(r"^\s*(async\s+)?def\s+", src, flags=re.M), f"Forbidden def in {nb_file.name} cell {idx}"
            if idx > 1:
                assert not re.search(
                    r"^\s*(import\s+|from\s+\S+\s+import\s+)", src, flags=re.M
                ), f"Mid-notebook import in {nb_file.name} cell {idx}"
    print("[PASS] Coding standards verified.")


def test_equity_and_drawdown_bounds() -> None:
    extreme_burns = pd.Series(np.full(200, -0.5), index=np.arange(1000, 1200))
    for mode in ("compound", "constant"):
        res = simulate_stake_pnl(extreme_burns, total_capital=1000.0, staking_mode=mode)
        eq = res["equity_curve"].to_numpy()
        assert np.all(eq >= 0.0), f"Negative equity in {mode}: min={eq.min()}"
        assert -100.0 <= res["money_drawdown_pct"] <= 0.0, f"Invalid DD% in {mode}: {res['money_drawdown_pct']}"
        assert res["max_drawdown_gbp"] <= 0.0

    # Wiped-out constant wallet must stay at 0 even if positive returns follow
    wipe_then_boom = pd.Series(
        np.concatenate([np.full(100, -0.5), np.full(100, 0.5)]),
        index=np.arange(1000, 1200),
    )
    res_wb = simulate_stake_pnl(wipe_then_boom, total_capital=1000.0, staking_mode="constant")
    assert np.all(res_wb["equity_curve"].to_numpy() >= 0.0)
    assert -100.0 <= res_wb["money_drawdown_pct"] <= 0.0
    assert res_wb["end_capital"] == 0.0, f"Wiped constant wallet recovered: {res_wb['end_capital']}"

    rng = np.random.default_rng(7)
    vol_corrs = pd.Series(rng.normal(0.005, 0.04, size=300), index=np.arange(900, 1200))
    for mode in ("compound", "constant"):
        res = simulate_stake_pnl(vol_corrs, total_capital=1000.0, staking_mode=mode)
        assert np.all(res["equity_curve"].to_numpy() >= 0.0)
        assert -100.0 <= res["money_drawdown_pct"] <= 0.0
    print("[PASS] Checks 1 & 2: Equity >= 0, ruin stays 0, money_drawdown_pct in [-100, 0].")


def test_constant_corr_single_digit_return() -> None:
    const_52_pre = pd.Series(np.full(52, 0.005), index=np.arange(1000, 1052))
    res_pre = simulate_stake_pnl(const_52_pre, total_capital=1000.0, staking_mode="compound")
    assert 0.5 < res_pre["total_return_pct"] < 5.0, f"Pre-1393 1yr return: {res_pre['total_return_pct']}"
    assert res_pre["score_basis"] == "CORR only (no True MMC / NMMC)"
    assert res_pre["mmc_status"] == "missing"

    const_52_r1393 = pd.Series(np.full(52, 0.005), index=np.arange(1393, 1445))
    res_r1393 = simulate_stake_pnl(const_52_r1393, total_capital=1000.0, staking_mode="compound")
    assert 1.0 < res_r1393["total_return_pct"] < 6.0, f"R1393+ 1yr return: {res_r1393['total_return_pct']}"
    print(
        f"[PASS] Check 3: Constant CORR=0.005 -> Pre-1393 1yr={res_pre['total_return_pct']:+.2f}%, "
        f"R1393+ 1yr={res_r1393['total_return_pct']:+.2f}%."
    )


def test_r1399_churn_and_nmr_gbp_separation() -> None:
    assert abs(compute_r1399_churn_retention(0.1) - 1.0) < 1e-12
    ret_02 = compute_r1399_churn_retention(0.2)
    assert abs(ret_02 - (2.0 / (1.0 + np.exp(1.0)))) < 1e-12
    mixed_eras = pd.Series([0.05, -0.05, 0.05, -0.05], index=[1395, 1396, 1399, 1400])
    rets_ch, _ = utils._compute_clipped_era_returns(mixed_eras, neutral_churn=0.2, payout_rule="auto")
    assert abs(rets_ch.loc[1395] - 0.40) < 1e-12
    assert abs(rets_ch.loc[1396] - (-0.40)) < 1e-12
    assert abs(rets_ch.loc[1399] - (0.40 * ret_02)) < 1e-12
    assert abs(rets_ch.loc[1400] - (-0.40)) < 1e-12

    const_52_pre = pd.Series(np.full(52, 0.005), index=np.arange(1000, 1052))
    fx_series = pd.Series(np.linspace(10.0, 20.0, 52), index=np.arange(1000, 1052))
    res_fx = simulate_stake_pnl(const_52_pre, total_capital=1000.0, nmr_gbp_prices=fx_series)
    assert abs(res_fx["start_nmr"] - 100.0) < 1e-9
    assert res_fx["end_nmr"] > 100.0
    assert abs(res_fx["end_capital"] - res_fx["end_nmr"] * 20.0) < 1e-6
    print("[PASS] R1399 churn retention and NMR/GBP separation verified.")


def test_versioned_candidate_scoring_and_drawdown_split() -> None:
    s_a = pd.Series([0.006 - 0.02 / np.sqrt(2), 0.006 + 0.02 / np.sqrt(2)])
    s_b = pd.Series([0.005 - 0.005 / np.sqrt(2), 0.005 + 0.005 / np.sqrt(2)])
    v0_a, ver0_a, _ = compute_versioned_candidate_score(s_a, score_version="v0_default")
    v0_b, ver0_b, _ = compute_versioned_candidate_score(s_b, score_version="v0_default")
    v1_a, ver1_a, _ = compute_versioned_candidate_score(s_a, score_version="v1_skepta")
    v1_b, ver1_b, _ = compute_versioned_candidate_score(s_b, score_version="v1_skepta")
    assert ver0_a == "v0_default" and ver0_b == "v0_default"
    assert abs(v0_a - 0.006) < 1e-12 and abs(v0_b - 0.005) < 1e-12 and v0_a > v0_b
    assert ver1_a == "v1_skepta" and ver1_b == "v1_skepta"
    assert abs(v1_a - (-0.004)) < 1e-12 and abs(v1_b - 0.0025) < 1e-12 and v1_b > v1_a

    v2_fb, ver2_fb, fb_flag = compute_versioned_candidate_score(s_b, era_mmcs=None, score_version="v2_skepta_mmc")
    assert ver2_fb == "v1_skepta" and fb_flag is True and abs(v2_fb - v1_b) < 1e-12

    mmc_s = pd.Series([0.003, 0.003])
    v2_real, ver2_real, fb_flag2 = compute_versioned_candidate_score(s_b, era_mmcs=mmc_s, score_version="v2_skepta_mmc")
    assert ver2_real == "v2_skepta_mmc" and fb_flag2 is False
    assert abs(v2_real - (0.005 + 2.0 * 0.003 - 0.5 * 0.005)) < 1e-12

    # Check evaluate_numerai_predictions uses corr_drawdown (not max_drawdown)
    rng = np.random.default_rng(1)
    y_true = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=300)
    y_pred = y_true + rng.normal(0.0, 0.4, size=300)
    eras = np.repeat(np.arange(1000, 1010), 30)
    ev = evaluate_numerai_predictions(y_true, y_pred, eras=eras, verbose=False)
    assert "corr_drawdown" in ev and "max_drawdown" not in ev
    assert callable(make_candidate_scorer("v1_skepta"))
    print("[PASS] Check 4: Versioned candidate_score and corr_drawdown split verified.")


def test_one_wallet_plots() -> None:
    rng = np.random.default_rng(11)
    vol_corrs = pd.Series(rng.normal(0.005, 0.03, size=120), index=np.arange(1000, 1120))
    dash_res = plot_stake_pnl_dashboard(vol_corrs, total_capital=1000.0, title="Test One-Wallet")
    assert "mean_ending_capital" not in dash_res and "window_outcomes_df" not in dash_res
    mc_res = plot_stake_pnl_monte_carlo(vol_corrs, total_capital=1000.0, n_sims=50, title="Test One-Wallet MC")
    assert "window_mc_df" not in mc_res and len(mc_res["final_stakes"]) == 50
    assert mc_res["sim_equity_paths"].shape == (50, len(vol_corrs))
    assert float(np.min(mc_res["sim_equity_paths"] - mc_res["start_capital"])) < 0.0
    plt.close("all")
    print("[PASS] Check 5: One-wallet dashboard & Monte Carlo plots verified.")


def test_date_aware_r1393_unwind_drawdown_fx_lag_and_circular_mc() -> None:
    # 1. Date-aware R1393+ & R1399+ switching on v6.0 remapped sequential eras (< 1393!)
    remapped_corrs = pd.Series([0.05, 0.05, 0.05], index=[1230, 1245, 1247])
    era_dates = pd.Series(
        ["2026-08-21", "2026-11-06", "2026-11-14"],
        index=[1230, 1245, 1247],
    )
    rets_date, meta_date = utils._compute_clipped_era_returns(
        remapped_corrs,
        neutral_churn=0.2,
        payout_rule="auto",
        era_dates=era_dates,
    )
    assert meta_date["n_pre_1393_eras"] == 1
    assert meta_date["n_r1393_eras"] == 2
    assert meta_date["n_r1399_eras"] == 1
    assert abs(rets_date.loc[1230] - 0.15) < 1e-12  # Pre-1393: 3 * 0.05
    assert abs(rets_date.loc[1245] - 0.40) < 1e-12  # R1393 (2026-11-06): 8 * 0.05, no churn
    assert abs(rets_date.loc[1247] - (0.40 * compute_r1399_churn_retention(0.2))) < 1e-12  # R1399 (2026-11-14)

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
    # Final net PnL is ~0, but the 6 consecutive burns during unwind must register a real drawdown!
    assert res_unwind["money_drawdown_pct"] < -15.0, f"Unwind drawdown masked: {res_unwind['money_drawdown_pct']}"

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
    # Round 0 opens at t=0 (FX=10) and settles at t=12 (FX=30) -> GBP PnL reflects settlement FX=30!
    expected_nmr_pnl_0 = (100.0 / 12.0) * (3.0 * 0.05)
    assert abs(res_fx_lag["round_pnls_gbp"].loc[1000] - expected_nmr_pnl_0 * 30.0) < 1e-6

    # 4. Circular block bootstrap default block_size=12
    mc_default = utils.simulate_stake_pnl_monte_carlo(
        pd.Series(np.random.default_rng(3).normal(0.01, 0.02, size=60), index=np.arange(1000, 1060)),
        n_sims=40,
    )
    assert mc_default["block_size"] == 12
    print("[PASS] Check 6: Date-aware R1393/R1399, unwind DD, settlement FX lag, and circular block_size=12 verified.")


if __name__ == "__main__":
    test_coding_standards()
    test_equity_and_drawdown_bounds()
    test_constant_corr_single_digit_return()
    test_r1399_churn_and_nmr_gbp_separation()
    test_versioned_candidate_scoring_and_drawdown_split()
    test_one_wallet_plots()
    test_date_aware_r1393_unwind_drawdown_fx_lag_and_circular_mc()
    print("ALL ACCEPTANCE CHECKS 1-6 PASSED!")
