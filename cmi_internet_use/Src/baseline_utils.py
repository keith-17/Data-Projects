"""baseline_utils.py

Handcrafted Autocorrelation & Circadian DSP Baseline for the Kaggle Child Mind
Institute — Problematic Internet Use (CMI PIU) competition.

Architectural & DSP lineage:
* Periodicity & fundamental preference: ``cr7_kimmitch`` ``PpgHr.swift``
  (linear drift removal, fundamental-peak preference >= 0.85 * max_peak, and
  Variant-A 3-point parabolic sub-lag interpolation).
* Circadian Cosinor rhythm: ``cr7_kimmitch`` ``CircadianEngine.swift``
  (24-hour single-component OLS Cosinor fit: mesor, amplitude, relative
  amplitude, acrophase hour).
* Sedentary bouts, quiet-hours, posture & self-calibrated motion:
  ``cr7_kimmitch`` ``SedentaryDetector.swift``, ``NightStandDown.swift``,
  ``WakeMotionRefinement.swift``, ``ImuFeatureExtractor.swift``, and
  ``SleepAccel.swift`` (never zero-fill missing wear windows: ``SleepDebt.swift``).
* Sleep regularity, calibration-invariant motion, posture & PRSA/CUSUM:
  ``OpenStrap/analytics`` + ``OpenStrap/edge`` (van Hees 2015/2018 5 deg / 5 min
  sustained inactivity, Phillips 2017 Sleep Regularity Index, Fisher circular
  social jetlag, self-calibrated ``g_ref``, Vaha-Ypya MAD, 15s ``dynAmp`` highpass,
  spherical rotation, upright night posture, Bauer PRSA ``AC/DC``, Page CUSUM).
* DSP rigor (``dsp-slides.pdf``): zero-mean autocovariance vs autocorrelation
  (Ch. Random Signals, slides 150-154), unbiased finite-overlap lag scaling
  ``1 / M_k``, zero-phase Butterworth ``sosfiltfilt`` anti-aliasing strictly
  below Nyquist before decimation (p. 180), and Hann-windowed Welch PSD
  periodogram energy ratios (slide 163 / p. 199).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import ndimage, optimize, signal as sps
from scipy.fft import next_fast_len
from sklearn.base import BaseEstimator

RAW_ACTIGRAPHY_DT_S: float = 5.0
RAW_ACTIGRAPHY_FS_HZ: float = 1.0 / RAW_ACTIGRAPHY_DT_S  # 0.2 Hz -> Nyquist = 0.1 Hz
NS_PER_HOUR: float = 3_600_000_000_000.0
SECONDS_PER_DAY: float = 86_400.0

LEAKY_TARGET_COLS = ("sii", "PCIAT-Season", "PCIAT-PCIAT_Total")
LEAKY_PREFIX = "PCIAT-"

# Module-level deterministic caches keyed strictly by input + preprocessing signature.
_GRID_CACHE: Dict[Tuple[Any, ...], Optional[Dict[str, Any]]] = {}
_HC_FEAT_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}

DEFAULT_HANDCRAFTED_PARAMS: Dict[str, Any] = {
    "bin_minutes": 15,
    "resample_method": "anti_alias_mean",
    "non_wear_policy": "mask",
    "min_bin_wear_frac": 0.25,
    "max_gap_bins": 4,
    "clip_enmo": 4.0,
    "acf_detrend": "linear",
    "acf_bias": "unbiased",
    "sedentary_threshold_g": 0.025,
    "period_band_hours": (18.0, 30.0),
    "fundamental_ratio": 0.85,
    "min_acf_strength": 0.08,
    "short_lags_h": (0.25, 0.5, 1.0, 2.0),
    "scoring_rule": "weighted_z_index",
    "coupling_alpha": 0.35,
    "threshold_mode": "optimize_qwk",
}


# ---------------------------------------------------------------------------
# Metric & Ordinal Threshold Utilities (Self-contained for standalone use)
# ---------------------------------------------------------------------------

def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int], min_rating: int = 0, max_rating: int = 3) -> float:
    """Compute exact Quadratic Weighted Kappa (QWK) on ordinal integers in ``[min_rating, max_rating]``."""
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    if len(yt) == 0 or len(yt) != len(yp):
        return 0.0
    yt = np.clip(yt, min_rating, max_rating) - min_rating
    yp = np.clip(yp, min_rating, max_rating) - min_rating
    n_classes = int(max_rating - min_rating + 1)

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(O, (yt, yp), 1.0)
    hist_t = O.sum(axis=1)
    hist_p = O.sum(axis=0)
    total = float(len(yt))
    E = np.outer(hist_t, hist_p) / max(total, 1.0)

    idx = np.arange(n_classes, dtype=np.float64)
    W = np.square(idx[:, None] - idx[None, :]) / max(float((n_classes - 1) ** 2), 1.0)

    denom = float(np.sum(W * E))
    if denom <= 1e-12:
        return 0.0
    num = float(np.sum(W * O))
    return float(1.0 - num / denom)


def apply_thresholds(y_continuous: Sequence[float], thresholds: Sequence[float] = (0.5, 1.5, 2.5)) -> np.ndarray:
    """Convert continuous severity scores into ordinal classes ``{0, 1, 2, 3}`` via sorted thresholds."""
    vals = np.asarray(y_continuous, dtype=np.float64).ravel()
    th = np.sort(np.asarray(thresholds, dtype=np.float64))
    return np.digitize(vals, th).astype(np.int64)


def optimize_thresholds_qwk(
    y_true: Sequence[int],
    y_continuous: Sequence[float],
    initial_thresholds: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Find monotonic cutpoints ``(t0, t1, t2)`` maximizing training-fold QWK.

    Uses alternating 1D exact coordinate search over empirical quantiles plus
    Nelder-Mead local polish so step-function QWK plateaus cannot trap optimization.
    """
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yc = np.asarray(y_continuous, dtype=np.float64).ravel()
    if len(yt) < 8 or float(np.nanstd(yc)) <= 1e-12:
        return np.array([0.5, 1.5, 2.5], dtype=np.float64)

    yc = np.nan_to_num(yc, nan=float(np.nanmedian(yc)))

    def _loss(th: np.ndarray) -> float:
        preds = apply_thresholds(yc, np.sort(th))
        return -quadratic_weighted_kappa(yt, preds)

    p0 = float(np.mean(yt <= 0))
    p1 = float(np.mean(yt <= 1))
    p2 = float(np.mean(yt <= 2))
    sd_yc = float(np.std(yc))
    min_gap = max(1e-3, 0.08 * sd_yc)

    q_init = np.quantile(yc, [np.clip(p0, 0.05, 0.90), np.clip(p1, 0.15, 0.96), np.clip(p2, 0.70, 0.99)])
    if int((yt == 3).sum()) == 0:
        q_init[2] = float(np.max(yc)) + min_gap
    q_init = np.array(
        [q_init[0], max(q_init[1], q_init[0] + min_gap), max(q_init[2], q_init[1] + min_gap)],
        dtype=np.float64,
    )

    starts: List[np.ndarray] = [
        q_init,
        np.array([0.5, 1.5, 2.5], dtype=np.float64),
        np.quantile(yc, [0.55, 0.82, 0.96]),
    ]
    if initial_thresholds is not None:
        starts.append(np.sort(np.asarray(initial_thresholds, dtype=np.float64)))

    best_th = q_init.copy()
    best_qwk = quadratic_weighted_kappa(yt, apply_thresholds(yc, best_th))

    for s in starts:
        cand = np.sort(s.copy())
        if int((yt == 3).sum()) == 0:
            cand[2] = max(cand[2], float(np.max(yc)) + min_gap)
        sc = quadratic_weighted_kappa(yt, apply_thresholds(yc, cand))
        if sc > best_qwk:
            best_qwk = sc
            best_th = cand

    # Coordinate-wise 1D grid sweep over empirical continuous score percentiles (immune to zero gradients)
    grid_vals = np.unique(np.quantile(yc, np.linspace(0.05, 0.995, 95)))
    if len(grid_vals) >= 6:
        cur = best_th.copy()
        for _ in range(3):
            improved = False
            for j in (0, 1, 2):
                if j == 2 and int((yt == 3).sum()) == 0:
                    continue
                lo_bound = cur[j - 1] + min_gap if j > 0 else -np.inf
                hi_bound = cur[j + 1] - min_gap if j < 2 else np.inf
                cands_j = grid_vals[(grid_vals > lo_bound) & (grid_vals < hi_bound)]
                for val in cands_j:
                    trial = cur.copy()
                    trial[j] = val
                    sc = quadratic_weighted_kappa(yt, apply_thresholds(yc, trial))
                    if sc > best_qwk + 1e-9:
                        best_qwk = sc
                        cur = trial.copy()
                        best_th = trial.copy()
                        improved = True
            if not improved:
                break

    # Local Nelder-Mead refinement around the global coordinate optimum
    res = optimize.minimize(_loss, np.sort(best_th), method="Nelder-Mead", options={"maxiter": 250, "xatol": 1e-4, "fatol": 1e-5})
    nm_cand = np.sort(res.x)
    if (nm_cand[1] - nm_cand[0]) >= min_gap and (nm_cand[2] - nm_cand[1]) >= min_gap:
        if quadratic_weighted_kappa(yt, apply_thresholds(yc, nm_cand)) > best_qwk:
            best_th = nm_cand
    return np.sort(best_th.astype(np.float64))


# ---------------------------------------------------------------------------
# DSP & cr7_kimmitch Math Primitives
# ---------------------------------------------------------------------------

def contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return ``[(start, end), ...]`` index slices of contiguous ``True`` runs without bridging gaps."""
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size == 0 or not m.any():
        return []
    padded = np.concatenate(([False], m, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def bounded_gap_fill(values: np.ndarray, mask: np.ndarray, max_gap: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate ONLY short gaps (length <= ``max_gap``), preserving long non-wear gaps as NaN."""
    x = np.asarray(values, dtype=np.float64).copy().ravel()
    m = np.asarray(mask, dtype=bool).copy().ravel() & np.isfinite(x)
    if max_gap <= 0 or m.all() or not m.any():
        return np.where(m, x, np.nan), m

    runs = contiguous_true_runs(~m)
    n = len(x)
    idx_all = np.arange(n, dtype=np.float64)
    obs_idx = np.flatnonzero(m)
    obs_val = x[obs_idx]
    interp_all = np.interp(idx_all, obs_idx.astype(np.float64), obs_val)

    for s, e in runs:
        if s > 0 and e < n and (e - s) <= max_gap:
            x[s:e] = interp_all[s:e]
            m[s:e] = True
    return np.where(m, x, np.nan), m


def zero_phase_lowpass_sos(
    x: np.ndarray,
    fs_hz: float,
    cutoff_hz: float,
    order: int = 4,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Zero-phase IIR Butterworth lowpass via ``sosfiltfilt`` (``dsp-slides.pdf`` p. 180).

    Strictly validates ``0 < cutoff_hz < fs_hz / 2`` (Nyquist) so anti-aliasing cannot distort phase.
    Falls back cleanly when a run is shorter than ``sosfiltfilt`` minimum pad length.
    """
    arr = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * float(fs_hz)
    if not (0.0 < float(cutoff_hz) < nyq):
        raise ValueError(f"Lowpass cutoff {cutoff_hz} Hz must lie strictly inside (0, Nyquist={nyq} Hz)")
    wn = float(cutoff_hz) / nyq
    sos = sps.butter(int(order), wn, btype="low", output="sos")
    min_len = 3 * (2 * sos.shape[0] + 1) + 1
    if arr.shape[axis] <= min_len:
        return arr.copy()
    return sps.sosfiltfilt(sos, arr, axis=axis)


def detrend_observed(values: np.ndarray, mask: np.ndarray, mode: str = "linear") -> np.ndarray:
    """Remove DC mean (``constant``, ``dsp-slides``) or OLS drift (``linear``, ``PpgHr``) over worn bins."""
    x = np.asarray(values, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel() & np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    n_valid = int(m.sum())
    if n_valid == 0:
        return out
    xm = x[m]
    if mode == "none":
        out[m] = xm
    elif mode == "constant":
        out[m] = xm - float(np.mean(xm))
    elif mode == "linear":
        if n_valid < 3:
            out[m] = xm - float(np.mean(xm))
        else:
            t = np.flatnonzero(m).astype(np.float64)
            tc = t - float(np.mean(t))
            denom = float(np.dot(tc, tc))
            if denom <= 1e-12:
                out[m] = xm - float(np.mean(xm))
            else:
                slope = float(np.dot(tc, xm - float(np.mean(xm)))) / denom
                intercept = float(np.mean(xm)) - slope * float(np.mean(t))
                out[m] = xm - (intercept + slope * t)
    else:
        raise ValueError(f"Unsupported detrend mode: {mode!r}")
    return out


def masked_unbiased_autocovariance(
    values: np.ndarray,
    mask: np.ndarray,
    max_lag: int,
    *,
    min_pair_count: int = 24,
    min_overlap_frac: float = 0.15,
    detrend: str = "linear",
    bias: str = "unbiased",
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FFT-accelerated masked autocovariance / normalized autocorrelation.

    Follows ``dsp-slides.pdf`` (slides 150-154, pp. 186-190):
    * Subtracts mean/linear trend first so autocovariance ``c_xx(k)`` is not dominated by DC energy.
    * Uses exact co-observed overlap count ``M_k = sum_n mask[n] * mask[n + k]`` as divisor when
      ``bias == 'unbiased'`` (generalizing ``1 / (N - k)`` to wear-masked time series), eliminating
      finite-sample triangular tapering that otherwise shrinks 24h-72h circadian lags.
    * Guards tail instability with ``M_k >= max(min_pair_count, min_overlap_frac * N_valid)``.
    """
    n = len(values)
    max_k = int(min(max_lag, max(0, n - 1)))
    acf = np.full(max_k + 1, np.nan, dtype=np.float64)
    counts = np.zeros(max_k + 1, dtype=np.float64)
    if n < 4 or max_k < 1:
        return acf, counts

    m = np.asarray(mask, dtype=bool).ravel() & np.isfinite(values)
    n_valid = int(m.sum())
    if n_valid < max(4, min_pair_count):
        return acf, counts

    xd = detrend_observed(values, m, mode=detrend)
    xz = np.where(m, xd, 0.0)
    mf = m.astype(np.float64)

    nfft = int(next_fast_len(2 * n - 1))
    fx = np.fft.rfft(xz, n=nfft)
    fm = np.fft.rfft(mf, n=nfft)
    num = np.fft.irfft(fx * np.conj(fx), n=nfft)[: max_k + 1]
    cnt = np.fft.irfft(fm * np.conj(fm), n=nfft)[: max_k + 1]
    cnt = np.rint(np.maximum(cnt, 0.0))
    counts[:] = cnt

    c0 = float(num[0] / max(cnt[0], 1.0)) if bias == "unbiased" else float(num[0] / n_valid)
    if not np.isfinite(c0) or c0 <= 1e-12:
        acf[0] = 1.0 if normalize else 0.0
        return acf, counts

    req = max(int(min_pair_count), int(np.ceil(float(min_overlap_frac) * n_valid)))
    valid_lags = cnt >= req
    if bias == "unbiased":
        raw = np.where(cnt > 0, num / np.maximum(cnt, 1.0), np.nan)
    else:
        raw = num / float(n_valid)

    if normalize:
        normed = raw / c0
        normed = np.where(valid_lags, np.clip(normed, -1.0, 1.0), np.nan)
        acf[:] = normed
        acf[0] = 1.0
    else:
        acf[:] = np.where(valid_lags, raw, np.nan)
    return acf, counts


def find_fundamental_acf_period(
    acf: np.ndarray,
    fs_hz: float,
    period_band_hours: Tuple[float, float] = (18.0, 30.0),
    fundamental_ratio: float = 0.85,
    min_strength: float = 0.08,
) -> Dict[str, float]:
    """Port of ``cr7_kimmitch`` ``PpgHr.swift`` fundamental-period selection & sub-lag interpolation.

    Prefers the shortest local-max lag inside ``period_band_hours`` achieving at least
    ``fundamental_ratio * max_peak`` (so 48h/72h harmonics never steal the ~24h circadian peak),
    then refines the discrete lag via Variant-A 3-point parabolic fit:
    ``delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)``.
    """
    out = {"period_h": np.nan, "peak_acf": np.nan, "concavity": np.nan, "usable": 0.0}
    if fs_hz <= 0 or len(acf) < 6:
        return out
    sec_per_bin = 1.0 / float(fs_hz)
    bins_per_hour = 3600.0 / sec_per_bin

    lo_lag = max(2, int(np.floor(float(period_band_hours[0]) * bins_per_hour)))
    hi_lag = min(len(acf) - 2, int(np.ceil(float(period_band_hours[1]) * bins_per_hour)))
    if hi_lag <= lo_lag:
        return out

    band_slice = acf[lo_lag - 1 : hi_lag + 2]
    if not np.isfinite(band_slice).any():
        return out

    peaks: List[int] = []
    for k in range(lo_lag, hi_lag + 1):
        y0, y1, y2 = acf[k - 1], acf[k], acf[k + 1]
        if np.isfinite([y0, y1, y2]).all() and y1 > y0 and y1 >= y2 and y1 > 0:
            peaks.append(k)

    if peaks:
        max_pk = max(float(acf[p]) for p in peaks)
        thresh = float(fundamental_ratio) * max_pk
        chosen = min(p for p in peaks if float(acf[p]) >= thresh)
    else:
        sub = acf[lo_lag : hi_lag + 1]
        if not np.isfinite(sub).any():
            return out
        chosen = lo_lag + int(np.nanargmax(sub))

    y1 = float(acf[chosen])
    if not np.isfinite(y1) or y1 < float(min_strength):
        out["peak_acf"] = y1 if np.isfinite(y1) else np.nan
        return out

    delta = 0.0
    concavity = 0.0
    if 0 < chosen < len(acf) - 1:
        y0, y2 = float(acf[chosen - 1]), float(acf[chosen + 1])
        if np.isfinite(y0) and np.isfinite(y2):
            denom = y0 - 2.0 * y1 + y2
            concavity = max(0.0, 2.0 * y1 - y0 - y2)
            if denom < -1e-12:
                delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))

    refined_bins = float(chosen) + delta
    out["period_h"] = float(refined_bins / bins_per_hour)
    out["peak_acf"] = float(np.clip(y1, -1.0, 1.0))
    out["concavity"] = float(concavity)
    out["usable"] = 1.0
    return out


def fit_circadian_cosinor_24h(
    t_hours: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    min_bins: int = 48,
) -> Dict[str, float]:
    """Port of ``cr7_kimmitch`` ``CircadianEngine.swift`` single-component 24h OLS Cosinor fit.

    Fits ``y(t) = M + beta * cos(2*pi*t/24) + gamma * sin(2*pi*t/24)`` strictly over observed
    worn samples:
    * ``amplitude = hypot(beta, gamma)``
    * ``rel_amplitude = amplitude / max(|M|, 1e-6)``
    * ``acrophase_h = (atan2(gamma, beta) mod 2*pi) * 24 / (2*pi)``
    """
    out = {
        "mesor": np.nan,
        "amplitude": np.nan,
        "rel_amplitude": np.nan,
        "acrophase_h": np.nan,
        "acrophase_sin": np.nan,
        "acrophase_cos": np.nan,
        "r2": np.nan,
    }
    m = np.asarray(mask, dtype=bool).ravel() & np.isfinite(values) & np.isfinite(t_hours)
    if int(m.sum()) < int(min_bins):
        return out

    th = np.asarray(t_hours[m], dtype=np.float64)
    y = np.asarray(values[m], dtype=np.float64)
    omega = 2.0 * np.pi / 24.0
    phi = omega * th
    X = np.column_stack([np.ones_like(phi), np.cos(phi), np.sin(phi)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    mesor, beta, gamma = float(coef[0]), float(coef[1]), float(coef[2])
    amp = float(np.hypot(beta, gamma))
    rel_amp = float(amp / max(abs(mesor), 1e-6))

    angle = float(np.arctan2(gamma, beta)) % (2.0 * np.pi)
    acro_h = float(angle * 24.0 / (2.0 * np.pi))

    yhat = X @ coef
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    r2 = float(1.0 - np.sum(np.square(y - yhat)) / ss_tot) if ss_tot > 1e-12 else 0.0

    out["mesor"] = mesor
    out["amplitude"] = amp
    out["rel_amplitude"] = rel_amp
    out["acrophase_h"] = acro_h
    out["acrophase_sin"] = float(np.sin(angle))
    out["acrophase_cos"] = float(np.cos(angle))
    out["r2"] = float(np.clip(r2, 0.0, 1.0))
    return out


def compute_welch_circadian_spectrum(
    values: np.ndarray,
    mask: np.ndarray,
    fs_hz: float,
    max_gap_bins: int = 4,
    bands_hours: Tuple[Tuple[float, float], ...] = ((16.0, 36.0), (4.0, 16.0), (1.0, 4.0)),
    nperseg_days: int = 4,
) -> Dict[str, float]:
    """Hann-windowed 50%-overlap Welch PSD (``dsp-slides.pdf`` slide 163 / p. 199) on worn runs.

    Respects Nyquist bounds (``hi <= fs_hz / 2``) and Parseval variance conservation across
    configurable period bands (default: circadian 16-36h, ultradian 4-16h, bout 1-4h).
    """
    out = {
        "welch_circadian_frac": np.nan,
        "welch_ultradian_frac": np.nan,
        "welch_bout_frac": np.nan,
        "welch_entropy": np.nan,
        "welch_peak_period_h": np.nan,
        "welch_parseval_ratio": np.nan,
    }
    filled, valid = bounded_gap_fill(values, mask, max_gap=max_gap_bins)
    runs = contiguous_true_runs(valid)
    if not runs or fs_hz <= 0:
        return out
    best_s, best_e = max(runs, key=lambda r: r[1] - r[0])
    seg = filled[best_s:best_e]
    n_seg = len(seg)
    bins_per_day = int(round(86400.0 * fs_hz))
    if n_seg < max(32, min(bins_per_day, 64)):
        return out

    seg_zm = seg - float(np.mean(seg))
    var_time = float(np.mean(np.square(seg_zm)))
    if var_time <= 1e-12:
        return out

    nperseg = min(n_seg, max(32, min(max(1, int(nperseg_days)) * bins_per_day, n_seg)))
    noverlap = nperseg // 2
    freqs, pxx = sps.welch(
        seg_zm,
        fs=float(fs_hz),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    if len(freqs) < 3:
        return out

    df = float(freqs[1] - freqs[0])
    total_power = float(np.sum(pxx) * df)
    if total_power <= 1e-15:
        return out

    nyq = 0.5 * float(fs_hz)
    names = ("welch_circadian_frac", "welch_ultradian_frac", "welch_bout_frac")
    for idx_b, bname in enumerate(names):
        if idx_b < len(bands_hours):
            lo_h, hi_h = map(float, bands_hours[idx_b])
            f_lo = 1.0 / (max(hi_h, 1e-3) * 3600.0)
            f_hi = min(nyq, 1.0 / (max(lo_h, 1e-3) * 3600.0))
            if f_hi > f_lo:
                bm = (freqs >= f_lo) & (freqs <= f_hi)
                out[bname] = float(np.sum(pxx[bm]) * df / total_power) if bm.any() else 0.0
            else:
                out[bname] = 0.0

    p_prob = pxx[1:] / max(float(np.sum(pxx[1:])), 1e-15)
    p_pos = p_prob[p_prob > 0]
    entropy = float(-np.sum(p_pos * np.log(p_pos)) / max(np.log(len(p_prob)), 1.0)) if len(p_prob) > 1 else 0.0

    peak_idx = 1 + int(np.argmax(pxx[1:]))
    peak_freq = float(freqs[peak_idx])
    peak_period_h = float(1.0 / (peak_freq * 3600.0)) if peak_freq > 1e-10 else np.nan

    out["welch_entropy"] = float(np.clip(entropy, 0.0, 1.0))
    out["welch_peak_period_h"] = float(np.clip(peak_period_h, 0.0, 200.0)) if np.isfinite(peak_period_h) else np.nan
    out["welch_parseval_ratio"] = float(total_power / var_time)
    return out


# ---------------------------------------------------------------------------
# OpenStrap/analytics (van Hees SIB, Phillips SRI, Circular SJL, PRSA, CUSUM)
# & mad-lab-fau/sleep_analysis (Cole-Kripke, Sadeh, Webster Rescoring)
# ---------------------------------------------------------------------------

def calibrate_g_ref(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    worn_mask: np.ndarray,
    epoch_samples: int = 12,
) -> float:
    """Self-calibrate gravitational reference norm ``g_ref`` from lowest-variance static epochs.

    Mirrors ``OpenStrap/analytics`` ``calibrateGRef`` (``enmo.dart``): identifies worn
    1-minute (12 x 5s) stationary windows near 1g with minimal triaxial standard deviation
    and returns the median vector magnitude, safely bounded in ``[0.85, 1.15]``.
    """
    m = np.asarray(worn_mask, dtype=bool)
    n = len(x)
    w = max(4, int(epoch_samples))
    n_blocks = n // w
    if n_blocks < 5:
        return 1.0
    lim = n_blocks * w
    xb = np.asarray(x[:lim], dtype=np.float64).reshape(n_blocks, w)
    yb = np.asarray(y[:lim], dtype=np.float64).reshape(n_blocks, w)
    zb = np.asarray(z[:lim], dtype=np.float64).reshape(n_blocks, w)
    mb = m[:lim].reshape(n_blocks, w).all(axis=1)
    if not np.isfinite(xb).all() or not np.isfinite(yb).all() or not np.isfinite(zb).all():
        mb &= np.isfinite(xb).all(axis=1) & np.isfinite(yb).all(axis=1) & np.isfinite(zb).all(axis=1)
    if mb.sum() < 5:
        return 1.0

    mag = np.sqrt(xb[mb] ** 2 + yb[mb] ** 2 + zb[mb] ** 2)
    block_mag = np.median(mag, axis=1)
    tri_std = np.sqrt(np.var(xb[mb], axis=1) + np.var(yb[mb], axis=1) + np.var(zb[mb], axis=1))
    valid_g = (block_mag >= 0.85) & (block_mag <= 1.15) & (tri_std < 0.025)
    if valid_g.sum() < 3:
        valid_g = (block_mag >= 0.85) & (block_mag <= 1.15)
    if not valid_g.any():
        return 1.0
    sub_std = tri_std[valid_g]
    q25 = float(np.percentile(sub_std, 25))
    static_mags = block_mag[valid_g][sub_std <= max(q25, 1e-4)]
    if len(static_mags) == 0:
        return 1.0
    return float(np.clip(np.median(static_mags), 0.85, 1.15))


def _apply_webster_rescoring(sleep_min: np.ndarray, valid_min: np.ndarray) -> np.ndarray:
    """Apply Webster (1982) 5 morphological sleep-wake rescoring rules at 60s resolution.

    Grounded in ``mad-lab-fau/sleep_analysis`` (``SleepWakeDetection`` with ``rescore_data=True``),
    applied strictly inside contiguous valid worn segments so missing runs are never bridged.
    """
    out = np.asarray(sleep_min, dtype=bool).copy()
    v = np.asarray(valid_min, dtype=bool)
    for s, e in contiguous_true_runs(v):
        seg = out[s:e].copy()
        n = len(seg)
        if n < 15:
            continue

        # Rule (d) & (e): short sleep bouts surrounded by longer wake periods -> Wake
        def _rescore_short_bouts(arr: np.ndarray, max_sleep_len: int, min_wake_pad: int) -> None:
            for rs, re in contiguous_true_runs(arr):
                if (re - rs) <= max_sleep_len:
                    left_ok = rs >= min_wake_pad and not arr[rs - min_wake_pad : rs].any()
                    right_ok = (n - re) >= min_wake_pad and not arr[re : re + min_wake_pad].any()
                    if left_ok and right_ok:
                        arr[rs:re] = False

        # Rules (a), (b), (c): >= W min preceding wake forces next S min sleep -> Wake
        def _rescore_after_wake(arr: np.ndarray, wake_run: int, sleep_cap: int) -> None:
            i = 0
            while i < n:
                if not arr[i]:
                    w_start = i
                    while i < n and not arr[i]:
                        i += 1
                    if (i - w_start) >= wake_run:
                        end_cap = min(n, i + sleep_cap)
                        arr[i:end_cap] = False
                else:
                    i += 1

        _rescore_after_wake(seg, 4, 1)
        _rescore_after_wake(seg, 10, 3)
        _rescore_after_wake(seg, 15, 4)
        _rescore_short_bouts(seg, 6, 10)
        _rescore_short_bouts(seg, 10, 20)
        out[s:e] = seg
    return out


def compute_fau_1min_sleep_heuristics(
    enmo_raw: np.ndarray,
    worn_raw: np.ndarray,
    dt_s: float = RAW_ACTIGRAPHY_DT_S,
    scale_g_to_counts: float = 1500.0,
) -> Dict[str, np.ndarray]:
    """Compute Cole-Kripke (1992), Sadeh (1994), and Webster (1982) 1-min sleep flags.

    Inspired by ``mad-lab-fau/sleep_analysis`` (``cole_kripke.py``, ``sadeh.py``,
    ``webster.py``). Converts calibrated 5s ENMO into 60-second epochs on contiguous
    worn intervals and evaluates published sleep/wake decision equations.
    """
    n_raw = len(enmo_raw)
    step = max(1, int(round(60.0 / max(dt_s, 1e-3))))
    n_min = (n_raw + step - 1) // step
    idx_min = np.arange(n_raw, dtype=np.int64) // step

    w_ok = np.asarray(worn_raw, dtype=bool) & np.isfinite(enmo_raw)
    cnt_ok = np.bincount(idx_min, weights=w_ok.astype(np.float64), minlength=n_min)
    valid_min = cnt_ok >= max(1.0, 0.5 * step)
    sum_enmo = np.bincount(idx_min, weights=np.where(w_ok, enmo_raw, 0.0), minlength=n_min)
    mean_enmo_min = np.where(valid_min, sum_enmo / np.maximum(cnt_ok, 1.0), np.nan)
    act_counts = np.where(valid_min, np.clip(mean_enmo_min * float(scale_g_to_counts), 0.0, 2000.0), np.nan)

    cole_sleep = np.zeros(n_min, dtype=bool)
    sadeh_sleep = np.zeros(n_min, dtype=bool)

    # Evaluate strictly inside contiguous valid 1-min runs (never bridge missing intervals)
    ck_kernel = np.array([67, 74, 230, 76, 58, 54, 106], dtype=np.float64) * 0.001  # reversed for conv: +2..0..-4
    for s, e in contiguous_true_runs(valid_min):
        seg = act_counts[s:e]
        if len(seg) < 12:
            continue
        # Cole-Kripke (1992): D = P * sum w_k * A_{t+k} < 1.0 -> Sleep (P=1.0 -> D = 0.001 * weighted sum)
        padded_ck = np.pad(seg, (4, 2), mode="edge")
        d_val = np.convolve(padded_ck, ck_kernel, mode="valid")
        # Align convolution output of length len(seg) so index i covers [i-4..i+2]
        cole_sleep[s:e] = d_val[: len(seg)] < 1.0

        # Sadeh (1994) 4-term pediatric/adult logistic score on 11-min centered & 6-min trailing stats
        sr = pd.Series(seg)
        mu_5 = sr.rolling(11, center=True, min_periods=1).mean().to_numpy()
        sd_6 = sr.rolling(6, center=False, min_periods=1).std(ddof=0).fillna(0.0).to_numpy()
        nat_6 = ((sr >= 50.0) & (sr < 100.0)).rolling(11, center=True, min_periods=1).sum().to_numpy()
        loc_act = np.log1p(np.maximum(seg, 0.0))
        ps = 7.601 - 0.065 * mu_5 - 1.08 * nat_6 - 0.056 * sd_6 - 0.703 * loc_act
        sadeh_sleep[s:e] = ps >= 0.0

    webster_sleep = _apply_webster_rescoring(sadeh_sleep, valid_min)

    return {
        "valid_min": valid_min,
        "cole_kripke_raw": cole_sleep[idx_min] & w_ok,
        "sadeh_raw": sadeh_sleep[idx_min] & w_ok,
        "webster_raw": webster_sleep[idx_min] & w_ok,
    }


def compute_phillips_sri(
    sleep_mask: np.ndarray,
    valid_mask: np.ndarray,
    bins_per_day: int,
) -> Dict[str, float]:
    """Compute Phillips (2017) Sleep Regularity Index (``OpenStrap/analytics`` ``sri.dart``).

    Evaluates 24-hour epoch-by-epoch sleep/wake state agreement across co-observed pairs
    separated by exact ``bins_per_day`` without fabricating sleep/wake in gaps.
    """
    out = {"sri": np.nan, "sri_pair_std": np.nan, "sri_cases": 0.0}
    sm = np.asarray(sleep_mask, dtype=bool).ravel()
    vm = np.asarray(valid_mask, dtype=bool).ravel()
    bpd = int(bins_per_day)
    if bpd < 4 or len(sm) <= bpd:
        return out

    co_obs = vm[:-bpd] & vm[bpd:]
    cases = int(co_obs.sum())
    out["sri_cases"] = float(cases)
    if cases < max(12, bpd // 2):
        return out

    agree = (sm[:-bpd] == sm[bpd:])
    sri = float(np.clip(200.0 * np.mean(agree[co_obs]) - 100.0, -100.0, 100.0))
    out["sri"] = sri

    # Day-to-day transition stability across valid 24h blocks
    n_days = len(co_obs) // bpd
    day_sris: List[float] = []
    for d in range(n_days):
        sl = slice(d * bpd, (d + 1) * bpd)
        if int(co_obs[sl].sum()) >= max(8, bpd // 4):
            day_sris.append(float(200.0 * np.mean(agree[sl][co_obs[sl]]) - 100.0))
    out["sri_pair_std"] = float(np.std(day_sris)) if len(day_sris) >= 2 else (0.0 if len(day_sris) == 1 else np.nan)
    return out


def _fisher_circular_stats(hours: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Return ``(circular_mean_hour in [0, 24), resultant_length R in [0, 1])``."""
    h = np.asarray(hours, dtype=np.float64)
    w = np.ones_like(h) if weights is None else np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    ok = np.isfinite(h) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan, np.nan
    theta = 2.0 * np.pi * (h[ok] % 24.0) / 24.0
    ww = w[ok]
    w_sum = float(np.sum(ww))
    c = float(np.sum(ww * np.cos(theta)) / w_sum)
    s = float(np.sum(ww * np.sin(theta)) / w_sum)
    R = float(np.clip(np.hypot(c, s), 0.0, 1.0))
    if R < 1e-8:
        return np.nan, 0.0
    mean_h = float((np.arctan2(s, c) % (2.0 * np.pi)) * 24.0 / (2.0 * np.pi))
    return mean_h, R


def compute_circular_sleep_regularity(
    sib_vals: np.ndarray,
    mask: np.ndarray,
    hour_of_day: np.ndarray,
    weekday: np.ndarray,
    bin_minutes: int,
) -> Dict[str, float]:
    """Fisher circular sleep midpoint regularity & Roenneberg Social Jetlag (``OpenStrap/analytics``).

    Groups observations into shifted noon-to-noon biological days so nocturnal sleep is never
    split across midnight, extracts sleep-weighted circular midpoints, and measures signed and
    absolute free-vs-workday phase shift in ``(-12, 12]`` hours plus weekend oversleep.
    """
    out = {
        "circ_sleep_mid_h": np.nan,
        "circ_sleep_mid_R": np.nan,
        "circ_sleep_mid_sd_h": np.nan,
        "circ_sleep_dur_h": np.nan,
        "circ_sjl_signed_h": np.nan,
        "circ_sjl_abs_h": np.nan,
        "circ_wknd_oversleep_h": np.nan,
    }
    m = np.asarray(mask, dtype=bool).ravel()
    sib = np.asarray(sib_vals, dtype=np.float64).ravel()
    hod = np.asarray(hour_of_day, dtype=np.float64).ravel()
    wd = np.asarray(weekday, dtype=np.float64).ravel()
    bin_h = float(max(bin_minutes, 1)) / 60.0
    bins_per_day = max(4, int(round(24.0 / bin_h)))

    if len(m) < bins_per_day or not np.isfinite(sib[m]).any():
        return out

    # Track noon-to-noon cycles using unwrapped cumulative elapsed hours anchored to hod[0]
    t_cum_h = float(hod[0] if np.isfinite(hod[0]) else 0.0) + np.arange(len(m), dtype=np.float64) * bin_h
    noon_day_idx = np.floor((t_cum_h - 12.0) / 24.0).astype(np.int64)

    mids, durs, is_free_day = [], [], []
    for d in np.unique(noon_day_idx):
        dm = m & (noon_day_idx == d) & np.isfinite(sib)
        if int(dm.sum()) < max(8, bins_per_day // 2):
            continue
        # Weight rest bins toward nighttime/main rest bout (sib >= 0.35)
        w_sleep = np.where(sib[dm] >= 0.35, sib[dm], 0.0)
        sleep_h = float(np.sum(np.clip(sib[dm], 0.0, 1.0)) * bin_h)
        if float(np.sum(w_sleep)) <= 0.5:
            continue
        mid_h, _ = _fisher_circular_stats(hod[dm], w_sleep)
        if not np.isfinite(mid_h):
            continue
        mids.append(mid_h)
        durs.append(sleep_h)
        wd_vals = wd[dm & np.isfinite(wd)]
        if len(wd_vals) > 0:
            # Determine morning-wake day (Saturday/Sunday = free day >= 6)
            night_end_wd = wd_vals[-1]
            is_free_day.append(bool(night_end_wd >= 6))
        else:
            is_free_day.append(None)  # type: ignore[arg-type]

    if not mids:
        return out

    mid_arr = np.asarray(mids, dtype=np.float64)
    dur_arr = np.asarray(durs, dtype=np.float64)
    mean_mid, R = _fisher_circular_stats(mid_arr)
    out["circ_sleep_mid_h"] = mean_mid
    out["circ_sleep_mid_R"] = R
    if np.isfinite(R):
        sd_rad = float(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-4)))))
        out["circ_sleep_mid_sd_h"] = float(min(12.0, sd_rad * 24.0 / (2.0 * np.pi)))
    out["circ_sleep_dur_h"] = float(np.mean(dur_arr))

    work_idx = [i for i, f in enumerate(is_free_day) if f is False]
    free_idx = [i for i, f in enumerate(is_free_day) if f is True]
    if work_idx and free_idx:
        mid_work, _ = _fisher_circular_stats(mid_arr[work_idx])
        mid_free, _ = _fisher_circular_stats(mid_arr[free_idx])
        if np.isfinite(mid_work) and np.isfinite(mid_free):
            # Shortest signed circular arc in (-12, 12]
            signed_sjl = float(((mid_free - mid_work + 12.0) % 24.0) - 12.0)
            if signed_sjl <= -12.0:
                signed_sjl += 24.0
            out["circ_sjl_signed_h"] = signed_sjl
            out["circ_sjl_abs_h"] = abs(signed_sjl)
        out["circ_wknd_oversleep_h"] = float(np.percentile(dur_arr[free_idx], 75) - np.median(dur_arr[work_idx]))
    return out


def compute_prsa_capacity(
    values: np.ndarray,
    mask: np.ndarray,
    l: int = 2,
    anchor_ratio_cap: float = 0.5,
) -> Dict[str, float]:
    """Bauer (2006) Phase-Rectified Signal Averaging ``DC`` & ``AC`` capacity (``OpenStrap/analytics``).

    Computes exact Haar step contrast ``(mean forward[0..L-1] - mean backward[-L..-1]) / 2``
    over gap-free upward (``DC``) and downward (``AC``) anchor points filtered for extreme jumps.
    """
    out = {
        "prsa_dc": np.nan,
        "prsa_ac": np.nan,
        "prsa_asym": np.nan,
        "prsa_dc_anchors": 0.0,
        "prsa_ac_anchors": 0.0,
    }
    x = np.asarray(values, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel() & np.isfinite(x)
    L = max(1, int(l))
    if int(m.sum()) < 2 * L + 2:
        return out

    obs = x[m]
    iqr_scale = max(float(np.percentile(obs, 95) - np.percentile(obs, 5)), 1e-6)
    max_step = max(float(anchor_ratio_cap) * iqr_scale * 2.0, 1e-6)

    dc_vals: List[np.ndarray] = []
    ac_vals: List[np.ndarray] = []
    for s, e in contiguous_true_runs(m):
        seg = x[s:e]
        if len(seg) < 2 * L + 1:
            continue
        # Evaluate anchor positions i in [L .. len(seg) - L]
        cs = np.concatenate([[0.0], np.cumsum(seg)])
        idx = np.arange(L, len(seg) - L + 1)
        fwd = (cs[idx + L] - cs[idx]) / float(L)
        bwd = (cs[idx] - cs[idx - L]) / float(L)
        haar = 0.5 * (fwd - bwd)  # for L=2 -> [X(0)+X(1)-X(-1)-X(-2)] / 4
        dx = seg[idx] - seg[idx - 1]
        clean_jump = np.abs(dx) <= max_step
        up = (dx > 0.0) & clean_jump
        dn = (dx < 0.0) & clean_jump
        if up.any():
            dc_vals.append(haar[up])
        if dn.any():
            ac_vals.append(haar[dn])

    if dc_vals:
        dc_all = np.concatenate(dc_vals)
        out["prsa_dc"] = float(np.mean(dc_all))
        out["prsa_dc_anchors"] = float(len(dc_all))
    if ac_vals:
        ac_all = np.concatenate(ac_vals)
        out["prsa_ac"] = float(np.mean(ac_all))
        out["prsa_ac_anchors"] = float(len(ac_all))
    if np.isfinite(out["prsa_dc"]) and np.isfinite(out["prsa_ac"]):
        dc_a, ac_a = abs(out["prsa_dc"]), abs(out["prsa_ac"])
        out["prsa_asym"] = float((dc_a - ac_a) / max(dc_a + ac_a, 1e-9))
    return out


def compute_cusum_volatility(
    values: np.ndarray,
    mask: np.ndarray,
    bins_per_day: int,
    k: float = 0.5,
    h: float = 3.0,
) -> Dict[str, float]:
    """Two-sided Page CUSUM volatility detector standardized by robust median/MAD (``OpenStrap/analytics``).

    Accumulators reset across every non-wear boundary so missingness never fabricates jumps.
    """
    out = {"cusum_max": np.nan, "cusum_count": 0.0, "cusum_frac": np.nan}
    x = np.asarray(values, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel() & np.isfinite(x)
    n_obs = int(m.sum())
    if n_obs < max(12, min(int(bins_per_day), 24)):
        return out

    obs = x[m]
    med = float(np.median(obs))
    mad = float(np.median(np.abs(obs - med)))
    sigma = max(1.4826 * mad, float(np.std(obs)) * 0.25, 1e-4)

    s_pos, s_neg = 0.0, 0.0
    max_s = 0.0
    alarm_steps = 0
    kf, hf = float(k), float(h)

    for s, e in contiguous_true_runs(m):
        s_pos = 0.0
        s_neg = 0.0
        for val in x[s:e]:
            z = (float(val) - med) / sigma
            s_pos = max(0.0, s_pos + z - kf)
            s_neg = max(0.0, s_neg - z - kf)
            cur = max(s_pos, s_neg)
            if cur > max_s:
                max_s = cur
            if cur > hf:
                alarm_steps += 1

    out["cusum_max"] = float(max_s)
    out["cusum_count"] = float(alarm_steps)
    out["cusum_frac"] = float(alarm_steps / max(n_obs, 1))
    return out


# ---------------------------------------------------------------------------
# Actigraphy Parquet Loading & Anti-Aliased Regular Grid Binning
# ---------------------------------------------------------------------------

_PARQUET_NEEDED_COLS = [
    "X", "Y", "Z", "enmo", "anglez", "non-wear_flag", "light",
    "time_of_day", "weekday", "relative_date_PCIAT",
]


def _load_parquet_arrays(parquet_path_or_df: Any) -> Optional[Dict[str, np.ndarray]]:
    if parquet_path_or_df is None or (isinstance(parquet_path_or_df, float) and np.isnan(parquet_path_or_df)):
        return None
    if isinstance(parquet_path_or_df, pd.DataFrame):
        df = parquet_path_or_df
    else:
        p = Path(str(parquet_path_or_df))
        if not p.exists():
            return None
        schema_names = set(pq.read_schema(p).names)
        cols = [c for c in _PARQUET_NEEDED_COLS if c in schema_names]
        df = pq.read_table(p, columns=cols).to_pandas()
    if df is None or len(df) == 0:
        return None
    return {col: df[col].to_numpy(dtype=np.float64, copy=False) for col in df.columns}


def bin_actigraphy_series(
    parquet_path_or_df: Any,
    *,
    bin_minutes: int = 15,
    resample_method: str = "anti_alias_mean",
    non_wear_policy: str = "mask",
    min_bin_wear_frac: float = 0.25,
    clip_enmo: float = 4.0,
) -> Optional[Dict[str, Any]]:
    """Aggregate 5-second actigraphy parquet onto a uniform multi-day grid.

    * Never collapses missing non-wear days onto adjacent worn timestamps (mirrors ``SleepDebt.swift``).
    * When ``resample_method == 'anti_alias_mean'``, runs zero-phase 4th-order Butterworth lowpass
      (`cutoff = 0.45 / bin_seconds < 0.1 Hz Nyquist`) over contiguous worn runs before downsampling.
    """
    res_meth = "mean" if (resample_method is None or str(resample_method).strip().lower() == "none") else str(resample_method)
    nw_pol = "keep_all" if (non_wear_policy is None or str(non_wear_policy).strip().lower() == "none") else str(non_wear_policy)
    clip_val = None if (clip_enmo is None or float(clip_enmo) <= 0.0) else float(clip_enmo)

    cache_key: Optional[Tuple[Any, ...]] = None
    if isinstance(parquet_path_or_df, (str, Path)):
        cache_key = (
            str(parquet_path_or_df),
            int(bin_minutes),
            res_meth,
            nw_pol,
            round(float(min_bin_wear_frac), 4),
            round(clip_val, 4) if clip_val is not None else None,
        )
        if cache_key in _GRID_CACHE:
            return _GRID_CACHE[cache_key]

    arrs = _load_parquet_arrays(parquet_path_or_df)
    if arrs is None:
        if cache_key is not None:
            _GRID_CACHE[cache_key] = None
        return None

    n_raw = len(arrs["enmo"])
    if n_raw < 12:
        if cache_key is not None:
            _GRID_CACHE[cache_key] = None
        return None

    rel_day = arrs.get("relative_date_PCIAT")
    tod = arrs.get("time_of_day")
    if rel_day is None or tod is None:
        t_raw_sec = np.arange(n_raw, dtype=np.float64) * RAW_ACTIGRAPHY_DT_S
    else:
        t_raw_sec = rel_day * SECONDS_PER_DAY + tod / 1e9

    finite_t = np.isfinite(t_raw_sec)
    if not finite_t.any():
        if cache_key is not None:
            _GRID_CACHE[cache_key] = None
        return None
    if not finite_t.all():
        t_raw_sec = np.interp(np.arange(n_raw), np.flatnonzero(finite_t), t_raw_sec[finite_t])

    # Sort defensively if timestamps have any non-monotonicity
    if np.any(np.diff(t_raw_sec) < 0):
        order = np.argsort(t_raw_sec, kind="mergesort")
        t_raw_sec = t_raw_sec[order]
        arrs = {k: v[order] for k, v in arrs.items()}

    raw_enmo = np.nan_to_num(arrs["enmo"], nan=np.nan)
    enmo = np.maximum(raw_enmo, 0.0) if clip_val is None else np.clip(raw_enmo, 0.0, clip_val)
    anglez = arrs.get("anglez", np.zeros(n_raw, dtype=np.float64))
    light = np.maximum(0.0, arrs.get("light", np.zeros(n_raw, dtype=np.float64)))
    non_wear = arrs.get("non-wear_flag", np.zeros(n_raw, dtype=np.float64))
    weekday = arrs.get("weekday", np.zeros(n_raw, dtype=np.float64))

    X = arrs.get("X", np.zeros(n_raw, dtype=np.float64))
    Y = arrs.get("Y", np.zeros(n_raw, dtype=np.float64))
    Z = arrs.get("Z", np.zeros(n_raw, dtype=np.float64))
    has_triaxial = bool(np.any((X != 0.0) | (Y != 0.0) | (Z != 0.0)))
    vec_mag = np.sqrt(X * X + Y * Y + Z * Z)
    mag_dev = np.abs(vec_mag - 1.0)
    # Discrete sample-to-sample vector jerk magnitude (g / 5s -> g/s via RAW_ACTIGRAPHY_FS_HZ)
    dx = np.diff(X, prepend=X[0])
    dy = np.diff(Y, prepend=Y[0])
    dz = np.diff(Z, prepend=Z[0])
    jerk_raw = np.sqrt(dx * dx + dy * dy + dz * dz) * RAW_ACTIGRAPHY_FS_HZ

    if nw_pol == "mask":
        worn_raw = np.isfinite(enmo) & (np.nan_to_num(non_wear, nan=1.0) == 0.0)
    elif nw_pol == "keep_all":
        worn_raw = np.isfinite(enmo)
    elif nw_pol in {"impute_zero", "impute_diurnal"}:
        if nw_pol == "impute_zero":
            enmo = np.where(np.nan_to_num(non_wear, nan=0.0) == 0.0, enmo, 0.0)
            worn_raw = np.isfinite(enmo)
        else:
            worn_raw = np.isfinite(enmo) & (np.nan_to_num(non_wear, nan=1.0) == 0.0)
    else:
        raise ValueError(f"Unsupported non_wear_policy: {non_wear_policy!r}")

    # OpenStrap/analytics high-res 5s kinematics & sleep heuristics PRIOR to bin lowpass:
    # 1) Self-calibrated g_ref, 15s trailing vector-highpass dynAmp, spherical rotation & pitch uprightness
    g_ref_cal = calibrate_g_ref(X, Y, Z, worn_raw) if has_triaxial else 1.0
    dyn_amp_raw = np.full(n_raw, np.nan, dtype=np.float64)
    rot_deg_raw = np.full(n_raw, np.nan, dtype=np.float64)
    upright_raw = np.full(n_raw, np.nan, dtype=np.float64)

    if has_triaxial:
        vm_safe = np.maximum(vec_mag, 1e-6)
        ux = np.where(worn_raw, X / vm_safe, np.nan)
        uy = np.where(worn_raw, Y / vm_safe, np.nan)
        uz = np.where(worn_raw, Z / vm_safe, np.nan)
        pitch_deg = np.arctan2(-X, np.sqrt(Y * Y + Z * Z)) * (180.0 / np.pi)
        upright_raw = np.where(worn_raw, (np.abs(pitch_deg) > 60.0).astype(np.float64), np.nan)

        dot_adj = np.clip(ux[1:] * ux[:-1] + uy[1:] * uy[:-1] + uz[1:] * uz[:-1], -1.0, 1.0)
        pair_ok = worn_raw[1:] & worn_raw[:-1] & np.isfinite(dot_adj)
        rot_deg_raw[1:] = np.where(pair_ok, np.arccos(dot_adj) * (180.0 / np.pi), np.nan)

        win_dyn = max(2, int(round(15.0 / RAW_ACTIGRAPHY_DT_S)))
        for s, e in contiguous_true_runs(worn_raw & np.isfinite(vec_mag)):
            seg_xyz = np.column_stack([X[s:e], Y[s:e], Z[s:e]])
            cs = np.vstack([np.zeros((1, 3), dtype=np.float64), np.cumsum(seg_xyz, axis=0)])
            idx_hi = np.arange(1, len(seg_xyz) + 1)
            idx_lo = np.maximum(0, idx_hi - win_dyn)
            cnt = (idx_hi - idx_lo)[:, None].astype(np.float64)
            trail_mean = (cs[idx_hi] - cs[idx_lo]) / cnt
            dyn_amp_raw[s:e] = np.linalg.norm(seg_xyz - trail_mean, axis=1)
    else:
        dyn_amp_raw = np.where(worn_raw, enmo, np.nan)

    # 2) OpenStrap / van Hees (2015/2018) 5-deg / 5-min sustained inactivity + 30-min worn-only bridge
    sib_raw = np.zeros(n_raw, dtype=bool)
    win_sib = max(6, int(round(300.0 / RAW_ACTIGRAPHY_DT_S)))  # 60 samples at 5s cadence
    if n_raw >= win_sib and np.isfinite(anglez[worn_raw]).any():
        ang_series = pd.Series(np.where(worn_raw & np.isfinite(anglez), anglez, np.nan))
        ang_range = (
            ang_series.rolling(win_sib, min_periods=win_sib).max()
            - ang_series.rolling(win_sib, min_periods=win_sib).min()
        ).to_numpy()
        win_ok = np.isfinite(ang_range) & (ang_range <= 5.0)
        if win_ok.any():
            ends = np.flatnonzero(win_ok)
            starts = ends - (win_sib - 1)
            diff_arr = np.zeros(n_raw + 1, dtype=np.int64)
            np.add.at(diff_arr, starts, 1)
            np.add.at(diff_arr, ends + 1, -1)
            sib_raw = (np.cumsum(diff_arr[:-1]) > 0) & worn_raw
            max_bridge = max(1, int(round(1800.0 / RAW_ACTIGRAPHY_DT_S)))
            for s, e in contiguous_true_runs(worn_raw):
                sub = sib_raw[s:e]
                for rs, re in contiguous_true_runs(~sub):
                    if 0 < rs and re < len(sub) and (re - rs) <= max_bridge:
                        sib_raw[s + rs : s + re] = True

    # 3) mad-lab-fau/sleep_analysis 60s sleep-wake heuristics (Cole-Kripke, Sadeh, Webster rescoring)
    fau_sleep = compute_fau_1min_sleep_heuristics(enmo, worn_raw, dt_s=RAW_ACTIGRAPHY_DT_S)

    bin_s = float(bin_minutes) * 60.0
    fs_bin = 1.0 / bin_s

    # Apply zero-phase anti-aliasing Butterworth lowpass per contiguous worn segment prior to decimation
    if res_meth in {"anti_alias_mean", "poly"} and int(bin_minutes) >= 1:
        cutoff_hz = min(0.45 / bin_s, 0.45 * RAW_ACTIGRAPHY_FS_HZ)
        if 0.0 < cutoff_hz < 0.5 * RAW_ACTIGRAPHY_FS_HZ:
            sos = sps.butter(4, float(cutoff_hz) / (0.5 * RAW_ACTIGRAPHY_FS_HZ), btype="low", output="sos")
            min_len = 3 * (2 * sos.shape[0] + 1) + 1
            for s, e in contiguous_true_runs(worn_raw):
                if (e - s) > max(30, min_len):
                    block = np.column_stack([enmo[s:e], anglez[s:e], light[s:e], mag_dev[s:e]])
                    filt = sps.sosfiltfilt(sos, block, axis=0)
                    enmo[s:e] = np.maximum(filt[:, 0], 0.0) if clip_val is None else np.clip(filt[:, 0], 0.0, clip_val)
                    anglez[s:e] = filt[:, 1]
                    light[s:e] = np.maximum(0.0, filt[:, 2])
                    mag_dev[s:e] = np.maximum(0.0, filt[:, 3])

    t0_s = float(np.floor(t_raw_sec[0] / bin_s) * bin_s)
    bin_idx = np.floor((t_raw_sec - t0_s) / bin_s).astype(np.int64)
    n_bins = int(bin_idx[-1]) + 1
    # Guard malformed extreme timestamp gaps (> 120 days -> cap safely using dense span around median)
    max_allowed_bins = int(round(120.0 * SECONDS_PER_DAY / bin_s))
    if n_bins <= 0 or n_bins > max_allowed_bins:
        # Fallback to dense index grid preserving exact 5-second step count
        t_raw_sec = np.arange(n_raw, dtype=np.float64) * RAW_ACTIGRAPHY_DT_S
        t0_s = 0.0
        bin_idx = np.floor(t_raw_sec / bin_s).astype(np.int64)
        n_bins = int(bin_idx[-1]) + 1

    expected_per_bin = bin_s / RAW_ACTIGRAPHY_DT_S
    min_samples_bin = max(1.0, float(min_bin_wear_frac) * expected_per_bin)

    w_float = worn_raw.astype(np.float64)
    total_counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    worn_counts = np.bincount(bin_idx, weights=w_float, minlength=n_bins)
    mask = worn_counts >= min_samples_bin
    denom = np.where(mask, np.maximum(worn_counts, 1.0), np.nan)

    def _wmean(arr: np.ndarray) -> np.ndarray:
        vals = np.where(worn_raw & np.isfinite(arr), arr, 0.0)
        return np.bincount(bin_idx, weights=vals, minlength=n_bins) / denom

    def _wmean_finite(arr: np.ndarray) -> np.ndarray:
        ok = worn_raw & np.isfinite(arr)
        cnt = np.bincount(bin_idx, weights=ok.astype(np.float64), minlength=n_bins)
        vals = np.where(ok, arr, 0.0)
        return np.where(mask & (cnt >= 1.0), np.bincount(bin_idx, weights=vals, minlength=n_bins) / np.maximum(cnt, 1.0), np.nan)

    enmo_bin = _wmean(enmo)
    log_enmo_bin = np.where(mask, np.log1p(np.maximum(enmo_bin, 0.0) * 1000.0), np.nan)
    enmo_sq = _wmean(np.square(enmo))
    enmo_std_bin = np.where(mask, np.sqrt(np.maximum(enmo_sq - np.square(enmo_bin), 0.0)), np.nan)
    anglez_bin = _wmean(anglez)
    anglez_sq = _wmean(np.square(anglez))
    anglez_var_bin = np.where(mask, np.maximum(anglez_sq - np.square(anglez_bin), 0.0), np.nan)
    light_bin = _wmean(light)
    jerk_sq_bin = _wmean(np.square(jerk_raw))
    jerk_rms_bin = np.where(mask, np.sqrt(np.maximum(jerk_sq_bin, 0.0)), np.nan)
    mag_dev_bin = _wmean(mag_dev)

    # Vaha-Ypya (2015) Mean Amplitude Deviation (MAD) + OpenStrap/van Hees/FAU binned channels
    if has_triaxial:
        bin_mean_mag = _wmean(vec_mag)
        mad_bin = _wmean_finite(np.where(mask[bin_idx], np.abs(vec_mag - bin_mean_mag[bin_idx]), np.nan))
    else:
        mad_bin = _wmean_finite(np.where(mask[bin_idx], np.abs(enmo - enmo_bin[bin_idx]), np.nan))
    dyn_amp_bin = _wmean_finite(dyn_amp_raw)
    rot_deg_bin = _wmean_finite(rot_deg_raw)
    upright_frac_bin = _wmean_finite(upright_raw)
    van_hees_sib_bin = _wmean(sib_raw.astype(np.float64))
    cole_kripke_bin = _wmean(fau_sleep["cole_kripke_raw"].astype(np.float64))
    sadeh_bin = _wmean(fau_sleep["sadeh_raw"].astype(np.float64))
    webster_bin = _wmean(fau_sleep["webster_raw"].astype(np.float64))

    # Local clock hour and ISO weekday (1..7) per bin via circular / modal bin assignment
    t_bin_s = t0_s + (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_s
    hour_of_day = np.mod(t_bin_s, SECONDS_PER_DAY) / 3600.0
    if tod is not None and len(tod) > 0:
        # Align clock hour exactly from observed time_of_day offset on first worn sample
        valid_tod = np.isfinite(tod)
        if valid_tod.any():
            i0 = int(np.flatnonzero(valid_tod)[0])
            exact_tod_offset = float(tod[i0] / 1e9) - np.mod(t_raw_sec[i0], SECONDS_PER_DAY)
            hour_of_day = np.mod(t_bin_s + exact_tod_offset, SECONDS_PER_DAY) / 3600.0
    t_hours = t_bin_s / 3600.0

    # Weekday majority/mean per bin for weekday vs weekend social jetlag
    wd_valid = np.isfinite(weekday) & (weekday >= 1) & (weekday <= 7)
    wd_sum = np.bincount(bin_idx, weights=np.where(wd_valid, weekday, 0.0), minlength=n_bins)
    wd_cnt = np.bincount(bin_idx, weights=wd_valid.astype(np.float64), minlength=n_bins)
    weekday_bin = np.where(wd_cnt > 0, np.rint(wd_sum / np.maximum(wd_cnt, 1.0)), np.nan)

    grid = {
        "n_raw": int(n_raw),
        "raw_wear_frac": float(np.mean(worn_raw)),
        "n_bins": int(n_bins),
        "bin_minutes": int(bin_minutes),
        "fs_bin_hz": float(fs_bin),
        "g_ref_cal": float(g_ref_cal),
        "mask": mask,
        "worn_frac_bin": np.where(total_counts > 0, worn_counts / np.maximum(total_counts, 1.0), 0.0),
        "enmo": enmo_bin,
        "log_enmo": log_enmo_bin,
        "enmo_std": enmo_std_bin,
        "anglez": anglez_bin,
        "anglez_var": anglez_var_bin,
        "light": light_bin,
        "jerk_rms": jerk_rms_bin,
        "mag_dev": mag_dev_bin,
        "mad": mad_bin,
        "dyn_amp": dyn_amp_bin,
        "rot_deg": rot_deg_bin,
        "upright_frac": upright_frac_bin,
        "van_hees_sib": van_hees_sib_bin,
        "cole_kripke_sleep": cole_kripke_bin,
        "sadeh_sleep": sadeh_bin,
        "webster_sleep": webster_bin,
        "hour_of_day": hour_of_day,
        "t_hours": t_hours,
        "weekday": weekday_bin,
    }
    if cache_key is not None:
        if len(_GRID_CACHE) > 2500:
            _GRID_CACHE.clear()
        _GRID_CACHE[cache_key] = grid
    return grid


# ---------------------------------------------------------------------------
# Handcrafted Autocorrelation Feature Extraction
# ---------------------------------------------------------------------------

def extract_handcrafted_autocorr_features(
    parquet_path_or_df: Any,
    *,
    bin_minutes: int = 15,
    resample_method: str = "anti_alias_mean",
    non_wear_policy: str = "mask",
    min_bin_wear_frac: float = 0.25,
    max_gap_bins: int = 4,
    clip_enmo: float = 4.0,
    acf_detrend: str = "linear",
    acf_bias: str = "unbiased",
    sedentary_threshold_g: float = 0.025,
    period_band_hours: Tuple[float, float] = (18.0, 30.0),
    fundamental_ratio: float = 0.85,
    min_acf_strength: float = 0.08,
    short_lags_h: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
) -> Dict[str, float]:
    """Extract DSP-rigorous autocorrelation, Cosinor, and sedentary bout handcrafted features."""
    keys = [
        "hc_has_actigraphy", "hc_wear_days", "hc_wear_frac", "hc_weekend_wear_frac",
        "hc_enmo_mean", "hc_enmo_std", "hc_enmo_p90",
        "hc_acf_24h", "hc_acf_12h", "hc_acf_48h", "hc_acf_light_24h",
        "hc_circadian_period_h", "hc_circadian_period_err_h",
        "hc_circadian_peak_acf", "hc_circadian_concavity",
        "hc_acf_short_mean", "hc_acf_sedentary_short_mean", "hc_acf_decay_tau_h",
        "hc_sedentary_frac", "hc_sedentary_max_bout_h",
        "hc_cosinor_mesor", "hc_cosinor_amp", "hc_cosinor_rel_amp",
        "hc_cosinor_acrophase_h", "hc_cosinor_r2", "hc_social_jetlag_h",
        "hc_night_day_enmo_ratio", "hc_late_night_enmo", "hc_late_night_light_p90",
        "hc_nocturnal_anglez_var", "hc_jerk_norm_ratio",
        "hc_welch_circadian_frac", "hc_welch_ultradian_frac", "hc_welch_entropy",
        # OpenStrap/analytics SRI, Circular SJL, Kinematics & PRSA/CUSUM
        "hc_van_hees_immobile_frac", "hc_sri", "hc_sri_pair_std",
        "hc_circ_sleep_mid_h", "hc_circ_sleep_mid_R", "hc_circ_sleep_mid_sd_h",
        "hc_circ_sleep_dur_h", "hc_circ_sjl_signed_h", "hc_circ_sjl_abs_h", "hc_circ_wknd_oversleep_h",
        "hc_g_ref_cal", "hc_dyn_amp_mean", "hc_mad_mean", "hc_rot_deg_mean", "hc_rot_deg_night",
        "hc_upright_night_frac", "hc_upright_day_frac",
        "hc_prsa_dc", "hc_prsa_ac", "hc_prsa_asym",
        "hc_cusum_max", "hc_cusum_count",
        # mad-lab-fau/sleep_analysis sleep heuristics
        "hc_cole_kripke_sleep_frac", "hc_sadeh_sleep_frac", "hc_webster_sleep_frac", "hc_sadeh_night_efficiency",
    ]
    out: Dict[str, float] = {k: np.nan for k in keys}
    out["hc_has_actigraphy"] = 0.0

    grid = bin_actigraphy_series(
        parquet_path_or_df,
        bin_minutes=bin_minutes,
        resample_method=resample_method,
        non_wear_policy=non_wear_policy,
        min_bin_wear_frac=min_bin_wear_frac,
        clip_enmo=clip_enmo,
    )
    if grid is None:
        return out

    mask = grid["mask"]
    n_valid = int(mask.sum())
    if n_valid < 16:
        return out

    fs_bin = grid["fs_bin_hz"]
    bin_h = float(bin_minutes) / 60.0
    bins_per_day = max(4, int(round(24.0 / bin_h)))

    out["hc_has_actigraphy"] = 1.0
    out["hc_wear_days"] = float(n_valid * bin_h / 24.0)
    out["hc_wear_frac"] = float(grid["raw_wear_frac"])

    enmo_v = grid["enmo"][mask]
    out["hc_enmo_mean"] = float(np.mean(enmo_v))
    out["hc_enmo_std"] = float(np.std(enmo_v))
    out["hc_enmo_p90"] = float(np.percentile(enmo_v, 90))

    # Weekend compliance
    wd_mask = mask & np.isfinite(grid["weekday"])
    if wd_mask.any():
        is_wknd = grid["weekday"][wd_mask] >= 6
        out["hc_weekend_wear_frac"] = float(np.mean(is_wknd))

    # 1. Masked unbiased autocovariance on log1p-stabilized ENMO + circadian periodicity
    max_lag_bins = min(grid["n_bins"] - 1, max(3 * bins_per_day + 4, int(round(72.0 / bin_h))))
    acf_enmo, _ = masked_unbiased_autocovariance(
        grid["log_enmo"],
        mask,
        max_lag=max_lag_bins,
        min_pair_count=max(12, bins_per_day // 2),
        detrend=acf_detrend,
        bias=acf_bias,
        normalize=True,
    )
    acf_light, _ = masked_unbiased_autocovariance(
        np.log1p(np.maximum(grid["light"], 0.0)),
        mask,
        max_lag=min(max_lag_bins, bins_per_day + 4),
        min_pair_count=max(12, bins_per_day // 2),
        detrend=acf_detrend,
        bias=acf_bias,
        normalize=True,
    )

    def _lag_peak(acf_arr: np.ndarray, target_h: float, tol_h: float = 1.0) -> float:
        c = int(round(target_h / bin_h))
        r = max(0, int(round(tol_h / bin_h)))
        lo, hi = max(1, c - r), min(len(acf_arr) - 1, c + r)
        if lo > hi:
            return np.nan
        sub = acf_arr[lo : hi + 1]
        return float(np.nanmax(sub)) if np.isfinite(sub).any() else np.nan

    out["hc_acf_24h"] = _lag_peak(acf_enmo, 24.0, tol_h=1.0)
    out["hc_acf_12h"] = _lag_peak(acf_enmo, 12.0, tol_h=0.5)
    out["hc_acf_48h"] = _lag_peak(acf_enmo, 48.0, tol_h=1.5)
    out["hc_acf_light_24h"] = _lag_peak(acf_light, 24.0, tol_h=1.0)

    # 2. cr7_kimmitch PpgHr fundamental-preferred parabolic period estimate
    per_info = find_fundamental_acf_period(
        acf_enmo,
        fs_bin,
        period_band_hours=period_band_hours,
        fundamental_ratio=fundamental_ratio,
        min_strength=min_acf_strength,
    )
    out["hc_circadian_period_h"] = per_info["period_h"]
    out["hc_circadian_period_err_h"] = (
        abs(per_info["period_h"] - 24.0) if np.isfinite(per_info["period_h"]) else np.nan
    )
    out["hc_circadian_peak_acf"] = per_info["peak_acf"]
    out["hc_circadian_concavity"] = per_info["concavity"]

    # 3. Short-lag sedentary bout autocorrelation (SedentaryDetector.swift)
    short_vals = []
    for lh in short_lags_h:
        k = max(1, int(round(float(lh) / bin_h)))
        if k < len(acf_enmo) and np.isfinite(acf_enmo[k]):
            short_vals.append(float(acf_enmo[k]))
    out["hc_acf_short_mean"] = float(np.mean(short_vals)) if short_vals else np.nan

    # Positive ACF integral time scale over 0..4h (behavioral bout persistence)
    max_tau_bins = min(len(acf_enmo) - 1, max(2, int(round(4.0 / bin_h))))
    pos_short = np.maximum(np.nan_to_num(acf_enmo[1 : max_tau_bins + 1], nan=0.0), 0.0)
    out["hc_acf_decay_tau_h"] = float(np.sum(pos_short) * bin_h)

    sed_bin = (grid["enmo"] < float(sedentary_threshold_g)).astype(np.float64)
    out["hc_sedentary_frac"] = float(np.mean(sed_bin[mask]))
    sed_runs = contiguous_true_runs(mask & (grid["enmo"] < float(sedentary_threshold_g)))
    out["hc_sedentary_max_bout_h"] = (
        float(max(e - s for s, e in sed_runs) * bin_h) if sed_runs else 0.0
    )
    acf_sed, _ = masked_unbiased_autocovariance(
        sed_bin,
        mask,
        max_lag=max(4, int(round(4.0 / bin_h))),
        min_pair_count=12,
        detrend="constant",
        bias=acf_bias,
        normalize=True,
    )
    sed_shorts = [
        float(acf_sed[max(1, int(round(float(lh) / bin_h)))])
        for lh in short_lags_h
        if max(1, int(round(float(lh) / bin_h))) < len(acf_sed)
        and np.isfinite(acf_sed[max(1, int(round(float(lh) / bin_h)))])
    ]
    out["hc_acf_sedentary_short_mean"] = float(np.mean(sed_shorts)) if sed_shorts else np.nan

    # 4. 24h Cosinor (CircadianEngine.swift) + Weekday vs Weekend social jetlag
    cos_all = fit_circadian_cosinor_24h(grid["t_hours"], grid["log_enmo"], mask, min_bins=max(24, bins_per_day))
    out["hc_cosinor_mesor"] = cos_all["mesor"]
    out["hc_cosinor_amp"] = cos_all["amplitude"]
    out["hc_cosinor_rel_amp"] = cos_all["rel_amplitude"]
    out["hc_cosinor_acrophase_h"] = cos_all["acrophase_h"]
    out["hc_cosinor_r2"] = cos_all["r2"]

    if wd_mask.any():
        cos_wkdy = fit_circadian_cosinor_24h(
            grid["t_hours"], grid["log_enmo"], mask & (grid["weekday"] <= 5), min_bins=24
        )
        cos_wknd = fit_circadian_cosinor_24h(
            grid["t_hours"], grid["log_enmo"], mask & (grid["weekday"] >= 6), min_bins=16
        )
        a1, a2 = cos_wkdy["acrophase_h"], cos_wknd["acrophase_h"]
        if np.isfinite(a1) and np.isfinite(a2):
            diff = abs(a2 - a1) % 24.0
            out["hc_social_jetlag_h"] = float(min(diff, 24.0 - diff))

    # 5. Nocturnal quiet-hours activity, late-night light, posture & self-calibrated jerk floor
    hod = grid["hour_of_day"]
    day_m = mask & (hod >= 8.0) & (hod < 21.0)
    night_m = mask & ((hod >= 22.0) | (hod < 6.0))
    late_m = mask & ((hod >= 23.0) | (hod < 4.0))

    day_enmo = float(np.mean(grid["enmo"][day_m])) if day_m.any() else np.nan
    night_enmo = float(np.mean(grid["enmo"][night_m])) if night_m.any() else np.nan
    if np.isfinite(day_enmo) and np.isfinite(night_enmo):
        out["hc_night_day_enmo_ratio"] = float(night_enmo / max(day_enmo, 1e-4))
    out["hc_late_night_enmo"] = float(np.mean(grid["enmo"][late_m])) if late_m.any() else np.nan
    out["hc_late_night_light_p90"] = (
        float(np.percentile(grid["light"][late_m], 90)) if late_m.any() else np.nan
    )
    out["hc_nocturnal_anglez_var"] = (
        float(np.median(grid["anglez_var"][night_m])) if night_m.any() else np.nan
    )

    jerk_v = grid["jerk_rms"][mask]
    jerk_v = jerk_v[np.isfinite(jerk_v)]
    if len(jerk_v) > 0:
        med_jerk = max(float(np.percentile(jerk_v, 25)), 1e-5)  # SleepAccel.swift quiescent jerk floor
        out["hc_jerk_norm_ratio"] = float(np.percentile(jerk_v, 90) / med_jerk)

    # 6. Welch PSD spectrum check (dsp-slides.pdf)
    spec = compute_welch_circadian_spectrum(grid["log_enmo"], mask, fs_bin, max_gap_bins=max_gap_bins)
    out["hc_welch_circadian_frac"] = spec["welch_circadian_frac"]
    out["hc_welch_ultradian_frac"] = spec["welch_ultradian_frac"]
    out["hc_welch_entropy"] = spec["welch_entropy"]

    # 7. OpenStrap/analytics van Hees SIB, Phillips SRI, circular SJL, kinematics & PRSA/CUSUM
    sib = grid["van_hees_sib"]
    sleep_m = mask & np.isfinite(sib) & (sib >= 0.65)
    out["hc_van_hees_immobile_frac"] = float(np.mean(sleep_m[mask]))
    sri_res = compute_phillips_sri(sleep_m, mask, bins_per_day)
    out["hc_sri"] = sri_res["sri"]
    out["hc_sri_pair_std"] = sri_res["sri_pair_std"]

    cs_res = compute_circular_sleep_regularity(sib, mask, hod, grid["weekday"], bin_minutes)
    for k, v in cs_res.items():
        out[f"hc_{k}"] = v

    out["hc_g_ref_cal"] = float(grid.get("g_ref_cal", 1.0))
    if np.isfinite(grid["dyn_amp"][mask]).any():
        out["hc_dyn_amp_mean"] = float(np.nanmean(grid["dyn_amp"][mask]))
    if np.isfinite(grid["mad"][mask]).any():
        out["hc_mad_mean"] = float(np.nanmean(grid["mad"][mask]))
    if np.isfinite(grid["rot_deg"][mask]).any():
        out["hc_rot_deg_mean"] = float(np.nanmean(grid["rot_deg"][mask]))
    if night_m.any() and np.isfinite(grid["rot_deg"][night_m]).any():
        out["hc_rot_deg_night"] = float(np.nanmean(grid["rot_deg"][night_m]))
    if late_m.any() and np.isfinite(grid["upright_frac"][late_m]).any():
        out["hc_upright_night_frac"] = float(np.nanmean(grid["upright_frac"][late_m]))
    if day_m.any() and np.isfinite(grid["upright_frac"][day_m]).any():
        out["hc_upright_day_frac"] = float(np.nanmean(grid["upright_frac"][day_m]))

    prsa_res = compute_prsa_capacity(grid["log_enmo"], mask, l=2, anchor_ratio_cap=0.5)
    out["hc_prsa_dc"] = prsa_res["prsa_dc"]
    out["hc_prsa_ac"] = prsa_res["prsa_ac"]
    out["hc_prsa_asym"] = prsa_res["prsa_asym"]

    cusum_res = compute_cusum_volatility(grid["log_enmo"], mask, bins_per_day, k=0.5, h=3.0)
    out["hc_cusum_max"] = cusum_res["cusum_max"]
    out["hc_cusum_count"] = cusum_res["cusum_count"]

    # 8. mad-lab-fau/sleep_analysis sleep-wake heuristics
    if np.isfinite(grid["cole_kripke_sleep"][mask]).any():
        out["hc_cole_kripke_sleep_frac"] = float(np.nanmean(grid["cole_kripke_sleep"][mask]))
    if np.isfinite(grid["sadeh_sleep"][mask]).any():
        out["hc_sadeh_sleep_frac"] = float(np.nanmean(grid["sadeh_sleep"][mask]))
    if np.isfinite(grid["webster_sleep"][mask]).any():
        out["hc_webster_sleep_frac"] = float(np.nanmean(grid["webster_sleep"][mask]))
    if night_m.any() and np.isfinite(grid["sadeh_sleep"][night_m]).any():
        out["hc_sadeh_night_efficiency"] = float(np.nanmean(grid["sadeh_sleep"][night_m]))
    return out


def batch_extract_handcrafted_autocorr_features(
    df: pd.DataFrame,
    *,
    path_col: str = "series_path",
    id_col: str = "id",
    bin_minutes: int = 15,
    resample_method: str = "anti_alias_mean",
    non_wear_policy: str = "mask",
    min_bin_wear_frac: float = 0.25,
    max_gap_bins: int = 4,
    clip_enmo: float = 4.0,
    acf_detrend: str = "linear",
    acf_bias: str = "unbiased",
    sedentary_threshold_g: float = 0.025,
    period_band_hours: Tuple[float, float] = (18.0, 30.0),
    fundamental_ratio: float = 0.85,
    min_acf_strength: float = 0.08,
    short_lags_h: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
    use_cache: bool = True,
) -> pd.DataFrame:
    """Extract handcrafted autocorrelation feature matrix aligned to ``df.index``."""
    ids = df[id_col].astype(str).tolist() if id_col in df.columns else [str(i) for i in range(len(df))]
    paths = df[path_col].tolist() if path_col in df.columns else [None] * len(df)
    cfg_sig = (
        int(bin_minutes),
        str(resample_method),
        str(non_wear_policy),
        round(float(min_bin_wear_frac), 4),
        int(max_gap_bins),
        round(float(clip_enmo), 4),
        str(acf_detrend),
        str(acf_bias),
        round(float(sedentary_threshold_g), 5),
        tuple(round(float(x), 3) for x in period_band_hours),
        round(float(fundamental_ratio), 4),
        round(float(min_acf_strength), 4),
        tuple(round(float(x), 4) for x in short_lags_h),
    )

    rows: List[Dict[str, float]] = []
    for sid, p in zip(ids, paths):
        row_key = (sid, str(p), cfg_sig)
        if use_cache and isinstance(p, (str, Path)) and row_key in _HC_FEAT_CACHE:
            rows.append(_HC_FEAT_CACHE[row_key])  # type: ignore[arg-type]
            continue
        feat = extract_handcrafted_autocorr_features(
            p,
            bin_minutes=bin_minutes,
            resample_method=resample_method,
            non_wear_policy=non_wear_policy,
            min_bin_wear_frac=min_bin_wear_frac,
            max_gap_bins=max_gap_bins,
            clip_enmo=clip_enmo,
            acf_detrend=acf_detrend,
            acf_bias=acf_bias,
            sedentary_threshold_g=sedentary_threshold_g,
            period_band_hours=period_band_hours,
            fundamental_ratio=fundamental_ratio,
            min_acf_strength=min_acf_strength,
            short_lags_h=short_lags_h,
        )
        if use_cache and isinstance(p, (str, Path)):
            if len(_HC_FEAT_CACHE) > 8000:
                _HC_FEAT_CACHE.clear()
            _HC_FEAT_CACHE[row_key] = feat  # type: ignore[assignment]
        rows.append(feat)

    return pd.DataFrame(rows, index=df.index)


# ---------------------------------------------------------------------------
# Handcrafted Autocorrelation Baseline Estimator
# ---------------------------------------------------------------------------

def _build_handcrafted_risk_matrix(
    df: pd.DataFrame,
    hc_df: pd.DataFrame,
    *,
    cgas_healthy_ref: float = 70.0,
) -> pd.DataFrame:
    """Construct clean, directional handcrafted risk terms (+ = higher PIU severity).

    Zero target leakage: explicitly refuses any ``PCIAT-*`` or ``sii`` column.
    Missing observations stay ``NaN`` so fold-standardized weighted averaging
    ignores missing instruments instead of distorting variance.
    """
    leaky = [c for c in df.columns if c in LEAKY_TARGET_COLS or str(c).startswith(LEAKY_PREFIX)]
    if leaky:
        raise ValueError(f"Leaky target columns detected in feature input: {leaky[:5]}")

    risk = pd.DataFrame(index=df.index, dtype=np.float64)

    def _col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.full(len(df), np.nan, dtype=np.float64)
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)

    # 1. Tabular clinical/behavioral domain indicators
    risk["risk_internet_hours"] = _col("PreInt_EduHx-computerinternet_hoursday")
    risk["risk_age"] = _col("Basic_Demos-Age")
    sds_t = _col("SDS-SDS_Total_T")
    sds_r = _col("SDS-SDS_Total_Raw")
    risk["risk_sleep_disturbance"] = np.where(np.isfinite(sds_t), sds_t, np.nan)
    # Harmonized physical inactivity (-PAQ score across Adolescent PAQ_A & Child PAQ_C)
    paq_a = _col("PAQ_A-PAQ_A_Total")
    paq_c = _col("PAQ_C-PAQ_C_Total")
    paq = np.where(np.isfinite(paq_a), paq_a, paq_c)
    risk["risk_inactivity"] = -paq
    cgas = _col("CGAS-CGAS_Score")
    risk["risk_cgas_deficit"] = cgas_healthy_ref - cgas
    # Normalized age * internet exposure interaction (clinically older youth + high screen use -> severe PIU)
    age = risk["risk_age"]
    hrs = risk["risk_internet_hours"]
    risk["risk_age_internet"] = np.where(np.isfinite(age) & np.isfinite(hrs), age * (hrs + 0.5), np.nan)
    if np.isfinite(sds_r).any() and not np.isfinite(sds_t).any():
        risk["risk_sleep_disturbance"] = sds_r

    # 2. Handcrafted Actigraphy Autocorrelation & Circadian indicators
    acf24 = hc_df["hc_acf_24h"].to_numpy(dtype=np.float64) if "hc_acf_24h" in hc_df else np.full(len(df), np.nan)
    risk["risk_acf_irregularity"] = np.where(np.isfinite(acf24), 1.0 - acf24, np.nan)

    sed_acf = hc_df["hc_acf_sedentary_short_mean"].to_numpy(dtype=np.float64) if "hc_acf_sedentary_short_mean" in hc_df else np.full(len(df), np.nan)
    short_acf = hc_df["hc_acf_short_mean"].to_numpy(dtype=np.float64) if "hc_acf_short_mean" in hc_df else np.full(len(df), np.nan)
    risk["risk_sedentary_acf"] = np.where(np.isfinite(sed_acf), sed_acf, short_acf)

    ng_ratio = hc_df["hc_night_day_enmo_ratio"].to_numpy(dtype=np.float64) if "hc_night_day_enmo_ratio" in hc_df else np.full(len(df), np.nan)
    risk["risk_nocturnal_ratio"] = np.where(np.isfinite(ng_ratio), np.log1p(np.maximum(ng_ratio, 0.0)), np.nan)

    acro = hc_df["hc_cosinor_acrophase_h"].to_numpy(dtype=np.float64) if "hc_cosinor_acrophase_h" in hc_df else np.full(len(df), np.nan)
    # Phase delay past midday (~14h normal peak -> later evening/night shift indicates delayed phase)
    risk["risk_acrophase_delay"] = np.where(
        np.isfinite(acro),
        np.cos(2.0 * np.pi * (acro - 20.0) / 24.0),  # +1 near 20:00-00:00 late peak, -1 near 08:00 morning peak
        np.nan,
    )

    per_err = hc_df["hc_circadian_period_err_h"].to_numpy(dtype=np.float64) if "hc_circadian_period_err_h" in hc_df else np.full(len(df), np.nan)
    risk["risk_period_drift"] = per_err

    sri = hc_df["hc_sri"].to_numpy(dtype=np.float64) if "hc_sri" in hc_df else np.full(len(df), np.nan)
    sjl = hc_df["hc_circ_sjl_abs_h"].to_numpy(dtype=np.float64) if "hc_circ_sjl_abs_h" in hc_df else np.full(len(df), np.nan)
    sri_term = np.where(np.isfinite(sri), (100.0 - sri) / 200.0, np.nan)
    risk["risk_sri_irregularity"] = np.where(
        np.isfinite(sri_term) & np.isfinite(sjl),
        0.70 * sri_term + 0.30 * (sjl / 4.0),
        np.where(np.isfinite(sri_term), sri_term, np.where(np.isfinite(sjl), sjl / 4.0, np.nan)),
    )

    up_n = hc_df["hc_upright_night_frac"].to_numpy(dtype=np.float64) if "hc_upright_night_frac" in hc_df else np.full(len(df), np.nan)
    risk["risk_upright_night"] = up_n

    cusum = hc_df["hc_cusum_max"].to_numpy(dtype=np.float64) if "hc_cusum_max" in hc_df else np.full(len(df), np.nan)
    prsa_a = hc_df["hc_prsa_asym"].to_numpy(dtype=np.float64) if "hc_prsa_asym" in hc_df else np.full(len(df), np.nan)
    risk["risk_cusum_prsa"] = np.where(
        np.isfinite(cusum) & np.isfinite(prsa_a),
        0.65 * (cusum / 4.0) + 0.35 * np.abs(prsa_a),
        np.where(np.isfinite(cusum), cusum / 4.0, np.abs(prsa_a)),
    )
    return risk


class HandcraftedAutocorrelationBaseline(BaseEstimator):
    """100% non-ML handcrafted autocorrelation & clinical domain baseline for CMI PIU (``sii`` in {0..3}).

    Zero supervised machine learning weights are fitted: regression/boosting models are never
    used. Feature contributions use explicit clinical + DSP prior weights combined via one of
    three deterministic handcrafted scoring rules:
    * ``'weighted_z_index'``: normalized weighted combination over available clinical and
      wearable autocorrelation/circadian risk terms (missingness never distorts variance).
    * ``'multiplicative_hazard'``: boosts base risk when high internet exposure co-occurs
      with disrupted 24h autocorrelation or high sedentary bout persistence.
    * ``'autocorr_gated_rules'``: dynamically gates actigraphy circadian terms by valid wear
      duration (``hc_wear_days >= 3``) while relying purely on clinical priors otherwise.

    Fitting estimates only fold standardization moments ``(mean, std)`` for unit alignment
    and optional monotonic cutpoints ``(t0, t1, t2)``.
    """

    def __init__(
        self,
        bin_minutes: int = 15,
        resample_method: str = "anti_alias_mean",
        non_wear_policy: str = "mask",
        min_bin_wear_frac: float = 0.25,
        max_gap_bins: int = 4,
        clip_enmo: float = 4.0,
        acf_detrend: str = "linear",
        acf_bias: str = "unbiased",
        sedentary_threshold_g: float = 0.025,
        period_band_hours: Tuple[float, float] = (18.0, 30.0),
        fundamental_ratio: float = 0.85,
        min_acf_strength: float = 0.08,
        short_lags_h: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
        scoring_rule: str = "weighted_z_index",
        coupling_alpha: float = 0.35,
        w_internet_hours: float = 1.35,
        w_age: float = 0.70,
        w_age_internet: float = 0.90,
        w_sleep_disturbance: float = 0.65,
        w_inactivity: float = 0.35,
        w_cgas_deficit: float = 0.35,
        w_acf_irregularity: float = 0.45,
        w_sedentary_acf: float = 0.40,
        w_nocturnal_ratio: float = 0.35,
        w_acrophase_delay: float = 0.30,
        w_period_drift: float = 0.25,
        w_sri_irregularity: float = 0.40,
        w_upright_night: float = 0.35,
        w_cusum_prsa: float = 0.25,
        threshold_mode: str = "optimize_qwk",
        id_col: str = "id",
        path_col: str = "series_path",
    ):
        self.bin_minutes = bin_minutes
        self.resample_method = resample_method
        self.non_wear_policy = non_wear_policy
        self.min_bin_wear_frac = min_bin_wear_frac
        self.max_gap_bins = max_gap_bins
        self.clip_enmo = clip_enmo
        self.acf_detrend = acf_detrend
        self.acf_bias = acf_bias
        self.sedentary_threshold_g = sedentary_threshold_g
        self.period_band_hours = period_band_hours
        self.fundamental_ratio = fundamental_ratio
        self.min_acf_strength = min_acf_strength
        self.short_lags_h = short_lags_h
        self.scoring_rule = scoring_rule
        self.coupling_alpha = coupling_alpha
        self.w_internet_hours = w_internet_hours
        self.w_age = w_age
        self.w_age_internet = w_age_internet
        self.w_sleep_disturbance = w_sleep_disturbance
        self.w_inactivity = w_inactivity
        self.w_cgas_deficit = w_cgas_deficit
        self.w_acf_irregularity = w_acf_irregularity
        self.w_sedentary_acf = w_sedentary_acf
        self.w_nocturnal_ratio = w_nocturnal_ratio
        self.w_acrophase_delay = w_acrophase_delay
        self.w_period_drift = w_period_drift
        self.w_sri_irregularity = w_sri_irregularity
        self.w_upright_night = w_upright_night
        self.w_cusum_prsa = w_cusum_prsa
        self.threshold_mode = threshold_mode
        self.id_col = id_col
        self.path_col = path_col

    def _prior_weights(self) -> Dict[str, float]:
        return {
            "risk_internet_hours": float(self.w_internet_hours),
            "risk_age": float(self.w_age),
            "risk_age_internet": float(self.w_age_internet),
            "risk_sleep_disturbance": float(self.w_sleep_disturbance),
            "risk_inactivity": float(self.w_inactivity),
            "risk_cgas_deficit": float(self.w_cgas_deficit),
            "risk_acf_irregularity": float(self.w_acf_irregularity),
            "risk_sedentary_acf": float(self.w_sedentary_acf),
            "risk_nocturnal_ratio": float(self.w_nocturnal_ratio),
            "risk_acrophase_delay": float(self.w_acrophase_delay),
            "risk_period_drift": float(self.w_period_drift),
            "risk_sri_irregularity": float(self.w_sri_irregularity),
            "risk_upright_night": float(self.w_upright_night),
            "risk_cusum_prsa": float(self.w_cusum_prsa),
        }

    def _prepare_clean_inputs(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_cols = [c for c in X.columns if c not in LEAKY_TARGET_COLS and not str(c).startswith(LEAKY_PREFIX)]
        return X[keep_cols]

    def _extract_hc_and_risk(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        clean_X = self._prepare_clean_inputs(X)
        # Map SequenceExtractor actigraphy features if present, else compute directly
        if "seq_acf_log_24h" in clean_X.columns and "seq_has_actigraphy" in clean_X.columns:
            hc_df = pd.DataFrame({
                "hc_has_actigraphy": clean_X.get("seq_has_actigraphy", 0.0),
                "hc_wear_days": clean_X.get("seq_wear_days", np.nan),
                "hc_acf_24h": clean_X.get("seq_acf_log_24h", np.nan),
                "hc_acf_short_mean": clean_X.get("seq_acf_short_mean", np.nan),
                "hc_acf_sedentary_short_mean": clean_X.get("seq_acf_sedentary_30m", np.nan),
                "hc_night_day_enmo_ratio": clean_X.get("seq_night_day_enmo_ratio", np.nan),
                "hc_cosinor_acrophase_h": clean_X.get("seq_cosinor_acrophase_h", np.nan),
                "hc_circadian_period_err_h": clean_X.get("seq_circadian_period_err_h", np.nan),
                "hc_sri": clean_X.get("seq_sri", np.nan),
                "hc_circ_sjl_abs_h": clean_X.get("seq_circ_sjl_abs_h", np.nan),
                "hc_upright_night_frac": clean_X.get("seq_upright_night_frac", np.nan),
                "hc_cusum_max": clean_X.get("seq_cusum_enmo_max", np.nan),
                "hc_prsa_asym": clean_X.get("seq_prsa_asym", np.nan),
            }, index=clean_X.index)
        elif "hc_acf_24h" in clean_X.columns and "hc_has_actigraphy" in clean_X.columns:
            hc_cols = [c for c in clean_X.columns if str(c).startswith("hc_")]
            hc_df = clean_X[hc_cols]
        else:
            hc_df = batch_extract_handcrafted_autocorr_features(
                clean_X,
                path_col=self.path_col,
                id_col=self.id_col,
                bin_minutes=self.bin_minutes,
                resample_method=self.resample_method,
                non_wear_policy=self.non_wear_policy,
                min_bin_wear_frac=self.min_bin_wear_frac,
                max_gap_bins=self.max_gap_bins,
                clip_enmo=self.clip_enmo,
                acf_detrend=self.acf_detrend,
                acf_bias=self.acf_bias,
                sedentary_threshold_g=self.sedentary_threshold_g,
                period_band_hours=self.period_band_hours,
                fundamental_ratio=self.fundamental_ratio,
                min_acf_strength=self.min_acf_strength,
                short_lags_h=self.short_lags_h,
            )
        risk_df = _build_handcrafted_risk_matrix(clean_X, hc_df)
        return hc_df, risk_df

    def fit(self, X: pd.DataFrame, y: Any):
        y_arr = np.asarray(y["sii"] if isinstance(y, pd.DataFrame) and "sii" in y.columns else y, dtype=np.int64).ravel()
        hc_df, risk_df = self._extract_hc_and_risk(X)

        self.risk_columns_ = list(risk_df.columns)
        self.hc_feature_names_ = list(hc_df.columns)
        self.feature_means_: Dict[str, float] = {}
        self.feature_stds_: Dict[str, float] = {}

        z_mat = np.full((len(risk_df), len(self.risk_columns_)), np.nan, dtype=np.float64)
        for j, col in enumerate(self.risk_columns_):
            vals = risk_df[col].to_numpy(dtype=np.float64)
            fin = np.isfinite(vals)
            mu = float(np.mean(vals[fin])) if fin.any() else 0.0
            sd = float(np.std(vals[fin])) if fin.sum() > 1 else 1.0
            if sd <= 1e-9:
                sd = 1.0
            self.feature_means_[col] = mu
            self.feature_stds_[col] = sd
            z_mat[:, j] = np.where(fin, np.clip((vals - mu) / sd, -4.0, 4.0), np.nan)

        prior_w = self._prior_weights()
        self.prior_weight_vec_ = np.array([prior_w.get(c, 0.0) for c in self.risk_columns_], dtype=np.float64)

        raw_scores = self._apply_handcrafted_rule(z_mat, hc_df)
        self.train_score_mean_ = float(np.mean(raw_scores))
        self.train_score_std_ = max(float(np.std(raw_scores)), 1e-6)

        if self.threshold_mode == "optimize_qwk":
            self.thresholds_ = optimize_thresholds_qwk(y_arr, raw_scores)
        elif self.threshold_mode == "quantile":
            p0, p1, p2 = float(np.mean(y_arr <= 0)), float(np.mean(y_arr <= 1)), float(np.mean(y_arr <= 2))
            self.thresholds_ = np.sort(np.quantile(raw_scores, [p0, p1, p2]))
        elif self.threshold_mode == "fixed_midpoint":
            self.thresholds_ = self.train_score_mean_ + (np.array([0.5, 1.5, 2.5]) - y_arr.mean()) * (
                self.train_score_std_ / max(y_arr.std(), 1e-3)
            )
        else:
            raise ValueError(f"Unsupported threshold_mode: {self.threshold_mode!r}")

        self.classes_ = np.array([0, 1, 2, 3], dtype=np.int64)
        self.last_train_qwk_ = quadratic_weighted_kappa(y_arr, apply_thresholds(raw_scores, self.thresholds_))
        return self

    def _transform_z(self, risk_df: pd.DataFrame) -> np.ndarray:
        z_mat = np.full((len(risk_df), len(self.risk_columns_)), np.nan, dtype=np.float64)
        for j, col in enumerate(self.risk_columns_):
            vals = risk_df[col].to_numpy(dtype=np.float64) if col in risk_df.columns else np.full(len(risk_df), np.nan)
            mu = self.feature_means_[col]
            sd = self.feature_stds_[col]
            fin = np.isfinite(vals)
            z_mat[:, j] = np.where(fin, np.clip((vals - mu) / sd, -4.0, 4.0), np.nan)
        return z_mat

    def _apply_handcrafted_rule(self, z_mat: np.ndarray, hc_df: pd.DataFrame) -> np.ndarray:
        """Evaluate exact deterministic handcrafted equation with zero ML regression."""
        avail = np.isfinite(z_mat).astype(np.float64)
        weights = np.broadcast_to(self.prior_weight_vec_[None, :], z_mat.shape).copy()

        if self.scoring_rule == "autocorr_gated_rules":
            wear_days = hc_df["hc_wear_days"].to_numpy(dtype=np.float64) if "hc_wear_days" in hc_df else np.zeros(len(z_mat))
            low_wear = (~np.isfinite(wear_days)) | (wear_days < 3.0)
            act_idx = [
                j for j, c in enumerate(self.risk_columns_)
                if any(t in c for t in ("acf", "nocturnal", "acrophase", "period", "sri", "upright", "cusum"))
            ]
            if act_idx:
                weights[np.ix_(low_wear, act_idx)] *= 0.25
        elif self.scoring_rule not in {"weighted_z_index", "multiplicative_hazard"}:
            raise ValueError(f"Unsupported scoring_rule: {self.scoring_rule!r}")

        eff_w = np.abs(weights) * avail
        signed = np.nan_to_num(z_mat, nan=0.0) * weights
        denom = np.sum(eff_w, axis=1)
        base_score = np.where(denom > 1e-9, np.sum(signed, axis=1) / np.maximum(denom, 1e-9), 0.0)

        if self.scoring_rule == "multiplicative_hazard":
            idx_ih = self.risk_columns_.index("risk_internet_hours")
            idx_acf = self.risk_columns_.index("risk_acf_irregularity")
            idx_sed = self.risk_columns_.index("risk_sedentary_acf")
            z_ih = np.nan_to_num(z_mat[:, idx_ih], nan=0.0)
            z_circ = 0.5 * (np.nan_to_num(z_mat[:, idx_acf], nan=0.0) + np.nan_to_num(z_mat[:, idx_sed], nan=0.0))
            hazard = np.maximum(0.0, z_ih) * np.maximum(0.0, z_circ)
            return base_score + float(self.coupling_alpha) * hazard
        return base_score

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Return continuous handcrafted severity risk scores for ``X``."""
        hc_df, risk_df = self._extract_hc_and_risk(X)
        z_mat = self._transform_z(risk_df)
        return self._apply_handcrafted_rule(z_mat, hc_df)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict integer classes ``{0, 1, 2, 3}`` via calibrated QWK thresholds."""
        return apply_thresholds(self.predict_score(X), self.thresholds_)

    def score(self, X: pd.DataFrame, y: Any) -> float:
        yt = y["sii"] if isinstance(y, pd.DataFrame) and "sii" in y.columns else y
        return quadratic_weighted_kappa(yt, self.predict(X))

    def get_feature_weights_df(self) -> pd.DataFrame:
        """Return inspectable table of handcrafted risk components and fixed domain weights."""
        return pd.DataFrame({
            "risk_term": self.risk_columns_,
            "prior_weight": self.prior_weight_vec_,
            "train_mean": [self.feature_means_[c] for c in self.risk_columns_],
            "train_std": [self.feature_stds_[c] for c in self.risk_columns_],
            "scoring_rule": self.scoring_rule,
        })
