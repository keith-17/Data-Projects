"""base_utils.py

Modular ``SequenceExtractor`` suite powered by verified open-source time-series
libraries (``seglearn``, ``statsmodels``, ``PyWavelets``, ``pyts``, ``scipy.signal``,
``scikit-optimize``) plus ``OpenStrap/analytics``, ``cr7_kimmitch``, and
``dsp-slides.pdf`` algorithms — with **ZERO supervised ML models**.

Lineage:
* Windowing & kinematic features: ``seglearn.transform.Segment``, ``FeatureRep``,
  ``PadTrunc``, and ``seglearn.feature_functions`` (``base_features``, safe
  ``all_features`` kinematic/EMG descriptors).
* Econometric & seasonal periodicity: ``statsmodels.tsa.stattools.pacf``,
  ``statsmodels.tsa.seasonal.seasonal_decompose``, and ``scipy.signal.lombscargle``.
* Time-frequency & recurrence dynamics: ``scipy.signal.stft``/``welch``,
  ``pywt.cwt`` + ``pywt.wavedec`` (DWT), and ``pyts.image.RecurrencePlot``.
* Sleep / circadian / motion physiology (``OpenStrap/analytics`` + ``cr7_kimmitch`` +
  ``mad-lab-fau/sleep_analysis``): van Hees 5-deg/5-min sustained inactivity,
  Phillips Sleep Regularity Index (SRI), Fisher circular social jetlag,
  calibration-invariant ``dynAmp`` & MAD, spherical wrist turn angle & upright
  sitting proxy, Bauer PRSA ``AC/DC``, Page CUSUM, Cole-Kripke / Sadeh / Webster
  sleep-wake architecture, causal vs centered NAT windows, and full missingness
  + leak-free imputation experimentation inside ``SequenceExtractor``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate as sp_interp, signal as sps
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import StandardScaler

try:
    import pywt
    PYWT_AVAILABLE = True
except Exception:  # pragma: no cover
    pywt = None
    PYWT_AVAILABLE = False

try:
    from seglearn import feature_functions as seg_ff
    from seglearn.transform import FeatureRep, PadTrunc, Segment
    SEGLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    seg_ff = FeatureRep = PadTrunc = Segment = None
    SEGLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import pacf as sm_pacf
    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover
    seasonal_decompose = sm_pacf = None
    STATSMODELS_AVAILABLE = False

try:
    from pyts.image import RecurrencePlot
    PYTS_AVAILABLE = True
except Exception:  # pragma: no cover
    RecurrencePlot = None
    PYTS_AVAILABLE = False

try:
    from skopt import Optimizer
    from skopt.space import Categorical as SkCategorical, Integer as SkInteger, Real as SkReal
    SKOPT_AVAILABLE = True
except Exception:  # pragma: no cover
    Optimizer = SkCategorical = SkInteger = SkReal = None
    SKOPT_AVAILABLE = False

try:
    import pycatch22
    PYCATCH22_AVAILABLE = True
except Exception:  # pragma: no cover
    pycatch22 = None
    PYCATCH22_AVAILABLE = False

try:
    from sktime.transformations.panel.catch22 import Catch22 as SktimeCatch22
    CATCH22_AVAILABLE = True
    SKTIME_AVAILABLE = True
except Exception:  # pragma: no cover
    SktimeCatch22 = None
    CATCH22_AVAILABLE = False
    SKTIME_AVAILABLE = importlib.util.find_spec("sktime") is not None

try:
    import tsaug
    TSAUG_AVAILABLE = True
except Exception:  # pragma: no cover
    tsaug = None
    TSAUG_AVAILABLE = False

try:
    import tsfel
    TSFEL_AVAILABLE = True
except Exception:  # pragma: no cover
    tsfel = None
    TSFEL_AVAILABLE = False

try:
    import tsfresh
    from tsfresh import extract_features as _tsfresh_extract
    from tsfresh.feature_extraction import MinimalFCParameters as _TsfreshMinimalFC
    TSFRESH_AVAILABLE = True
except Exception:  # pragma: no cover
    tsfresh = _tsfresh_extract = _TsfreshMinimalFC = None
    TSFRESH_AVAILABLE = False

try:
    from tsflex.features import (
        FeatureCollection as TsflexFeatureCollection,
        MultipleFeatureDescriptors as TsflexMultipleFeatureDescriptors,
    )
    TSFLEX_AVAILABLE = True
except Exception:  # pragma: no cover
    TsflexFeatureCollection = TsflexMultipleFeatureDescriptors = None
    TSFLEX_AVAILABLE = False

try:
    import functime  # noqa: F401
    FUNCTIME_AVAILABLE = True
except Exception:  # pragma: no cover
    FUNCTIME_AVAILABLE = False

TSAI_AVAILABLE = importlib.util.find_spec("tsai") is not None

_CATCH22_CACHE: Dict[bool, Any] = {}
_TSFEL_DOMAIN_CACHE: Dict[str, Dict[str, Any]] = {}

try:
    from baseline_utils import (
        LEAKY_PREFIX,
        LEAKY_TARGET_COLS,
        RAW_ACTIGRAPHY_FS_HZ,
        apply_thresholds,
        bin_actigraphy_series,
        bounded_gap_fill,
        compute_circular_sleep_regularity,
        compute_cusum_volatility,
        compute_phillips_sri,
        compute_prsa_capacity,
        compute_welch_circadian_spectrum,
        contiguous_true_runs,
        find_fundamental_acf_period,
        fit_circadian_cosinor_24h,
        masked_unbiased_autocovariance,
        optimize_thresholds_qwk,
        quadratic_weighted_kappa,
    )
except ImportError:
    from Src.baseline_utils import (
        LEAKY_PREFIX,
        LEAKY_TARGET_COLS,
        RAW_ACTIGRAPHY_FS_HZ,
        apply_thresholds,
        bin_actigraphy_series,
        bounded_gap_fill,
        compute_circular_sleep_regularity,
        compute_cusum_volatility,
        compute_phillips_sri,
        compute_prsa_capacity,
        compute_welch_circadian_spectrum,
        contiguous_true_runs,
        find_fundamental_acf_period,
        fit_circadian_cosinor_24h,
        masked_unbiased_autocovariance,
        optimize_thresholds_qwk,
        quadratic_weighted_kappa,
    )


ID_COL = "id"
TARGET_COL = "sii"
PATH_COL = "series_path"

_SEQ_FEATURE_CACHE: Dict[Tuple[Any, ...], Dict[str, float]] = {}


class InvalidParams(ValueError):
    """Raised when a hyperparameter candidate violates DSP, Nyquist, or schema bounds."""


class InvalidExtractorParams(InvalidParams):
    """Alias matching uploaded ``base_utils.InvalidExtractorParams``."""


def library_status() -> Dict[str, bool]:
    """Report exact local availability of open-source time-series / DSP / optimization libraries."""
    return {
        "scipy": True,
        "pywt": bool(PYWT_AVAILABLE),
        "seglearn": bool(SEGLEARN_AVAILABLE),
        "tsaug": bool(TSAUG_AVAILABLE),
        "tsfel": bool(TSFEL_AVAILABLE),
        "tsfresh": bool(TSFRESH_AVAILABLE),
        "pycatch22": bool(PYCATCH22_AVAILABLE),
        "tsflex": bool(TSFLEX_AVAILABLE),
        "sktime": bool(SKTIME_AVAILABLE),
        "catch22_sktime": bool(CATCH22_AVAILABLE),
        "statsmodels": bool(STATSMODELS_AVAILABLE),
        "pyts": bool(PYTS_AVAILABLE),
        "skopt": bool(SKOPT_AVAILABLE),
        "tsai": bool(TSAI_AVAILABLE),
        "functime": bool(FUNCTIME_AVAILABLE),
    }


def _structured(value: Any) -> Any:
    """Decode JSON or Python-literal strings produced by skopt Categorical encoding."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] in "{[(":
        try:
            return json.loads(text)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception:
                return value
    if text == "None":
        return None
    if text == "True":
        return True
    if text == "False":
        return False
    return value


def _pipe_modes(value: Any, allowed: Iterable[str], *, name: str) -> set:
    if value is None or str(value).strip().lower() in {"", "none"}:
        return set()
    allowed_set = set(allowed)
    if str(value).strip().lower() == "all":
        return set(allowed_set)
    parts = [p.strip() for p in str(value).split("|") if p.strip()]
    unknown = [p for p in parts if p not in allowed_set]
    if unknown:
        raise InvalidExtractorParams(f"{name} received unknown modes {unknown}; allowed={sorted(allowed_set)}")
    return set(parts)


def _one_sided_psd(coefficients: np.ndarray, nfft: int) -> np.ndarray:
    """Convert real-signal one-sided STFT coefficients to one-sided PSD (``base_utils.py``)."""
    power = np.abs(coefficients) ** 2
    weights = np.full(power.shape[0], 2.0)
    weights[0] = 1.0
    if nfft % 2 == 0 and len(weights) > 1:
        weights[-1] = 1.0
    return power * weights[:, None]


def _stft_frame_coverage(
    frame_times: np.ndarray,
    signal_length: int,
    sampling_rate: float,
    nperseg: int,
    window_type: Any,
) -> np.ndarray:
    """Fraction of squared STFT window weight backed by real observed samples (``base_utils.py``)."""
    window = np.asarray(sps.get_window(window_type, nperseg, fftbins=True), dtype=float)
    sample_centers = np.rint(np.asarray(frame_times) * sampling_rate).astype(int)
    frame_samples = sample_centers[:, None] - nperseg // 2 + np.arange(nperseg, dtype=int)[None, :]
    observed = (frame_samples >= 0) & (frame_samples < signal_length)
    weights = np.square(window)
    return (observed * weights[None, :]).sum(axis=1) / max(float(weights.sum()), 1e-12)


# ---------------------------------------------------------------------------
# Data Discovery, Safe Loading & Competition Scoring
# ---------------------------------------------------------------------------

def find_data_root(local_dir: str = "Data", kaggle_root: str = "/kaggle/input") -> Path:
    """Locate CMI PIU dataset directory locally or in Kaggle."""
    cwd = Path.cwd()
    candidates = [
        cwd / local_dir,
        cwd.parent / local_dir,
        cwd / "data",
        cwd.parent / "data",
        Path(kaggle_root) / "child-mind-institute-problematic-internet-use",
        Path(kaggle_root),
    ]
    for p in candidates:
        if (p / "train.csv").exists():
            return p
    return candidates[0]


def _resolve_series_map(series_dir: Path) -> Dict[str, str]:
    if not series_dir.is_dir():
        return {}
    mapping: Dict[str, str] = {}
    for entry in os.scandir(series_dir):
        if entry.is_dir() and entry.name.startswith("id="):
            sid = entry.name.split("=", 1)[1]
            p_file = Path(entry.path) / "part-0.parquet"
            if p_file.exists():
                mapping[sid] = str(p_file)
            else:
                parquets = list(Path(entry.path).glob("*.parquet"))
                if parquets:
                    mapping[sid] = str(parquets[0])
    return mapping


def load_cmi_dataset(
    data_root: Optional[Path] = None,
    *,
    split: str = "train",
    dropna_target: bool = True,
    max_subjects: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load CMI PIU tabular data with matched ``series_path`` parquet pointers."""
    root = Path(data_root) if data_root is not None else find_data_root()
    csv_name = "train.csv" if split == "train" else "test.csv"
    series_folder = "series_train.parquet" if split == "train" else "series_test.parquet"
    df = pd.read_csv(root / csv_name)

    if split == "train" and dropna_target and TARGET_COL in df.columns:
        df = df[df[TARGET_COL].notna()].copy()
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    series_map = _resolve_series_map(root / series_folder)
    df[PATH_COL] = df[ID_COL].astype(str).map(series_map)
    df["has_actigraphy"] = df[PATH_COL].notna().astype(int)

    if max_subjects is not None and int(max_subjects) < len(df):
        if split == "train" and TARGET_COL in df.columns:
            strata = df[TARGET_COL].astype(str) + "_" + df["has_actigraphy"].astype(str)
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=int(max_subjects), random_state=int(random_state)
            )
            idx, _ = next(splitter.split(df, strata))
            df = df.iloc[idx].copy()
        else:
            df = df.sample(n=int(max_subjects), random_state=int(random_state)).copy()

    return df.reset_index(drop=True)


def strip_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip ``sii`` and all ``PCIAT-*`` columns absent from Kaggle test data."""
    keep = [c for c in df.columns if c not in LEAKY_TARGET_COLS and not str(c).startswith(LEAKY_PREFIX)]
    return df[keep]


def competition_score(y_true: Any, y_pred: Any) -> float:
    """CMI PIU competition metric: Quadratic Weighted Kappa (QWK)."""
    yt = y_true[TARGET_COL] if isinstance(y_true, pd.DataFrame) and TARGET_COL in y_true.columns else y_true
    return float(quadratic_weighted_kappa(yt, y_pred, min_rating=0, max_rating=3))


def candidate_score(estimator: Any, X: pd.DataFrame, y: Any, sample_weight: Optional[Any] = None) -> float:
    """Sklearn CV scorer returning Quadratic Weighted Kappa (competition metric)."""
    try:
        return competition_score(y, estimator.predict(X))
    except Exception:
        return -1.0


def candidate_accuracy_pct(estimator: Any, X: pd.DataFrame, y: Any, sample_weight: Optional[Any] = None) -> float:
    """Sklearn CV scorer returning normal exact-match percentage score (0.0 to 100.0; -1.0 on failure)."""
    try:
        yt = (
            np.asarray(y[TARGET_COL], dtype=np.int64).ravel()
            if isinstance(y, pd.DataFrame) and TARGET_COL in y.columns
            else np.asarray(y, dtype=np.int64).ravel()
        )
        yp = np.asarray(estimator.predict(X), dtype=np.int64).ravel()
        if len(yt) == 0 or len(yt) != len(yp):
            return -1.0
        return float(100.0 * np.mean(yt == yp))
    except Exception:
        return -1.0


def evaluate_holdout(
    y_true_df: Any,
    y_pred: Sequence[int],
    *,
    target_col: str = TARGET_COL,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate holdout predictions with normal accuracy (%), competition QWK, MAE & confusion matrix."""
    yt = (
        np.asarray(y_true_df[target_col], dtype=np.int64).ravel()
        if isinstance(y_true_df, pd.DataFrame) and target_col in y_true_df.columns
        else np.asarray(y_true_df, dtype=np.int64).ravel()
    )
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    qwk = quadratic_weighted_kappa(yt, yp)
    acc = float(np.mean(yt == yp))
    acc_pct = float(100.0 * acc)
    mae = float(mean_absolute_error(yt, yp))
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    report = classification_report(yt, yp, labels=[0, 1, 2, 3], zero_division=0, digits=4)
    if verbose:
        print(f"Holdout Normal Score (Exact Accuracy): {acc_pct:.2f}% | MAE: {mae:.4f}")
        print(f"Holdout Competition Score (QWK): {qwk:.5f}")
        print("Confusion Matrix (rows=true 0..3, cols=pred 0..3):")
        print(cm)
        print(report)
    return {
        "competition_score": qwk,
        "qwk": qwk,
        "accuracy": acc,
        "accuracy_pct": acc_pct,
        "mae": mae,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "pred_distribution": {int(k): int(v) for k, v in zip(*np.unique(yp, return_counts=True))},
    }


def export_submission(
    model: Any,
    test_df: pd.DataFrame,
    output_path: Optional[Any] = "submission.csv",
    *,
    sample_sub_path: Optional[Any] = None,
) -> pd.DataFrame:
    """Generate and validate Kaggle-compliant ``submission.csv`` (``id,sii`` integers ``0..3``)."""
    preds = np.asarray(model.predict(strip_leaky_columns(test_df)), dtype=np.int64).ravel()
    preds = np.clip(preds, 0, 3)
    sub = pd.DataFrame({ID_COL: test_df[ID_COL].astype(str).values, TARGET_COL: preds})

    if sample_sub_path is not None and Path(sample_sub_path).exists():
        ref = pd.read_csv(sample_sub_path)
        sub = ref[[ID_COL]].astype(str).merge(sub, on=ID_COL, how="left")
        sub[TARGET_COL] = sub[TARGET_COL].fillna(0).astype(np.int64)
        if len(sub) != len(ref):
            raise ValueError(f"Submission row count mismatch: got {len(sub)}, expected {len(ref)}")

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_p, index=False)
    return sub


# ---------------------------------------------------------------------------
# Modular Open-Source Sub-Extractors
# ---------------------------------------------------------------------------

class HoneycombBase(BaseEstimator, TransformerMixin):
    """Base sklearn transformer matching uploaded ``base_utils.py`` convention."""

    def fit(self, X: Any, y: Any = None):
        return self


NOOP_MOTION_ATOMS = (
    "enmo_ac_rms", "jerk_rms", "jerk_floor", "posture_var",
    "motion_cv", "sedentary_frac", "dog_var", "baevsky_si",
)
NOOP_PERIODICITY_MODES = (None, "none", "acf_circadian", "acf_kimmitch", "cosinor", "acf_kimmitch|cosinor")
NOOP_SPECTRAL_ATOMS = ("band_fractions", "spectral_entropy", "peak_period", "parseval_ratio")
NOOP_QUALITY_MODES = (None, "none", "basic")


class NoopActigraphyOptions(BaseEstimator):
    """Nested sklearn config for ``cr7_kimmitch`` / NOOP descriptors inside SequenceExtractor."""

    def __init__(
        self,
        motion_stats: Optional[str] = "kimmitch",
        periodicity: Optional[str] = "acf_kimmitch|cosinor",
        spectral_stats: Optional[str] = "all",
        quality: Optional[str] = "basic",
        detrend: str = "linear",
        period_band_hours: Tuple[float, float] = (18.0, 30.0),
        fundamental_ratio: float = 0.85,
        min_acf_strength: float = 0.08,
        nyquist_policy: str = "clip",
    ):
        self.motion_stats = motion_stats
        self.periodicity = periodicity
        self.spectral_stats = spectral_stats
        self.quality = quality
        self.detrend = detrend
        self.period_band_hours = period_band_hours
        self.fundamental_ratio = fundamental_ratio
        self.min_acf_strength = min_acf_strength
        self.nyquist_policy = nyquist_policy

    def mode_names(self) -> Tuple[set, set]:
        raw_m = self.motion_stats
        if raw_m == "kimmitch":
            raw_m = "enmo_ac_rms|jerk_rms|jerk_floor|posture_var|motion_cv|sedentary_frac|dog_var|baevsky_si"
        elif raw_m == "basic":
            raw_m = "enmo_ac_rms|sedentary_frac"
        motion = _pipe_modes(raw_m, NOOP_MOTION_ATOMS, name="noop__motion_stats")
        spectral = _pipe_modes(self.spectral_stats, NOOP_SPECTRAL_ATOMS, name="noop__spectral_stats")
        if self.periodicity not in NOOP_PERIODICITY_MODES:
            raise InvalidExtractorParams(f"noop__periodicity must be one of {NOOP_PERIODICITY_MODES}")
        if self.quality not in NOOP_QUALITY_MODES:
            raise InvalidExtractorParams(f"noop__quality must be one of {NOOP_QUALITY_MODES}")
        if self.nyquist_policy not in {"clip", "error"}:
            raise InvalidExtractorParams("noop__nyquist_policy must be 'clip' or 'error'")
        return motion, spectral


class SignalCleaner(HoneycombBase):
    """Signal cleaning, anti-alias binning, missingness/gap imputation & baseline separation."""

    def __init__(
        self,
        bin_minutes: int = 15,
        resample_method: str = "anti_alias_mean",
        non_wear_policy: str = "mask",
        interp_mode: str = "bounded_linear",
        min_bin_wear_frac: float = 0.25,
        max_gap_bins: int = 4,
        impute_long_gaps: bool = False,
        min_valid_wear_days: float = 0.0,
        clip_enmo: float = 4.0,
        linear_acc_mode: str = "rolling_mean",
        baseline_window_bins: int = 16,
        highpass_hz: float = 0.00015,
    ):
        self.bin_minutes = bin_minutes
        self.resample_method = resample_method
        self.non_wear_policy = non_wear_policy
        self.interp_mode = interp_mode
        self.min_bin_wear_frac = min_bin_wear_frac
        self.max_gap_bins = max_gap_bins
        self.impute_long_gaps = impute_long_gaps
        self.min_valid_wear_days = min_valid_wear_days
        self.clip_enmo = clip_enmo
        self.linear_acc_mode = linear_acc_mode
        self.baseline_window_bins = baseline_window_bins
        self.highpass_hz = highpass_hz

    def _add_linear_motion(self, grid: Dict[str, Any]) -> Dict[str, Any]:
        """Separate slow baseline posture/activity envelope from rapid transitions (``base_utils.py``)."""
        enmo_f, mask_f = grid["enmo_filled"], grid["mask_filled"]
        n = len(enmo_f)
        fs_bin = float(grid["fs_bin_hz"])
        mode = str(self.linear_acc_mode or "none")
        lin = np.full(n, np.nan, dtype=np.float64)

        if mode == "none":
            grid["lin_enmo"] = np.where(mask_f, enmo_f - float(np.mean(enmo_f[mask_f])), np.nan)
            return grid

        if mode == "highpass_sos":
            nyq = 0.5 * fs_bin
            fc = float(self.highpass_hz)
            if 0.0 < fc < nyq:
                sos = sps.butter(2, fc / nyq, btype="high", output="sos")
                min_len = 3 * (2 * sos.shape[0] + 1) + 1
                for s, e in contiguous_true_runs(mask_f):
                    if (e - s) > min_len:
                        lin[s:e] = sps.sosfiltfilt(sos, enmo_f[s:e])
                    else:
                        lin[s:e] = enmo_f[s:e] - float(np.mean(enmo_f[s:e]))
            else:
                mode = "rolling_mean"

        if mode == "rolling_mean":
            if int(self.baseline_window_bins) <= 0:
                grid["lin_enmo"] = np.where(mask_f, enmo_f - float(np.mean(enmo_f[mask_f])), np.nan)
                return grid
            w = max(3, int(self.baseline_window_bins))
            k = np.ones(w, dtype=np.float64)
            num = np.convolve(np.where(mask_f, enmo_f, 0.0), k, mode="same")
            den = np.convolve(mask_f.astype(np.float64), k, mode="same")
            base = np.where(den >= 1.0, num / np.maximum(den, 1.0), np.nan)
            lin = np.where(mask_f & np.isfinite(base), enmo_f - base, np.nan)

        grid["lin_enmo"] = lin
        return grid

    @staticmethod
    def _impute_1d_channel(
        values: np.ndarray,
        obs_mask: np.ndarray,
        hour_of_day: np.ndarray,
        *,
        mode: str,
        max_gap: int,
        impute_long_gaps: bool,
        nonnegative: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Impute missing actigraphy bins honestly without distorting non-imputed runs."""
        x = np.asarray(values, dtype=np.float64).copy()
        m = np.asarray(obs_mask, dtype=bool).copy()
        n = len(x)
        obs_idx = np.flatnonzero(m & np.isfinite(x))
        if len(obs_idx) == 0 or mode == "none":
            return x, m

        obs_vals = x[obs_idx]
        global_med = float(np.median(obs_vals))
        lo_bound = 0.0 if nonnegative else float(np.min(obs_vals))
        hi_bound = float(np.max(obs_vals))

        target_fill = np.zeros(n, dtype=bool)
        if impute_long_gaps:
            target_fill = ~m
        elif max_gap > 0:
            for s, e in contiguous_true_runs(~m):
                is_internal = (s > 0) and (e < n)
                if is_internal and (e - s) <= int(max_gap):
                    target_fill[s:e] = True

        if not target_fill.any():
            return x, m

        idx_fill = np.flatnonzero(target_fill)
        if mode == "zero":
            fill_val = 0.0 if nonnegative else global_med
            x[idx_fill] = fill_val
        elif mode == "bounded_ffill":
            s_full = pd.Series(np.where(m, x, np.nan)).ffill().bfill().fillna(global_med).to_numpy()
            x[idx_fill] = s_full[idx_fill]
        elif mode == "bounded_pchip" and len(obs_idx) >= 4:
            pchip = sp_interp.PchipInterpolator(obs_idx.astype(np.float64), obs_vals, extrapolate=False)
            pred = pchip(idx_fill.astype(np.float64))
            pred = np.where(np.isfinite(pred), pred, np.interp(idx_fill, obs_idx, obs_vals))
            x[idx_fill] = np.clip(pred, lo_bound, hi_bound)
        elif mode == "bounded_cubic" and len(obs_idx) >= 4:
            cs = sp_interp.CubicSpline(obs_idx.astype(np.float64), obs_vals, extrapolate=False)
            pred = cs(idx_fill.astype(np.float64))
            pred = np.where(np.isfinite(pred), pred, np.interp(idx_fill, obs_idx, obs_vals))
            x[idx_fill] = np.clip(pred, lo_bound, hi_bound)
        elif mode == "diurnal_median":
            hod = np.asarray(hour_of_day, dtype=np.float64)
            obs_hours = np.floor(hod[obs_idx]).astype(np.int64) % 24
            hour_template = np.full(24, np.nan, dtype=np.float64)
            for h in range(24):
                hm = obs_hours == h
                if hm.any():
                    hour_template[h] = float(np.median(obs_vals[hm]))
            obs_h = np.flatnonzero(np.isfinite(hour_template))
            if len(obs_h) == 0:
                hour_template[:] = global_med
            elif len(obs_h) < 24:
                ext_idx = np.concatenate([obs_h - 24, obs_h, obs_h + 24]).astype(np.float64)
                ext_val = np.tile(hour_template[obs_h], 3)
                hour_template = np.interp(np.arange(24, dtype=np.float64), ext_idx, ext_val)
            target_h = np.floor(hod[idx_fill]).astype(np.int64) % 24
            x[idx_fill] = hour_template[target_h]
        else:
            x[idx_fill] = np.clip(np.interp(idx_fill, obs_idx, obs_vals), lo_bound, hi_bound)

        m[idx_fill] = True
        return x, m

    def clean_series(self, parquet_path_or_df: Any) -> Optional[Dict[str, Any]]:
        grid = bin_actigraphy_series(
            parquet_path_or_df,
            bin_minutes=int(self.bin_minutes),
            resample_method=("none" if self.resample_method is None else str(self.resample_method)),
            non_wear_policy=("none" if self.non_wear_policy is None else str(self.non_wear_policy)),
            min_bin_wear_frac=float(self.min_bin_wear_frac),
            clip_enmo=(None if self.clip_enmo is None else float(self.clip_enmo)),
        )
        if grid is None or not grid["mask"].any():
            return None

        bin_h = float(grid["bin_minutes"]) / 60.0
        wear_days = float(grid["mask"].sum()) * bin_h / 24.0
        if wear_days < float(self.min_valid_wear_days):
            return None

        eff_mode = "diurnal_median" if self.non_wear_policy == "impute_diurnal" else str(self.interp_mode)
        eff_long = bool(self.impute_long_gaps) or (self.non_wear_policy == "impute_diurnal")
        gap_lim = int(self.max_gap_bins)

        raw_mask = np.asarray(grid["mask"], dtype=bool).copy()
        enmo_f, mask_f = self._impute_1d_channel(
            grid["enmo"], raw_mask, grid["hour_of_day"], mode=eff_mode, max_gap=gap_lim, impute_long_gaps=eff_long, nonnegative=True
        )
        log_f, _ = self._impute_1d_channel(
            grid["log_enmo"], raw_mask, grid["hour_of_day"], mode=eff_mode, max_gap=gap_lim, impute_long_gaps=eff_long, nonnegative=True
        )
        lgt_f, _ = self._impute_1d_channel(
            grid["light"], raw_mask, grid["hour_of_day"], mode=eff_mode, max_gap=gap_lim, impute_long_gaps=eff_long, nonnegative=True
        )
        ang_f, _ = self._impute_1d_channel(
            grid["anglez"], raw_mask, grid["hour_of_day"], mode=eff_mode, max_gap=gap_lim, impute_long_gaps=eff_long, nonnegative=False
        )
        dyn_f, _ = self._impute_1d_channel(
            grid["dyn_amp"], raw_mask, grid["hour_of_day"], mode=eff_mode, max_gap=gap_lim, impute_long_gaps=eff_long, nonnegative=True
        )

        out = dict(grid)
        out["raw_mask"] = raw_mask
        out["mask_filled"] = mask_f
        out["imputed_mask"] = mask_f & (~raw_mask)
        out["enmo_filled"] = enmo_f
        out["log_enmo_filled"] = log_f
        out["light_filled"] = lgt_f
        out["anglez_filled"] = ang_f
        out["dyn_amp_filled"] = dyn_f
        return self._add_linear_motion(out)


class MotionFilter(HoneycombBase):
    """Scalar random-walk Kalman smoother & dead-reckoning excursion integrals (``base_utils.py``)."""

    def __init__(
        self,
        motion_filter_mode: Optional[str] = "kalman",
        kalman_process_noise: float = 1e-3,
        kalman_measurement_noise: float = 1e-2,
        use_dead_reckoning: bool = True,
        dead_reckoning_detrend: bool = True,
    ):
        self.motion_filter_mode = motion_filter_mode
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.use_dead_reckoning = use_dead_reckoning
        self.dead_reckoning_detrend = dead_reckoning_detrend

    def _kalman_filter_1d(self, signal: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(signal, dtype=np.float64), nan=0.0)
        if len(arr) == 0:
            return arr
        Q = float(self.kalman_process_noise)
        R = float(self.kalman_measurement_noise)
        x = float(arr[0])
        P = 1.0
        out = np.empty_like(arr)
        for i in range(len(arr)):
            P_pred = P + Q
            denom = P_pred + R
            K = (P_pred / denom) if denom > 1e-12 else 1.0
            x = x + K * (arr[i] - x)
            P = (1.0 - K) * P_pred
            out[i] = x
        return out

    @staticmethod
    def _linear_detrend(seg: np.ndarray) -> np.ndarray:
        if len(seg) <= 1:
            return seg * 0.0
        return seg - np.linspace(seg[0], seg[-1], len(seg))

    def transform_grid(self, grid: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(grid)
        mask_f = grid["mask_filled"]
        enmo_f = grid["enmo_filled"]
        dt_h = float(grid["bin_minutes"]) / 60.0

        kal = np.full_like(enmo_f, np.nan)
        if self.motion_filter_mode == "kalman":
            for s, e in contiguous_true_runs(mask_f):
                kal[s:e] = self._kalman_filter_1d(enmo_f[s:e])
        else:
            kal[mask_f] = enmo_f[mask_f]
        out["enmo_kalman"] = kal

        if self.use_dead_reckoning:
            dyn = np.where(mask_f & np.isfinite(grid["lin_enmo"]), grid["lin_enmo"], 0.0)
            vel = np.full_like(enmo_f, np.nan)
            pos = np.full_like(enmo_f, np.nan)
            for s, e in contiguous_true_runs(mask_f):
                v_seg = np.cumsum(dyn[s:e] * dt_h)
                if self.dead_reckoning_detrend:
                    v_seg = self._linear_detrend(v_seg)
                p_seg = np.cumsum(v_seg * dt_h)
                if self.dead_reckoning_detrend:
                    p_seg = self._linear_detrend(p_seg)
                vel[s:e] = v_seg
                pos[s:e] = p_seg
            out["dr_vel"] = vel
            out["dr_pos"] = pos
        return out


class ActigraphyBranchExtractor(HoneycombBase):
    """Build parallel branches (``raw|smoothed|jerk|velocity|linear|kalman``) & statistics."""

    def __init__(
        self,
        acc_modes: str = "raw|smoothed|jerk|velocity|linear|kalman",
        smooth_window_bins: int = 4,
        smooth_alpha: Optional[float] = None,
        branch_stats: str = "mean|std|p90|rms",
    ):
        self.acc_modes = acc_modes
        self.smooth_window_bins = smooth_window_bins
        self.smooth_alpha = smooth_alpha
        self.branch_stats = branch_stats

    def build_and_summarize(self, grid: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        modes = _pipe_modes(
            self.acc_modes,
            ("raw", "smoothed", "jerk", "velocity", "linear", "kalman"),
            name="acc_modes",
        )
        stats_wanted = _pipe_modes(
            self.branch_stats,
            ("mean", "std", "min", "max", "p90", "rms", "abs_mean"),
            name="branch_stats",
        )
        mask_f = grid["mask_filled"]
        enmo_f = grid["enmo_filled"]
        dt_h = float(grid["bin_minutes"]) / 60.0
        branches: Dict[str, np.ndarray] = {}

        if "raw" in modes:
            branches["enmo_raw"] = enmo_f
            branches["log_enmo_raw"] = grid["log_enmo_filled"]
            branches["dyn_amp_raw"] = grid["dyn_amp_filled"]
            branches["anglez_raw"] = grid["anglez_filled"]
            branches["light_raw"] = grid["light_filled"]

        if "smoothed" in modes:
            w = int(self.smooth_window_bins)
            sm = np.full_like(enmo_f, np.nan)
            if self.smooth_alpha is not None and float(self.smooth_alpha) > 0.0:
                a = min(1.0, float(self.smooth_alpha))
                for s, e in contiguous_true_runs(mask_f):
                    sm[s:e] = pd.Series(enmo_f[s:e]).ewm(alpha=a, adjust=False).mean().to_numpy()
            elif w > 1:
                k = np.ones(w, dtype=np.float64) / float(w)
                num = np.convolve(np.where(mask_f, enmo_f, 0.0), k, mode="same")
                den = np.convolve(mask_f.astype(np.float64), k, mode="same")
                sm = np.where(den >= 0.5, num / np.maximum(den, 1e-6), np.nan)
            else:
                sm[mask_f] = enmo_f[mask_f]
            branches["enmo_smooth"] = sm

        if "jerk" in modes:
            enmo_d = np.full_like(enmo_f, np.nan)
            ang_d = np.full_like(enmo_f, np.nan)
            adj = mask_f[1:] & mask_f[:-1]
            enmo_d[1:] = np.where(adj, np.abs(enmo_f[1:] - enmo_f[:-1]) / max(dt_h, 1e-6), np.nan)
            ang_d[1:] = np.where(adj, np.abs(grid["anglez_filled"][1:] - grid["anglez_filled"][:-1]) / max(dt_h, 1e-6), np.nan)
            branches["enmo_jerk"] = enmo_d
            branches["anglez_rate"] = ang_d

        if "velocity" in modes and "dr_vel" in grid:
            branches["dr_velocity"] = grid["dr_vel"]
            branches["dr_displacement"] = grid["dr_pos"]

        if "linear" in modes and "lin_enmo" in grid:
            branches["lin_enmo"] = grid["lin_enmo"]

        if "kalman" in modes and "enmo_kalman" in grid:
            branches["enmo_kalman"] = grid["enmo_kalman"]
            branches["kalman_innov"] = np.where(mask_f, enmo_f - grid["enmo_kalman"], np.nan)

        feats: Dict[str, float] = {}
        for bname, arr in branches.items():
            vals = arr[np.isfinite(arr)]
            for st in stats_wanted:
                key = f"seq_br_{bname}_{st}"
                if len(vals) == 0:
                    feats[key] = np.nan
                elif st == "mean":
                    feats[key] = float(np.mean(vals))
                elif st == "std":
                    feats[key] = float(np.std(vals))
                elif st == "min":
                    feats[key] = float(np.min(vals))
                elif st == "max":
                    feats[key] = float(np.max(vals))
                elif st == "p90":
                    feats[key] = float(np.percentile(vals, 90))
                elif st == "rms":
                    feats[key] = float(np.sqrt(np.mean(np.square(vals))))
                elif st == "abs_mean":
                    feats[key] = float(np.mean(np.abs(vals)))
        return branches, feats


class SeglearnWindowFeatureExtractor(HoneycombBase):
    """Open-source ``seglearn`` (``Segment`` + ``FeatureRep``) multi-window actigraphy extractor.

    Applies ``seglearn.transform.Segment(width, overlap)`` across contiguous worn stretches of
    ``(log_enmo, dyn_amp, anglez, lin_enmo)`` followed by ``seglearn.transform.FeatureRep``
    to extract statistical, kinematic/EMG (``waveform_length``, ``zero_crossings``,
    ``slope_sign_changes``, ``willison_amplitude``, ``emg_var``), and cross-channel ``corr``
    summaries across diurnal sliding windows (mean & inter-window std).
    """

    def __init__(
        self,
        seglearn_window_days: float = 1.0,
        seglearn_overlap: float = 0.5,
        seglearn_feature_set: str = "safe_all",
    ):
        self.seglearn_window_days = seglearn_window_days
        self.seglearn_overlap = seglearn_overlap
        self.seglearn_feature_set = seglearn_feature_set

    @staticmethod
    def _feature_dict(mode: str) -> Dict[str, Any]:
        if mode is None or str(mode).strip().lower() in {"", "none"}:
            return {}
        if not SEGLEARN_AVAILABLE or seg_ff is None:
            return {}
        if mode == "base":
            return dict(seg_ff.base_features())
        if mode == "emg_kinematic":
            all_f = seg_ff.all_features()
            keep = (
                "mean", "std", "mad", "variation", "skew", "kurt",
                "mean_abs_diff", "corr", "zero_crossings", "slope_sign_changes",
                "waveform_length", "emg_var", "willison_amplitude",
            )
            return {k: all_f[k] for k in keep if k in all_f}
        # Default 'safe_all': excludes gmean/hmean (log on negative anglez) & hist4
        all_f = seg_ff.all_features()
        return {k: v for k, v in all_f.items() if k not in {"gmean", "hmean", "hist4"}}

    def extract(self, grid: Dict[str, Any], branches: Mapping[str, np.ndarray]) -> Dict[str, float]:
        if not SEGLEARN_AVAILABLE or Segment is None or FeatureRep is None:
            return {}
        f_dict = self._feature_dict(str(self.seglearn_feature_set))
        if not f_dict:
            return {}

        runs = contiguous_true_runs(grid["mask_filled"])
        bins_day = max(4, int(round(24.0 * 60.0 / max(int(grid["bin_minutes"]), 1))))
        width = max(8, int(round(float(self.seglearn_window_days) * bins_day)))

        ch_names = ("log_enmo", "dyn_amp", "anglez", "lin_enmo")
        seqs: List[np.ndarray] = []
        for s, e in runs:
            if (e - s) >= width:
                mat = np.column_stack([
                    np.nan_to_num(grid["log_enmo_filled"][s:e], nan=0.0),
                    np.nan_to_num(grid["dyn_amp_filled"][s:e], nan=0.0),
                    np.nan_to_num(grid["anglez_filled"][s:e], nan=0.0),
                    np.nan_to_num(branches.get("lin_enmo", grid.get("lin_enmo", np.zeros(grid["n_bins"])))[s:e], nan=0.0),
                ]).astype(np.float64)
                seqs.append(mat)

        if not seqs:
            return {"seq_seglearn_n_windows": 0.0}

        seg = Segment(width=width, overlap=float(np.clip(self.seglearn_overlap, 0.0, 0.85)))
        y_dummy = np.zeros(len(seqs), dtype=np.int64)
        Xt, _, _ = seg.fit_transform(seqs, y_dummy)
        if len(Xt) == 0:
            return {"seq_seglearn_n_windows": 0.0}

        rep = FeatureRep(features=f_dict)
        F = rep.fit_transform(Xt)
        out: Dict[str, float] = {"seq_seglearn_n_windows": float(len(Xt))}
        labels = list(rep.f_labels)
        for idx, lbl in enumerate(labels):
            col_vals = F[:, idx]
            fin = np.isfinite(col_vals)
            # Clean readable feature naming replacing channel index _0.._3 with channel name
            pretty = lbl
            for c_idx, cname in enumerate(ch_names):
                if lbl.endswith(f"_{c_idx}"):
                    pretty = f"{lbl[: -len(f'_{c_idx}')]}_{cname}"
                    break
            out[f"seq_seg_{pretty}_wmean"] = float(np.mean(col_vals[fin])) if fin.any() else np.nan
            if len(col_vals) > 1:
                out[f"seq_seg_{pretty}_wstd"] = float(np.std(col_vals[fin])) if fin.any() else np.nan
        return out


class StatsmodelsAndAutocorrExtractor(HoneycombBase):
    """Unbiased ACF + ``Cr7_kimmitch`` PpgHr + ``statsmodels`` PACF & seasonal_decompose + ``lombscargle``."""

    def __init__(
        self,
        acf_detrend: str = "linear",
        acf_bias: str = "unbiased",
        short_lags_h: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
        period_band_hours: Tuple[float, float] = (18.0, 30.0),
        fundamental_ratio: float = 0.85,
        min_acf_strength: float = 0.08,
        sedentary_threshold_g: float = 0.025,
    ):
        self.acf_detrend = acf_detrend
        self.acf_bias = acf_bias
        self.short_lags_h = short_lags_h
        self.period_band_hours = period_band_hours
        self.fundamental_ratio = fundamental_ratio
        self.min_acf_strength = min_acf_strength
        self.sedentary_threshold_g = sedentary_threshold_g

    def extract(self, grid: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        mask = grid["mask"]
        fs_bin = float(grid["fs_bin_hz"])
        bin_m = int(grid["bin_minutes"])
        bin_h = float(bin_m) / 60.0
        bins_day = max(4, int(round(24.0 / bin_h)))
        max_lag = min(grid["n_bins"] - 1, max(3 * bins_day + 4, int(round(72.0 / bin_h))))

        acf_log, _ = masked_unbiased_autocovariance(
            grid["log_enmo"], mask, max_lag=max_lag, min_pair_count=max(12, bins_day // 2),
            detrend=str(self.acf_detrend), bias=str(self.acf_bias), normalize=True,
        )
        acf_raw, _ = masked_unbiased_autocovariance(
            grid["enmo"], mask, max_lag=min(max_lag, bins_day + 4), min_pair_count=max(12, bins_day // 2),
            detrend=str(self.acf_detrend), bias=str(self.acf_bias), normalize=True,
        )
        acf_dyn, _ = masked_unbiased_autocovariance(
            grid["dyn_amp"], mask, max_lag=min(max_lag, bins_day + 4), min_pair_count=max(12, bins_day // 2),
            detrend=str(self.acf_detrend), bias=str(self.acf_bias), normalize=True,
        )
        acf_lgt, _ = masked_unbiased_autocovariance(
            np.log1p(np.maximum(grid["light"], 0.0)), mask, max_lag=min(max_lag, bins_day + 4),
            min_pair_count=max(12, bins_day // 2), detrend=str(self.acf_detrend), bias=str(self.acf_bias), normalize=True,
        )
        acf_ang, _ = masked_unbiased_autocovariance(
            grid["anglez"], mask, max_lag=min(max_lag, bins_day + 4), min_pair_count=max(12, bins_day // 2),
            detrend=str(self.acf_detrend), bias=str(self.acf_bias), normalize=True,
        )

        def _pk(arr: np.ndarray, tgt_h: float, tol_h: float = 1.0) -> float:
            c = int(round(tgt_h / bin_h))
            r = max(0, int(round(tol_h / bin_h)))
            lo, hi = max(1, c - r), min(len(arr) - 1, c + r)
            if lo > hi:
                return np.nan
            sub = arr[lo : hi + 1]
            return float(np.nanmax(sub)) if np.isfinite(sub).any() else np.nan

        out["seq_acf_log_24h"] = _pk(acf_log, 24.0, 1.0)
        out["seq_acf_log_12h"] = _pk(acf_log, 12.0, 0.5)
        out["seq_acf_log_48h"] = _pk(acf_log, 48.0, 1.5)
        out["seq_acf_log_72h"] = _pk(acf_log, 72.0, 2.0)
        out["seq_acf_enmo_24h"] = _pk(acf_raw, 24.0, 1.0)
        out["seq_acf_dyn_amp_24h"] = _pk(acf_dyn, 24.0, 1.0)
        out["seq_acf_light_24h"] = _pk(acf_lgt, 24.0, 1.0)
        out["seq_acf_anglez_24h"] = _pk(acf_ang, 24.0, 1.0)

        short_vals = []
        raw_lags = _structured(self.short_lags_h)
        lag_seq = () if (raw_lags is None or str(raw_lags).strip().lower() in {"", "none"}) else raw_lags
        for idx_l, lh in enumerate(lag_seq):
            k = max(1, int(round(float(lh) / bin_h)))
            val = float(acf_log[k]) if k < len(acf_log) and np.isfinite(acf_log[k]) else np.nan
            out[f"seq_acf_lag{idx_l}"] = val
            if np.isfinite(val):
                short_vals.append(val)
        out["seq_acf_short_mean"] = float(np.mean(short_vals)) if short_vals else np.nan

        k4 = min(len(acf_log) - 1, max(2, int(round(4.0 / bin_h))))
        out["seq_acf_decay_tau_h"] = float(np.sum(np.maximum(np.nan_to_num(acf_log[1 : k4 + 1], nan=0.0), 0.0)) * bin_h)

        sed_ind = np.where(mask, (grid["enmo"] < float(self.sedentary_threshold_g)).astype(np.float64), np.nan)
        acf_sed, _ = masked_unbiased_autocovariance(
            sed_ind, mask, max_lag=max(4, int(round(4.0 / bin_h))),
            min_pair_count=12, detrend="constant", bias=str(self.acf_bias), normalize=True,
        )
        sed_k = max(1, int(round(0.5 / bin_h)))
        out["seq_acf_sedentary_30m"] = float(acf_sed[sed_k]) if sed_k < len(acf_sed) and np.isfinite(acf_sed[sed_k]) else np.nan

        pinfo = find_fundamental_acf_period(
            acf_log, fs_bin,
            period_band_hours=tuple(map(float, _structured(self.period_band_hours))),
            fundamental_ratio=float(self.fundamental_ratio),
            min_strength=float(self.min_acf_strength),
        )
        out["seq_circadian_period_h"] = pinfo["period_h"]
        out["seq_circadian_period_err_h"] = abs(pinfo["period_h"] - 24.0) if np.isfinite(pinfo["period_h"]) else np.nan
        out["seq_circadian_peak_acf"] = pinfo["peak_acf"]
        out["seq_circadian_concavity"] = pinfo["concavity"]

        # statsmodels PACF & additive seasonal decomposition on longest contiguous filled span
        runs = contiguous_true_runs(grid["mask_filled"])
        if STATSMODELS_AVAILABLE and sm_pacf is not None and seasonal_decompose is not None and runs:
            s, e = max(runs, key=lambda r: r[1] - r[0])
            seg = grid["log_enmo_filled"][s:e]
            if len(seg) >= 24 and float(np.std(seg)) > 1e-9:
                max_p_lag = min(12, len(seg) // 2 - 1)
                if max_p_lag >= 4:
                    p_vals = sm_pacf(seg, nlags=max_p_lag, method="yw")
                    out["seq_sm_pacf_lag1"] = float(p_vals[1])
                    out["seq_sm_pacf_lag2"] = float(p_vals[2])
                    out["seq_sm_pacf_lag4"] = float(p_vals[4])
            if len(seg) >= 2 * bins_day and float(np.std(seg)) > 1e-9:
                dec = seasonal_decompose(seg, model="additive", period=bins_day, extrapolate_trend="freq")
                var_sr = float(np.var(dec.seasonal + dec.resid))
                var_tr = float(np.var(dec.trend + dec.resid))
                var_r = float(np.var(dec.resid))
                out["seq_sm_seasonal_strength"] = float(max(0.0, 1.0 - var_r / max(var_sr, 1e-12)))
                out["seq_sm_trend_strength"] = float(max(0.0, 1.0 - var_r / max(var_tr, 1e-12)))

        # scipy.signal.lombscargle over exact uneven worn timestamps near 24h circadian band (HRVFreqDomain.swift)
        if int(mask.sum()) >= 36:
            th = grid["t_hours"][mask] * 3600.0
            yv = grid["log_enmo"][mask]
            yv = yv - float(np.mean(yv))
            if float(np.std(yv)) > 1e-9:
                periods_h = np.linspace(18.0, 30.0, 25)
                ang_freqs = 2.0 * np.pi / (periods_h * 3600.0)
                lsp_pow = sps.lombscargle(th, yv, ang_freqs, precenter=True, normalize=True)
                best_idx = int(np.argmax(lsp_pow))
                out["seq_lsp_circadian_norm_power"] = float(lsp_pow[best_idx])
                out["seq_lsp_peak_period_h"] = float(periods_h[best_idx])
        return out


class SleepRegularityAndOrientationExtractor(HoneycombBase):
    """``OpenStrap/analytics`` van Hees SIB, Phillips SRI, circular SJL, dynAmp, MAD & posture proxy."""

    def extract(self, grid: Dict[str, Any]) -> Dict[str, float]:
        mask = grid["mask"]
        bin_m = int(grid["bin_minutes"])
        bins_day = max(4, int(round(24.0 * 60.0 / max(bin_m, 1))))
        hod = grid["hour_of_day"]
        out: Dict[str, float] = {}

        sib_vals = grid.get("van_hees_sib", np.full(len(mask), np.nan))
        sleep_m = mask & np.isfinite(sib_vals) & (sib_vals >= 0.65)
        out["seq_van_hees_immobile_frac"] = float(np.mean(sleep_m[mask])) if mask.any() else np.nan

        sri_res = compute_phillips_sri(sleep_m, mask, bins_day)
        out["seq_sri"] = sri_res["sri"]
        out["seq_sri_pair_std"] = sri_res["sri_pair_std"]

        cs_res = compute_circular_sleep_regularity(sib_vals, mask, hod, grid["weekday"], bin_m)
        for k, v in cs_res.items():
            out[f"seq_{k}"] = v

        # Classic Cosinor + IS / IV / M10-L5
        y = grid["log_enmo"]
        enmo = grid["enmo"]
        cos_all = fit_circadian_cosinor_24h(grid["t_hours"], y, mask, min_bins=24)
        for k, v in cos_all.items():
            out[f"seq_cosinor_{k}"] = v

        vy = y[mask]
        var_all = float(np.var(vy)) if len(vy) > 1 else np.nan
        if len(vy) >= 24 and np.isfinite(var_all) and var_all > 1e-12:
            gm = float(np.mean(vy))
            hi_int = np.floor(hod[mask]).astype(np.int64) % 24
            hm = [float(np.mean(vy[hi_int == h])) for h in range(24) if (hi_int == h).any()]
            out["seq_circadian_IS"] = float(np.clip(np.mean(np.square(np.asarray(hm) - gm)) / var_all, 0.0, 1.0)) if len(hm) >= 12 else np.nan
            adj = mask[1:] & mask[:-1]
            out["seq_circadian_IV"] = float(np.clip(np.mean(np.square(y[1:][adj] - y[:-1][adj])) / var_all, 0.0, 3.0)) if adj.sum() >= 12 else np.nan
            h_prof = np.array([float(np.mean(enmo[mask][hi_int == h])) if (hi_int == h).any() else np.nan for h in range(24)])
            if np.isfinite(h_prof).sum() >= 18:
                ext_p = np.concatenate([h_prof, h_prof])
                m10_w = [float(np.nanmean(ext_p[h : h + 10])) for h in range(24) if np.isfinite(ext_p[h : h + 10]).any()]
                l5_w = [float(np.nanmean(ext_p[h : h + 5])) for h in range(24) if np.isfinite(ext_p[h : h + 5]).any()]
                if m10_w and l5_w:
                    m10, l5 = max(m10_w), min(l5_w)
                    out["seq_m10"], out["seq_l5"] = float(m10), float(l5)
                    out["seq_m10_l5_ra"] = float((m10 - l5) / max(m10 + l5, 1e-6))

        # Calibration-invariant motion (dynAmp, MAD, g_ref) + spherical rotation + upright night sitting
        out["seq_g_ref_cal"] = float(grid.get("g_ref_cal", 1.0))
        if "dyn_amp" in grid and np.isfinite(grid["dyn_amp"][mask]).any():
            out["seq_dyn_amp_mean"] = float(np.nanmean(grid["dyn_amp"][mask]))
            out["seq_dyn_amp_p90"] = float(np.nanpercentile(grid["dyn_amp"][mask], 90))
        if "mad" in grid and np.isfinite(grid["mad"][mask]).any():
            out["seq_mad_mean"] = float(np.nanmean(grid["mad"][mask]))
        if "rot_deg" in grid and np.isfinite(grid["rot_deg"][mask]).any():
            out["seq_rot_deg_mean"] = float(np.nanmean(grid["rot_deg"][mask]))
        night_m = mask & ((hod >= 22.0) | (hod < 6.0))
        late_m = mask & ((hod >= 23.0) | (hod < 4.0))
        day_m = mask & (hod >= 8.0) & (hod < 21.0)
        if "rot_deg" in grid and night_m.any() and np.isfinite(grid["rot_deg"][night_m]).any():
            out["seq_rot_deg_night"] = float(np.nanmean(grid["rot_deg"][night_m]))
        if "upright_frac" in grid:
            if late_m.any() and np.isfinite(grid["upright_frac"][late_m]).any():
                out["seq_upright_night_frac"] = float(np.nanmean(grid["upright_frac"][late_m]))
            if day_m.any() and np.isfinite(grid["upright_frac"][day_m]).any():
                out["seq_upright_day_frac"] = float(np.nanmean(grid["upright_frac"][day_m]))

        d_val = float(np.mean(enmo[day_m])) if day_m.any() else np.nan
        n_val = float(np.mean(enmo[night_m])) if night_m.any() else np.nan
        out["seq_night_day_enmo_ratio"] = float(n_val / max(d_val, 1e-4)) if np.isfinite(d_val) and np.isfinite(n_val) else np.nan
        out["seq_late_night_light_p90"] = float(np.percentile(grid["light"][late_m], 90)) if late_m.any() else np.nan
        return out


class SleepArchitectureAndMultiWindowExtractor(HoneycombBase):
    """``mad-lab-fau/sleep_analysis`` sleep-wake heuristics, Webster endpoints & NAT multi-window dynamics.

    Implements:
    * Cole-Kripke (1992), Sadeh (1994), and Webster (1982) rescoring sleep fractions
      plus nocturnal sleep endpoints (TST, WASO, wake bouts, fragmentation index)
      mirrored from ``sleep_analysis/classification/heuristic_algorithms`` & ``scoring.py``.
    * Causal vs centered rolling windows & NAT band density mirrored from
      ``sleep_analysis/feature_extraction/mesa_datasst/actigraphy.py``.
    * Moving-average baseline subtraction + first-order derivative (FOD) burstiness
      mirrored from ``sleep_analysis/preprocessing/d04_main/movement_radar.py``.
    """

    def __init__(self, fau_windows: Tuple[int, ...] = (4, 8), nat_thresholds_g: Tuple[float, float, float] = (0.02, 0.05, 0.15)):
        self.fau_windows = fau_windows
        self.nat_thresholds_g = nat_thresholds_g

    def extract(self, grid: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {
            "seq_cole_kripke_sleep_frac": np.nan,
            "seq_sadeh_sleep_frac": np.nan,
            "seq_webster_sleep_frac": np.nan,
            "seq_sadeh_night_efficiency": np.nan,
            "seq_sadeh_vanhees_agree": np.nan,
            "seq_webster_tst_h": np.nan,
            "seq_webster_waso_min": np.nan,
            "seq_webster_wake_bouts": np.nan,
            "seq_webster_fragmentation_idx": np.nan,
            "seq_fau_nat_low_frac": np.nan,
            "seq_fau_nat_mod_frac": np.nan,
            "seq_fau_nat_vig_frac": np.nan,
            "seq_fau_anyact_density": np.nan,
            "seq_fau_skew_w8": np.nan,
            "seq_fau_kurt_w8": np.nan,
            "seq_fau_causal_centered_std_diff": np.nan,
            "seq_radar_fod_burst_frac": np.nan,
            "seq_radar_fod_burst_energy": np.nan,
        }
        mask = grid["mask"]
        mask_f = grid.get("mask_filled", mask)
        hod = grid["hour_of_day"]
        bin_m = int(grid["bin_minutes"])
        bin_h = float(bin_m) / 60.0
        bins_day = max(4, int(round(24.0 / bin_h)))

        ck = grid.get("cole_kripke_sleep")
        sa = grid.get("sadeh_sleep")
        wb = grid.get("webster_sleep")
        sib = grid.get("van_hees_sib")

        if ck is not None and np.isfinite(ck[mask]).any():
            out["seq_cole_kripke_sleep_frac"] = float(np.nanmean(ck[mask]))
        if sa is not None and np.isfinite(sa[mask]).any():
            out["seq_sadeh_sleep_frac"] = float(np.nanmean(sa[mask]))
            night_m = mask & ((hod >= 22.0) | (hod < 6.0))
            if night_m.any() and np.isfinite(sa[night_m]).any():
                out["seq_sadeh_night_efficiency"] = float(np.nanmean(sa[night_m]))
            if sib is not None and np.isfinite(sib[mask]).any():
                sa_b = sa[mask] >= 0.5
                sib_b = sib[mask] >= 0.5
                out["seq_sadeh_vanhees_agree"] = float(np.mean(sa_b == sib_b))

        if wb is not None and np.isfinite(wb[mask]).any():
            out["seq_webster_sleep_frac"] = float(np.nanmean(wb[mask]))
            # Shifted noon-to-noon sleep endpoints (main sleep interval onset -> offset)
            t_cum_h = float(hod[0] if np.isfinite(hod[0]) else 0.0) + np.arange(len(mask), dtype=np.float64) * bin_h
            noon_day = np.floor((t_cum_h - 12.0) / 24.0).astype(np.int64)
            tst_list, waso_list, wb_list, frag_list = [], [], [], []
            wb_bin = mask & np.isfinite(wb) & (wb >= 0.5)
            for d in np.unique(noon_day):
                dm = mask & (noon_day == d)
                if int(dm.sum()) < max(8, bins_day // 2):
                    continue
                d_wb = wb_bin[dm]
                sl_runs = contiguous_true_runs(d_wb)
                if not sl_runs:
                    continue
                # Bed interval anchored from first sleep bout onset to final sleep bout offset
                s_on = sl_runs[0][0]
                s_off = sl_runs[-1][1]
                span = d_wb[s_on:s_off]
                obs_span = dm[dm][s_on:s_off]
                if len(span) < 4:
                    continue
                tst_h = float(np.sum(span) * bin_h)
                wake_runs = contiguous_true_runs(~span)
                waso_min = float(np.sum(~span) * bin_m)
                n_wake_bouts = float(len(wake_runs))
                tst_list.append(tst_h)
                waso_list.append(waso_min)
                wb_list.append(n_wake_bouts)
                frag_list.append(float(n_wake_bouts / max(tst_h, 0.25)))
            if tst_list:
                out["seq_webster_tst_h"] = float(np.mean(tst_list))
                out["seq_webster_waso_min"] = float(np.mean(waso_list))
                out["seq_webster_wake_bouts"] = float(np.mean(wb_list))
                out["seq_webster_fragmentation_idx"] = float(np.mean(frag_list))

        # Causal vs centered multi-window stats & NAT intensity bands (FAU actigraphy.py)
        enmo_f = grid["enmo_filled"]
        if int(mask_f.sum()) >= 12:
            enmo_v = enmo_f[mask_f]
            t1, t2, t3 = map(float, self.nat_thresholds_g)
            out["seq_fau_nat_low_frac"] = float(np.mean((enmo_v >= t1) & (enmo_v < t2)))
            out["seq_fau_nat_mod_frac"] = float(np.mean((enmo_v >= t2) & (enmo_v < t3)))
            out["seq_fau_nat_vig_frac"] = float(np.mean(enmo_v >= t3))
            out["seq_fau_anyact_density"] = float(np.mean(enmo_v > t1))

            sr_all = pd.Series(np.where(mask_f, enmo_f, np.nan))
            w8 = max(4, int(self.fau_windows[-1]))
            s_c = sr_all.rolling(w8, center=False, min_periods=4).std().to_numpy()
            s_ctr = sr_all.rolling(w8, center=True, min_periods=4).std().to_numpy()
            sk_c = sr_all.rolling(w8, center=False, min_periods=4).skew().to_numpy()
            ku_c = sr_all.rolling(w8, center=False, min_periods=4).kurt().to_numpy()
            valid_st = mask_f & np.isfinite(s_c) & np.isfinite(s_ctr)
            if valid_st.any():
                out["seq_fau_causal_centered_std_diff"] = float(np.mean(np.abs(s_ctr[valid_st] - s_c[valid_st])))
            if np.isfinite(sk_c[mask_f]).any():
                out["seq_fau_skew_w8"] = float(np.nanmean(sk_c[mask_f]))
            if np.isfinite(ku_c[mask_f]).any():
                out["seq_fau_kurt_w8"] = float(np.nanmean(ku_c[mask_f]))

            # Radar-style MA removal -> Z-scale -> FOD |diff| -> MinMax threshold burstiness (movement_radar.py)
            ma = sr_all.rolling(max(6, bins_day // 2), center=True, min_periods=2).mean().to_numpy()
            hp = np.where(mask_f, enmo_f - np.nan_to_num(ma, nan=float(np.mean(enmo_v))), np.nan)
            hp_obs = hp[mask_f]
            hp_z = (hp_obs - float(np.mean(hp_obs))) / max(float(np.std(hp_obs)), 1e-6)
            fod = np.abs(np.diff(hp_z, prepend=hp_z[0]))
            fod_rng = max(float(np.max(fod) - np.min(fod)), 1e-6)
            fod_norm = (fod - float(np.min(fod))) / fod_rng
            out["seq_radar_fod_burst_frac"] = float(np.mean(fod_norm > 0.20))
            out["seq_radar_fod_burst_energy"] = float(np.mean(np.where(fod_norm > 0.20, fod_norm, 0.0)))
        return out


class PRSAAndChangepointExtractor(HoneycombBase):
    """Bauer PRSA ``AC/DC`` asymmetry & two-sided Page CUSUM volatility (``OpenStrap/analytics``)."""

    def __init__(self, prsa_l: int = 2, anchor_ratio_cap: float = 0.5, cusum_k: float = 0.5, cusum_h: float = 3.0):
        self.prsa_l = prsa_l
        self.anchor_ratio_cap = anchor_ratio_cap
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h

    def extract(self, grid: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        mask = grid["mask"]
        bins_day = max(4, int(round(24.0 * 60.0 / max(int(grid["bin_minutes"]), 1))))

        p_enmo = compute_prsa_capacity(grid["log_enmo"], mask, l=int(self.prsa_l), anchor_ratio_cap=float(self.anchor_ratio_cap))
        for k, v in p_enmo.items():
            out[f"seq_{k}"] = v

        p_dyn = compute_prsa_capacity(grid["dyn_amp"], mask, l=int(self.prsa_l), anchor_ratio_cap=float(self.anchor_ratio_cap))
        out["seq_prsa_dyn_ac"] = p_dyn["prsa_ac"]
        out["seq_prsa_dyn_dc"] = p_dyn["prsa_dc"]

        c_enmo = compute_cusum_volatility(grid["log_enmo"], mask, bins_day, k=float(self.cusum_k), h=float(self.cusum_h))
        out["seq_cusum_enmo_max"] = c_enmo["cusum_max"]
        out["seq_cusum_enmo_count"] = c_enmo["cusum_count"]

        hod = grid["hour_of_day"]
        late_only = grid["mask"] & ((hod >= 22.0) | (hod < 4.0))
        c_lgt = compute_cusum_volatility(np.log1p(np.maximum(grid["light"], 0.0)), late_only, bins_day, k=float(self.cusum_k), h=float(self.cusum_h))
        out["seq_cusum_night_light_max"] = c_lgt["cusum_max"]
        return out


class STFTAndWelchExtractor(HoneycombBase):
    """Windowed ``scipy.signal.stft`` edge-corrected spectrogram + Hann ``scipy.signal.welch`` PSD."""

    def __init__(
        self,
        stft_window_days: float = 2.0,
        stft_overlap_ratio: float = 0.5,
        stft_window_type: str = "hann",
        stft_scaling: str = "psd",
        stft_detrend: str = "constant",
        stft_bands_hours: Tuple[Tuple[float, float], ...] = ((16.0, 36.0), (4.0, 16.0), (1.0, 4.0)),
        max_gap_bins: int = 4,
    ):
        self.stft_window_days = stft_window_days
        self.stft_overlap_ratio = stft_overlap_ratio
        self.stft_window_type = stft_window_type
        self.stft_scaling = stft_scaling
        self.stft_detrend = stft_detrend
        self.stft_bands_hours = stft_bands_hours
        self.max_gap_bins = max_gap_bins

    def extract(self, grid: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {
            "seq_stft_mean_power": np.nan, "seq_stft_power_std": np.nan, "seq_stft_peak_period_h": np.nan,
            "seq_stft_centroid_hz": np.nan, "seq_stft_spread_hz": np.nan, "seq_stft_entropy_bits": np.nan,
            "seq_stft_temporal_power_slope": np.nan, "seq_stft_flux_mean": np.nan, "seq_stft_flux_max": np.nan,
            "seq_stft_circadian_frac": np.nan, "seq_stft_ultradian_frac": np.nan, "seq_stft_bout_frac": np.nan,
        }
        spec = compute_welch_circadian_spectrum(grid["log_enmo"], grid["mask"], grid["fs_bin_hz"], max_gap_bins=int(self.max_gap_bins))
        for k, v in spec.items():
            out[f"seq_{k}"] = v

        runs = contiguous_true_runs(grid["mask_filled"])
        if not runs:
            return out
        s, e = max(runs, key=lambda r: r[1] - r[0])
        seg = grid["log_enmo_filled"][s:e]
        fs = float(grid["fs_bin_hz"])
        if len(seg) < 16 or fs <= 0:
            return out

        bins_per_day = max(8, int(round(86400.0 * fs)))
        nperseg = max(16, min(len(seg), int(round(float(self.stft_window_days) * bins_per_day))))
        noverlap = min(nperseg - 1, max(0, int(round(nperseg * float(self.stft_overlap_ratio)))))
        sc_mode = "psd" if self.stft_scaling == "psd" else "spectrum"
        win_arg = "boxcar" if (self.stft_window_type is None or str(self.stft_window_type).strip().lower() in {"", "none"}) else str(self.stft_window_type)
        det_arg: Any = False if (self.stft_detrend is None or self.stft_detrend is False or str(self.stft_detrend).strip().lower() in {"", "none"}) else str(self.stft_detrend)
        f, t, Zxx = sps.stft(
            seg, fs=fs, window=win_arg, nperseg=nperseg,
            noverlap=noverlap, detrend=det_arg, return_onesided=True,
            scaling=sc_mode, boundary="zeros", padded=True,
        )
        rep = _one_sided_psd(Zxx, nperseg)
        cov = _stft_frame_coverage(t, len(seg), fs, nperseg, win_arg)
        rep = np.divide(rep, cov[None, :], out=np.zeros_like(rep), where=cov[None, :] > 0)
        if rep.size == 0 or len(f) < 2:
            return out

        df = float(f[1] - f[0])
        mean_psd = np.mean(rep, axis=1)
        mean_pwr = float(np.sum(mean_psd) * df)
        frame_pwr = np.sum(rep, axis=0) * df
        weights = mean_psd * df
        wsum = float(np.sum(weights))

        out["seq_stft_mean_power"] = mean_pwr
        out["seq_stft_power_std"] = float(np.std(frame_pwr))
        if len(f) > 1 and np.sum(mean_psd[1:]) > 0:
            pk_f = float(f[1 + int(np.argmax(mean_psd[1:]))])
            out["seq_stft_peak_period_h"] = float(1.0 / (pk_f * 3600.0)) if pk_f > 1e-10 else np.nan
        centroid = float(np.sum(f * weights) / wsum) if wsum > 1e-15 else 0.0
        spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * weights) / wsum)) if wsum > 1e-15 else 0.0
        out["seq_stft_centroid_hz"] = centroid
        out["seq_stft_spread_hz"] = spread
        p = weights[weights > 0] / max(wsum, 1e-15)
        out["seq_stft_entropy_bits"] = float(-np.sum(p * np.log2(p))) if len(p) else 0.0
        if len(t) > 1:
            tc = t - float(np.mean(t))
            den = float(np.dot(tc, tc))
            out["seq_stft_temporal_power_slope"] = float(np.dot(tc, frame_pwr - np.mean(frame_pwr)) / den) if den > 1e-12 else 0.0
        else:
            out["seq_stft_temporal_power_slope"] = 0.0

        f_dist = rep / np.maximum(rep.sum(axis=0, keepdims=True), 1e-15)
        flux = np.abs(np.diff(f_dist, axis=1)).sum(axis=0) if rep.shape[1] > 1 else np.zeros(1)
        out["seq_stft_flux_mean"] = float(np.mean(flux))
        out["seq_stft_flux_max"] = float(np.max(flux))

        nyq = 0.5 * fs
        bands = tuple(tuple(map(float, b)) for b in _structured(self.stft_bands_hours))
        for bname, (lo_h, hi_h) in zip(("circadian", "ultradian", "bout"), bands):
            f_lo = 1.0 / (max(hi_h, 1e-3) * 3600.0)
            f_hi = min(nyq, 1.0 / (max(lo_h, 1e-3) * 3600.0))
            if f_hi > f_lo and mean_pwr > 1e-15:
                bm = (f >= f_lo) & (f <= f_hi)
                bp = float(np.mean(np.sum(rep[bm], axis=0) * df)) if bm.any() else 0.0
                out[f"seq_stft_{bname}_frac"] = float(bp / mean_pwr)
            else:
                out[f"seq_stft_{bname}_frac"] = 0.0
        return out


class WaveletAndRecurrenceExtractor(HoneycombBase):
    """Open-source ``pywt`` (CWT + orthogonal DWT) and ``pyts.image.RecurrencePlot`` features."""

    def __init__(
        self,
        cwt_wavelet: str = "morl",
        cwt_n_scales: int = 16,
        cwt_min_period_h: float = 2.0,
        cwt_max_period_h: float = 36.0,
        dwt_wavelet: str = "db4",
        dwt_level: int = 4,
    ):
        self.cwt_wavelet = cwt_wavelet
        self.cwt_n_scales = cwt_n_scales
        self.cwt_min_period_h = cwt_min_period_h
        self.cwt_max_period_h = cwt_max_period_h
        self.dwt_wavelet = dwt_wavelet
        self.dwt_level = dwt_level

    def extract(self, grid: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {
            "seq_cwt_mean_mag": np.nan, "seq_cwt_std_mag": np.nan, "seq_cwt_power_mean": np.nan,
            "seq_cwt_dominant_period_h": np.nan, "seq_cwt_scale_entropy": np.nan,
            "seq_cwt_circadian_energy": np.nan, "seq_cwt_ultradian_energy": np.nan,
            "seq_cwt_ultradian_circadian_ratio": np.nan,
        }
        runs = contiguous_true_runs(grid["mask_filled"])
        if not runs:
            return out
        s, e = max(runs, key=lambda r: r[1] - r[0])
        seg = grid["log_enmo_filled"][s:e]
        fs = float(grid["fs_bin_hz"])
        if len(seg) < 24 or fs <= 0:
            return out
        seg_zm = seg - float(np.mean(seg))

        # 1. PyWavelets CWT
        cwt_wav = str(self.cwt_wavelet or "none").strip()
        if PYWT_AVAILABLE and pywt is not None and cwt_wav.lower() not in {"", "none"} and int(self.cwt_n_scales) > 0:
            wlet = pywt.ContinuousWavelet(cwt_wav)
            fc = float(pywt.central_frequency(wlet))
            nyq = 0.5 * fs
            f_hi = min(0.9 * nyq, 1.0 / (max(float(self.cwt_min_period_h), 0.5) * 3600.0))
            f_lo = max(1.0 / (max(float(self.cwt_max_period_h), 4.0) * 3600.0), 2.0 * fs / max(len(seg), 1))
            if f_hi > f_lo:
                s_min = fc * fs / f_hi
                s_max = fc * fs / f_lo
                if np.isfinite([s_min, s_max]).all() and s_max > s_min:
                    widths = np.geomspace(max(1.0, s_min), s_max, max(4, int(self.cwt_n_scales)))
                    coeffs, freqs = pywt.cwt(seg_zm, widths, cwt_wav, sampling_period=1.0 / fs)
                    mag = np.abs(coeffs)
                    valid = np.ones(mag.shape, dtype=bool)
                    half_w = max(abs(float(wlet.lower_bound)), abs(float(wlet.upper_bound)))
                    for idx, sc in enumerate(widths):
                        edge = int(np.ceil(half_w * float(sc) * 0.5))
                        if edge > 0:
                            valid[idx, : min(edge, len(seg))] = False
                            valid[idx, max(0, len(seg) - edge) :] = False
                    if not valid.any():
                        valid[:] = True
                    valid_counts = valid.sum(axis=1)
                    scale_mean = np.divide(np.where(valid, mag, 0.0).sum(axis=1), np.maximum(valid_counts, 1))
                    valid_vals = mag[valid]
                    periods_h = 1.0 / (np.maximum(freqs, 1e-12) * 3600.0)

                    out["seq_cwt_mean_mag"] = float(np.mean(valid_vals))
                    out["seq_cwt_std_mag"] = float(np.std(valid_vals))
                    out["seq_cwt_power_mean"] = float(np.mean(np.square(valid_vals)))
                    out["seq_cwt_dominant_period_h"] = float(periods_h[int(np.argmax(scale_mean))])
                    p_s = scale_mean[scale_mean > 0] / max(float(np.sum(scale_mean)), 1e-15)
                    out["seq_cwt_scale_entropy"] = float(-np.sum(p_s * np.log2(p_s))) if len(p_s) else 0.0

                    circ_m = (periods_h >= 16.0) & (periods_h <= 36.0)
                    ult_m = (periods_h >= 4.0) & (periods_h < 16.0)
                    circ_e = float(np.mean(scale_mean[circ_m])) if circ_m.any() else np.nan
                    ult_e = float(np.mean(scale_mean[ult_m])) if ult_m.any() else np.nan
                    out["seq_cwt_circadian_energy"] = circ_e
                    out["seq_cwt_ultradian_energy"] = ult_e
                    out["seq_cwt_ultradian_circadian_ratio"] = (
                        float(ult_e / max(circ_e, 1e-9)) if np.isfinite(circ_e) and np.isfinite(ult_e) else np.nan
                    )

        # 2. PyWavelets multi-resolution orthogonal DWT energy distribution
        dwt_wav = str(self.dwt_wavelet or "none").strip()
        if PYWT_AVAILABLE and pywt is not None and dwt_wav.lower() not in {"", "none"} and int(self.dwt_level) > 0:
            max_lvl = pywt.dwt_max_level(len(seg_zm), dwt_wav)
            lvl = max(0, min(int(self.dwt_level), max_lvl))
            if lvl >= 1:
                d_coeffs = pywt.wavedec(seg_zm, dwt_wav, level=lvl)
                energies = np.array([float(np.sum(np.square(c))) for c in d_coeffs], dtype=np.float64)
                e_tot = max(float(np.sum(energies)), 1e-12)
                fracs = energies / e_tot
                out["seq_dwt_approx_frac"] = float(fracs[0])
                out["seq_dwt_detail_high_frac"] = float(fracs[-1])
                p_d = fracs[fracs > 0]
                out["seq_dwt_entropy"] = float(-np.sum(p_d * np.log2(p_d)) / max(np.log2(len(fracs)), 1.0)) if len(fracs) > 1 else 0.0

        # 3. pyts RecurrencePlot on folded daily profile / contiguous diurnal span
        if PYTS_AVAILABLE and RecurrencePlot is not None:
            bins_day = max(4, int(round(86400.0 * fs)))
            tail_len = min(len(seg_zm), 2 * bins_day)
            if tail_len >= 24 and float(np.std(seg_zm[-tail_len:])) > 1e-9:
                rp = RecurrencePlot(threshold="point", percentage=20)
                R = rp.fit_transform(seg_zm[-tail_len:][None, :])[0]
                out["seq_pyts_rec_rate"] = float(np.mean(R))
                # Diagonal recurrence at +/- 1 day shift measures nonlinear 24h recurrence determinism
                if tail_len > bins_day + 4:
                    diag_24 = np.diag(R, k=bins_day)
                    out["seq_pyts_rec_diag24h"] = float(np.mean(diag_24)) if len(diag_24) else np.nan
        return out


TSFEL_DOMAINS = ("statistical", "temporal", "spectral")
TSFEL_DEFAULT_FEATURES = (
    "Mean", "Standard deviation", "Skewness", "Kurtosis", "Interquartile range", "Max", "Min",
    "Zero crossing rate", "Mean absolute diff", "Entropy", "Slope", "Peak to peak distance",
    "Max power spectrum", "Spectral centroid", "Spectral entropy", "Fundamental frequency",
)
TSFRESH_PROFILES = ("minimal", "efficient", "none", None)
AUGMENTATIONS = (
    "jitter", "noise", "scaling", "magnitude_warp", "time_warp",
    "time_shift", "crop", "temporal_mask", "channel_drop",
)
_TIMING_OPS = {"time_warp", "time_shift", "crop"}


def _clean_lib_col(name: str) -> str:
    return (
        str(name)
        .replace('"', "")
        .replace("'", "")
        .replace(" ", "_")
        .replace("__", "_")
    )


class TsfelVectorizedExtractor(HoneycombBase):
    """Real ``tsfel`` + ``tsfresh`` (``utils_v3.py``) multi-domain static extractor.

    Mirrors ``multi-branch_version_3.ipynb`` & ``utils_v3.py``:
    * Dispatches to installed ``tsfel.time_series_features_extractor`` for the requested
      ``tsfel_domains`` and ``tsfel_features`` (with duplicate-label protection on ``tsfel==0.2.0``).
    * Dispatches to installed ``tsfresh.extract_features`` using circadian/FFT/autocorrelation
      ``MinimalFCParameters`` / ``efficient`` feature sets (single-process ``n_jobs=0``).
    * Falls back cleanly to the vectorized 16-feature implementation when requested or if absent.
    """

    def __init__(
        self,
        static_backend: str = "tsfel+tsfresh",
        tsfel_domains: str = "statistical|temporal|spectral",
        tsfel_features: Optional[Any] = None,
        tsfresh_profile: str = "minimal",
        tsfresh_fft_bands: Any = ((0.00001, 0.00003), (0.00003, 0.00012), (0.00012, 0.00035)),
        static_channels: str = "log_enmo|dyn_amp|anglez|light|lin_enmo|enmo_kalman",
    ):
        self.static_backend = static_backend
        self.tsfel_domains = tsfel_domains
        self.tsfel_features = tsfel_features
        self.tsfresh_profile = tsfresh_profile
        self.tsfresh_fft_bands = tsfresh_fft_bands
        self.static_channels = static_channels

    def _wanted_tsfel_features(self) -> Tuple[set, Any]:
        domains = _pipe_modes(self.tsfel_domains, TSFEL_DOMAINS, name="tsfel_domains")
        raw = _structured(self.tsfel_features)
        if isinstance(raw, dict):
            return domains, raw
        if raw is None:
            wanted = set(TSFEL_DEFAULT_FEATURES)
        elif isinstance(raw, str) and raw.strip().lower() in {"", "none"}:
            return domains, set()
        elif isinstance(raw, str) and "|" in raw:
            wanted = {f.strip() for f in raw.split("|") if f.strip()}
        elif isinstance(raw, str):
            wanted = {raw.strip()}
        else:
            wanted = set(raw)
        return domains, wanted

    def _tsfel_config(self) -> Dict[str, Any]:
        domains, wanted = self._wanted_tsfel_features()
        if isinstance(wanted, dict):
            cfg: Dict[str, Any] = {}
            for dom, feats_req in wanted.items():
                if not isinstance(feats_req, Mapping):
                    continue
                if dom not in _TSFEL_DOMAIN_CACHE:
                    _TSFEL_DOMAIN_CACHE[dom] = tsfel.get_features_by_domain(dom).get(dom, {})
                dom_d = _TSFEL_DOMAIN_CACHE[dom]
                sub: Dict[str, Any] = {}
                for feat, spec in feats_req.items():
                    use_val = spec.get("use", "yes") if isinstance(spec, Mapping) else "yes"
                    if str(use_val).lower() == "no":
                        continue
                    if isinstance(spec, Mapping) and "function" in spec:
                        sub[feat] = dict(spec)
                    elif feat in dom_d:
                        merged = dict(dom_d[feat], use="yes")
                        if isinstance(spec, Mapping) and isinstance(spec.get("parameters"), Mapping):
                            p = dict(merged.get("parameters", {}) or {})
                            p.update(spec["parameters"])
                            merged["parameters"] = p
                        sub[feat] = merged
                if sub:
                    cfg[dom] = sub
            return cfg
        if not wanted:
            return {}
        cfg = {}
        for dom in sorted(domains):
            if dom not in _TSFEL_DOMAIN_CACHE:
                _TSFEL_DOMAIN_CACHE[dom] = tsfel.get_features_by_domain(dom).get(dom, {})
            dom_d = _TSFEL_DOMAIN_CACHE[dom]
            sub = {feat: dict(spec, use="yes") for feat, spec in dom_d.items() if feat in wanted}
            if sub:
                cfg[dom] = sub
        return cfg

    def _tsfresh_settings(self, fs: float, n: int) -> Dict[str, Any]:
        if str(self.tsfresh_profile) == "minimal" and _TsfreshMinimalFC is not None:
            settings = dict(_TsfreshMinimalFC())
            settings["mean_abs_change"] = None
            settings["absolute_sum_of_changes"] = None
            settings["skewness"] = None
            settings["kurtosis"] = None
            return settings
        bands = _structured(self.tsfresh_fft_bands)
        coeffs = []
        for lo, hi in bands:
            centre = 0.5 * (float(lo) + float(hi))
            k = int(round(centre * n / max(float(fs), 1e-12)))
            if 0 < k <= n // 2:
                coeffs.append(k)
        settings: Dict[str, Any] = {
            "mean": None, "standard_deviation": None, "skewness": None, "kurtosis": None,
            "minimum": None, "maximum": None, "median": None, "root_mean_square": None,
            "mean_abs_change": None, "absolute_sum_of_changes": None,
            "fft_aggregated": [{"aggtype": a} for a in ("centroid", "variance", "skew", "kurtosis")],
        }
        if coeffs:
            settings["fft_coefficient"] = [{"coeff": k, "attr": "abs"} for k in sorted(set(coeffs))]
        bins_day = max(4, int(round(86400.0 * fs)))
        lags = sorted({lag for lag in (2, max(2, bins_day // 4), bins_day) if lag < max(n - 2, 2)})
        if lags:
            settings["autocorrelation"] = [{"lag": lag} for lag in lags]
        settings["number_peaks"] = [{"n": 2}, {"n": 4}]
        return settings

    @staticmethod
    def _extract_1d_fallback(vals: np.ndarray, fs: float, domains: set) -> Dict[str, float]:
        out: Dict[str, float] = {}
        n = len(vals)
        if n < 8:
            return out
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        centered = vals - mean_v
        if "statistical" in domains:
            q25, q75 = np.percentile(vals, [25.0, 75.0])
            out.update({"mean": mean_v, "std": std_v, "iqr": float(q75 - q25), "max": float(np.max(vals)), "min": float(np.min(vals))})
            z = centered / max(std_v, 1e-12)
            out["skew"] = float(np.mean(z ** 3)) if std_v > 1e-12 else 0.0
            out["kurtosis"] = float(np.mean(z ** 4) - 3.0) if std_v > 1e-12 else 0.0
        if "temporal" in domains:
            diffs = np.diff(vals)
            above = vals >= np.median(vals)
            t = np.arange(n, dtype=np.float64) - 0.5 * (n - 1)
            den = float(np.dot(t, t))
            out.update({
                "mad": float(np.mean(np.abs(diffs))),
                "p2p": float(np.ptp(vals)),
                "zcr": float(np.mean(above[1:] != above[:-1])),
                "slope": float(np.dot(t, centered) / den) if den > 1e-12 else 0.0,
            })
        if "spectral" in domains:
            hann = np.hanning(n)
            psd = (np.abs(np.fft.rfft(centered * hann)) ** 2) / max(float(np.sum(hann ** 2)), 1e-12)
            freqs = np.fft.rfftfreq(n, d=1.0 / max(fs, 1e-12))
            if len(freqs) > 1 and np.sum(psd[1:]) > 1e-15:
                p1, f1 = psd[1:], freqs[1:]
                prob = p1 / float(np.sum(p1))
                pos = prob[prob > 0]
                out["max_psd"] = float(np.max(p1))
                out["spectral_centroid"] = float(np.sum(f1 * prob))
                out["spectral_entropy"] = float(-np.sum(pos * np.log2(pos)) / max(np.log2(len(p1)), 1.0)) if len(p1) > 1 else 0.0
                out["fundamental_freq"] = float(f1[int(np.argmax(p1))])
            else:
                out["max_psd"] = out["spectral_centroid"] = out["spectral_entropy"] = out["fundamental_freq"] = 0.0
        return out

    def extract(self, grid: Dict[str, Any], branches: Mapping[str, np.ndarray], *, run_tsfel: bool = True, run_tsfresh: bool = True) -> Dict[str, float]:
        backend = str(self.static_backend or "tsfel+tsfresh").lower()
        if backend == "none":
            return {}
        domains = _pipe_modes(self.tsfel_domains, TSFEL_DOMAINS, name="tsfel_domains")
        channels = _pipe_modes(
            self.static_channels,
            ("log_enmo", "dyn_amp", "anglez", "light", "lin_enmo", "enmo_kalman"),
            name="static_channels",
        )
        if not channels:
            return {}

        cand_map = {
            "log_enmo": grid.get("log_enmo_filled"),
            "dyn_amp": grid.get("dyn_amp_filled"),
            "anglez": grid.get("anglez_filled"),
            "light": grid.get("light_filled"),
            "lin_enmo": branches.get("lin_enmo", grid.get("lin_enmo")),
            "enmo_kalman": branches.get("enmo_kalman", grid.get("enmo_kalman")),
        }
        fs = float(grid["fs_bin_hz"])
        mask_f = grid.get("mask_filled", grid["mask"])
        runs = contiguous_true_runs(mask_f)
        if not runs:
            return {}
        s, e = max(runs, key=lambda r: r[1] - r[0])
        if (e - s) < 16:
            return {}

        ch_data: Dict[str, np.ndarray] = {}
        for ch in sorted(channels):
            arr = cand_map.get(ch)
            if arr is not None:
                seg = np.asarray(arr[s:e], dtype=np.float64)
                if np.isfinite(seg).sum() >= 12:
                    ch_data[ch] = np.nan_to_num(seg, nan=float(np.nanmedian(seg)))
        if not ch_data:
            return {}

        out: Dict[str, float] = {}
        tsf_prof = str(self.tsfresh_profile or "none").strip().lower()
        want_tsfel = run_tsfel and backend in {"tsfel", "tsfel+tsfresh", "auto"}
        want_tsfresh = run_tsfresh and (tsf_prof not in {"", "none"}) and backend in {"tsfresh", "tsfel+tsfresh", "auto"}

        if want_tsfel and domains:
            if TSFEL_AVAILABLE and tsfel is not None:
                cfg = self._tsfel_config()
                if cfg:
                    frame = pd.DataFrame(ch_data)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df_t = tsfel.time_series_features_extractor(cfg, frame, fs=fs, verbose=0)
                    for col in df_t.columns:
                        val = float(df_t[col].iloc[0])
                        out[f"seq_tsfel_{_clean_lib_col(col)}"] = val if np.isfinite(val) else np.nan
            else:
                for ch, vals in ch_data.items():
                    for k, v in self._extract_1d_fallback(vals, fs, domains).items():
                        out[f"seq_tsfel_{ch}_{k}"] = v
        elif backend == "vectorized" and domains:
            for ch, vals in ch_data.items():
                for k, v in self._extract_1d_fallback(vals, fs, domains).items():
                    out[f"seq_tsfel_{ch}_{k}"] = v

        if want_tsfresh and TSFRESH_AVAILABLE and _tsfresh_extract is not None:
            n_seg = len(next(iter(ch_data.values())))
            settings = self._tsfresh_settings(fs, n_seg)
            t_axis = np.arange(n_seg, dtype=np.int64)
            frames = [
                pd.DataFrame({"id": 0, "time": t_axis, "kind": ch, "value": vals})
                for ch, vals in ch_data.items()
            ]
            df_long = pd.concat(frames, ignore_index=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_fr = _tsfresh_extract(
                    df_long,
                    column_id="id",
                    column_sort="time",
                    column_kind="kind",
                    column_value="value",
                    default_fc_parameters=settings,
                    disable_progressbar=True,
                    n_jobs=0,
                )
            for col in df_fr.columns:
                val = float(df_fr[col].iloc[0])
                out[f"seq_tsfresh_{_clean_lib_col(col)}"] = val if np.isfinite(val) else np.nan
        return out


class Catch22SktimeExtractor(HoneycombBase):
    """Canonical ``catch22`` / ``catch24`` extractor via ``pycatch22`` (C++ fast-path) or ``sktime.Catch22``."""

    def __init__(self, catch24: bool = True, catch22_channels: str = "log_enmo|dyn_amp"):
        self.catch24 = catch24
        self.catch22_channels = catch22_channels

    @staticmethod
    def _get_sktime_transformer(catch24: bool):
        if not CATCH22_AVAILABLE or SktimeCatch22 is None:
            return None
        key = bool(catch24)
        if key not in _CATCH22_CACHE:
            _CATCH22_CACHE[key] = SktimeCatch22(catch24=key, replace_nans=True, col_names="short_str_feat")
        return _CATCH22_CACHE[key]

    def extract(self, grid: Dict[str, Any]) -> Dict[str, float]:
        if not PYCATCH22_AVAILABLE and not CATCH22_AVAILABLE:
            return {}
        channels = _pipe_modes(self.catch22_channels, ("log_enmo", "dyn_amp", "anglez", "lin_enmo"), name="catch22_channels")
        if not channels:
            return {}
        runs = contiguous_true_runs(grid["mask_filled"])
        if not runs:
            return {}
        s, e = max(runs, key=lambda r: r[1] - r[0])
        if (e - s) < 24:
            return {}

        cand_map = {
            "log_enmo": grid.get("log_enmo_filled"),
            "dyn_amp": grid.get("dyn_amp_filled"),
            "anglez": grid.get("anglez_filled"),
            "lin_enmo": grid.get("lin_enmo"),
        }
        out: Dict[str, float] = {}
        c24 = bool(self.catch24)

        if PYCATCH22_AVAILABLE and pycatch22 is not None:
            for ch in sorted(channels):
                arr = cand_map.get(ch)
                if arr is None:
                    continue
                seg = np.asarray(arr[s:e], dtype=np.float64)
                if len(seg) < 24 or float(np.nanstd(seg)) <= 1e-12:
                    continue
                seg_clean = np.nan_to_num(seg, nan=float(np.nanmedian(seg)))
                res = pycatch22.catch22_all(seg_clean.tolist(), catch24=c24)
                for nm, val in zip(res["names"], res["values"]):
                    vf = float(val)
                    out[f"seq_c22_{ch}_{nm}"] = vf if np.isfinite(vf) else np.nan
            return out

        # Fallback: sktime Catch22
        c22 = self._get_sktime_transformer(c24)
        if c22 is None:
            return {}
        step = max(1, (e - s) // 672)
        arrays, valid_names = [], []
        for ch in sorted(channels):
            arr = cand_map.get(ch)
            if arr is None:
                continue
            seg = np.asarray(arr[s:e:step], dtype=np.float64)
            if len(seg) >= 24 and float(np.nanstd(seg)) > 1e-12:
                arrays.append(np.nan_to_num(seg, nan=float(np.nanmedian(seg))))
                valid_names.append(ch)
        if not arrays:
            return {}
        L = min(len(a) for a in arrays)
        panel = np.stack([a[:L] for a in arrays], axis=0)[:, None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_df = c22.fit_transform(panel)
        for row_i, ch in enumerate(valid_names):
            for col in c_df.columns:
                val = float(c_df[col].iloc[row_i])
                out[f"seq_c22_{ch}_{col}"] = val if np.isfinite(val) else np.nan
        return out


class TsflexMultiScaleExtractor(HoneycombBase):
    """Multi-scale strided circadian envelope extractor powered by ``tsflex.features.FeatureCollection``."""

    def __init__(
        self,
        tsflex_windows: Tuple[str, ...] = ("2h", "6h", "12h"),
        tsflex_strides: str = "2h",
        tsflex_channels: str = "log_enmo|dyn_amp",
    ):
        self.tsflex_windows = tsflex_windows
        self.tsflex_strides = tsflex_strides
        self.tsflex_channels = tsflex_channels

    def extract(self, grid: Dict[str, Any]) -> Dict[str, float]:
        if not TSFLEX_AVAILABLE or TsflexFeatureCollection is None or TsflexMultipleFeatureDescriptors is None:
            return {}
        channels = _pipe_modes(self.tsflex_channels, ("log_enmo", "dyn_amp", "anglez", "lin_enmo"), name="tsflex_channels")
        if not channels:
            return {}
        runs = contiguous_true_runs(grid["mask_filled"])
        if not runs:
            return {}
        s, e = max(runs, key=lambda r: r[1] - r[0])
        if (e - s) < 16:
            return {}

        cand_map = {
            "log_enmo": grid.get("log_enmo_filled"),
            "dyn_amp": grid.get("dyn_amp_filled"),
            "anglez": grid.get("anglez_filled"),
            "lin_enmo": grid.get("lin_enmo"),
        }
        cols: Dict[str, np.ndarray] = {}
        for ch in sorted(channels):
            arr = cand_map.get(ch)
            if arr is not None:
                seg = np.asarray(arr[s:e], dtype=np.float64)
                cols[ch] = np.nan_to_num(seg, nan=float(np.nanmedian(seg)) if np.isfinite(seg).any() else 0.0)
        if not cols:
            return {}

        raw_wins = _structured(self.tsflex_windows)
        if raw_wins is None or (isinstance(raw_wins, str) and raw_wins.strip().lower() in {"", "none"}):
            return {}
        wins = [str(w) for w in raw_wins if str(w).strip().lower() not in {"", "none"}]
        if not wins:
            return {}
        bin_m = int(grid["bin_minutes"])
        idx = pd.date_range("2024-01-01", periods=e - s, freq=f"{bin_m}min")
        df_ts = pd.DataFrame(cols, index=idx)
        fc = TsflexFeatureCollection(
            [
                TsflexMultipleFeatureDescriptors(
                    functions=[np.mean, np.std, np.ptp],
                    series_names=list(cols.keys()),
                    windows=wins,
                    strides=str(self.tsflex_strides),
                )
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_fl = fc.calculate(df_ts, return_df=True, show_progress=False)

        out: Dict[str, float] = {}
        if isinstance(df_fl, pd.DataFrame) and len(df_fl) > 0:
            for col in df_fl.columns:
                vals = pd.to_numeric(df_fl[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
                cname = _clean_lib_col(col)
                if len(vals) > 0:
                    out[f"seq_tsflex_{cname}_mean"] = float(np.mean(vals))
                    out[f"seq_tsflex_{cname}_std"] = float(np.std(vals))
                    out[f"seq_tsflex_{cname}_p90"] = float(np.percentile(vals, 90))
        return out


class TSAugmentor(BaseEstimator):
    """Time-series chunk augmentor powered by ``tsaug`` (ported from ``utils_v3.py``).

    Configurable via sklearn ``estimator__augmentor__*`` params in Grid/Bayesian searches.
    Baseline defaults set ``prob=0.0`` (inactive during baseline fit unless enabled).
    """

    def __init__(
        self,
        prob: float = 0.0,
        per_aug_prob: float = 0.5,
        augmentations: Optional[Any] = None,
        n_copies: int = 1,
        jitter_sigma: float = 0.0,
        noise_std: float = 0.0,
        scaling_sigma: float = 0.0,
        warp_sigma: float = 0.0,
        warp_num_knots: int = 4,
        time_warp_max_speed_ratio: float = 1.0,
        time_warp_n_speed_change: int = 3,
        time_shift_frac: float = 0.0,
        crop_frac_range: Tuple[float, float] = (1.0, 1.0),
        temporal_num_masks: int = 0,
        temporal_mask_frac: float = 0.0,
        temporal_mask_fill: str = "ffill",
        channel_drop_prob: float = 0.0,
        random_state: Optional[int] = 42,
    ):
        self.prob = prob
        self.per_aug_prob = per_aug_prob
        self.augmentations = augmentations
        self.n_copies = n_copies
        self.jitter_sigma = jitter_sigma
        self.noise_std = noise_std
        self.scaling_sigma = scaling_sigma
        self.warp_sigma = warp_sigma
        self.warp_num_knots = warp_num_knots
        self.time_warp_max_speed_ratio = time_warp_max_speed_ratio
        self.time_warp_n_speed_change = time_warp_n_speed_change
        self.time_shift_frac = time_shift_frac
        self.crop_frac_range = crop_frac_range
        self.temporal_num_masks = temporal_num_masks
        self.temporal_mask_frac = temporal_mask_frac
        self.temporal_mask_fill = temporal_mask_fill
        self.channel_drop_prob = channel_drop_prob
        self.random_state = random_state

    def allowed_ops(self) -> set:
        value = _structured(self.augmentations)
        if value is None:
            return set(AUGMENTATIONS)
        if isinstance(value, str):
            if value.strip().lower() in {"", "none"}:
                return set()
            value = value.split("|")
        return {str(v).strip() for v in value if str(v).strip().lower() not in {"", "none"}}

    def active_ops(self) -> List[str]:
        allowed = self.allowed_ops()
        lo, hi = tuple(map(float, _structured(self.crop_frac_range)))
        strengths = {
            "jitter": float(self.jitter_sigma) > 0,
            "noise": float(self.noise_std) > 0,
            "scaling": float(self.scaling_sigma) > 0,
            "magnitude_warp": float(self.warp_sigma) > 0,
            "time_warp": float(self.time_warp_max_speed_ratio) > 1.0,
            "time_shift": float(self.time_shift_frac) > 0,
            "crop": hi < 1.0,
            "temporal_mask": int(self.temporal_num_masks) > 0 and float(self.temporal_mask_frac) > 0,
            "channel_drop": float(self.channel_drop_prob) > 0,
        }
        return [op for op in AUGMENTATIONS if op in allowed and strengths[op]]

    def enabled(self) -> bool:
        return int(self.n_copies) > 0 and float(self.prob) > 0 and float(self.per_aug_prob) > 0 and bool(self.active_ops())

    def _tsaug_op(self, op: str, length: int, seed: int):
        if op == "jitter":
            return tsaug.AddNoise(scale=float(self.jitter_sigma), normalize=False, seed=seed)
        if op == "noise":
            return tsaug.AddNoise(scale=float(self.noise_std), normalize=False, seed=seed)
        if op == "magnitude_warp":
            return tsaug.Drift(max_drift=float(self.warp_sigma), n_drift_points=int(self.warp_num_knots), kind="multiplicative", per_channel=True, normalize=False, seed=seed)
        if op == "time_warp":
            return tsaug.TimeWarp(n_speed_change=int(self.time_warp_n_speed_change), max_speed_ratio=float(self.time_warp_max_speed_ratio), seed=seed)
        if op == "crop":
            lo, hi = tuple(map(float, _structured(self.crop_frac_range)))
            sizes = sorted({int(np.clip(round(f * length), 2, length)) for f in np.linspace(lo, hi, 6)})
            return tsaug.Crop(size=sizes, resize=length, seed=seed)
        if op == "temporal_mask":
            n_masks = max(1, int(self.temporal_num_masks))
            size = max(1, int(round(float(self.temporal_mask_frac) * length / n_masks)))
            return tsaug.Dropout(p=float(self.temporal_mask_frac), size=[size], fill=str(self.temporal_mask_fill), per_channel=False, seed=seed)
        if op == "channel_drop":
            return tsaug.Dropout(p=float(self.channel_drop_prob), size=[length], fill=0.0, per_channel=True, seed=seed)
        return None

    def augment_array(self, x: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Augment ``(N, T, C)`` windows using installed ``tsaug`` operations."""
        arr = np.asarray(x, dtype=np.float64).copy()
        if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] < 4:
            return arr
        if rng is None:
            rng = np.random.default_rng(self.random_state)
        n, length, channels = arr.shape
        gate = rng.random(n) < float(self.prob)
        if not gate.any():
            return arr
        ref = np.maximum(np.std(arr, axis=(0, 1)), 1e-6)
        for op in self.active_ops():
            fire = gate & (rng.random(n) < float(self.per_aug_prob))
            idx = np.flatnonzero(fire)
            if not len(idx):
                continue
            sub = arr[idx]
            seed = int(rng.integers(0, 2 ** 31 - 1))
            if op == "time_shift":
                max_s = max(1, int(round(float(self.time_shift_frac) * length)))
                for r_i in range(len(sub)):
                    sh = int(rng.integers(-max_s, max_s + 1))
                    sub[r_i] = np.roll(sub[r_i], sh, axis=0)
            elif op == "scaling":
                sub = sub * rng.normal(1.0, float(self.scaling_sigma), size=(len(idx), 1, channels))
            elif TSAUG_AVAILABLE and tsaug is not None:
                aug_op = self._tsaug_op(op, length, seed)
                if aug_op is not None:
                    sub = (aug_op.augment(sub / ref) * ref) if op == "noise" else aug_op.augment(sub)
            arr[idx] = sub
        return arr.astype(x.dtype if x.dtype == np.float32 else np.float64)


class NoopPhysiologyExtractor(HoneycombBase):
    """``cr7_kimmitch`` / NOOP descriptors: DoG variability, Baevsky SI, quiescent jerk floor & sedentary bouts."""

    def __init__(self, noop: Optional[NoopActigraphyOptions] = None, sedentary_threshold_g: float = 0.025):
        self.noop = noop
        self.sedentary_threshold_g = sedentary_threshold_g

    @staticmethod
    def _dog_var(x: np.ndarray, bin_minutes: int) -> float:
        if len(x) < 6:
            return np.nan
        s_short = max(1.0, 60.0 / max(bin_minutes, 1) * 0.5)
        s_long = max(s_short + 1.0, 240.0 / max(bin_minutes, 1) * 0.5)

        def _g(arr: np.ndarray, sigma: float) -> np.ndarray:
            r = max(1, min(len(arr) // 2, int(round(3.0 * sigma))))
            g = np.arange(-r, r + 1, dtype=np.float64)
            k = np.exp(-0.5 * (g / sigma) ** 2)
            k /= k.sum()
            return np.convolve(np.pad(arr, r, mode="edge"), k, mode="valid")

        return float(np.sqrt(np.mean(np.square(_g(x, s_short) - _g(x, s_long)))))

    @staticmethod
    def _baevsky(x: np.ndarray) -> Tuple[float, float, float]:
        if len(x) < 8:
            return np.nan, np.nan, np.nan
        p2, p98 = np.percentile(x, [2.0, 98.0])
        mxdmn = float(max(0.0, p98 - p2))
        if mxdmn <= 1e-9:
            return 100.0, 0.0, 0.0
        clipped = x[(x >= p2) & (x <= p98)]
        if len(clipped) < 4:
            clipped = x
        counts, edges = np.histogram(clipped, bins=16, range=(float(p2), float(p98)))
        pk = int(np.argmax(counts))
        amo = 100.0 * float(counts[pk]) / float(len(clipped))
        mo = 0.5 * float(edges[pk] + edges[pk + 1])
        scale = max(abs(mo), float(np.std(clipped)), 1e-3)
        return amo, mxdmn, float(amo / (2.0 * scale * max(mxdmn, 1e-3)))

    def extract(self, grid: Dict[str, Any]) -> Dict[str, float]:
        mask = grid["mask"]
        bin_m = int(grid["bin_minutes"])
        bin_h = float(bin_m) / 60.0
        enmo_v = grid["enmo"][mask]
        sed_m = mask & (grid["enmo"] < float(self.sedentary_threshold_g))
        runs = contiguous_true_runs(sed_m)

        out: Dict[str, float] = {
            "seq_sedentary_frac": float(np.mean(enmo_v < float(self.sedentary_threshold_g))),
            "seq_sedentary_max_bout_h": float(max(e - s for s, e in runs) * bin_h) if runs else 0.0,
        }
        opts = self.noop if self.noop is not None else NoopActigraphyOptions()
        m_atoms, _ = opts.mode_names()
        jerk_v = grid["jerk_rms"][mask]
        jerk_v = jerk_v[np.isfinite(jerk_v)]
        if "enmo_ac_rms" in m_atoms:
            out["noop_enmo_ac_rms"] = float(np.std(enmo_v))
        if "motion_cv" in m_atoms:
            out["noop_motion_cv"] = float(np.std(enmo_v) / max(float(np.mean(enmo_v)), 1e-6))
        if "posture_var" in m_atoms:
            av = grid["anglez_var"][mask]
            out["noop_posture_var"] = float(np.nanmedian(av)) if np.isfinite(av).any() else np.nan
        if "jerk_floor" in m_atoms and len(jerk_v) > 0:
            j_med = max(float(np.median(jerk_v)), 1e-5)
            out["noop_jerk_floor"] = j_med
            out["noop_move_fraction"] = float(np.mean(jerk_v > 2.0 * j_med))
            out["noop_still_fraction"] = float(np.mean(jerk_v <= 1.25 * j_med))
            out["noop_jerk_gate_ratio"] = float(np.percentile(jerk_v, 95) / j_med)
        if "dog_var" in m_atoms and grid["mask_filled"].any():
            out["noop_dog_variability"] = self._dog_var(grid["log_enmo_filled"][grid["mask_filled"]], bin_m)
        if "baevsky_si" in m_atoms:
            amo, mxdmn, si = self._baevsky(grid["log_enmo"][mask])
            out["noop_baevsky_amo_pct"] = amo
            out["noop_baevsky_mxdmn"] = mxdmn
            out["noop_baevsky_si"] = si
        return out


DEFAULT_SEQUENCE_EXTRACTOR_PARAMS: Dict[str, Any] = {
    "native_sampling_rate": RAW_ACTIGRAPHY_FS_HZ,  # 0.2 Hz (5s -> exact van Hees 5s epoch)
    "bin_minutes": 15,                             # 96 bins/day -> Nyquist = 2 cyc/hour
    "resample_method": "anti_alias_mean",          # zero-phase Butterworth sosfiltfilt before decimation
    "non_wear_policy": "mask",                     # "mask", "keep_all", "impute_zero", "impute_diurnal"
    "interp_mode": "bounded_linear",               # "none", "bounded_linear", "bounded_pchip", "bounded_cubic", "bounded_ffill", "diurnal_median", "zero"
    "min_bin_wear_frac": 0.25,
    "max_gap_bins": 4,                             # bounded fill up to 1h gaps unless impute_long_gaps=True
    "impute_long_gaps": False,
    "min_valid_wear_days": 0.0,
    "clip_enmo": 4.0,
    "linear_acc_mode": "rolling_mean",
    "baseline_window_bins": 16,
    "highpass_hz": 0.00015,
    "motion_filter_mode": "kalman",
    "kalman_process_noise": 1e-3,
    "kalman_measurement_noise": 1e-2,
    "use_dead_reckoning": True,
    "dead_reckoning_detrend": True,
    "acc_modes": "raw|smoothed|jerk|velocity|linear|kalman",
    "smooth_window_bins": 4,
    "smooth_alpha": None,
    "sedentary_threshold_g": 0.025,
    "feature_families": (
        "autocorr|circadian|sleep_arch|motion|spectral|stft|wavelet|seglearn|catch22|tsfel|tsfresh|tsflex|prsa_cusum|branches|compliance"
    ),
    "acf_detrend": "linear",
    "acf_bias": "unbiased",
    "short_lags_h": (0.25, 0.5, 1.0, 2.0),
    "period_band_hours": (18.0, 30.0),
    "fundamental_ratio": 0.85,
    "min_acf_strength": 0.08,
    "stft_window_days": 2.0,
    "stft_overlap_ratio": 0.5,
    "stft_window_type": "hann",
    "cwt_wavelet": "morl",
    "cwt_n_scales": 16,
    "cwt_min_period_h": 2.0,
    "cwt_max_period_h": 36.0,
    "dwt_wavelet": "db4",
    "dwt_level": 4,
    "seglearn_window_days": 1.0,
    "seglearn_overlap": 0.5,
    "seglearn_feature_set": "safe_all",
    "catch22_catch24": True,
    "catch22_channels": "log_enmo|dyn_amp",
    "static_backend": "tsfel+tsfresh",
    "tsfel_domains": "statistical|temporal|spectral",
    "tsfel_features": None,
    "tsfresh_profile": "minimal",
    "tsfresh_fft_bands": ((0.00001, 0.00003), (0.00003, 0.00012), (0.00012, 0.00035)),
    "static_channels": "log_enmo|dyn_amp|anglez|light|lin_enmo|enmo_kalman",
    "tsflex_windows": ("2h", "6h", "12h"),
    "tsflex_strides": "2h",
    "tsflex_channels": "log_enmo|dyn_amp",
    "chunk_window_days": 3.0,
    "chunk_overlap": 0.5,
    "max_windows_per_sequence": 12,
    "include_tabular_engineered": True,
    "harmonize_instruments": True,
    "missing_feature_strategy": "keep_nan",        # "keep_nan", "median", "mean", "constant_zero", "knn", "iterative", "group_age_median"
    "add_missing_indicators": True,
    "impute_knn_neighbors": 5,
    "impute_max_iter": 10,
    "use_cache": True,
}


class SequenceExtractor(HoneycombBase):
    """Modular Multi-Library ``SequenceExtractor`` for CMI PIU.

    Combines ``SignalCleaner`` (including time-series gap/diurnal imputation & missingness tracking),
    ``MotionFilter``, ``ActigraphyBranchExtractor``, ``SeglearnWindowFeatureExtractor``,
    ``StatsmodelsAndAutocorrExtractor``, ``SleepRegularityAndOrientationExtractor``,
    ``SleepArchitectureAndMultiWindowExtractor`` (``mad-lab-fau/sleep_analysis`` Cole-Kripke,
    Sadeh, Webster endpoints, causal vs centered NAT windows & FOD bursts),
    ``PRSAAndChangepointExtractor``, ``STFTAndWelchExtractor``, ``WaveletAndRecurrenceExtractor``,
    ``TsfelVectorizedExtractor``, ``NoopPhysiologyExtractor``, and fold-safe tabular/feature
    missingness & imputation experiments (``SimpleImputer``, ``KNNImputer``, ``IterativeImputer``,
    ``MissingIndicator``, ``group_age_median``).
    """

    def __init__(
        self,
        noop: Optional[NoopActigraphyOptions] = None,
        native_sampling_rate: float = RAW_ACTIGRAPHY_FS_HZ,
        bin_minutes: int = 15,
        resample_method: str = "anti_alias_mean",
        non_wear_policy: str = "mask",
        interp_mode: str = "bounded_linear",
        min_bin_wear_frac: float = 0.25,
        max_gap_bins: int = 4,
        impute_long_gaps: bool = False,
        min_valid_wear_days: float = 0.0,
        clip_enmo: float = 4.0,
        linear_acc_mode: str = "rolling_mean",
        baseline_window_bins: int = 16,
        highpass_hz: float = 0.00015,
        motion_filter_mode: Optional[str] = "kalman",
        kalman_process_noise: float = 1e-3,
        kalman_measurement_noise: float = 1e-2,
        use_dead_reckoning: bool = True,
        dead_reckoning_detrend: bool = True,
        acc_modes: str = "raw|smoothed|jerk|velocity|linear|kalman",
        smooth_window_bins: int = 4,
        smooth_alpha: Optional[float] = None,
        sedentary_threshold_g: float = 0.025,
        feature_families: str = "autocorr|circadian|sleep_arch|motion|spectral|stft|wavelet|seglearn|catch22|tsfel|tsfresh|tsflex|prsa_cusum|branches|compliance",
        acf_detrend: str = "linear",
        acf_bias: str = "unbiased",
        short_lags_h: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
        period_band_hours: Tuple[float, float] = (18.0, 30.0),
        fundamental_ratio: float = 0.85,
        min_acf_strength: float = 0.08,
        stft_window_days: float = 2.0,
        stft_overlap_ratio: float = 0.5,
        stft_window_type: str = "hann",
        cwt_wavelet: str = "morl",
        cwt_n_scales: int = 16,
        cwt_min_period_h: float = 2.0,
        cwt_max_period_h: float = 36.0,
        dwt_wavelet: str = "db4",
        dwt_level: int = 4,
        seglearn_window_days: float = 1.0,
        seglearn_overlap: float = 0.5,
        seglearn_feature_set: str = "safe_all",
        catch22_catch24: bool = True,
        catch22_channels: str = "log_enmo|dyn_amp",
        static_backend: str = "tsfel+tsfresh",
        tsfel_domains: str = "statistical|temporal|spectral",
        tsfel_features: Optional[Any] = None,
        tsfresh_profile: str = "minimal",
        tsfresh_fft_bands: Any = ((0.00001, 0.00003), (0.00003, 0.00012), (0.00012, 0.00035)),
        static_channels: str = "log_enmo|dyn_amp|anglez|light|lin_enmo|enmo_kalman",
        tsflex_windows: Tuple[str, ...] = ("2h", "6h", "12h"),
        tsflex_strides: str = "2h",
        tsflex_channels: str = "log_enmo|dyn_amp",
        chunk_window_days: float = 3.0,
        chunk_overlap: float = 0.5,
        max_windows_per_sequence: int = 12,
        include_tabular_engineered: bool = True,
        harmonize_instruments: bool = True,
        missing_feature_strategy: str = "keep_nan",
        add_missing_indicators: bool = True,
        impute_knn_neighbors: int = 5,
        impute_max_iter: int = 10,
        use_cache: bool = True,
        id_col: str = ID_COL,
        path_col: str = PATH_COL,
    ):
        self.noop = noop
        self.native_sampling_rate = native_sampling_rate
        self.bin_minutes = bin_minutes
        self.resample_method = resample_method
        self.non_wear_policy = non_wear_policy
        self.interp_mode = interp_mode
        self.min_bin_wear_frac = min_bin_wear_frac
        self.max_gap_bins = max_gap_bins
        self.impute_long_gaps = impute_long_gaps
        self.min_valid_wear_days = min_valid_wear_days
        self.clip_enmo = clip_enmo
        self.linear_acc_mode = linear_acc_mode
        self.baseline_window_bins = baseline_window_bins
        self.highpass_hz = highpass_hz
        self.motion_filter_mode = motion_filter_mode
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.use_dead_reckoning = use_dead_reckoning
        self.dead_reckoning_detrend = dead_reckoning_detrend
        self.acc_modes = acc_modes
        self.smooth_window_bins = smooth_window_bins
        self.smooth_alpha = smooth_alpha
        self.sedentary_threshold_g = sedentary_threshold_g
        self.feature_families = feature_families
        self.acf_detrend = acf_detrend
        self.acf_bias = acf_bias
        self.short_lags_h = short_lags_h
        self.period_band_hours = period_band_hours
        self.fundamental_ratio = fundamental_ratio
        self.min_acf_strength = min_acf_strength
        self.stft_window_days = stft_window_days
        self.stft_overlap_ratio = stft_overlap_ratio
        self.stft_window_type = stft_window_type
        self.cwt_wavelet = cwt_wavelet
        self.cwt_n_scales = cwt_n_scales
        self.cwt_min_period_h = cwt_min_period_h
        self.cwt_max_period_h = cwt_max_period_h
        self.dwt_wavelet = dwt_wavelet
        self.dwt_level = dwt_level
        self.seglearn_window_days = seglearn_window_days
        self.seglearn_overlap = seglearn_overlap
        self.seglearn_feature_set = seglearn_feature_set
        self.catch22_catch24 = catch22_catch24
        self.catch22_channels = catch22_channels
        self.static_backend = static_backend
        self.tsfel_domains = tsfel_domains
        self.tsfel_features = tsfel_features
        self.tsfresh_profile = tsfresh_profile
        self.tsfresh_fft_bands = tsfresh_fft_bands
        self.static_channels = static_channels
        self.tsflex_windows = tsflex_windows
        self.tsflex_strides = tsflex_strides
        self.tsflex_channels = tsflex_channels
        self.chunk_window_days = chunk_window_days
        self.chunk_overlap = chunk_overlap
        self.max_windows_per_sequence = max_windows_per_sequence
        self.include_tabular_engineered = include_tabular_engineered
        self.harmonize_instruments = harmonize_instruments
        self.missing_feature_strategy = missing_feature_strategy
        self.add_missing_indicators = add_missing_indicators
        self.impute_knn_neighbors = impute_knn_neighbors
        self.impute_max_iter = impute_max_iter
        self.use_cache = use_cache
        self.id_col = id_col
        self.path_col = path_col
        self._rebuild_components()

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = super().get_params(deep=deep)
        if deep and self.noop is None:
            params.update({f"noop__{k}": v for k, v in NoopActigraphyOptions().get_params(deep=True).items()})
        return params

    def set_params(self, **params: Any):
        clean = {k: _structured(v) for k, v in params.items()}
        if any(name.startswith("noop__") for name in clean) and clean.get("noop", self.noop) is None:
            clean["noop"] = NoopActigraphyOptions()
        super().set_params(**clean)
        self._rebuild_components()
        return self

    def _rebuild_components(self) -> None:
        raw_short_lags = _structured(self.short_lags_h)
        short_lags = () if (raw_short_lags is None or str(raw_short_lags).strip().lower() in {"", "none"}) else tuple(map(float, raw_short_lags))
        raw_tsflex_w = _structured(self.tsflex_windows)
        tsflex_wins = () if (raw_tsflex_w is None or str(raw_tsflex_w).strip().lower() in {"", "none"}) else tuple(map(str, raw_tsflex_w))

        self.cleaner = SignalCleaner(
            bin_minutes=int(self.bin_minutes),
            resample_method=("none" if self.resample_method is None else str(self.resample_method)),
            non_wear_policy=("none" if self.non_wear_policy is None else str(self.non_wear_policy)),
            interp_mode=("none" if self.interp_mode is None else str(self.interp_mode)),
            min_bin_wear_frac=float(self.min_bin_wear_frac),
            max_gap_bins=int(self.max_gap_bins),
            impute_long_gaps=bool(self.impute_long_gaps),
            min_valid_wear_days=float(self.min_valid_wear_days),
            clip_enmo=(None if self.clip_enmo is None else float(self.clip_enmo)),
            linear_acc_mode=("none" if self.linear_acc_mode is None else str(self.linear_acc_mode)),
            baseline_window_bins=int(self.baseline_window_bins),
            highpass_hz=float(self.highpass_hz),
        )
        self.motion_filter = MotionFilter(
            motion_filter_mode=self.motion_filter_mode,
            kalman_process_noise=float(self.kalman_process_noise),
            kalman_measurement_noise=float(self.kalman_measurement_noise),
            use_dead_reckoning=bool(self.use_dead_reckoning),
            dead_reckoning_detrend=bool(self.dead_reckoning_detrend),
        )
        self.branch_extractor = ActigraphyBranchExtractor(
            acc_modes=("none" if self.acc_modes is None else str(self.acc_modes)),
            smooth_window_bins=int(self.smooth_window_bins),
            smooth_alpha=self.smooth_alpha,
        )
        self.autocorr_sm_extractor = StatsmodelsAndAutocorrExtractor(
            acf_detrend=("none" if self.acf_detrend is None else str(self.acf_detrend)),
            acf_bias=str(self.acf_bias),
            short_lags_h=short_lags,
            period_band_hours=tuple(map(float, _structured(self.period_band_hours))),
            fundamental_ratio=float(self.fundamental_ratio),
            min_acf_strength=float(self.min_acf_strength),
            sedentary_threshold_g=float(self.sedentary_threshold_g),
        )
        self.sleep_sri_extractor = SleepRegularityAndOrientationExtractor()
        self.sleep_arch_extractor = SleepArchitectureAndMultiWindowExtractor()
        self.prsa_cusum_extractor = PRSAAndChangepointExtractor()
        self.stft_welch_extractor = STFTAndWelchExtractor(
            stft_window_days=float(self.stft_window_days),
            stft_overlap_ratio=float(self.stft_overlap_ratio),
            stft_window_type=("none" if self.stft_window_type is None else str(self.stft_window_type)),
            max_gap_bins=int(self.max_gap_bins),
        )
        self.wavelet_rec_extractor = WaveletAndRecurrenceExtractor(
            cwt_wavelet=("none" if self.cwt_wavelet is None else str(self.cwt_wavelet)),
            cwt_n_scales=int(self.cwt_n_scales),
            cwt_min_period_h=float(self.cwt_min_period_h),
            cwt_max_period_h=float(self.cwt_max_period_h),
            dwt_wavelet=("none" if self.dwt_wavelet is None else str(self.dwt_wavelet)),
            dwt_level=int(self.dwt_level),
        )
        self.seglearn_extractor = SeglearnWindowFeatureExtractor(
            seglearn_window_days=float(self.seglearn_window_days),
            seglearn_overlap=float(self.seglearn_overlap),
            seglearn_feature_set=("none" if self.seglearn_feature_set is None else str(self.seglearn_feature_set)),
        )
        self.catch22_extractor = Catch22SktimeExtractor(
            catch24=bool(self.catch22_catch24),
            catch22_channels=("none" if self.catch22_channels is None else str(self.catch22_channels)),
        )
        self.tsfel_extractor = TsfelVectorizedExtractor(
            static_backend=("none" if self.static_backend is None else str(self.static_backend)),
            tsfel_domains=("none" if self.tsfel_domains is None else str(self.tsfel_domains)),
            tsfel_features=self.tsfel_features,
            tsfresh_profile=("none" if self.tsfresh_profile is None else str(self.tsfresh_profile)),
            tsfresh_fft_bands=_structured(self.tsfresh_fft_bands),
            static_channels=("none" if self.static_channels is None else str(self.static_channels)),
        )
        self.tsflex_extractor = TsflexMultiScaleExtractor(
            tsflex_windows=tsflex_wins,
            tsflex_strides=str(self.tsflex_strides),
            tsflex_channels=("none" if self.tsflex_channels is None else str(self.tsflex_channels)),
        )
        self.noop_extractor = NoopPhysiologyExtractor(
            noop=self.noop if self.noop is not None else NoopActigraphyOptions(),
            sedentary_threshold_g=float(self.sedentary_threshold_g),
        )

    def validate(self) -> "SequenceExtractor":
        validate_sequence_extractor_params(self.get_params(deep=False))
        return self

    def _extract_single_actigraphy(self, parquet_path_or_df: Any, families: set) -> Dict[str, float]:
        row: Dict[str, float] = {
            "seq_has_actigraphy": 0.0,
            "seq_missing_bin_frac": 1.0,
            "seq_imputed_bin_frac": 0.0,
        }
        clean_grid = self.cleaner.clean_series(parquet_path_or_df)
        if clean_grid is None or int(clean_grid["mask"].sum()) < 16:
            return row

        row["seq_has_actigraphy"] = 1.0
        bin_h = float(clean_grid["bin_minutes"]) / 60.0
        raw_m = clean_grid["raw_mask"]
        fill_m = clean_grid["mask_filled"]
        imp_m = clean_grid["imputed_mask"]
        n_valid = int(raw_m.sum())

        row["seq_missing_bin_frac"] = float(1.0 - np.mean(raw_m))
        row["seq_imputed_bin_frac"] = float(np.sum(imp_m) / max(int(fill_m.sum()), 1))
        gap_runs = contiguous_true_runs(~raw_m)
        row["seq_max_unworn_gap_h"] = float(max((e - s for s, e in gap_runs), default=0) * bin_h)
        row["seq_n_nonwear_bouts"] = float(len(gap_runs))
        if imp_m.any() and raw_m.any():
            raw_mu = float(np.nanmean(clean_grid["enmo"][raw_m]))
            fill_mu = float(np.nanmean(clean_grid["enmo_filled"][fill_m]))
            row["seq_imputation_shift_enmo"] = float(fill_mu - raw_mu)
        else:
            row["seq_imputation_shift_enmo"] = 0.0

        if not families:
            return row

        grid = self.motion_filter.transform_grid(clean_grid)
        branches, br_feats = self.branch_extractor.build_and_summarize(grid)

        if "compliance" in families:
            row["seq_wear_days"] = float(n_valid * bin_h / 24.0)
            row["seq_wear_frac"] = float(grid["raw_wear_frac"])
            row["seq_bin_span_days"] = float(grid["n_bins"] * bin_h / 24.0)
            wd_ok = raw_m & np.isfinite(grid["weekday"])
            row["seq_weekend_wear_frac"] = float(np.mean(grid["weekday"][wd_ok] >= 6)) if wd_ok.any() else np.nan
            hod = grid["hour_of_day"]
            night_slots = (hod >= 22.0) | (hod < 6.0)
            row["seq_night_wear_frac"] = float(np.mean(raw_m[night_slots])) if night_slots.any() else np.nan

        if "branches" in families:
            row.update(br_feats)
        if "autocorr" in families:
            row.update(self.autocorr_sm_extractor.extract(grid))
        if "circadian" in families:
            row.update(self.sleep_sri_extractor.extract(grid))
        if "sleep_arch" in families:
            row.update(self.sleep_arch_extractor.extract(grid))
        if "motion" in families:
            row.update(self.noop_extractor.extract(grid))
        if "prsa_cusum" in families:
            row.update(self.prsa_cusum_extractor.extract(grid))
        if "spectral" in families or "stft" in families:
            row.update(self.stft_welch_extractor.extract(grid))
        if "wavelet" in families:
            row.update(self.wavelet_rec_extractor.extract(grid))
        if "seglearn" in families:
            row.update(self.seglearn_extractor.extract(grid, branches))
        if "catch22" in families:
            row.update(self.catch22_extractor.extract(grid))
        if "tsfel" in families or "tsfresh" in families:
            row.update(
                self.tsfel_extractor.extract(
                    grid,
                    branches,
                    run_tsfel=("tsfel" in families),
                    run_tsfresh=("tsfresh" in families),
                )
            )
        if "tsflex" in families:
            row.update(self.tsflex_extractor.extract(grid))
        return row

    @staticmethod
    def _build_tabular_block(
        X: pd.DataFrame,
        include_engineered: bool = True,
        harmonize_instruments: bool = True,
    ) -> pd.DataFrame:
        clean = strip_leaky_columns(X)
        drop_meta = {ID_COL, PATH_COL, "has_actigraphy"}
        num_cols, season_cols = [], []
        for col in clean.columns:
            if col in drop_meta or str(col).startswith(("hc_", "seq_", "noop_", "miss_")):
                continue
            if str(col).endswith("-Season") or clean[col].dtype == object:
                season_cols.append(col)
            else:
                num_cols.append(col)

        tab = clean[num_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)
        for scol in season_cols:
            tab[f"tab_{scol}_code"] = (
                clean[scol].astype(str).map({"Spring": 0.0, "Summer": 1.0, "Fall": 2.0, "Winter": 3.0}).astype(np.float64)
            )

        reconstructed_bmi = np.zeros(len(tab), dtype=np.float64)
        if harmonize_instruments and "Physical-Weight" in tab.columns and "Physical-Height" in tab.columns:
            bmi_raw = tab["Physical-BMI"] if "Physical-BMI" in tab.columns else pd.Series(np.nan, index=tab.index)
            wt = pd.to_numeric(tab["Physical-Weight"], errors="coerce")
            ht = pd.to_numeric(tab["Physical-Height"], errors="coerce")
            calc_mask = bmi_raw.isna() & (wt > 15.0) & (ht > 20.0)
            if calc_mask.any():
                bmi_est = np.clip(703.0 * wt / np.square(ht), 10.0, 60.0)
                tab.loc[calc_mask, "Physical-BMI"] = bmi_est.loc[calc_mask]
                reconstructed_bmi = calc_mask.to_numpy(dtype=np.float64)
            tab["eng_BMI_reconstructed"] = reconstructed_bmi

        if include_engineered:
            paq_a = pd.to_numeric(tab.get("PAQ_A-PAQ_A_Total", np.nan), errors="coerce")
            paq_c = pd.to_numeric(tab.get("PAQ_C-PAQ_C_Total", np.nan), errors="coerce")
            tab["eng_PAQ_harmonized"] = paq_a.combine_first(paq_c)
            age = pd.to_numeric(tab.get("Basic_Demos-Age", np.nan), errors="coerce")
            ihrs = pd.to_numeric(tab.get("PreInt_EduHx-computerinternet_hoursday", np.nan), errors="coerce")
            sds_t = pd.to_numeric(tab.get("SDS-SDS_Total_T", np.nan), errors="coerce")
            bmi = pd.to_numeric(tab.get("Physical-BMI", np.nan), errors="coerce")
            ffm = pd.to_numeric(tab.get("BIA-BIA_FFM", np.nan), errors="coerce")
            fat = pd.to_numeric(tab.get("BIA-BIA_Fat", np.nan), errors="coerce")
            wt = pd.to_numeric(tab.get("Physical-Weight", np.nan), errors="coerce")

            tab["eng_Internet_x_Age"] = ihrs * age
            tab["eng_Internet_x_SDS"] = ihrs * sds_t
            tab["eng_BMI_x_Age"] = bmi * age
            tab["eng_FFM_Weight_Ratio"] = ffm / np.maximum(wt, 1.0)
            tab["eng_Fat_FFM_Ratio"] = fat / np.maximum(ffm, 1.0)
            tab["eng_n_valid_instruments"] = clean[num_cols].notna().sum(axis=1).astype(np.float64)
        return tab

    @staticmethod
    def _age_group_labels(df: pd.DataFrame) -> np.ndarray:
        if "Basic_Demos-Age" not in df.columns:
            return np.full(len(df), "unknown", dtype=object)
        age = pd.to_numeric(df["Basic_Demos-Age"], errors="coerce").to_numpy(dtype=np.float64)
        out = np.full(len(df), "unknown", dtype=object)
        out[np.isfinite(age) & (age < 10.0)] = "child"
        out[np.isfinite(age) & (age >= 10.0) & (age < 14.0)] = "early_teen"
        out[np.isfinite(age) & (age >= 14.0)] = "adolescent"
        return out

    @staticmethod
    def _build_missing_indicators_df(tab_df: pd.DataFrame, seq_df: pd.DataFrame) -> pd.DataFrame:
        miss = pd.DataFrame(index=tab_df.index, dtype=np.float64)
        n_tab = max(tab_df.shape[1], 1)
        n_seq = max(seq_df.shape[1], 1)
        miss["miss_tab_frac"] = tab_df.isna().sum(axis=1).astype(np.float64) / float(n_tab)
        miss["miss_seq_frac"] = seq_df.isna().sum(axis=1).astype(np.float64) / float(n_seq)
        miss["miss_total_missing_cells"] = (tab_df.isna().sum(axis=1) + seq_df.isna().sum(axis=1)).astype(np.float64)
        miss["miss_actigraphy_absent"] = (
            1.0 - pd.to_numeric(seq_df.get("seq_has_actigraphy", 0.0), errors="coerce").fillna(0.0)
        )
        miss["miss_flag_internet_hours"] = (
            tab_df["PreInt_EduHx-computerinternet_hoursday"].isna().astype(np.float64)
            if "PreInt_EduHx-computerinternet_hoursday" in tab_df.columns
            else 1.0
        )
        miss["miss_flag_sds"] = tab_df["SDS-SDS_Total_T"].isna().astype(np.float64) if "SDS-SDS_Total_T" in tab_df.columns else 1.0
        paq_a = tab_df["PAQ_A-PAQ_A_Total"].isna() if "PAQ_A-PAQ_A_Total" in tab_df.columns else pd.Series(True, index=tab_df.index)
        paq_c = tab_df["PAQ_C-PAQ_C_Total"].isna() if "PAQ_C-PAQ_C_Total" in tab_df.columns else pd.Series(True, index=tab_df.index)
        miss["miss_flag_paq"] = (paq_a & paq_c).astype(np.float64)
        miss["miss_flag_cgas"] = tab_df["CGAS-CGAS_Score"].isna().astype(np.float64) if "CGAS-CGAS_Score" in tab_df.columns else 1.0
        miss["miss_flag_bia"] = tab_df["BIA-BIA_FFM"].isna().astype(np.float64) if "BIA-BIA_FFM" in tab_df.columns else 1.0
        miss["miss_flag_physical"] = tab_df["Physical-BMI"].isna().astype(np.float64) if "Physical-BMI" in tab_df.columns else 1.0
        return miss

    def _extract_raw_blocks(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.validate()
        self._rebuild_components()
        cfg_sig = self._cfg_signature()
        families = _pipe_modes(
            self.feature_families,
            (
                "autocorr", "circadian", "sleep_arch", "motion", "spectral", "stft",
                "wavelet", "seglearn", "catch22", "tsfel", "tsfresh", "tsflex",
                "prsa_cusum", "branches", "compliance",
            ),
            name="feature_families",
        )
        ids = X[self.id_col].astype(str).tolist() if self.id_col in X.columns else list(map(str, range(len(X))))
        paths = X[self.path_col].tolist() if self.path_col in X.columns else [None] * len(X)
        seq_rows: List[Dict[str, float]] = []
        for sid, p in zip(ids, paths):
            row_key = (sid, str(p), cfg_sig)
            if bool(self.use_cache) and isinstance(p, (str, Path)) and row_key in _SEQ_FEATURE_CACHE:
                seq_rows.append(_SEQ_FEATURE_CACHE[row_key])
                continue
            r = self._extract_single_actigraphy(p, families)
            if bool(self.use_cache) and isinstance(p, (str, Path)):
                if len(_SEQ_FEATURE_CACHE) > 8000:
                    _SEQ_FEATURE_CACHE.clear()
                _SEQ_FEATURE_CACHE[row_key] = r
            seq_rows.append(r)

        seq_df = pd.DataFrame(seq_rows, index=X.index, dtype=np.float64)
        tab_df = self._build_tabular_block(
            X,
            include_engineered=bool(self.include_tabular_engineered),
            harmonize_instruments=bool(self.harmonize_instruments),
        )
        return tab_df, seq_df

    def _cfg_signature(self) -> Tuple[Any, ...]:
        p = {
            k: repr(_structured(v))
            for k, v in sorted(self.get_params(deep=True).items())
            if k not in {
                "use_cache", "chunk_window_days", "chunk_overlap", "max_windows_per_sequence",
                "missing_feature_strategy", "add_missing_indicators", "impute_knn_neighbors", "impute_max_iter",
            }
        }
        return tuple(p.items())

    def fit(self, X: pd.DataFrame, y: Any = None):
        """Fit missing-data indicators and imputation statistics strictly on training samples."""
        tab_df, seq_df = self._extract_raw_blocks(X)
        raw_df = pd.concat([tab_df, seq_df], axis=1)
        self._base_feature_cols_ = list(raw_df.columns)
        self._tab_cols_ = list(tab_df.columns)
        self._miss_indicator_ = None
        self._miss_indicator_cols_: List[str] = []
        self._scaler_ = None
        self._imputer_ = None
        self._iter_core_idx_: Optional[np.ndarray] = None
        self._age_group_medians_: Dict[str, np.ndarray] = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mat = raw_df.to_numpy(dtype=np.float64)
            self._fallback_medians_ = np.nan_to_num(np.nanmedian(mat, axis=0), nan=0.0) if len(mat) else np.zeros(mat.shape[1])

            if bool(self.add_missing_indicators) and len(tab_df) > 0:
                mi = MissingIndicator(features="missing-only", error_on_new=False)
                tab_mat = tab_df.to_numpy(dtype=np.float64)
                mi.fit(tab_mat)
                self._miss_indicator_ = mi
                self._miss_indicator_cols_ = [f"miss_ind_{self._tab_cols_[idx]}" for idx in mi.features_]

            strat = "keep_nan" if (self.missing_feature_strategy is None or str(self.missing_feature_strategy).strip().lower() in {"", "none", "keep_nan"}) else str(self.missing_feature_strategy)

            if strat == "keep_nan":
                pass
            elif strat == "constant_zero":
                imp = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
                imp.fit(mat)
                self._imputer_ = imp
            elif strat in {"median", "mean"}:
                imp = SimpleImputer(strategy=strat, keep_empty_features=True)
                imp.fit(mat)
                self._imputer_ = imp
            elif strat == "group_age_median":
                groups = self._age_group_labels(X)
                for g in np.unique(groups):
                    gm = groups == g
                    if gm.any():
                        g_med = np.nanmedian(mat[gm], axis=0)
                        g_med = np.where(np.isfinite(g_med), g_med, self._fallback_medians_)
                        self._age_group_medians_[str(g)] = g_med
            elif strat == "knn":
                mat_prefill = np.where(np.isnan(mat) & np.all(np.isnan(mat), axis=0)[None, :], 0.0, mat)
                scaler = StandardScaler()
                scaled = scaler.fit_transform(mat_prefill)
                k = max(1, min(int(self.impute_knn_neighbors), max(1, len(mat) - 1)))
                imp = KNNImputer(n_neighbors=k, weights="distance", keep_empty_features=True)
                imp.fit(scaled)
                self._scaler_ = scaler
                self._imputer_ = imp
            elif strat == "iterative":
                # Run MICE cleanly on core clinical + primary domain features (avoid 500x seglearn/tsfel blow-up)
                core_idx = [
                    j for j, c in enumerate(self._base_feature_cols_)
                    if not str(c).startswith(("seq_seg_", "seq_tsfel_", "seq_tsfresh_", "seq_tsflex_", "seq_c22_"))
                ]
                if not core_idx:
                    core_idx = list(range(min(64, len(self._base_feature_cols_))))
                c_idx = np.asarray(core_idx, dtype=np.int64)
                sub = mat[:, c_idx]
                sub_prefill = np.where(np.isnan(sub) & np.all(np.isnan(sub), axis=0)[None, :], 0.0, sub)
                scaler = StandardScaler()
                scaled = scaler.fit_transform(sub_prefill)
                n_near = min(10, max(1, len(c_idx) - 1))
                imp = IterativeImputer(
                    max_iter=max(1, min(int(self.impute_max_iter), 4)),
                    n_nearest_features=n_near,
                    tol=1e-2,
                    random_state=42,
                    keep_empty_features=True,
                )
                imp.fit(scaled)
                self._iter_core_idx_ = c_idx
                self._scaler_ = scaler
                self._imputer_ = imp
            else:
                raise InvalidExtractorParams(f"Unsupported missing_feature_strategy: {strat!r}")

        self._is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract unified multi-library actigraphy + tabular features with optional leak-free imputation."""
        tab_df, seq_df = self._extract_raw_blocks(X)
        raw_df = pd.concat([tab_df, seq_df], axis=1)
        if getattr(self, "_is_fitted_", False) and getattr(self, "_base_feature_cols_", None):
            raw_df = raw_df.reindex(columns=self._base_feature_cols_)
            tab_df = tab_df.reindex(columns=self._tab_cols_)

        miss_summary = self._build_missing_indicators_df(tab_df, seq_df)
        blocks: List[pd.DataFrame] = []

        strat = "keep_nan" if (self.missing_feature_strategy is None or str(self.missing_feature_strategy).strip().lower() in {"", "none", "keep_nan"}) else str(self.missing_feature_strategy)
        if getattr(self, "_is_fitted_", False) and strat != "keep_nan":
            mat = raw_df.to_numpy(dtype=np.float64, copy=True)
            if strat == "group_age_median":
                groups = self._age_group_labels(X)
                for r_i in range(len(mat)):
                    g = str(groups[r_i])
                    fill_vec = self._age_group_medians_.get(g, self._fallback_medians_)
                    nan_cols = ~np.isfinite(mat[r_i])
                    if nan_cols.any():
                        mat[r_i, nan_cols] = fill_vec[nan_cols]
            elif strat == "knn" and self._scaler_ is not None and self._imputer_ is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bad_cols = ~np.isfinite(self._scaler_.mean_) | (self._scaler_.scale_ <= 0)
                    mat_in = mat.copy()
                    if np.any(bad_cols):
                        mat_in[:, bad_cols] = 0.0
                    scaled = self._scaler_.transform(mat_in)
                    scaled_imp = self._imputer_.transform(scaled)
                    mat_imp = self._scaler_.inverse_transform(scaled_imp)
                    if np.any(bad_cols):
                        mat_imp[:, bad_cols] = self._fallback_medians_[bad_cols]
                    mat = mat_imp
            elif strat == "iterative" and self._scaler_ is not None and self._imputer_ is not None and self._iter_core_idx_ is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    c_idx = self._iter_core_idx_
                    sub = mat[:, c_idx].copy()
                    bad_cols = ~np.isfinite(self._scaler_.mean_) | (self._scaler_.scale_ <= 0)
                    if np.any(bad_cols):
                        sub[:, bad_cols] = 0.0
                    scaled = self._scaler_.transform(sub)
                    scaled_imp = self._imputer_.transform(scaled)
                    sub_imp = self._scaler_.inverse_transform(scaled_imp)
                    if np.any(bad_cols):
                        sub_imp[:, bad_cols] = self._fallback_medians_[c_idx][bad_cols]
                    mat[:, c_idx] = sub_imp
                    # Fill remaining auxiliary descriptor columns cleanly via training medians
                    still_nan = ~np.isfinite(mat)
                    if still_nan.any():
                        r_idx, col_idx = np.where(still_nan)
                        mat[r_idx, col_idx] = self._fallback_medians_[col_idx]
            elif self._imputer_ is not None:
                mat = self._imputer_.transform(mat)
                still_nan = ~np.isfinite(mat)
                if still_nan.any():
                    r_idx, c_idx = np.where(still_nan)
                    mat[r_idx, c_idx] = self._fallback_medians_[c_idx]
            feat_out = pd.DataFrame(mat, index=raw_df.index, columns=raw_df.columns, dtype=np.float64)
        else:
            feat_out = raw_df

        blocks.append(feat_out)
        if bool(self.add_missing_indicators):
            miss_summary["miss_post_impute_applied"] = float(strat != "keep_nan")
            blocks.append(miss_summary)
            if getattr(self, "_is_fitted_", False) and self._miss_indicator_ is not None and self._miss_indicator_cols_:
                ind_mat = self._miss_indicator_.transform(tab_df.to_numpy(dtype=np.float64)).astype(np.float64)
                ind_df = pd.DataFrame(ind_mat, index=raw_df.index, columns=self._miss_indicator_cols_, dtype=np.float64)
                blocks.append(ind_df)
        return pd.concat(blocks, axis=1)

    def transform_chunks(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Use ``seglearn.transform.Segment`` & ``PadTrunc`` to build sliding multi-day tensors ``(N, T, C)``."""
        self.validate()
        self._rebuild_components()
        bins_per_day = max(4, int(round(24.0 * 60.0 / max(int(self.bin_minutes), 1))))
        width = max(8, int(round(float(self.chunk_window_days) * bins_per_day)))
        max_w = max(1, int(self.max_windows_per_sequence))

        channel_order = (
            "log_enmo_filled", "dyn_amp_filled", "enmo_smooth",
            "lin_enmo", "enmo_kalman", "anglez_filled", "light_filled",
        )
        raw_mats: List[np.ndarray] = []
        mask_mats: List[np.ndarray] = []
        imp_mats: List[np.ndarray] = []
        kept_ids: List[str] = []

        ids = X[self.id_col].astype(str).tolist() if self.id_col in X.columns else list(map(str, range(len(X))))
        paths = X[self.path_col].tolist() if self.path_col in X.columns else [None] * len(X)

        for sid, p in zip(ids, paths):
            clean_g = self.cleaner.clean_series(p)
            if clean_g is None or not clean_g["mask_filled"].any():
                continue
            g = self.motion_filter.transform_grid(clean_g)
            branches, _ = self.branch_extractor.build_and_summarize(g)
            g["enmo_smooth"] = branches.get("enmo_smooth", g["enmo_filled"])
            n = len(g["mask_filled"])
            mat = np.column_stack([np.nan_to_num(g.get(ch, np.zeros(n)), nan=0.0) for ch in channel_order]).astype(np.float32)
            raw_mats.append(mat)
            mask_mats.append(g["mask_filled"].astype(np.float32)[:, None])
            imp_mats.append(g["imputed_mask"].astype(np.float32)[:, None])
            kept_ids.append(sid)

        if not raw_mats:
            return {
                "chunks": np.zeros((0, width, len(channel_order)), dtype=np.float32),
                "masks": np.zeros((0, width), dtype=bool),
                "imputed_masks": np.zeros((0, width), dtype=bool),
                "sequence_ids": np.asarray([]),
                "channel_names": channel_order,
                "padded_full_shape": (0, 0, len(channel_order)),
            }

        if SEGLEARN_AVAILABLE and Segment is not None and PadTrunc is not None:
            pad = PadTrunc(width=width * 4)
            padded_arr, _, _ = pad.fit_transform(raw_mats, np.zeros(len(raw_mats), dtype=int))
            long_mats = [m if len(m) >= width else np.pad(m, ((0, width - len(m)), (0, 0))) for m in raw_mats]
            long_msks = [m if len(m) >= width else np.pad(m, ((0, width - len(m)), (0, 0))) for m in mask_mats]
            long_imps = [m if len(m) >= width else np.pad(m, ((0, width - len(m)), (0, 0))) for m in imp_mats]
            y_idx = np.arange(len(long_mats), dtype=int)
            seg = Segment(width=width, overlap=float(np.clip(self.chunk_overlap, 0.0, 0.85)))
            Xt, yt, _ = seg.fit_transform(long_mats, y_idx)
            Mt, _, _ = seg.fit_transform(long_msks, y_idx)
            It, _, _ = seg.fit_transform(long_imps, y_idx)
            keep_rows: List[int] = []
            for s_idx in range(len(kept_ids)):
                pos = np.flatnonzero(yt == s_idx)
                if len(pos) > max_w:
                    pos = pos[np.linspace(0, len(pos) - 1, max_w, dtype=int)]
                keep_rows.extend(pos.tolist())
            Xt = Xt[keep_rows].astype(np.float32)
            Mt = (Mt[keep_rows, :, 0] > 0.5)
            It = (It[keep_rows, :, 0] > 0.5)
            seq_ids = np.asarray([kept_ids[int(i)] for i in yt[keep_rows]])
            padded_shape = padded_arr.shape
        else:
            Xt = np.zeros((0, width, len(channel_order)), dtype=np.float32)
            Mt = np.zeros((0, width), dtype=bool)
            It = np.zeros((0, width), dtype=bool)
            seq_ids = np.asarray([])
            padded_shape = (0, 0, len(channel_order))

        return {
            "chunks": Xt,
            "masks": Mt,
            "imputed_masks": It,
            "sequence_ids": seq_ids,
            "channel_names": channel_order,
            "padded_full_shape": padded_shape,
        }


def validate_sequence_extractor_params(params: Mapping[str, Any]) -> None:
    """Validate DSP, Nyquist, STFT, wavelet, seglearn, missing/imputation & window parameters."""
    bin_min = int(params.get("bin_minutes", 15))
    if bin_min <= 0 or bin_min > 360:
        raise InvalidExtractorParams(f"bin_minutes must be in [1, 360], got {bin_min}")
    fs_bin = 1.0 / (float(bin_min) * 60.0)
    nyq = 0.5 * fs_bin
    lin_mode = params.get("linear_acc_mode", "rolling_mean")
    if lin_mode not in {"rolling_mean", "highpass_sos", "none", None}:
        raise InvalidExtractorParams(f"Unsupported linear_acc_mode: {lin_mode!r}")
    if lin_mode == "highpass_sos":
        hp = float(params.get("highpass_hz", 0.00015))
        if not (0.0 < hp < nyq):
            raise InvalidExtractorParams(f"highpass_hz ({hp}) must be < Nyquist ({nyq:.6f} Hz)")
    clip_e = params.get("clip_enmo", 4.0)
    if clip_e is not None and float(clip_e) < 0.0:
        raise InvalidExtractorParams(f"clip_enmo must be None or >= 0, got {clip_e!r}")
    if params.get("resample_method", "anti_alias_mean") not in {"anti_alias_mean", "poly", "mean", "none", None}:
        raise InvalidExtractorParams("resample_method must be 'anti_alias_mean', 'poly', 'mean', or 'none'")
    if params.get("non_wear_policy", "mask") not in {"mask", "impute_zero", "impute_diurnal", "keep_all", "none", None}:
        raise InvalidExtractorParams("non_wear_policy must be 'mask', 'impute_zero', 'impute_diurnal', 'keep_all', or 'none'")
    if params.get("motion_filter_mode", "kalman") not in {"kalman", "none", None}:
        raise InvalidExtractorParams("motion_filter_mode must be 'kalman' or 'none'")
    if params.get("interp_mode", "bounded_linear") not in {
        "none", None, "zero", "bounded_ffill", "bounded_linear", "bounded_pchip", "bounded_cubic", "diurnal_median"
    }:
        raise InvalidExtractorParams(f"Unsupported interp_mode: {params.get('interp_mode')!r}")
    if params.get("missing_feature_strategy", "keep_nan") not in {
        "keep_nan", "none", None, "median", "mean", "constant_zero", "knn", "iterative", "group_age_median"
    }:
        raise InvalidExtractorParams(f"Unsupported missing_feature_strategy: {params.get('missing_feature_strategy')!r}")
    if params.get("seglearn_feature_set", "safe_all") not in {"safe_all", "base", "emg_kinematic", "none", None}:
        raise InvalidExtractorParams(f"Unsupported seglearn_feature_set: {params.get('seglearn_feature_set')!r}")
    if params.get("acf_detrend", "linear") not in {"linear", "constant", "none", None}:
        raise InvalidExtractorParams("acf_detrend must be 'linear', 'constant', or 'none'")
    if params.get("acf_bias", "unbiased") not in {"unbiased", "biased"}:
        raise InvalidExtractorParams("acf_bias must be 'unbiased' or 'biased'")
    if float(params.get("min_bin_wear_frac", 0.25)) < 0.0 or int(params.get("max_gap_bins", 4)) < 0:
        raise InvalidExtractorParams("min_bin_wear_frac and max_gap_bins must be >= 0")
    if float(params.get("min_valid_wear_days", 0.0)) < 0.0:
        raise InvalidExtractorParams("min_valid_wear_days must be >= 0")
    if int(params.get("cwt_n_scales", 16)) < 0 or int(params.get("dwt_level", 4)) < 0:
        raise InvalidExtractorParams("cwt_n_scales and dwt_level must be >= 0")
    band = _structured(params.get("period_band_hours", (18.0, 30.0)))
    if not isinstance(band, (tuple, list)) or len(band) != 2 or not (0 < band[0] < band[1]):
        raise InvalidExtractorParams(f"period_band_hours must satisfy 0 < lo < hi, got {band!r}")
    if params.get("static_backend", "tsfel+tsfresh") not in {
        "tsfel", "tsfresh", "tsfel+tsfresh", "auto", "vectorized", "none", None
    }:
        raise InvalidExtractorParams(f"Unsupported static_backend: {params.get('static_backend')!r}")
    if params.get("tsfresh_profile", "minimal") not in TSFRESH_PROFILES:
        raise InvalidExtractorParams(f"tsfresh_profile must be one of {TSFRESH_PROFILES}")
    _pipe_modes(
        params.get(
            "feature_families",
            "autocorr|circadian|sleep_arch|motion|spectral|stft|wavelet|seglearn|catch22|tsfel|tsfresh|tsflex|prsa_cusum|branches|compliance",
        ),
        (
            "autocorr", "circadian", "sleep_arch", "motion", "spectral", "stft",
            "wavelet", "seglearn", "catch22", "tsfel", "tsfresh", "tsflex",
            "prsa_cusum", "branches", "compliance",
        ),
        name="feature_families",
    )


def inactive_sequence_parameters(params: Mapping[str, Any]) -> set:
    """Identify inactive ``extractor__*`` params so GridSearchCV deduplication skips clones."""
    p = {
        (k[len("estimator__"):] if k.startswith("estimator__") else k): _structured(v)
        for k, v in params.items()
    }
    prefix = "estimator__" if any(k.startswith("estimator__") for k in params) else ""
    inactive: set = set()
    families = _pipe_modes(
        p.get(
            "extractor__feature_families",
            "autocorr|circadian|sleep_arch|motion|spectral|stft|wavelet|seglearn|catch22|tsfel|tsfresh|tsflex|prsa_cusum|branches|compliance",
        ),
        (
            "autocorr", "circadian", "sleep_arch", "motion", "spectral", "stft",
            "wavelet", "seglearn", "catch22", "tsfel", "tsfresh", "tsflex",
            "prsa_cusum", "branches", "compliance",
        ),
        name="feature_families",
    )
    if not families:
        for name in (
            "linear_acc_mode", "baseline_window_bins", "highpass_hz", "motion_filter_mode",
            "kalman_process_noise", "kalman_measurement_noise", "use_dead_reckoning", "dead_reckoning_detrend",
            "acc_modes", "smooth_window_bins", "smooth_alpha", "sedentary_threshold_g",
        ):
            inactive.add(f"{prefix}extractor__{name}")
    if "autocorr" not in families:
        for name in ("acf_detrend", "acf_bias", "short_lags_h", "period_band_hours", "fundamental_ratio", "min_acf_strength"):
            inactive.add(f"{prefix}extractor__{name}")
    if "stft" not in families:
        for name in ("stft_window_days", "stft_overlap_ratio", "stft_window_type"):
            inactive.add(f"{prefix}extractor__{name}")
    if "wavelet" not in families:
        for name in ("cwt_wavelet", "cwt_n_scales", "cwt_min_period_h", "cwt_max_period_h", "dwt_wavelet", "dwt_level"):
            inactive.add(f"{prefix}extractor__{name}")
    else:
        cwt_w = str(p.get("extractor__cwt_wavelet", "morl") or "none").lower()
        if cwt_w == "none" or int(p.get("extractor__cwt_n_scales", 16)) <= 0:
            for name in ("cwt_n_scales", "cwt_min_period_h", "cwt_max_period_h"):
                inactive.add(f"{prefix}extractor__{name}")
        dwt_w = str(p.get("extractor__dwt_wavelet", "db4") or "none").lower()
        if dwt_w == "none" or int(p.get("extractor__dwt_level", 4)) <= 0:
            inactive.add(f"{prefix}extractor__dwt_level")
    if "seglearn" not in families or p.get("extractor__seglearn_feature_set", "safe_all") in ("none", None):
        for name in ("seglearn_window_days", "seglearn_overlap"):
            inactive.add(f"{prefix}extractor__{name}")
        if "seglearn" not in families:
            inactive.add(f"{prefix}extractor__seglearn_feature_set")
    if "catch22" not in families:
        for name in ("catch24", "catch22_catch24", "catch22_channels"):
            inactive.add(f"{prefix}extractor__{name}")
    sb = p.get("extractor__static_backend", "tsfel+tsfresh")
    if "tsfel" not in families or sb in ("tsfresh", "none", None):
        for name in ("tsfel_domains", "tsfel_features"):
            inactive.add(f"{prefix}extractor__{name}")
    if "tsfresh" not in families or sb in ("tsfel", "none", None):
        for name in ("tsfresh_profile", "tsfresh_fft_bands"):
            inactive.add(f"{prefix}extractor__{name}")
    elif p.get("extractor__tsfresh_profile", "minimal") in ("none", None):
        inactive.add(f"{prefix}extractor__tsfresh_fft_bands")
    if "tsflex" not in families:
        for name in ("tsflex_windows", "tsflex_strides", "tsflex_channels"):
            inactive.add(f"{prefix}extractor__{name}")
    if "tsfel" not in families and "tsfresh" not in families:
        for name in ("static_backend", "static_channels"):
            inactive.add(f"{prefix}extractor__{name}")
    lin_m = p.get("extractor__linear_acc_mode", "rolling_mean")
    if lin_m != "highpass_sos":
        inactive.add(f"{prefix}extractor__highpass_hz")
    if lin_m != "rolling_mean":
        inactive.add(f"{prefix}extractor__baseline_window_bins")
    if int(p.get("extractor__max_gap_bins", 4)) <= 0 and not p.get("extractor__impute_long_gaps", False) and p.get("extractor__non_wear_policy", "mask") != "impute_diurnal":
        inactive.add(f"{prefix}extractor__interp_mode")
    if p.get("extractor__motion_filter_mode", "kalman") != "kalman":
        for name in ("kalman_process_noise", "kalman_measurement_noise"):
            inactive.add(f"{prefix}extractor__{name}")
    if not p.get("extractor__use_dead_reckoning", True):
        inactive.add(f"{prefix}extractor__dead_reckoning_detrend")
    if p.get("extractor__missing_feature_strategy", "keep_nan") != "knn":
        inactive.add(f"{prefix}extractor__impute_knn_neighbors")
    if p.get("extractor__missing_feature_strategy", "keep_nan") != "iterative":
        inactive.add(f"{prefix}extractor__impute_max_iter")
    return inactive


# ---------------------------------------------------------------------------
# Pure Handcrafted Sequence Baseline Estimator (ZERO ML Models)
# ---------------------------------------------------------------------------

class HandcraftedSequenceBaselineEstimator(BaseEstimator):
    """Purely handcrafted estimator combining ``SequenceExtractor`` open-source DSP features & domain rules.

    **Zero supervised machine learning models**: uses explicit domain priors across clinical
    assessment terms + open-source actigraphy calculations (`seq_acf_*`, `seq_sri`,
    `seq_circ_sjl_abs_h`, `seq_upright_night_frac`, `seq_cusum_enmo_max`, `seq_prsa_asym`,
    `seq_stft_flux_mean`, `seq_cwt_ultradian_circadian_ratio`, `seq_seg_*` waveform variance).
    """

    def __init__(
        self,
        extractor: Optional[SequenceExtractor] = None,
        augmentor: Optional[TSAugmentor] = None,
        scoring_rule: str = "autocorr_spectral_rule",
        w_internet_hours: float = 1.40,
        w_age: float = 0.70,
        w_age_internet: float = 0.90,
        w_sleep_disturbance: float = 0.65,
        w_inactivity: float = 0.35,
        w_cgas_deficit: float = 0.35,
        w_acf_irregularity: float = 0.45,
        w_sedentary_acf: float = 0.40,
        w_sri_irregularity: float = 0.40,
        w_nocturnal_ratio: float = 0.35,
        w_upright_night: float = 0.35,
        w_acrophase_delay: float = 0.30,
        w_period_drift: float = 0.25,
        w_stft_flux: float = 0.25,
        w_cwt_ultradian: float = 0.25,
        w_seglearn_kinematic: float = 0.25,
        w_cusum_prsa: float = 0.20,
        w_sleep_fragmentation: float = 0.25,
        coupling_alpha: float = 0.35,
        threshold_calibration: str = "optimize_qwk",
        primary_target: str = TARGET_COL,
        validation_group_column: Optional[str] = None,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.extractor = extractor
        self.augmentor = augmentor
        self.scoring_rule = scoring_rule
        self.w_internet_hours = w_internet_hours
        self.w_age = w_age
        self.w_age_internet = w_age_internet
        self.w_sleep_disturbance = w_sleep_disturbance
        self.w_inactivity = w_inactivity
        self.w_cgas_deficit = w_cgas_deficit
        self.w_acf_irregularity = w_acf_irregularity
        self.w_sedentary_acf = w_sedentary_acf
        self.w_sri_irregularity = w_sri_irregularity
        self.w_nocturnal_ratio = w_nocturnal_ratio
        self.w_upright_night = w_upright_night
        self.w_acrophase_delay = w_acrophase_delay
        self.w_period_drift = w_period_drift
        self.w_stft_flux = w_stft_flux
        self.w_cwt_ultradian = w_cwt_ultradian
        self.w_seglearn_kinematic = w_seglearn_kinematic
        self.w_cusum_prsa = w_cusum_prsa
        self.w_sleep_fragmentation = w_sleep_fragmentation
        self.coupling_alpha = coupling_alpha
        self.threshold_calibration = threshold_calibration
        self.primary_target = primary_target
        self.validation_group_column = validation_group_column
        self.verbose = verbose
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = super().get_params(deep=deep)
        if deep and self.extractor is None:
            params["extractor"] = None
            for k, v in SequenceExtractor().get_params(deep=True).items():
                params[f"extractor__{k}"] = v
        if deep and self.augmentor is None:
            params["augmentor"] = None
            for k, v in TSAugmentor().get_params(deep=True).items():
                params[f"augmentor__{k}"] = v
        return params

    def set_params(self, **params: Any):
        clean = {k: _structured(v) for k, v in params.items()}
        if any(name.startswith("extractor__") for name in clean) and clean.get("extractor", self.extractor) is None:
            clean["extractor"] = SequenceExtractor()
        if any(name.startswith("augmentor__") for name in clean) and clean.get("augmentor", self.augmentor) is None:
            clean["augmentor"] = TSAugmentor()
        return super().set_params(**clean)

    def _get_extractor(self) -> SequenceExtractor:
        return self.extractor if self.extractor is not None else SequenceExtractor()

    def _get_augmentor(self) -> TSAugmentor:
        return self.augmentor if self.augmentor is not None else TSAugmentor()

    def _build_domain_terms(self, F: pd.DataFrame) -> pd.DataFrame:
        r = pd.DataFrame(index=F.index, dtype=np.float64)

        def _col(name: str) -> np.ndarray:
            if name not in F.columns:
                return np.full(len(F), np.nan, dtype=np.float64)
            return pd.to_numeric(F[name], errors="coerce").to_numpy(dtype=np.float64)

        r["term_internet_hours"] = _col("PreInt_EduHx-computerinternet_hoursday")
        r["term_age"] = _col("Basic_Demos-Age")
        ih, ag = r["term_internet_hours"], r["term_age"]
        r["term_age_internet"] = np.where(np.isfinite(ih) & np.isfinite(ag), ag * (ih + 0.5), np.nan)
        r["term_sleep_disturbance"] = _col("SDS-SDS_Total_T")
        paq = _col("eng_PAQ_harmonized")
        if not np.isfinite(paq).any():
            paq_a, paq_c = _col("PAQ_A-PAQ_A_Total"), _col("PAQ_C-PAQ_C_Total")
            paq = np.where(np.isfinite(paq_a), paq_a, paq_c)
        r["term_inactivity"] = -paq
        r["term_cgas_deficit"] = 70.0 - _col("CGAS-CGAS_Score")

        acf24 = _col("seq_acf_log_24h")
        ra = _col("seq_m10_l5_ra")
        r["term_acf_irregularity"] = np.where(
            np.isfinite(acf24) & np.isfinite(ra),
            0.65 * (1.0 - acf24) + 0.35 * (1.0 - ra),
            np.where(np.isfinite(acf24), 1.0 - acf24, np.nan),
        )
        sed_acf = _col("seq_acf_sedentary_30m")
        sh_acf = _col("seq_acf_short_mean")
        r["term_sedentary_acf"] = np.where(np.isfinite(sed_acf), sed_acf, sh_acf)

        sri = _col("seq_sri")
        sjl = _col("seq_circ_sjl_abs_h")
        sri_term = np.where(np.isfinite(sri), (100.0 - sri) / 200.0, np.nan)
        r["term_sri_irregularity"] = np.where(
            np.isfinite(sri_term) & np.isfinite(sjl),
            0.70 * sri_term + 0.30 * (sjl / 4.0),
            np.where(np.isfinite(sri_term), sri_term, np.where(np.isfinite(sjl), sjl / 4.0, np.nan)),
        )

        ng = _col("seq_night_day_enmo_ratio")
        r["term_nocturnal_ratio"] = np.where(np.isfinite(ng), np.log1p(np.maximum(ng, 0.0)), np.nan)
        r["term_upright_night"] = _col("seq_upright_night_frac")
        acro = _col("seq_cosinor_acrophase_h")
        r["term_acrophase_delay"] = np.where(np.isfinite(acro), np.cos(2.0 * np.pi * (acro - 20.0) / 24.0), np.nan)
        r["term_period_drift"] = _col("seq_circadian_period_err_h")

        if self.scoring_rule in {"autocorr_spectral_rule", "multiplicative_hazard"}:
            r["term_stft_flux"] = _col("seq_stft_flux_mean")
            r["term_cwt_ultradian"] = _col("seq_cwt_ultradian_circadian_ratio")
            # seglearn inter-window instability across diurnal segments (inverse regularity)
            seg_wstd = _col("seq_seg_variation_log_enmo_wstd")
            seg_var = _col("seq_seg_std_dyn_amp_wstd")
            r["term_seglearn_kinematic"] = np.where(
                np.isfinite(seg_wstd) & np.isfinite(seg_var),
                0.5 * seg_wstd + 0.5 * seg_var,
                np.where(np.isfinite(seg_wstd), seg_wstd, seg_var),
            )
            cusum = _col("seq_cusum_enmo_max")
            prsa_a = _col("seq_prsa_asym")
            r["term_cusum_prsa"] = np.where(
                np.isfinite(cusum) & np.isfinite(prsa_a),
                0.65 * (cusum / 4.0) + 0.35 * np.abs(prsa_a),
                np.where(np.isfinite(cusum), cusum / 4.0, np.abs(prsa_a)),
            )
            frag = _col("seq_webster_fragmentation_idx")
            sa_eff = _col("seq_sadeh_night_efficiency")
            r["term_sleep_fragmentation"] = np.where(
                np.isfinite(frag) & np.isfinite(sa_eff),
                0.65 * frag + 0.35 * (1.0 - sa_eff),
                np.where(np.isfinite(frag), frag, 1.0 - sa_eff),
            )
        return r

    def _prior_weights(self) -> Dict[str, float]:
        return {
            "term_internet_hours": float(self.w_internet_hours),
            "term_age": float(self.w_age),
            "term_age_internet": float(self.w_age_internet),
            "term_sleep_disturbance": float(self.w_sleep_disturbance),
            "term_inactivity": float(self.w_inactivity),
            "term_cgas_deficit": float(self.w_cgas_deficit),
            "term_acf_irregularity": float(self.w_acf_irregularity),
            "term_sedentary_acf": float(self.w_sedentary_acf),
            "term_sri_irregularity": float(self.w_sri_irregularity),
            "term_nocturnal_ratio": float(self.w_nocturnal_ratio),
            "term_upright_night": float(self.w_upright_night),
            "term_acrophase_delay": float(self.w_acrophase_delay),
            "term_period_drift": float(self.w_period_drift),
            "term_stft_flux": float(self.w_stft_flux),
            "term_cwt_ultradian": float(self.w_cwt_ultradian),
            "term_seglearn_kinematic": float(self.w_seglearn_kinematic),
            "term_cusum_prsa": float(self.w_cusum_prsa),
            "term_sleep_fragmentation": float(self.w_sleep_fragmentation),
        }

    def _evaluate_handcrafted_scores(self, z_mat: np.ndarray) -> np.ndarray:
        avail = np.isfinite(z_mat).astype(np.float64)
        w = self.weight_vec_[None, :]
        eff_w = np.abs(w) * avail
        denom = np.sum(eff_w, axis=1)
        base = np.where(denom > 1e-9, np.sum(np.nan_to_num(z_mat, nan=0.0) * w, axis=1) / np.maximum(denom, 1e-9), 0.0)

        if self.scoring_rule == "multiplicative_hazard":
            idx_ih = self.term_names_.index("term_internet_hours")
            idx_acf = self.term_names_.index("term_acf_irregularity")
            idx_sri = self.term_names_.index("term_sri_irregularity")
            z_ih = np.nan_to_num(z_mat[:, idx_ih], nan=0.0)
            z_circ = 0.5 * (np.nan_to_num(z_mat[:, idx_acf], nan=0.0) + np.nan_to_num(z_mat[:, idx_sri], nan=0.0))
            return base + float(self.coupling_alpha) * np.maximum(0.0, z_ih) * np.maximum(0.0, z_circ)
        return base

    def fit(self, X: pd.DataFrame, y: Any):
        self.fit_failed_ = False
        y_arr = np.asarray(
            y[self.primary_target] if isinstance(y, pd.DataFrame) and self.primary_target in y.columns else y,
            dtype=np.int64,
        ).ravel()

        ext = self._get_extractor()
        ext.validate()
        self.extractor_ = ext
        aug = self._get_augmentor()
        self.augmentor_ = aug
        self.augmented_chunk_count_ = 0
        if aug.enabled():
            ch_res = ext.transform_chunks(strip_leaky_columns(X))
            if ch_res["chunks"].shape[0] > 0:
                rng = np.random.default_rng(self.random_state)
                aug_chunks = aug.augment_array(ch_res["chunks"], rng=rng)
                self.augmented_chunk_count_ = int(aug_chunks.shape[0])
        feat_df = ext.fit_transform(strip_leaky_columns(X), y_arr)
        self.feature_names_in_ = list(feat_df.columns)

        terms = self._build_domain_terms(feat_df)
        self.term_names_ = list(terms.columns)
        self.term_means_: Dict[str, float] = {}
        self.term_stds_: Dict[str, float] = {}

        z_mat = np.full((len(terms), len(self.term_names_)), np.nan, dtype=np.float64)
        for j, col in enumerate(self.term_names_):
            v = terms[col].to_numpy(dtype=np.float64)
            fin = np.isfinite(v)
            mu = float(np.mean(v[fin])) if fin.any() else 0.0
            sd = float(np.std(v[fin])) if fin.sum() > 1 else 1.0
            if sd <= 1e-9:
                sd = 1.0
            self.term_means_[col] = mu
            self.term_stds_[col] = sd
            z_mat[:, j] = np.where(fin, np.clip((v - mu) / sd, -4.0, 4.0), np.nan)

        pw = self._prior_weights()
        self.weight_vec_ = np.array([pw.get(c, 0.0) for c in self.term_names_], dtype=np.float64)
        scores = self._evaluate_handcrafted_scores(z_mat)
        self.train_score_mean_ = float(np.mean(scores))
        self.train_score_std_ = max(float(np.std(scores)), 1e-6)

        if self.threshold_calibration == "optimize_qwk":
            self.thresholds_ = optimize_thresholds_qwk(y_arr, scores)
        elif self.threshold_calibration == "quantile":
            p0, p1, p2 = float(np.mean(y_arr <= 0)), float(np.mean(y_arr <= 1)), float(np.mean(y_arr <= 2))
            self.thresholds_ = np.sort(np.quantile(scores, [p0, p1, p2]))
        elif self.threshold_calibration == "fixed_midpoint":
            self.thresholds_ = self.train_score_mean_ + (np.array([0.5, 1.5, 2.5]) - y_arr.mean()) * (
                self.train_score_std_ / max(y_arr.std(), 1e-3)
            )
        elif self.threshold_calibration in ("none", None):
            self.thresholds_ = np.array([0.0, 0.75, 1.5], dtype=np.float64)
        else:
            raise InvalidParams(f"Unsupported threshold_calibration: {self.threshold_calibration!r}")

        self.train_qwk_ = quadratic_weighted_kappa(y_arr, apply_thresholds(scores, self.thresholds_))
        self.classes_ = np.array([0, 1, 2, 3], dtype=np.int64)
        return self

    def predict_continuous(self, X: pd.DataFrame) -> np.ndarray:
        feat_df = self.extractor_.transform(strip_leaky_columns(X))
        terms = self._build_domain_terms(feat_df)
        z_mat = np.full((len(terms), len(self.term_names_)), np.nan, dtype=np.float64)
        for j, col in enumerate(self.term_names_):
            v = terms[col].to_numpy(dtype=np.float64)
            fin = np.isfinite(v)
            z_mat[:, j] = np.where(fin, np.clip((v - self.term_means_[col]) / self.term_stds_[col], -4.0, 4.0), np.nan)
        return self._evaluate_handcrafted_scores(z_mat)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return apply_thresholds(self.predict_continuous(X), self.thresholds_)

    def score(self, X: pd.DataFrame, y: Any) -> float:
        return competition_score(y, self.predict(X))

    def get_active_rules_df(self) -> pd.DataFrame:
        """Inspect handcrafted domain weights and fold standardization moments."""
        return pd.DataFrame({
            "handcrafted_term": self.term_names_,
            "prior_weight": self.weight_vec_,
            "train_mean": [self.term_means_[c] for c in self.term_names_],
            "train_std": [self.term_stds_[c] for c in self.term_names_],
            "scoring_rule": self.scoring_rule,
        })


# ---------------------------------------------------------------------------
# Search Space, Grid Deduplication & HierarchicalBayesSearchCV (Grid & Bayes only)
# ---------------------------------------------------------------------------

class ParamSpec:
    """Unified parameter descriptor rendered cleanly to Grid or Bayesian skopt dimensions."""

    def __init__(self, grid_values: Sequence[Any], bayes_dim: Optional[Any] = None):
        self.grid_values = list(grid_values)
        self.bayes_dim = bayes_dim

    def to_skopt(self):
        if self.bayes_dim is not None:
            return self.bayes_dim
        return SkCategorical([_encode_skopt_item(v) for v in self.grid_values])


def _encode_skopt_item(val: Any) -> Any:
    if isinstance(val, (tuple, list, dict)):
        return json.dumps(val, sort_keys=True)
    return val


def _decode_skopt_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: _structured(v) for k, v in params.items()}


def prepare_bayesian_space(space: Mapping[str, Any], skopt_available: bool = SKOPT_AVAILABLE) -> Dict[str, Any]:
    if not skopt_available or SkCategorical is None:
        return {}
    out: Dict[str, Any] = {}
    for k, v in space.items():
        if isinstance(v, ParamSpec):
            out[k] = v.to_skopt()
        elif isinstance(v, (list, tuple)):
            out[k] = SkCategorical([_encode_skopt_item(x) for x in v])
        else:
            out[k] = v
    return out


def inactive_parameters_for_candidate(params: Mapping[str, Any]) -> set:
    p = {
        (k[len("estimator__"):] if k.startswith("estimator__") else k): _structured(v)
        for k, v in params.items()
    }
    prefix = "estimator__" if any(k.startswith("estimator__") for k in params) else ""
    inactive = inactive_sequence_parameters(params)

    rule = p.get("scoring_rule", "autocorr_spectral_rule")
    if rule == "weighted_z_index":
        for name in (
            "w_stft_flux", "w_cwt_ultradian", "w_seglearn_kinematic",
            "w_cusum_prsa", "w_sleep_fragmentation", "coupling_alpha",
        ):
            inactive.add(f"{prefix}{name}")
    elif rule == "autocorr_spectral_rule":
        inactive.add(f"{prefix}coupling_alpha")
    if float(p.get("augmentor__prob", 0.0)) <= 0.0:
        for name in (
            "per_aug_prob", "augmentations", "n_copies", "jitter_sigma", "noise_std",
            "scaling_sigma", "warp_sigma", "warp_num_knots", "time_warp_max_speed_ratio",
            "time_warp_n_speed_change", "time_shift_frac", "crop_frac_range",
            "temporal_num_masks", "temporal_mask_frac", "temporal_mask_fill", "channel_drop_prob",
        ):
            inactive.add(f"{prefix}augmentor__{name}")
    return inactive


def candidate_error(params: Mapping[str, Any], base_params: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    merged = dict(base_params or {})
    merged.update(_decode_skopt_params(params))
    ext_kwargs = {
        k.split("extractor__", 1)[1]: v
        for k, v in merged.items()
        if "extractor__" in k and "noop__" not in k
    }
    try:
        validate_sequence_extractor_params(ext_kwargs)
    except InvalidParams as exc:
        return str(exc)
    return None


def _canonical_candidate_key(params: Mapping[str, Any], base_params: Optional[Mapping[str, Any]] = None) -> str:
    merged = dict(base_params or {})
    merged.update(_decode_skopt_params(params))
    inactive = inactive_parameters_for_candidate(merged)
    eff = {k: repr(v) for k, v in sorted(params.items()) if k not in inactive}
    return hashlib.md5(json.dumps(eff, sort_keys=True).encode()).hexdigest()


def deduplicate_param_grid(param_grid: Mapping[str, Sequence[Any]], *, base_params: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    grid = list(ParameterGrid(dict(param_grid)))
    seen: set = set()
    kept: List[Dict[str, Any]] = []
    for cand in grid:
        if candidate_error(cand, base_params=base_params) is not None:
            continue
        sig = _canonical_candidate_key(cand, base_params=base_params)
        if sig not in seen:
            seen.add(sig)
            kept.append(cand)
    return [{k: [v] for k, v in cand.items()} for cand in kept]


def refit_valid_candidate(cv_results: Any, X: Optional[pd.DataFrame] = None, y: Optional[Any] = None) -> int:
    res = getattr(cv_results, "cv_results_", cv_results)
    means = np.asarray(res.get("mean_test_qwk", res.get("mean_test_score", [])), dtype=float)
    valid = np.where(np.isfinite(means) & (means > -0.5))[0]
    if len(valid) == 0:
        return int(np.nanargmax(means)) if len(means) else 0
    return int(valid[int(np.argmax(means[valid]))])


def _evaluate_candidate_cv(
    estimator: Any,
    params: Mapping[str, Any],
    X: pd.DataFrame,
    y: Any,
    groups: Optional[Any],
    cv: Any,
    scoring: Any,
    *,
    error_score: float = -1.0,
) -> Tuple[List[float], List[float], float]:
    t0 = time.perf_counter()
    qwk_scores: List[float] = []
    acc_pct_scores: List[float] = []
    use_default_qwk = scoring is None or scoring is candidate_score
    for tr_idx, va_idx in cv.split(X, y, groups):
        cand_est = clone(estimator).set_params(**params)
        y_tr = y.iloc[tr_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)[tr_idx]
        y_va = y.iloc[va_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)[va_idx]
        try:
            cand_est.fit(X.iloc[tr_idx], y_tr)
            preds = np.asarray(cand_est.predict(X.iloc[va_idx]), dtype=np.int64).ravel()
            yt_va = (
                np.asarray(y_va[TARGET_COL], dtype=np.int64).ravel()
                if isinstance(y_va, pd.DataFrame) and TARGET_COL in y_va.columns
                else np.asarray(y_va, dtype=np.int64).ravel()
            )
            acc_pct = float(100.0 * np.mean(yt_va == preds)) if len(yt_va) == len(preds) and len(yt_va) > 0 else -1.0
            if use_default_qwk:
                qwk = float(competition_score(yt_va, preds))
            else:
                qwk = float(scoring(cand_est, X.iloc[va_idx], y_va))
            qwk_scores.append(qwk if np.isfinite(qwk) else error_score)
            acc_pct_scores.append(acc_pct if np.isfinite(acc_pct) else -1.0)
        except Exception:
            qwk_scores.append(error_score)
            acc_pct_scores.append(-1.0)
    return qwk_scores, acc_pct_scores, float(time.perf_counter() - t0)


class HierarchicalBayesSearchCV(BaseEstimator):
    """Bayesian optimization over DSP, ``seglearn``, ``SequenceExtractor`` & handcrafted rule parameters."""

    def __init__(
        self,
        estimator: Any,
        search_spaces: Mapping[str, Any],
        n_iter: int = 8,
        cv: Any = None,
        scoring: Any = None,
        refit: Any = True,
        random_state: int = 42,
        verbose: int = 1,
        error_score: Any = -1.0,
        n_jobs: int = 1,
        x0: Optional[Mapping[str, Any]] = None,
    ):
        self.estimator = estimator
        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.random_state = random_state
        self.verbose = verbose
        self.error_score = error_score
        self.n_jobs = n_jobs
        self.x0 = x0

    @staticmethod
    def _default_starting_point(keys: Sequence[str], dimensions: Sequence[Any], base_params: Mapping[str, Any], user_x0: Optional[Mapping[str, Any]]) -> List[Any]:
        point: List[Any] = []
        for k, dim in zip(keys, dimensions):
            if user_x0 is not None and k in user_x0:
                val = _encode_skopt_item(user_x0[k])
                cats = getattr(dim, "categories", None)
                if cats is None or val in cats:
                    point.append(val)
                    continue
            if hasattr(dim, "categories"):
                cats = list(dim.categories)
                bv = _encode_skopt_item(base_params.get(k)) if k in base_params else None
                point.append(bv if bv in cats else cats[0])
            else:
                low, high = getattr(dim, "low", 0.0), getattr(dim, "high", 1.0)
                is_int = isinstance(low, (int, np.integer)) and isinstance(high, (int, np.integer))
                bv = base_params.get(k)
                if isinstance(bv, (int, float, np.number)) and low <= float(bv) <= high:
                    point.append(int(round(float(bv))) if is_int else float(bv))
                else:
                    point.append(int(low) if is_int else float(low))
        return point

    def fit(self, X: pd.DataFrame, y: Any, groups: Optional[Any] = None):
        if not SKOPT_AVAILABLE or Optimizer is None:
            raise ImportError("scikit-optimize is required for HierarchicalBayesSearchCV.")
        cv = self.cv or StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(self.random_state))
        space_dict = prepare_bayesian_space(self.search_spaces, skopt_available=True)
        keys = list(space_dict.keys())
        dimensions = [space_dict[k] for k in keys]
        opt = Optimizer(dimensions, random_state=int(self.random_state))

        base_params = self.estimator.get_params(deep=True)
        cache: Dict[str, Tuple[List[float], List[float], float]] = {}
        all_params: List[Dict[str, Any]] = []
        all_fold_qwk: List[List[float]] = []
        all_fold_acc_pct: List[List[float]] = []
        all_times: List[float] = []

        init_pt = self._default_starting_point(keys, dimensions, base_params, self.x0)
        scorer = self.scoring or candidate_score
        best_so_far = -np.inf
        total = max(1, int(self.n_iter))

        for it in range(total):
            x_pt = init_pt if it == 0 else opt.ask()
            decoded = _decode_skopt_params(dict(zip(keys, x_pt)))
            sig = _canonical_candidate_key(decoded, base_params=base_params)
            if sig in cache:
                fold_qwks, fold_accs, elapsed = cache[sig]
                cached_tag = " (cached)"
            else:
                fold_qwks, fold_accs, elapsed = _evaluate_candidate_cv(
                    self.estimator, decoded, X, y, groups, cv, scorer,
                    error_score=-1.0 if self.error_score == "raise" else float(self.error_score),
                )
                cache[sig] = (fold_qwks, fold_accs, elapsed)
                cached_tag = ""
            ok_idx = [i for i, (a, q) in enumerate(zip(fold_accs, fold_qwks)) if np.isfinite(a) and a >= 0.0 and np.isfinite(q)]
            mean_acc = float(np.mean([fold_accs[i] for i in ok_idx])) if ok_idx else np.nan
            mean_qwk = float(np.mean([fold_qwks[i] for i in ok_idx])) if ok_idx else float(np.mean(fold_qwks))
            best_so_far = max(best_so_far, mean_qwk)
            opt.tell(x_pt, -mean_qwk)
            all_params.append(decoded)
            all_fold_qwk.append(fold_qwks)
            all_fold_acc_pct.append(fold_accs)
            all_times.append(elapsed)
            if self.verbose and int(self.verbose) > 0:
                sr = decoded.get("estimator__scoring_rule", decoded.get("scoring_rule", "?"))
                bm = decoded.get("estimator__extractor__bin_minutes", decoded.get("extractor__bin_minutes", "?"))
                fold_str = " | ".join(f"F{i + 1}: {fold_accs[i]:.2f}% -> QWK={fold_qwks[i]:.5f}" for i in ok_idx)
                acc_lbl = f"{mean_acc:.2f}%" if np.isfinite(mean_acc) else "nan%"
                print(
                    f"[Bayes {it + 1}/{total}] rule={sr} bin={bm}m | "
                    f"Normal={acc_lbl} -> Comp QWK={mean_qwk:.5f} | best={best_so_far:.5f} ({elapsed:.2f}s){cached_tag}",
                    flush=True,
                )
                if ok_idx:
                    print(f"  Successful folds ({len(ok_idx)}/{len(fold_qwks)}): {fold_str}", flush=True)

        means = np.array([float(np.mean(s)) for s in all_fold_qwk], dtype=float)
        stds = np.array([float(np.std(s)) for s in all_fold_qwk], dtype=float)
        acc_means = np.array(
            [float(np.mean([a for a in s if np.isfinite(a) and a >= 0.0])) if any(np.isfinite(a) and a >= 0.0 for a in s) else np.nan for s in all_fold_acc_pct],
            dtype=float,
        )
        acc_stds = np.array(
            [float(np.std([a for a in s if np.isfinite(a) and a >= 0.0])) if any(np.isfinite(a) and a >= 0.0 for a in s) else np.nan for s in all_fold_acc_pct],
            dtype=float,
        )
        ranks = (len(means) - np.argsort(np.argsort(means))).astype(int)
        n_splits = len(all_fold_qwk[0]) if all_fold_qwk else 1

        self.cv_results_ = {
            "params": all_params,
            "mean_test_score": means,
            "std_test_score": stds,
            "rank_test_score": ranks,
            "mean_test_qwk": means.copy(),
            "std_test_qwk": stds.copy(),
            "rank_test_qwk": ranks.copy(),
            "mean_test_accuracy_pct": acc_means,
            "std_test_accuracy_pct": acc_stds,
            "mean_fit_time": np.array(all_times, dtype=float),
        }
        for i in range(n_splits):
            self.cv_results_[f"split{i}_test_score"] = np.array([s[i] for s in all_fold_qwk], dtype=float)
            self.cv_results_[f"split{i}_test_qwk"] = np.array([s[i] for s in all_fold_qwk], dtype=float)
            self.cv_results_[f"split{i}_test_accuracy_pct"] = np.array([s[i] for s in all_fold_acc_pct], dtype=float)

        self.best_index_ = int(refit_valid_candidate(self.cv_results_)) if callable(self.refit) else int(np.argmax(means))
        self.best_params_ = dict(all_params[self.best_index_])
        self.best_score_ = float(means[self.best_index_])
        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.best_estimator_.predict(X)


def extract_successful_fold_scores(search_or_results: Any, *, cand_index: Optional[int] = None) -> pd.DataFrame:
    """Return normal percentage score and competition QWK for each successful CV fold."""
    cv_res = getattr(search_or_results, "cv_results_", search_or_results)
    if not cv_res:
        return pd.DataFrame(columns=["fold", "normal_score_pct", "competition_qwk"])
    idx = int(getattr(search_or_results, "best_index_", 0) if cand_index is None else cand_index)

    qwk_keys = sorted(
        [k for k in cv_res if k.startswith("split") and k.endswith("_test_qwk")],
        key=lambda s: int(s[len("split"):].split("_", 1)[0]),
    )
    if not qwk_keys:
        qwk_keys = sorted(
            [k for k in cv_res if k.startswith("split") and k.endswith("_test_score")],
            key=lambda s: int(s[len("split"):].split("_", 1)[0]),
        )

    rows: List[Dict[str, Any]] = []
    for qk in qwk_keys:
        fold_num = int(qk[len("split"):].split("_", 1)[0])
        ak = f"split{fold_num}_test_accuracy_pct"
        q_arr = np.asarray(cv_res[qk], dtype=float)
        if idx >= len(q_arr):
            continue
        q_val = float(q_arr[idx])
        if ak in cv_res:
            a_val = float(np.asarray(cv_res[ak], dtype=float)[idx])
            if not (np.isfinite(a_val) and a_val >= 0.0 and np.isfinite(q_val)):
                continue
        else:
            if not np.isfinite(q_val):
                continue
            a_val = np.nan
        rows.append({
            "fold": fold_num + 1,
            "normal_score_pct": round(a_val, 2) if np.isfinite(a_val) else np.nan,
            "competition_qwk": round(q_val, 5),
        })
    return pd.DataFrame(rows, columns=["fold", "normal_score_pct", "competition_qwk"])


def build_quick_results_df(search_or_results: Any) -> pd.DataFrame:
    cv_res = getattr(search_or_results, "cv_results_", search_or_results)
    if not cv_res:
        return pd.DataFrame()
    df = pd.DataFrame(cv_res)
    if "params" in df.columns:
        p_df = pd.json_normalize(df["params"]).rename(columns=lambda c: c.replace("estimator__", ""))
        df = pd.concat([df.drop(columns=["params"]), p_df], axis=1)
    if "rank_test_qwk" in df.columns:
        df = df.sort_values("rank_test_qwk", kind="stable").reset_index(drop=True)
    elif "rank_test_score" in df.columns:
        df = df.sort_values("rank_test_score", kind="stable").reset_index(drop=True)
    elif "mean_test_qwk" in df.columns:
        df = df.sort_values("mean_test_qwk", ascending=False, kind="stable").reset_index(drop=True)
    elif "mean_test_score" in df.columns:
        df = df.sort_values("mean_test_score", ascending=False, kind="stable").reset_index(drop=True)
    return df


def split_stratified_protocol(
    experiment_df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    train_size: float = 0.8,
    n_splits: int = 5,
    random_state: int = 42,
):
    sii_num = experiment_df[target_col].astype(int)
    sii_str = sii_num.astype(str)
    req_min = max(2, int(n_splits))
    if "has_actigraphy" in experiment_df.columns:
        joint = sii_str + "_" + experiment_df["has_actigraphy"].astype(int).astype(str)
        strata = joint if int(joint.value_counts().min()) >= req_min else sii_str
    else:
        strata = sii_str

    vc = strata.value_counts()
    if int(vc.min()) < 2:
        collapsed = sii_num.copy()
        for val in sorted(collapsed.unique(), reverse=True):
            if int((collapsed == val).sum()) < 2 and val > 0:
                collapsed = collapsed.replace(val, val - 1)
        strata = collapsed.astype(str)

    if int(strata.value_counts().min()) >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=float(train_size), random_state=int(random_state))
        tr_idx, te_idx = next(splitter.split(experiment_df, strata))
    else:
        from sklearn.model_selection import ShuffleSplit
        splitter = ShuffleSplit(n_splits=1, train_size=float(train_size), random_state=int(random_state))
        tr_idx, te_idx = next(splitter.split(experiment_df))

    X_train = experiment_df.iloc[tr_idx].reset_index(drop=True)
    X_test = experiment_df.iloc[te_idx].reset_index(drop=True)

    tr_strata = strata.iloc[tr_idx].reset_index(drop=True)
    y_tr_vals = X_train[target_col].astype(int)
    min_tr_cls = int(y_tr_vals.value_counts().min()) if len(y_tr_vals) else 0
    if int(n_splits) <= 1:
        if min_tr_cls >= 2:
            cv_obj = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(random_state))
        else:
            from sklearn.model_selection import ShuffleSplit
            cv_obj = ShuffleSplit(n_splits=1, test_size=0.2, random_state=int(random_state))
    elif min_tr_cls >= int(n_splits):
        cv_obj = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    elif min_tr_cls >= 2:
        cv_obj = StratifiedKFold(n_splits=min_tr_cls, shuffle=True, random_state=int(random_state))
    else:
        from sklearn.model_selection import KFold
        cv_obj = KFold(n_splits=min(int(n_splits), max(2, len(X_train) // 3)), shuffle=True, random_state=int(random_state))
    return X_train, X_test, tr_strata, cv_obj


def run_validation_searches(
    search: Any,
    experiment_df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    validation_param_space: Optional[Mapping[str, Any]] = None,
    train_size: float = 0.8,
    n_splits: int = 5,
    random_state: int = 42,
    results_dir: Optional[Any] = "results",
    experiment_name: str = "baseline_cmi_piu",
    search_mode: str = "bayes",
    timestamp: Optional[str] = None,
    save_results: bool = True,
    keep_history: bool = False,
) -> Dict[str, Any]:
    """Run stratified CV search and holdout evaluation, optionally saving CV and summary files."""
    persist = bool(save_results) and results_dir is not None
    out_dir = Path(results_dir) if persist else None
    prefix = f"{timestamp or time.strftime('%Y%m%d_%H%M%S')}_" if (keep_history or timestamp) else ""
    modes = list((validation_param_space or {}).get("validation_mode", ["stratified_subject"]))

    protocol_results: List[Dict[str, Any]] = []
    last_run: Dict[str, Any] = {"experiment_succeeded": False}

    for v_mode in modes:
        X_train_full, X_test_full, tr_strata, cv_obj = split_stratified_protocol(
            experiment_df, target_col=target_col, train_size=train_size, n_splits=n_splits, random_state=random_state
        )
        y_train = X_train_full[target_col].astype(int).reset_index(drop=True)
        y_test = X_test_full[target_col].astype(int).reset_index(drop=True)
        X_train = strip_leaky_columns(X_train_full)
        X_test = strip_leaky_columns(X_test_full)

        print(
            f"\nValidation protocol: mode={v_mode} | "
            f"Train: {len(X_train)} subjects (act={int(X_train['has_actigraphy'].sum())}) | "
            f"Holdout: {len(X_test)} subjects (act={int(X_test['has_actigraphy'].sum())})",
            flush=True,
        )

        run_search = clone(search)
        run_search.cv = cv_obj
        run_search.fit(X_train, y_train, groups=tr_strata if v_mode == "stratified_actigraphy" else None)

        best_model = getattr(run_search, "best_estimator_", None)
        best_idx = int(getattr(run_search, "best_index_", 0))
        cv_res = getattr(run_search, "cv_results_", {})
        cv_means = cv_res.get("mean_test_qwk", cv_res.get("mean_test_score", []))
        raw_best_score = getattr(run_search, "best_score_", np.nan)
        if not np.isfinite(raw_best_score) and len(cv_means) > best_idx:
            raw_best_score = cv_means[best_idx]
        best_score = float(raw_best_score)
        best_params = dict(getattr(run_search, "best_params_", {}))

        fold_df = extract_successful_fold_scores(run_search, cand_index=best_idx)
        acc_means = cv_res.get("mean_test_accuracy_pct", [])
        raw_best_acc_pct = acc_means[best_idx] if len(acc_means) > best_idx else (
            float(fold_df["normal_score_pct"].mean()) if len(fold_df) else np.nan
        )
        best_acc_pct = float(raw_best_acc_pct) if raw_best_acc_pct is not None and np.isfinite(raw_best_acc_pct) else np.nan

        if len(fold_df) > 0:
            print(f"\nSuccessful Folds Summary ({len(fold_df)} folds) — Normal Percentage Score -> Competition Score (QWK):")
            for _, frow in fold_df.iterrows():
                pct_s = f"{frow['normal_score_pct']:.2f}%" if pd.notna(frow["normal_score_pct"]) else "N/A"
                print(f"  Fold {int(frow['fold'])}: Normal Score = {pct_s} | Competition Score (QWK) = {frow['competition_qwk']:.5f}")
            mean_pct_s = f"{fold_df['normal_score_pct'].mean():.2f}%" if fold_df["normal_score_pct"].notna().any() else "N/A"
            print(f"  Mean across {len(fold_df)} successful folds: Normal Score = {mean_pct_s} | Competition Score (QWK) = {fold_df['competition_qwk'].mean():.5f}")

        eval_res: Dict[str, Any] = {}
        if best_model is not None and len(X_test) > 0:
            y_pred = best_model.predict(X_test)
            eval_res = evaluate_holdout(y_test, y_pred, target_col=target_col, verbose=True)

        row_summary = {
            "validation_mode": v_mode,
            "search_mode": search_mode,
            "best_cv_normal_score_pct": best_acc_pct,
            "best_cv_qwk": best_score,
            "successful_folds": int(len(fold_df)),
            "holdout_normal_score_pct": eval_res.get("accuracy_pct", np.nan),
            "holdout_qwk": eval_res.get("competition_score", np.nan),
            "holdout_acc": eval_res.get("accuracy", np.nan),
            "holdout_mae": eval_res.get("mae", np.nan),
            "n_train": int(len(X_train)),
            "n_holdout": int(len(X_test)),
        }
        protocol_results.append(row_summary)

        exp_path_str: Optional[str] = None
        summary_path_str: Optional[str] = None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            cv_df = build_quick_results_df(run_search)
            exp_csv = out_dir / f"{prefix}{experiment_name}_{v_mode}_{search_mode}_cv.csv"
            cv_df.to_csv(exp_csv, index=False)

            summary_json = out_dir / f"{prefix}{experiment_name}_summary.json"
            summary_json.write_text(
                json.dumps(
                    {
                        "protocol_results": protocol_results,
                        "successful_fold_scores": fold_df.to_dict(orient="records"),
                        "best_params": {k: repr(v) for k, v in best_params.items()},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            exp_path_str = str(exp_csv)
            summary_path_str = str(summary_json)

        last_run = {
            "experiment_succeeded": best_model is not None,
            "validation_mode": v_mode,
            "search": run_search,
            "best_estimator": best_model,
            "best_params": best_params,
            "best_cv_accuracy_pct": best_acc_pct,
            "best_cv_qwk": best_score,
            "fold_scores_df": fold_df,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "holdout_results": eval_res,
            "experiments_path": exp_path_str,
            "summary_path": summary_path_str,
        }

    return {
        "last_run": last_run,
        "protocol_results": protocol_results,
        "experiments_path": last_run.get("experiments_path"),
        "summary_path": last_run.get("summary_path"),
    }
