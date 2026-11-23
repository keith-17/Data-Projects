"""lgbm_utils.py

LightGBM Estimator, LightGBM-aware Hyperparameter Searchers (Bayesian / Evolutionary / Grid),
Competition Submission Validator, and Trained-Model Permutation Importance for the Kaggle
Child Mind Institute — Problematic Internet Use (CMI PIU) competition.

Designed to drop seamlessly into the ``baseline.ipynb`` / ``base_utils.py`` pipeline:
* Reuses ``SequenceExtractor``, ``TSAugmentor``, ``NoopActigraphyOptions``, ``run_validation_searches``,
  ``evaluate_holdout``, ``export_submission``, and exact Quadratic Weighted Kappa (QWK) scoring from
  ``base_utils.py`` and ``baseline_utils.py``.
* Supports both ``LGBMRegressor`` (MSE / Huber / Fair / MAE / Poisson / Tweedie trees directly
  aligned with ordinal QWK loss) and ``LGBMClassifier`` (expected-value probability projection
  or argmax) across ``gbdt``, ``dart``, and ``goss`` boosting types.
* Leverages LightGBM's native missing-value split routing (or explicit imputation strategies),
  leak-free internal Out-Of-Fold (OOF) early stopping + ordinal QWK threshold calibration, and
  gain/split feature importance attribution.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold, StratifiedShuffleSplit

try:
    from baseline_utils import (
        apply_thresholds,
        optimize_thresholds_qwk,
        quadratic_weighted_kappa,
    )
    from base_utils import (
        ID_COL,
        SKOPT_AVAILABLE,
        TARGET_COL,
        ECat,
        EFloat,
        EInt,
        InvalidParams,
        Optimizer,
        SequenceExtractor,
        TSAugmentor,
        _accumulate_candidate_scores,
        _decode_skopt_params,
        _encode_skopt_item,
        _fmt_pct,
        _ga_gen_summary_line,
        _log_fold_error,
        _populate_param_columns,
        _propagate_verbose,
        _structured,
        candidate_score,
        competition_score,
        inactive_sequence_parameters,
        prepare_bayesian_space,
        prepare_evolutionary_space,
        refit_valid_candidate,
        strip_leaky_columns,
        validate_sequence_extractor_params,
    )
except ImportError:
    from Src.baseline_utils import (
        apply_thresholds,
        optimize_thresholds_qwk,
        quadratic_weighted_kappa,
    )
    from Src.base_utils import (
        ID_COL,
        SKOPT_AVAILABLE,
        TARGET_COL,
        ECat,
        EFloat,
        EInt,
        InvalidParams,
        Optimizer,
        SequenceExtractor,
        TSAugmentor,
        _accumulate_candidate_scores,
        _decode_skopt_params,
        _encode_skopt_item,
        _fmt_pct,
        _ga_gen_summary_line,
        _log_fold_error,
        _populate_param_columns,
        _propagate_verbose,
        _structured,
        candidate_score,
        competition_score,
        inactive_sequence_parameters,
        prepare_bayesian_space,
        prepare_evolutionary_space,
        refit_valid_candidate,
        strip_leaky_columns,
        validate_sequence_extractor_params,
    )


DEFAULT_LGBM_ESTIMATOR_PARAMS: Dict[str, Any] = {
    "task_mode": "regressor",
    "boosting_type": "gbdt",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "min_child_weight": 1e-3,
    "min_split_gain": 0.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "max_bin": 255,
    "extra_trees": False,
    "path_smooth": 0.0,
    "drop_rate": 0.1,
    "reg_objective": "regression",
    "clf_prediction_mode": "expected_value",
    "early_stopping_rounds": 40,
    "class_weight": None,
    "sample_weight_mode": "none",
    "nan_handling": "native",
    "drop_all_nan_cols": True,
    "threshold_calibration": "optimize_qwk",
    "threshold_calibration_source": "oof",
}

_SUPPORTED_REG_OBJECTIVES = {"regression", "huber", "fair", "regression_l1", "poisson", "tweedie"}
_SUPPORTED_BOOSTING_TYPES = {"gbdt", "dart", "goss"}


# ---------------------------------------------------------------------------
# Coercion, Validation, Competition Submission Check & Feature-Family Helpers
# ---------------------------------------------------------------------------

def _is_none_like(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in {"", "none", "null", "nan"}:
        return True
    if isinstance(val, (float, np.floating)) and not np.isfinite(val):
        return True
    return False


def _coerce_bool(val: Any, default: bool = False) -> bool:
    v = _structured(val)
    if v is None:
        return default
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        low = v.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no", "none", ""}:
            return False
    return bool(v)


def _coerce_opt_int(val: Any, *, min_val: Optional[int] = None) -> Optional[int]:
    v = _structured(val)
    if _is_none_like(v):
        return None
    iv = int(round(float(v)))
    if min_val is not None and iv < min_val:
        return None
    return iv


def _coerce_opt_float(val: Any) -> Optional[float]:
    v = _structured(val)
    if _is_none_like(v):
        return None
    fv = float(v)
    return fv if np.isfinite(fv) else None


def _coerce_max_depth(val: Any) -> int:
    v = _structured(val)
    if _is_none_like(v):
        return -1
    iv = int(round(float(v)))
    return -1 if iv <= 0 else iv


def _coerce_class_weight(val: Any) -> Optional[str]:
    v = _structured(val)
    if _is_none_like(v):
        return None
    s = str(v).strip().lower()
    if s != "balanced":
        raise InvalidParams(f"Unsupported class_weight for LightGBM: {val!r} (expected None or 'balanced')")
    return s


def _feature_family_label(col_name: str) -> str:
    """Map an extracted feature column to its coherent semantic family (aligned with ``base_utils.py``)."""
    c = str(col_name)
    if c.startswith("miss_"):
        return "missing_indicators"
    if c.startswith("noop_") or c in {"seq_sedentary_frac", "seq_sedentary_max_bout_h"}:
        return "noop_all"
    if c.startswith(("seq_acf_", "seq_pacf_", "seq_m10_", "seq_sri", "seq_circ_", "seq_cosinor_", "seq_circadian_", "seq_stl_", "seq_lomb_")):
        return "autocorr_circadian"
    if c.startswith(("seq_lag_", "seq_lead_", "seq_presleep_", "seq_day_lag1_", "seq_roll_")):
        return "lag_lead_dynamics"
    if c.startswith(("seq_sadeh_", "seq_ck_", "seq_webster_", "seq_nat_", "seq_fod_", "seq_sib_", "seq_night_", "seq_upright_")):
        return "sleep_architecture"
    if c.startswith(("seq_stft_", "seq_welch_", "seq_spec_", "seq_cwt_", "seq_dwt_", "seq_rp_")):
        return "stft_wavelet_spectral"
    if c.startswith(("seq_seg_", "seq_br_", "seq_cusum_", "seq_prsa_")):
        return "seglearn_kinematic"
    if c.startswith(("seq_tsfel_", "seq_tsfresh_", "seq_tsflex_", "seq_c22_")):
        return "static_open_source"
    if c.startswith("seq_"):
        return "actigraphy_compliance"
    return "clinical_tabular"


def validate_lgbm_params(params: Mapping[str, Any]) -> None:
    """Validate LightGBM hyperparameters prior to CV fitting."""
    task_mode = str(_structured(params.get("task_mode", "regressor"))).strip().lower()
    if task_mode not in {"regressor", "classifier"}:
        raise InvalidParams(f"task_mode must be 'regressor' or 'classifier', got {task_mode!r}")

    boosting_type = str(_structured(params.get("boosting_type", "gbdt"))).strip().lower()
    if boosting_type not in _SUPPORTED_BOOSTING_TYPES:
        raise InvalidParams(f"Unsupported boosting_type: {boosting_type!r}")

    n_est = _coerce_opt_int(params.get("n_estimators", 300), min_val=1)
    if n_est is None or n_est < 1:
        raise InvalidParams(f"n_estimators must be >= 1, got {params.get('n_estimators')!r}")

    lr = float(_structured(params.get("learning_rate", 0.03)))
    if not np.isfinite(lr) or not (0.0 < lr <= 1.0):
        raise InvalidParams(f"learning_rate must be in (0, 1], got {lr!r}")

    num_leaves = _coerce_opt_int(params.get("num_leaves", 31), min_val=2)
    if num_leaves is None or num_leaves < 2:
        raise InvalidParams(f"num_leaves must be >= 2, got {params.get('num_leaves')!r}")

    _coerce_max_depth(params.get("max_depth", 6))

    min_child_samples = _coerce_opt_int(params.get("min_child_samples", 20), min_val=1)
    if min_child_samples is None or min_child_samples < 1:
        raise InvalidParams(f"min_child_samples must be >= 1, got {params.get('min_child_samples')!r}")

    min_child_weight = float(_structured(params.get("min_child_weight", 1e-3)))
    if not np.isfinite(min_child_weight) or min_child_weight < 0.0:
        raise InvalidParams(f"min_child_weight must be >= 0.0, got {min_child_weight!r}")

    min_split_gain = float(_structured(params.get("min_split_gain", 0.0)))
    if not np.isfinite(min_split_gain) or min_split_gain < 0.0:
        raise InvalidParams(f"min_split_gain must be >= 0.0, got {min_split_gain!r}")

    subsample = float(_structured(params.get("subsample", 0.8)))
    if not np.isfinite(subsample) or not (0.0 < subsample <= 1.0):
        raise InvalidParams(f"subsample must be in (0, 1], got {subsample!r}")

    subsample_freq = int(round(float(_structured(params.get("subsample_freq", 1)))))
    if subsample_freq < 0:
        raise InvalidParams(f"subsample_freq must be >= 0, got {subsample_freq!r}")

    colsample = float(_structured(params.get("colsample_bytree", 0.8)))
    if not np.isfinite(colsample) or not (0.0 < colsample <= 1.0):
        raise InvalidParams(f"colsample_bytree must be in (0, 1], got {colsample!r}")

    reg_alpha = float(_structured(params.get("reg_alpha", 0.1)))
    reg_lambda = float(_structured(params.get("reg_lambda", 1.0)))
    if not np.isfinite(reg_alpha) or reg_alpha < 0.0:
        raise InvalidParams(f"reg_alpha must be >= 0.0, got {reg_alpha!r}")
    if not np.isfinite(reg_lambda) or reg_lambda < 0.0:
        raise InvalidParams(f"reg_lambda must be >= 0.0, got {reg_lambda!r}")

    max_bin = _coerce_opt_int(params.get("max_bin", 255), min_val=15)
    if max_bin is None or max_bin < 15:
        raise InvalidParams(f"max_bin must be >= 15, got {params.get('max_bin')!r}")

    path_smooth = float(_structured(params.get("path_smooth", 0.0)))
    if not np.isfinite(path_smooth) or path_smooth < 0.0:
        raise InvalidParams(f"path_smooth must be >= 0.0, got {path_smooth!r}")

    drop_rate = float(_structured(params.get("drop_rate", 0.1)))
    if not np.isfinite(drop_rate) or not (0.0 <= drop_rate <= 1.0):
        raise InvalidParams(f"drop_rate must be in [0, 1], got {drop_rate!r}")

    reg_obj = str(_structured(params.get("reg_objective", "regression"))).strip().lower()
    if reg_obj not in _SUPPORTED_REG_OBJECTIVES:
        raise InvalidParams(f"Unsupported reg_objective: {reg_obj!r}")

    clf_pred_mode = str(_structured(params.get("clf_prediction_mode", "expected_value"))).strip().lower()
    if clf_pred_mode not in {"expected_value", "argmax"}:
        raise InvalidParams(f"Unsupported clf_prediction_mode: {clf_pred_mode!r}")

    es_raw = _structured(params.get("early_stopping_rounds", 40))
    if not _is_none_like(es_raw) and int(round(float(es_raw))) < 0:
        raise InvalidParams(f"early_stopping_rounds must be None or >= 0, got {es_raw!r}")

    nan_h = str(_structured(params.get("nan_handling", "native"))).strip().lower()
    if nan_h not in {"native", "median", "sentinel", "zero"}:
        raise InvalidParams(f"Unsupported nan_handling: {nan_h!r}")

    sw_mode = str(_structured(params.get("sample_weight_mode", "none"))).strip().lower()
    if sw_mode not in {"none", "balanced", "sqrt_balanced"}:
        raise InvalidParams(f"Unsupported sample_weight_mode: {sw_mode!r}")

    tc = _structured(params.get("threshold_calibration", "optimize_qwk"))
    tc_norm = "none" if _is_none_like(tc) else str(tc).strip().lower()
    if tc_norm not in {"optimize_qwk", "quantile", "fixed_midpoint", "round", "none"}:
        raise InvalidParams(f"Unsupported threshold_calibration: {tc!r}")

    tc_src = str(_structured(params.get("threshold_calibration_source", "oof"))).strip().lower()
    if tc_src not in {"oof", "oob_or_oof", "train"}:
        raise InvalidParams(f"Unsupported threshold_calibration_source: {tc_src!r}")

    _coerce_class_weight(params.get("class_weight", None))


def verify_competition_submission(
    submission_df: pd.DataFrame,
    sample_sub_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Validate ``submission.csv`` against CMI PIU Kaggle competition rules.

    Checks:
    1. Exact schema ``['id', 'sii']``.
    2. Zero missing values in ``id`` or ``sii``.
    3. Integer-valued predictions strictly in ``{0, 1, 2, 3}``.
    4. Exact row count and ordered ``id`` match against ``sample_submission.csv`` if supplied.
    """
    if list(submission_df.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"Submission columns must be {[ID_COL, TARGET_COL]}, got {list(submission_df.columns)}")
    if submission_df[ID_COL].isna().any() or submission_df[TARGET_COL].isna().any():
        raise ValueError("Submission contains NaN/null values.")
    if not pd.api.types.is_integer_dtype(submission_df[TARGET_COL]):
        raise ValueError(f"Submission '{TARGET_COL}' must have an integer dtype, got {submission_df[TARGET_COL].dtype}")

    vals = submission_df[TARGET_COL].to_numpy(dtype=np.int64)
    invalid_vals = set(np.unique(vals).tolist()) - {0, 1, 2, 3}
    if invalid_vals:
        raise ValueError(f"Submission '{TARGET_COL}' contains values outside {{0, 1, 2, 3}}: {sorted(invalid_vals)}")

    matches_sample_sub = None
    if sample_sub_path is not None and Path(sample_sub_path).exists():
        ref = pd.read_csv(sample_sub_path)
        if len(submission_df) != len(ref):
            raise ValueError(f"Submission row count {len(submission_df)} != sample_submission {len(ref)}")
        if not np.array_equal(submission_df[ID_COL].astype(str).to_numpy(), ref[ID_COL].astype(str).to_numpy()):
            raise ValueError("Submission 'id' sequence does not match sample_submission.csv.")
        matches_sample_sub = True

    dist = {int(k): int(v) for k, v in zip(*np.unique(vals, return_counts=True))}
    return {
        "valid": True,
        "n_rows": int(len(submission_df)),
        "columns": list(submission_df.columns),
        "sii_dtype": str(submission_df[TARGET_COL].dtype),
        "matches_sample_submission": matches_sample_sub,
        "pred_distribution": dist,
    }


# ---------------------------------------------------------------------------
# LightGBM Sequence Estimator
# ---------------------------------------------------------------------------

class LightGBMSequenceEstimator(BaseEstimator):
    """End-to-end LightGBM estimator wrapping ``SequenceExtractor`` & ``TSAugmentor``.

    Key design features for CMI PIU (ordinal ``sii`` in ``{0, 1, 2, 3}``, metric: QWK):
    * **Leak-free featurization & imputation**: ``SequenceExtractor`` + feature sanitizer are
      strictly fit on training-fold subjects only.
    * **Native missingness & dual task modes**: Supports ``task_mode='regressor'`` (L2 / Huber /
      Fair / L1 / Poisson / Tweedie) and ``task_mode='classifier'`` (expected-value probability
      projection or argmax), with native LightGBM ``NaN`` split routing by default.
    * **Internal 3-fold OOF early stopping & QWK threshold calibration**: Estimates optimal
      ``effective_n_estimators_`` via validation early stopping (when enabled and not DART) and
      calibrates ordinal thresholds ``(t0, t1, t2)`` on leak-free out-of-fold predictions so cutpoints
      do not overfit in-sample tree predictions.
    """

    def __init__(
        self,
        extractor: Optional[SequenceExtractor] = None,
        augmentor: Optional[TSAugmentor] = None,
        task_mode: str = "regressor",
        boosting_type: str = "gbdt",
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        max_depth: Optional[int] = 6,
        min_child_samples: int = 20,
        min_child_weight: float = 1e-3,
        min_split_gain: float = 0.0,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        max_bin: int = 255,
        extra_trees: bool = False,
        path_smooth: float = 0.0,
        drop_rate: float = 0.1,
        reg_objective: str = "regression",
        clf_prediction_mode: str = "expected_value",
        early_stopping_rounds: Optional[int] = 40,
        class_weight: Optional[str] = None,
        sample_weight_mode: str = "none",
        nan_handling: str = "native",
        drop_all_nan_cols: bool = True,
        threshold_calibration: str = "optimize_qwk",
        threshold_calibration_source: str = "oof",
        feature_selection_method: Optional[str] = None,
        static_top_k: Optional[int] = None,
        feature_selection_percentile: Optional[float] = None,
        filter_method: Optional[str] = None,
        filter_threshold: Optional[float] = None,
        max_outlier_features: Optional[int] = None,
        validation_split: float = 0.20,
        invalid_candidate_policy: str = "penalize",
        invalid_candidate_score: float = -1.0,
        candidate_failure_log: Optional[str] = None,
        primary_target: str = TARGET_COL,
        validation_group_column: Optional[str] = None,
        lgbm_n_jobs: int = -1,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.extractor = extractor
        self.augmentor = augmentor
        self.task_mode = task_mode
        self.boosting_type = boosting_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.max_bin = max_bin
        self.extra_trees = extra_trees
        self.path_smooth = path_smooth
        self.drop_rate = drop_rate
        self.reg_objective = reg_objective
        self.clf_prediction_mode = clf_prediction_mode
        self.early_stopping_rounds = early_stopping_rounds
        self.class_weight = class_weight
        self.sample_weight_mode = sample_weight_mode
        self.nan_handling = nan_handling
        self.drop_all_nan_cols = drop_all_nan_cols
        self.threshold_calibration = threshold_calibration
        self.threshold_calibration_source = threshold_calibration_source
        self.feature_selection_method = feature_selection_method
        self.static_top_k = static_top_k
        self.feature_selection_percentile = feature_selection_percentile
        self.filter_method = filter_method
        self.filter_threshold = filter_threshold
        self.max_outlier_features = max_outlier_features
        self.validation_split = validation_split
        self.invalid_candidate_policy = invalid_candidate_policy
        self.invalid_candidate_score = invalid_candidate_score
        self.candidate_failure_log = candidate_failure_log
        self.primary_target = primary_target
        self.validation_group_column = validation_group_column
        self.lgbm_n_jobs = lgbm_n_jobs
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
        ext = clone(self.extractor) if self.extractor is not None else SequenceExtractor()
        if self.feature_selection_method is not None:
            ext.feature_selection_method = self.feature_selection_method
        if self.static_top_k is not None and ext.static_top_k is None:
            ext.static_top_k = self.static_top_k
        if self.feature_selection_percentile is not None and ext.feature_selection_percentile is None:
            ext.feature_selection_percentile = self.feature_selection_percentile
        if self.filter_method is not None:
            ext.filter_method = self.filter_method
        if self.filter_threshold is not None:
            ext.filter_threshold = self.filter_threshold
        if self.max_outlier_features is not None:
            ext.max_outlier_features = self.max_outlier_features
        return ext

    def _get_augmentor(self) -> TSAugmentor:
        return clone(self.augmentor) if self.augmentor is not None else TSAugmentor()

    def _effective_es_rounds(self) -> Optional[int]:
        bt = str(_structured(self.boosting_type)).strip().lower()
        if bt == "dart":
            return None
        es = _coerce_opt_int(self.early_stopping_rounds, min_val=1)
        return es

    def _build_lgbm_model(
        self,
        *,
        n_estimators_override: Optional[int] = None,
        seed_offset: int = 0,
        reg_objective_override: Optional[str] = None,
        n_samples_train: Optional[int] = None,
    ):
        task_mode = str(_structured(self.task_mode)).strip().lower()
        req_bt = str(_structured(self.boosting_type)).strip().lower()
        n_est = max(1, int(round(float(_structured(n_estimators_override or self.n_estimators)))))
        lr = float(_structured(self.learning_rate))
        num_leaves = max(2, int(round(float(_structured(self.num_leaves)))))
        max_d = _coerce_max_depth(self.max_depth)
        if max_d > 0:
            num_leaves = min(num_leaves, max(2, 2 ** min(max_d, 10)))

        mcs = max(1, int(round(float(_structured(self.min_child_samples)))))
        if n_samples_train is not None and n_samples_train >= 4:
            mcs = min(mcs, max(1, n_samples_train // 3))

        mcw = float(_structured(self.min_child_weight))
        msg = float(_structured(self.min_split_gain))
        colsample = float(_structured(self.colsample_bytree))
        ra = float(_structured(self.reg_alpha))
        rl = float(_structured(self.reg_lambda))
        mb = max(15, int(round(float(_structured(self.max_bin)))))
        et = _coerce_bool(self.extra_trees, default=False)
        ps = float(_structured(self.path_smooth))
        dr = float(_structured(self.drop_rate))
        n_jobs = int(_structured(self.lgbm_n_jobs))
        rs = int(_structured(self.random_state)) + int(seed_offset)
        lgbm_verb = 0 if (int(self.verbose or 0) >= 3 and seed_offset == 0) else -1

        common_kwargs: Dict[str, Any] = {
            "n_estimators": n_est,
            "learning_rate": lr,
            "num_leaves": num_leaves,
            "max_depth": max_d,
            "min_child_samples": mcs,
            "min_child_weight": mcw,
            "min_split_gain": msg,
            "colsample_bytree": colsample,
            "reg_alpha": ra,
            "reg_lambda": rl,
            "max_bin": mb,
            "extra_trees": et,
            "path_smooth": ps,
            "n_jobs": n_jobs,
            "random_state": rs,
            "verbosity": lgbm_verb,
            "force_col_wise": True,
        }

        if req_bt == "goss":
            common_kwargs["boosting_type"] = "gbdt"
            common_kwargs["data_sample_strategy"] = "goss"
            common_kwargs["top_rate"] = 0.2
            common_kwargs["other_rate"] = 0.1
            common_kwargs["subsample"] = 1.0
            common_kwargs["subsample_freq"] = 0
        else:
            common_kwargs["boosting_type"] = req_bt
            sub = float(_structured(self.subsample))
            sub_f = int(round(float(_structured(self.subsample_freq))))
            common_kwargs["subsample"] = sub
            common_kwargs["subsample_freq"] = sub_f if sub < 1.0 else 0
            if req_bt == "dart":
                common_kwargs["drop_rate"] = dr

        if task_mode == "regressor":
            obj = reg_objective_override or str(_structured(self.reg_objective)).strip().lower()
            return LGBMRegressor(objective=obj, **common_kwargs)

        cw = _coerce_class_weight(self.class_weight)
        return LGBMClassifier(class_weight=cw, **common_kwargs)

    def _fit_feature_sanitizer(self, feat_df: pd.DataFrame, support_mask: np.ndarray) -> None:
        raw_cols = list(feat_df.columns)
        sub_df = feat_df.iloc[np.flatnonzero(support_mask)] if support_mask.any() else feat_df
        mat = sub_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
        mat[~np.isfinite(mat)] = np.nan

        if _coerce_bool(self.drop_all_nan_cols, default=True) and mat.shape[1] > 1:
            keep_mask = ~np.all(np.isnan(mat), axis=0)
            if not keep_mask.any():
                keep_mask[0] = True
            self.feature_names_out_ = [c for c, k in zip(raw_cols, keep_mask) if k]
            mat = mat[:, keep_mask]
        else:
            self.feature_names_out_ = raw_cols

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            meds = np.nanmedian(mat, axis=0)
            mins = np.nanmin(mat, axis=0)
            stds = np.nanstd(mat, axis=0)

        meds = np.where(np.isfinite(meds), meds, 0.0)
        stds = np.where(np.isfinite(stds) & (stds > 1e-6), stds, 1.0)
        sentinels = np.where(np.isfinite(mins), mins - 5.0 * stds, -999.0)

        self.col_medians_ = meds.astype(np.float64)
        self.col_stds_ = stds.astype(np.float64)
        self.col_sentinels_ = sentinels.astype(np.float64)
        self.effective_nan_handling_ = str(_structured(self.nan_handling)).strip().lower()

    def _transform_feature_matrix(self, feat_df: pd.DataFrame, *, nan_mode: Optional[str] = None) -> np.ndarray:
        aligned = feat_df.reindex(columns=self.feature_names_out_)
        mat = aligned.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
        mat[~np.isfinite(mat)] = np.nan
        mode = nan_mode or self.effective_nan_handling_

        if mode == "native":
            return mat.astype(np.float32)

        nan_mask = np.isnan(mat)
        if nan_mask.any():
            r_idx, c_idx = np.where(nan_mask)
            if mode == "sentinel":
                mat[r_idx, c_idx] = self.col_sentinels_[c_idx]
            elif mode == "zero":
                mat[r_idx, c_idx] = 0.0
            else:
                mat[r_idx, c_idx] = self.col_medians_[c_idx]
        return mat.astype(np.float32)

    def _apply_training_augmentation(
        self,
        feat_df_fit: pd.DataFrame,
        y_fit: np.ndarray,
        X_fit_raw: pd.DataFrame,
        ext: SequenceExtractor,
        aug: TSAugmentor,
    ) -> Tuple[pd.DataFrame, np.ndarray, int]:
        """Generate training-fold augmented copies via modality dropout & actigraphy perturbations."""
        self.augmented_chunk_count_ = 0
        if not aug.enabled():
            return feat_df_fit, y_fit, 0

        rng = np.random.default_rng(int(_structured(self.random_state)))
        active_ops = aug.active_ops()
        if active_ops:
            try:
                ch_res = ext.transform_chunks(strip_leaky_columns(X_fit_raw))
                if ch_res["chunks"].shape[0] > 0:
                    aug_chunks = aug.augment_array(ch_res["chunks"], rng=rng)
                    self.augmented_chunk_count_ = int(aug_chunks.shape[0])
            except Exception:
                self.augmented_chunk_count_ = 0

        n_copies = max(1, min(int(_structured(aug.n_copies)), 4))
        seq_cols = [
            c for c in feat_df_fit.columns
            if str(c).startswith(("seq_", "noop_")) and c not in {"seq_has_actigraphy"}
        ]
        if not seq_cols:
            return feat_df_fit, y_fit, 0

        has_act = (
            (pd.to_numeric(feat_df_fit["seq_has_actigraphy"], errors="coerce").fillna(0.0).to_numpy() > 0.5)
            if "seq_has_actigraphy" in feat_df_fit.columns
            else np.ones(len(feat_df_fit), dtype=bool)
        )
        act_idx = np.flatnonzero(has_act)
        if len(act_idx) == 0:
            return feat_df_fit, y_fit, 0

        seq_mat_ref = feat_df_fit.iloc[act_idx][seq_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            seq_stds = np.nanstd(seq_mat_ref, axis=0)
        seq_stds = np.where(np.isfinite(seq_stds) & (seq_stds > 1e-6), seq_stds, 0.0)

        p_gate = float(np.clip(aug.prob, 0.0, 1.0))
        p_op = float(np.clip(aug.per_aug_prob, 0.0, 1.0))
        p_sensor_drop = max(float(aug.sensor_drop_prob), float(aug.sensor_drop_joint_prob))
        p_chan_drop = float(aug.channel_drop_prob) if "channel_drop" in active_ops else 0.0
        noise_scale = max(
            float(aug.jitter_sigma) if "jitter" in active_ops else 0.0,
            float(aug.noise_std) if "noise" in active_ops else 0.0,
            0.25 * float(aug.warp_sigma) if "magnitude_warp" in active_ops else 0.0,
            0.15 * float(aug.time_shift_frac) if "time_shift" in active_ops else 0.0,
        )
        scale_sigma = float(aug.scaling_sigma) if "scaling" in active_ops else 0.0

        aug_dfs: List[pd.DataFrame] = [feat_df_fit]
        aug_ys: List[np.ndarray] = [y_fit]
        total_added = 0

        for _ in range(n_copies):
            pick = act_idx[rng.random(len(act_idx)) < p_gate]
            if len(pick) == 0:
                continue
            copy_df = feat_df_fit.iloc[pick].copy()
            copy_y = y_fit[pick].copy()
            mat = copy_df[seq_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)

            if noise_scale > 0.0 and rng.random() < p_op:
                eff_sigma = min(noise_scale, 0.5)
                mat = mat + rng.normal(0.0, eff_sigma, size=mat.shape) * seq_stds[None, :]
            if scale_sigma > 0.0 and rng.random() < p_op:
                eff_s = min(scale_sigma, 0.5)
                mat = mat * rng.normal(1.0, eff_s, size=mat.shape)
            if p_chan_drop > 0.0 and rng.random() < p_op:
                drop_c = rng.random(mat.shape) < min(p_chan_drop, 0.75)
                mat[drop_c] = np.nan

            copy_df[seq_cols] = mat

            if p_sensor_drop > 0.0:
                drop_rows = rng.random(len(copy_df)) < p_sensor_drop
                if drop_rows.any():
                    copy_df.loc[copy_df.index[drop_rows], seq_cols] = np.nan
                    if "seq_has_actigraphy" in copy_df.columns:
                        copy_df.loc[copy_df.index[drop_rows], "seq_has_actigraphy"] = 0.0
                    if "miss_actigraphy_absent" in copy_df.columns:
                        copy_df.loc[copy_df.index[drop_rows], "miss_actigraphy_absent"] = 1.0

            aug_dfs.append(copy_df)
            aug_ys.append(copy_y)
            total_added += len(copy_df)

        if total_added == 0:
            return feat_df_fit, y_fit, 0
        return pd.concat(aug_dfs, axis=0, ignore_index=True), np.concatenate(aug_ys, axis=0), total_added

    def _compute_sample_weights(self, y_arr: np.ndarray) -> Optional[np.ndarray]:
        if str(_structured(self.task_mode)).strip().lower() != "regressor":
            return None
        mode = str(_structured(self.sample_weight_mode)).strip().lower()
        if mode == "none":
            return None
        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) < 2:
            return None
        n = float(len(y_arr))
        k = float(len(classes))
        raw_w = {int(c): n / max(k * float(cnt), 1.0) for c, cnt in zip(classes, counts)}
        if mode == "sqrt_balanced":
            raw_w = {c: float(np.sqrt(w)) for c, w in raw_w.items()}
        weights = np.array([raw_w[int(v)] for v in y_arr], dtype=np.float64)
        mean_w = float(np.mean(weights))
        return weights / mean_w if mean_w > 0 else weights

    def _continuous_from_model(self, model: Any, X_mat: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if str(_structured(self.task_mode)).strip().lower() == "regressor":
                return np.asarray(model.predict(X_mat), dtype=np.float64).ravel()
            proba = np.asarray(model.predict_proba(X_mat), dtype=np.float64)
        cls_vals = np.asarray(model.classes_, dtype=np.float64).ravel()
        return proba @ cls_vals

    def _safe_reg_objective(self, y_arr: np.ndarray) -> Optional[str]:
        if str(_structured(self.task_mode)).strip().lower() != "regressor":
            return None
        obj = str(_structured(self.reg_objective)).strip().lower()
        if obj in {"poisson", "tweedie"} and float(np.sum(np.maximum(y_arr, 0))) <= 0.0:
            return "regression"
        return obj

    def _fit_single_lgbm(
        self,
        model: Any,
        X_mat: np.ndarray,
        y_arr: np.ndarray,
        sample_weight: Optional[np.ndarray],
        *,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        early_stopping_rounds: Optional[int] = None,
    ) -> Any:
        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None and early_stopping_rounds is not None and early_stopping_rounds > 0:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(int(early_stopping_rounds), verbose=False),
                lgb.log_evaluation(period=0),
            ]
        y_fit = np.maximum(y_arr, 0) if (
            str(_structured(self.task_mode)).strip().lower() == "regressor"
            and str(getattr(model, "objective", "")) in {"poisson", "tweedie"}
        ) else y_arr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X_mat, y_fit, **fit_kwargs)
            except Exception as exc:
                # Fallback if eval_set/objective fails on tiny imbalanced splits
                if "eval_set" in fit_kwargs:
                    fit_kwargs.pop("eval_set", None)
                    fit_kwargs.pop("callbacks", None)
                    model.fit(X_mat, y_fit, **fit_kwargs)
                else:
                    raise exc
        return model

    def _compute_oof_calibration_and_best_iter(
        self,
        X_orig_mat: np.ndarray,
        y_orig: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, int]:
        """Run internal stratified OOF CV to gather early-stopping iteration & leak-free OOF scores."""
        max_n_est = max(1, int(round(float(_structured(self.n_estimators)))))
        es_rounds = self._effective_es_rounds()
        tc_src = str(_structured(self.threshold_calibration_source)).strip().lower()
        need_oof_scores = tc_src in {"oof", "oob_or_oof"}

        if len(y_orig) < 12 or len(np.unique(y_orig)) < 2 or (not need_oof_scores and es_rounds is None):
            return None, None, ("train" if not need_oof_scores else "train_fallback"), max_n_est

        _, counts = np.unique(y_orig, return_counts=True)
        min_cls = int(counts.min()) if len(counts) else 0
        if min_cls >= 3:
            splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(_structured(self.random_state)))
        elif min_cls == 2:
            splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=int(_structured(self.random_state)))
        else:
            splitter = KFold(n_splits=min(3, max(2, len(y_orig) // 4)), shuffle=True, random_state=int(_structured(self.random_state)))

        task_mode = str(_structured(self.task_mode)).strip().lower()
        oof = np.full(len(y_orig), np.nan, dtype=np.float64)
        fold_best_iters: List[int] = []

        for fold_idx, (tr_i, va_i) in enumerate(splitter.split(X_orig_mat, y_orig)):
            y_tr_f = y_orig[tr_i]
            y_va_f = y_orig[va_i]
            if len(np.unique(y_tr_f)) < 2:
                continue

            safe_obj = self._safe_reg_objective(y_tr_f)
            fold_model = self._build_lgbm_model(
                n_estimators_override=max_n_est,
                seed_offset=100 + fold_idx,
                reg_objective_override=safe_obj,
                n_samples_train=len(tr_i),
            )
            sw_tr = self._compute_sample_weights(y_tr_f)

            can_es = (
                es_rounds is not None
                and es_rounds > 0
                and len(va_i) >= 4
                and (task_mode == "regressor" or set(np.unique(y_va_f)).issubset(set(np.unique(y_tr_f))))
            )
            self._fit_single_lgbm(
                fold_model,
                X_orig_mat[tr_i],
                y_tr_f,
                sw_tr,
                eval_set=[(X_orig_mat[va_i], y_va_f)] if can_es else None,
                early_stopping_rounds=es_rounds if can_es else None,
            )
            best_it = getattr(fold_model, "best_iteration_", None)
            if can_es and best_it is not None and int(best_it) > 0:
                fold_best_iters.append(int(best_it))
            oof[va_i] = self._continuous_from_model(fold_model, X_orig_mat[va_i])

        if fold_best_iters:
            eff_n = int(np.clip(round(float(np.median(fold_best_iters))), min(10, max_n_est), max_n_est))
        else:
            eff_n = max_n_est

        fin = np.isfinite(oof)
        if need_oof_scores and int(fin.sum()) >= max(8, int(0.75 * len(y_orig))):
            n_folds_lbl = "3fold" if min_cls >= 3 or min_cls < 2 else "2fold"
            return oof[fin], y_orig[fin], f"oof_{n_folds_lbl}", eff_n
        return None, None, ("train" if not need_oof_scores else "train_fallback"), eff_n

    def _fit_thresholds(self, y_cal: np.ndarray, scores_cal: np.ndarray) -> np.ndarray:
        tc = _structured(self.threshold_calibration)
        mode = "none" if _is_none_like(tc) else str(tc).strip().lower()
        sc = np.asarray(scores_cal, dtype=np.float64).ravel()
        yc = np.asarray(y_cal, dtype=np.int64).ravel()
        sd = float(np.nanstd(sc)) if len(sc) else 0.0
        if mode in {"fixed_midpoint", "round", "none"} or len(sc) < 4 or not np.isfinite(sd) or sd <= 1e-12:
            return np.array([0.5, 1.5, 2.5], dtype=np.float64)
        if mode == "optimize_qwk":
            return optimize_thresholds_qwk(yc, sc)
        if mode == "quantile":
            p0 = float(np.clip(np.mean(yc <= 0), 0.05, 0.90))
            p1 = float(np.clip(np.mean(yc <= 1), 0.15, 0.96))
            p2 = float(np.clip(np.mean(yc <= 2), 0.70, 0.99))
            q = np.sort(np.quantile(sc, [p0, p1, p2])).astype(np.float64)
            min_gap = max(1e-4, 0.02 * sd)
            if int((yc == 3).sum()) == 0:
                q[2] = max(q[2], float(np.nanmax(sc)) + min_gap)
            q[1] = max(q[1], q[0] + min_gap)
            q[2] = max(q[2], q[1] + min_gap)
            return q
        raise InvalidParams(f"Unsupported threshold_calibration: {self.threshold_calibration!r}")

    def fit(self, X: pd.DataFrame, y: Any):
        t_start = time.perf_counter()
        v_lvl = int(self.verbose or 0)
        self.fit_failed_ = False
        validate_lgbm_params(self.get_params(deep=False))

        y_arr = np.asarray(
            y[self.primary_target] if isinstance(y, pd.DataFrame) and self.primary_target in y.columns else y,
            dtype=np.int64,
        ).ravel()

        ids_all = X[ID_COL].astype(str).to_numpy() if ID_COL in X.columns else np.arange(len(X)).astype(str)
        v_split = float(self.validation_split or 0.0)
        if 0.0 < v_split < 0.9 and len(X) >= 10:
            try:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=v_split, random_state=int(_structured(self.random_state)))
                tr_i, va_i = next(sss.split(X, y_arr))
                self.train_sequence_ids_ = ids_all[tr_i]
                self.validation_sequence_ids_ = ids_all[va_i]
            except Exception:
                cut = max(1, int(round(len(X) * (1.0 - v_split))))
                self.train_sequence_ids_ = ids_all[:cut]
                self.validation_sequence_ids_ = ids_all[cut:]
        else:
            self.train_sequence_ids_ = ids_all
            self.validation_sequence_ids_ = np.asarray([], dtype=str)
        self.internal_validation_held_out_ = False

        ext = self._get_extractor()
        ext.validate()
        self.extractor_ = ext
        aug = self._get_augmentor()
        self.augmentor_ = aug

        n_act = int(X["has_actigraphy"].sum()) if isinstance(X, pd.DataFrame) and "has_actigraphy" in X.columns else 0
        if v_lvl >= 2:
            fams_str = str(getattr(ext, "feature_families", "none"))
            if len(fams_str) > 64:
                fams_str = fams_str[:61] + "..."
            print(
                f"  [LGBM fit] Extracting features: n={len(X)} (act={n_act}) | bin={ext.bin_minutes}m | "
                f"missing={ext.missing_feature_strategy} | families={fams_str}",
                flush=True,
            )

        t_ext0 = time.perf_counter()
        X_clean = strip_leaky_columns(X)
        feat_df = ext.fit_transform(X_clean, y_arr)
        t_ext = time.perf_counter() - t_ext0
        if v_lvl >= 1:
            print(f"  {ext.format_cache_log(elapsed_s=t_ext)}", flush=True)
        self.feature_names_in_ = list(feat_df.columns)
        self.selected_static_cols_ = list(getattr(ext, "selected_static_cols_", []))

        support_mask = np.asarray(getattr(ext, "support_mask_", np.ones(len(X), dtype=bool)), dtype=bool)
        if len(support_mask) != len(X) or not support_mask.any():
            support_mask = np.ones(len(X), dtype=bool)
        self.support_mask_ = support_mask
        self.n_samples_removed_ = int(getattr(ext, "n_samples_removed_", int((~support_mask).sum())))
        self.n_samples_retained_ = int(getattr(ext, "n_samples_retained_", int(support_mask.sum())))

        # Fit leak-free feature sanitizer strictly on retained training-fold rows
        self._fit_feature_sanitizer(feat_df, support_mask)

        sup_idx = np.flatnonzero(support_mask)
        feat_fit_orig = feat_df.iloc[sup_idx].reset_index(drop=True)
        y_fit_orig = y_arr[sup_idx]
        X_fit_raw = X_clean.iloc[sup_idx].reset_index(drop=True)

        raw_nan_pct = float(100.0 * feat_fit_orig.reindex(columns=self.feature_names_out_).isna().to_numpy().mean()) if len(feat_fit_orig) and self.feature_names_out_ else 0.0

        # Optional training-only augmentation
        feat_fit_train, y_fit_train, n_aug_rows = self._apply_training_augmentation(
            feat_fit_orig, y_fit_orig, X_fit_raw, ext, aug
        )
        self.n_augmented_rows_ = int(n_aug_rows)
        aug_applied = n_aug_rows > 0

        X_orig_mat = self._transform_feature_matrix(feat_fit_orig)
        X_train_mat = self._transform_feature_matrix(feat_fit_train) if aug_applied else X_orig_mat
        sw_train = self._compute_sample_weights(y_fit_train)

        # Internal 3-fold OOF early stopping & leak-free calibration scores
        t_cal0 = time.perf_counter()
        cal_scores_oof, cal_y_oof, cal_src, eff_n_est = self._compute_oof_calibration_and_best_iter(
            X_orig_mat, y_fit_orig
        )
        self.effective_n_estimators_ = int(eff_n_est)

        safe_obj = self._safe_reg_objective(y_fit_train)
        lgbm_model = self._build_lgbm_model(
            n_estimators_override=self.effective_n_estimators_,
            reg_objective_override=safe_obj,
            n_samples_train=len(y_fit_train),
        )

        if v_lvl >= 2:
            cls_name = type(lgbm_model).__name__
            extra_lbl = (
                f"obj={lgbm_model.objective} sw={self.sample_weight_mode}"
                if str(_structured(self.task_mode)).strip().lower() == "regressor"
                else f"cw={lgbm_model.class_weight} pred={self.clf_prediction_mode}"
            )
            print(
                f"  [LGBM fit] Features ready in {t_ext:.2f}s: retained={self.n_samples_retained_}/{len(X)} "
                f"(outliers_dropped={self.n_samples_removed_}, aug=+{self.n_augmented_rows_}) | "
                f"features={len(self.feature_names_out_)} ({raw_nan_pct:.1f}% NaN -> {self.effective_nan_handling_})",
                flush=True,
            )
            print(
                f"  [LGBM fit] Training {cls_name}(bt={self.boosting_type}, n_est={self.effective_n_estimators_}/{self.n_estimators}, "
                f"lr={lgbm_model.learning_rate}, leaves={lgbm_model.num_leaves}, depth={lgbm_model.max_depth}, "
                f"min_child={lgbm_model.min_child_samples}, {extra_lbl}) on {X_train_mat.shape}...",
                flush=True,
            )

        t_lgb0 = time.perf_counter()
        lgbm_model = self._fit_single_lgbm(lgbm_model, X_train_mat, y_fit_train, sw_train)
        t_lgb = time.perf_counter() - t_lgb0
        self.model_ = lgbm_model

        # Extract gain and split importances
        n_feat = len(self.feature_names_out_)
        try:
            booster = lgbm_model.booster_
            gain_imp = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
            split_imp = np.asarray(booster.feature_importance(importance_type="split"), dtype=np.float64)
        except Exception:
            gain_imp = np.asarray(getattr(lgbm_model, "feature_importances_", np.zeros(n_feat)), dtype=np.float64)
            split_imp = gain_imp.copy()

        if len(gain_imp) != n_feat:
            gain_imp = np.zeros(n_feat, dtype=np.float64)
            split_imp = np.zeros(n_feat, dtype=np.float64)
        self.feature_importances_gain_ = gain_imp
        self.feature_importances_split_ = split_imp
        gain_sum = float(np.sum(gain_imp))
        self.feature_importances_ = (gain_imp / gain_sum) if gain_sum > 0.0 else np.zeros(n_feat, dtype=np.float64)

        use_argmax = (
            str(_structured(self.task_mode)).strip().lower() == "classifier"
            and str(_structured(self.clf_prediction_mode)).strip().lower() == "argmax"
        )
        if use_argmax:
            self.thresholds_ = np.array([0.5, 1.5, 2.5], dtype=np.float64)
            self.calibration_source_ = "argmax"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_preds = np.clip(np.asarray(lgbm_model.predict(X_orig_mat), dtype=np.int64).ravel(), 0, 3)
            self.calib_qwk_ = float(quadratic_weighted_kappa(y_fit_orig, train_preds))
            self.calib_accuracy_pct_ = float(100.0 * np.mean(y_fit_orig == train_preds)) if len(y_fit_orig) else np.nan
        else:
            if cal_scores_oof is not None and cal_y_oof is not None:
                cal_scores, cal_y = cal_scores_oof, cal_y_oof
                self.calibration_source_ = cal_src
            else:
                cal_scores = self._continuous_from_model(lgbm_model, X_orig_mat)
                cal_y = y_fit_orig
                self.calibration_source_ = cal_src
            self.thresholds_ = self._fit_thresholds(cal_y, cal_scores)
            cal_preds = apply_thresholds(cal_scores, self.thresholds_)
            self.calib_qwk_ = float(quadratic_weighted_kappa(cal_y, cal_preds))
            self.calib_accuracy_pct_ = float(100.0 * np.mean(cal_y == cal_preds)) if len(cal_y) else np.nan
            train_scores = self._continuous_from_model(lgbm_model, X_orig_mat)
            train_preds = apply_thresholds(train_scores, self.thresholds_)
        t_cal = time.perf_counter() - t_cal0

        self.train_qwk_ = float(quadratic_weighted_kappa(y_fit_orig, train_preds))
        self.train_accuracy_pct_ = float(100.0 * np.mean(y_fit_orig == train_preds)) if len(y_fit_orig) else np.nan
        self.history_: Dict[str, List[float]] = {
            "loss": [float(1.0 - self.train_qwk_)],
            "val_loss": [float(1.0 - self.calib_qwk_)],
            "qwk": [float(self.train_qwk_)],
            "calib_qwk": [float(self.calib_qwk_)],
            "accuracy_pct": [float(self.train_accuracy_pct_)],
            "calib_accuracy_pct": [float(self.calib_accuracy_pct_)],
        }
        self.classes_ = np.array([0, 1, 2, 3], dtype=np.int64)

        if v_lvl >= 2:
            t_tot = time.perf_counter() - t_start
            th_str = "[" + ", ".join(f"{float(t):.3f}" for t in self.thresholds_) + "]"
            print(
                f"  [LGBM fit] Done in {t_tot:.2f}s (ext={t_ext:.2f}s, lgbm={t_lgb:.2f}s, oof/cal={t_cal:.2f}s) | "
                f"cal_src={self.calibration_source_} | thresholds={th_str} | "
                f"calib: Normal={_fmt_pct(self.calib_accuracy_pct_)} -> QWK={self.calib_qwk_:.5f} | "
                f"train: Normal={_fmt_pct(self.train_accuracy_pct_)} -> QWK={self.train_qwk_:.5f}",
                flush=True,
            )
        if v_lvl >= 3 and len(self.feature_names_out_) > 0:
            top_gain = self.get_feature_importances_df().head(5)
            top_str = ", ".join(f"{r['feature']}={r['lgbm_importance']:.4f}" for _, r in top_gain.iterrows())
            print(f"  [LGBM fit] Top 5 gain features: {top_str}", flush=True)
        return self

    def get_history_dict(self) -> Dict[str, List[float]]:
        return dict(getattr(self, "history_", {}))

    def predict_continuous_from_features(self, feat_df: pd.DataFrame) -> np.ndarray:
        X_mat = self._transform_feature_matrix(feat_df)
        return self._continuous_from_model(self.model_, X_mat)

    def predict_from_features(self, feat_df: pd.DataFrame) -> np.ndarray:
        X_mat = self._transform_feature_matrix(feat_df)
        if (
            str(_structured(self.task_mode)).strip().lower() == "classifier"
            and str(_structured(self.clf_prediction_mode)).strip().lower() == "argmax"
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.clip(np.asarray(self.model_.predict(X_mat), dtype=np.int64).ravel(), 0, 3)
        scores = self._continuous_from_model(self.model_, X_mat)
        return apply_thresholds(scores, self.thresholds_)

    def predict_continuous(self, X: pd.DataFrame) -> np.ndarray:
        feat_df = self.extractor_.transform(strip_leaky_columns(X))
        return self.predict_continuous_from_features(feat_df)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if str(_structured(self.task_mode)).strip().lower() != "classifier":
            raise AttributeError("predict_proba is only available when task_mode='classifier'.")
        feat_df = self.extractor_.transform(strip_leaky_columns(X))
        X_mat = self._transform_feature_matrix(feat_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_p = np.asarray(self.model_.predict_proba(X_mat), dtype=np.float64)
        full_p = np.zeros((len(X_mat), 4), dtype=np.float64)
        for j, cls in enumerate(self.model_.classes_):
            c_int = int(cls)
            if 0 <= c_int <= 3:
                full_p[:, c_int] = raw_p[:, j]
        return full_p

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        feat_df = self.extractor_.transform(strip_leaky_columns(X))
        return self.predict_from_features(feat_df)

    def score(self, X: pd.DataFrame, y: Any) -> float:
        return competition_score(y, self.predict(X))

    def get_feature_importances_df(self) -> pd.DataFrame:
        """Return LightGBM Gain & Split feature importances sorted descending by normalized Gain."""
        cols = list(getattr(self, "feature_names_out_", []))
        imps = np.asarray(getattr(self, "feature_importances_", np.zeros(len(cols))), dtype=np.float64)
        gains = np.asarray(getattr(self, "feature_importances_gain_", np.zeros(len(cols))), dtype=np.float64)
        splits = np.asarray(getattr(self, "feature_importances_split_", np.zeros(len(cols))), dtype=np.float64)
        df = pd.DataFrame({
            "feature": cols,
            "family": [_feature_family_label(c) for c in cols],
            "lgbm_importance": imps,
            "lgbm_gain": gains,
            "lgbm_split": splits,
        })
        return df.sort_values("lgbm_importance", ascending=False, kind="stable").reset_index(drop=True)

    def get_model_summary(self) -> Dict[str, Any]:
        """Return compact diagnostic summary of the fitted LightGBM estimator."""
        return {
            "task_mode": str(self.task_mode),
            "boosting_type": str(self.boosting_type),
            "n_estimators_requested": int(round(float(_structured(self.n_estimators)))),
            "effective_n_estimators": int(getattr(self, "effective_n_estimators_", self.n_estimators)),
            "learning_rate": float(_structured(self.learning_rate)),
            "num_leaves": int(round(float(_structured(self.num_leaves)))),
            "max_depth": _coerce_max_depth(self.max_depth),
            "n_features_in": int(len(getattr(self, "feature_names_out_", []))),
            "n_samples_retained": int(getattr(self, "n_samples_retained_", 0)),
            "n_samples_removed": int(getattr(self, "n_samples_removed_", 0)),
            "n_augmented_rows": int(getattr(self, "n_augmented_rows_", 0)),
            "effective_nan_handling": getattr(self, "effective_nan_handling_", self.nan_handling),
            "calibration_source": getattr(self, "calibration_source_", None),
            "thresholds": [round(float(t), 5) for t in getattr(self, "thresholds_", [])],
            "calib_qwk": round(float(getattr(self, "calib_qwk_", np.nan)), 5),
            "train_qwk": round(float(getattr(self, "train_qwk_", np.nan)), 5),
        }


# ---------------------------------------------------------------------------
# LightGBM-Aware Inactive Parameter Pruning, Deduplication & Searchers
# ---------------------------------------------------------------------------

def inactive_lgbm_parameters_for_candidate(params: Mapping[str, Any]) -> set:
    """Identify inactive ``extractor__*``, ``augmentor__*``, and LightGBM hyperparameters."""
    p = {
        (k[len("estimator__"):] if k.startswith("estimator__") else k): _structured(v)
        for k, v in params.items()
    }
    prefix = "estimator__" if any(k.startswith("estimator__") for k in params) else ""
    inactive = inactive_sequence_parameters(params)

    # Inactive augmentor params when prob == 0.0
    if float(p.get("augmentor__prob", 0.0) or 0.0) <= 0.0:
        for name in (
            "per_aug_prob", "augmentations", "n_copies", "jitter_sigma", "noise_std",
            "scaling_sigma", "warp_sigma", "warp_num_knots", "time_warp_max_speed_ratio",
            "time_warp_n_speed_change", "time_shift_frac", "crop_frac_range",
            "temporal_num_masks", "temporal_mask_frac", "temporal_mask_fill",
            "channel_drop_prob", "sensor_drop_prob", "sensor_drop_joint_prob",
        ):
            inactive.add(f"{prefix}augmentor__{name}")

    task_mode = str(p.get("task_mode", "regressor")).strip().lower()
    if task_mode == "regressor":
        for name in ("class_weight", "clf_prediction_mode"):
            inactive.add(f"{prefix}{name}")
    else:
        for name in ("reg_objective", "sample_weight_mode"):
            inactive.add(f"{prefix}{name}")
        if str(p.get("clf_prediction_mode", "expected_value")).strip().lower() == "argmax":
            inactive.add(f"{prefix}threshold_calibration")
            inactive.add(f"{prefix}threshold_calibration_source")

    bt = str(p.get("boosting_type", "gbdt")).strip().lower()
    if bt != "dart":
        inactive.add(f"{prefix}drop_rate")
    else:
        inactive.add(f"{prefix}early_stopping_rounds")

    if bt == "goss":
        inactive.add(f"{prefix}subsample")
        inactive.add(f"{prefix}subsample_freq")
    elif float(p.get("subsample", 0.8)) >= 1.0:
        inactive.add(f"{prefix}subsample_freq")

    tc = p.get("threshold_calibration", "optimize_qwk")
    tc_norm = "none" if _is_none_like(tc) else str(tc).strip().lower()
    es = _coerce_opt_int(p.get("early_stopping_rounds", 40), min_val=1)
    if tc_norm in {"fixed_midpoint", "round", "none"} and (es is None or bt == "dart"):
        inactive.add(f"{prefix}threshold_calibration_source")

    return inactive


def lgbm_candidate_error(params: Mapping[str, Any], base_params: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Validate both ``SequenceExtractor`` DSP bounds and LightGBM parameter bounds."""
    merged = dict(base_params or {})
    merged.update(_decode_skopt_params(params))
    ext_kwargs = {
        k.split("extractor__", 1)[1]: v
        for k, v in merged.items()
        if "extractor__" in k and "noop__" not in k
    }
    lgbm_kwargs = {
        (k[len("estimator__"):] if k.startswith("estimator__") else k): v
        for k, v in merged.items()
        if "extractor__" not in k and "augmentor__" not in k and k not in {"estimator", "steps", "memory", "verbose", "transform_input"}
    }
    try:
        validate_sequence_extractor_params(ext_kwargs)
        validate_lgbm_params(lgbm_kwargs)
    except InvalidParams as exc:
        return str(exc)
    return None


def _canonical_lgbm_candidate_key(params: Mapping[str, Any], base_params: Optional[Mapping[str, Any]] = None) -> str:
    merged = dict(base_params or {})
    merged.update(_decode_skopt_params(params))
    inactive = inactive_lgbm_parameters_for_candidate(merged)
    eff = {k: repr(v) for k, v in sorted(params.items()) if k not in inactive}
    return hashlib.md5(json.dumps(eff, sort_keys=True).encode()).hexdigest()


def deduplicate_lgbm_param_grid(
    param_grid: Mapping[str, Sequence[Any]],
    *,
    base_params: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Expand and deduplicate a LightGBM grid search space, dropping invalid & equivalent clones."""
    grid = list(ParameterGrid(dict(param_grid)))
    seen: set = set()
    kept: List[Dict[str, Any]] = []
    for cand in grid:
        if lgbm_candidate_error(cand, base_params=base_params) is not None:
            continue
        sig = _canonical_lgbm_candidate_key(cand, base_params=base_params)
        if sig not in seen:
            seen.add(sig)
            kept.append(cand)
    return [{k: [v] for k, v in cand.items()} for cand in kept]


def _lgbm_candidate_brief(decoded: Mapping[str, Any]) -> str:
    """Compact summary of active LightGBM & key extractor params for verbose >= 2 logging."""
    def _g(key: str, default: Any = "?") -> Any:
        return decoded.get(f"estimator__{key}", decoded.get(key, default))

    tm = str(_g("task_mode", "regressor")).strip().lower()
    extra = (
        f"obj={_g('reg_objective')} sw={_g('sample_weight_mode')}"
        if tm == "regressor"
        else f"cw={_g('class_weight')} pred={_g('clf_prediction_mode')}"
    )
    return (
        f"task={tm} bt={_g('boosting_type')} trees={_g('n_estimators')} lr={_g('learning_rate')} "
        f"leaves={_g('num_leaves')} depth={_g('max_depth')} min_child={_g('min_child_samples')} "
        f"{extra} nan={_g('nan_handling')} cal={_g('threshold_calibration')}({_g('threshold_calibration_source')}) | "
        f"bin={_g('extractor__bin_minutes')}m miss={_g('extractor__missing_feature_strategy')} "
        f"fs={_g('extractor__feature_selection_method')} aug_p={_g('augmentor__prob')}"
    )


def _evaluate_lgbm_candidate_cv(
    estimator: Any,
    params: Mapping[str, Any],
    X: pd.DataFrame,
    y: Any,
    groups: Optional[Any],
    cv: Any,
    scoring: Any,
    *,
    error_score: float = -1.0,
    verbose: int = 0,
) -> Tuple[List[float], List[float], float]:
    """CV evaluator with verbose fold progress, exception reporting, and verbosity propagation."""
    t0 = time.perf_counter()
    v_lvl = int(verbose or 0)
    qwk_scores: List[float] = []
    acc_pct_scores: List[float] = []
    use_default_qwk = scoring is None or scoring is candidate_score

    splits = list(cv.split(X, y, groups))
    n_splits = len(splits)
    for f_idx, (tr_idx, va_idx) in enumerate(splits):
        t_f0 = time.perf_counter()
        cand_est = clone(estimator).set_params(**params)
        _propagate_verbose(cand_est, v_lvl)

        y_tr = y.iloc[tr_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)[tr_idx]
        y_va = y.iloc[va_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)[va_idx]
        if v_lvl >= 2:
            print(f"  [Fold {f_idx + 1}/{n_splits}] train={len(tr_idx)} val={len(va_idx)}", flush=True)
        try:
            cand_est.fit(X.iloc[tr_idx], y_tr)
            if v_lvl >= 2:
                print(f"  [Fold {f_idx + 1}/{n_splits}] Scoring val={len(va_idx)}...", flush=True)
            t_p0 = time.perf_counter()
            preds = np.asarray(cand_est.predict(X.iloc[va_idx]), dtype=np.int64).ravel()
            t_pred = time.perf_counter() - t_p0
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
            qwk_val = qwk if np.isfinite(qwk) else error_score
            acc_val = acc_pct if np.isfinite(acc_pct) else -1.0
            qwk_scores.append(qwk_val)
            acc_pct_scores.append(acc_val)
            if v_lvl >= 2:
                print(
                    f"  [Fold {f_idx + 1}/{n_splits}] val Normal={acc_val:.2f}% -> Comp QWK={qwk_val:.5f} "
                    f"(pred={t_pred:.2f}s, fold={time.perf_counter() - t_f0:.2f}s)",
                    flush=True,
                )
        except Exception as exc:
            qwk_scores.append(error_score)
            acc_pct_scores.append(-1.0)
            _log_fold_error(f_idx, n_splits, exc, v_lvl)
    return qwk_scores, acc_pct_scores, float(time.perf_counter() - t0)


def _assemble_search_cv_results(
    all_params: List[Dict[str, Any]],
    all_fold_qwk: List[List[float]],
    all_fold_acc_pct: List[List[float]],
    all_times: List[float],
) -> Dict[str, Any]:
    """Assemble sklearn-compatible ``cv_results_`` dictionary for Bayes / GA searches."""
    means = np.array([float(np.mean(s)) for s in all_fold_qwk], dtype=float)
    stds = np.array([float(np.std(s)) for s in all_fold_qwk], dtype=float)
    acc_means = np.array(
        [
            float(np.mean([a for a in s if np.isfinite(a) and a >= 0.0]))
            if any(np.isfinite(a) and a >= 0.0 for a in s)
            else np.nan
            for s in all_fold_acc_pct
        ],
        dtype=float,
    )
    acc_stds = np.array(
        [
            float(np.std([a for a in s if np.isfinite(a) and a >= 0.0]))
            if any(np.isfinite(a) and a >= 0.0 for a in s)
            else np.nan
            for s in all_fold_acc_pct
        ],
        dtype=float,
    )
    ranks = (len(means) - np.argsort(np.argsort(means))).astype(int)
    n_splits = len(all_fold_qwk[0]) if all_fold_qwk else 1

    cv_results: Dict[str, Any] = {
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
        cv_results[f"split{i}_test_score"] = np.array([s[i] for s in all_fold_qwk], dtype=float)
        cv_results[f"split{i}_test_qwk"] = np.array([s[i] for s in all_fold_qwk], dtype=float)
        cv_results[f"split{i}_test_accuracy_pct"] = np.array([s[i] for s in all_fold_acc_pct], dtype=float)
    _populate_param_columns(cv_results, all_params)
    return cv_results


class LGBMHierarchicalBayesSearchCV(BaseEstimator):
    """Bayesian optimization over ``SequenceExtractor``, ``TSAugmentor`` & LightGBM hyperparameters."""

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
    def _default_starting_point(
        keys: Sequence[str],
        dimensions: Sequence[Any],
        base_params: Mapping[str, Any],
        user_x0: Optional[Mapping[str, Any]],
    ) -> List[Any]:
        point: List[Any] = []
        for k, dim in zip(keys, dimensions):
            if user_x0 is not None and k in user_x0:
                val = _encode_skopt_item(user_x0[k])
                cats = getattr(dim, "categories", None)
                if cats is None or val in cats:
                    point.append(val)
                    continue
            bv = 0.0 if "augmentor__prob" in k else base_params.get(k)
            if hasattr(dim, "categories"):
                cats = list(dim.categories)
                enc_bv = "None" if bv is None else _encode_skopt_item(bv)
                if enc_bv in cats:
                    point.append(enc_bv)
                elif str(enc_bv) in cats:
                    point.append(str(enc_bv))
                else:
                    point.append(cats[0])
            else:
                low, high = getattr(dim, "low", 0.0), getattr(dim, "high", 1.0)
                is_int = isinstance(low, (int, np.integer)) and isinstance(high, (int, np.integer))
                if isinstance(bv, (int, float, np.number)) and low <= float(bv) <= high:
                    point.append(int(round(float(bv))) if is_int else float(bv))
                else:
                    point.append(int(low) if is_int else float(low))
        return point

    def fit(self, X: pd.DataFrame, y: Any, groups: Optional[Any] = None):
        if not SKOPT_AVAILABLE or Optimizer is None:
            raise ImportError("scikit-optimize is required for LGBMHierarchicalBayesSearchCV.")
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
        best_acc_so_far = np.nan
        total = max(1, int(self.n_iter))
        v_lvl = int(self.verbose or 0)

        for it in range(total):
            x_pt = init_pt if it == 0 else opt.ask()
            decoded = _decode_skopt_params(dict(zip(keys, x_pt)))
            sig = _canonical_lgbm_candidate_key(decoded, base_params=base_params)
            if v_lvl >= 2:
                print(f"\n[Bayes {it + 1}/{total} START] {_lgbm_candidate_brief(decoded)}", flush=True)
            if v_lvl >= 3:
                inactive = inactive_lgbm_parameters_for_candidate({**base_params, **decoded})
                active_d = {k: v for k, v in sorted(decoded.items()) if k not in inactive}
                print(f"  [Active params ({len(active_d)})]: {active_d}", flush=True)
            if sig in cache:
                fold_qwks, fold_accs, elapsed = cache[sig]
                cached_tag = " (cached)"
            else:
                err = lgbm_candidate_error(decoded, base_params=base_params)
                if err is not None:
                    if v_lvl >= 1:
                        print(f"  [Bayes {it + 1}/{total} INVALID] {err}", flush=True)
                    n_sp = cv.get_n_splits(X, y, groups) if hasattr(cv, "get_n_splits") else 1
                    err_val = -1.0 if self.error_score == "raise" else float(self.error_score)
                    fold_qwks, fold_accs, elapsed = [err_val] * n_sp, [-1.0] * n_sp, 0.0
                else:
                    fold_qwks, fold_accs, elapsed = _evaluate_lgbm_candidate_cv(
                        self.estimator,
                        decoded,
                        X,
                        y,
                        groups,
                        cv,
                        scorer,
                        error_score=-1.0 if self.error_score == "raise" else float(self.error_score),
                        verbose=v_lvl,
                    )
                cache[sig] = (fold_qwks, fold_accs, elapsed)
                cached_tag = ""

            ok_idx, mean_acc, mean_qwk, best_so_far, best_acc_so_far, acc_lbl, best_acc_lbl, fold_str = (
                _accumulate_candidate_scores(fold_accs, fold_qwks, best_so_far, best_acc_so_far)
            )
            opt.tell(x_pt, -mean_qwk)
            all_params.append(decoded)
            all_fold_qwk.append(fold_qwks)
            all_fold_acc_pct.append(fold_accs)
            all_times.append(elapsed)

            if v_lvl >= 1:
                tm = decoded.get("estimator__task_mode", decoded.get("task_mode", "?"))
                bt = decoded.get("estimator__boosting_type", decoded.get("boosting_type", "?"))
                nt = decoded.get("estimator__n_estimators", decoded.get("n_estimators", "?"))
                lr = decoded.get("estimator__learning_rate", decoded.get("learning_rate", "?"))
                bm = decoded.get("estimator__extractor__bin_minutes", decoded.get("extractor__bin_minutes", "?"))
                print(
                    f"[Bayes {it + 1}/{total}] task={tm} bt={bt} trees={nt} lr={lr} bin={bm}m | "
                    f"Normal={acc_lbl} -> Comp QWK={mean_qwk:.5f} | "
                    f"best: Normal={best_acc_lbl} -> QWK={best_so_far:.5f} ({elapsed:.2f}s){cached_tag}",
                    flush=True,
                )
                if ok_idx and (v_lvl >= 2 or len(fold_qwks) > 1):
                    print(f"  Successful folds ({len(ok_idx)}/{len(fold_qwks)}): {fold_str}", flush=True)

        self.cv_results_ = _assemble_search_cv_results(all_params, all_fold_qwk, all_fold_acc_pct, all_times)
        means = self.cv_results_["mean_test_qwk"]
        self.best_index_ = int(refit_valid_candidate(self.cv_results_)) if callable(self.refit) else int(np.argmax(means))
        self.best_params_ = dict(all_params[self.best_index_])
        self.best_score_ = float(means[self.best_index_])
        if self.refit:
            if v_lvl >= 1:
                best_cv_acc = float(self.cv_results_["mean_test_accuracy_pct"][self.best_index_])
                print(
                    f"\n[Bayes REFIT] Refitting best candidate #{self.best_index_ + 1} "
                    f"(CV Normal={_fmt_pct(best_cv_acc)} -> Comp QWK={self.best_score_:.5f}): "
                    f"{_lgbm_candidate_brief(self.best_params_)}",
                    flush=True,
                )
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            _propagate_verbose(self.best_estimator_, v_lvl)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.best_estimator_.predict(X)


class LGBMGASearchCV(BaseEstimator):
    """Evolutionary (Genetic Algorithm) search over ``SequenceExtractor``, ``TSAugmentor`` & LightGBM."""

    def __init__(
        self,
        estimator: Any,
        param_distributions: Mapping[str, Any],
        population_size: int = 8,
        generations: int = 4,
        tournament_size: int = 3,
        elitism: int = 2,
        crossover_probability: float = 0.7,
        mutation_probability: float = 0.25,
        cv: Any = None,
        scoring: Any = None,
        refit: Any = True,
        random_state: int = 42,
        verbose: int = 1,
        error_score: Any = -1.0,
        n_jobs: int = 1,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.random_state = random_state
        self.verbose = verbose
        self.error_score = error_score
        self.n_jobs = n_jobs

    @staticmethod
    def _default_seed_candidate(space_dict: Mapping[str, Any], base_params: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        cand: Dict[str, Any] = {}
        for k, dim in space_dict.items():
            bv = 0.0 if "augmentor__prob" in k else base_params.get(k)
            if isinstance(dim, ECat):
                cand[k] = bv if bv in dim.categories else dim.categories[0]
            elif isinstance(dim, EInt):
                if isinstance(bv, (int, float, np.number)) and dim.low <= int(round(float(bv))) <= dim.high:
                    cand[k] = int(round(float(bv)))
                else:
                    cand[k] = dim.sample(rng)
            elif isinstance(dim, EFloat):
                if isinstance(bv, (int, float, np.number)) and dim.low <= float(bv) <= dim.high:
                    cand[k] = float(bv)
                else:
                    cand[k] = dim.sample(rng)
            else:
                cand[k] = dim.sample(rng)
        return cand

    def fit(self, X: pd.DataFrame, y: Any, groups: Optional[Any] = None):
        rng = np.random.default_rng(int(self.random_state))
        cv = self.cv or StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(self.random_state))
        space_dict = prepare_evolutionary_space(self.param_distributions)
        keys = list(space_dict.keys())
        base_params = self.estimator.get_params(deep=True)
        scorer = self.scoring or candidate_score

        pop_size = max(2, int(self.population_size))
        n_gen = max(1, int(self.generations))
        t_size = max(1, min(int(self.tournament_size), pop_size))
        n_elite = max(0, min(int(self.elitism), pop_size - 1))
        v_lvl = int(self.verbose or 0)

        cache: Dict[str, Tuple[List[float], List[float], float]] = {}
        all_params: List[Dict[str, Any]] = []
        all_fold_qwk: List[List[float]] = []
        all_fold_acc_pct: List[List[float]] = []
        all_times: List[float] = []
        best_so_far = -np.inf
        best_acc_so_far = np.nan

        population: List[Dict[str, Any]] = [self._default_seed_candidate(space_dict, base_params, rng)]
        while len(population) < pop_size:
            population.append({k: space_dict[k].sample(rng) for k in keys})

        for gen in range(n_gen):
            gen_scores: List[float] = []
            gen_accs: List[float] = []
            for c_idx, raw_cand in enumerate(population):
                decoded = _decode_skopt_params(raw_cand)
                sig = _canonical_lgbm_candidate_key(decoded, base_params=base_params)
                if v_lvl >= 2:
                    print(f"\n[GA gen {gen + 1}/{n_gen} | cand {c_idx + 1}/{pop_size} START] {_lgbm_candidate_brief(decoded)}", flush=True)
                if v_lvl >= 3:
                    inactive = inactive_lgbm_parameters_for_candidate({**base_params, **decoded})
                    active_d = {k: v for k, v in sorted(decoded.items()) if k not in inactive}
                    print(f"  [Active params ({len(active_d)})]: {active_d}", flush=True)
                if sig in cache:
                    fold_qwks, fold_accs, elapsed = cache[sig]
                    cached_tag = " (cached)"
                else:
                    err = lgbm_candidate_error(decoded, base_params=base_params)
                    if err is not None:
                        if v_lvl >= 1:
                            print(f"  [GA gen {gen + 1}/{n_gen} | cand {c_idx + 1}/{pop_size} INVALID] {err}", flush=True)
                        n_sp = cv.get_n_splits(X, y, groups) if hasattr(cv, "get_n_splits") else 1
                        err_val = -1.0 if self.error_score == "raise" else float(self.error_score)
                        fold_qwks, fold_accs, elapsed = [err_val] * n_sp, [-1.0] * n_sp, 0.0
                    else:
                        fold_qwks, fold_accs, elapsed = _evaluate_lgbm_candidate_cv(
                            self.estimator,
                            decoded,
                            X,
                            y,
                            groups,
                            cv,
                            scorer,
                            error_score=-1.0 if self.error_score == "raise" else float(self.error_score),
                            verbose=v_lvl,
                        )
                    cache[sig] = (fold_qwks, fold_accs, elapsed)
                    cached_tag = ""

                ok_idx, mean_acc, mean_qwk, best_so_far, best_acc_so_far, acc_lbl, best_acc_lbl, fold_str = (
                    _accumulate_candidate_scores(fold_accs, fold_qwks, best_so_far, best_acc_so_far)
                )
                gen_scores.append(mean_qwk)
                gen_accs.append(mean_acc)
                all_params.append(decoded)
                all_fold_qwk.append(fold_qwks)
                all_fold_acc_pct.append(fold_accs)
                all_times.append(elapsed)

                if v_lvl >= 1:
                    tm = decoded.get("estimator__task_mode", decoded.get("task_mode", "?"))
                    bt = decoded.get("estimator__boosting_type", decoded.get("boosting_type", "?"))
                    nt = decoded.get("estimator__n_estimators", decoded.get("n_estimators", "?"))
                    lr = decoded.get("estimator__learning_rate", decoded.get("learning_rate", "?"))
                    bm = decoded.get("estimator__extractor__bin_minutes", decoded.get("extractor__bin_minutes", "?"))
                    print(
                        f"[GA gen {gen + 1}/{n_gen} | cand {c_idx + 1}/{pop_size}] task={tm} bt={bt} trees={nt} lr={lr} bin={bm}m | "
                        f"Normal={acc_lbl} -> Comp QWK={mean_qwk:.5f} | "
                        f"best: Normal={best_acc_lbl} -> QWK={best_so_far:.5f} ({elapsed:.2f}s){cached_tag}",
                        flush=True,
                    )
                    if ok_idx and (v_lvl >= 2 or len(fold_qwks) > 1):
                        print(f"  Successful folds ({len(ok_idx)}/{len(fold_qwks)}): {fold_str}", flush=True)

            if v_lvl >= 1 and gen_scores:
                print(
                    _ga_gen_summary_line(gen, n_gen, gen_scores, gen_accs, best_so_far, best_acc_so_far),
                    flush=True,
                )

            if gen + 1 >= n_gen:
                break

            order = np.argsort(-np.asarray(gen_scores, dtype=float), kind="stable")
            next_pop: List[Dict[str, Any]] = [dict(population[int(i)]) for i in order[:n_elite]]

            def _tournament_pick() -> Dict[str, Any]:
                competitors = rng.choice(pop_size, size=t_size, replace=False)
                winner = int(competitors[int(np.argmax([gen_scores[int(i)] for i in competitors]))])
                return dict(population[winner])

            while len(next_pop) < pop_size:
                p1 = _tournament_pick()
                p2 = _tournament_pick()
                child = dict(p1)
                if rng.random() < float(self.crossover_probability):
                    for k in keys:
                        if rng.random() < 0.5:
                            child[k] = p2[k]
                for k in keys:
                    if rng.random() < float(self.mutation_probability):
                        child[k] = space_dict[k].mutate(child[k], rng)
                next_pop.append(child)
            population = next_pop

        self.cv_results_ = _assemble_search_cv_results(all_params, all_fold_qwk, all_fold_acc_pct, all_times)
        means = self.cv_results_["mean_test_qwk"]
        self.best_index_ = int(refit_valid_candidate(self.cv_results_)) if callable(self.refit) else int(np.argmax(means))
        self.best_params_ = dict(all_params[self.best_index_])
        self.best_score_ = float(means[self.best_index_])
        if self.refit:
            if v_lvl >= 1:
                best_cv_acc = float(self.cv_results_["mean_test_accuracy_pct"][self.best_index_])
                print(
                    f"\n[GA REFIT] Refitting best candidate #{self.best_index_ + 1} "
                    f"(CV Normal={_fmt_pct(best_cv_acc)} -> Comp QWK={self.best_score_:.5f}): "
                    f"{_lgbm_candidate_brief(self.best_params_)}",
                    flush=True,
                )
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            _propagate_verbose(self.best_estimator_, v_lvl)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.best_estimator_.predict(X)


# ---------------------------------------------------------------------------
# Trained-Model Permutation Importance (LightGBM)
# ---------------------------------------------------------------------------

def compute_lgbm_permutation_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: Any,
    *,
    repeats: int = 3,
    random_state: int = 42,
    top_k_individual: int = 25,
    include_noop_individual: bool = True,
    require_internal_validation: bool = False,
    output_dir: Optional[Any] = None,
    prefix: str = "permutation_importance",
) -> pd.DataFrame:
    """Compute trained-model permutation importance for ``LightGBMSequenceEstimator``.

    Evaluates on the provided holdout split ``(X, y)`` by default (or internal
    ``validation_sequence_ids_`` when ``require_internal_validation=True``), permuting
    semantic feature families, top-K LightGBM gain features, individual ``noop_*`` features,
    and selected static features.
    """
    est = estimator.named_steps["estimator"] if hasattr(estimator, "named_steps") and "estimator" in estimator.named_steps else estimator
    ext = getattr(est, "extractor_", None) or est._get_extractor()

    y_ser = y[TARGET_COL] if isinstance(y, pd.DataFrame) and TARGET_COL in y.columns else pd.Series(np.asarray(y).ravel(), index=X.index)
    val_ids = getattr(est, "validation_sequence_ids_", np.asarray([]))
    scope_name = f"holdout_split (n={len(X)})"
    if bool(require_internal_validation) and len(val_ids) > 0 and ID_COL in X.columns:
        val_mask = X[ID_COL].astype(str).isin(set(map(str, val_ids))).to_numpy()
        if int(val_mask.sum()) >= 5:
            X_eval = X.loc[val_mask].reset_index(drop=True)
            y_eval = np.asarray(y_ser.loc[val_mask], dtype=np.int64).ravel()
            scope_name = f"internal_validation (n={len(X_eval)})"
        else:
            X_eval = X.reset_index(drop=True)
            y_eval = np.asarray(y_ser, dtype=np.int64).ravel()
    else:
        X_eval = X.reset_index(drop=True)
        y_eval = np.asarray(y_ser, dtype=np.int64).ravel()

    feat_df = ext.transform(strip_leaky_columns(X_eval))
    base_preds = est.predict_from_features(feat_df)
    base_qwk = float(competition_score(y_eval, base_preds))
    base_f1 = float(f1_score(y_eval, base_preds, average="macro", zero_division=0))

    active_cols = [c for c in getattr(est, "feature_names_out_", list(feat_df.columns)) if c in feat_df.columns]
    family_map: Dict[str, List[str]] = {}
    for c in active_cols:
        fam = _feature_family_label(c)
        family_map.setdefault(fam, []).append(c)

    groups: List[Tuple[str, str, List[str]]] = []
    family_order = (
        "clinical_tabular",
        "missing_indicators",
        "autocorr_circadian",
        "lag_lead_dynamics",
        "sleep_architecture",
        "stft_wavelet_spectral",
        "seglearn_kinematic",
        "static_open_source",
        "actigraphy_compliance",
        "noop_all",
    )
    for fam in family_order:
        cols = family_map.get(fam, [])
        if cols:
            groups.append(("family", fam, cols))

    seen_indiv: set = set()
    lgbm_imp_df = est.get_feature_importances_df()
    for feat_name in lgbm_imp_df["feature"].head(max(0, int(top_k_individual))).tolist():
        if feat_name in feat_df.columns and feat_df[feat_name].notna().sum() >= 2:
            groups.append(("lgbm_feature", feat_name, [feat_name]))
            seen_indiv.add(feat_name)

    if bool(include_noop_individual):
        for nc in family_map.get("noop_all", []):
            if nc not in seen_indiv and feat_df[nc].notna().sum() >= 2:
                groups.append(("noop_feature", nc, [nc]))
                seen_indiv.add(nc)
        for sc in getattr(est, "selected_static_cols_", [])[:15]:
            if sc not in seen_indiv and sc in feat_df.columns and feat_df[sc].notna().sum() >= 2:
                groups.append(("selected_static", sc, [sc]))
                seen_indiv.add(sc)

    rng = np.random.default_rng(int(random_state))
    n_rep = max(1, int(repeats))
    rows: List[Dict[str, Any]] = []
    permuted_col_union: set = set()

    for kind, label, cols in groups:
        qwk_drops: List[float] = []
        f1_drops: List[float] = []
        perm_qwks: List[float] = []
        permuted_col_union.update(cols)

        for _ in range(n_rep):
            perm_idx = rng.permutation(len(X_eval))
            f_mod = feat_df.copy()
            for c in cols:
                f_mod[c] = f_mod[c].to_numpy()[perm_idx]
            p_preds = est.predict_from_features(f_mod)
            p_qwk = float(competition_score(y_eval, p_preds))
            p_f1 = float(f1_score(y_eval, p_preds, average="macro", zero_division=0))
            perm_qwks.append(p_qwk)
            qwk_drops.append(base_qwk - p_qwk)
            f1_drops.append(base_f1 - p_f1)

        rows.append({
            "kind": kind,
            "feature_or_group": label,
            "n_columns": len(cols),
            "importance_mean": float(np.mean(qwk_drops)),
            "importance_std": float(np.std(qwk_drops)),
            "qwk_drop_mean": float(np.mean(qwk_drops)),
            "qwk_drop_std": float(np.std(qwk_drops)),
            "macro_f1_drop_mean": float(np.mean(f1_drops)),
            "permuted_qwk_mean": float(np.mean(perm_qwks)),
            "baseline_qwk": base_qwk,
        })

    imp_df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False, kind="stable").reset_index(drop=True)
    saved_paths: Dict[str, str] = {}
    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        csv_p = out_p / f"{prefix}.csv"
        json_p = out_p / f"{prefix}.json"
        png_p = out_p / f"{prefix}.png"
        imp_df.to_csv(csv_p, index=False)
        json_p.write_text(
            json.dumps(
                {
                    "scope": scope_name,
                    "baseline_qwk": base_qwk,
                    "baseline_macro_f1": base_f1,
                    "repeats": n_rep,
                    "top_importances": imp_df.head(25).to_dict(orient="records"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        saved_paths["csv"] = str(csv_p)
        saved_paths["json"] = str(json_p)
        try:
            import matplotlib.pyplot as plt

            top_plot = imp_df.head(18).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.32 * len(top_plot))))
            ax.barh(top_plot["feature_or_group"], top_plot["importance_mean"], xerr=top_plot["importance_std"], color="#2563eb", alpha=0.85)
            ax.axvline(0.0, color="#64748b", linestyle="--", linewidth=1.0)
            ax.set_xlabel("Permutation Importance (Competition QWK drop)")
            ax.set_title(f"LightGBM Permutation Importance ({scope_name})")
            fig.tight_layout()
            fig.savefig(png_p, dpi=140)
            plt.close(fig)
            saved_paths["png"] = str(png_p)
        except Exception:
            pass

    imp_df.attrs["scope"] = scope_name
    imp_df.attrs["baseline_qwk"] = base_qwk
    imp_df.attrs["baseline_macro_f1"] = base_f1
    imp_df.attrs["permutable_fraction"] = float(len(permuted_col_union) / max(len(active_cols), 1))
    imp_df.attrs["paths"] = saved_paths
    return imp_df
