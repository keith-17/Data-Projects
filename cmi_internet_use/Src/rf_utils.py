"""rf_utils.py

Random Forest Estimator, RF-aware Hyperparameter Searchers (Bayesian / Evolutionary / Grid),
and Trained-Model Permutation Importance for the Kaggle Child Mind Institute — Problematic
Internet Use (CMI PIU) competition.

Designed to drop seamlessly into the ``baseline.ipynb`` / ``base_utils.py`` pipeline:
* Reuses ``SequenceExtractor``, ``TSAugmentor``, ``NoopActigraphyOptions``, ``run_validation_searches``,
  ``evaluate_holdout``, ``export_submission``, and Quadratic Weighted Kappa (QWK) scoring from
  ``base_utils.py`` and ``baseline_utils.py``.
* Supports both ``RandomForestRegressor`` (directly aligned with QWK quadratic loss) and
  ``RandomForestClassifier`` (expected-value probability projection or argmax) with leak-free
  Out-Of-Bag (OOB) / Out-Of-Fold (OOF) ordinal threshold calibration.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
        _evaluate_candidate_cv,
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
        _evaluate_candidate_cv,
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


DEFAULT_RF_ESTIMATOR_PARAMS: Dict[str, Any] = {
    "task_mode": "regressor",
    "n_estimators": 300,
    "reg_criterion": "squared_error",
    "clf_criterion": "gini",
    "clf_prediction_mode": "expected_value",
    "max_depth": 12,
    "min_samples_split": 4,
    "min_samples_leaf": 3,
    "min_weight_fraction_leaf": 0.0,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "max_samples": None,
    "ccp_alpha": 0.0,
    "class_weight": None,
    "sample_weight_mode": "none",
    "nan_handling": "median",
    "drop_all_nan_cols": True,
    "threshold_calibration": "optimize_qwk",
    "threshold_calibration_source": "oob_or_oof",
}


# ---------------------------------------------------------------------------
# Coercion, Validation & Semantic Feature-Family Helpers
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


def _coerce_max_features(val: Any) -> Any:
    v = _structured(val)
    if _is_none_like(v):
        return None
    if isinstance(v, str):
        low = v.strip().lower()
        if low in {"sqrt", "log2"}:
            return low
        if low == "all":
            return None
        try:
            fv = float(low)
        except ValueError as exc:
            raise InvalidParams(f"Unsupported max_features: {val!r}") from exc
    elif isinstance(v, (int, np.integer)):
        iv = int(v)
        return None if iv <= 0 else iv
    elif isinstance(v, (float, np.floating)):
        fv = float(v)
    else:
        raise InvalidParams(f"Unsupported max_features: {val!r}")

    if not np.isfinite(fv) or fv <= 0.0:
        raise InvalidParams(f"max_features must be > 0, got {val!r}")
    if fv >= 1.0:
        return 1.0 if fv == 1.0 else int(round(fv))
    return float(fv)


def _coerce_min_samples(val: Any, *, is_leaf: bool) -> Any:
    v = _structured(val)
    if isinstance(v, str):
        v = float(v.strip())
    if isinstance(v, (int, np.integer)):
        iv = int(v)
        return max(1 if is_leaf else 2, iv)
    fv = float(v)
    if not np.isfinite(fv) or fv <= 0.0:
        return 1 if is_leaf else 2
    if fv < 1.0:
        upper = 0.5 if is_leaf else 1.0
        return float(min(fv, upper))
    return max(1 if is_leaf else 2, int(round(fv)))


def _coerce_max_samples(val: Any, *, bootstrap: bool) -> Optional[float]:
    if not bootstrap:
        return None
    fv = _coerce_opt_float(val)
    if fv is None or fv >= 1.0:
        return None
    if fv <= 0.0:
        raise InvalidParams(f"max_samples must be in (0, 1], got {val!r}")
    return float(fv)


def _coerce_class_weight(val: Any) -> Optional[str]:
    v = _structured(val)
    if _is_none_like(v):
        return None
    s = str(v).strip().lower()
    if s not in {"balanced", "balanced_subsample"}:
        raise InvalidParams(f"Unsupported class_weight: {val!r}")
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


def validate_rf_params(params: Mapping[str, Any]) -> None:
    """Validate Random Forest hyperparameters prior to CV fitting."""
    task_mode = str(_structured(params.get("task_mode", "regressor"))).strip().lower()
    if task_mode not in {"regressor", "classifier"}:
        raise InvalidParams(f"task_mode must be 'regressor' or 'classifier', got {task_mode!r}")

    n_est = _coerce_opt_int(params.get("n_estimators", 300), min_val=1)
    if n_est is None or n_est < 1:
        raise InvalidParams(f"n_estimators must be >= 1, got {params.get('n_estimators')!r}")

    reg_crit = str(_structured(params.get("reg_criterion", "squared_error"))).strip().lower()
    if reg_crit not in {"squared_error", "friedman_mse", "absolute_error", "poisson"}:
        raise InvalidParams(f"Unsupported reg_criterion: {reg_crit!r}")

    clf_crit = str(_structured(params.get("clf_criterion", "gini"))).strip().lower()
    if clf_crit not in {"gini", "entropy", "log_loss"}:
        raise InvalidParams(f"Unsupported clf_criterion: {clf_crit!r}")

    clf_pred_mode = str(_structured(params.get("clf_prediction_mode", "expected_value"))).strip().lower()
    if clf_pred_mode not in {"expected_value", "argmax"}:
        raise InvalidParams(f"Unsupported clf_prediction_mode: {clf_pred_mode!r}")

    nan_h = str(_structured(params.get("nan_handling", "median"))).strip().lower()
    if nan_h not in {"median", "sentinel", "zero", "native"}:
        raise InvalidParams(f"Unsupported nan_handling: {nan_h!r}")

    sw_mode = str(_structured(params.get("sample_weight_mode", "none"))).strip().lower()
    if sw_mode not in {"none", "balanced", "sqrt_balanced"}:
        raise InvalidParams(f"Unsupported sample_weight_mode: {sw_mode!r}")

    tc = _structured(params.get("threshold_calibration", "optimize_qwk"))
    tc_norm = "none" if _is_none_like(tc) else str(tc).strip().lower()
    if tc_norm not in {"optimize_qwk", "quantile", "fixed_midpoint", "round", "none"}:
        raise InvalidParams(f"Unsupported threshold_calibration: {tc!r}")

    tc_src = str(_structured(params.get("threshold_calibration_source", "oob_or_oof"))).strip().lower()
    if tc_src not in {"oob_or_oof", "train"}:
        raise InvalidParams(f"Unsupported threshold_calibration_source: {tc_src!r}")

    if float(_structured(params.get("ccp_alpha", 0.0))) < 0.0:
        raise InvalidParams("ccp_alpha must be >= 0.0")
    if float(_structured(params.get("min_impurity_decrease", 0.0))) < 0.0:
        raise InvalidParams("min_impurity_decrease must be >= 0.0")
    mwfl = float(_structured(params.get("min_weight_fraction_leaf", 0.0)))
    if not (0.0 <= mwfl <= 0.5):
        raise InvalidParams(f"min_weight_fraction_leaf must be in [0.0, 0.5], got {mwfl}")

    _coerce_max_features(params.get("max_features", "sqrt"))
    _coerce_class_weight(params.get("class_weight", None))
    bs = _coerce_bool(params.get("bootstrap", True), default=True)
    _coerce_max_samples(params.get("max_samples", None), bootstrap=bs)


# ---------------------------------------------------------------------------
# Random Forest Sequence Estimator
# ---------------------------------------------------------------------------

class RandomForestSequenceEstimator(BaseEstimator):
    """End-to-end Random Forest estimator wrapping ``SequenceExtractor`` & ``TSAugmentor``.

    Key design features for CMI PIU (ordinal ``sii`` in ``{0, 1, 2, 3}``, metric: QWK):
    * **Leak-free featurization & imputation**: ``SequenceExtractor`` + RF fallback NaN sanitizer
      are strictly fit on training-fold subjects only.
    * **Dual task modes**: ``task_mode='regressor'`` (MSE/Friedman-MSE trees aligned with QWK's
      quadratic penalty) or ``task_mode='classifier'`` (expected-value probability projection or
      argmax).
    * **Out-of-sample ordinal threshold calibration**: Calibrates QWK thresholds ``(t0, t1, t2)``
      using native Random Forest Out-Of-Bag (OOB) predictions (or internal 3-fold OOF fallback
      when ``bootstrap=False`` or augmentation is enabled) so threshold cutpoints never overfit
      in-sample tree memorization.
    * **Actigraphy-aware augmentation**: When ``TSAugmentor`` is enabled, applies training-only
      actigraphy modality dropout and signal-feature perturbations.
    """

    def __init__(
        self,
        extractor: Optional[SequenceExtractor] = None,
        augmentor: Optional[TSAugmentor] = None,
        task_mode: str = "regressor",
        n_estimators: int = 300,
        reg_criterion: str = "squared_error",
        clf_criterion: str = "gini",
        clf_prediction_mode: str = "expected_value",
        max_depth: Optional[int] = 12,
        min_samples_split: Any = 4,
        min_samples_leaf: Any = 3,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Any = "sqrt",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        max_samples: Optional[float] = None,
        ccp_alpha: float = 0.0,
        class_weight: Optional[str] = None,
        sample_weight_mode: str = "none",
        nan_handling: str = "median",
        drop_all_nan_cols: bool = True,
        threshold_calibration: str = "optimize_qwk",
        threshold_calibration_source: str = "oob_or_oof",
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
        rf_n_jobs: int = -1,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.extractor = extractor
        self.augmentor = augmentor
        self.task_mode = task_mode
        self.n_estimators = n_estimators
        self.reg_criterion = reg_criterion
        self.clf_criterion = clf_criterion
        self.clf_prediction_mode = clf_prediction_mode
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.ccp_alpha = ccp_alpha
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
        self.rf_n_jobs = rf_n_jobs
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

    def _build_rf_model(self, *, use_oob: bool, seed_offset: int = 0):
        task_mode = str(self.task_mode).strip().lower()
        n_est = max(1, int(round(float(_structured(self.n_estimators)))))
        max_d = _coerce_opt_int(self.max_depth, min_val=1)
        min_split = _coerce_min_samples(self.min_samples_split, is_leaf=False)
        min_leaf = _coerce_min_samples(self.min_samples_leaf, is_leaf=True)
        min_wfl = float(_structured(self.min_weight_fraction_leaf))
        max_feat = _coerce_max_features(self.max_features)
        max_ln = _coerce_opt_int(self.max_leaf_nodes, min_val=2)
        min_imp = float(_structured(self.min_impurity_decrease))
        bs = _coerce_bool(self.bootstrap, default=True)
        max_samp = _coerce_max_samples(self.max_samples, bootstrap=bs)
        ccp = float(_structured(self.ccp_alpha))
        n_jobs = int(_structured(self.rf_n_jobs))
        rs = int(_structured(self.random_state)) + int(seed_offset)
        oob = bool(use_oob and bs and n_est >= 10)
        rf_verbose = max(0, int(self.verbose or 0) - 2) if seed_offset == 0 else 0

        if task_mode == "regressor":
            crit = str(_structured(self.reg_criterion)).strip().lower()
            return RandomForestRegressor(
                n_estimators=n_est,
                criterion=crit,
                max_depth=max_d,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf,
                min_weight_fraction_leaf=min_wfl,
                max_features=max_feat,
                max_leaf_nodes=max_ln,
                min_impurity_decrease=min_imp,
                bootstrap=bs,
                oob_score=oob,
                max_samples=max_samp,
                ccp_alpha=ccp,
                n_jobs=n_jobs,
                verbose=rf_verbose,
                random_state=rs,
            )

        crit = str(_structured(self.clf_criterion)).strip().lower()
        cw = _coerce_class_weight(self.class_weight)
        return RandomForestClassifier(
            n_estimators=n_est,
            criterion=crit,
            max_depth=max_d,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            min_weight_fraction_leaf=min_wfl,
            max_features=max_feat,
            max_leaf_nodes=max_ln,
            min_impurity_decrease=min_imp,
            bootstrap=bs,
            oob_score=oob,
            max_samples=max_samp,
            ccp_alpha=ccp,
            class_weight=cw,
            n_jobs=n_jobs,
            verbose=rf_verbose,
            random_state=rs,
        )

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

        req_nan = str(_structured(self.nan_handling)).strip().lower()
        task_mode = str(self.task_mode).strip().lower()
        if req_nan == "native" and task_mode == "regressor" and str(_structured(self.reg_criterion)).strip().lower() == "absolute_error":
            # scikit-learn RF absolute_error does not support native NaN splits
            req_nan = "median"
        self.effective_nan_handling_ = req_nan

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

        # Precompute column scales on actigraphy-present training subjects only
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
        if str(self.task_mode).strip().lower() != "regressor":
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
        if str(self.task_mode).strip().lower() == "regressor":
            return np.asarray(model.predict(X_mat), dtype=np.float64).ravel()
        proba = np.asarray(model.predict_proba(X_mat), dtype=np.float64)
        cls_vals = np.asarray(model.classes_, dtype=np.float64).ravel()
        return proba @ cls_vals

    def _fit_single_rf(self, model: Any, X_mat: np.ndarray, y_arr: np.ndarray, sample_weight: Optional[np.ndarray]) -> Any:
        try:
            model.fit(X_mat, y_arr, sample_weight=sample_weight)
            return model
        except ValueError as exc:
            # Automatic graceful fallback if sklearn RF does not support native NaNs for this version/criterion
            msg = str(exc).lower()
            if self.effective_nan_handling_ == "native" and ("nan" in msg or "missing" in msg):
                self.effective_nan_handling_ = "median"
                X_clean = X_mat.copy()
                nan_m = np.isnan(X_clean)
                if nan_m.any():
                    r_i, c_i = np.where(nan_m)
                    X_clean[r_i, c_i] = self.col_medians_[c_i].astype(np.float32)
                model.fit(X_clean, y_arr, sample_weight=sample_weight)
                return model
            raise

    def _compute_calibration_scores(
        self,
        X_orig_mat: np.ndarray,
        y_orig: np.ndarray,
        fitted_model: Any,
        *,
        aug_applied: bool,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Produce honest out-of-sample (OOB or 3-fold OOF) continuous scores for threshold calibration."""
        src = str(_structured(self.threshold_calibration_source)).strip().lower()
        if src == "train" or len(y_orig) < 10:
            return self._continuous_from_model(fitted_model, X_orig_mat), y_orig, "train"

        # 1. Zero-cost Out-Of-Bag calibration when bootstrap=True and no augmented rows were concatenated
        if _coerce_bool(self.bootstrap, default=True) and not aug_applied:
            oob_scores: Optional[np.ndarray] = None
            if str(self.task_mode).strip().lower() == "regressor" and hasattr(fitted_model, "oob_prediction_"):
                oob_scores = np.asarray(fitted_model.oob_prediction_, dtype=np.float64).ravel()
            elif str(self.task_mode).strip().lower() == "classifier" and hasattr(fitted_model, "oob_decision_function_"):
                oob_df = np.asarray(fitted_model.oob_decision_function_, dtype=np.float64)
                cls_vals = np.asarray(fitted_model.classes_, dtype=np.float64).ravel()
                if oob_df.ndim == 2 and oob_df.shape[1] == len(cls_vals):
                    row_ok = np.all(np.isfinite(oob_df), axis=1) & (np.sum(oob_df, axis=1) > 0)
                    oob_scores = np.full(len(y_orig), np.nan, dtype=np.float64)
                    oob_scores[row_ok] = oob_df[row_ok] @ cls_vals

            if oob_scores is not None and len(oob_scores) == len(y_orig):
                fin = np.isfinite(oob_scores)
                if int(fin.sum()) >= max(8, int(0.8 * len(y_orig))):
                    return oob_scores[fin], y_orig[fin], "oob"

        # 2. Leak-free 3-fold OOF fallback (when bootstrap=False, augmented, or small n_estimators)
        _, counts = np.unique(y_orig, return_counts=True)
        min_cls = int(counts.min()) if len(counts) else 0
        if min_cls >= 3:
            splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(_structured(self.random_state)))
        elif min_cls == 2:
            splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=int(_structured(self.random_state)))
        else:
            splitter = KFold(n_splits=min(3, max(2, len(y_orig) // 3)), shuffle=True, random_state=int(_structured(self.random_state)))

        oof = np.full(len(y_orig), np.nan, dtype=np.float64)
        for fold_idx, (tr_i, va_i) in enumerate(splitter.split(X_orig_mat, y_orig)):
            fold_rf = self._build_rf_model(use_oob=False, seed_offset=100 + fold_idx)
            sw_tr = self._compute_sample_weights(y_orig[tr_i])
            self._fit_single_rf(fold_rf, X_orig_mat[tr_i], y_orig[tr_i], sw_tr)
            oof[va_i] = self._continuous_from_model(fold_rf, X_orig_mat[va_i])

        fin = np.isfinite(oof)
        if int(fin.sum()) >= 8:
            return oof[fin], y_orig[fin], "oof_3fold"
        return self._continuous_from_model(fitted_model, X_orig_mat), y_orig, "train_fallback"

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
        validate_rf_params(self.get_params(deep=False))

        y_arr = np.asarray(
            y[self.primary_target] if isinstance(y, pd.DataFrame) and self.primary_target in y.columns else y,
            dtype=np.int64,
        ).ravel()

        # Track split IDs (matching HandcraftedSequenceBaselineEstimator API)
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
                f"  [RF fit] Extracting features: n={len(X)} (act={n_act}) | bin={ext.bin_minutes}m | "
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

        # Enable native OOB scoring when bootstrap=True and no augmented duplicates exist
        use_oob = _coerce_bool(self.bootstrap, default=True) and not aug_applied
        rf_model = self._build_rf_model(use_oob=use_oob)

        if v_lvl >= 2:
            cls_name = type(rf_model).__name__
            extra_lbl = (
                f"sw={self.sample_weight_mode}"
                if str(self.task_mode).strip().lower() == "regressor"
                else f"cw={rf_model.class_weight} pred={self.clf_prediction_mode}"
            )
            print(
                f"  [RF fit] Features ready in {t_ext:.2f}s: retained={self.n_samples_retained_}/{len(X)} "
                f"(outliers_dropped={self.n_samples_removed_}, aug=+{self.n_augmented_rows_}) | "
                f"features={len(self.feature_names_out_)} ({raw_nan_pct:.1f}% NaN -> {self.effective_nan_handling_})",
                flush=True,
            )
            print(
                f"  [RF fit] Training {cls_name}(n_estimators={rf_model.n_estimators}, criterion={rf_model.criterion}, "
                f"max_depth={rf_model.max_depth}, min_leaf={rf_model.min_samples_leaf}, max_feat={rf_model.max_features}, "
                f"bootstrap={rf_model.bootstrap}, ccp={rf_model.ccp_alpha}, {extra_lbl}) on {X_train_mat.shape}...",
                flush=True,
            )

        t_rf0 = time.perf_counter()
        rf_model = self._fit_single_rf(rf_model, X_train_mat, y_fit_train, sw_train)
        t_rf = time.perf_counter() - t_rf0
        self.model_ = rf_model
        # Re-align X_orig_mat if native NaN handling fell back to median
        if self.effective_nan_handling_ != str(_structured(self.nan_handling)).strip().lower():
            X_orig_mat = self._transform_feature_matrix(feat_fit_orig)

        self.oob_score_ = float(getattr(rf_model, "oob_score_", np.nan)) if use_oob else np.nan
        self.feature_importances_ = np.asarray(getattr(rf_model, "feature_importances_", np.zeros(len(self.feature_names_out_))), dtype=np.float64)

        # Calibrate ordinal thresholds using out-of-sample (OOB / OOF) continuous predictions
        t_cal0 = time.perf_counter()
        use_argmax = (
            str(self.task_mode).strip().lower() == "classifier"
            and str(_structured(self.clf_prediction_mode)).strip().lower() == "argmax"
        )
        if use_argmax:
            self.thresholds_ = np.array([0.5, 1.5, 2.5], dtype=np.float64)
            self.calibration_source_ = "argmax"
            train_preds = np.asarray(rf_model.predict(X_orig_mat), dtype=np.int64).ravel()
            self.calib_qwk_ = float(quadratic_weighted_kappa(y_fit_orig, train_preds))
            self.calib_accuracy_pct_ = float(100.0 * np.mean(y_fit_orig == train_preds)) if len(y_fit_orig) else np.nan
        else:
            cal_scores, cal_y, cal_src = self._compute_calibration_scores(
                X_orig_mat, y_fit_orig, rf_model, aug_applied=aug_applied
            )
            self.calibration_source_ = cal_src
            self.thresholds_ = self._fit_thresholds(cal_y, cal_scores)
            cal_preds = apply_thresholds(cal_scores, self.thresholds_)
            self.calib_qwk_ = float(quadratic_weighted_kappa(cal_y, cal_preds))
            self.calib_accuracy_pct_ = float(100.0 * np.mean(cal_y == cal_preds)) if len(cal_y) else np.nan
            train_scores = self._continuous_from_model(rf_model, X_orig_mat)
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
            oob_str = f"{self.oob_score_:.4f}" if np.isfinite(self.oob_score_) else "N/A"
            th_str = "[" + ", ".join(f"{float(t):.3f}" for t in self.thresholds_) + "]"
            print(
                f"  [RF fit] Done in {t_tot:.2f}s (ext={t_ext:.2f}s, rf={t_rf:.2f}s, cal={t_cal:.2f}s) | "
                f"OOB={oob_str} | cal_src={self.calibration_source_} | thresholds={th_str} | "
                f"calib: Normal={_fmt_pct(self.calib_accuracy_pct_)} -> QWK={self.calib_qwk_:.5f} | "
                f"train: Normal={_fmt_pct(self.train_accuracy_pct_)} -> QWK={self.train_qwk_:.5f}",
                flush=True,
            )
        if v_lvl >= 3 and len(self.feature_names_out_) > 0:
            top_mdi = self.get_feature_importances_df().head(5)
            top_str = ", ".join(f"{r['feature']}={r['rf_importance']:.4f}" for _, r in top_mdi.iterrows())
            print(f"  [RF fit] Top 5 MDI features: {top_str}", flush=True)
        return self

    def get_history_dict(self) -> Dict[str, List[float]]:
        return dict(getattr(self, "history_", {}))

    def predict_continuous_from_features(self, feat_df: pd.DataFrame) -> np.ndarray:
        X_mat = self._transform_feature_matrix(feat_df)
        return self._continuous_from_model(self.model_, X_mat)

    def predict_from_features(self, feat_df: pd.DataFrame) -> np.ndarray:
        X_mat = self._transform_feature_matrix(feat_df)
        if (
            str(self.task_mode).strip().lower() == "classifier"
            and str(_structured(self.clf_prediction_mode)).strip().lower() == "argmax"
        ):
            return np.clip(np.asarray(self.model_.predict(X_mat), dtype=np.int64).ravel(), 0, 3)
        scores = self._continuous_from_model(self.model_, X_mat)
        return apply_thresholds(scores, self.thresholds_)

    def predict_continuous(self, X: pd.DataFrame) -> np.ndarray:
        feat_df = self.extractor_.transform(strip_leaky_columns(X))
        return self.predict_continuous_from_features(feat_df)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if str(self.task_mode).strip().lower() != "classifier":
            raise AttributeError("predict_proba is only available when task_mode='classifier'.")
        feat_df = self.extractor_.transform(strip_leaky_columns(X))
        X_mat = self._transform_feature_matrix(feat_df)
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
        """Return Random Forest MDI (impurity-reduction) feature importances sorted descending."""
        cols = list(getattr(self, "feature_names_out_", []))
        imps = np.asarray(getattr(self, "feature_importances_", np.zeros(len(cols))), dtype=np.float64)
        df = pd.DataFrame({
            "feature": cols,
            "family": [_feature_family_label(c) for c in cols],
            "rf_importance": imps,
        })
        return df.sort_values("rf_importance", ascending=False, kind="stable").reset_index(drop=True)

    def get_model_summary(self) -> Dict[str, Any]:
        """Return compact diagnostic summary of the fitted Random Forest estimator."""
        return {
            "task_mode": str(self.task_mode),
            "n_estimators": int(getattr(self.model_, "n_estimators", self.n_estimators)),
            "n_features_in": int(len(getattr(self, "feature_names_out_", []))),
            "n_samples_retained": int(getattr(self, "n_samples_retained_", 0)),
            "n_samples_removed": int(getattr(self, "n_samples_removed_", 0)),
            "n_augmented_rows": int(getattr(self, "n_augmented_rows_", 0)),
            "effective_nan_handling": getattr(self, "effective_nan_handling_", self.nan_handling),
            "calibration_source": getattr(self, "calibration_source_", None),
            "thresholds": [round(float(t), 5) for t in getattr(self, "thresholds_", [])],
            "oob_score": None if not np.isfinite(getattr(self, "oob_score_", np.nan)) else round(float(self.oob_score_), 5),
            "calib_qwk": round(float(getattr(self, "calib_qwk_", np.nan)), 5),
            "train_qwk": round(float(getattr(self, "train_qwk_", np.nan)), 5),
        }


# ---------------------------------------------------------------------------
# RF-Aware Inactive Parameter Pruning, Deduplication & Searchers
# ---------------------------------------------------------------------------

def inactive_rf_parameters_for_candidate(params: Mapping[str, Any]) -> set:
    """Identify inactive ``extractor__*``, ``augmentor__*``, and Random Forest params."""
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

    # Inactive RF params conditioned on task_mode, bootstrap, and threshold_calibration
    task_mode = str(p.get("task_mode", "regressor")).strip().lower()
    if task_mode == "regressor":
        for name in ("clf_criterion", "class_weight", "clf_prediction_mode"):
            inactive.add(f"{prefix}{name}")
    else:
        for name in ("reg_criterion", "sample_weight_mode"):
            inactive.add(f"{prefix}{name}")
        if str(p.get("clf_prediction_mode", "expected_value")).strip().lower() == "argmax":
            inactive.add(f"{prefix}threshold_calibration")
            inactive.add(f"{prefix}threshold_calibration_source")

    if not _coerce_bool(p.get("bootstrap", True), default=True):
        inactive.add(f"{prefix}max_samples")

    tc = p.get("threshold_calibration", "optimize_qwk")
    tc_norm = "none" if _is_none_like(tc) else str(tc).strip().lower()
    if tc_norm in {"fixed_midpoint", "round", "none"}:
        inactive.add(f"{prefix}threshold_calibration_source")

    return inactive


def rf_candidate_error(params: Mapping[str, Any], base_params: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Validate both ``SequenceExtractor`` DSP bounds and Random Forest parameter bounds."""
    merged = dict(base_params or {})
    merged.update(_decode_skopt_params(params))
    ext_kwargs = {
        k.split("extractor__", 1)[1]: v
        for k, v in merged.items()
        if "extractor__" in k and "noop__" not in k
    }
    rf_kwargs = {
        (k[len("estimator__"):] if k.startswith("estimator__") else k): v
        for k, v in merged.items()
        if "extractor__" not in k and "augmentor__" not in k and k not in {"estimator", "steps", "memory", "verbose", "transform_input"}
    }
    try:
        validate_sequence_extractor_params(ext_kwargs)
        validate_rf_params(rf_kwargs)
    except InvalidParams as exc:
        return str(exc)
    return None


def _canonical_rf_candidate_key(params: Mapping[str, Any], base_params: Optional[Mapping[str, Any]] = None) -> str:
    merged = dict(base_params or {})
    merged.update(_decode_skopt_params(params))
    inactive = inactive_rf_parameters_for_candidate(merged)
    eff = {k: repr(v) for k, v in sorted(params.items()) if k not in inactive}
    return hashlib.md5(json.dumps(eff, sort_keys=True).encode()).hexdigest()


def deduplicate_rf_param_grid(
    param_grid: Mapping[str, Sequence[Any]],
    *,
    base_params: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Expand and deduplicate a Random Forest grid search space, dropping invalid & equivalent clones."""
    grid = list(ParameterGrid(dict(param_grid)))
    seen: set = set()
    kept: List[Dict[str, Any]] = []
    for cand in grid:
        if rf_candidate_error(cand, base_params=base_params) is not None:
            continue
        sig = _canonical_rf_candidate_key(cand, base_params=base_params)
        if sig not in seen:
            seen.add(sig)
            kept.append(cand)
    return [{k: [v] for k, v in cand.items()} for cand in kept]


def _rf_candidate_brief(decoded: Mapping[str, Any]) -> str:
    """Compact summary of active RF & key extractor params for verbose >= 2 logging."""
    def _g(key: str, default: Any = "?") -> Any:
        return decoded.get(f"estimator__{key}", decoded.get(key, default))

    tm = str(_g("task_mode", "regressor")).strip().lower()
    crit = _g("reg_criterion") if tm == "regressor" else _g("clf_criterion")
    extra = (
        f"sw={_g('sample_weight_mode')}"
        if tm == "regressor"
        else f"cw={_g('class_weight')} pred={_g('clf_prediction_mode')}"
    )
    return (
        f"task={tm} crit={crit} trees={_g('n_estimators')} depth={_g('max_depth')} "
        f"leaf={_g('min_samples_leaf')} max_feat={_g('max_features')} boot={_g('bootstrap')} "
        f"ccp={_g('ccp_alpha')} {extra} nan={_g('nan_handling')} cal={_g('threshold_calibration')}({_g('threshold_calibration_source')}) | "
        f"bin={_g('extractor__bin_minutes')}m miss={_g('extractor__missing_feature_strategy')} "
        f"fs={_g('extractor__feature_selection_method')} aug_p={_g('augmentor__prob')}"
    )


def _evaluate_rf_candidate_cv(
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


class RFHierarchicalBayesSearchCV(BaseEstimator):
    """Bayesian optimization over ``SequenceExtractor``, ``TSAugmentor`` & Random Forest hyperparameters."""

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
            raise ImportError("scikit-optimize is required for RFHierarchicalBayesSearchCV.")
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
            sig = _canonical_rf_candidate_key(decoded, base_params=base_params)
            if v_lvl >= 2:
                print(f"\n[Bayes {it + 1}/{total} START] {_rf_candidate_brief(decoded)}", flush=True)
            if v_lvl >= 3:
                inactive = inactive_rf_parameters_for_candidate({**base_params, **decoded})
                active_d = {k: v for k, v in sorted(decoded.items()) if k not in inactive}
                print(f"  [Active params ({len(active_d)})]: {active_d}", flush=True)
            if sig in cache:
                fold_qwks, fold_accs, elapsed = cache[sig]
                cached_tag = " (cached)"
            else:
                err = rf_candidate_error(decoded, base_params=base_params)
                if err is not None:
                    if v_lvl >= 1:
                        print(f"  [Bayes {it + 1}/{total} INVALID] {err}", flush=True)
                    n_sp = cv.get_n_splits(X, y, groups) if hasattr(cv, "get_n_splits") else 1
                    err_val = -1.0 if self.error_score == "raise" else float(self.error_score)
                    fold_qwks, fold_accs, elapsed = [err_val] * n_sp, [-1.0] * n_sp, 0.0
                else:
                    fold_qwks, fold_accs, elapsed = _evaluate_rf_candidate_cv(
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
                nt = decoded.get("estimator__n_estimators", decoded.get("n_estimators", "?"))
                md = decoded.get("estimator__max_depth", decoded.get("max_depth", "?"))
                bm = decoded.get("estimator__extractor__bin_minutes", decoded.get("extractor__bin_minutes", "?"))
                print(
                    f"[Bayes {it + 1}/{total}] task={tm} trees={nt} depth={md} bin={bm}m | "
                    f"Normal={acc_lbl} -> Comp QWK={mean_qwk:.5f} | "
                    f"best: Normal={best_acc_lbl} -> QWK={best_so_far:.5f} ({elapsed:.2f}s){cached_tag}",
                    flush=True,
                )
                if ok_idx and (v_lvl >= 2 or len(fold_qwks) > 1):
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
        _populate_param_columns(self.cv_results_, all_params)

        self.best_index_ = int(refit_valid_candidate(self.cv_results_)) if callable(self.refit) else int(np.argmax(means))
        self.best_params_ = dict(all_params[self.best_index_])
        self.best_score_ = float(means[self.best_index_])
        if self.refit:
            if v_lvl >= 1:
                best_cv_acc = float(self.cv_results_["mean_test_accuracy_pct"][self.best_index_])
                print(
                    f"\n[Bayes REFIT] Refitting best candidate #{self.best_index_ + 1} "
                    f"(CV Normal={_fmt_pct(best_cv_acc)} -> Comp QWK={self.best_score_:.5f}): "
                    f"{_rf_candidate_brief(self.best_params_)}",
                    flush=True,
                )
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            _propagate_verbose(self.best_estimator_, v_lvl)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.best_estimator_.predict(X)


class RFGASearchCV(BaseEstimator):
    """Evolutionary (Genetic Algorithm) search over ``SequenceExtractor``, ``TSAugmentor`` & Random Forest."""

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
                sig = _canonical_rf_candidate_key(decoded, base_params=base_params)
                if v_lvl >= 2:
                    print(f"\n[GA gen {gen + 1}/{n_gen} | cand {c_idx + 1}/{pop_size} START] {_rf_candidate_brief(decoded)}", flush=True)
                if v_lvl >= 3:
                    inactive = inactive_rf_parameters_for_candidate({**base_params, **decoded})
                    active_d = {k: v for k, v in sorted(decoded.items()) if k not in inactive}
                    print(f"  [Active params ({len(active_d)})]: {active_d}", flush=True)
                if sig in cache:
                    fold_qwks, fold_accs, elapsed = cache[sig]
                    cached_tag = " (cached)"
                else:
                    err = rf_candidate_error(decoded, base_params=base_params)
                    if err is not None:
                        if v_lvl >= 1:
                            print(f"  [GA gen {gen + 1}/{n_gen} | cand {c_idx + 1}/{pop_size} INVALID] {err}", flush=True)
                        n_sp = cv.get_n_splits(X, y, groups) if hasattr(cv, "get_n_splits") else 1
                        err_val = -1.0 if self.error_score == "raise" else float(self.error_score)
                        fold_qwks, fold_accs, elapsed = [err_val] * n_sp, [-1.0] * n_sp, 0.0
                    else:
                        fold_qwks, fold_accs, elapsed = _evaluate_rf_candidate_cv(
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
                    nt = decoded.get("estimator__n_estimators", decoded.get("n_estimators", "?"))
                    md = decoded.get("estimator__max_depth", decoded.get("max_depth", "?"))
                    bm = decoded.get("estimator__extractor__bin_minutes", decoded.get("extractor__bin_minutes", "?"))
                    print(
                        f"[GA gen {gen + 1}/{n_gen} | cand {c_idx + 1}/{pop_size}] task={tm} trees={nt} depth={md} bin={bm}m | "
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
        _populate_param_columns(self.cv_results_, all_params)

        self.best_index_ = int(refit_valid_candidate(self.cv_results_)) if callable(self.refit) else int(np.argmax(means))
        self.best_params_ = dict(all_params[self.best_index_])
        self.best_score_ = float(means[self.best_index_])
        if self.refit:
            if v_lvl >= 1:
                best_cv_acc = float(self.cv_results_["mean_test_accuracy_pct"][self.best_index_])
                print(
                    f"\n[GA REFIT] Refitting best candidate #{self.best_index_ + 1} "
                    f"(CV Normal={_fmt_pct(best_cv_acc)} -> Comp QWK={self.best_score_:.5f}): "
                    f"{_rf_candidate_brief(self.best_params_)}",
                    flush=True,
                )
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            _propagate_verbose(self.best_estimator_, v_lvl)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.best_estimator_.predict(X)


# ---------------------------------------------------------------------------
# Trained-Model Permutation Importance (Random Forest)
# ---------------------------------------------------------------------------

def compute_rf_permutation_importance(
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
    """Compute trained-model permutation importance for ``RandomForestSequenceEstimator``.

    Mirrors ``base_utils.compute_permutation_importance``:
    * Evaluates on the provided holdout split ``(X, y)`` by default (recommended for Random
      Forest so in-sample tree memorization cannot bias permutation drops), or on internal
      ``validation_sequence_ids_`` if ``require_internal_validation=True`` and present.
    * Permutes coherent feature families (``clinical_tabular``, ``missing_indicators``,
      ``autocorr_circadian``, ``lag_lead_dynamics``, ``sleep_architecture``,
      ``stft_wavelet_spectral``, ``seglearn_kinematic``, ``static_open_source``, ``noop_all``),
      the top-K individual Random Forest features ranked by MDI importance, plus individual
      ``noop_*`` and ANOVA-selected static features.
    * Populates ``importance_df.attrs`` (``scope``, ``baseline_macro_f1``, ``baseline_qwk``,
      ``permutable_fraction``, ``paths``) and optionally exports CSV, JSON, and PNG plot.
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
    rf_imp_df = est.get_feature_importances_df()
    for feat_name in rf_imp_df["feature"].head(max(0, int(top_k_individual))).tolist():
        if feat_name in feat_df.columns and feat_df[feat_name].notna().sum() >= 2:
            groups.append(("rf_feature", feat_name, [feat_name]))
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
            ax.barh(top_plot["feature_or_group"], top_plot["importance_mean"], xerr=top_plot["importance_std"], color="#10b981", alpha=0.85)
            ax.axvline(0.0, color="#64748b", linestyle="--", linewidth=1.0)
            ax.set_xlabel("Permutation Importance (Competition QWK drop)")
            ax.set_title(f"Random Forest Permutation Importance ({scope_name})")
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
