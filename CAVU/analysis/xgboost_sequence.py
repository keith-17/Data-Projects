"""Fold-local feature extraction and dated forecasting for the CAVU notebook.

The estimator is model-agnostic: parameters are tunable through sklearn's
``extractor__...`` and ``estimator__...`` namespaces inside a Pipeline.
"""

from numbers import Integral, Real
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _numeric_frame(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with named numeric columns.")
    if not X.columns.is_unique or not all(isinstance(c, str) for c in X.columns):
        raise ValueError("X must have unique string column names.")
    if not len(X.columns):
        raise ValueError("X must contain at least one feature.")
    invalid = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
    if invalid:
        raise ValueError(f"Features must be numeric; encode these columns first: {invalid}")
    frame = X.astype(float)
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("X contains infinity; use NaN for missing feature values.")
    return frame


def _target_series(y, index):
    if isinstance(y, pd.Series) and not y.index.equals(index):
        raise ValueError("The y index and order must exactly match X.")
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or len(values) != len(index):
        raise ValueError("y must be one-dimensional and have the same number of rows as X.")
    if not np.isfinite(values).all():
        raise ValueError("Target values must be finite; missing targets cannot be imputed.")
    return pd.Series(values, index=index, name=getattr(y, "name", None))


class CavuSequenceExtractor(BaseEstimator):
    """Construct features, filter training rows, impute, then select features.

    ``filter`` retains the supplied numeric feature definitions. ``forecast``
    accepts a bank from :func:`build_forecast_bank`; horizon is fixed per search.
    Outlier bounds, medians and absolute Pearson rankings use this fit's rows
    only. All-missing training features use zero because no median exists.
    Prediction never filters rows. Rolling standard deviation uses ``ddof=0``.
    """

    def __init__(self, mode="filter", horizon_weeks=1, history_window=8,
                 include_lags=True, include_rolling=True, filter_method="none",
                 filter_threshold=1.5, max_outlier_features=0,
                 select_features=False, n_features=12):
        self.mode = mode
        self.horizon_weeks = horizon_weeks
        self.history_window = history_window
        self.include_lags = include_lags
        self.include_rolling = include_rolling
        self.filter_method = filter_method
        self.filter_threshold = filter_threshold
        self.max_outlier_features = max_outlier_features
        self.select_features = select_features
        self.n_features = n_features

    def _construct(self, X):
        X = _numeric_frame(X)
        if self.mode == "filter":
            return X.copy()
        if self.mode != "forecast":
            raise ValueError("mode must be 'filter' or 'forecast'.")
        _positive_int(self.horizon_weeks, "horizon_weeks")
        _positive_int(self.history_window, "history_window")
        bank_horizon = X.attrs.get("horizon_weeks", self.horizon_weeks)
        if bank_horizon != self.horizon_weeks:
            raise ValueError("horizon_weeks must match the feature bank and stay fixed within one search.")
        histories, known = {}, []
        for col in X:
            match = re.fullmatch(r"hist__(.+)__lag_([1-9][0-9]*)", col)
            if match:
                histories.setdefault(match.group(1), {})[int(match.group(2))] = col
            elif col.startswith(("cal__", "id__")):
                known.append(col)
            else:
                raise ValueError(f"Unexpected forecast column {col!r}; use a dated feature bank.")
        if not histories:
            raise ValueError("Forecast mode requires hist__<feature>__lag_<n> columns.")
        result = X[known].copy()
        lags = range(self.horizon_weeks, self.horizon_weeks + self.history_window)
        for feature, columns in histories.items():
            missing = [lag for lag in lags if lag not in columns]
            if missing:
                raise ValueError(f"History bank lacks {feature!r} lags {missing}.")
            window = X[[columns[lag] for lag in lags]]
            if self.include_lags:
                result[window.columns] = window
            if self.include_rolling:
                complete = window.notna().all(axis=1)
                for stat in ("mean", "std", "min", "max"):
                    values = window.std(axis=1, ddof=0) if stat == "std" else getattr(window, stat)(axis=1)
                    result[f"roll__{feature}__{stat}"] = values.where(complete)
        if not len(result.columns):
            raise ValueError("The configured extractor produces no features.")
        return result

    def fit_resample(self, X, y):
        features = self._construct(X)
        target = _target_series(y, features.index)
        if self.filter_method not in {"none", "iqr", "zscore"}:
            raise ValueError("filter_method must be 'none', 'iqr', or 'zscore'.")
        if not isinstance(self.filter_threshold, Real) or not np.isfinite(self.filter_threshold) or self.filter_threshold <= 0:
            raise ValueError("filter_threshold must be a positive finite number.")
        if isinstance(self.max_outlier_features, bool) or not isinstance(self.max_outlier_features, Integral) or self.max_outlier_features < 0:
            raise ValueError("max_outlier_features must be a nonnegative integer.")
        _positive_int(self.n_features, "n_features")
        lower = pd.Series(-np.inf, index=features.columns)
        upper = pd.Series(np.inf, index=features.columns)
        if self.filter_method == "iqr":
            q1, q3 = features.quantile(.25), features.quantile(.75)
            spread = q3 - q1
            usable = spread > np.finfo(float).eps
            lower[usable] = q1[usable] - self.filter_threshold * spread[usable]
            upper[usable] = q3[usable] + self.filter_threshold * spread[usable]
        elif self.filter_method == "zscore":
            mean, spread = features.mean(), features.std(ddof=0)
            usable = spread > np.finfo(float).eps
            lower[usable] = mean[usable] - self.filter_threshold * spread[usable]
            upper[usable] = mean[usable] + self.filter_threshold * spread[usable]
        counts = ((features < lower) | (features > upper)).sum(axis=1)
        mask = (counts <= self.max_outlier_features).to_numpy()
        minimum = max(2, int(np.ceil(.1 * len(features))))
        if mask.sum() < minimum:
            raise ValueError(f"Filtering retained {mask.sum()} of {len(features)} rows; at least {minimum} are required. Relax the filter or provide more training data.")
        retained, target = features.loc[mask].copy(), target.loc[mask].copy()
        self.imputation_values_ = retained.median().fillna(0.0)
        retained = retained.fillna(self.imputation_values_)
        self.feature_scores_ = pd.Series(0.0, index=retained.columns)
        varying = retained.nunique() > 1
        if target.nunique() > 1 and varying.any():
            self.feature_scores_.loc[varying] = retained.loc[:, varying].corrwith(target).abs().fillna(0.0)
        selected = list(retained.columns)
        if self.select_features:
            selected = self.feature_scores_.sort_values(ascending=False, kind="stable").index[:self.n_features].tolist()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.constructed_feature_names_ = np.asarray(features.columns, dtype=object)
        self.selected_feature_names_ = np.asarray(selected, dtype=object)
        self.n_features_in_, self.n_features_out_ = len(X.columns), len(selected)
        self.n_samples_in_, self.n_samples_retained_ = len(X), int(mask.sum())
        self.n_samples_removed_ = len(X) - self.n_samples_retained_
        self.sample_mask_ = self.support_mask_ = mask
        self.filter_bounds_ = pd.DataFrame({"lower": lower, "upper": upper})
        return retained[selected], target

    def fit(self, X, y):
        self.fit_resample(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_feature_names_")
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Prediction columns and order must exactly match training X.")
        features = self._construct(X)
        return features.fillna(self.imputation_values_).loc[:, self.selected_feature_names_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_feature_names_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features must match fitted feature_names_in_.")
        return self.selected_feature_names_.copy()


class SequenceFilteredRegressor(RegressorMixin, BaseEstimator):
    """Clone and fit an extractor and regressor together within each CV fold.

    Fit keywords are forwarded, so unsupported arguments fail explicitly.
    ``sample_weight`` and ``base_margin`` are filtered with training rows.
    External evaluation sets and early stopping require separate, fold-local
    evaluation plumbing and are deliberately rejected here.
    """

    def __init__(self, extractor, estimator, scale_features=False,
                 clip_nonnegative=False, verbose=0):
        self.extractor = extractor
        self.estimator = estimator
        self.scale_features = scale_features
        self.clip_nonnegative = clip_nonnegative
        self.verbose = verbose

    def fit(self, X, y, **fit_params):
        params = self.estimator.get_params(deep=True)
        if any(k.split("__")[-1] == "early_stopping_rounds" and v is not None for k, v in params.items()) or fit_params.get("early_stopping_rounds") is not None or "eval_set" in fit_params:
            raise ValueError("Early stopping/eval_set requires fold-local evaluation plumbing; do not pass the external test set. Disable early_stopping_rounds for this wrapper.")
        self.extractor_ = clone(self.extractor)
        self.estimator_ = clone(self.estimator)
        features, target = self.extractor_.fit_resample(X, y)
        for name in ("sample_weight", "base_margin"):
            if name in fit_params and fit_params[name] is not None:
                values = _target_series(fit_params[name], X.index)
                fit_params[name] = values.iloc[self.extractor_.sample_mask_].to_numpy()
        self.scaler_ = StandardScaler().fit(features) if self.scale_features else None
        fitted_X = self.scaler_.transform(features) if self.scaler_ is not None else features
        self.estimator_.fit(fitted_X, target, **fit_params)
        self.feature_names_in_ = self.extractor_.feature_names_in_.copy()
        self.feature_names_out_ = self.extractor_.get_feature_names_out()
        self.selected_feature_names_ = self.feature_names_out_.copy()
        self.n_features_in_ = self.extractor_.n_features_in_
        self.n_features_out_ = self.extractor_.n_features_out_
        self.n_samples_removed_ = self.extractor_.n_samples_removed_
        if self.verbose:
            print(f"Retained {len(features)}/{len(X)} training rows; fitted {features.shape[1]} features.")
        return self

    def predict(self, X):
        check_is_fitted(self, "feature_names_out_")
        features = self.extractor_.transform(X)
        if self.scaler_ is not None:
            features = self.scaler_.transform(features)
        prediction = np.asarray(self.estimator_.predict(features))
        return np.maximum(prediction, 0) if self.clip_nonnegative else prediction

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.extractor_.get_feature_names_out(input_features)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.estimator_.feature_importances_)

    @property
    def feature_importances_series_(self):
        return pd.Series(self.feature_importances_, index=self.get_feature_names_out(), name="importance")


def build_forecast_bank(weekly_df, weekly_meta, horizon_weeks=1,
                        max_history_weeks=8, history_columns=("occupancy",),
                        key_columns=("terminal_num", "carpark_name_num")):
    """Return indexed ``(X, y, metadata)`` using exact key/date history joins.

    Keep rows with dated history coverage at every lag from ``horizon_weeks``
    through ``horizon_weeks + max_history_weeks - 1``. Missing dated rows never
    become zeros; missing feature values remain NaN for fold-local imputation.
    Target occupancy must be finite. Keys must already be numerically encoded.
    Calendar features describe the target week and retain its ISO year.
    """
    _positive_int(horizon_weeks, "horizon_weeks")
    _positive_int(max_history_weeks, "max_history_weeks")
    if not isinstance(weekly_df, pd.DataFrame) or not isinstance(weekly_meta, pd.DataFrame):
        raise TypeError("weekly_df and weekly_meta must be pandas DataFrames.")
    if not weekly_df.index.is_unique or not weekly_meta.index.is_unique:
        raise ValueError("weekly_df and weekly_meta must have unique indices.")
    if not weekly_df.columns.is_unique or not weekly_meta.columns.is_unique:
        raise ValueError("weekly_df and weekly_meta must have unique column names.")
    histories, keys = list(history_columns), list(key_columns)
    if not histories or not keys or len(set(histories)) != len(histories) or len(set(keys)) != len(keys):
        raise ValueError("history_columns and key_columns must be nonempty and unique.")
    required = set(histories + keys + ["occupancy"])
    if not required.issubset(weekly_df.columns) or "week_start" not in weekly_meta:
        raise ValueError(f"weekly_df requires columns {sorted(required)} and weekly_meta requires week_start.")
    if not weekly_df.index.isin(weekly_meta.index).all():
        raise ValueError("weekly_meta is missing indices required by weekly_df.")
    values = _numeric_frame(weekly_df[list(dict.fromkeys(histories + keys))])
    if values[keys].isna().any().any():
        raise ValueError("Entity keys must not contain missing values.")
    target = _target_series(weekly_df["occupancy"], weekly_df.index)
    meta = weekly_meta.loc[weekly_df.index].copy()
    dates = pd.to_datetime(meta["week_start"], errors="raise")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("week_start must contain complete dates at midnight.")
    if not dates.dt.dayofweek.eq(0).all():
        raise ValueError("week_start must be a Monday for every weekly row.")
    iso = dates.dt.isocalendar()
    if "weekofyear" in weekly_df:
        supplied_weeks = pd.to_numeric(weekly_df["weekofyear"], errors="coerce")
        if not supplied_weeks.eq(iso.week).fillna(False).all():
            raise ValueError("weekly_df.weekofyear does not match weekly_meta.week_start; refresh the dated metadata.")
    lookup_keys = weekly_df[keys].copy()
    lookup_keys["week_start"] = dates
    if lookup_keys.duplicated().any():
        raise ValueError("Each entity key + week_start combination must be unique.")
    source = values[histories].copy()
    source.index = pd.MultiIndex.from_frame(lookup_keys)
    bank = pd.DataFrame(index=weekly_df.index)
    bank["cal__iso_year"], bank["cal__week"], bank["cal__month"] = iso.year.astype(int), iso.week.astype(int), dates.dt.month
    phase = 2 * np.pi * (iso.week.astype(float) - 1) / 52.1775
    bank["cal__week_sin"], bank["cal__week_cos"] = np.sin(phase), np.cos(phase)
    for key in keys:
        bank[f"id__{key}"] = values[key]
        meta[key] = weekly_df[key]
    coverage = np.ones(len(bank), dtype=bool)
    for lag in range(horizon_weeks, horizon_weeks + max_history_weeks):
        requested = lookup_keys.copy()
        requested["week_start"] = dates - pd.Timedelta(weeks=lag)
        requested_index = pd.MultiIndex.from_frame(requested)
        coverage &= requested_index.isin(source.index)
        lagged = source.reindex(requested_index)
        for feature in histories:
            bank[f"hist__{feature}__lag_{lag}"] = lagged[feature].to_numpy()
    meta["week_start"] = dates
    meta["origin_week"] = dates - pd.Timedelta(weeks=horizon_weeks)
    meta["iso_year"] = iso.year.astype(int)
    meta["original_index"] = weekly_df.index.to_numpy()
    if not coverage.any():
        raise ValueError("No forecast rows have complete dated history coverage; shorten max_history_weeks or provide more weekly history.")
    bank.attrs["horizon_weeks"] = horizon_weeks
    bank.attrs["max_history_weeks"] = max_history_weeks
    bank.attrs["n_rows_without_history"] = int((~coverage).sum())
    return bank.loc[coverage].copy(), target.loc[coverage].copy(), meta.loc[coverage].copy()


def chronological_week_splits(dates, horizon_weeks, n_splits=3, test_size=.2):
    """Expanding, date-grouped splits, returned as positional index arrays.

    Integer ``test_size`` specifies weeks per validation fold; a float in (0,1)
    uses ``ceil(test_size * total_unique_weeks)`` per fold. The final n_splits
    disjoint blocks validate. Training target dates must be <= the earliest
    validation origin (first validation date minus horizon weeks). Input order
    may be arbitrary: entities sharing a date always stay together.
    """
    _positive_int(horizon_weeks, "horizon_weeks")
    _positive_int(n_splits, "n_splits")
    dates = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))
    if dates.isna().any():
        raise ValueError("Split dates must not contain missing dates.")
    weeks = dates.unique().sort_values()
    if isinstance(test_size, Integral) and not isinstance(test_size, bool):
        _positive_int(test_size, "test_size")
        width = int(test_size)
    elif isinstance(test_size, Real) and 0 < test_size < 1:
        width = int(np.ceil(float(test_size) * len(weeks)))
    else:
        raise ValueError("test_size must be a positive integer or a float between zero and one.")
    first = len(weeks) - n_splits * width
    if width == 0 or first <= 0:
        raise ValueError("Not enough unique weeks for the requested validation folds and initial training period.")
    result = []
    for start in range(first, len(weeks), width):
        validation_weeks = weeks[start:start + width]
        cutoff = validation_weeks[0] - pd.Timedelta(weeks=horizon_weeks)
        train = np.flatnonzero(dates <= cutoff)
        validation = np.flatnonzero(dates.isin(validation_weeks))
        if not len(train):
            raise ValueError("The forecast horizon leaves an empty training fold; reduce n_splits/test_size or provide more weeks.")
        result.append((train, validation))
    return result
