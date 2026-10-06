import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.utils.validation import check_is_fitted


def expand_booking_dates_fast(df):
    """Ultra-fast date expansion using numpy"""
    start_dates = df["started_at_dt"].dt.normalize()
    end_dates = df["closed_at_dt"].dt.normalize()
    days_diff = (end_dates - start_dates).dt.days + 1

    # Create arrays for expansion
    booking_indices = np.repeat(df.index.values, days_diff.values)
    day_offsets = np.concatenate([np.arange(days) for days in days_diff.values])

    # Create expanded dataframe
    expanded_df = df.iloc[booking_indices].copy()
    expanded_df['date_occupied'] = (
            start_dates.iloc[booking_indices].reset_index(drop=True) +
            pd.to_timedelta(day_offsets, unit='D')
    )

    return expanded_df.reset_index(drop=True)


def expand_booking_dates_optimized(df):
    """Fastest approach using pure pandas vectorization"""
    # Calculate date ranges directly
    starts = df['started_at_dt'].dt.normalize()
    ends = df['closed_at_dt'].dt.normalize()

    # Use list comprehension with zip (much faster than apply)
    df = df.copy()
    df['occupancy_days'] = [
        pd.date_range(s, e, freq='D')
        for s, e in zip(starts, ends)
    ]

    return df.explode('occupancy_days').rename(
        columns={'occupancy_days': 'date_occupied'}
    ).reset_index(drop=True)


def make_group_cv_splitter(groups, n_splits=5, test_size=0.25, random_state=42):
    group_count = pd.Series(groups).nunique()
    if group_count < 2:
        raise ValueError('Grouped model search requires at least two distinct groups.')
    if n_splits <= 1:
        return GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
    return GroupKFold(n_splits=min(int(n_splits), group_count))


class SequenceExtractor:
    def __init__(self, method='none', threshold=1.5, max_outlier_features=0):
        self.method = method
        self.threshold = threshold
        self.max_outlier_features = max_outlier_features

    def fit_resample(self, X, y):
        X_values = np.asarray(X, dtype=float)
        y_values = np.asarray(y)
        if X_values.ndim != 2 or len(X_values) != len(y_values):
            raise ValueError('X must be 2D and have the same number of rows as y.')
        if self.method not in {'none', 'iqr', 'zscore'}:
            raise ValueError("method must be 'none', 'iqr', or 'zscore'.")

        outlier_counts = np.zeros(len(X_values), dtype=int)
        if self.method == 'iqr':
            for column_index in range(X_values.shape[1]):
                column = X_values[:, column_index]
                finite_values = column[np.isfinite(column)]
                if finite_values.size == 0:
                    continue
                lower_quartile, upper_quartile = np.percentile(finite_values, [25, 75])
                interquartile_range = upper_quartile - lower_quartile
                if interquartile_range <= np.finfo(float).eps:
                    continue
                lower_bound = lower_quartile - self.threshold * interquartile_range
                upper_bound = upper_quartile + self.threshold * interquartile_range
                outlier_counts += np.isfinite(column) & ((column < lower_bound) | (column > upper_bound))
        elif self.method == 'zscore':
            for column_index in range(X_values.shape[1]):
                column = X_values[:, column_index]
                finite_values = column[np.isfinite(column)]
                if finite_values.size < 2:
                    continue
                standard_deviation = finite_values.std()
                if standard_deviation <= np.finfo(float).eps:
                    continue
                z_scores = np.abs((column - finite_values.mean()) / standard_deviation)
                outlier_counts += np.isfinite(column) & (z_scores > self.threshold)

        self.support_mask_ = outlier_counts <= self.max_outlier_features
        if self.support_mask_.sum() < 2:
            self.support_mask_ = np.ones(len(X_values), dtype=bool)
        self.n_samples_removed_ = int((~self.support_mask_).sum())
        return X_values[self.support_mask_], y_values[self.support_mask_]


class SequenceFilteredRandomForestRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        filter_method='none',
        filter_threshold=1.5,
        max_outlier_features=0,
        select_features=False,
        n_features=8,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    ):
        self.filter_method = filter_method
        self.filter_threshold = filter_threshold
        self.max_outlier_features = max_outlier_features
        self.select_features = select_features
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        input_feature_names = list(X.columns) if hasattr(X, 'columns') else None
        X_values = np.asarray(X, dtype=float)
        y_values = np.asarray(y)
        if X_values.ndim != 2 or len(X_values) != len(y_values):
            raise ValueError('X must be 2D and have the same number of rows as y.')

        self.sequence_extractor_ = SequenceExtractor(
            method=self.filter_method,
            threshold=self.filter_threshold,
            max_outlier_features=self.max_outlier_features,
        )
        X_filtered, y_filtered = self.sequence_extractor_.fit_resample(X_values, y_values)

        self.n_features_in_ = X_values.shape[1]
        self.feature_names_in_ = np.asarray(
            input_feature_names if input_feature_names is not None else [f'x{index}' for index in range(self.n_features_in_)],
            dtype=object,
        )
        if self.select_features:
            if self.n_features is None or int(self.n_features) < 1:
                raise ValueError('n_features must be a positive integer when select_features=True.')
            centered_X = X_filtered - X_filtered.mean(axis=0)
            centered_y = y_filtered.astype(float) - y_filtered.astype(float).mean()
            denominator = np.sqrt((centered_X ** 2).sum(axis=0) * (centered_y ** 2).sum())
            correlations = np.divide(
                np.abs(centered_X.T @ centered_y),
                denominator,
                out=np.zeros(X_filtered.shape[1], dtype=float),
                where=denominator > np.finfo(float).eps,
            )
            feature_count = min(int(self.n_features), X_filtered.shape[1])
            selected_indices = np.argsort(correlations)[::-1][:feature_count]
            self.selected_feature_indices_ = np.sort(selected_indices)
        else:
            self.selected_feature_indices_ = np.arange(X_filtered.shape[1])

        self.selected_feature_names_ = self.feature_names_in_[self.selected_feature_indices_]
        X_selected = X_filtered[:, self.selected_feature_indices_]
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X_selected, y_filtered)
        self.feature_importances_ = self.model_.feature_importances_
        self.n_training_samples_ = len(y_filtered)
        return self

    def predict(self, X):
        check_is_fitted(self, 'model_')
        X_values = np.asarray(X, dtype=float)
        return self.model_.predict(X_values[:, self.selected_feature_indices_])