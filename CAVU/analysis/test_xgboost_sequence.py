"""Regression tests for calendar-safe sequence features and fold-local fitting.

Run from this directory with ``python -m unittest test_xgboost_sequence -v``.
The Bayesian/XGBoost smoke test is optional when those packages are absent.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from xgboost_sequence import (
    CavuSequenceExtractor,
    SequenceFilteredRegressor,
    build_forecast_bank,
    chronological_week_splits,
)

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer
    from xgboost import XGBRegressor
except ImportError:
    BayesSearchCV = None


KEYS = ("terminal_num", "carpark_name_num")


def weekly_panel(periods=42, shuffle=False):
    """Distinct unit levels make accidental cross-unit histories conspicuous."""
    weeks = pd.date_range("2022-11-07", periods=periods, freq="7D")
    records = []
    dates = []
    for terminal, carpark, level in [(1, 0, 100.0), (1, 1, 1000.0), (2, 0, 5000.0)]:
        for offset, week in enumerate(weeks):
            records.append(
                {
                    "terminal_num": terminal,
                    "carpark_name_num": carpark,
                    "occupancy": level + offset * 10.0,
                    "total_flights": 50.0 + offset,
                    "total_passengers": 1000.0 + offset * 100.0,
                    # These target-week fields must never become predictors.
                    "bookings_count": level + offset * 3.0,
                    "total_revenue": level * 20.0 + offset,
                }
            )
            dates.append(week)
    index = pd.Index(1003 + np.arange(len(records)) * 17, name="source_row")
    frame = pd.DataFrame(records, index=index)
    meta = pd.DataFrame({"week_start": dates}, index=index)
    if shuffle:
        frame = frame.sample(frac=1, random_state=11)
        meta = meta.sample(frac=1, random_state=23)
    return frame, meta


def source_row(frame, meta, week, terminal=1, carpark=0):
    mask = (
        frame.terminal_num.eq(terminal)
        & frame.carpark_name_num.eq(carpark)
        & meta.week_start.reindex(frame.index).eq(pd.Timestamp(week))
    )
    return frame.index[mask][0]


class ForecastBankTests(unittest.TestCase):
    def test_horizons_use_exact_calendar_lags_and_target_calendar(self):
        frame, meta = weekly_panel(shuffle=True)
        target = pd.Timestamp("2023-03-27")  # Week offset 20, across the year boundary.
        for horizon in (1, 4):
            with self.subTest(horizon=horizon):
                X, y, bank_meta = build_forecast_bank(
                    frame, meta, horizon_weeks=horizon, max_history_weeks=8
                )
                for terminal, carpark, level in [(1, 0, 100.0), (1, 1, 1000.0), (2, 0, 5000.0)]:
                    row = source_row(frame, meta, target, terminal, carpark)
                    self.assertEqual(X.loc[row, f"hist__occupancy__lag_{horizon}"], level + (20 - horizon) * 10)
                    self.assertEqual(X.loc[row, f"hist__occupancy__lag_{horizon + 7}"], level + (13 - horizon) * 10)
                    self.assertEqual(y.loc[row], frame.loc[row, "occupancy"])
                    self.assertEqual(bank_meta.loc[row, "week_start"], target)
                    self.assertEqual(bank_meta.loc[row, "origin_week"], target - pd.Timedelta(weeks=horizon))
                    self.assertEqual(X.loc[row, "cal__iso_year"], 2023)
                    self.assertEqual(X.loc[row, "cal__week"], target.isocalendar().week)
                    self.assertEqual(X.loc[row, "cal__month"], target.month)
                self.assertTrue(all(name.startswith(("hist__", "cal__", "id__")) for name in X.columns))
                self.assertFalse(any("revenue" in name or "bookings_count" in name for name in X.columns))

    def test_future_outcomes_do_not_change_predictors_at_earlier_origins(self):
        frame, meta = weekly_panel(shuffle=True)
        cutoff = pd.Timestamp("2023-03-20")
        changed = frame.copy()
        future = meta.week_start.reindex(frame.index).gt(cutoff)
        changed.loc[future, ["occupancy", "total_flights", "total_passengers", "bookings_count", "total_revenue"]] = 999999.0
        for horizon in (1, 4):
            with self.subTest(horizon=horizon):
                X, y, bank_meta = build_forecast_bank(frame, meta, horizon_weeks=horizon, max_history_weeks=8)
                other_X, other_y, _ = build_forecast_bank(changed, meta, horizon_weeks=horizon, max_history_weeks=8)
                earlier = bank_meta.index[bank_meta.origin_week.le(cutoff)]
                pd.testing.assert_frame_equal(X.loc[earlier], other_X.loc[earlier])
                forecast_rows = bank_meta.index[bank_meta.origin_week.eq(cutoff)]
                self.assertTrue((y.loc[forecast_rows] != other_y.loc[forecast_rows]).all())

    def test_missing_week_never_becomes_the_previous_observation(self):
        frame, meta = weekly_panel()
        missing_week = pd.Timestamp("2023-01-30")  # Offset 12.
        missing_row = source_row(frame, meta, missing_week)
        frame = frame.drop(index=missing_row)
        for horizon in (1, 4):
            with self.subTest(horizon=horizon):
                X, _, bank_meta = build_forecast_bank(frame, meta, horizon_weeks=horizon, max_history_weeks=3)
                for lag in range(horizon, horizon + 3):
                    affected_week = missing_week + pd.Timedelta(weeks=lag)
                    affected = source_row(frame, meta, affected_week)
                    unaffected_unit = source_row(frame, meta, affected_week, carpark=1)
                    self.assertNotIn(affected, X.index)
                    self.assertIn(unaffected_unit, X.index)
                next_week = missing_week + pd.Timedelta(weeks=horizon + 3)
                next_row = source_row(frame, meta, next_week)
                self.assertIn(next_row, X.index)
                expected_row = source_row(frame, meta, next_week - pd.Timedelta(weeks=horizon))
                self.assertEqual(X.loc[next_row, f"hist__occupancy__lag_{horizon}"], frame.loc[expected_row, "occupancy"])
                self.assertTrue(bank_meta.week_start.ge(meta.week_start.min() + pd.Timedelta(weeks=horizon + 2)).all())

    def test_shuffled_inputs_keep_source_index_and_target_alignment(self):
        frame, meta = weekly_panel(shuffle=True)
        X, y, bank_meta = build_forecast_bank(frame, meta, horizon_weeks=4, max_history_weeks=8)
        expected_index = frame.index[
            meta.week_start.reindex(frame.index).ge(meta.week_start.min() + pd.Timedelta(weeks=11))
        ]
        pd.testing.assert_index_equal(X.index, expected_index)
        pd.testing.assert_index_equal(y.index, expected_index)
        pd.testing.assert_index_equal(bank_meta.index, expected_index)
        np.testing.assert_array_equal(y.to_numpy(), frame.loc[expected_index, "occupancy"].to_numpy())
        pd.testing.assert_series_equal(bank_meta.week_start, meta.loc[expected_index, "week_start"], check_names=False)

    def test_duplicate_unit_week_is_rejected_even_with_different_row_index(self):
        frame, meta = weekly_panel()
        duplicate = frame.iloc[[10]].copy()
        duplicate.index = pd.Index([999999], name=frame.index.name)
        duplicate_meta = meta.iloc[[10]].copy()
        duplicate_meta.index = duplicate.index
        with self.assertRaises(ValueError):
            build_forecast_bank(pd.concat([frame, duplicate]), pd.concat([meta, duplicate_meta]))

    def test_invalid_metadata_and_duplicate_source_indices_are_rejected(self):
        frame, meta = weekly_panel()
        with self.assertRaises(ValueError):
            build_forecast_bank(frame, meta.drop(index=frame.index[0]))
        with self.assertRaises(ValueError):
            build_forecast_bank(pd.concat([frame, frame.iloc[[0]]]), meta)
        for offset in (pd.Timedelta(days=1), pd.Timedelta(hours=12)):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                build_forecast_bank(frame, meta.assign(week_start=meta.week_start + offset))
        dated_frame = frame.assign(weekofyear=meta.week_start.dt.isocalendar().week.astype(int))
        build_forecast_bank(dated_frame, meta)
        with self.assertRaises(ValueError):
            build_forecast_bank(dated_frame.assign(weekofyear=dated_frame.weekofyear + 1), meta)


class ExtractorTests(unittest.TestCase):
    def test_training_filter_aligns_target_and_holdout_uses_training_medians(self):
        index = pd.Index([f"train-{i}" for i in range(8)])
        X = pd.DataFrame(
            {"level": [10, 11, 9, 10, 12, 11, 10, 10000], "other": [2, 3, 2, np.nan, 3, 2, 2, 3]},
            index=index,
        )
        y = pd.Series(np.arange(8, dtype=float) * 7, index=index)
        extractor = CavuSequenceExtractor(mode="filter", filter_method="iqr", filter_threshold=1.5)
        filtered_X, filtered_y = extractor.fit_resample(X, y)
        pd.testing.assert_index_equal(filtered_X.index, index[:-1])
        pd.testing.assert_index_equal(filtered_y.index, index[:-1])
        np.testing.assert_array_equal(filtered_y, y.iloc[:-1])
        train_before = extractor.transform(X)
        holdout = pd.DataFrame(
            {"level": [np.nan, 1e9, -1e9], "other": [np.nan, -1e9, 1e9]},
            index=["missing", "large", "small"],
        )
        transformed = extractor.transform(holdout)
        pd.testing.assert_index_equal(transformed.index, holdout.index)
        self.assertEqual(transformed.loc["missing", "level"], 10.0)
        self.assertEqual(transformed.loc["missing", "other"], 2.0)
        self.assertEqual(transformed.loc["large", "level"], 1e9)
        self.assertEqual(transformed.loc["small", "level"], -1e9)
        pd.testing.assert_frame_equal(extractor.transform(X), train_before)
        np.testing.assert_array_equal(extractor.get_feature_names_out(), transformed.columns)

    def test_feature_selection_is_fitted_on_training_target_only(self):
        signal = np.arange(48, dtype=float)
        X = pd.DataFrame({"signal": signal, "unrelated": np.tile([1.0, -1.0, -1.0, 1.0], 12)})
        y = pd.Series(3.0 * signal + 4.0, index=X.index)
        extractor = CavuSequenceExtractor(mode="filter", select_features=True, n_features=1)
        train_X, _ = extractor.fit_resample(X, y)
        self.assertEqual(list(train_X.columns), ["signal"])
        heldout = pd.DataFrame({"signal": [np.nan, -2.0], "unrelated": [1e12, -1e12]}, index=[100, 101])
        projected = extractor.transform(heldout)
        self.assertEqual(list(projected.columns), ["signal"])
        self.assertEqual(projected.loc[100, "signal"], np.median(signal))
        pd.testing.assert_frame_equal(extractor.transform(X), train_X)

    def test_target_index_mismatch_cannot_silently_pair_wrong_rows(self):
        X = pd.DataFrame({"signal": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
        y = pd.Series([4.0, 5.0, 6.0], index=[30, 20, 10])
        with self.assertRaises(ValueError):
            CavuSequenceExtractor(mode="filter").fit_resample(X, y)

    def test_forecast_window_controls_lags_and_rolling_statistics(self):
        frame, meta = weekly_panel()
        X, y, _ = build_forecast_bank(frame, meta, horizon_weeks=4, max_history_weeks=8)
        extractor = CavuSequenceExtractor(
            mode="forecast", horizon_weeks=4, history_window=3, include_lags=True, include_rolling=True
        )
        selected, _ = extractor.fit_resample(X, y)
        lag_columns = [name for name in selected if name.startswith("hist__")]
        self.assertEqual(set(lag_columns), {f"hist__occupancy__lag_{lag}" for lag in (4, 5, 6)})
        expected = X[[f"hist__occupancy__lag_{lag}" for lag in (4, 5, 6)]]
        np.testing.assert_allclose(selected["roll__occupancy__mean"], expected.mean(axis=1))
        np.testing.assert_allclose(selected["roll__occupancy__std"], expected.std(axis=1, ddof=0))
        np.testing.assert_allclose(selected["roll__occupancy__min"], expected.min(axis=1))
        np.testing.assert_allclose(selected["roll__occupancy__max"], expected.max(axis=1))
        rolling_only = clone(extractor).set_params(include_lags=False)
        rolling_X, _ = rolling_only.fit_resample(X, y)
        self.assertFalse(any(name.startswith("hist__") for name in rolling_X.columns))
        self.assertTrue(any(name.startswith("roll__") for name in rolling_X.columns))
        with self.assertRaises(ValueError):
            clone(extractor).set_params(horizon_weeks=1).fit_resample(X, y)


class WrapperAndSearchTests(unittest.TestCase):
    def test_wrapper_clones_components_exposes_controls_and_keeps_prediction_rows(self):
        extractor = CavuSequenceExtractor(mode="filter", filter_method="iqr")
        estimator = DummyRegressor(strategy="mean")
        model = SequenceFilteredRegressor(extractor=extractor, estimator=estimator, scale_features=True)
        copied = clone(model)
        self.assertIsNot(copied.extractor, extractor)
        self.assertIsNot(copied.estimator, estimator)
        self.assertIn("extractor__filter_method", copied.get_params(deep=True))
        self.assertIn("estimator__strategy", copied.get_params(deep=True))
        copied.set_params(extractor__history_window=4, estimator__strategy="mean")
        self.assertEqual(copied.extractor.history_window, 4)
        X = pd.DataFrame({"x": [9, 10, 11, 9, 10, 11, 10, 10000]}, index=np.arange(8) * 5)
        y = pd.Series([1., 2., 3., 4., 5., 6., 7., 99999.], index=X.index)
        copied.fit(X, y)
        self.assertIsNot(copied.extractor_, copied.extractor)
        self.assertIsNot(copied.estimator_, copied.estimator)
        heldout = pd.DataFrame({"x": [np.nan, -1e9, 1e9]}, index=[50, 60, 70])
        predictions = copied.predict(heldout)
        self.assertEqual(predictions.shape, (len(heldout),))
        np.testing.assert_allclose(predictions, np.mean(y.iloc[:-1]))

    def test_chronological_folds_keep_whole_weeks_and_purge_to_origin(self):
        frame, meta = weekly_panel(shuffle=True)
        X, _, bank_meta = build_forecast_bank(frame, meta, horizon_weeks=4, max_history_weeks=4)
        dates = bank_meta.week_start
        folds = chronological_week_splits(dates, horizon_weeks=4, n_splits=3, test_size=5)
        self.assertEqual(len(folds), 3)
        previous_train = set()
        previous_validation = set()
        for train_index, validation_index in folds:
            train_dates, validation_dates = dates.iloc[train_index], dates.iloc[validation_index]
            self.assertTrue(len(train_index) > 0 and len(validation_index) > 0)
            self.assertFalse(set(train_index) & set(validation_index))
            self.assertTrue(previous_train.issubset(set(train_index)))
            self.assertFalse(previous_validation & set(validation_index))
            self.assertLessEqual(train_dates.max(), validation_dates.min() - pd.Timedelta(weeks=4))
            self.assertEqual(validation_dates.nunique(), 5)
            self.assertEqual(set(validation_index), set(np.flatnonzero(dates.isin(validation_dates.unique()))))
            self.assertEqual(set(train_index), set(np.flatnonzero(dates.le(validation_dates.min() - pd.Timedelta(weeks=4)))))
            previous_train = set(train_index)
            previous_validation.update(validation_index)
        self.assertEqual(len(X), len(dates))

    @staticmethod
    def search_inputs():
        frame, meta = weekly_panel(periods=36, shuffle=True)
        X, y, bank_meta = build_forecast_bank(frame, meta, horizon_weeks=2, max_history_weeks=4)
        folds = chronological_week_splits(bank_meta.week_start, horizon_weeks=2, n_splits=2, test_size=4)
        return X, y, folds

    @staticmethod
    def pipeline(estimator):
        return Pipeline([
            ("model", SequenceFilteredRegressor(
                extractor=CavuSequenceExtractor(mode="forecast", horizon_weeks=2, history_window=4),
                estimator=estimator,
                clip_nonnegative=True,
            ))
        ])

    def test_two_candidate_grid_search_tunes_extraction_filter_and_model(self):
        X, y, folds = self.search_inputs()
        candidates = [
            {
                "model__extractor__history_window": [2],
                "model__extractor__include_lags": [True],
                "model__extractor__include_rolling": [False],
                "model__extractor__filter_method": ["none"],
                "model__extractor__select_features": [False],
                "model__estimator__max_depth": [2],
            },
            {
                "model__extractor__history_window": [4],
                "model__extractor__include_lags": [False],
                "model__extractor__include_rolling": [True],
                "model__extractor__filter_method": ["iqr"],
                "model__extractor__filter_threshold": [1.5],
                "model__extractor__select_features": [True],
                "model__extractor__n_features": [3],
                "model__estimator__max_depth": [3],
            },
        ]
        search = GridSearchCV(
            self.pipeline(DecisionTreeRegressor(random_state=42)), candidates,
            cv=folds, scoring="neg_mean_absolute_error", error_score="raise", n_jobs=1,
        ).fit(X, y)
        self.assertEqual(len(search.cv_results_["params"]), 2)
        self.assertTrue(np.isfinite(search.cv_results_["mean_test_score"]).all())
        self.assertEqual(search.predict(X.iloc[:5]).shape, (5,))
        self.assertIn("model__extractor__history_window", search.best_params_)
        self.assertIn("model__extractor__filter_method", search.best_params_)
        self.assertIn("model__estimator__max_depth", search.best_params_)

    @unittest.skipIf(BayesSearchCV is None, "optional xgboost and scikit-optimize packages are required")
    def test_two_iteration_bayesian_search_fits_xgboost_pipeline(self):
        X, y, folds = self.search_inputs()
        estimator = XGBRegressor(
            objective="reg:squarederror", n_estimators=5, max_depth=2,
            learning_rate=0.1, random_state=42, n_jobs=1, verbosity=0,
        )
        search = BayesSearchCV(
            self.pipeline(estimator),
            {
                "model__extractor__history_window": Categorical([2, 4]),
                "model__extractor__filter_method": Categorical(["none", "iqr"]),
                "model__extractor__include_lags": Categorical([True, False]),
                "model__extractor__n_features": Integer(2, 5),
                "model__extractor__select_features": Categorical([True, False]),
                "model__estimator__max_depth": Integer(1, 3),
            },
            n_iter=2, cv=folds, scoring="neg_mean_absolute_error",
            error_score="raise", random_state=42, n_jobs=1,
        ).fit(X, y)
        self.assertEqual(len(search.cv_results_["params"]), 2)
        self.assertTrue(np.isfinite(search.cv_results_["mean_test_score"]).all())
        predictions = search.predict(X.iloc[:7])
        self.assertEqual(predictions.shape, (7,))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())


if __name__ == "__main__":
    unittest.main()
