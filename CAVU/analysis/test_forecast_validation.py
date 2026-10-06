"""Focused tests for forecast-time isolation and pricing comparison semantics."""
import unittest
import numpy as np
import pandas as pd
from forecast_validation import (
    EvaluationConfig, build_features, evaluate_revenue, oracle_price_revenue,
    scenario_revenue, threshold_pricing_policy,
)


class EvaluationTests(unittest.TestCase):
    def panel(self):
        return pd.DataFrame({'terminal': 'T1', 'carpark_name': 'A',
            'week_start': pd.date_range('2023-01-02', periods=24, freq='7D'),
            'occupancy': np.arange(24.) * 10 + 100, 'passengers': np.arange(24.) + 1000,
            'flights': np.arange(24.) + 50})

    def test_future_outcomes_cannot_change_current_features_or_baselines(self):
        for horizon in [1, 4]:
            panel = self.panel()
            original, features, flights = build_features(panel, horizon)
            row = original.loc[original.week_start == pd.Timestamp('2023-04-17')]
            changed = panel.copy()
            changed.loc[changed.week_start > row.origin_week.iloc[0], ['occupancy', 'passengers', 'flights']] = 999999
            modified, _, _ = build_features(changed, horizon)
            other = modified.loc[modified.week_start == row.week_start.iloc[0]]
            pd.testing.assert_frame_equal(row[features + flights], other[features + flights])
            self.assertNotIn('occupancy', features + flights)
            self.assertNotIn('bookings_count', features + flights)

    def test_calendar_boundary_uses_dates_not_week_number(self):
        panel = self.panel()
        panel['week_start'] = pd.date_range('2022-11-07', periods=24, freq='7D')
        frame, _, _ = build_features(panel, 1)
        row = frame.loc[frame.week_start == pd.Timestamp('2023-01-09')].iloc[0]
        self.assertEqual(row.origin_week, pd.Timestamp('2023-01-02'))
        self.assertEqual(row.recent_occupancy, 180.)

    def test_missing_calendar_week_fails(self):
        with self.assertRaises(ValueError):
            build_features(self.panel().drop(index=12), 1)

    def test_policy_threshold_boundaries(self):
        np.testing.assert_allclose(threshold_pricing_policy([49, 50, 85, 86], 100), [.8, 1, 1, 1.3])

    def test_perfect_prediction_is_not_a_price_oracle(self):
        perfect_policy = scenario_revenue(60, threshold_pricing_policy(60, 100), 100)
        inaccurate_policy = scenario_revenue(60, threshold_pricing_policy(40, 100), 100)
        self.assertGreater(inaccurate_policy, perfect_policy)
        self.assertAlmostEqual(perfect_policy, 1200.)
        self.assertAlmostEqual(float(oracle_price_revenue(60, 100)), 1254.7674631095)

    def test_oracle_bounds_all_allowed_actions(self):
        actual = np.arange(151, dtype=float)
        for elasticity in [-.5, -1., -1.2, -1.5, -2.]:
            oracle = oracle_price_revenue(actual, 100, elasticity=elasticity)
            for price in [.8, 1., 1.3]:
                self.assertTrue(np.all(oracle + 1e-8 >= scenario_revenue(actual, price, 100, elasticity=elasticity)))

    def test_unit_elasticity_is_price_invariant_when_unconstrained(self):
        for multiplier in [.8, 1., 1.3]:
            self.assertAlmostEqual(float(scenario_revenue(20, multiplier, 100, elasticity=-1.)), 400.)

    def test_invalid_inputs_fail_instead_of_becoming_hidden_fallbacks(self):
        for actual, capacity in [(np.nan, 100), (-1, 100), (20, 0), (20, np.inf)]:
            with self.assertRaises(ValueError):
                scenario_revenue(actual, 1, capacity)

    def test_identical_forecasts_have_identical_revenue(self):
        predictions = pd.DataFrame({'terminal': ['T1'], 'carpark_name': ['A'],
            'target_week': [pd.Timestamp('2023-06-05')], 'issue_date': [pd.Timestamp('2023-06-05')],
            'horizon_weeks': [1], 'actual_space_days': [60.], 'capacity_proxy_space_days': [100.],
            'recent_mean_prediction': [40.], 'last_week_prediction': [40.], 'model_prediction': [40.]})
        ledger, summary = evaluate_revenue(predictions, EvaluationConfig())
        rf = summary.loc[summary.method == 'Selected Random Forest']
        np.testing.assert_allclose(rf.uplift_gbp, 0)
        self.assertTrue((ledger.regret_to_price_oracle >= -1e-8).all())
        self.assertTrue((ledger.signed_gap_to_perfect_same_policy < 0).any())


if __name__ == '__main__':
    unittest.main()
