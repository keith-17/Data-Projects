"""Forward evaluation and an explicitly assumed pricing scenario.

Consumes the existing expanded-booking cache without changing its preparation.
The pricing function is a replaceable boundary for a separately owned policy.
No actual operator forecasts or production pricing model are supplied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

KEYS = ['terminal', 'carpark_name']
RULES = ['Recent 4-week mean', 'Last observed week']


@dataclass(frozen=True)
class EvaluationConfig:
    horizons: tuple = (1, 4)
    test_weeks: int = 16
    seed: int = 42
    trees: int = 150
    reference_price_per_space_day: float = 20.0
    elasticities: tuple = (-0.5, -1.0, -1.2, -1.5, -2.0)
    capacity_scales: tuple = (1.0, 1.25, 1.5)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_reference_panel(expanded_bookings, flights_csv):
    """Read target counts from the user's existing cache; read flights for past features.

    Zero rows in the cache are treated as zero occupied space-days, not independently
    verified observations. Complete Monday-Sunday weeks only; no pooled week numbers.
    Realized passenger/flight counts are used only after their week has ended, assuming
    immediate availability then. Production reporting delays need a further lag.
    """
    flights = pd.read_csv(flights_csv)
    required = {'flight_date', 'terminal_name', 'pax_quantity', 'flight_number_full_icao'}
    if not required.issubset(flights):
        raise ValueError(f'Missing flight columns: {required - set(flights)}')
    flights['date'] = pd.to_datetime(flights.flight_date, errors='raise').dt.normalize()
    start, end = flights.date.min(), flights.date.max()
    daily_parts = []
    for chunk in pd.read_csv(expanded_bookings, usecols=KEYS + ['date_occupied'], chunksize=250_000):
        chunk['date'] = pd.to_datetime(chunk.date_occupied, errors='raise').dt.normalize()
        chunk = chunk.loc[chunk.date.between(start, end)]
        daily_parts.append(chunk.groupby(KEYS + ['date']).size().rename('occupancy').reset_index())
    daily = pd.concat(daily_parts, ignore_index=True).groupby(KEYS + ['date']).occupancy.sum().reset_index()
    if daily.empty:
        raise ValueError('No booking observations overlap the flight dates.')
    first_monday = start + pd.Timedelta(days=(7 - start.weekday()) % 7)
    last_monday = end - pd.Timedelta(days=end.weekday())
    if last_monday + pd.Timedelta(days=6) > end:
        last_monday -= pd.Timedelta(weeks=1)
    weeks = pd.date_range(first_monday, last_monday, freq='7D')
    daily['week_start'] = daily.date - pd.to_timedelta(daily.date.dt.weekday, unit='D')
    weekly_counts = daily.groupby(KEYS + ['week_start']).occupancy.sum()
    pairs = daily[KEYS].drop_duplicates().sort_values(KEYS)
    index = pd.MultiIndex.from_tuples(
        [(t, p, w) for t, p in pairs.itertuples(index=False, name=None) for w in weeks],
        names=KEYS + ['week_start'],
    )
    panel = weekly_counts.reindex(index, fill_value=0).rename('occupancy').reset_index()
    flights['week_start'] = flights.date - pd.to_timedelta(flights.date.dt.weekday, unit='D')
    flight_daily = flights.dropna(subset=['terminal_name']).groupby(['terminal_name', 'date', 'week_start']).agg(
        passengers=('pax_quantity', 'sum'), flights=('flight_number_full_icao', 'nunique')
    ).reset_index()
    fw = flight_daily.groupby(['terminal_name', 'week_start'])[['passengers', 'flights']].sum().reset_index()
    panel = panel.merge(fw, left_on=['terminal', 'week_start'], right_on=['terminal_name', 'week_start'], how='left', validate='many_to_one')
    panel[['passengers', 'flights']] = panel[['passengers', 'flights']].fillna(0)
    return panel.drop(columns='terminal_name'), daily, {
        'flight_rows': len(flights), 'flight_start': str(start.date()), 'flight_end': str(end.date()),
        'missing_terminal_flight_rows': int(flights.terminal_name.isna().sum()),
        'complete_weeks': len(weeks), 'target_unit': 'occupied space-days per car park per week',
        'target_provenance': 'Existing expanded-booking cache; upstream preparation inherited, not re-audited.',
        'zero_assumption': 'Absence of cache rows is treated as zero occupancy; operational completeness not established.',
    }


def build_features(panel, horizon):
    """Features for target week t use observations no later than week t-horizon."""
    if horizon < 1:
        raise ValueError('horizon must be at least one week')
    pieces = []
    history = ['recent_occupancy', 'previous_occupancy', 'four_weeks_back', 'mean_4', 'mean_8', 'std_4', 'recent_change']
    flight_features = ['recent_passengers', 'mean_passengers_4', 'recent_flights', 'mean_flights_4']
    for _, group in panel.groupby(KEYS, sort=True):
        group = group.sort_values('week_start').copy()
        if not (group.week_start.diff().dropna() == pd.Timedelta(weeks=1)).all():
            raise ValueError('Panel must contain a continuous weekly calendar for each park.')
        known = group.occupancy.shift(horizon)
        group['recent_occupancy'] = known
        group['previous_occupancy'] = group.occupancy.shift(horizon + 1)
        group['four_weeks_back'] = group.occupancy.shift(horizon + 3)
        group['mean_4'] = known.rolling(4, min_periods=4).mean()
        group['mean_8'] = known.rolling(8, min_periods=8).mean()
        group['std_4'] = known.rolling(4, min_periods=4).std(ddof=0)
        group['recent_change'] = group.recent_occupancy - group.previous_occupancy
        for feature in ['passengers', 'flights']:
            observed = group[feature].shift(horizon)
            group[f'recent_{feature}'] = observed
            group[f'mean_{feature}_4'] = observed.rolling(4, min_periods=4).mean()
        group['origin_week'] = group.week_start - pd.Timedelta(weeks=horizon)
        group['issue_date'] = group.origin_week + pd.Timedelta(weeks=1)
        phase = 2 * np.pi * group.week_start.dt.dayofyear / 365.25
        group['annual_sin'], group['annual_cos'] = np.sin(phase), np.cos(phase)
        pieces.append(group)
    frame = pd.concat(pieces, ignore_index=True).dropna(subset=history + flight_features)
    return frame, KEYS + history + ['annual_sin', 'annual_cos'], flight_features


def model_pipeline(features, parameters, config):
    preprocess = ColumnTransformer([
        ('park', OneHotEncoder(handle_unknown='ignore', sparse_output=False), KEYS),
        ('numeric', 'passthrough', [c for c in features if c not in KEYS]),
    ])
    return Pipeline([
        ('features', preprocess),
        ('model', RandomForestRegressor(n_estimators=config.trees, random_state=config.seed, n_jobs=2, **parameters)),
    ])


def select_model(frame, history_features, flight_features, first_test_week, horizon, config):
    """Select RF configuration and flight ablation using only pre-evaluation outcomes."""
    cutoff = first_test_week - pd.Timedelta(weeks=horizon)
    development_weeks = sorted(frame.loc[frame.week_start <= cutoff, 'week_start'].unique())
    if len(development_weeks) < 16:
        raise ValueError('At least 16 usable development weeks required.')
    validation_weeks = [pd.Timestamp(development_weeks[i]) for i in [-9, -5, -1]]
    rows = []
    for include_flights in [False, True]:
        features = history_features + (flight_features if include_flights else [])
        for depth in [8, None]:
            for leaf in [2, 5]:
                params = {'max_depth': depth, 'min_samples_leaf': leaf, 'max_features': 1.0}
                errors = []
                for week in validation_weeks:
                    origin = week - pd.Timedelta(weeks=horizon)
                    train, valid = frame.loc[frame.week_start <= origin], frame.loc[frame.week_start == week]
                    if len(train) == 0 or len(valid) == 0:
                        raise ValueError('Empty chronological validation split.')
                    model = model_pipeline(features, params, config).fit(train[features], train.occupancy)
                    errors.extend(np.abs(valid.occupancy.to_numpy() - model.predict(valid[features])))
                rows.append({'include_flights': include_flights, **params, 'development_mae': float(np.mean(errors))})
    winner = min(rows, key=lambda row: row['development_mae'])
    return winner, rows, validation_weeks, cutoff


def threshold_pricing_policy(predicted_space_days, capacity_space_days):
    """Frozen illustrative policy, NOT a supplied production pricing model.

    Replace this function with the other team's policy adapter when available.
    That adapter must receive forecast-time information only and apply identically
    to every forecast, including perfect information. Do not tune on holdout revenue.
    """
    predicted, capacity = _validated(predicted_space_days, capacity_space_days)
    rate = predicted / capacity
    return np.where(rate > .85, 1.30, np.where(rate < .50, .80, 1.00))


def _validated(values, capacity):
    values, capacity = np.broadcast_arrays(np.asarray(values, float), np.asarray(capacity, float))
    if not np.isfinite(values).all() or not np.isfinite(capacity).all() or (values < 0).any() or (capacity <= 0).any():
        raise ValueError('Values must be finite and nonnegative; capacities finite and positive.')
    return values, capacity


def scenario_revenue(reference_demand, multipliers, capacity, reference_price=20., elasticity=-1.2):
    """Assume observed space-days are unconstrained demand at reference_price.

    Demand response is an imposed constant elasticity, not a causal estimate.
    Weekly aggregation does not model daily peaks, stay overlap, displacement,
    existing reservations, costs or competitor responses. Output is revenue, not profit.
    """
    actual, capacity = _validated(reference_demand, capacity)
    multipliers = np.broadcast_to(np.asarray(multipliers, float), actual.shape)
    if not np.isfinite(multipliers).all() or (multipliers <= 0).any() or not np.isfinite(reference_price) or reference_price <= 0 or not np.isfinite(elasticity):
        raise ValueError('Price and multipliers must be finite and positive, elasticity finite.')
    demand = actual * multipliers ** elasticity
    return np.minimum(demand, capacity) * reference_price * multipliers


def oracle_price_revenue(reference_demand, capacity, reference_price=20., elasticity=-1.2):
    candidates = [scenario_revenue(reference_demand, m, capacity, reference_price, elasticity) for m in [.8, 1., 1.3]]
    return np.max(candidates, axis=0)


def evaluate_revenue(predictions, config, pricing_policy=threshold_pricing_policy):
    """Paired comparison; shared policy, demand response, capacity and evaluation rows."""
    records = []
    for horizon, rows in predictions.groupby('horizon_weeks'):
        for scale in config.capacity_scales:
            cap = rows.capacity_proxy_space_days.to_numpy() * scale
            for elasticity in config.elasticities:
                actual = rows.actual_space_days.to_numpy()
                perfect_actions = pricing_policy(actual, cap)
                perfect = scenario_revenue(actual, perfect_actions, cap, config.reference_price_per_space_day, elasticity)
                oracle = oracle_price_revenue(actual, cap, config.reference_price_per_space_day, elasticity)
                for method, column in [('Recent 4-week mean', 'recent_mean_prediction'), ('Last observed week', 'last_week_prediction'), ('Selected Random Forest', 'model_prediction'), ('Perfect information, same policy', None), ('Always reference price', 'constant')]:
                    actions = perfect_actions if column is None else np.ones(len(rows)) if column == 'constant' else pricing_policy(rows[column].to_numpy(), cap)
                    revenue = scenario_revenue(actual, actions, cap, config.reference_price_per_space_day, elasticity)
                    part = rows[KEYS + ['target_week', 'issue_date']].copy()
                    part['horizon_weeks'], part['capacity_scale'], part['elasticity'], part['method'] = horizon, scale, elasticity, method
                    part['price_multiplier'], part['revenue'] = actions, revenue
                    part['same_policy_perfect_revenue'], part['price_oracle_revenue'] = perfect, oracle
                    part['signed_gap_to_perfect_same_policy'] = perfect - revenue
                    part['regret_to_price_oracle'] = oracle - revenue
                    part['action_agreement_with_perfect'] = actions == perfect_actions
                    if (oracle + 1e-7 < revenue).any():
                        raise AssertionError('Oracle bound failed; policy uses prices outside the oracle action set.')
                    records.append(part)
    ledger = pd.concat(records, ignore_index=True)
    summary = ledger.groupby(['horizon_weeks', 'capacity_scale', 'elasticity', 'method']).agg(
        revenue=('revenue', 'sum'), same_policy_perfect_revenue=('same_policy_perfect_revenue', 'sum'),
        price_oracle_revenue=('price_oracle_revenue', 'sum'),
        signed_gap_to_perfect_same_policy=('signed_gap_to_perfect_same_policy', 'sum'),
        regret_to_price_oracle=('regret_to_price_oracle', 'sum'),
        action_agreement=('action_agreement_with_perfect', 'mean'),
        evaluated_park_weeks=('revenue', 'size'),
    ).reset_index()
    baseline = summary.loc[summary.method == RULES[0], ['horizon_weeks', 'capacity_scale', 'elasticity', 'revenue']].rename(columns={'revenue': 'baseline_revenue'})
    summary = summary.merge(baseline, on=['horizon_weeks', 'capacity_scale', 'elasticity'], validate='many_to_one')
    summary['uplift_gbp'] = summary.revenue - summary.baseline_revenue
    summary['uplift_pct'] = np.where(summary.baseline_revenue > 0, 100 * summary.uplift_gbp / summary.baseline_revenue, np.nan)
    return ledger, summary


def run_evaluation(panel, daily, output_dir, config=EvaluationConfig(), source_paths=(), provenance=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weeks = sorted(panel.week_start.unique())
    test_weeks = [pd.Timestamp(w) for w in weeks[-config.test_weeks:]]
    cohort_cutoff = test_weeks[0] - pd.Timedelta(weeks=max(config.horizons))
    eligible = panel.loc[panel.week_start <= cohort_cutoff].groupby(KEYS).occupancy.agg(lambda x: (x > 0).sum())
    eligible = eligible[eligible >= 8].index
    panel = panel.set_index(KEYS).loc[eligible].reset_index()
    selection, predictions = {}, []
    for horizon in config.horizons:
        frame, history_features, flight_features = build_features(panel, horizon)
        winner, trials, validation_weeks, cutoff = select_model(frame, history_features, flight_features, test_weeks[0], horizon, config)
        features = history_features + (flight_features if winner['include_flights'] else [])
        parameters = {k: winner[k] for k in ['max_depth', 'min_samples_leaf', 'max_features']}
        selection[str(horizon)] = {'winner': winner, 'trials': trials, 'features': features, 'validation_weeks': [str(w.date()) for w in validation_weeks], 'selection_latest_outcome_week': str(cutoff.date())}
        print(f'Horizon {horizon}: selected {winner}', flush=True)
        frozen_capacity = daily.loc[daily.date < cutoff + pd.Timedelta(weeks=1)].groupby(KEYS).occupancy.max() * 7
        for target_week in test_weeks:
            origin = target_week - pd.Timedelta(weeks=horizon)
            train = frame.loc[frame.week_start <= origin]
            test = frame.loc[frame.week_start == target_week]
            if test.empty or len(test) != len(eligible):
                raise ValueError('Incomplete evaluation cohort; do not silently drop predictions.')
            if not (train.week_start.max() <= origin and test.origin_week.eq(origin).all()):
                raise AssertionError('Forecast-time information boundary failed.')
            model = model_pipeline(features, parameters, config).fit(train[features], train.occupancy)
            prediction = np.maximum(0., model.predict(test[features]))
            result = test[KEYS + ['week_start', 'issue_date', 'origin_week']].rename(columns={'week_start': 'target_week'}).copy()
            result['horizon_weeks'] = horizon
            result['actual_space_days'] = test.occupancy.to_numpy()
            result['recent_mean_prediction'] = test.mean_4.to_numpy()
            result['last_week_prediction'] = test.recent_occupancy.to_numpy()
            result['model_prediction'] = prediction
            result['capacity_proxy_space_days'] = frozen_capacity.reindex(pd.MultiIndex.from_frame(test[KEYS])).to_numpy()
            result['training_last_target_week'] = train.week_start.max()
            result['training_rows'] = len(train)
            if result.isna().any().any():
                raise ValueError('Missing forecast, capacity or audit field.')
            predictions.append(result)
    predictions = pd.concat(predictions, ignore_index=True)
    metrics = []
    columns = [(RULES[0], 'recent_mean_prediction'), (RULES[1], 'last_week_prediction'), ('Selected Random Forest', 'model_prediction')]
    for horizon, rows in predictions.groupby('horizon_weeks'):
        for method, col in columns:
            error = rows[col] - rows.actual_space_days
            metrics.append({'horizon_weeks': horizon, 'method': method, 'mae': mean_absolute_error(rows.actual_space_days, rows[col]), 'rmse': float(np.sqrt(mean_squared_error(rows.actual_space_days, rows[col]))), 'r2': r2_score(rows.actual_space_days, rows[col]), 'wape_pct': 100 * error.abs().sum() / rows.actual_space_days.sum(), 'bias': error.mean(), 'p95_absolute_error': error.abs().quantile(.95), 'park_weeks': len(rows)})
    metrics = pd.DataFrame(metrics)
    revenue, revenue_summary = evaluate_revenue(predictions, config)
    failures = predictions.copy()
    failures['model_absolute_error'] = (failures.model_prediction - failures.actual_space_days).abs()
    failures['recent_mean_absolute_error'] = (failures.recent_mean_prediction - failures.actual_space_days).abs()
    failures['last_week_absolute_error'] = (failures.last_week_prediction - failures.actual_space_days).abs()
    failures.sort_values(['horizon_weeks', 'model_absolute_error'], ascending=[True, False]).to_csv(output / 'failure_cases.csv', index=False)
    failures.groupby(['horizon_weeks', 'terminal'])[['model_absolute_error', 'recent_mean_absolute_error', 'last_week_absolute_error']].mean().to_csv(output / 'terminal_mae.csv')
    predictions.to_csv(output / 'predictions.csv', index=False)
    metrics.to_csv(output / 'forecast_metrics.csv', index=False)
    revenue.to_csv(output / 'revenue_ledger.csv', index=False)
    revenue_summary.to_csv(output / 'revenue_sensitivity.csv', index=False)
    panel.to_csv(output / 'reference_weekly_panel.csv', index=False)
    manifest = {'config': asdict(config), 'selection': selection, 'provenance': provenance, 'cohort': [list(key) for key in eligible], 'cohort_latest_outcome_week': str(cohort_cutoff.date()), 'test_start': str(test_weeks[0].date()), 'test_end': str(test_weeks[-1].date()), 'python': platform.python_version(), 'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib']}, 'input_hashes': {str(p): sha256(p) for p in source_paths}, 'code_sha256': sha256(__file__), 'limitations': ['Retrospective re-evaluation of previously explored data, not a fresh prospective holdout.', 'Existing target cache and its upstream selection are inherited.', 'Past realized flight information assumed available immediately after week end.', 'No observed human forecasts, production pricing policy, measured physical capacity or estimated demand elasticity.', 'Capacity proxy frozen before evaluation; observed demand assumed at GBP20 per occupied space-day.', 'Reference threshold policy may earn less with perfect information; gaps are signed.', 'Weekly scenario omits daily capacity, booking inventory, costs and substitution.', 'Each horizon evaluated separately; never sum horizons as incremental revenue.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    make_plots(predictions, revenue, revenue_summary, output)
    return metrics, revenue_summary, manifest


def make_plots(predictions, revenue, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for horizon, rows in predictions.groupby('horizon_weeks'):
        totals = rows.groupby('target_week')[['actual_space_days', 'recent_mean_prediction', 'last_week_prediction', 'model_prediction']].sum()
        fig, ax = plt.subplots(figsize=(11, 5))
        for col, label in [('actual_space_days', 'Observed'), ('recent_mean_prediction', RULES[0]), ('last_week_prediction', RULES[1]), ('model_prediction', 'Selected Random Forest')]:
            ax.plot(totals.index, totals[col], marker='o', markersize=3, label=label)
        ax.set(title=f'{horizon}-week horizon: evaluated weeks only', ylabel='Occupied space-days across evaluated parks', xlabel='Target week')
        ax.legend(); ax.grid(alpha=.2); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(output / f'forecast_h{horizon}.png', dpi=160); plt.close(fig)
    rf = summary.loc[summary.method == 'Selected Random Forest']
    fig, axes = plt.subplots(1, len(rf.horizon_weeks.unique()), figsize=(11, 4), squeeze=False)
    for ax, (horizon, group) in zip(axes.flat, rf.groupby('horizon_weeks')):
        for scale, scenario in group.groupby('capacity_scale'):
            ax.plot(scenario.elasticity, scenario.uplift_pct, marker='o', label=f'Capacity proxy x{scale:g}')
        ax.axhline(0, color='black', linewidth=.8)
        ax.set(title=f'{horizon}-week horizon', xlabel='Assumed elasticity', ylabel='Revenue uplift vs recent mean (%)')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Scenario sensitivity; fixed illustrative pricing policy')
    fig.tight_layout(); fig.savefig(output / 'revenue_sensitivity.png', dpi=160); plt.close(fig)
    chosen = revenue.loc[(revenue.elasticity == -1.2) & (revenue.capacity_scale == 1.) & (revenue.horizon_weeks == 1)]
    weekly = chosen.groupby(['target_week', 'method']).revenue.sum().unstack()
    fig, ax = plt.subplots(figsize=(11, 5))
    for method in [RULES[0], RULES[1], 'Selected Random Forest', 'Perfect information, same policy', 'Always reference price']:
        ax.plot(weekly.index, weekly[method].cumsum(), label=method)
    ax.set(title='Cumulative scenario revenue over evaluated weeks', ylabel='Scenario revenue (GBP)', xlabel='Target week')
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(output / 'revenue_cumulative.png', dpi=160); plt.close(fig)
    gaps = chosen.groupby('method').signed_gap_to_perfect_same_policy.sum().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(gaps.index, gaps.values / 1000)
    ax.axvline(0, color='black', linewidth=.8)
    ax.set(title='Perfect information through the same policy: signed revenue gap', xlabel='Gap (GBP thousands); negative means more revenue than perfect-information policy')
    ax.grid(axis='x', alpha=.2); fig.tight_layout()
    fig.savefig(output / 'same_policy_gap.png', dpi=160); plt.close(fig)
