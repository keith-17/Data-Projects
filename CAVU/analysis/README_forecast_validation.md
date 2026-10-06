# Forecast and revenue validation

Use `validated_forecast_and_revenue.ipynb` for the revised evaluation. The original `analysis.ipynb` and `general_utils.py` have not been edited by this revision; their existing local edits and older outputs remain in place.

The new notebook reads `temp_csv/final_bookings_df.csv` (the existing expanded-booking cache) and `../raw_data/flights_sample.csv`. It does not rebuild or review the upstream booking preparation. Results inherit the coverage and selection assumptions of that cache. Rebuilding that cache is outside this change's scope.

## Reproduce

Open a Python environment with the packages in `requirements_forecast_validation.txt`, open the notebook with this analysis directory as the working directory, and run its cells in order. A notebook frontend normally supplies IPython for display. The core module can also run without a notebook:

```python
from pathlib import Path
from forecast_validation import EvaluationConfig, load_reference_panel, run_evaluation
root = Path.cwd()
cache = root / 'temp_csv' / 'final_bookings_df.csv'
flights = root.parent / 'raw_data' / 'flights_sample.csv'
panel, daily, provenance = load_reference_panel(cache, flights)
metrics, revenue, manifest = run_evaluation(
    panel, daily, root / 'reports' / 'validated_forecast_v1',
    EvaluationConfig(), source_paths=[cache, flights], provenance=provenance,
)
```

Run the focused tests with `python -m unittest test_forecast_validation -v` from this directory.

## Fixed experimental choices

- Complete Monday-Sunday weeks, identified by dates rather than pooled week numbers.
- Last 16 weeks as the retrospective evaluation period; next-week and four-week target horizons reported separately.
- Fixed cohort chosen using earlier history: at least 8 positive weeks by the earliest evaluation origin.
- Forecast inputs use completed weeks only. Actual reporting delays are not provided; immediate availability at week end is an explicit assumption.
- Recent four-week mean as the primary planning rule; last observed week as a stronger responsiveness check. Neither represents observed human judgment.
- Random Forest configurations and inclusion of past-flight features selected on development MAE at three earlier origins. No evaluation-period revenue tuning.
- Hyperparameters fixed during evaluation. Training expands as outcomes become available.
- Reused historical data: this is not a fresh prospective holdout and no statistical significance claim is made.

## Interface to the pricing team

`threshold_pricing_policy(forecast, capacity)` is a frozen illustrative policy. No production pricing model has been supplied. Replace this boundary with the other team's policy, keeping its parameters and forecast-time inputs identical across forecasting methods. If its available price actions differ, update the oracle's action set as well.

Do not retrain or optimise the pricing policy to favour a particular forecast on evaluation weeks. All demand-response scenarios and capacities are shared across methods within a horizon.

Report separately:

1. Predictive error.
2. Revenue under the same fixed policy.
3. Perfect information through that same policy, with a **signed** gap.
4. The additional price-optimising oracle, with nonnegative regret within its assumed world.
5. A constant reference-price comparator.

A policy can be suboptimal even with perfect information. Negative signed gaps are possible and must not be clipped to zero or presented as implementation errors.

## Meaning of the scenario

The target and capacity are weekly space-days. The price is GBP20 per space-day. Observed occupancy is treated as unconstrained reference-price demand, an unverified assumption. Capacity is seven times the greatest observed daily occupancy available before the first evaluation forecast, held fixed, with sensitivity scales. It is not measured physical capacity.

The scenario omits daily capacity peaks, booking lead times and inventory already sold, overlapping stays, customer substitution, costs and competitor responses. Elasticity is imposed, not estimated. Outputs are conditional simulated revenue, not observed revenue, savings, profit or an estimate of causal deployment impact.

## Results and audit files

`reports/validated_forecast_v1/` contains the predictions, forecast metrics, revenue ledger, sensitivity table, failure cases, terminal error summary, charts and manifest. The manifest records parameters, feature choices, versions and hashes of the inputs and code.

The notebook embeds plots. The generated report directory is already excluded by the repository's existing ignore rules. The reported revenues cover 16 evaluated weeks at each horizon. Do not annualise them or add horizons together.

The current run passed 9 focused tests. Forecast-time fields, matching cohorts, finite predictions and unique keys were checked. Re-fitting the selected model at the first evaluation origin reproduced predictions at both horizons. Only presentation and failure-table additions followed the numerical run; the manifest records the original and final code hashes.

## Conclusion supported by this run

At one week, Random Forest improves on the four-week mean but loses to last week's occupancy. At four weeks, it loses to both rules. The one-week default scenario improves revenue against the four-week mean but also loses to last week's rule, and uplift changes sign under elasticity sensitivity. A constant price beats the illustrative perfect-information threshold policy in that scenario. These results support careful baseline and policy evaluation, not general superiority to operators or readiness for production.
