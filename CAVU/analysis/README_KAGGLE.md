# Run the checkpoint-based CAVU notebook on Kaggle

## Upload these two files

1. Create a **private Kaggle Dataset** from `kaggle_inputs.zip` (1.67 MB). The archive contains the data checkpoints, flights CSV and Python helpers. Keep the files together so the manifest can find them.
2. Import `analysis_kaggle.ipynb` into a Kaggle Notebook, then attach that Dataset using the notebook's input panel.
3. Run from the top. If the setup cell needs to install missing packages, enable Internet in the notebook settings. CPU is sufficient to run the current configuration; the existing XGBoost settings use CPU even if a GPU is selected.

The first cell locates the dataset automatically. If several copies are attached, set `DATA_ROOT` to the required `/kaggle/input/...` folder.

## What is reused

- The existing `temp_csv/final_bookings_df.csv` contains 4,072,417 already-expanded booking-day rows. It was aggregated locally into 9,704 daily rows in `daily_bookings_checkpoint.csv`.
- The chunked aggregation preserves the notebook's counts, sums, means and minima. Means use their non-missing counts as weights across chunks. The first 500,000 rows were checked against the original direct aggregation, and the full occupancy total reconciles to the checkpoint row count.
- `booking_weekly_checkpoint.csv` contains the reservation counts needed by the bookings-versus-flights chart. It was prepared once locally using the original confirmation/date/terminal-recovery rules before car-park/day expansion. Raw reservation records are not included in the upload.
- `flights_sample.csv`, `general_utils.py` and `xgboost_sequence.py` are included.
- The 2.66 GB hourly CSV and its weekly partitions are not required by this notebook's weekly demand workflow.

The Kaggle copy starts from these checkpoints and then runs the lightweight flight-feature, join and weekly-aggregation steps. It does not rerun booking cleaning, terminal recovery, melt or explode. Raw-record inspection cells remain in the original local notebook.

## Settings and outputs

Your current model flags, XGBoost search spaces, 50 Bayesian iterations and verbosity controls were retained. The original `analysis.ipynb` was not edited for this conversion. The Kaggle copy includes small compatibility adjustments for holiday date matching, DataFrame formatting and the ARIMA input index. Those preserve the underlying input values and ordering.

Outputs are written under `/kaggle/working/cavu_outputs/`:

- `weekly_df.csv` and `weekly_meta.csv` preserve the row index needed for forecast dates.
- `reports/` contains the Excel report.
- Setting `xgb_save_results=True` also exports XGBoost search results, holdout predictions and metrics.

Use **Save Version / Save & Run All** when you want a saved execution and downloadable outputs. Input datasets are read-only; generated files are written to the working directory. [Kaggle staff explanation of input/output directories](https://www.kaggle.com/general/223073), [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks).

## Validation scope

The notebook passed a local execution check across its 48 code cells, including data loading, feature preparation, ARIMA, Random Forest, PCA/SVM, a short Bayesian XGBoost run, diagnostics, both revenue sections and Excel export. Test-only controls reduced XGBoost to two search iterations and disabled the TensorFlow neural-network branch. The delivered notebook retains the original flags and full configured search budget.

The generated model frame contained 814 weekly rows and the joined daily frame contained 5,390 rows. Bundle hashes are recorded in `cavu_checkpoint_manifest.json` and checked by the setup cell. This was a local execution test, not a hosted Kaggle run; the TensorFlow training branch has not been tested in this conversion.
