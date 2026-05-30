Round 2 — Target & gap calculation details

Objective
- Define the real target `Potential = true demand ceiling (latent)` as a practical proxy: `max_monthly_volume_per_outlet`.

Datasets used
- `data/silver/cleaned/transactions_cleaned.parquet` — transactional rows with at least `outlet_id`, `date`, `volume_liters`.
- `data/gold/outlet_features.parquet` — feature table produced by `src/gold/build_outlet_features.py`, contains `avg_monthly_volume` (see caveat below).

Computed fields (what I created)

1) `monthly_volume` (intermediate)
- How: aggregate transactions into monthly totals per outlet.
- Pandas steps used:

```python
txn_df['date'] = pd.to_datetime(txn_df['date'])
txn_df['month'] = txn_df['date'].dt.to_period('M')
monthly = (
    txn_df
    .groupby(['outlet_id','month'], observed=True)['volume_liters']
    .sum()
    .reset_index()
    .rename(columns={'volume_liters': 'monthly_volume'})
)
```

2) `max_monthly_volume` (the `Potential` target)
- How: for each outlet take the maximum value of `monthly_volume` across observed months.
- Pandas:

```python
potential = (
    monthly
    .groupby('outlet_id', observed=True)['monthly_volume']
    .max()
    .reset_index()
    .rename(columns={'monthly_volume':'max_monthly_volume'})
)
```

3) `actual_avg_monthly_volume`
- Source: the pipeline's `avg_monthly_volume` column from `data/gold/outlet_features.parquet`.
- Important caveat: in the current `src/gold/build_outlet_features.py` the column `avg_monthly_volume` is produced by a simple `.agg(avg_monthly_volume=('volume_liters','mean'))` on `txn_df.groupby('outlet_id')`. That computes the mean transaction `volume_liters` per outlet, not the mean of monthly totals. It therefore may under- or over-estimate a true monthly average depending on transaction frequency.
- Recommendation: replace or augment with a true monthly average computed as `monthly.groupby('outlet_id')['monthly_volume'].mean()` for consistency with `max_monthly_volume`.

4) `gap_volume` and `gap_pct`
- Formulas used:

```text
gap_volume = max_monthly_volume - actual_avg_monthly_volume
gap_pct = gap_volume / (actual_avg_monthly_volume + 1e-9)
```

- Notes: `gap_volume` can be negative if `actual_avg_monthly_volume` > `max_monthly_volume` (e.g., if recent months have spikes). We left negatives as-is for diagnostics; optionally clip to zero to represent only positive upside.

Saved artifact
- `data/models/outlet_potential_target.parquet` — contains at least these columns:
  - `outlet_id`, `max_monthly_volume`, `actual_avg_monthly_volume`, `gap_volume`, `gap_pct`

Operational wiring updated
- `run_pipeline.py` now executes the feature build and scoring chain in dependency order:
  - `src/gold/feature_poi.py`
  - `src/gold/build_outlet_features.py`
  - `src/models/build_potential_target.py`
  - `src/models/cluster_profiles.py`
  - `src/models/opportunity_recommendations.py`
- The dashboard at `output/dashboard/app.py` reads the refreshed artifacts and now has room for the new POI / mobility feature layer.

Next recommended changes
- Recompute `actual_avg_monthly_volume` as true monthly means and re-run the build so `gap` aligns with the `max_monthly_volume` definition.
- Add a timestamp and provenance (dataset hash, pipeline run id) into the saved Parquet metadata or a companion CSV for reproducibility.

If you want, I will: (a) update `src/gold/build_outlet_features.py` to write a `true_avg_monthly_volume` column computed from the monthly aggregation, (b) re-run the pipeline and update downstream artifacts and the dashboard.

---

Update — 2026-05-30
- Implemented true-monthly-average fix in `src/gold/build_outlet_features.py`:
  - Compute monthly totals per outlet and `true_avg_monthly_volume = monthly.groupby('outlet_id')['monthly_volume'].mean()`.
  - Merge into `txn_features` and set `avg_monthly_volume` to that active-month mean (falls back to the previous transaction-mean when missing).
  - Added `avg_monthly_volume_calendar = total_volume / total_months_in_dataset` as a calendar-month baseline.
- Rebuilt and validated artifacts:
  - Rebuilt `data/gold/outlet_features.parquet` (now contains `avg_monthly_volume`, `avg_monthly_volume_calendar`, and month-based `active_months`).
  - Recomputed `data/models/outlet_potential_target.parquet` and `data/models/opportunity_recommendations.parquet` to pick up the corrected averages.
  - Spot-checked `OUT_00001`: `transaction_mean=64.23`, `avg_monthly_volume=533.42` (active-month mean), `avg_monthly_volume_calendar=340.80`.
- Notes and recommendation:
  - `avg_monthly_volume` is the mean over active months (months with transactions). If you prefer calendar-month averages (including months with zero sales), use `avg_monthly_volume_calendar` for gap calculations — this will typically increase estimated gaps for outlets with many inactive months.
  - Ensure `max_monthly_volume` is not accidentally included in `outlet_features` used for training (avoid leakage).
  - If you want I can switch the gap calculation to use `avg_monthly_volume_calendar` and re-run scoring and dashboard.

Leakage audit and dual-model rerun — 2026-05-30

- What I rechecked:
  - `avg_to_max_ratio` is directly target-derived because it is computed as `avg_monthly_volume / max_monthly_volume`.
  - That means it can leak the answer into training, so it was excluded from the new training pass.

- What I reran:
  - `run_pipeline.py --stage gold` to refresh `src/gold/feature_poi.py`, `src/gold/build_outlet_features.py`, and `src/models/build_potential_target.py`.
  - `run_pipeline.py --stage predict` to refresh `src/models/cluster_profiles.py` and `src/models/opportunity_recommendations.py` on the rebuilt gold data.
  - `src/models/train_dual_potential_models.py` to train two separate model versions on the refreshed artifacts.

- Version A — Peak Sales model:
  - Purpose: keep the current peak-sales objective, but remove the direct leakage feature `avg_to_max_ratio`.
  - Hold-out performance: Blend RMSE **218.31**, R² **0.7933**.
  - Saved outputs:
    - `data/models/peak_sales_predictions.parquet`
    - `models/peak_sales_lgbm_model.txt`
    - `models/peak_sales_feature_importance.csv`

- Version B — Market Capacity model:
  - Purpose: predict latent market capacity using only location, accessibility, POI, competition, and static outlet signals.
  - Excluded features:
    - `avg_monthly_volume`, `avg_monthly_volume_calendar`, `total_volume`, `total_revenue`, `transaction_count`, `volume_std`, `volume_cv`, `sales_frequency`, `avg_to_max_ratio`
  - Hold-out performance: Blend RMSE **197.21**, R² **0.8313**.
  - Saved outputs:
    - `data/models/market_capacity_predictions.parquet`
    - `models/market_capacity_lgbm_model.txt`
    - `models/market_capacity_feature_importance.csv`

- Comparison output:
  - `data/models/model_comparison.parquet` now contains `predicted_peak_sales`, `predicted_market_capacity`, `current_sales`, `peak_gap`, and `market_gap`.
  - `market_gap = predicted_market_capacity - current_sales` is the preferred ranking signal for latent-demand work.
  - Top market-gap outlets are now dominated by cases where current sales are far below the predicted capacity, which is the business pattern we wanted to surface.

- Notes:
  - LightGBM and XGBoost were installed into the active `.venv` during the earlier training run, and the new dual trainer reused that environment.
  - The gold rebuild changed the merged feature table shape, so the new model script now fills missing transport-specific POI columns with zero when they are absent in the latest POI output.
  - If you want the dashboard to show both model versions and the new market-gap ranking, I can wire that next.


Round 2 — Feature layer upgrade notes

What was added
- Distance decay layer in `src/gold/feature_poi.py`.
- Competition density layer in `src/gold/build_outlet_features.py`.

How the new features are calculated

1) `poi_influence_score`
- Built from all POIs within a 1000 meter radius around each outlet.
- Each POI contributes a weight of:

```text
weight = exp(-distance / 200)
```

- The score is the sum of all weighted POIs inside the decay window.

2) `accessibility_weighted_score`
- Uses the same distance-decay weights as `poi_influence_score`.
- Only POIs whose `poi_category` is `transport` contribute to this score.
- This is a weighted accessibility proxy, not a raw count.

3) `competitor_count_500m`
- Counts outlet-to-outlet neighbors within 500 meters.
- Self is excluded from the count.

4) `competitor_count_1km`
- Counts outlet-to-outlet neighbors within 1 kilometer.
- Self is excluded from the count.

5) `nearest_competitor_distance`
- Minimum distance in meters from the outlet to any other outlet in the candidate neighborhood.
- Computed from projected coordinates using the local distance approximation in the competition block.

6) `saturation_score`
- A simple saturation proxy combining nearby competitor density and nearest competitor proximity.
- It increases when an outlet has more competitors nearby and/or a closer nearest competitor.

Saved outputs after the feature update
- `data/gold/poi_features.parquet`
- `data/gold/outlet_features.parquet`
- `data/models/cluster_profiles.parquet`
- `data/models/opportunity_recommendations.parquet`

Validation results
- Syntax check passed for `src/gold/feature_poi.py` and `src/gold/build_outlet_features.py`.
- Rebuild passed and the final outlet feature table now contains the new columns.
- Confirmed columns present in `data/gold/outlet_features.parquet`:
  - `poi_influence_score`
  - `accessibility_weighted_score`
  - `competitor_count_500m`
  - `competitor_count_1km`
  - `nearest_competitor_distance`
  - `saturation_score`

What this changes for Round 2
- The model now has a stronger demand-side signal from POI influence.
- The model now has explicit competitive pressure features instead of relying only on cluster heuristics.
- These features should be used before retraining or rescoring the potential model.
- The predict stage now consumes the updated mobility / POI feature layer so the dashboard and ranked recommendations stay aligned with the rebuilt gold outputs.

Round 2 — Model training — 2026-05-30

- What we did:
  - Ensured the training environment had the preferred GBM libraries. The training script attempted to import `lightgbm` and `xgboost`; when missing it installed them into the active `.venv` and retried.
  - Added a robust fallback: if LightGBM were unavailable after install, the script falls back to scikit-learn's `HistGradientBoostingRegressor`. The RandomForest baseline remains as an ensemble member.
  - Fixed runtime issues observed earlier (PyArrow-backed arrays causing sklearn indexing errors and `src` import path when running the script directly).

- Key training details:
  - CV strategy: 5-fold GroupKFold (groups = `geo_cell`), hold-out 80/20 stratified by log-binned target.
  - Models trained: LightGBM (preferred) + RandomForest (baseline) → 70% LGBM + 30% RF ensemble.
  - LightGBM average best iterations across folds: ~1999 (used to retrain final GBM with reduced lr).

- Final performance (hold-out):
  - Blend (GBM 70% + RF 30%) Hold-out RMSE: **135.95**
  - Blend Hold-out R²: **0.9198**
  - Blend Hold-out MdAPE: **0.5%**

- Artifacts saved (paths):
  - Predictions: `data/models/outlet_potential_predictions.parquet`
  - GBM model: `models/lgbm_potential_model.txt` (LightGBM format) or `models/skl_gbm_potential_model.pkl` when sklearn fallback
  - RF model: `models/rf_potential_model.pkl`
  - Scaler + encoders: `models/scaler.pkl`, `models/label_encoders.pkl`
  - Feature importances: `models/feature_importance.csv`

- Quick validation / top opportunities (top 10 shown):
  - OUT_01146 — predicted_potential 2054.01, avg_monthly_volume 1139.88, opportunity_pct 80.2%
  - OUT_08278 — predicted_potential 2055.23, avg_monthly_volume 1148.26, opportunity_pct 79.0%
  - OUT_00579 — predicted_potential 2067.01, avg_monthly_volume 1163.75, opportunity_pct 77.6%
  - OUT_10728 — predicted_potential 2079.90, avg_monthly_volume 1176.91, opportunity_pct 76.7%
  - OUT_08484 — predicted_potential 2069.93, avg_monthly_volume 1169.92, opportunity_pct 76.9%

- Notes & next recommended steps:
  - The training run successfully installed LightGBM/XGBoost into the `.venv` and used LightGBM for the main model; fallback is available and tested.
  - Decide whether gap calculations should use `avg_monthly_volume` (active-month mean) or `avg_monthly_volume_calendar` (calendar-month mean). `calendar` yields larger, more conservative gaps for outlets with inactive months.
  - Surface the new predictions and `feature_importance.csv` in the dashboard; I can patch `output/dashboard/app.py` to show top N outlets, hold-out metrics and a feature-importance plot.
  - Add a reproducibility entry (pipeline run id, venv packages list or `pip freeze`, timestamp) alongside saved Parquet outputs.



