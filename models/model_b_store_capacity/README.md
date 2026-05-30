# Model B - Store Capacity

Estimates store capacity from outlet_size and cooler_count as the capacity proxy.

## Feature Rules
- Uses only outlet_size and cooler_count.

## Opportunity Definition
- Opportunity = StoreCapacity - CurrentSales

## Metrics
- RMSE: 198.47
- MAE: 78.59
- R2: 0.8291
- MAPE: 18.1%
- MdAPE: 16.0%

## Files
- model.pkl: pickled model bundle and metadata
- lgbm_model.txt: LightGBM booster snapshot
- predictions.parquet: model output parquet file
- feature_importance.txt: plain-text feature importance summary

## Source Artifacts
- models\store_capacity_rf_model.pkl
- models\store_capacity_lgbm_model.txt
- data\models\store_capacity_predictions.parquet
- models\store_capacity_feature_importance.csv
