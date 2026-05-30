# Model B - Market Capacity

Estimates latent market capacity using location, POI, competition, accessibility, mobility, commercial activity, and outlet static signals.

## Feature Rules
- Uses location, POI, competition, accessibility, mobility, commercial activity, outlet_size, and cooler_count. This is the broader market-capacity model.

## Opportunity Definition
- market_gap = predicted_value - current_sales

## Metrics
- RMSE: 184.06
- MAE: 68.17
- R2: 0.8530
- MAPE: 17.7%
- MdAPE: 14.2%

## Files
- model.pkl: pickled model bundle and metadata
- lgbm_model.txt: LightGBM booster snapshot
- predictions.parquet: model output parquet file
- feature_importance.txt: plain-text feature importance summary

## Source Artifacts
- models\market_capacity_rf_model.pkl
- models\market_capacity_lgbm_model.txt
- data\models\market_capacity_predictions.parquet
- models\market_capacity_feature_importance.csv
