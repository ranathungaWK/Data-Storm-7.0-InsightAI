# Model B ? Market Capacity

Estimates latent market capacity from location, POI, competition, accessibility, mobility, and commercial activity signals.

## Feature Rules
- Excludes outlet_size, cooler_count, and sales history features.

## Opportunity Definition
- market_gap = predicted_value - current_sales

## Metrics
- RMSE: 184.06
- MAE: 68.17
- R?: 0.8530
- MAPE: 17.7%
- MdAPE: 14.2%

## Files
- `model.pkl`: pickled model bundle and metadata
- `lgbm_model.txt`: LightGBM booster snapshot
- `predictions.parquet`: model output parquet
