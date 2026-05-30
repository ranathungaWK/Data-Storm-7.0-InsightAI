# Model C ? Market Demand

Predicts latent market demand using only POI, competition, accessibility, mobility, and commercial activity.

## Feature Rules
- Uses only POI, competition, accessibility, mobility, and commercial activity.

## Opportunity Definition
- Opportunity = MarketDemand - CurrentSales

## Metrics
- RMSE: 270.79
- MAE: 177.83
- R?: 0.6819
- MAPE: 75.5%
- MdAPE: 47.4%

## Files
- `model.pkl`: pickled model bundle and metadata
- `lgbm_model.txt`: LightGBM booster snapshot
- `predictions.parquet`: model output parquet file
