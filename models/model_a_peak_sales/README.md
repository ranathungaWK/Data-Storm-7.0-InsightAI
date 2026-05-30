# Model A ? Peak Sales

Predicts the highest monthly sales an outlet can achieve using historical demand signals plus location and business context.

## Feature Rules
- Uses history features; this is the historical peak-sales model.

## Opportunity Definition
- Opportunity = PeakSales - CurrentSales

## Metrics
- RMSE: 204.29
- MAE: 55.03
- R?: 0.8190
- MAPE: 11.2%
- MdAPE: 8.6%

## Files
- `model.pkl`: pickled model bundle and metadata
- `lgbm_model.txt`: LightGBM booster snapshot
- `predictions.parquet`: model output parquet file
