# Model A - Peak Sales

Predicts the highest monthly sales an outlet can achieve using historical demand signals plus location and business context.

## Feature Rules
- Uses history features plus location and business context. This is the historical peak-sales model.

## Opportunity Definition
- Opportunity = PeakSales - CurrentSales

## Metrics
- RMSE: 204.29
- MAE: 55.03
- R2: 0.8190
- MAPE: 11.2%
- MdAPE: 8.6%

## Files
- model.pkl: pickled model bundle and metadata
- lgbm_model.txt: LightGBM booster snapshot
- predictions.parquet: model output parquet file
- feature_importance.txt: plain-text feature importance summary

## Source Artifacts
- models\peak_sales_rf_model.pkl
- models\peak_sales_lgbm_model.txt
- data\models\peak_sales_predictions.parquet
- models\peak_sales_feature_importance.csv
