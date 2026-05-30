# Model C - Market Demand

Predicts latent market demand using only POI, competition, accessibility, mobility, and commercial activity.

## Feature Rules
- Uses only POI, competition, accessibility, mobility, and commercial activity.

## Opportunity Definition
- Opportunity = MarketDemand - CurrentSales

## Metrics
- RMSE: 270.79
- MAE: 177.83
- R2: 0.6819
- MAPE: 75.5%
- MdAPE: 47.4%

## Files
- model.pkl: pickled model bundle and metadata
- lgbm_model.txt: LightGBM booster snapshot
- predictions.parquet: model output parquet file
- feature_importance.txt: plain-text feature importance summary

## Source Artifacts
- models\model_c_market_demand_rf_model.pkl
- models\model_c_market_demand_lgbm_model.txt
- data\models\model_c_market_demand_predictions.parquet
- models\market_demand_feature_importance.csv
