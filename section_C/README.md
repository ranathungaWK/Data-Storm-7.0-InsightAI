# Section C — Mathematical Framework

## Story

We estimate *Potential Sales* (latent demand) rather than short-term predicted sales. This is the uncapped peak an outlet can achieve given its history and location context.

### Step 1 — Observed sales

Current sales: `CurrentSales` (observed 30/90/365-day aggregates used in models).

### Step 2 — Train Model A

Target: Peak Historical Sales (monthly peak observed historically).

Features: history, business context, location context (POI, competition, accessibility, mobility, outlet profile).

### Step 3 — Predict

Predicted peak: `\hat{Y}_{peak} = f(X)` where $X$ = history + competition + POI + accessibility + outlet profile.

### Step 4 — Opportunity

Opportunity for outlet $i$:

$$
Opportunity_i = \max\left(0, \hat{Y}_{peak,i} - CurrentSales_i\right)
$$

## Model metrics (from pipeline)

```
RMSE 204.29
MAE 55.03
R^2 0.819
```

## Feature importance

See `report_assets/feature_importance.svg` (top features), `report_assets/model_metrics.svg` for metric visualizations, and `report_assets/top100_allocations.csv` for the top-100 allocation evidence.
