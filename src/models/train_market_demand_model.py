"""
Model C — Market Demand.

This model estimates latent market demand using only:
- POI
- Competition
- Accessibility
- Mobility
- Commercial activity

It excludes:
- outlet_size
- cooler_count
- sales history features

Target:
- max_monthly_volume

Ranking:
- Opportunity = MarketDemand - CurrentSales
- CurrentSales uses avg_monthly_volume when available, otherwise avg_monthly_volume_calendar.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import pickle
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import RobustScaler

from src.utils.logger import get_logger

logger = get_logger("models.train_market_demand_model")

FEATURE_PATH = Path("data/gold/outlet_features.parquet")
TARGET_PATH = Path("data/models/outlet_potential_target.parquet")
OUTPUT_DIR = Path("data/models")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = ["outlet_type"]

MARKET_DEMAND_FEATURES = [
    "poi_influence_score",
    "accessibility_weighted_score",
    "commercial_score",
    "accessibility_score",
    "mobility_score",
    "restaurant",
    "public_transit_facility_or_service",
    "train_station",
    "ground_transport_facility_or_service",
    "air_transport_facility_or_service",
    "competitor_count_500m",
    "competitor_count_1km",
    "nearest_competitor_distance",
    "saturation_score",
    "outlet_type",
]

LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
}


def ensure_lightgbm():
    try:
        return importlib.import_module("lightgbm")
    except Exception:
        logger.info("LightGBM not found — attempting install into active venv...")
        try:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "lightgbm",
                "xgboost",
                "--prefer-binary",
            ])
        except subprocess.CalledProcessError as exc:
            logger.warning(f"Pip install failed: {exc}")
            return None
        try:
            return importlib.import_module("lightgbm")
        except Exception as exc:
            logger.warning(f"Import after install still failed: {exc}")
            return None


lgb = ensure_lightgbm()


def regression_metrics(y_true, y_pred, label="") -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)
    mdape = float(np.median(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)
    logger.info(
        f"[{label}] RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}  MAPE={mape:.1f}%  MdAPE={mdape:.1f}%"
    )
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "mdape": mdape}


def load_data() -> pd.DataFrame:
    logger.info("Loading features and target...")
    feat = pd.read_parquet(FEATURE_PATH)
    targ = pd.read_parquet(TARGET_PATH)
    df = feat.merge(targ, on="outlet_id", how="inner")
    df = df.dropna(subset=["max_monthly_volume"]).copy()
    logger.info(f"Merged dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def current_sales_column(df: pd.DataFrame) -> str:
    if "avg_monthly_volume" in df.columns:
        return "avg_monthly_volume"
    if "avg_monthly_volume_calendar" in df.columns:
        return "avg_monthly_volume_calendar"
    raise KeyError("No current sales baseline column found")


def build_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    state: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, RobustScaler, dict, np.ndarray, dict]:
    df = df.copy()

    for col in feature_columns:
        if col not in df.columns:
            logger.warning(f"Feature '{col}' not found — filling with 0")
            df[col] = 0

    numeric_cols = [c for c in feature_columns if c not in CATEGORICAL_COLS]
    if state is None:
        numeric_medians = {col: df[col].median() for col in numeric_cols}
        numeric_caps = {col: df[col].quantile(0.999) for col in numeric_cols}
        cat_levels = {
            col: df[col].fillna("__MISSING__").astype(str).dropna().astype(str).unique().tolist()
            for col in CATEGORICAL_COLS
        }
    else:
        numeric_medians = state["numeric_medians"]
        numeric_caps = state["numeric_caps"]
        cat_levels = state["cat_levels"]

    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(numeric_medians.get(col, 0))

    le_map = {}
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("__MISSING__").astype(str)
        levels = cat_levels.get(col, [])
        level_to_code = {value: index for index, value in enumerate(levels)}
        df[col] = df[col].map(level_to_code).fillna(-1).astype(int)
        le_map[col] = levels
        logger.info(f"Encoded '{col}': {len(levels)} classes")

    for col in numeric_cols:
        p999 = numeric_caps.get(col, None)
        if p999 is not None and p999 > 0:
            df[col] = df[col].clip(upper=p999)

    y = df["max_monthly_volume"].astype(float).to_numpy()
    y_log = np.log1p(y)
    outlet_ids = df["outlet_id"].to_numpy()

    if "geo_cell" in df.columns:
        df["geo_cell"] = df["geo_cell"].fillna(df["lat_bucket"].astype(str) + "_" + df["lon_bucket"].astype(str))
        groups = df["geo_cell"].astype(str).to_numpy()
    else:
        groups = (df["lat_bucket"].astype(str) + "_" + df["lon_bucket"].astype(str)).to_numpy()

    X_raw = df[feature_columns].copy().reset_index(drop=True)
    scaler = state["scaler"] if state is not None else RobustScaler()
    X_scaled = X_raw.copy()
    scale_cols = [c for c in feature_columns if c not in CATEGORICAL_COLS]
    if state is None:
        X_scaled[scale_cols] = scaler.fit_transform(X_raw[scale_cols])
    else:
        X_scaled[scale_cols] = scaler.transform(X_raw[scale_cols])

    fitted_state = {
        "numeric_medians": numeric_medians,
        "numeric_caps": numeric_caps,
        "cat_levels": cat_levels,
        "scaler": scaler,
    }

    return X_raw, X_scaled, y, y_log, groups, scaler, le_map, outlet_ids, fitted_state


def train_lgbm_cv(X: pd.DataFrame, y_log: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    gkf = GroupKFold(n_splits=n_splits)
    oof_log = np.zeros(len(y_log))
    models = []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
        logger.info(f"  LGBM fold {fold + 1}/{n_splits} — train={len(tr_idx):,}  val={len(val_idx):,}")
        tr_data = lgb.Dataset(X.iloc[tr_idx], label=y_log[tr_idx])
        val_data = lgb.Dataset(X.iloc[val_idx], label=y_log[val_idx], reference=tr_data)
        model = lgb.train(
            LGBM_PARAMS,
            tr_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False), lgb.log_evaluation(period=-1)],
        )
        oof_log[val_idx] = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
        models.append(model)
        logger.info(f"  Fold {fold + 1} best iteration: {model.best_iteration}")
    return np.expm1(oof_log), models


def train_lgbm_final(X: pd.DataFrame, y_log: np.ndarray, rounds: int):
    logger.info(f"Retraining final LGBM on full train set (rounds={rounds}) ...")
    tr_data = lgb.Dataset(X, label=y_log)
    return lgb.train({**LGBM_PARAMS, "learning_rate": 0.02}, tr_data, num_boost_round=rounds)


def train_rf_cv(X_scaled: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y))
    models = []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
        logger.info(f"  RF fold {fold + 1}/{n_splits}")
        model = RandomForestRegressor(
            n_estimators=300,
            max_features=0.6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled.iloc[tr_idx], y[tr_idx])
        oof[val_idx] = model.predict(X_scaled.iloc[val_idx])
        models.append(model)
    return oof, models


def log_feature_importance(models: list, feature_names: list, output_path: Path):
    importance_sum = np.zeros(len(feature_names), dtype=float)
    for model in models:
        if hasattr(model, "feature_importance"):
            try:
                importance_sum += model.feature_importance(importance_type="gain")
                continue
            except Exception:
                pass
        if hasattr(model, "feature_importances_"):
            importance_sum += np.asarray(model.feature_importances_)
            continue
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance_sum / max(len(models), 1)}).sort_values("importance", ascending=False)
    importance_df.to_csv(output_path, index=False)
    logger.info("\n" + importance_df.head(20).to_string(index=False))
    return importance_df


class BlendPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, gbm_model, rf_model, scaler, feature_columns: list[str]):
        self.gbm_model = gbm_model
        self.rf_model = rf_model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.scale_columns = [column for column in feature_columns if column not in CATEGORICAL_COLS]

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)
        raw = X[self.feature_columns].copy()
        scaled = raw.copy()
        if self.scale_columns:
            scaled[self.scale_columns] = self.scaler.transform(raw[self.scale_columns])
        gbm_pred = np.expm1(self.gbm_model.predict(raw))
        rf_pred = self.rf_model.predict(scaled)
        return 0.7 * gbm_pred + 0.3 * rf_pred


def main():
    logger.info("═" * 65)
    logger.info("  MODEL C — MARKET DEMAND TRAINING START")
    logger.info("═" * 65)

    df = load_data()
    sales_col = current_sales_column(df)
    y = df["max_monthly_volume"].astype(float).to_numpy()
    y_log = np.log1p(y)
    groups = df["geo_cell"].astype(str).to_numpy() if "geo_cell" in df.columns else (df["lat_bucket"].astype(str) + "_" + df["lon_bucket"].astype(str)).to_numpy()

    split_seed = 42
    strat_bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=split_seed,
        stratify=strat_bins,
    )

    logger.info(f"Train: {len(train_idx):,} | Hold-out test: {len(test_idx):,}")
    logger.info("Model C feature set: POI + competition + accessibility + mobility + commercial activity")

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    X_train_raw, X_train_scaled, y_train, y_log_train, grp_train, scaler, le_map, ids_train, fit_state = build_matrix(
        train_df,
        MARKET_DEMAND_FEATURES,
    )
    X_test_raw, X_test_scaled, y_test, y_log_test, grp_test, _, _, ids_test, _ = build_matrix(
        test_df,
        MARKET_DEMAND_FEATURES,
        state=fit_state,
    )

    oof_gbm, cv_models = train_lgbm_cv(X_train_raw, y_log_train, grp_train)
    regression_metrics(y_train, oof_gbm, label="Model C LGBM OOF")
    best_rounds = int(np.mean([m.best_iteration for m in cv_models]))
    final_gbm = train_lgbm_final(X_train_raw, y_log_train, best_rounds)
    test_preds = np.expm1(final_gbm.predict(X_test_raw))
    regression_metrics(y_test, test_preds, label="Model C LGBM Hold-out")

    oof_rf, rf_models = train_rf_cv(X_train_scaled, y_train, grp_train)
    regression_metrics(y_train, oof_rf, label="Model C RF OOF")
    test_preds_rf = rf_models[0].predict(X_test_scaled)
    regression_metrics(y_test, test_preds_rf, label="Model C RF Hold-out")

    blend_oof = 0.7 * oof_gbm + 0.3 * oof_rf
    blend_test = 0.7 * test_preds + 0.3 * test_preds_rf
    regression_metrics(y_train, blend_oof, label="Model C Blend OOF")
    metrics_test = regression_metrics(y_test, blend_test, label="Model C Blend Hold-out")

    holdout_current_sales = test_df[sales_col].to_numpy() if sales_col in test_df.columns else y_test
    holdout_predicted_market_demand = np.maximum(blend_test, holdout_current_sales)

    holdout_df = pd.DataFrame(
        {
            "outlet_id": ids_test,
            "current_sales": holdout_current_sales,
            "predicted_market_demand": np.round(holdout_predicted_market_demand, 2),
            "opportunity": np.round(holdout_predicted_market_demand - holdout_current_sales, 2),
            "opportunity_pct": np.round((holdout_predicted_market_demand - holdout_current_sales) / (holdout_current_sales + 1e-6) * 100, 1),
            "actual_max_monthly": y_test,
            "split": "test",
        }
    )

    train_current_sales = train_df[sales_col].to_numpy() if sales_col in train_df.columns else y_train
    train_predicted_market_demand = 0.7 * np.expm1(final_gbm.predict(X_train_raw)) + 0.3 * rf_models[0].predict(X_train_scaled)
    train_df_out = pd.DataFrame(
        {
            "outlet_id": ids_train,
            "current_sales": train_current_sales,
            "predicted_market_demand": np.round(np.maximum(train_predicted_market_demand, train_current_sales), 2),
            "opportunity": np.round(np.maximum(train_predicted_market_demand, train_current_sales) - train_current_sales, 2),
            "opportunity_pct": np.round((np.maximum(train_predicted_market_demand, train_current_sales) - train_current_sales) / (train_current_sales + 1e-6) * 100, 1),
            "actual_max_monthly": y_train,
            "split": "train",
        }
    )

    predictions = pd.concat([train_df_out, holdout_df], ignore_index=True)
    predictions = predictions.sort_values(["split", "opportunity"], ascending=[True, False]).reset_index(drop=True)

    feature_importance_path = MODEL_DIR / "market_demand_feature_importance.csv"
    log_feature_importance(cv_models, list(X_train_raw.columns), feature_importance_path)

    perm_model = BlendPredictor(final_gbm, rf_models[0], scaler, MARKET_DEMAND_FEATURES)
    perm = permutation_importance(
        perm_model,
        X_test_raw,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    permutation_df = pd.DataFrame(
        {
            "feature": MARKET_DEMAND_FEATURES,
            "rmse_increase_mean": perm.importances_mean,
            "rmse_increase_std": perm.importances_std,
        }
    ).sort_values("rmse_increase_mean", ascending=False)
    permutation_path = OUTPUT_DIR / "market_demand_permutation_importance.csv"
    permutation_df.to_csv(permutation_path, index=False)

    ranking_df = holdout_df.sort_values("opportunity", ascending=False).reset_index(drop=True)
    ranking_path = OUTPUT_DIR / "model_c_opportunity_ranking.parquet"
    ranking_df.to_parquet(ranking_path, index=False)

    output_path = OUTPUT_DIR / "model_c_market_demand_predictions.parquet"
    predictions.to_parquet(output_path, index=False)

    if lgb is not None:
        final_gbm.save_model(str(MODEL_DIR / "model_c_market_demand_lgbm_model.txt"))
    with open(MODEL_DIR / "model_c_market_demand_rf_model.pkl", "wb") as f:
        pickle.dump(rf_models[0], f)
    with open(MODEL_DIR / "model_c_market_demand_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "model_c_market_demand_label_encoders.pkl", "wb") as f:
        pickle.dump(le_map, f)

    summary_path = MODEL_DIR / "model_c_market_demand_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Model C — Market Demand",
                "",
                f"- Current sales column: {sales_col}",
                f"- Hold-out RMSE: {metrics_test['rmse']:.2f}",
                f"- Hold-out R²: {metrics_test['r2']:.4f}",
                f"- Hold-out MAE: {metrics_test['mae']:.2f}",
                "",
                "## Top 10 opportunity outlets",
                ranking_df[["outlet_id", "predicted_market_demand", "current_sales", "opportunity"]].head(10).to_string(index=False),
            ]
        ),
        encoding="utf-8",
    )

    logger.info(f"Saved market demand predictions -> {output_path}")
    logger.info(f"Saved opportunity ranking -> {ranking_path}")
    logger.info(f"Saved permutation importance -> {permutation_path}")
    logger.info(f"Saved summary -> {summary_path}")
    logger.info("═" * 65)
    logger.info("  MODEL C — MARKET DEMAND TRAINING COMPLETE")
    logger.info("═" * 65)


if __name__ == "__main__":
    main()
