"""
Dual outlet potential trainers.

Version A: peak-sales model
- Keeps the historical demand and geography feature stack, but removes the
    direct target-derived ratio feature `avg_to_max_ratio`.

Version B: store-capacity model
- Uses the outlet static capacity signals (outlet_size and cooler_count) as
    the primary explanatory features for capacity.

Outputs:
- data/models/peak_sales_predictions.parquet
- data/models/store_capacity_predictions.parquet
- data/models/model_comparison.parquet
- models/peak_sales_lgbm_model.txt
- models/store_capacity_lgbm_model.txt
- models/peak_sales_feature_importance.csv
- models/store_capacity_feature_importance.csv
"""

from __future__ import annotations

from pathlib import Path
import importlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import RobustScaler

from src.utils.logger import get_logger

logger = get_logger("models.train_dual_potential_models")


def ensure_lightgbm():
    try:
        return importlib.import_module("lightgbm")
    except Exception:
        logger.info("LightGBM not found — attempting install into active venv...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "xgboost", "--prefer-binary"])
        except subprocess.CalledProcessError as exc:
            logger.warning(f"Pip install failed: {exc}")
            return None
        try:
            return importlib.import_module("lightgbm")
        except Exception as exc:
            logger.warning(f"Import after install still failed: {exc}")
            return None


lgb = ensure_lightgbm()

FEATURE_PATH = Path("data/gold/outlet_features.parquet")
TARGET_PATH = Path("data/models/outlet_potential_target.parquet")
OUTPUT_DIR = Path("data/models")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = ["outlet_size", "outlet_type"]

PEAK_HISTORY_FEATURES = [
    "avg_monthly_volume_calendar",
    "avg_monthly_volume",
    "volume_std",
    "volume_cv",
    "active_months",
    "transaction_count",
    "sku_diversity",
    "distributor_diversity",
    "total_volume",
    "total_revenue",
    "sales_frequency",
]

LOCATION_SIGNAL_FEATURES = [
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
    "outlet_size",
    "cooler_count",
    "outlet_type",
    "master_completeness_score",
]

MODEL_A_FEATURES = PEAK_HISTORY_FEATURES + LOCATION_SIGNAL_FEATURES + [
    "revenue_per_tx",
    "volume_efficiency",
    "competitive_pressure",
]

MODEL_B_FEATURES = ["outlet_size", "cooler_count"]

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


def regression_metrics(y_true, y_pred, label="") -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    mdape = np.median(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
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


def build_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    include_history_derived: bool,
    state: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, RobustScaler, dict, np.ndarray, dict]:
    df = df.copy()

    if include_history_derived:
        df["revenue_per_tx"] = df["total_revenue"] / df["transaction_count"].clip(lower=1)
        df["volume_efficiency"] = (
            df["avg_monthly_volume"] / df["avg_monthly_volume_calendar"].clip(lower=1)
        ).clip(upper=5)
        df["competitive_pressure"] = df["saturation_score"] / df["nearest_competitor_distance"].clip(lower=1)

    for col in feature_columns:
        if col not in df.columns:
            logger.warning(f"Feature '{col}' not found — filling with 0")
            df[col] = 0

    numeric_cols = [c for c in feature_columns if c not in CATEGORICAL_COLS]
    if state is None:
        numeric_medians = {col: df[col].median() for col in numeric_cols}
        numeric_caps = {col: df[col].quantile(0.999) for col in numeric_cols}
        cat_levels = {col: df[col].fillna("__MISSING__").astype(str).dropna().astype(str).unique().tolist() for col in CATEGORICAL_COLS}
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


def train_version(
    *,
    name: str,
    feature_columns: list[str],
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
    y_log: np.ndarray,
    groups: np.ndarray,
    include_history_derived: bool,
    output_prefix: str,
    floor_to_calendar_avg: bool,
) -> dict:
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Build train matrix and reuse the fitted preprocessing state on the hold-out set.
    X_train_raw, X_train_scaled, y_train, y_log_train, grp_train, scaler, le_map, ids_train, fit_state = build_matrix(
        train_df, feature_columns, include_history_derived
    )
    X_test_raw, X_test_scaled, y_test, y_log_test, grp_test, _, _, ids_test, _ = build_matrix(
        test_df, feature_columns, include_history_derived, state=fit_state
    )

    logger.info(f"\n=== {name} ===")
    logger.info(f"Train rows: {len(y_train):,} | Hold-out rows: {len(y_test):,} | Features: {len(feature_columns)}")

    oof_gbm, cv_models = train_lgbm_cv(X_train_raw, y_log_train, grp_train)
    regression_metrics(y_train, oof_gbm, label=f"{name} LGBM OOF")
    best_rounds = int(np.mean([m.best_iteration for m in cv_models]))
    final_gbm = train_lgbm_final(X_train_raw, y_log_train, best_rounds)
    test_preds = np.expm1(final_gbm.predict(X_test_raw))
    regression_metrics(y_test, test_preds, label=f"{name} LGBM Hold-out")

    oof_rf, rf_models = train_rf_cv(X_train_scaled, y_train, grp_train)
    regression_metrics(y_train, oof_rf, label=f"{name} RF OOF")
    test_preds_rf = rf_models[0].predict(X_test_scaled)
    regression_metrics(y_test, test_preds_rf, label=f"{name} RF Hold-out")

    blend_oof = 0.7 * oof_gbm + 0.3 * oof_rf
    blend_test = 0.7 * test_preds + 0.3 * test_preds_rf
    regression_metrics(y_train, blend_oof, label=f"{name} Blend OOF")
    metrics_test = regression_metrics(y_test, blend_test, label=f"{name} Blend Hold-out")

    if floor_to_calendar_avg:
        obs_avg = test_df["avg_monthly_volume_calendar"].to_numpy() if "avg_monthly_volume_calendar" in test_df.columns else y_test
        blend_test = np.maximum(blend_test, obs_avg)

    all_test = pd.DataFrame(
        {
            "outlet_id": ids_test,
            "actual_max_monthly": y_test,
            "current_sales": test_df["avg_monthly_volume_calendar"].to_numpy() if "avg_monthly_volume_calendar" in test_df.columns else y_test,
            "predicted_value": np.round(blend_test, 2),
            "split": "test",
        }
    )

    all_train = pd.DataFrame(
        {
            "outlet_id": ids_train,
            "actual_max_monthly": y_train,
            "current_sales": train_df["avg_monthly_volume_calendar"].to_numpy() if "avg_monthly_volume_calendar" in train_df.columns else y_train,
            "predicted_value": np.round(0.7 * np.expm1(final_gbm.predict(X_train_raw)) + 0.3 * rf_models[0].predict(X_train_scaled), 2),
            "split": "train",
        }
    )

    all_rows = pd.concat([all_train, all_test], ignore_index=True)
    all_rows["predicted_value"] = np.maximum(all_rows["predicted_value"], all_rows["current_sales"])
    all_rows["gap_value"] = np.round(all_rows["predicted_value"] - all_rows["current_sales"], 2)
    all_rows["gap_pct"] = np.round(all_rows["gap_value"] / (all_rows["current_sales"] + 1e-6) * 100, 1)

    feature_importance_path = MODEL_DIR / f"{output_prefix}_feature_importance.csv"
    log_feature_importance(cv_models, list(X_train_raw.columns), feature_importance_path)

    if lgb is not None:
        final_gbm.save_model(str(MODEL_DIR / f"{output_prefix}_lgbm_model.txt"))
    with open(MODEL_DIR / f"{output_prefix}_rf_model.pkl", "wb") as f:
        import pickle
        pickle.dump(rf_models[0], f)
    with open(MODEL_DIR / f"{output_prefix}_scaler.pkl", "wb") as f:
        import pickle
        pickle.dump(scaler, f)
    with open(MODEL_DIR / f"{output_prefix}_label_encoders.pkl", "wb") as f:
        import pickle
        pickle.dump(le_map, f)

    pred_path = OUTPUT_DIR / f"{output_prefix}_predictions.parquet"
    all_rows.to_parquet(pred_path, index=False)
    logger.info(f"Saved {name} predictions -> {pred_path}")

    return {
        "name": name,
        "metrics": metrics_test,
        "predictions": all_rows,
        "predictions_path": pred_path,
        "best_rounds": best_rounds,
        "feature_importance_path": feature_importance_path,
    }


def main():
    logger.info("═" * 65)
    logger.info("  DUAL OUTLET POTENTIAL TRAINING START")
    logger.info("═" * 65)

    df = load_data()
    y = df["max_monthly_volume"].astype(float).to_numpy()
    y_log = np.log1p(y)
    groups = df["geo_cell"].astype(str).to_numpy() if "geo_cell" in df.columns else (df["lat_bucket"].astype(str) + "_" + df["lon_bucket"].astype(str)).to_numpy()
    outlet_ids = df["outlet_id"].to_numpy()

    # explicit leak check: this ratio is target-derived and is excluded from training
    if "avg_to_max_ratio" in df.columns:
        logger.warning("avg_to_max_ratio exists in source data; it will not be used for training.")
    else:
        logger.info("avg_to_max_ratio is computed in older code paths only; excluded here by design.")

    split_seed = 42
    strat_bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=split_seed,
        stratify=strat_bins,
    )

    logger.info(f"Train: {len(train_idx):,} | Hold-out test: {len(test_idx):,}")

    model_a = train_version(
        name="Version A - Peak Sales",
        feature_columns=MODEL_A_FEATURES,
        df=df,
        train_idx=train_idx,
        test_idx=test_idx,
        y=y,
        y_log=y_log,
        groups=groups,
        include_history_derived=True,
        output_prefix="peak_sales",
        floor_to_calendar_avg=True,
    )

    model_b = train_version(
        name="Version B - Store Capacity",
        feature_columns=MODEL_B_FEATURES,
        df=df,
        train_idx=train_idx,
        test_idx=test_idx,
        y=y,
        y_log=y_log,
        groups=groups,
        include_history_derived=False,
        output_prefix="store_capacity",
        floor_to_calendar_avg=True,
    )

    comparison = model_a["predictions"][ ["outlet_id", "predicted_value", "current_sales", "gap_value", "gap_pct"] ].merge(
        model_b["predictions"][ ["outlet_id", "predicted_value", "gap_value", "gap_pct"] ],
        on="outlet_id",
        how="inner",
        suffixes=("_peak_sales", "_market_capacity"),
    )
    comparison = comparison.rename(
        columns={
            "predicted_value_peak_sales": "predicted_peak_sales",
            "predicted_value_market_capacity": "predicted_store_capacity",
            "current_sales_peak_sales": "current_sales",
            "gap_value_peak_sales": "peak_gap",
            "gap_pct_peak_sales": "peak_gap_pct",
            "gap_value_market_capacity": "market_gap",
            "gap_pct_market_capacity": "market_gap_pct",
        }
    )
    comparison["market_gap"] = comparison["predicted_store_capacity"] - comparison["current_sales"]
    comparison["market_gap_pct"] = np.round(comparison["market_gap"] / (comparison["current_sales"] + 1e-6) * 100, 1)
    comparison = comparison.sort_values("market_gap", ascending=False)
    comparison_path = OUTPUT_DIR / "model_comparison.parquet"
    comparison.to_parquet(comparison_path, index=False)
    logger.info(f"Saved comparison -> {comparison_path}")

    logger.info("\nTop 10 market gap outlets:")
    logger.info("\n" + comparison[["outlet_id", "predicted_peak_sales", "predicted_store_capacity", "current_sales", "market_gap", "market_gap_pct"]].head(10).to_string(index=False))

    logger.info("\nSummary metrics")
    logger.info(f"Peak-sales blend hold-out RMSE: {model_a['metrics']['rmse']:.2f} | R²: {model_a['metrics']['r2']:.4f}")
    logger.info(f"Store-capacity blend hold-out RMSE: {model_b['metrics']['rmse']:.2f} | R²: {model_b['metrics']['r2']:.4f}")

    logger.info("═" * 65)
    logger.info("  DUAL OUTLET POTENTIAL TRAINING COMPLETE")
    logger.info("═" * 65)


if __name__ == "__main__":
    main()
