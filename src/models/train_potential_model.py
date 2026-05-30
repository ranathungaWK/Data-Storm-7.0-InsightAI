"""
╔══════════════════════════════════════════════════════════════════════════╗
║           OUTLET POTENTIAL MODEL  —  train_potential_model.py           ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  WHAT IS THIS MODEL?                                                     ║
║  ─────────────────────────────────────────────────────────────────────── ║
║  This is a "Sales Potential Ceiling" model.                              ║
║                                                                          ║
║  Every outlet has an OBSERVED sales level (what it actually sells        ║
║  month to month) and a LATENT CEILING (what it COULD sell given its      ║
║  location, size, customer traffic, and market conditions).               ║
║                                                                          ║
║  The gap between the two is the GROWTH OPPORTUNITY.                      ║
║                                                                          ║
║  WHY DO WE USE IT?                                                       ║
║  ─────────────────────────────────────────────────────────────────────── ║
║  • Sales teams can't visit every outlet — this model RANKS them by       ║
║    opportunity size so effort goes where ROI is highest.                 ║
║  • Outlets performing near their ceiling need retention, not push.       ║
║  • Outlets performing far below ceiling are UNDER-SERVED — prime         ║
║    targets for range extension, cooler placement, promotions.            ║
║                                                                          ║
║  WHAT IS THE TARGET?                                                     ║
║  ─────────────────────────────────────────────────────────────────────── ║
║  y = max_monthly_volume  (from outlet_potential_target.parquet)          ║
║                                                                          ║
║  This is the highest monthly volume ever recorded at that outlet.        ║
║  It is a proxy for the DEMAND CEILING — the best month represents        ║
║  what the outlet is capable of when conditions are right.                ║
║                                                                          ║
║  We log-transform it (log1p) before training because sales data is       ║
║  heavily right-skewed — a few big outlets would dominate the loss        ║
║  function and the model would ignore small outlets entirely.             ║
║                                                                          ║
║  TRAIN / VALIDATION STRATEGY                                             ║
║  ─────────────────────────────────────────────────────────────────────── ║
║  • 80 / 20 stratified hold-out split (main evaluation set)               ║
║  • 5-Fold GroupKFold on geo_cell (grid cell ~1km²) so spatially          ║
║    adjacent outlets never leak between train and validation folds.       ║
║  • OOF (Out-Of-Fold) predictions are assembled and evaluated once        ║
║    on the full training set, then the final model is retrained on        ║
║    100% of train data and evaluated on the hold-out set.                 ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import sys
import subprocess
import importlib

# LightGBM will be attempted to be installed/imported at runtime; set placeholder
lgb = None

# Make sure project root is on sys.path so `src` imports work when script is run directly
sys.path.insert(0, str(Path.cwd()))

from src.utils.logger import get_logger

logger = get_logger("models.train_potential_model")


def ensure_lightgbm():
    """Try to import LightGBM; if missing, attempt to install it into the current venv.
    Returns the imported module or None on failure.
    """
    try:
        return importlib.import_module("lightgbm")
    except Exception:
        logger.info("LightGBM not found — attempting to install lightgbm and xgboost into active venv...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "xgboost", "--prefer-binary"]) 
        except subprocess.CalledProcessError as e:
            logger.warning(f"Pip install failed: {e}")
            return None

        try:
            return importlib.import_module("lightgbm")
        except Exception as e:
            logger.warning(f"Import after install still failed: {e}")
            return None


# Attempt to ensure LightGBM is available for the rest of this script
lgb = ensure_lightgbm()


# ══════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════

FEATURE_PATH = Path("data/gold/outlet_features.parquet")
TARGET_PATH  = Path("data/models/outlet_potential_target.parquet")
OUTPUT_PRED  = Path("data/models/outlet_potential_predictions.parquet")
MODEL_DIR    = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# FEATURE REGISTRY
# Features are grouped so it is easy to ablate / swap groups.
# NOTE: zero-variance columns from a broken POI pipeline are excluded.
#       Re-add education, healthcare, shopping etc. once POI is fixed.
# ══════════════════════════════════════════════════════════════════════════

DEMAND_FEATURES = [
    "avg_monthly_volume_calendar",   # calendar-normalised avg — preferred over raw avg
    "avg_monthly_volume",            # raw avg — extra signal
    "volume_std",                    # spread of monthly volumes
    "volume_cv",                     # coefficient of variation (std / mean)
    "active_months",                 # how long the outlet has been active
    "transaction_count",             # total transactions in history
    "sku_diversity",                 # breadth of product range
    "distributor_diversity",         # number of distinct distributors
    "total_volume",                  # lifetime volume
    "total_revenue",                 # lifetime revenue
    "sales_frequency",               # avg transactions per month
]

GEO_FEATURES = [
    "poi_influence_score",           # distance-decay weighted POI density
    "accessibility_weighted_score",  # transport-specific decay score
    "commercial_score",              # restaurant + shopping + accommodation count
    "accessibility_score",           # raw transport POI count
    "mobility_score",                # transit mobility proxy
    # POI sub-counts (only non-zero ones in current data)
    "restaurant",
    "public_transit_facility_or_service",
    "train_station",
    "ground_transport_facility_or_service",
    "air_transport_facility_or_service",
]

COMPETITION_FEATURES = [
    "competitor_count_500m",
    "competitor_count_1km",
    "nearest_competitor_distance",
    "saturation_score",
]

STATIC_FEATURES = [
    "outlet_size",           # categorical → will be encoded
    "cooler_count",
    "outlet_type",           # categorical → will be encoded
    "master_completeness_score",
]

# Categorical columns that need encoding
CATEGORICAL_COLS = ["outlet_size", "outlet_type"]

# All features used for training
FEATURE_COLUMNS = DEMAND_FEATURES + GEO_FEATURES + COMPETITION_FEATURES + STATIC_FEATURES

# Features that will be scaled (tree models don't need scaling but RF benefits
# and it helps inspect feature magnitudes)
SCALE_COLS = [f for f in FEATURE_COLUMNS if f not in CATEGORICAL_COLS]


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & MERGE
# ══════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    logger.info("Loading features ...")
    feat = pd.read_parquet(FEATURE_PATH)

    logger.info("Loading target ...")
    targ = pd.read_parquet(TARGET_PATH)

    df = feat.merge(targ, on="outlet_id", how="inner")
    logger.info(f"Merged dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

    missing_target = df["max_monthly_volume"].isna().sum()
    if missing_target > 0:
        logger.warning(f"Dropping {missing_target} rows with null target")
        df = df.dropna(subset=["max_monthly_volume"])

    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    """
    Returns:
        X_raw   — DataFrame with all features (for LightGBM, handles categoricals natively)
        X_scaled— numpy array scaled with RobustScaler  (for RF)
        y       — raw target (max_monthly_volume)
        y_log   — log1p(y) used as the actual training target
        groups  — geo_cell group labels for GroupKFold
        scaler  — fitted RobustScaler (save for inference)
        le_map  — dict of LabelEncoders keyed by column name
    """
    df = df.copy()

    # ── Derived features ────────────────────────────────────────────────
    # Ratio of current avg to max: how far below ceiling the outlet sits
    # (will be 1.0 for the outlet whose max == avg, < 1 otherwise)
    df["avg_to_max_ratio"] = (
        df["avg_monthly_volume_calendar"]
        / (df["max_monthly_volume"].clip(lower=1))
    ).clip(upper=1.0)

    # Revenue per transaction — proxy for basket size / product mix
    df["revenue_per_tx"] = df["total_revenue"] / df["transaction_count"].clip(lower=1)

    # Volume per active month vs calendar avg (captures seasonality gap)
    df["volume_efficiency"] = (
        df["avg_monthly_volume"] / df["avg_monthly_volume_calendar"].clip(lower=1)
    ).clip(upper=5)

    # Distance to nearest competitor relative to saturation (richer signal)
    df["competitive_pressure"] = (
        df["saturation_score"] / df["nearest_competitor_distance"].clip(lower=1)
    )

    EXTRA_FEATURES = [
        "avg_to_max_ratio",
        "revenue_per_tx",
        "volume_efficiency",
        "competitive_pressure",
    ]

    all_features = FEATURE_COLUMNS + EXTRA_FEATURES

    # ── Handle missing values ────────────────────────────────────────────
    for col in all_features:
        if col not in df.columns:
            logger.warning(f"Feature '{col}' not found — filling with 0")
            df[col] = 0

    # Fill numeric nulls with median (robust to outliers)
    for col in [c for c in all_features if c not in CATEGORICAL_COLS]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # ── Encode categoricals ──────────────────────────────────────────────
    le_map = {}
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("__MISSING__").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_map[col] = le
        logger.info(f"Encoded '{col}': {len(le.classes_)} classes")

    # ── Clip extreme outliers (beyond 99.9th percentile) ─────────────────
    # Prevents a handful of extreme outlets from dominating gradients
    numeric_feats = [c for c in all_features if c not in CATEGORICAL_COLS]
    for col in numeric_feats:
        p999 = df[col].quantile(0.999)
        if p999 > 0:
            df[col] = df[col].clip(upper=p999)

    # ── Target ──────────────────────────────────────────────────────────
    y     = df["max_monthly_volume"].astype(float).values
    y_log = np.log1p(y)

    # ── Groups for spatial CV ────────────────────────────────────────────
    if "geo_cell" in df.columns:
        df["geo_cell"] = df["geo_cell"].fillna(
            df["lat_bucket"].astype(str) + "_" + df["lon_bucket"].astype(str)
        )
        groups = df["geo_cell"].values
    else:
        # Fallback: bucket by rounded lat/lon
        groups = (df["lat_bucket"].astype(str) + "_" + df["lon_bucket"].astype(str)).values

    n_groups = pd.Series(groups).nunique()
    logger.info(f"Spatial groups (geo_cells): {n_groups:,}")

    # ── Build feature matrix ─────────────────────────────────────────────
    X_raw = df[all_features].copy()

    # RobustScaler (median + IQR) — less sensitive to outliers than StandardScaler
    scaler = RobustScaler()
    scale_cols = [c for c in all_features if c not in CATEGORICAL_COLS]
    X_scaled              = X_raw.copy()
    X_scaled[scale_cols]  = scaler.fit_transform(X_raw[scale_cols])

    logger.info(f"Feature matrix: {X_raw.shape[0]:,} rows × {X_raw.shape[1]} features")
    logger.info(f"Target  — mean={y.mean():.1f}  median={np.median(y):.1f}  "
                f"p95={np.percentile(y,95):.1f}  max={y.max():.1f}")

    return X_raw, X_scaled, y, y_log, groups, scaler, le_map, df["outlet_id"].values


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — METRICS HELPER
# ══════════════════════════════════════════════════════════════════════════

def regression_metrics(y_true, y_pred, label="") -> dict:
    """Compute and log a full suite of regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = np.mean(np.abs(y_true - y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    # Median absolute percentage error (robust to outliers)
    mdape = np.median(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

    logger.info(
        f"[{label}] RMSE={rmse:.2f}  MAE={mae:.2f}  "
        f"R²={r2:.4f}  MAPE={mape:.1f}%  MdAPE={mdape:.1f}%"
    )
    return dict(rmse=rmse, mae=mae, r2=r2, mape=mape, mdape=mdape)


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN LIGHTGBM  (primary model)
# ══════════════════════════════════════════════════════════════════════════

LGBM_PARAMS = {
    "objective":         "regression_l1",  # MAE loss — robust to sales outliers
    "metric":            "rmse",
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 30,               # avoid overfitting on sparse geo-cells
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,              # L1
    "reg_lambda":        1.0,              # L2
    "verbosity":         -1,
    "seed":              42,
    "n_jobs":            -1,
}

def train_lgbm_cv(
    X: pd.DataFrame,
    y_log: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, list]:
    """
    5-Fold GroupKFold cross-validation.
    Groups = geo_cell so outlets in the same ~1km² area never split across folds.
    Returns OOF predictions (in original scale) and list of trained models.
    """
    gkf     = GroupKFold(n_splits=n_splits)
    oof_log = np.zeros(len(y_log))
    models  = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
        logger.info(f"  LGBM fold {fold+1}/{n_splits} — "
                    f"train={len(tr_idx):,}  val={len(val_idx):,}")

        tr_data  = lgb.Dataset(X.iloc[tr_idx],  label=y_log[tr_idx])
        val_data = lgb.Dataset(X.iloc[val_idx], label=y_log[val_idx],
                               reference=tr_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=-1),          # silent
        ]

        model = lgb.train(
            LGBM_PARAMS,
            tr_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        oof_log[val_idx] = model.predict(
            X.iloc[val_idx], num_iteration=model.best_iteration
        )
        models.append(model)
        logger.info(f"  Fold {fold+1} best iteration: {model.best_iteration}")

    oof_preds = np.expm1(oof_log)
    return oof_preds, models


def train_lgbm_final(
    X: pd.DataFrame,
    y_log: np.ndarray,
    best_rounds: int,
) -> lgb.Booster:
    """Retrain on 100% of training data using the avg best iteration from CV."""
    logger.info(f"Retraining final LGBM on full train set (rounds={best_rounds}) ...")
    tr_data = lgb.Dataset(X, label=y_log)
    model   = lgb.train(
        {**LGBM_PARAMS, "learning_rate": 0.02},  # lower LR for final model
        tr_data,
        num_boost_round=best_rounds,
    )
    return model


def train_sklearn_gbm_cv(
    X: pd.DataFrame,
    y_log: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, list]:
    """Cross-validated training using scikit-learn's HistGradientBoostingRegressor
    as a fallback when LightGBM is unavailable. Returns OOF preds (original scale)
    and the list of trained estimators.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_log = np.zeros(len(y_log))
    models = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups=groups)):
        logger.info(f"  SKL-GBM fold {fold+1}/{n_splits} — train={len(tr_idx):,}  val={len(val_idx):,}")
        model = HistGradientBoostingRegressor(
            loss="least_absolute_deviation",
            learning_rate=0.05,
            max_iter=1000,
            max_leaf_nodes=63,
            min_samples_leaf=30,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=42,
        )
        model.fit(X.iloc[tr_idx], y_log[tr_idx])
        # predict log-scale
        oof_log[val_idx] = model.predict(X.iloc[val_idx])
        models.append(model)
        logger.info(f"  Fold {fold+1} n_iter_={getattr(model, 'n_iter_', 'NA')}")

    oof_preds = np.expm1(oof_log)
    return oof_preds, models


def train_sklearn_gbm_final(
    X: pd.DataFrame,
    y_log: np.ndarray,
    best_iters: int,
) -> HistGradientBoostingRegressor:
    """Retrain a final sklearn GBDT on full training set using best_iters."""
    logger.info(f"Retraining final SKL-GBM on full train set (max_iter={best_iters}) ...")
    model = HistGradientBoostingRegressor(
        loss="least_absolute_deviation",
        learning_rate=0.02,
        max_iter=max(10, best_iters),
        max_leaf_nodes=63,
        min_samples_leaf=30,
        early_stopping=False,
        random_state=42,
    )
    model.fit(X, y_log)
    return model


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — TRAIN RANDOM FOREST  (reference / ensemble member)
# ══════════════════════════════════════════════════════════════════════════

def train_rf_cv(
    X_scaled: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, list]:
    """RF trained on raw target (no log transform needed — RF is scale-invariant)."""
    gkf   = GroupKFold(n_splits=n_splits)
    oof   = np.zeros(len(y))
    models = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
        logger.info(f"  RF fold {fold+1}/{n_splits}")
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


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════

def log_feature_importance(models: list, feature_names: list):
    importance_sum = np.zeros(len(feature_names))
    for m in models:
        # lightgbm.Booster exposes feature_importance(importance_type='gain')
        if hasattr(m, "feature_importance"):
            try:
                importance_sum += m.feature_importance(importance_type="gain")
                continue
            except Exception:
                pass
        # scikit-learn estimators expose feature_importances_
        if hasattr(m, "feature_importances_"):
            importance_sum += getattr(m, "feature_importances_")
            continue
        # last resort: try attribute 'importance_'
        if hasattr(m, "importance_"):
            importance_sum += getattr(m, "importance_")
            continue
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importance_sum / len(models),
    }).sort_values("importance", ascending=False)

    logger.info("── Top 20 Feature Importances (avg gain across CV folds) ──")
    logger.info("\n" + importance_df.head(20).to_string(index=False))
    importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    return importance_df


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("═" * 65)
    logger.info("  OUTLET POTENTIAL MODEL — TRAINING START")
    logger.info("═" * 65)

    # ── Load ──────────────────────────────────────────────────────────────
    df = load_data()

    # ── Preprocess ────────────────────────────────────────────────────────
    X_raw, X_scaled, y, y_log, groups, scaler, le_map, outlet_ids = preprocess(df)

    # ── Hold-out split (80/20) ────────────────────────────────────────────
    # Convert possible pyarrow-backed series/arrays into numpy-backed objects
    # so sklearn's train_test_split doesn't hit pyarrow indexing issues.
    if hasattr(X_raw, "reset_index"):
        X_raw = X_raw.reset_index(drop=True)
    if hasattr(X_scaled, "reset_index"):
        X_scaled = X_scaled.reset_index(drop=True)

    groups = np.asarray(groups).astype(object)
    outlet_ids = np.asarray(outlet_ids).astype(object)
    y = np.asarray(y)
    y_log = np.asarray(y_log)

    # Stratify by log-binned target so both splits see the full range
    y_bin = pd.qcut(y_log, q=10, labels=False, duplicates="drop")

    (X_train_raw, X_test_raw,
     X_train_sc,  X_test_sc,
     y_train,     y_test,
     y_log_train, y_log_test,
     grp_train,   _,
     ids_train,   ids_test) = train_test_split(
        X_raw, X_scaled, y, y_log, groups, outlet_ids,
        test_size=0.2,
        random_state=42,
        stratify=y_bin,
    )

    logger.info(f"Train: {len(y_train):,}  |  Hold-out test: {len(y_test):,}")

    # ── Gradient-boosted tree model (preferred: LightGBM; fallback: sklearn HistGradientBoosting)
    if lgb is not None:
        logger.info("\n── LightGBM  5-Fold GroupKFold CV ──")
        oof_gbm, cv_models = train_lgbm_cv(X_train_raw, y_log_train, grp_train)
        regression_metrics(y_train, oof_gbm, label="LGBM OOF (train)")

        # avg best iteration across folds → use for final retraining
        avg_best_rounds = int(np.mean([m.best_iteration for m in cv_models]))
        logger.info(f"Average best iteration across folds: {avg_best_rounds}")

        # ── LightGBM final model (retrain on full train set) ──────────────────
        final_gbm = train_lgbm_final(X_train_raw, y_log_train, avg_best_rounds)
        test_preds_gbm = np.expm1(final_gbm.predict(X_test_raw))
        regression_metrics(y_test, test_preds_gbm, label="LGBM Hold-out")
    else:
        logger.info("\n── SKLearn HistGradientBoosting 5-Fold GroupKFold CV (fallback) ──")
        oof_gbm, cv_models = train_sklearn_gbm_cv(X_train_raw, y_log_train, grp_train)
        regression_metrics(y_train, oof_gbm, label="SKL-GBM OOF (train)")

        # avg best iterations across folds → use for final retraining if available
        avg_best_rounds = int(np.mean([getattr(m, "n_iter_", 100) for m in cv_models]))
        logger.info(f"Average n_iter_ across folds: {avg_best_rounds}")

        final_gbm = train_sklearn_gbm_final(X_train_raw, y_log_train, avg_best_rounds)
        test_preds_gbm = np.expm1(final_gbm.predict(X_test_raw))
        regression_metrics(y_test, test_preds_gbm, label="SKL-GBM Hold-out")

    # ── RandomForest CV (reference baseline) ─────────────────────────────
    logger.info("\n── RandomForest  5-Fold GroupKFold CV ──")
    oof_rf, rf_models = train_rf_cv(X_train_sc, y_train, grp_train)
    regression_metrics(y_train, oof_rf, label="RF OOF (train)")

    final_rf = rf_models[0]   # use first fold model as artifact (or retrain below)
    test_preds_rf = final_rf.predict(X_test_sc)
    regression_metrics(y_test, test_preds_rf, label="RF Hold-out")

    # ── Ensemble blend (LGBM 70% + RF 30%) ───────────────────────────────
    logger.info("\n── Ensemble blend (GBM 0.70 + RF 0.30) ──")
    oof_blend  = 0.70 * oof_gbm  + 0.30 * oof_rf
    test_blend = 0.70 * test_preds_gbm + 0.30 * test_preds_rf
    regression_metrics(y_train, oof_blend,  label="Blend OOF (train)")
    metrics_test = regression_metrics(y_test, test_blend, label="Blend Hold-out")

    # ── Sanity check: predicted ceiling ≥ observed avg ────────────────────
    # A predicted ceiling below the outlet's own average is logically impossible
    avg_vol = df.set_index("outlet_id")["avg_monthly_volume_calendar"]

    # ── Feature importance ────────────────────────────────────────────────
    log_feature_importance(cv_models, list(X_raw.columns))

    # ── Assemble full predictions on ALL data ─────────────────────────────
    # Use the final LGBM + first RF fold for full-dataset inference
    all_preds_gbm = np.expm1(final_gbm.predict(X_raw))
    all_preds_rf  = rf_models[0].predict(X_scaled)
    all_preds     = 0.70 * all_preds_gbm + 0.30 * all_preds_rf

    # Floor: predicted_potential must be ≥ observed average
    obs_avg = df["avg_monthly_volume_calendar"].values
    all_preds = np.maximum(all_preds, obs_avg)

    # ── Save predictions ──────────────────────────────────────────────────
    split_label = np.where(np.isin(outlet_ids, ids_test), "test", "train")

    out = pd.DataFrame({
        "outlet_id":            outlet_ids,
        "predicted_potential":  np.round(all_preds, 2),
        "actual_max_monthly":   y,
        "avg_monthly_volume":   df["avg_monthly_volume_calendar"].values,
        "opportunity_gap":      np.round(np.maximum(all_preds - obs_avg, 0), 2),
        "opportunity_pct":      np.round(
            np.maximum(all_preds - obs_avg, 0) / (obs_avg + 1e-6) * 100, 1
        ),
        "split":                split_label,
    }).sort_values("opportunity_gap", ascending=False)

    out.to_parquet(OUTPUT_PRED, index=False)
    logger.info(f"\nSaved predictions → {OUTPUT_PRED}")
    logger.info(f"Top 10 highest-opportunity outlets:\n"
                f"{out[['outlet_id','predicted_potential','avg_monthly_volume','opportunity_gap','opportunity_pct']].head(10).to_string(index=False)}")

    # ── Save model artefacts ──────────────────────────────────────────────
    # Save GBM model: LightGBM has native save API; sklearn models are pickled
    if lgb is not None:
        final_gbm.save_model(str(MODEL_DIR / "lgbm_potential_model.txt"))
    else:
        with open(MODEL_DIR / "skl_gbm_potential_model.pkl", "wb") as f:
            pickle.dump(final_gbm, f)
    with open(MODEL_DIR / "rf_potential_model.pkl", "wb") as f:
        pickle.dump(rf_models[0], f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "label_encoders.pkl", "wb") as f:
        pickle.dump(le_map, f)

    logger.info("\n── Saved artefacts ──")
    logger.info(f"  {MODEL_DIR}/lgbm_potential_model.txt")
    logger.info(f"  {MODEL_DIR}/rf_potential_model.pkl")
    logger.info(f"  {MODEL_DIR}/scaler.pkl")
    logger.info(f"  {MODEL_DIR}/label_encoders.pkl")
    logger.info(f"  {MODEL_DIR}/feature_importance.csv")

    logger.info("\n═" * 65)
    logger.info("  TRAINING COMPLETE")
    logger.info(f"  Hold-out RMSE : {metrics_test['rmse']:.2f}")
    logger.info(f"  Hold-out R²   : {metrics_test['r2']:.4f}")
    logger.info(f"  Hold-out MdAPE: {metrics_test['mdape']:.1f}%")
    logger.info("═" * 65)


if __name__ == "__main__":
    main()