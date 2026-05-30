from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.utils.logger import get_logger

logger = get_logger("gold.build_outlet_features")


# ============================================================
# PATHS
# ============================================================

MASTER_PATH = Path(
    "data/silver/cleaned/outlet_master_cleaned.parquet"
)
COORD_PATH = Path(
    "data/silver/cleaned/coordinates_cleaned.parquet"
)


TXN_PATH = Path(
    "data/silver/cleaned/transactions_cleaned.parquet"
)

POI_PATH = Path(
    "data/gold/poi_features.parquet"
)

OUTPUT_PATH = Path(
    "data/gold/outlet_features.parquet"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

logger.info("Loading datasets...")

master_df = pd.read_parquet(MASTER_PATH)

coord_df = pd.read_parquet(COORD_PATH)

txn_df = pd.read_parquet(TXN_PATH)

poi_df = pd.read_parquet(POI_PATH)

logger.info(
    f"Master: {master_df.shape}"
)

logger.info(
    f"Coordinates: {coord_df.shape}"
)

logger.info(
    f"Transactions: {txn_df.shape}"
)

logger.info(
    f"POI: {poi_df.shape}"
)


# ============================================================
# COMPETITION FEATURES
# ============================================================

logger.info("Building competition density features...")

coord_xy = coord_df[["longitude", "latitude"]].to_numpy(dtype=float)
coord_xy_3857 = coord_df.copy()

# project to meters using a lightweight approximation (works on small lon/lat spans)
# for Sri Lanka-scale distances, scale degrees to meters via EPSG:3857 transform approximation
# better CRS handling would require geopandas, but this keeps the pipeline dependency-light.
lon = coord_df["longitude"].to_numpy(dtype=float)
lat = coord_df["latitude"].to_numpy(dtype=float)
earth_radius = 6378137.0
x = np.radians(lon) * earth_radius
y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * earth_radius
coord_xy_3857 = np.column_stack([x, y])

tree = cKDTree(coord_xy_3857)

count_500m = []
count_1km = []
nearest_distance = []
saturation_score = []

neighbors_1km = tree.query_ball_point(coord_xy_3857, r=1000)

for idx, nbrs in enumerate(neighbors_1km):
    # remove self if present
    nbrs = [n for n in nbrs if n != idx]
    count_1km.append(len(nbrs))

    close_500 = [n for n in nbrs if np.hypot(*(coord_xy_3857[n] - coord_xy_3857[idx])) <= 500]
    count_500m.append(len(close_500))

    if nbrs:
        dists = [np.hypot(*(coord_xy_3857[n] - coord_xy_3857[idx])) for n in nbrs]
        nearest = float(min(dists))
    else:
        nearest = np.nan
    nearest_distance.append(nearest)

    # simple saturation proxy: denser nearby competitors + closer nearest competitor => higher saturation
    density_component = len(nbrs) / 10.0
    proximity_component = 1.0 / (1.0 + (nearest if np.isfinite(nearest) else 1000.0) / 250.0)
    saturation_score.append(density_component + proximity_component)

competition_df = pd.DataFrame(
    {
        "outlet_id": coord_df["outlet_id"],
        "competitor_count_500m": count_500m,
        "competitor_count_1km": count_1km,
        "nearest_competitor_distance": nearest_distance,
        "saturation_score": saturation_score,
    }
)


# ============================================================
# TRANSACTION FEATURES
# ============================================================

logger.info(
    "Building transaction aggregations..."
)

txn_features = (

    txn_df

    .groupby("outlet_id")

    .agg(

        total_volume=(
            "volume_liters",
            "sum"
        ),

        total_revenue=(
            "total_bill_value",
            "sum"
        ),

        avg_monthly_volume=(
            "volume_liters",
            "mean"
        ),

        avg_monthly_revenue=(
            "total_bill_value",
            "mean"
        ),

        volume_std=(
            "volume_liters",
            "std"
        ),

        transaction_count=(
            "sku_id",
            "count"
        ),

        sku_diversity=(
            "sku_id",
            "nunique"
        ),

        distributor_diversity=(
            "distributor_id",
            "nunique"
        ),

        active_months=(
            "date",
            "nunique"
        )

    )

    .reset_index()

)

# ============================================================
# TRUE MONTHLY AVERAGE (fix for avg_monthly_volume semantics)
# Compute monthly totals per outlet and take their mean so
# `avg_monthly_volume` reflects average monthly sales (not average transaction size).
# ============================================================

# ensure date is datetime
if not pd.api.types.is_datetime64_any_dtype(txn_df["date"]):
    txn_df["date"] = pd.to_datetime(txn_df["date"])

txn_df["month"] = txn_df["date"].dt.to_period("M")

monthly = (
    txn_df
    .groupby(["outlet_id", "month"], observed=True)["volume_liters"]
    .sum()
    .reset_index()
    .rename(columns={"volume_liters": "monthly_volume"})
)

true_monthly_avg = (
    monthly
    .groupby("outlet_id", observed=True)["monthly_volume"]
    .mean()
    .reset_index()
    .rename(columns={"monthly_volume": "true_avg_monthly_volume"})
)

# merge true monthly average into txn_features and prefer it over transaction-mean
txn_features = txn_features.merge(
    true_monthly_avg,
    on="outlet_id",
    how="left",
)

txn_features["avg_monthly_volume"] = (
    txn_features["true_avg_monthly_volume"].fillna(txn_features["avg_monthly_volume"])
)

# drop helper column
txn_features = txn_features.drop(columns=["true_avg_monthly_volume"], errors="ignore")


# ============================================================
# Active months (months with transactions) should be counted in months, not unique dates.
# Replace `active_months` aggregated from unique `date` values with month-based counts.
# Also compute calendar-month average (span-based) for comparison.
# ============================================================

# recompute active_months as number of unique months with activity
active_months_by_month = (
    monthly
    .groupby("outlet_id", observed=True)["month"]
    .nunique()
    .reset_index()
    .rename(columns={"month": "active_months_months"})
)

txn_features = txn_features.merge(
    active_months_by_month,
    on="outlet_id",
    how="left",
)

# if active_months_months is missing, fall back to previous active_months value (which may be date-based)
txn_features["active_months"] = (
    txn_features["active_months_months"].fillna(txn_features.get("active_months", 0)).astype(int)
)

# drop helper column
txn_features = txn_features.drop(columns=["active_months_months"], errors="ignore")

# dataset calendar span in months (inclusive)
min_date = txn_df["date"].min()
max_date = txn_df["date"].max()
if pd.notnull(min_date) and pd.notnull(max_date):
    total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
else:
    total_months = None

if total_months and total_months > 0:
    txn_features["avg_monthly_volume_calendar"] = (
        txn_features["total_volume"] / float(total_months)
    )
else:
    txn_features["avg_monthly_volume_calendar"] = txn_features["avg_monthly_volume"]

# ============================================================
# STABILITY FEATURES
# ============================================================

txn_features["volume_cv"] = (

    txn_features["volume_std"]

    /

    (
        txn_features["avg_monthly_volume"]
        + 1e-6
    )

)

txn_features["sales_frequency"] = (

    txn_features["active_months"]

    / 12

)

logger.info(
    f"Transaction features: {txn_features.shape}"
)


# ============================================================
# MERGE ALL FEATURES
# ============================================================

logger.info(
    "Merging feature tables..."
)

feature_df = (

    master_df

    .merge(
        coord_df,
        on="outlet_id",
        how="left"
    )

    .merge(
        txn_features,
        on="outlet_id",
        how="left"
    )

    .merge(
        poi_df,
        on="outlet_id",
        how="left"
    )

    .merge(
        competition_df,
        on="outlet_id",
        how="left"
    )

)


# ============================================================
# MOBILITY SCORE
# Derived from `accessibility_weighted_score` (distance-decay transport influence).
# Scaled to a small range so downstream thresholds (e.g. >=1) remain meaningful.
# If the source column is missing, default to 0.
# ============================================================
if "accessibility_weighted_score" in feature_df.columns:
    max_acc = feature_df["accessibility_weighted_score"].max() or 0.0
    if max_acc > 0:
        feature_df["mobility_score"] = (
            feature_df["accessibility_weighted_score"] / (max_acc + 1e-6)
        ) * 3.0
    else:
        feature_df["mobility_score"] = 0.0
else:
    feature_df["mobility_score"] = 0.0

# keep mobility_score numeric
try:
    feature_df["mobility_score"] = feature_df["mobility_score"].astype(float)
except Exception:
    pass


# ============================================================
# OPTIONAL FEATURE SAFETY
# ============================================================

optional_features = [

    "commercial_score",
    "accessibility_score",
    "education_score",
    "healthcare_score",
    "lifestyle_score"

]

for col in optional_features:

    if col not in feature_df.columns:

        logger.warning(
            f"Missing optional feature: {col}"
        )

        feature_df[col] = 0


# ============================================================
# NULL HANDLING
# ============================================================

numeric_columns = feature_df.select_dtypes(
    include=["number"]
).columns

feature_df[numeric_columns] = (

    feature_df[numeric_columns]

    .fillna(0)

)

logger.info(
    f"Final feature table shape: {feature_df.shape}"
)


# ============================================================
# SAVE
# ============================================================

feature_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

logger.info(
    f"Saved outlet features -> {OUTPUT_PATH}"
)
