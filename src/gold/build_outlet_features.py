from pathlib import Path

import pandas as pd

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

)


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