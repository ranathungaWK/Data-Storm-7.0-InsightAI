from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from src.utils.logger import get_logger

logger = get_logger("models.potential_scoring")


# ============================================================
# PATHS
# ============================================================

FEATURE_PATH = Path(
    "data/gold/outlet_features.parquet"
)

CLUSTER_PATH = Path(
    "data/models/outlet_clusters.parquet"
)

OUTPUT_PATH = Path(
    "data/models/outlet_potential_scores.parquet"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

logger.info(
    "Loading feature datasets..."
)

features_df = pd.read_parquet(
    FEATURE_PATH
)

cluster_df = pd.read_parquet(
    CLUSTER_PATH
)

logger.info(
    f"Features: {features_df.shape}"
)

logger.info(
    f"Clusters: {cluster_df.shape}"
)


# ============================================================
# MERGE
# ============================================================

df = features_df.merge(
    cluster_df[
        ["outlet_id", "cluster"]
    ],
    on="outlet_id",
    how="left"
)

logger.info(
    f"Merged dataset: {df.shape}"
)


# ============================================================
# REQUIRED FEATURES
# ============================================================

required_columns = [

    # actual performance
    "total_volume",
    "total_bill_value",

    # business capability
    "sku_diversity",
    "distributor_diversity",

    # spatial intelligence
    "commercial_score",
    "mobility_score",
    "education_score",
    "healthcare_score",
    "lifestyle_score",

    # consistency
    "sales_frequency",

    # clustering
    "cluster"
]

missing_cols = [
    col for col in required_columns
    if col not in df.columns
]

if missing_cols:

    logger.warning(
        f"Missing columns: {missing_cols}"
    )

    for col in missing_cols:

        df[col] = 0


# ============================================================
# CLUSTER BENCHMARKS
# ============================================================

logger.info(
    "Building cluster benchmarks..."
)

cluster_benchmarks = (

    df.groupby("cluster")
    .agg(

        cluster_avg_volume=(
            "total_volume",
            "mean"
        ),

        cluster_avg_bill=(
            "total_bill_value",
            "mean"
        ),

        cluster_avg_sku=(
            "sku_diversity",
            "mean"
        ),

        cluster_avg_commercial=(
            "commercial_score",
            "mean"
        ),

        cluster_avg_mobility=(
            "mobility_score",
            "mean"
        ),

        cluster_avg_frequency=(
            "sales_frequency",
            "mean"
        )

    )
    .reset_index()

)

logger.info(
    f"Cluster benchmark shape: "
    f"{cluster_benchmarks.shape}"
)


# ============================================================
# MERGE BENCHMARKS
# ============================================================

df = df.merge(
    cluster_benchmarks,
    on="cluster",
    how="left"
)


# ============================================================
# EXPECTED PERFORMANCE
# ============================================================

logger.info(
    "Estimating expected outlet performance..."
)

df["expected_volume"] = (

    (
        df["cluster_avg_volume"] * 0.40
    )

    +

    (
        df["commercial_score"]
        /
        (
            df["cluster_avg_commercial"]
            + 1e-6
        )
    )
    *
    df["cluster_avg_volume"]
    * 0.20

    +

    (
        df["mobility_score"]
        /
        (
            df["cluster_avg_mobility"]
            + 1e-6
        )
    )
    *
    df["cluster_avg_volume"]
    * 0.15

    +

    (
        df["sku_diversity"]
        /
        (
            df["cluster_avg_sku"]
            + 1e-6
        )
    )
    *
    df["cluster_avg_volume"]
    * 0.15

    +

    (
        df["sales_frequency"]
        /
        (
            df["cluster_avg_frequency"]
            + 1e-6
        )
    )
    *
    df["cluster_avg_volume"]
    * 0.10

)


# ============================================================
# POTENTIAL GAP
# ============================================================

logger.info(
    "Computing hidden potential gaps..."
)

df["raw_potential_gap"] = (

    df["expected_volume"]
    -
    df["total_volume"]

)

# only positive opportunity

df["raw_potential_gap"] = (
    df["raw_potential_gap"]
    .clip(lower=0)
)


# ============================================================
# NORMALIZE SCORE
# ============================================================

logger.info(
    "Normalizing potential scores..."
)

scaler = MinMaxScaler()

df["potential_score"] = scaler.fit_transform(

    df[["raw_potential_gap"]]

) * 100


# ============================================================
# POTENTIAL TIERS
# ============================================================

def assign_potential_tier(score):

    if score >= 80:
        return "Very High"

    elif score >= 60:
        return "High"

    elif score >= 40:
        return "Medium"

    elif score >= 20:
        return "Low"

    else:
        return "Very Low"


df["potential_tier"] = (
    df["potential_score"]
    .apply(assign_potential_tier)
)


# ============================================================
# GLOBAL RANK
# ============================================================

df["potential_rank"] = (

    df["potential_score"]
    .rank(
        ascending=False,
        method="dense"
    )

    .astype(int)

)


# ============================================================
# TOP OUTLETS
# ============================================================

top_outlets = (

    df[

        [
            "outlet_id",
            "cluster",
            "total_volume",
            "expected_volume",
            "potential_score",
            "potential_tier",
            "potential_rank"
        ]

    ]

    .sort_values(
        by="potential_score",
        ascending=False
    )

    .head(20)

)

logger.info(
    "\nTop hidden opportunity outlets:\n"
)

logger.info(
    f"\n{top_outlets}"
)


# ============================================================
# SAVE
# ============================================================

output_columns = [

    "outlet_id",
    "cluster",

    "total_volume",
    "expected_volume",

    "raw_potential_gap",

    "potential_score",
    "potential_tier",
    "potential_rank"
]

final_df = df[
    output_columns
]

final_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

logger.info(
    f"Saved potential scores -> {OUTPUT_PATH}"
)