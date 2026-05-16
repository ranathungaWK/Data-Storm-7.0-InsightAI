from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("models.cluster_profiles")


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
    "data/models/cluster_profiles.parquet"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

logger.info("Loading datasets...")

features_df = pd.read_parquet(FEATURE_PATH)

clusters_df = pd.read_parquet(CLUSTER_PATH)

df = features_df.merge(
    clusters_df,
    on="outlet_id",
    how="left"
)

logger.info(
    f"Merged shape: {df.shape}"
)


# ============================================================
# PROFILE FEATURES
# ============================================================

profile_columns = [

    "total_volume",
    "avg_monthly_volume",
    "total_bill_value",

    "commercial_score",
    "mobility_score",
    "education_score",
    "healthcare_score",
    "lifestyle_score",

    "sales_frequency",
    "sku_diversity"

]

profile_columns = [
    c for c in profile_columns
    if c in df.columns
]

logger.info(
    f"Using profile columns: {profile_columns}"
)


# ============================================================
# BUILD CLUSTER PROFILES
# ============================================================

cluster_profiles = (

    df.groupby("cluster")[profile_columns]

    .mean()

    .round(2)

    .reset_index()

)

# ============================================================
# OUTLET COUNTS
# ============================================================

counts = (

    df.groupby("cluster")
    .size()
    .reset_index(name="outlet_count")

)

cluster_profiles = cluster_profiles.merge(
    counts,
    on="cluster",
    how="left"
)

# ============================================================
# RANK CLUSTERS
# ============================================================

cluster_profiles["cluster_score"] = (

    cluster_profiles["avg_monthly_volume"]

    +

    cluster_profiles["commercial_score"] * 5

    +

    cluster_profiles["mobility_score"] * 3

)

cluster_profiles = cluster_profiles.sort_values(
    by="cluster_score",
    ascending=False
)

logger.info(
    "\nCluster Profiles:\n"
)

logger.info(cluster_profiles)

# ============================================================
# SAVE
# ============================================================

cluster_profiles.to_parquet(
    OUTPUT_PATH,
    index=False
)

logger.info(
    f"Saved cluster profiles -> {OUTPUT_PATH}"
)