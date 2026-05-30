from pathlib import Path

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from src.utils.logger import get_logger

logger = get_logger(
    "models.opportunity_recommendations"
)


# ============================================================
# PATHS
# ============================================================

FEATURE_PATH = Path(
    "data/gold/outlet_features.parquet"
)

CLUSTER_PATH = Path(
    "data/models/outlet_clusters.parquet"
)

PROFILE_PATH = Path(
    "data/models/cluster_profiles.parquet"
)

OUTPUT_PATH = Path(
    "data/models/opportunity_recommendations.parquet"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

logger.info("Loading datasets...")

features_df = pd.read_parquet(
    FEATURE_PATH
)

clusters_df = pd.read_parquet(
    CLUSTER_PATH
)

profiles_df = pd.read_parquet(
    PROFILE_PATH
)

# ============================================================
# MERGE
# ============================================================

df = (

    features_df

    .merge(
        clusters_df,
        on="outlet_id",
        how="left"
    )

)

logger.info(
    f"Merged shape: {df.shape}"
)

# ============================================================
# GET CLUSTER BENCHMARKS
# ============================================================

benchmark_map = profiles_df.set_index(
    "cluster"
)["avg_monthly_volume"].to_dict()

df["expected_volume"] = (
    df["cluster"]
    .map(benchmark_map)
)

# ============================================================
# PERFORMANCE GAP
# ============================================================

df["volume_gap"] = (

    df["expected_volume"]

    -

    df["avg_monthly_volume"]

)

# ============================================================
# OPPORTUNITY CONDITIONS
# ============================================================

df["high_commercial"] = (
    df["commercial_score"] >= 2
)

df["high_mobility"] = (
    df["mobility_score"] >= 1
)

df["high_lifestyle"] = (
    df["lifestyle_score"] >= 1
)

# ============================================================
# RAW OPPORTUNITY SCORE
# ============================================================

df["raw_opportunity_score"] = (

    df["volume_gap"] * 0.6

    +

    df["commercial_score"] * 10

    +

    df["mobility_score"] * 5

    +

    df["lifestyle_score"] * 5

)

# ============================================================
# NORMALIZE SCORE
# ============================================================

scaler = MinMaxScaler()

df["opportunity_score"] = scaler.fit_transform(

    df[["raw_opportunity_score"]]

) * 100

# ============================================================
# TIERS
# ============================================================

def assign_tier(score):

    if score >= 80:
        return "Very High"

    elif score >= 60:
        return "High"

    elif score >= 40:
        return "Medium"

    else:
        return "Low"


df["opportunity_tier"] = (
    df["opportunity_score"]
    .apply(assign_tier)
)

# ============================================================
# RANK
# ============================================================

df = df.sort_values(
    by="opportunity_score",
    ascending=False
)

df["opportunity_rank"] = range(
    1,
    len(df) + 1
)

# ============================================================
# FINAL OUTPUT
# ============================================================

output_columns = [

    "outlet_id",

    "cluster",

    "avg_monthly_volume",

    "expected_volume",

    "volume_gap",

    "commercial_score",

    "mobility_score",

    "lifestyle_score",

    "opportunity_score",

    "opportunity_tier",

    "opportunity_rank"

]

recommendation_df = df[
    output_columns
]

# ============================================================
# PREVIEW
# ============================================================

logger.info(
    "\nTop Opportunity Outlets:\n"
)

logger.info(
    recommendation_df.head(20)
)

# ============================================================
# SAVE
# ============================================================

recommendation_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

logger.info(
    f"Saved recommendations -> {OUTPUT_PATH}"
)