from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from src.utils.logger import get_logger


logger = get_logger("models.clustering")


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "data/gold/outlet_features.parquet"
)

OUTPUT_PATH = Path(
    "data/models/outlet_clusters.parquet"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

logger.info(
    "Loading outlet feature dataset..."
)

df = pd.read_parquet(INPUT_PATH)

logger.info(
    f"Loaded shape: {df.shape}"
)

required_features = [

    "cooler_count",
    "commercial_score",
    "accessibility_score",
    "lifestyle_score",
    "sku_diversity",
    "avg_monthly_volume",
    "avg_monthly_bill"

]

available_features = [
    col for col in required_features
    if col in df.columns
]

missing_features = [
    col for col in required_features
    if col not in df.columns
]

logger.warning(
    f"Missing features: {missing_features}"
)

model_df = df[available_features].copy()

# ============================================================
# SELECT FEATURES
# ============================================================

feature_columns = [

    # spatial
    "commercial_score",
    "accessibility_score",
    "lifestyle_score",

    # outlet
    "cooler_count",

    # transactions
    "avg_monthly_volume",
    "avg_monthly_revenue",
    "sku_diversity",
    "active_months"
]


# ============================================================
# OPTIONAL CATEGORICAL FEATURES
# ============================================================

categorical_columns = []

if "outlet_size" in df.columns:
    categorical_columns.append("outlet_size")

if "outlet_type" in df.columns:
    categorical_columns.append("outlet_type")


# ============================================================
# LABEL ENCODING
# ============================================================

for col in categorical_columns:

    logger.info(
        f"Encoding categorical column: {col}"
    )

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )

    feature_columns.append(col)


# ============================================================
# BUILD MODEL DATASET
# ============================================================

model_df = df[
    ["outlet_id"] + feature_columns
].copy()

# fill missing values

model_df[feature_columns] = (
    model_df[feature_columns]
    .fillna(0)
)

logger.info(
    f"Model dataset shape: {model_df.shape}"
)


# ============================================================
# SCALING
# ============================================================

logger.info(
    "Scaling features..."
)

scaler = StandardScaler()

X = scaler.fit_transform(
    model_df[feature_columns]
)

logger.info(
    f"Scaled matrix shape: {X.shape}"
)


# ============================================================
# KMEANS CLUSTERING
# ============================================================

N_CLUSTERS = 8

logger.info(
    f"Running KMeans (k={N_CLUSTERS})..."
)

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=20
)

clusters = kmeans.fit_predict(X)

model_df["cluster"] = clusters


# ============================================================
# EVALUATION
# ============================================================

score = silhouette_score(
    X,
    clusters
)

logger.info(
    f"Silhouette Score: {score:.4f}"
)


# ============================================================
# CLUSTER SUMMARY
# ============================================================

cluster_summary = (
    model_df
    .groupby("cluster")
    .size()
    .reset_index(name="outlet_count")
)

logger.info(
    f"\n{cluster_summary}"
)


# ============================================================
# SAVE OUTPUT
# ============================================================

output_df = model_df[
    ["outlet_id", "cluster"]
]

output_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

logger.info(
    f"Saved clusters -> {OUTPUT_PATH}"
)