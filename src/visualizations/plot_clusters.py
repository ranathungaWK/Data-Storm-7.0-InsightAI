from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger("visualization.plot_clusters")


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
    "reports/figures/cluster_pca.png"
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

df = features_df.merge(
    clusters_df,
    on="outlet_id",
    how="left"
)

logger.info(
    f"Merged shape: {df.shape}"
)

# ============================================================
# FEATURE SELECTION
# ============================================================

model_features = [

    "avg_monthly_volume",
    "commercial_score",
    "mobility_score",
    "education_score",
    "healthcare_score",
    "lifestyle_score",
    "sales_frequency",
    "sku_diversity"

]

model_features = [
    c for c in model_features
    if c in df.columns
]

logger.info(
    f"Using features: {model_features}"
)

X = df[model_features].fillna(0)

# ============================================================
# SCALE
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# PCA
# ============================================================

logger.info("Running PCA...")

pca = PCA(n_components=2)

components = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame({

    "PC1": components[:, 0],
    "PC2": components[:, 1],
    "cluster": df["cluster"]

})

# ============================================================
# PLOT
# ============================================================

logger.info("Generating plot...")

plt.figure(figsize=(12, 8))

scatter = plt.scatter(

    plot_df["PC1"],
    plot_df["PC2"],

    c=plot_df["cluster"],

    alpha=0.6

)

plt.title(
    "Outlet Cluster Visualization (PCA)"
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300
)

logger.info(
    f"Saved PCA plot -> {OUTPUT_PATH}"
)