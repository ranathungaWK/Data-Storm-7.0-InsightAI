from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("gold.feature_poi_local")


# ============================================================
# PATHS
# ============================================================

OUTLET_PATH = Path(
    "data/silver/cleaned/coordinates_cleaned.parquet"
)

POI_PATH = Path(
    "data/overture/sri_lanka_places.parquet"
)

OUTPUT_PATH = Path(
    "data/gold/poi_features.parquet"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

SEARCH_RADIUS_METERS = 300


# ============================================================
# LOAD OUTLETS
# ============================================================

logger.info("Loading outlet coordinates...")

outlets = pd.read_parquet(
    OUTLET_PATH
)

outlets_gdf = gpd.GeoDataFrame(
    outlets,
    geometry=gpd.points_from_xy(
        outlets.longitude,
        outlets.latitude
    ),
    crs="EPSG:4326"
)

logger.info(
    f"Loaded outlets: {len(outlets_gdf):,}"
)


# ============================================================
# LOAD POIs
# ============================================================

logger.info("Loading Overture POIs...")

pois = gpd.read_parquet(
    POI_PATH
)

logger.info(
    f"Loaded POIs: {len(pois):,}"
)


# ============================================================
# CATEGORY EXTRACTION
# ============================================================

pois["poi_category"] = (
    pois["basic_category"]
    .fillna("unknown")
    .astype(str)
    .str.lower()
)

TARGET_CATEGORIES = [
    "education",
    "transport",
    "healthcare",
    "shopping",
    "restaurant",
    "accommodation",
    "entertainment"
]

pois = pois[
    pois["poi_category"]
    .isin(TARGET_CATEGORIES)
]

logger.info(
    f"Filtered POIs: {len(pois):,}"
)


# ============================================================
# PROJECT CRS
# ============================================================

outlets_gdf = outlets_gdf.to_crs(
    epsg=3857
)

pois = pois.to_crs(
    epsg=3857
)

# ============================================================
# BUFFER OUTLETS
# ============================================================

outlets_gdf["geometry"] = (
    outlets_gdf.geometry.buffer(
        SEARCH_RADIUS_METERS
    )
)

logger.info(
    "Created outlet buffers"
)


# ============================================================
# SPATIAL JOIN
# ============================================================

logger.info(
    "Running spatial join..."
)

joined = gpd.sjoin(
    pois,
    outlets_gdf,
    predicate="within",
    how="inner"
)

logger.info(
    f"Spatial matches: {len(joined):,}"
)


# ============================================================
# AGGREGATE COUNTS
# ============================================================

feature_df = (

    joined.groupby(
        ["outlet_id", "poi_category"]
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()

)

# ============================================================
# ENSURE ALL COLUMNS EXIST
# ============================================================

for col in TARGET_CATEGORIES:

    if col not in feature_df.columns:

        feature_df[col] = 0


# ============================================================
# DERIVED FEATURES
# ============================================================

feature_df["commercial_score"] = (
    feature_df["shopping"]
    + feature_df["restaurant"]
    + feature_df["accommodation"]
)

feature_df["accessibility_score"] = (
    feature_df["transport"]
)

feature_df["education_score"] = (
    feature_df["education"]
)

feature_df["healthcare_score"] = (
    feature_df["healthcare"]
)

feature_df["lifestyle_score"] = (
    feature_df["entertainment"]
)

# ============================================================
# SAFE OPTIONAL FEATURE DEFAULTS
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