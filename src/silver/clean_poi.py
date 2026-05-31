"""Silver Layer — POI data standardization."""

from pathlib import Path

import geopandas as gpd

from src.utils.logger import get_logger

logger = get_logger("silver.clean_poi")


INPUT_PATH = Path("data/overture/sri_lanka_places.parquet")
OUTPUT_PATH = Path("data/silver/cleaned/poi_cleaned.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def clean_poi(config=None):
    """Standardize and validate scraped POI data."""
    logger.info("Loading raw POI data...")
    poi_df = gpd.read_parquet(INPUT_PATH)
    poi_df.columns = [column.strip().lower() for column in poi_df.columns]

    if "basic_category" in poi_df.columns:
        poi_df["basic_category"] = poi_df["basic_category"].astype(str).str.strip().str.lower()

    if "taxonomy" in poi_df.columns:
        poi_df["taxonomy"] = poi_df["taxonomy"].astype(str).str.strip()

    logger.info(f"Cleaned POI shape: {poi_df.shape}")
    poi_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved cleaned POI -> {OUTPUT_PATH}")
    return poi_df


if __name__ == "__main__":
    clean_poi()
