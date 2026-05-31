"""Silver Layer — Seasonality data validation and normalization."""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("silver.clean_seasonality")


INPUT_PATH = Path("src/bronze/distributor_seasonality_details.csv")
OUTPUT_PATH = Path("data/silver/cleaned/seasonality_cleaned.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


SEASONALITY_SCORE_MAP = {
    "favorable": 2,
    "moderate": 1,
    "un-favorable": -1,
    "unfavorable": -1,
}


def clean_seasonality(config=None):
    """Validate seasonality index values and completeness."""
    logger.info("Loading raw seasonality data...")
    seasonality_df = pd.read_csv(INPUT_PATH)
    seasonality_df.columns = [column.strip().lower() for column in seasonality_df.columns]

    if "seasonality_index" in seasonality_df.columns:
        seasonality_df["seasonality_index"] = (
            seasonality_df["seasonality_index"].astype(str).str.strip().str.lower()
        )
        seasonality_df["seasonality_score"] = (
            seasonality_df["seasonality_index"].map(SEASONALITY_SCORE_MAP).fillna(0).astype(int)
        )

    if {"year", "month"}.issubset(seasonality_df.columns):
        seasonality_df["year"] = pd.to_numeric(seasonality_df["year"], errors="coerce").astype("Int64")
        seasonality_df["month"] = pd.to_numeric(seasonality_df["month"], errors="coerce").astype("Int64")

    seasonality_df = seasonality_df.dropna(subset=[col for col in ["distributor_id", "year", "month"] if col in seasonality_df.columns]).reset_index(drop=True)

    logger.info(f"Cleaned seasonality shape: {seasonality_df.shape}")
    seasonality_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved cleaned seasonality -> {OUTPUT_PATH}")
    return seasonality_df


if __name__ == "__main__":
    clean_seasonality()
