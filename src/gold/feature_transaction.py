import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("gold.feature_transaction")

# FEATURE ENGINE FUNCTION

def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Building transaction-level features...")

    # GROUP BY OUTLET (core behavioral unit)

    grouped = df.groupby("outlet_id")

    features = grouped.agg(
        total_volume=("censored_volume", "sum"),
        avg_volume=("censored_volume", "mean"),
        std_volume=("censored_volume", "std"),
        max_volume=("censored_volume", "max"),
        min_volume=("censored_volume", "min"),
        transaction_count=("censored_volume", "count"),
    ).reset_index()

    # DERIVED BEHAVIORAL SIGNALS

    features["volatility"] = (
        features["std_volume"] /
        (features["avg_volume"] + 1e-6)
    )

    features["range_volume"] = (
        features["max_volume"] - features["min_volume"]
    )

    # STABILITY SCORE (IMPORTANT FOR CLUSTERING)

    features["stability_score"] = 1 / (
        features["volatility"] + 1e-6
    )

    # LOG TRANSFORM (for skewed distributions)

    features["log_total_volume"] = np.log1p(
        features["total_volume"]
    )
    logger.info(f"Generated features: {features.shape}")

    return features