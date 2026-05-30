"""
Gold Layer — Outlet Profile Feature Engineering (Enhanced)
----------------------------------------------------------
Adds latent potential signals + normalization + spatial readiness
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

logger = get_logger("gold.feature_outlet_profile")


def build_outlet_profile(transactions: pd.DataFrame,
                         outlet_master: pd.DataFrame) -> pd.DataFrame:

    logger.info("Building enhanced outlet profile features...")

    # STEP 1 — TRANSACTION AGGREGATION

    tx = transactions.groupby("outlet_id").agg(
        total_volume=("censored_volume", "sum"),
        avg_volume=("censored_volume", "mean"),
        std_volume=("censored_volume", "std"),
        max_volume=("censored_volume", "max"),
        min_volume=("censored_volume", "min"),
        transaction_count=("censored_volume", "count"),
    ).reset_index()

    # STEP 2 — BEHAVIORAL STABILITY FEATURES

    tx["volatility"] = tx["std_volume"] / (tx["avg_volume"] + 1e-6)
    tx["stability_score"] = 1 / (tx["volatility"] + 1e-6)

    tx["range_volume"] = tx["max_volume"] - tx["min_volume"]

    tx["volume_per_transaction"] = (
        tx["total_volume"] / (tx["transaction_count"] + 1e-6)
    )

    # STEP 3 — LATENT POTENTIAL SIGNALS (KEY ADDITION)

    tx["peak_to_avg_ratio"] = (
        tx["max_volume"] / (tx["avg_volume"] + 1e-6)
    )

    tx["demand_headroom_proxy"] = (
        tx["max_volume"] - tx["avg_volume"]
    )

    tx["consistency_index"] = (
        tx["avg_volume"] / (tx["volatility"] + 1e-6)
    )

    # STEP 4 — LOG TRANSFORMS (SKEW HANDLING)

    tx["log_total_volume"] = np.log1p(tx["total_volume"])
    tx["log_avg_volume"] = np.log1p(tx["avg_volume"])

    # STEP 5 — MERGE OUTLET MASTER

    outlet_master.columns = [
        c.strip().lower() for c in outlet_master.columns
    ]

    profile = tx.merge(
        outlet_master,
        on="outlet_id",
        how="left"
    )

    # STEP 6 — MISSINGNESS SIGNAL (DATA QUALITY AS FEATURE)

    profile["master_data_completeness"] = (
        1 - profile.isnull().mean(axis=1)
    )

    # STEP 7 — NORMALIZED INTENSITY SCORE (IMPORTANT FIX)

    raw_score = (
        profile["log_total_volume"] *
        profile["stability_score"]
    ).values.reshape(-1, 1)

    scaler = StandardScaler()
    profile["demand_intensity_score"] = scaler.fit_transform(raw_score)

    # STEP 8 — FINAL CLEANUP

    profile = profile.replace([np.inf, -np.inf], np.nan)
    profile = profile.fillna(0)

    logger.info(f"Outlet profile shape: {profile.shape}")

    return profile