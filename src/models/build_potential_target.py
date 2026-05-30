from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("models.build_potential_target")


# Paths (mirror repository conventions)
TXN_PATH = Path("data/silver/cleaned/transactions_cleaned.parquet")
FEATURE_PATH = Path("data/gold/outlet_features.parquet")
OUTPUT_PATH = Path("data/models/outlet_potential_target.parquet")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():

    logger.info("Loading transactions and features...")

    txn_df = pd.read_parquet(TXN_PATH)
    feat_df = pd.read_parquet(FEATURE_PATH)

    logger.info(f"Transactions: {txn_df.shape}")
    logger.info(f"Features: {feat_df.shape}")

    # ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(txn_df["date"]):
        txn_df["date"] = pd.to_datetime(txn_df["date"])

    # monthly aggregation per outlet
    txn_df["month"] = txn_df["date"].dt.to_period("M")

    monthly = (
        txn_df
        .groupby(["outlet_id", "month"], observed=True)
        ["volume_liters"]
        .sum()
        .reset_index()
        .rename(columns={"volume_liters": "monthly_volume"})
    )

    logger.info(f"Monthly aggregation shape: {monthly.shape}")

    # compute max monthly volume per outlet = potential
    potential = (
        monthly
        .groupby("outlet_id", observed=True)["monthly_volume"]
        .max()
        .reset_index()
        .rename(columns={"monthly_volume": "max_monthly_volume"})
    )

    logger.info(f"Potential (max monthly) shape: {potential.shape}")

    # merge with features to get actual (avg_monthly_volume)
    merged = (
        potential
        .merge(
            feat_df[
                ["outlet_id", "avg_monthly_volume"]
                ]
            .drop_duplicates(subset=["outlet_id"]),
            on="outlet_id",
            how="left"
        )
    )

    # define actual sales (fallback to avg_monthly_volume, else 0)
    merged["avg_monthly_volume"] = merged["avg_monthly_volume"].fillna(0)
    merged = merged.rename(columns={"avg_monthly_volume": "actual_avg_monthly_volume"})

    # gap = potential - actual
    merged["gap_volume"] = merged["max_monthly_volume"] - merged["actual_avg_monthly_volume"]
    merged["gap_pct"] = merged["gap_volume"] / (merged["actual_avg_monthly_volume"] + 1e-9)

    # clip negative gaps to 0 (if actual > potential) — keep as negative for diagnostics? keep negative to show oversaturated outlets
    # merged["gap_volume"] = merged["gap_volume"].clip(lower=0)

    logger.info(f"Saving potential/gap -> {OUTPUT_PATH}")

    merged.to_parquet(OUTPUT_PATH, index=False)

    logger.info("Done.")


if __name__ == "__main__":
    main()
