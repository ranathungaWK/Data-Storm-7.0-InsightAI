"""
Silver Layer — Transactions Cleaning Pipeline
---------------------------------------------
Forensic + Demand-aware + Feature-ready processing
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.silver.dq_checks import (
    check_duplicates,
    check_nulls,
    check_value_range,
    check_negative_volumes,
    check_zero_volumes
)

from src.silver.quarantine import QuarantineManager

from src.utils.logger import get_logger

logger = get_logger("silver.clean_transactions")

# PATHS

INPUT_PATH = Path("src/bronze/transactions_history_final.csv")

OUTPUT_PATH = Path("data/silver/cleaned/transactions_cleaned.parquet")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# LOAD DATA

df = pd.read_csv(INPUT_PATH)

logger.info(f"Loaded transactions: {df.shape}")

# QUARANTINE INIT

qm = QuarantineManager()

# NORMALIZATION

df.columns = [c.strip().lower() for c in df.columns]

# ============================================================
# DATE CONSTRUCTION
# ============================================================

df["date"] = pd.to_datetime(
    dict(
        year=df["year"],
        month=df["month"],
        day=1
    ),
    errors="coerce"
)

logger.info("Constructed synthetic monthly date column")

# DQ 1 — DUPLICATES (Business Key)

dup = check_duplicates(
    df=df,
    key_columns=[
        "outlet_id",
        "year",
        "month",
        "distributor_id",
        "sku_id"
    ],
    dataset_name="transactions",
    id_column="outlet_id"
)

qm.add_rejections(dup["failed_records"])

df = df.drop(index=dup["failed_records"]["row_index"].unique())

# DQ 2 — NULL CHECK (critical fields)

null = check_nulls(
    df=df,
    mandatory_columns=["outlet_id", "volume_liters"],
    dataset_name="transactions"
)

qm.add_rejections(null["failed_records"])

df = df.drop(index=null["failed_records"]["row_index"].unique())

# DQ 3 — VALUE RANGE (basic sanity check)

# Example: no absurd extreme values (domain dependent)
range_check = check_value_range(
    df=df,
    column="volume_liters",
    min_val=-500,
    max_val=50000,
    dataset_name="transactions"
)

qm.add_rejections(range_check["failed_records"])

df = df.drop(index=range_check["failed_records"]["row_index"].unique())

# FORENSIC LAYER (DO NOT DROP)

neg = check_negative_volumes(
    df=df,
    volume_column="volume_liters",
    dataset_name="transactions"
)

zero = check_zero_volumes(
    df=df,
    volume_column="volume_liters",
    dataset_name="transactions"
)

# These are NOT dropped — only tagged
qm.add_rejections(neg["failed_records"])
qm.add_rejections(zero["failed_records"])

# DEMAND CENSORING LOGIC (CORE ML PREP)

df["censored_volume"] = df["volume_liters"].copy()

# Option A: keep negatives as 0 for modeling stability
df.loc[df["censored_volume"] < 0, "censored_volume"] = 0

# Option B: cap extreme outliers (winsorization style)
upper_cap = df["censored_volume"].quantile(0.99)
df["censored_volume"] = np.minimum(df["censored_volume"], upper_cap)

# FINAL CLEANING

df = df.reset_index(drop=True)

logger.info(f"Final transactions shape: {df.shape}")

# SAVE SILVER DATASET

df.to_parquet(OUTPUT_PATH, index=False)

logger.info(f"Saved transactions → {OUTPUT_PATH}")

# FLUSH QUARANTINE

qm.flush()

logger.info("Transactions pipeline completed")