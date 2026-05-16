"""
Silver Layer — Outlet Master Cleaning Pipeline
----------------------------------------------
Entity standardization + segmentation readiness
"""

from pathlib import Path
import pandas as pd

from src.silver.dq_checks import (
    check_duplicates,
    check_nulls
)

from src.silver.quarantine import QuarantineManager

from src.utils.logger import get_logger

logger = get_logger("silver.clean_outlet_master")


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "src/bronze/outlet_master.csv"
)

OUTPUT_PATH = Path(
    "data/silver/cleaned/outlet_master_cleaned.parquet"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

logger.info("Loading outlet master dataset...")

df = pd.read_csv(INPUT_PATH)

logger.info(f"Loaded shape: {df.shape}")


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

df.columns = [
    c.strip().lower()
    for c in df.columns
]

logger.info(
    f"Normalized columns: {df.columns.tolist()}"
)


# ============================================================
# QUARANTINE INIT
# ============================================================

qm = QuarantineManager()


# ============================================================
# DQ CHECK — DUPLICATES
# ============================================================

dup = check_duplicates(
    df=df,
    key_columns=["outlet_id"],
    dataset_name="outlet_master",
    id_column="outlet_id"
)

qm.add_rejections(
    dup["failed_records"]
)

df = df.drop(
    index=dup["failed_records"]["row_index"].unique()
)

logger.info(
    f"Removed duplicate outlet IDs"
)


# ============================================================
# DQ CHECK — NULLS
# ============================================================

mandatory_columns = ["outlet_id"]

nulls = check_nulls(
    df=df,
    mandatory_columns=mandatory_columns,
    dataset_name="outlet_master",
    id_column="outlet_id"
)

qm.add_rejections(
    nulls["failed_records"]
)

df = df.drop(
    index=nulls["failed_records"]["row_index"].unique()
)

logger.info(
    f"Removed null violations"
)


# ============================================================
# TEXT STANDARDIZATION
# ============================================================

text_columns = [
    col for col in df.columns
    if df[col].dtype == "object"
]

for col in text_columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

logger.info(
    "Standardized categorical text fields"
)


# ============================================================
# MISSINGNESS FEATURES
# ============================================================

df["master_completeness_score"] = (
    1 - df.isnull().mean(axis=1)
)

logger.info(
    "Generated completeness score"
)


# ============================================================
# FINAL CLEANUP
# ============================================================

df = df.reset_index(drop=True)

logger.info(
    f"Final cleaned shape: {df.shape}"
)


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

df.to_parquet(
    OUTPUT_PATH,
    index=False
)

logger.info(
    f"Saved cleaned outlet master -> {OUTPUT_PATH}"
)


# ============================================================
# FLUSH QUARANTINE
# ============================================================

qm.flush()

logger.info(
    "Outlet master pipeline completed"
)