"""
Holiday Cleaning Pipeline
"""


from pathlib import Path

import pandas as pd

from src.silver.dq_checks import (
    check_duplicates,
    check_nulls,
)

from src.silver.quarantine import (
    QuarantineManager
)

from src.utils.logger import get_logger


logger = get_logger("silver.clean_holidays")


# PATH CONFIGURATION


INPUT_PATH = (
    Path("src/bronze/holiday_list.csv")
)

OUTPUT_PATH = (
    Path("data/silver/cleaned/holidays_cleaned.parquet")
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

# LOAD DATA

logger.info("Loading holiday dataset...")

holiday_df = pd.read_csv(INPUT_PATH)

logger.info(
    f"Loaded {len(holiday_df):,} rows"
)


# INITIALIZE QUARANTINE SYSTEM

qm = QuarantineManager()


# DQ CHECK — DUPLICATES

duplicate_result = check_duplicates(
    df=holiday_df,
    key_columns=holiday_df.columns.tolist(),
    dataset_name="holidays",
    id_column=holiday_df.columns[0]
)

qm.add_rejections(
    duplicate_result["failed_records"]
)

duplicate_indices = (
    duplicate_result["failed_records"]
    ["row_index"]
    .unique()
)

holiday_df = holiday_df.drop(
    index=duplicate_indices
)

logger.info(
    f"Removed {len(duplicate_indices):,} duplicates"
)


# DQ CHECK — NULLS

mandatory_columns = holiday_df.columns.tolist()

null_result = check_nulls(
    df=holiday_df,
    mandatory_columns=mandatory_columns,
    dataset_name="holidays",
    id_column=holiday_df.columns[0]
)

qm.add_rejections(
    null_result["failed_records"]
)

null_indices = (
    null_result["failed_records"]
    ["row_index"]
    .unique()
)

holiday_df = holiday_df.drop(
    index=null_indices
)

logger.info(
    f"Removed {len(null_indices):,} null violations"
)


# DATE STANDARDIZATION

date_columns = [
    col for col in holiday_df.columns
    if "date" in col.lower()
]

logger.info(
    f"Detected date columns: {date_columns}"
)

for col in date_columns:

    holiday_df[col] = pd.to_datetime(
        holiday_df[col],
        errors="coerce"
    )

    invalid_dates = holiday_df[
        holiday_df[col].isna()
    ]

    if not invalid_dates.empty:

        logger.warning(
            f"{len(invalid_dates):,} invalid "
            f"dates detected in '{col}'"
        )

        invalid_dates = invalid_dates.copy()

        invalid_dates["dataset"] = "holidays"
        invalid_dates["rule_id"] = "DQ_006"
        invalid_dates["check_type"] = "INVALID_DATE"
        invalid_dates["severity"] = "CRITICAL"
        invalid_dates["column"] = col
        invalid_dates["reason"] = (
            "Invalid date format"
        )

        qm.add_rejections(
            invalid_dates
        )

        holiday_df = holiday_df[
            holiday_df[col].notna()
        ]


# FINAL STANDARDIZATION

holiday_df.columns = [
    col.strip().lower()
    for col in holiday_df.columns
]

holiday_df = holiday_df.reset_index(
    drop=True
)

logger.info(
    f"Final cleaned shape: {holiday_df.shape}"
)


# SAVE SILVER DATASET

holiday_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

logger.info(
    f"Saved cleaned dataset -> {OUTPUT_PATH}"
)


# FLUSH QUARANTINE RECORDS

qm.flush()

logger.info(
    "Holiday cleaning pipeline completed."
)