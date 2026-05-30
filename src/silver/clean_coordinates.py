"""
Silver Layer — Coordinate Cleaning Pipeline
------------------------------------------
Geospatial validation + anomaly detection + clustering readiness
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.silver.dq_checks import (
    check_nulls,
    check_value_range,
    check_duplicates,
    _create_rejection_df
)

from src.silver.quarantine import QuarantineManager

from src.utils.logger import get_logger

logger = get_logger("silver.clean_coordinates")


# PATHS

INPUT_PATH = Path("src/bronze/outlet_coordinates.csv")

OUTPUT_PATH = Path("data/silver/cleaned/coordinates_cleaned.parquet")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# LOAD DATA

df = pd.read_csv(INPUT_PATH)

df.columns = [c.strip().lower() for c in df.columns]

logger.info(f"Loaded coordinates: {df.shape}")


qm = QuarantineManager()


# STEP 1 — NULL CHECKS

null_result = check_nulls(
    df=df,
    mandatory_columns=["latitude", "longitude"],
    dataset_name="coordinates"
)

qm.add_rejections(null_result["failed_records"])

df = df.drop(
    index=null_result["failed_records"]["row_index"].unique()
)


# STEP 2 — GEOGRAPHIC RANGE VALIDATION

# Earth constraints
lat_range = check_value_range(
    df=df,
    column="latitude",
    min_val=-90,
    max_val=90,
    dataset_name="coordinates"
)

lon_range = check_value_range(
    df=df,
    column="longitude",
    min_val=-180,
    max_val=180,
    dataset_name="coordinates"
)

qm.add_rejections(lat_range["failed_records"])
qm.add_rejections(lon_range["failed_records"])

bad_idx = set(
    lat_range["failed_records"]["row_index"].tolist()
    +
    lon_range["failed_records"]["row_index"].tolist()
)

df = df.drop(index=list(bad_idx))


# STEP 3 — DUPLICATE LOCATION DETECTION

df["coord_hash"] = (
    df["latitude"].round(5).astype(str)
    + "_"
    + df["longitude"].round(5).astype(str)
)

duplicate_result = check_duplicates(
    df=df,
    key_columns=["coord_hash"],
    dataset_name="coordinates",
    id_column="outlet_id"
)

qm.add_rejections(
    duplicate_result["failed_records"]
)

df = df.drop(
    index=duplicate_result["failed_records"]["row_index"].unique()
)


# STEP 4 — GEOSPATIAL OUTLIER DETECTION

# detect extreme spatial noise using IQR logic
lat_q1, lat_q3 = df["latitude"].quantile([0.25, 0.75])
lon_q1, lon_q3 = df["longitude"].quantile([0.25, 0.75])

lat_iqr = lat_q3 - lat_q1
lon_iqr = lon_q3 - lon_q1

spatial_outliers = df[
    (df["latitude"] < lat_q1 - 3 * lat_iqr) |
    (df["latitude"] > lat_q3 + 3 * lat_iqr) |
    (df["longitude"] < lon_q1 - 3 * lon_iqr) |
    (df["longitude"] > lon_q3 + 3 * lon_iqr)
]

from src.silver.dq_checks import _create_rejection_df

outlier_rejections = _create_rejection_df(
    df=spatial_outliers,
    dataset_name="coordinates",
    rule_id="DQ_007",
    check_type="SPATIAL_OUTLIER",
    severity="WARNING",
    column="latitude|longitude",
    reason="Spatial coordinate outlier detected",
    id_column="outlet_id"
)

qm.add_rejections(outlier_rejections)

df = df.drop(index=spatial_outliers.index)


# STEP 5 — FEATURE ENGINEERING (CLUSTER READY)

df["lat_bucket"] = df["latitude"].round(2)
df["lon_bucket"] = df["longitude"].round(2)

df["geo_cell"] = (
    df["lat_bucket"].astype(str)
    + "_"
    + df["lon_bucket"].astype(str)
)

df = df.drop(columns=["coord_hash"])


# SAVE SILVER DATASET

df.reset_index(drop=True, inplace=True)

df.to_parquet(OUTPUT_PATH, index=False)

logger.info(f"Saved cleaned coordinates → {OUTPUT_PATH}")


# FLUSH QUARANTINE


qm.flush()

logger.info("Coordinates pipeline completed")