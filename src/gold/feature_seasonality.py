"""
Gold Layer — Seasonality Feature Engineering
-------------------------------------------
Transforms time into predictive demand signals.
"""

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("gold.feature_seasonality")


# MAIN FEATURE FUNCTION

def build_seasonality_features(
    df: pd.DataFrame,
    holiday_df: pd.DataFrame,
    distributor_df: pd.DataFrame
) -> pd.DataFrame:

    logger.info("Building seasonality features...")

    # DATE NORMALIZATION

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    # HOLIDAY FEATURE ENGINEERING

    holiday_df["date"] = pd.to_datetime(holiday_df["date"], errors="coerce")

    df = df.merge(
        holiday_df[["date"]],
        on="date",
        how="left",
        indicator=True
    )

    df["is_holiday"] = np.where(df["_merge"] == "both", 1, 0)

    df.drop(columns=["_merge"], inplace=True)

    # HOLIDAY DISTANCE FEATURE

    holiday_dates = holiday_df["date"].dropna().unique()

    df["days_to_nearest_holiday"] = df["date"].apply(
        lambda x: np.min(np.abs((holiday_dates - x).astype("timedelta64[D]")))
        if pd.notnull(x) else np.nan
    )

    # WEEKEND FEATURE

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # MONTHLY SEASONALITY INDEX

    df["monthly_seasonality"] = (
        df.groupby("month")["censored_volume"]
        .transform("mean")
        .fillna(0)
    )

    # DISTRIBUTOR SEASONALITY (BUSINESS SIGNAL)

    if "month" in distributor_df.columns:

        distributor_df["month"] = distributor_df["month"].astype(int)

        df = df.merge(
            distributor_df,
            on="month",
            how="left"
        )

    # ROLLING BEHAVIOR (TEMPORAL MEMORY)

    df = df.sort_values(["outlet_id", "date"])

    df["rolling_7d_mean"] = (
        df.groupby("outlet_id")["censored_volume"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    df["rolling_30d_mean"] = (
        df.groupby("outlet_id")["censored_volume"]
        .transform(lambda x: x.rolling(30, min_periods=1).mean())
    )

    # TREND SIGNAL

    df["trend_signal"] = (
        df["rolling_7d_mean"] - df["rolling_30d_mean"]
    )

    logger.info(f"Seasonality features created: {df.shape}")

    return df