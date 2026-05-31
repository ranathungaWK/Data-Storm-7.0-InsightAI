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

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    df["date_key"] = df["date"].dt.strftime("%Y-%m-%d")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    # HOLIDAY FEATURE ENGINEERING

    holiday_df["date"] = pd.to_datetime(holiday_df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    holiday_lookup = holiday_df[["date"]].dropna().drop_duplicates().copy()
    holiday_lookup["date_key"] = holiday_lookup["date"].dt.strftime("%Y-%m-%d")

    df = df.merge(
        holiday_lookup[["date_key"]],
        on="date_key",
        how="left",
        indicator=True
    )

    df["is_holiday"] = np.where(df["_merge"] == "both", 1, 0)

    df.drop(columns=["_merge"], inplace=True)

    # HOLIDAY DISTANCE FEATURE

    holiday_dates = pd.to_datetime(
        holiday_lookup["date"],
        errors="coerce"
    ).dropna().to_numpy(dtype="datetime64[ns]")

    def nearest_holiday_days(x: pd.Timestamp) -> float:
        if pd.isna(x) or holiday_dates.size == 0:
            return np.nan
        x_value = np.datetime64(x.to_datetime64())
        deltas = np.abs((holiday_dates - x_value) / np.timedelta64(1, "D"))
        return float(np.min(deltas))

    df["days_to_nearest_holiday"] = df["date"].apply(nearest_holiday_days)

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
        distributor_monthly = (
            distributor_df
            .groupby("month", as_index=False)
            .mean(numeric_only=True)
        )

        df = df.merge(
            distributor_monthly,
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

    return df.drop(columns=["date_key"])