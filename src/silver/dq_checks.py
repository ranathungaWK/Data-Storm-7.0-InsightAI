#Industrial-Grade Reusable Data Quality Framework

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("silver.dq_checks")


# REJECTION SCHEMA

REJECTION_COLUMNS = [
    "dataset",
    "rule_id",
    "check_type",
    "severity",
    "column",
    "key_value",
    "row_index",
    "reason",
    "original_value",
    "timestamp",
]

# INTERNAL UTILITIES

def _current_timestamp() -> str:
    return datetime.now().isoformat()


def _empty_rejection_df() -> pd.DataFrame:
    return pd.DataFrame(columns=REJECTION_COLUMNS)


def _build_summary(
    dataset_name: str,
    rule_id: str,
    check_type: str,
    rows_checked: int,
    violations: int,
    severity: str,
) -> dict:

    violation_rate = (
        round((violations / rows_checked) * 100, 4)
        if rows_checked > 0
        else 0
    )

    return {
        "dataset": dataset_name,
        "rule_id": rule_id,
        "check_type": check_type,
        "severity": severity,
        "rows_checked": rows_checked,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "timestamp": _current_timestamp(),
    }


def _package_result(
    failed_records: pd.DataFrame,
    summary: dict,
) -> dict:

    return {
        "failed_records": failed_records,
        "summary": summary,
    }


def _create_rejection_df(
    df: pd.DataFrame,
    dataset_name: str,
    rule_id: str,
    check_type: str,
    severity: str,
    column: str,
    reason: str,
    original_value_column: Optional[str] = None,
    id_column: str = "outlet_id",
) -> pd.DataFrame:

    if df.empty:
        return _empty_rejection_df()

    rejection_df = pd.DataFrame({
        "dataset": dataset_name,
        "rule_id": rule_id,
        "check_type": check_type,
        "severity": severity,
        "column": column,
        "key_value": (
            df[id_column].astype(str)
            if id_column in df.columns
            else ""
        ),
        "row_index": df.index,
        "reason": reason,
        "original_value": (
            df[original_value_column].astype(str)
            if original_value_column
            else ""
        ),
        "timestamp": _current_timestamp(),
    })

    return rejection_df[REJECTION_COLUMNS]


# STRUCTURAL CHECKS

def check_duplicates(
    df: pd.DataFrame,
    key_columns: list[str],
    dataset_name: str,
    id_column: str = "outlet_id",
    rule_id: str = "DQ_001",
) -> dict:
    """
    Detect duplicate records based on composite key.
    """

    duplicate_mask = df.duplicated(
        subset=key_columns,
        keep="first"
    )

    failed_df = df[duplicate_mask]

    logger.info(
        f"[{dataset_name}] Duplicate check "
        f"({key_columns}) -> {len(failed_df):,} duplicates"
    )

    rejection_df = _create_rejection_df(
        df=failed_df,
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="DUPLICATE",
        severity="CRITICAL",
        column="|".join(key_columns),
        reason=f"Duplicate composite key: {key_columns}",
        id_column=id_column,
    )

    summary = _build_summary(
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="DUPLICATE",
        rows_checked=len(df),
        violations=len(rejection_df),
        severity="CRITICAL",
    )

    return _package_result(rejection_df, summary)


def check_nulls(
    df: pd.DataFrame,
    mandatory_columns: list[str],
    dataset_name: str,
    id_column: str = "outlet_id",
    rule_id: str = "DQ_002",
) -> dict:
    """
    Detect null or empty mandatory fields.
    """

    all_failures = []

    for col in mandatory_columns:

        null_mask = (
            df[col].isna()
            |
            (df[col].astype(str).str.strip() == "")
        )

        failed_df = df[null_mask]

        logger.info(
            f"[{dataset_name}] Null check '{col}' "
            f"-> {len(failed_df):,} violations"
        )

        rejection_df = _create_rejection_df(
            df=failed_df,
            dataset_name=dataset_name,
            rule_id=rule_id,
            check_type="NULL_CHECK",
            severity="CRITICAL",
            column=col,
            reason=f"Mandatory column '{col}' is null/empty",
            original_value_column=col,
            id_column=id_column,
        )

        all_failures.append(rejection_df)

    final_rejections = (
        pd.concat(all_failures, ignore_index=True)
        if all_failures
        else _empty_rejection_df()
    )

    summary = _build_summary(
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="NULL_CHECK",
        rows_checked=len(df),
        violations=len(final_rejections),
        severity="CRITICAL",
    )

    return _package_result(final_rejections, summary)


def check_referential_integrity(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    fk_column: str,
    pk_column: str,
    dataset_name: str,
    ref_dataset_name: str,
    id_column: str = "outlet_id",
    rule_id: str = "DQ_003",
) -> dict:
    """
    Validate foreign key integrity.
    """

    valid_keys = set(
        ref_df[pk_column]
        .dropna()
        .unique()
    )

    orphan_mask = ~df[fk_column].isin(valid_keys)

    failed_df = df[orphan_mask]

    logger.info(
        f"[{dataset_name}] Referential integrity "
        f"{fk_column} -> "
        f"{ref_dataset_name}.{pk_column} "
        f"-> {len(failed_df):,} violations"
    )

    rejection_df = _create_rejection_df(
        df=failed_df,
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="REFERENTIAL_INTEGRITY",
        severity="CRITICAL",
        column=fk_column,
        reason=(
            f"Foreign key not found in "
            f"{ref_dataset_name}.{pk_column}"
        ),
        original_value_column=fk_column,
        id_column=id_column,
    )

    summary = _build_summary(
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="REFERENTIAL_INTEGRITY",
        rows_checked=len(df),
        violations=len(rejection_df),
        severity="CRITICAL",
    )

    return _package_result(rejection_df, summary)


def check_value_range(
    df: pd.DataFrame,
    column: str,
    min_val: float,
    max_val: float,
    dataset_name: str,
    id_column: str = "outlet_id",
    rule_id: str = "DQ_004",
) -> dict:
    """
    Detect out-of-range numeric values.
    """

    range_mask = (
        (df[column] < min_val)
        |
        (df[column] > max_val)
    )

    failed_df = df[range_mask]

    logger.info(
        f"[{dataset_name}] Range check '{column}' "
        f"[{min_val}, {max_val}] "
        f"-> {len(failed_df):,} violations"
    )

    rejection_df = _create_rejection_df(
        df=failed_df,
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="VALUE_RANGE",
        severity="CRITICAL",
        column=column,
        reason=(
            f"Value outside allowed range "
            f"[{min_val}, {max_val}]"
        ),
        original_value_column=column,
        id_column=id_column,
    )

    summary = _build_summary(
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="VALUE_RANGE",
        rows_checked=len(df),
        violations=len(rejection_df),
        severity="CRITICAL",
    )

    return _package_result(rejection_df, summary)


def check_format(
    df: pd.DataFrame,
    column: str,
    regex_pattern: str,
    dataset_name: str,
    id_column: str = "outlet_id",
    rule_id: str = "DQ_005",
) -> dict:
    """
    Validate regex format compliance.
    """

    non_null_df = df[df[column].notna()].copy()

    invalid_mask = ~non_null_df[column].astype(str).str.match(
        regex_pattern
    )

    failed_df = non_null_df[invalid_mask]

    logger.info(
        f"[{dataset_name}] Format check '{column}' "
        f"-> {len(failed_df):,} violations"
    )

    rejection_df = _create_rejection_df(
        df=failed_df,
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="FORMAT_CHECK",
        severity="CRITICAL",
        column=column,
        reason=f"Invalid format for '{column}'",
        original_value_column=column,
        id_column=id_column,
    )

    summary = _build_summary(
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="FORMAT_CHECK",
        rows_checked=len(df),
        violations=len(rejection_df),
        severity="CRITICAL",
    )

    return _package_result(rejection_df, summary)

# FORENSIC CHECKS

def check_negative_volumes(
    df: pd.DataFrame,
    volume_column: str,
    dataset_name: str = "transactions",
    id_column: str = "outlet_id",
    rule_id: str = "FQ_001",
) -> dict:
    """
    Detect negative transaction volumes.

    IMPORTANT:
    These may represent:
    - returns
    - reversals
    - corrections

    Therefore:
    - tagged as WARNING
    - NOT automatically removed
    """

    negative_mask = df[volume_column] < 0

    failed_df = df[negative_mask]

    logger.info(
        f"[{dataset_name}] Negative volume forensic check "
        f"-> {len(failed_df):,} observations"
    )

    rejection_df = _create_rejection_df(
        df=failed_df,
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="NEGATIVE_VOLUME",
        severity="WARNING",
        column=volume_column,
        reason=(
            "Negative transaction volume detected "
            "(possible reversal/return)"
        ),
        original_value_column=volume_column,
        id_column=id_column,
    )

    summary = _build_summary(
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="NEGATIVE_VOLUME",
        rows_checked=len(df),
        violations=len(rejection_df),
        severity="WARNING",
    )

    return _package_result(rejection_df, summary)


def check_zero_volumes(
    df: pd.DataFrame,
    volume_column: str,
    dataset_name: str = "transactions",
    id_column: str = "outlet_id",
    rule_id: str = "FQ_002",
) -> dict:
    """
    Detect zero transaction volumes.

    Often:
    - system adjustments
    - placeholder records
    - fee entries
    """

    zero_mask = df[volume_column] == 0

    failed_df = df[zero_mask]

    logger.info(
        f"[{dataset_name}] Zero volume forensic check "
        f"-> {len(failed_df):,} observations"
    )

    rejection_df = _create_rejection_df(
        df=failed_df,
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="ZERO_VOLUME",
        severity="INFO",
        column=volume_column,
        reason="Zero-volume transaction detected",
        original_value_column=volume_column,
        id_column=id_column,
    )

    summary = _build_summary(
        dataset_name=dataset_name,
        rule_id=rule_id,
        check_type="ZERO_VOLUME",
        rows_checked=len(df),
        violations=len(rejection_df),
        severity="INFO",
    )

    return _package_result(rejection_df, summary)