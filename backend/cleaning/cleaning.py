"""
Automated and Custom Data Cleaning module.
Features 6-13:
6. Missing Value Imputation (mean, median, mode, constant, forward/backward fill)
7. Missing Value Dropping (threshold-based row/column removal)
8. Duplicate Removal (exact and partial duplicate purging)
9. Outlier Detection & Treatment (Z-score and IQR-based anomaly flagging with capping/clipping/dropping)
10. Column Header Standardization (automatic whitespace removal and case formatting)
11. Text & String Cleaning (whitespace trimming, capitalization fixes, special character stripping)
12. Data Type Casting (manual and automatic conversion of column types)
13. Data Lineage / Audit Trail (chronological logging of all applied cleaning steps)
"""

import re
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class AuditLogger:
    """
    Feature 13: Data Lineage / Audit Trail tracker.
    Maintains a structured, chronological record of every applied data transformation.
    """
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def log(self, action: str, details: str, rows_affected: int = 0, columns_affected: Optional[List[str]] = None):
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "details": details,
            "rows_affected": rows_affected,
            "columns_affected": columns_affected or []
        }
        self.logs.append(entry)

    def get_logs(self) -> List[Dict[str, Any]]:
        return self.logs

    def to_dataframe(self) -> pd.DataFrame:
        if not self.logs:
            return pd.DataFrame(columns=["timestamp", "action", "details", "rows_affected", "columns_affected"])
        return pd.DataFrame(self.logs)

    def clear(self):
        self.logs.clear()


def standardize_column_headers(
    df: pd.DataFrame,
    case_style: str = "snake_case",
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 10: Column Header Standardization.
    Removes whitespace, strips special characters, formats case (snake_case, lower_case, upper_case, camelCase).
    """
    result = df.copy()
    old_columns = list(result.columns)
    new_columns = []

    for col in old_columns:
        c = str(col).strip()
        c = re.sub(r"[^\w\s]", "", c)
        c = re.sub(r"[\s]+", "_", c)

        if case_style == "snake_case":
            c = c.lower()
        elif case_style == "lower_case":
            c = c.lower().replace("_", " ")
        elif case_style == "upper_case":
            c = c.upper()
        elif case_style == "camelCase":
            parts = c.split("_")
            c = parts[0].lower() + "".join(word.capitalize() for word in parts[1:])

        new_columns.append(c)

    result.columns = new_columns
    if logger:
        logger.log(
            action="Standardize Column Headers",
            details=f"Formatted columns to {case_style}",
            rows_affected=0,
            columns_affected=new_columns
        )
    return result


def impute_missing_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: str = "mean",
    fill_value: Any = None,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 6: Missing Value Imputation.
    Supported strategies: mean, median, mode, constant, ffill, bfill.
    """
    result = df.copy()
    target_cols = columns if columns is not None else list(result.columns)
    affected_rows_count = 0

    for col in target_cols:
        if col not in result.columns:
            continue
        missing_count = int(result[col].isna().sum())
        if missing_count == 0:
            continue

        affected_rows_count += missing_count

        if strategy == "mean" and pd.api.types.is_numeric_dtype(result[col]):
            mean_val = result[col].mean()
            result[col] = result[col].fillna(mean_val)
        elif strategy == "median" and pd.api.types.is_numeric_dtype(result[col]):
            median_val = result[col].median()
            result[col] = result[col].fillna(median_val)
        elif strategy == "mode":
            mode_vals = result[col].mode()
            if len(mode_vals) > 0:
                fill_mode = mode_vals.iloc[0]
                if isinstance(result[col].dtype, pd.CategoricalDtype):
                    if fill_mode not in result[col].cat.categories:
                        result[col] = result[col].cat.add_categories([fill_mode])
                result[col] = result[col].fillna(fill_mode)
        elif strategy == "constant":
            if isinstance(result[col].dtype, pd.CategoricalDtype):
                if fill_value not in result[col].cat.categories:
                    result[col] = result[col].cat.add_categories([fill_value])
            result[col] = result[col].fillna(fill_value)
        elif strategy == "ffill":
            result[col] = result[col].ffill()
        elif strategy == "bfill":
            result[col] = result[col].bfill()

    if logger:
        logger.log(
            action=f"Impute Missing Values ({strategy})",
            details=f"Imputed missing entries using strategy: {strategy}",
            rows_affected=affected_rows_count,
            columns_affected=target_cols
        )
    return result


def drop_missing_values(
    df: pd.DataFrame,
    how: str = "any",
    row_threshold_pct: Optional[float] = None,
    col_threshold_pct: Optional[float] = None,
    columns: Optional[List[str]] = None,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 7: Missing Value Dropping with threshold-based row and column removal.
    """
    result = df.copy()
    initial_rows = len(result)
    initial_cols = len(result.columns)

    if col_threshold_pct is not None:
        col_thresh = (100.0 - col_threshold_pct) / 100.0 * initial_rows
        result = result.dropna(axis=1, thresh=int(col_thresh))

    if row_threshold_pct is not None:
        total_cols = len(result.columns)
        row_thresh = int((100.0 - row_threshold_pct) / 100.0 * total_cols)
        result = result.dropna(axis=0, thresh=row_thresh)
    elif columns:
        result = result.dropna(axis=0, how=how, subset=columns)
    else:
        result = result.dropna(axis=0, how=how)

    rows_dropped = initial_rows - len(result)
    cols_dropped = initial_cols - len(result.columns)

    if logger:
        logger.log(
            action="Drop Missing Values",
            details=f"Dropped {rows_dropped} rows and {cols_dropped} columns based on criteria",
            rows_affected=rows_dropped,
            columns_affected=[]
        )
    return result


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 8: Duplicate Removal (exact and partial duplicate purging).
    """
    initial_rows = len(df)
    result = df.drop_duplicates(subset=subset, keep=keep)  # type: ignore
    rows_dropped = initial_rows - len(result)

    if logger:
        subset_desc = f"subset {subset}" if subset else "entire row match"
        logger.log(
            action="Remove Duplicates",
            details=f"Purged {rows_dropped} duplicate rows ({subset_desc}, keep='{keep}')",
            rows_affected=rows_dropped,
            columns_affected=subset or list(df.columns)
        )
    return result


def treat_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
    action: str = "cap",
    logger: Optional[AuditLogger] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Feature 9: Outlier Detection & Treatment.
    Z-score and IQR-based anomaly flagging with capping/clipping/dropping.
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not a valid numeric column.")

    result = df.copy()
    series = result[column].dropna()

    if method == "iqr":
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - (threshold * iqr)
        upper_bound = q75 + (threshold * iqr)
    elif method == "zscore":
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0:
            lower_bound, upper_bound = mean, mean
        else:
            lower_bound = mean - (threshold * std)
            upper_bound = mean + (threshold * std)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    outlier_mask = (result[column] < lower_bound) | (result[column] > upper_bound)
    outlier_count = int(outlier_mask.sum())

    if action == "cap":
        result[column] = result[column].clip(lower=lower_bound, upper=upper_bound)
        details = f"Capped {outlier_count} outliers to bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
        rows_affected = outlier_count
    elif action == "drop":
        initial_len = len(result)
        result = result[~outlier_mask].reset_index(drop=True)
        rows_affected = initial_len - len(result)
        details = f"Dropped {rows_affected} outlier rows outside bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
    elif action == "flag":
        flag_col = f"{column}_is_outlier"
        result[flag_col] = outlier_mask
        rows_affected = outlier_count
        details = f"Flagged {outlier_count} outliers in column {flag_col}"
    else:
        raise ValueError(f"Unknown outlier treatment action: {action}")

    if logger:
        logger.log(
            action=f"Outlier Treatment ({method.upper()})",
            details=details,
            rows_affected=rows_affected,
            columns_affected=[column]
        )

    stats_info = {
        "method": method,
        "action": action,
        "threshold": threshold,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outliers_detected": outlier_count
    }
    return result, stats_info


def clean_text_columns(
    df: pd.DataFrame,
    columns: List[str],
    strip_whitespace: bool = True,
    case_transformation: Optional[str] = None,
    remove_special_chars: bool = False,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 11: Text & String Cleaning (whitespace trimming, capitalization fixes, special character stripping).
    Safely handles both object/string dtypes and category dtypes without crashing or skipping.
    """
    result = df.copy()

    for col in columns:
        if col not in result.columns:
            continue

        was_category = isinstance(result[col].dtype, pd.CategoricalDtype)
        # Convert to string series for clean vectorized manipulations
        s = result[col].astype(str)

        if strip_whitespace:
            s = s.str.strip()

        if case_transformation == "lower":
            s = s.str.lower()
        elif case_transformation == "upper":
            s = s.str.upper()
        elif case_transformation == "title":
            s = s.str.title()

        if remove_special_chars:
            s = s.apply(lambda x: re.sub(r"[^\w\s]", "", str(x)) if pd.notna(x) else x)

        if was_category:
            result[col] = s.astype("category")
        else:
            result[col] = s

    if logger:
        logger.log(
            action="Text & String Cleaning",
            details=f"Cleaned string formatting for {columns}",
            rows_affected=len(result),
            columns_affected=columns
        )
    return result


def cast_data_types(
    df: pd.DataFrame,
    type_conversions: Dict[str, str],
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 12: Data Type Casting (manual and automatic conversion of column types).
    Valid targets: 'int', 'float', 'string', 'datetime', 'boolean', 'category'.
    """
    result = df.copy()
    succeeded_cols = []

    for col, target_type in type_conversions.items():
        if col not in result.columns:
            continue
        target = target_type.lower()
        try:
            if target in ["int", "int64", "integer"]:
                result[col] = pd.to_numeric(result[col], errors="coerce").astype("Int64")
            elif target in ["float", "float64"]:
                result[col] = pd.to_numeric(result[col], errors="coerce").astype(float)
            elif target in ["str", "string"]:
                result[col] = result[col].astype(str)
            elif target in ["datetime", "date"]:
                result[col] = pd.to_datetime(result[col], errors="coerce")
            elif target in ["bool", "boolean"]:
                result[col] = result[col].astype(bool)
            elif target in ["category", "categorical"]:
                result[col] = result[col].astype("category")
            succeeded_cols.append(col)
        except Exception as e:
            raise ValueError(f"Failed casting {col} to {target}: {str(e)}")

    if logger:
        logger.log(
            action="Data Type Casting",
            details=f"Converted types: {type_conversions}",
            rows_affected=len(result),
            columns_affected=succeeded_cols
        )
    return result


def run_automated_cleaning(
    df: pd.DataFrame,
    logger: Optional[AuditLogger] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive one-click Auto-Clean execution.
    Cleans text even if previously downcast to category, removes duplicates,
    imputes missing values, and caps extreme outliers.
    """
    audit = logger if logger is not None else AuditLogger()
    cleaned = df.copy()

    # 1. Standardize headers
    cleaned = standardize_column_headers(cleaned, case_style="snake_case", logger=audit)

    # 2. Purge exact duplicates
    cleaned = remove_duplicates(cleaned, logger=audit)

    # 3. Clean text & string columns (including category dtypes)
    text_and_cat_cols = [
        c for c in cleaned.columns
        if cleaned[c].dtype == "object"
        or isinstance(cleaned[c].dtype, (pd.StringDtype, pd.CategoricalDtype))
    ]
    if text_and_cat_cols:
        cleaned = clean_text_columns(cleaned, columns=text_and_cat_cols, strip_whitespace=True, logger=audit)

    # 4. Impute missing values (median for numeric, mode for categorical/text)
    for col in cleaned.columns:
        if cleaned[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned = impute_missing_values(cleaned, columns=[col], strategy="median", logger=audit)
            else:
                cleaned = impute_missing_values(cleaned, columns=[col], strategy="mode", logger=audit)

    # 5. Outlier clipping on numeric columns (safe IQR 3.0 threshold)
    num_cols = [c for c in cleaned.columns if pd.api.types.is_numeric_dtype(cleaned[c])]
    outlier_summary = {}
    for nc in num_cols:
        cleaned, info = treat_outliers(cleaned, column=nc, method="iqr", threshold=3.0, action="cap", logger=audit)
        outlier_summary[nc] = info["outliers_detected"]

    summary = {
        "final_rows": len(cleaned),
        "final_columns": len(cleaned.columns),
        "steps_executed": len(audit.get_logs()),
        "outlier_caps": outlier_summary
    }

    return cleaned, summary
