"""
Automated and Custom Data Cleaning module.
Features 6-13:
6. Missing Value Imputation (mean, median, mode, constant, forward/backward fill)
7. Missing Value Dropping (threshold-based with math.ceil non-null rounding)
8. Duplicate Removal (exact and partial duplicate purging)
9. Outlier Detection & Treatment (Z-score and IQR-based anomaly flagging with capping/clipping/dropping)
10. Column Header Standardization (collision-proof naming with explicit mapping)
11. Text & String Cleaning (preserving nulls, whitespace trimming, casing, special character stripping)
12. Data Type Casting (with unparsed token tracking and robust boolean parsing)
13. Data Lineage / Audit Trail (chronological logging of all applied cleaning steps)
"""

import re
import math
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
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Feature 10: Column Header Standardization with Collision Guards.
    Formats column names safely, preventing duplicate header creation by appending _2, _3.
    Returns modified DataFrame and the explicit mapping dictionary (Original -> New).
    """
    result = df.copy()
    old_columns = list(result.columns)
    new_columns = []
    mapping = {}
    seen_counts: Dict[str, int] = {}

    for col in old_columns:
        c = str(col).strip()
        # Preserve meaningful symbols like currency ($), percent (%), underscores
        c = re.sub(r"[\s\-]+", "_", c)
        c = re.sub(r"[^\w$%_]", "", c)

        if case_style == "snake_case":
            c = c.lower()
        elif case_style == "lower_case":
            c = c.lower().replace("_", " ")
        elif case_style == "upper_case":
            c = c.upper()
        elif case_style == "camelCase":
            parts = c.split("_")
            c = parts[0].lower() + "".join(word.capitalize() for word in parts[1:])

        # Collision guard
        if c in seen_counts:
            seen_counts[c] += 1
            final_col = f"{c}_{seen_counts[c]}"
        else:
            seen_counts[c] = 1
            final_col = c

        new_columns.append(final_col)
        mapping[str(col)] = final_col

    result.columns = new_columns
    if logger:
        logger.log(
            action="Standardize Column Headers",
            details=f"Formatted columns to {case_style}. Mapping: {mapping}",
            rows_affected=0,
            columns_affected=new_columns
        )
    return result, mapping


def impute_missing_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: str = "mean",
    fill_value: Any = None,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 6: Missing Value Imputation.
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
    Feature 7: Missing Value Dropping with math.ceil non-null requirement rounding.
    Prevents silent distortion of missing value thresholds.
    """
    result = df.copy()
    initial_rows = len(result)
    initial_cols = len(result.columns)

    if col_threshold_pct is not None:
        # Minimum non-null rows required = ceil((100 - pct) / 100 * initial_rows)
        min_non_null_rows = math.ceil((100.0 - col_threshold_pct) / 100.0 * initial_rows)
        result = result.dropna(axis=1, thresh=min_non_null_rows)

    if row_threshold_pct is not None:
        total_cols = len(result.columns)
        min_non_null_cols = math.ceil((100.0 - row_threshold_pct) / 100.0 * total_cols)
        result = result.dropna(axis=0, thresh=min_non_null_cols)
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
    Feature 8: Duplicate Removal.
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
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not a valid numeric column.")

    result = df.copy()
    series = result[column].dropna()

    if len(series) == 0:
        return result, {"method": method, "action": action, "outliers_detected": 0}

    if method == "iqr":
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        if iqr == 0.0:
            # Handle zero-IQR bounds (e.g., highly repeated values): fall back to min/max
            lower_bound = series.min()
            upper_bound = series.max()
        else:
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
    Feature 11: Text & String Cleaning with strict null preservation.
    """
    result = df.copy()

    for col in columns:
        if col not in result.columns:
            continue

        was_category = isinstance(result[col].dtype, pd.CategoricalDtype)
        series = result[col].copy()

        def clean_val(val):
            if pd.isna(val):
                return np.nan
            v = str(val)
            if strip_whitespace:
                v = v.strip()
            if case_transformation == "lower":
                v = v.lower()
            elif case_transformation == "upper":
                v = v.upper()
            elif case_transformation == "title":
                v = v.title()
            if remove_special_chars:
                v = re.sub(r"[^\w\s]", "", v)
            return v

        cleaned_series = series.apply(clean_val)

        if was_category:
            result[col] = cleaned_series.astype("category")
        else:
            result[col] = cleaned_series

    if logger:
        logger.log(
            action="Text & String Cleaning",
            details=f"Cleaned string formatting for {columns} (preserved nulls)",
            rows_affected=len(result),
            columns_affected=columns
        )
    return result


BOOLEAN_TRUE_VALUES = {"true", "yes", "y", "1", "1.0", "t"}
BOOLEAN_FALSE_VALUES = {"false", "no", "n", "0", "0.0", "f"}


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """
    Explicit boolean parser.
    """
    def to_bool(val):
        if pd.isna(val):
            return pd.NA
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        v_str = str(val).strip().lower()
        if v_str in BOOLEAN_TRUE_VALUES:
            return True
        elif v_str in BOOLEAN_FALSE_VALUES:
            return False
        else:
            raise ValueError(f"Cannot parse ambiguous value '{val}' as boolean.")

    return series.apply(to_bool).astype("boolean")


def cast_data_types(
    df: pd.DataFrame,
    type_conversions: Dict[str, str],
    logger: Optional[AuditLogger] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Feature 12: Data Type Casting with unparsed token accounting.
    Returns converted DataFrame and dictionary of unparsed tokens coerced to NaN/NaT.
    """
    result = df.copy()
    succeeded_cols = []
    unparsed_tokens = {}

    for col, target_type in type_conversions.items():
        if col not in result.columns:
            continue
        target = target_type.lower()
        initial_nans = int(result[col].isna().sum())

        try:
            if target in ["int", "int64", "integer"]:
                coerced = pd.to_numeric(result[col], errors="coerce")
                result[col] = coerced.astype("Int64")
            elif target in ["float", "float64"]:
                coerced = pd.to_numeric(result[col], errors="coerce")
                result[col] = coerced.astype(float)
            elif target in ["str", "string"]:
                result[col] = result[col].astype("string")
            elif target in ["datetime", "date"]:
                result[col] = pd.to_datetime(result[col], errors="coerce")
            elif target in ["bool", "boolean"]:
                result[col] = parse_boolean_series(result[col])
            elif target in ["category", "categorical"]:
                result[col] = result[col].astype("category")

            new_nans = int(result[col].isna().sum())
            unparsed = max(0, new_nans - initial_nans)
            if unparsed > 0:
                unparsed_tokens[col] = unparsed

            succeeded_cols.append(col)
        except Exception as e:
            raise ValueError(f"Failed casting {col} to {target}: {str(e)}")

    if logger:
        logger.log(
            action="Data Type Casting",
            details=f"Converted types: {type_conversions}. Coerced to NaN: {unparsed_tokens}",
            rows_affected=len(result),
            columns_affected=succeeded_cols
        )
    return result, unparsed_tokens


def analyze_cleaning_recommendations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates actionable, non-destructive cleaning recommendations for user inspection.
    """
    recs = {
        "header_standardization_needed": any(not c.isidentifier() for c in df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_columns": {},
        "outlier_candidates": {}
    }

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            recs["missing_columns"][col] = {
                "count": n_missing,
                "pct": round(n_missing / len(df) * 100, 2),
                "suggested_imputation": "median" if pd.api.types.is_numeric_dtype(df[col]) else "mode"
            }

        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            if len(s) > 10:
                q25, q75 = s.quantile(0.25), s.quantile(0.75)
                iqr = q75 - q25
                outliers = int(((s < q25 - 3.0 * iqr) | (s > q75 + 3.0 * iqr)).sum())
                if outliers > 0:
                    recs["outlier_candidates"][col] = {
                        "count": outliers,
                        "pct": round(outliers / len(s) * 100, 2)
                    }

    return recs


def run_automated_cleaning(
    df: pd.DataFrame,
    logger: Optional[AuditLogger] = None,
    impute_missing: bool = False,
    cap_outliers: bool = False,
    standardize_headers: bool = True,
    purge_duplicates: bool = True,
    clean_strings: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Explicit, controlled cleaning pipeline.
    """
    audit = logger if logger is not None else AuditLogger()
    cleaned = df.copy()

    # 1. Standardize headers with collision guards
    header_mapping = {}
    if standardize_headers:
        cleaned, header_mapping = standardize_column_headers(cleaned, case_style="snake_case", logger=audit)

    # 2. Purge exact duplicates
    if purge_duplicates:
        cleaned = remove_duplicates(cleaned, logger=audit)

    # 3. Clean text columns without stringifying NaNs
    if clean_strings:
        text_cols = [
            c for c in cleaned.columns
            if cleaned[c].dtype == "object"
            or isinstance(cleaned[c].dtype, (pd.StringDtype, pd.CategoricalDtype))
        ]
        if text_cols:
            cleaned = clean_text_columns(cleaned, columns=text_cols, strip_whitespace=True, logger=audit)

    # 4. Optional missing value imputation (opt-in)
    if impute_missing:
        for col in cleaned.columns:
            if cleaned[col].isna().sum() > 0:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned = impute_missing_values(cleaned, columns=[col], strategy="median", logger=audit)
                else:
                    cleaned = impute_missing_values(cleaned, columns=[col], strategy="mode", logger=audit)

    # 5. Optional outlier treatment (opt-in)
    outlier_summary = {}
    if cap_outliers:
        num_cols = [c for c in cleaned.columns if pd.api.types.is_numeric_dtype(cleaned[c])]
        for nc in num_cols:
            cleaned, info = treat_outliers(cleaned, column=nc, method="iqr", threshold=3.0, action="cap", logger=audit)
            outlier_summary[nc] = info["outliers_detected"]

    summary = {
        "final_rows": len(cleaned),
        "final_columns": len(cleaned.columns),
        "steps_executed": len(audit.get_logs()),
        "header_mapping": header_mapping,
        "imputation_performed": impute_missing,
        "outlier_capping_performed": cap_outliers,
        "outlier_caps": outlier_summary
    }

    return cleaned, summary
