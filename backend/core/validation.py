"""
Centralized Data Validation Layer for DataSight Analytics.
Provides reusable validation routines for all analytical modules:
- Finite value checks (np.isfinite rejection of Inf, -Inf)
- Minimum row & column checks
- Zero-variance and constant column checks
- Duplicate column name detection
- Cardinality guards for pivoting, groupbys, and categorical plotting
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when dataset fails mathematical, structural, or resource validation."""
    pass


def validate_dataframe_not_empty(df: pd.DataFrame, min_rows: int = 1) -> None:
    """Ensure dataframe is non-null and meets minimum row requirements."""
    if df is None:
        raise ValidationError("DataFrame is None.")
    if df.empty or len(df) < min_rows:
        raise ValidationError(f"DataFrame must contain at least {min_rows} row(s); got {len(df) if df is not None else 0}.")


def validate_unique_columns(df: pd.DataFrame) -> None:
    """Ensure column names are strictly unique to prevent silent alignment errors."""
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        duplicates = [c for c in cols if cols.count(c) > 1]
        raise ValidationError(f"DataFrame contains duplicate column names: {list(set(duplicates))}")


def validate_numeric_finite_column(
    df: pd.DataFrame,
    column: str,
    allow_nan: bool = True,
    min_valid: int = 2
) -> np.ndarray:
    """
    Validates that a column exists, is numeric, and contains finite values.
    Rejects Inf and -Inf. Returns non-null finite numpy array.
    """
    if column not in df.columns:
        raise ValidationError(f"Column '{column}' does not exist in DataFrame.")

    s = df[column]
    if not pd.api.types.is_numeric_dtype(s):
        raise ValidationError(f"Column '{column}' is not numeric (dtype: {s.dtype}).")

    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

    # Check for infinite values
    inf_count = int(np.isinf(arr).sum())
    if inf_count > 0:
        raise ValidationError(f"Column '{column}' contains {inf_count} infinite value(s) (Inf / -Inf).")

    clean = arr[~np.isnan(arr)]
    if len(clean) < min_valid:
        raise ValidationError(f"Column '{column}' contains fewer than {min_valid} valid finite observations.")

    return clean


def validate_non_zero_variance(arr: np.ndarray, col_name: str = "feature") -> None:
    """Validates that a numeric array has non-zero variance."""
    if len(arr) < 2:
        raise ValidationError(f"Cannot compute variance for '{col_name}' with fewer than 2 elements.")
    var_val = float(np.var(arr, ddof=1))
    if var_val == 0.0 or np.isnan(var_val):
        raise ValidationError(f"Feature '{col_name}' has zero variance (constant value {arr[0]}).")


def validate_cardinality_guard(df: pd.DataFrame, column: str, max_cardinality: int = 100) -> int:
    """
    Guards against memory exhaustion during pivoting, groupbys, or plotting
    by capping the number of distinct categories.
    """
    if column not in df.columns:
        raise ValidationError(f"Column '{column}' does not exist.")
    n_unique = int(df[column].nunique(dropna=True))
    if n_unique > max_cardinality:
        raise ValidationError(
            f"Cardinality Violation: Column '{column}' has {n_unique} unique categories, "
            f"exceeding the safety limit of {max_cardinality}. Please group or filter categories first."
        )
    return n_unique
