"""
Centralized Data Validation Layer for DataSight Analytics.
Provides reusable validation routines for all analytical modules:
- Finite value checks (np.isfinite rejection of Inf, -Inf)
- Minimum row & column checks
- Zero-variance and constant column checks
- Duplicate column name detection
- Cardinality guards for pivoting, groupbys, and categorical plotting
"""

from typing import List, Optional, Tuple, Dict, Any, Union
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


def sanitize_and_report_numeric_policy(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    allow_inf: bool = False,
    coerce_invalid: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Global NaN/Inf/NaT Policy:
    Ensures uniform accounting and handling of missing, infinite, and unparseable values across modules.
    Returns: (cleaned_df, report)
    """
    target_cols = columns if columns is not None else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    report: Dict[str, Any] = {
        "columns_evaluated": target_cols,
        "dropped_rows_count": 0,
        "infinite_values_found": {},
        "nan_values_found": {},
        "action_taken": "coerced" if coerce_invalid else "validated"
    }

    result = df.copy()

    for col in target_cols:
        if col not in result.columns:
            continue
        s = result[col]
        nan_count = int(s.isna().sum())
        if nan_count > 0:
            report["nan_values_found"][col] = nan_count

        if pd.api.types.is_numeric_dtype(s):
            arr = s.to_numpy(dtype=float)
            inf_count = int(np.isinf(arr).sum())
            if inf_count > 0:
                report["infinite_values_found"][col] = inf_count
                if not allow_inf:
                    if coerce_invalid:
                        result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                    else:
                        raise ValidationError(
                            f"Column '{col}' violates policy with {inf_count} infinite value(s). "
                            f"Infinite values must be sanitized before processing."
                        )

    return result, report


def validate_merge_safety(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    how: str,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    on: Optional[Union[str, List[str]]] = None,
    max_output_rows: int = 500000
) -> Dict[str, Any]:
    """
    Resource guard: checks for potential Cartesian explosion before performing merge.
    """
    l_keys = [on] if isinstance(on, str) else (on or ([left_on] if isinstance(left_on, str) else (left_on or [])))
    r_keys = [on] if isinstance(on, str) else (on or ([right_on] if isinstance(right_on, str) else (right_on or [])))

    if not l_keys or not r_keys:
        return {"is_safe": True, "estimated_max_rows": len(left_df)}

    # Estimate join explosion potential by checking max duplicate key frequency
    max_left_key_freq = int(left_df.groupby(l_keys, observed=True).size().max()) if len(left_df) > 0 else 0
    max_right_key_freq = int(right_df.groupby(r_keys, observed=True).size().max()) if len(right_df) > 0 else 0

    estimated_max = max(len(left_df), len(right_df))
    if max_left_key_freq > 1 and max_right_key_freq > 1:
        # Many-to-many relationship
        estimated_max = min(len(left_df) * max_right_key_freq, max_output_rows + 1)

    if estimated_max > max_output_rows:
        raise ValidationError(
            f"Merge Resource Guard: Potential Cartesian explosion detected! "
            f"Estimated possible join size could reach {estimated_max:,} rows, "
            f"exceeding maximum permitted safety ceiling of {max_output_rows:,} rows."
        )

    return {
        "is_safe": True,
        "max_left_key_freq": max_left_key_freq,
        "max_right_key_freq": max_right_key_freq,
        "is_many_to_many": bool(max_left_key_freq > 1 and max_right_key_freq > 1)
    }

