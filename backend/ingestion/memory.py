"""
Memory management and optimization layer for data processing pipelines.
Provides precision-safe numeric downcasting with nullable integer support,
RAM tracking, and selective garbage collection.
"""

import gc
import psutil
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute total deep memory usage of a pandas DataFrame.
    Returns metrics in bytes, megabytes, and human-readable string.
    """
    if df is None or df.empty:
        return {"bytes": 0, "mb": 0.0, "readable": "0.00 MB"}

    total_bytes = int(df.memory_usage(deep=True).sum())
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024

    if total_gb >= 1.0:
        readable = f"{total_gb:.2f} GB"
    else:
        readable = f"{total_mb:.2f} MB"

    return {
        "bytes": total_bytes,
        "mb": round(total_mb, 2),
        "readable": readable
    }


def get_system_memory() -> Dict[str, Any]:
    """
    Get current system virtual memory statistics.
    """
    try:
        mem = psutil.virtual_memory()
        return {
            "total_mb": round(mem.total / (1024 * 1024), 2),
            "available_mb": round(mem.available / (1024 * 1024), 2),
            "used_mb": round(mem.used / (1024 * 1024), 2),
            "percent": mem.percent
        }
    except Exception:
        return {
            "total_mb": 0.0,
            "available_mb": 0.0,
            "used_mb": 0.0,
            "percent": 0.0
        }


def optimize_dataframe_memory(
    df: pd.DataFrame,
    downcast_integers: bool = True,
    downcast_floats: bool = False,
    convert_categories: bool = False,
    category_threshold: float = 0.20
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Precision-safe memory optimization layer.

    Nullable Integer Safety:
    Uses Pandas nullable integer dtypes ("Int8", "Int16", "Int32", "Int64") when missing
    values are present, preventing crashes caused by NumPy integer conversions.
    """
    if df is None or df.empty:
        return df, {"initial_mb": 0.0, "final_mb": 0.0, "reduction_percent": 0.0}

    initial_usage = get_memory_usage(df)
    initial_mb = initial_usage["mb"]
    n_rows = len(df)

    optimized_df = df.copy()

    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        has_nulls = bool(optimized_df[col].isna().any())

        # Safe Integer downcasting (supporting nullable Int dtypes)
        if downcast_integers and (pd.api.types.is_integer_dtype(col_type) or (has_nulls and pd.api.types.is_float_dtype(col_type))):
            s = optimized_df[col]
            # Check if float series actually represents whole numbers with NaNs
            is_pseudo_int = False
            if has_nulls and pd.api.types.is_float_dtype(col_type):
                non_nulls = s.dropna()
                if len(non_nulls) > 0 and (non_nulls % 1 == 0).all():
                    is_pseudo_int = True

            if pd.api.types.is_integer_dtype(col_type) or is_pseudo_int:
                c_min = s.min()
                c_max = s.max()
                if pd.notna(c_min) and pd.notna(c_max):
                    if has_nulls or is_pseudo_int:
                        # Use Pandas Nullable Integer types
                        if c_min >= -128 and c_max <= 127:
                            optimized_df[col] = s.astype("Int8")
                        elif c_min >= -32768 and c_max <= 32767:
                            optimized_df[col] = s.astype("Int16")
                        elif c_min >= -2147483648 and c_max <= 2147483647:
                            optimized_df[col] = s.astype("Int32")
                        else:
                            optimized_df[col] = s.astype("Int64")
                    else:
                        # Standard NumPy non-nullable types
                        if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                            optimized_df[col] = s.astype(np.int8)
                        elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                            optimized_df[col] = s.astype(np.int16)
                        elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                            optimized_df[col] = s.astype(np.int32)
                        else:
                            optimized_df[col] = s.astype(np.int64)

        # Opt-in Float downcasting (disabled by default)
        elif downcast_floats and pd.api.types.is_float_dtype(col_type):
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            if pd.notna(c_min) and pd.notna(c_max):
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)

        # Opt-in Categorical conversion (disabled by default)
        if convert_categories and (col_type == "object" or col_type == "string"):
            num_unique = optimized_df[col].nunique(dropna=False)
            if n_rows > 0 and (num_unique / n_rows) <= category_threshold:
                try:
                    optimized_df[col] = optimized_df[col].astype("category")
                except Exception:
                    pass

    final_usage = get_memory_usage(optimized_df)
    final_mb = final_usage["mb"]
    saved_mb = max(0.0, initial_mb - final_mb)
    reduction = round((saved_mb / initial_mb * 100) if initial_mb > 0 else 0.0, 2)

    metrics = {
        "initial_mb": initial_mb,
        "final_mb": final_mb,
        "saved_mb": round(saved_mb, 2),
        "reduction_percent": reduction,
        "float_downcasting_applied": downcast_floats,
        "categorical_conversion_applied": convert_categories
    }

    return optimized_df, metrics


def free_memory(force: bool = False):
    """Explicitly trigger garbage collection only when requested."""
    if force:
        gc.collect()
