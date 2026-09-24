"""
Memory management and optimization layer for data processing pipelines.
Provides precision-safe numeric downcasting, RAM tracking, and selective garbage collection.
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
    Memory optimization layer with strict precision preservation.

    Rules:
    1. Integer downcasting is safe and enabled by default (int64 -> int32/int16/int8 based on min/max bounds).
    2. Float downcasting (float64 -> float32) is DISABLED by default to prevent silent loss of
       precision in scientific or financial metrics. Must be explicitly requested.
    3. Low-cardinality conversion to categorical is DISABLED by default so that storage
       optimization does not conflate with analytical semantic type detection.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to optimize
    downcast_integers : bool
        Whether to downcast integer columns safely
    downcast_floats : bool
        Whether to downcast float columns (opt-in only)
    convert_categories : bool
        Whether to convert object columns to category dtype (opt-in only)
    category_threshold : float
        Ratio of unique values to total rows below which a column can be categorical

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Optimized dataframe and performance impact metrics
    """
    if df is None or df.empty:
        return df, {"initial_mb": 0.0, "final_mb": 0.0, "reduction_percent": 0.0}

    initial_usage = get_memory_usage(df)
    initial_mb = initial_usage["mb"]
    n_rows = len(df)

    optimized_df = df.copy()

    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype

        # Safe Integer downcasting
        if downcast_integers and pd.api.types.is_integer_dtype(col_type):
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            if pd.notna(c_min) and pd.notna(c_max):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
                else:
                    optimized_df[col] = optimized_df[col].astype(np.int64)

        # Opt-in Float downcasting (never default)
        if downcast_floats and pd.api.types.is_float_dtype(col_type):
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            if pd.notna(c_min) and pd.notna(c_max):
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)

        # Opt-in Categorical conversion
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
    """
    Explicitly trigger garbage collection only when requested.
    """
    if force:
        gc.collect()
