"""
Time-Series Analysis module.
Features 32-36:
32. Temporal Resampling (daily, weekly, monthly, quarterly aggregation)
33. Rolling & Moving Averages (sliding window metrics)
34. Period-over-Period Growth (MoM and YoY calculations)
35. Cumulative Sums & Running Totals
36. Seasonality & Trend Decomposition (trend, seasonal, and residual splitting)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from backend.ingestion.memory import free_memory


def resample_temporal_data(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    frequency: str = "M",  # D, W, M, Q, Y
    aggregation: str = "sum"  # sum, mean, max, min, count
) -> pd.DataFrame:
    """
    Feature 32: Temporal Resampling (daily, weekly, monthly, quarterly aggregation).
    """
    sub = df[[date_column, value_column]].dropna().copy()
    sub[date_column] = pd.to_datetime(sub[date_column])
    sub = sub.sort_values(by=date_column)
    sub = sub.set_index(date_column)

    resampled = sub.resample(frequency).agg(aggregation).reset_index()
    resampled.columns = ["date", f"{value_column}_{aggregation}"]
    free_memory()
    return resampled


def compute_rolling_metrics(
    df: pd.DataFrame,
    value_column: str,
    window_size: int = 7,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Feature 33: Rolling & Moving Averages (sliding window metrics).
    Computes rolling mean, std, min, max.
    """
    if metrics is None:
        metrics = ["mean", "std"]

    result = df.copy()
    roller = result[value_column].rolling(window=window_size)

    for m in metrics:
        col_name = f"{value_column}_rolling_{m}_{window_size}"
        if m == "mean":
            result[col_name] = roller.mean()
        elif m == "std":
            result[col_name] = roller.std()
        elif m == "min":
            result[col_name] = roller.min()
        elif m == "max":
            result[col_name] = roller.max()

    free_memory()
    return result


def compute_period_over_period_growth(
    df: pd.DataFrame,
    value_column: str,
    periods: int = 1,
    growth_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Feature 34: Period-over-Period Growth (MoM and YoY calculations).
    Computes absolute diff and percentage change.
    """
    result = df.copy()
    target_name = growth_col_name if growth_col_name else f"{value_column}_pct_change_{periods}"
    abs_name = f"{value_column}_diff_{periods}"

    result[abs_name] = result[value_column].diff(periods=periods)
    result[target_name] = (result[value_column].pct_change(periods=periods) * 100).round(2)

    free_memory()
    return result


def compute_cumulative_totals(
    df: pd.DataFrame,
    value_column: str,
    new_column_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Feature 35: Cumulative Sums & Running Totals.
    """
    result = df.copy()
    col_name = new_column_name if new_column_name else f"{value_column}_cumsum"
    result[col_name] = result[value_column].cumsum()
    free_memory()
    return result


def decompose_seasonality_trend(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    model: str = "additive",  # additive or multiplicative
    period: Optional[int] = None
) -> Dict[str, Any]:
    """
    Feature 36: Seasonality & Trend Decomposition (trend, seasonal, and residual splitting).
    """
    sub = df[[date_column, value_column]].dropna().copy()
    sub[date_column] = pd.to_datetime(sub[date_column])
    sub = sub.sort_values(by=date_column)
    
    # Check if regular frequency can be inferred or use index
    ts = sub.set_index(date_column)[value_column]

    if period is None:
        period = 12 if len(ts) >= 24 else 4
    if len(ts) < 2 * period:
        period = max(2, len(ts) // 3)

    if period < 2 or len(ts) < 2 * period:
        raise ValueError(f"Insufficient data points ({len(ts)}) for seasonal decomposition with period {period}.")

    # Multiplicative requires strictly positive data
    if model == "multiplicative" and (ts <= 0).any():
        model = "additive"

    decomposition = seasonal_decompose(ts, model=model, period=period, extrapolate_trend="freq")

    decomp_df = pd.DataFrame({
        "date": ts.index,
        "observed": decomposition.observed.values,
        "trend": decomposition.trend.values,
        "seasonal": decomposition.seasonal.values,
        "residual": decomposition.resid.values
    })

    free_memory()
    return {
        "model": model,
        "period": period,
        "data": decomp_df
    }
