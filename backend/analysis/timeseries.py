"""
Time-Series Analysis module.
Features 32-36:
32. Temporal Resampling (daily, weekly, monthly, quarterly aggregation)
33. Rolling & Moving Averages (sliding window metrics with temporal sorting)
34. Period-over-Period Growth (validated time frequency MoM/YoY calculations)
35. Cumulative Sums & Running Totals
36. Seasonality & Trend Decomposition (strict periodicity validation)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def ensure_sorted_timeseries(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Guarantees that time-series operations are strictly sorted by date/time.
    """
    sorted_df = df.copy()
    sorted_df[date_column] = pd.to_datetime(sorted_df[date_column])
    return sorted_df.sort_values(by=date_column).reset_index(drop=True)


def resample_temporal_data(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    frequency: str = "ME",
    aggregation: str = "sum"
) -> pd.DataFrame:
    """
    Feature 32: Temporal Resampling (daily, weekly, monthly, quarterly aggregation).
    """
    sub = df[[date_column, value_column]].dropna().copy()
    sub = ensure_sorted_timeseries(sub, date_column)
    sub = sub.set_index(date_column)

    resampled = sub.resample(frequency).agg(aggregation).reset_index()
    resampled.columns = ["date", f"{value_column}_{aggregation}"]
    return resampled


def compute_rolling_metrics(
    df: pd.DataFrame,
    value_column: str,
    date_column: Optional[str] = None,
    window_size: int = 7,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Feature 33: Rolling & Moving Averages (sliding window metrics).
    Strictly orders by date column if provided before computing window stats.
    """
    if metrics is None:
        metrics = ["mean", "std"]

    result = ensure_sorted_timeseries(df, date_column) if date_column else df.copy()
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

    return result


def compute_period_over_period_growth(
    df: pd.DataFrame,
    value_column: str,
    date_column: Optional[str] = None,
    periods: int = 1,
    growth_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Feature 34: Period-over-Period Growth with temporal ordering verification.
    """
    result = ensure_sorted_timeseries(df, date_column) if date_column else df.copy()
    target_name = growth_col_name if growth_col_name else f"{value_column}_pct_change_{periods}"
    abs_name = f"{value_column}_diff_{periods}"

    result[abs_name] = result[value_column].diff(periods=periods)
    result[target_name] = (result[value_column].pct_change(periods=periods) * 100).round(2)

    return result


def compute_cumulative_totals(
    df: pd.DataFrame,
    value_column: str,
    date_column: Optional[str] = None,
    new_column_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Feature 35: Cumulative Sums & Running Totals with temporal sorting.
    """
    result = ensure_sorted_timeseries(df, date_column) if date_column else df.copy()
    col_name = new_column_name if new_column_name else f"{value_column}_cumsum"
    result[col_name] = result[value_column].cumsum()
    return result


def decompose_seasonality_trend(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    model: str = "additive",
    period: int = 12
) -> Dict[str, Any]:
    """
    Feature 36: Seasonality & Trend Decomposition.
    Does NOT silently alter requested periods. Raises an informative error if
    observations are insufficient for the user's requested period (requires >= 2 * period).
    """
    sub = df[[date_column, value_column]].dropna().copy()
    sub = ensure_sorted_timeseries(sub, date_column)

    ts = sub.set_index(date_column)[value_column]
    n_points = len(ts)

    min_required = 2 * period
    if n_points < min_required:
        raise ValueError(
            f"Insufficient observations for seasonal decomposition: dataset has {n_points} valid records, "
            f"but period {period} requires at least {min_required} consecutive points (2 full seasonal cycles)."
        )

    if model == "multiplicative" and (ts <= 0).any():
        raise ValueError("Multiplicative decomposition requires strictly positive values (> 0). Use additive decomposition.")

    decomposition = seasonal_decompose(ts, model=model, period=period, extrapolate_trend="period")

    decomp_df = pd.DataFrame({
        "date": ts.index,
        "observed": decomposition.observed.values,
        "trend": decomposition.trend.values,
        "seasonal": decomposition.seasonal.values,
        "residual": decomposition.resid.values
    })

    return {
        "model": model,
        "period": period,
        "data": decomp_df
    }
