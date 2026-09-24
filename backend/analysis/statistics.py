"""
Exploratory Data Analysis & Statistics module.
Features 14-19:
14. Univariate Summary Statistics (mean, median, standard deviation, variance, min, max, percentiles)
15. Categorical Frequency Distribution (value counts, percentages, unique value tallies)
16. Missingness Matrix (visual heatmaps of missing data patterns)
17. Correlation Matrix (Pearson, Spearman, and Kendall coefficients)
18. Multi-Collinearity Detection (automated warnings for correlated predictor variables / VIF)
19. Skewness & Kurtosis Analysis (symmetry and tail heaviness measurements)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats


def compute_univariate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 14: Univariate Summary Statistics.
    Computes count, missing, mean, std, var, median, min, max, 25%, 75%, IQR for all numeric columns.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()

    records = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) == 0:
            continue

        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        iqr = q75 - q25

        records.append({
            "Column": col,
            "Count": int(s.count()),
            "Missing": int(num_df[col].isna().sum()),
            "Missing_Pct": round(float(num_df[col].isna().mean() * 100), 2),
            "Mean": round(float(s.mean()), 4),
            "Median": round(float(s.median()), 4),
            "Std_Dev": round(float(s.std()), 4),
            "Variance": round(float(s.var()), 4),
            "Min": round(float(s.min()), 4),
            "Q25": round(q25, 4),
            "Q75": round(q75, 4),
            "IQR": round(iqr, 4),
            "Max": round(float(s.max()), 4),
        })

    return pd.DataFrame(records)


def compute_categorical_distribution(df: pd.DataFrame, max_categories: int = 20) -> Dict[str, pd.DataFrame]:
    """
    Feature 15: Categorical Frequency Distribution.
    Value counts, percentages, unique value tallies.
    """
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    distributions = {}

    for col in cat_cols:
        val_counts = df[col].value_counts(dropna=False).head(max_categories)
        total = len(df[col])
        pcts = (val_counts / total * 100).round(2)

        dist_df = pd.DataFrame({
            "Category": val_counts.index.astype(str),
            "Count": val_counts.values,
            "Percentage": pcts.values
        })
        distributions[col] = dist_df

    return distributions


def compute_missingness_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Feature 16: Missingness Matrix data structures for heatmaps and summary bars.
    """
    missing_counts = df.isna().sum()
    missing_pcts = (missing_counts / len(df) * 100).round(2)

    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": missing_counts.values,
        "Missing_Percentage": missing_pcts.values
    }).sort_values(by="Missing_Count", ascending=False)

    # Boolean matrix sampled if rows > 1000 for visualization performance
    if len(df) > 1000:
        sample_df = df.sample(n=1000, random_state=42)
    else:
        sample_df = df

    matrix_bool = sample_df.isna().astype(int)

    return {
        "summary": summary_df,
        "sample_matrix": matrix_bool,
        "total_missing_cells": int(missing_counts.sum()),
        "columns_with_missing": list(missing_counts[missing_counts > 0].index)
    }


def compute_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Feature 17: Correlation Matrix (Pearson, Spearman, and Kendall coefficients).
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return pd.DataFrame()

    corr = num_df.corr(method=method)  # type: ignore
    return corr.round(4)


def detect_multicollinearity(df: pd.DataFrame, threshold: float = 0.80) -> List[Dict[str, Any]]:
    """
    Feature 18: Multi-Collinearity Detection.
    Automated warnings for correlated predictor variables.
    """
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.shape[1] < 2:
        return []

    corr_matrix = num_df.corr(method="pearson").abs()
    high_corr_pairs = []

    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            coeff = corr_matrix.loc[col1, col2]
            if coeff >= threshold:
                high_corr_pairs.append({
                    "column_1": col1,
                    "column_2": col2,
                    "correlation": round(float(coeff), 4),
                    "severity": "Critical" if coeff >= 0.90 else "High"
                })

    return sorted(high_corr_pairs, key=lambda x: x["correlation"], reverse=True)


def compute_skewness_kurtosis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 19: Skewness & Kurtosis Analysis.
    Symmetry and tail heaviness measurements.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()

    records = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) < 3:
            continue

        skew_val = float(stats.skew(s, bias=False))
        kurt_val = float(stats.kurtosis(s, bias=False))  # Fisher kurtosis (normal == 0)

        # Interpretation
        if abs(skew_val) < 0.5:
            skew_desc = "Fairly Symmetrical"
        elif skew_val > 0.5:
            skew_desc = "Positively Skewed (Right-tailed)"
        else:
            skew_desc = "Negatively Skewed (Left-tailed)"

        if abs(kurt_val) < 0.5:
            kurt_desc = "Mesokurtic (Normal-like tails)"
        elif kurt_val > 0.5:
            kurt_desc = "Leptokurtic (Heavy tails, prone to outliers)"
        else:
            kurt_desc = "Platykurtic (Light tails, fewer outliers)"

        records.append({
            "Column": col,
            "Skewness": round(skew_val, 4),
            "Skew_Classification": skew_desc,
            "Kurtosis": round(kurt_val, 4),
            "Kurt_Classification": kurt_desc
        })

    return pd.DataFrame(records)
