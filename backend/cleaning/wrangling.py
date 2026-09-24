"""
Advanced Data Wrangling & Transformations module.
Features 20-26:
20. Custom Formula / Equation Builder (creating calculated columns via mathematical expressions like `Revenue - Cost`)
21. Binning & Discretization (grouping continuous numerical data into discrete ranges)
22. SQL-like Groupby & Aggregations (grouping by categories and applying summary functions)
23. Merging & Joining (combining datasets using inner, left, right, or outer joins)
24. Text Regex Extraction (pulling patterns like emails, phone numbers, zip codes)
25. Data Pivoting & Unpivoting (reshaping between wide and long formats)
26. Custom Filtering & Querying (row filtering using custom expression builders)
"""

import re
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from backend.ingestion.memory import free_memory
from backend.cleaning.cleaning import AuditLogger


def add_custom_formula_column(
    df: pd.DataFrame,
    new_column_name: str,
    expression: str,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 20: Custom Formula / Equation Builder.
    Evaluates vector formulas such as `df['revenue'] - df['cost']` or expressions like `revenue * 1.15`.
    """
    result = df.copy()
    
    # Safe evaluation environment using pandas eval
    try:
        result[new_column_name] = result.eval(expression)
    except Exception:
        # Fallback to local dict eval if columns have spaces or special characters
        local_env = {col: result[col] for col in result.columns}
        local_env["np"] = np
        try:
            result[new_column_name] = eval(expression, {"__builtins__": {}}, local_env)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {str(e)}")

    if logger:
        logger.log(
            action="Custom Formula Column",
            details=f"Created '{new_column_name}' with expression: {expression}",
            rows_affected=len(result),
            columns_affected=[new_column_name]
        )
    free_memory()
    return result


def bin_continuous_column(
    df: pd.DataFrame,
    column: str,
    new_column_name: Optional[str] = None,
    bins: Union[int, List[float]] = 5,
    labels: Optional[List[str]] = None,
    bin_type: str = "equal_width",  # equal_width or quantile
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 21: Binning & Discretization.
    Groups continuous numerical data into discrete ranges using equal-width or quantile binning.
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not a valid numeric column.")

    target_name = new_column_name if new_column_name else f"{column}_binned"
    result = df.copy()

    if bin_type == "equal_width":
        result[target_name] = pd.cut(result[column], bins=bins, labels=labels, include_lowest=True)
    elif bin_type == "quantile":
        num_quantiles = bins if isinstance(bins, int) else 4
        result[target_name] = pd.qcut(result[column], q=num_quantiles, labels=labels, duplicates="drop")
    else:
        raise ValueError(f"Unknown bin_type '{bin_type}'. Must be 'equal_width' or 'quantile'.")

    result[target_name] = result[target_name].astype(str)

    if logger:
        logger.log(
            action="Binning / Discretization",
            details=f"Binned '{column}' into '{target_name}' using {bin_type} ({bins} bins)",
            rows_affected=len(result),
            columns_affected=[target_name]
        )
    return result


def aggregate_groupby(
    df: pd.DataFrame,
    group_columns: List[str],
    aggregations: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Feature 22: SQL-like Groupby & Aggregations.
    Groups by one or more categories and applies aggregate functions (mean, sum, count, std, min, max, median).
    """
    for col in group_columns:
        if col not in df.columns:
            raise ValueError(f"Group column '{col}' not found.")

    grouped = df.groupby(group_columns).agg(aggregations)
    # Flatten hierarchical columns if multiple aggregations were selected
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]
    
    result = grouped.reset_index()
    free_memory()
    return result


def merge_datasets(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    how: str = "inner",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    on: Optional[Union[str, List[str]]] = None,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 23: Merging & Joining.
    Combines datasets using inner, left, right, or outer joins.
    """
    result = pd.merge(
        left=left_df,
        right=right_df,
        how=how,  # type: ignore
        left_on=left_on,
        right_on=right_on,
        on=on,
        suffixes=("_left", "_right")
    )

    if logger:
        logger.log(
            action=f"Merge Datasets ({how.upper()})",
            details=f"Merged datasets on keys: on={on}, left_on={left_on}, right_on={right_on}",
            rows_affected=len(result),
            columns_affected=list(result.columns)
        )
    free_memory()
    return result


def extract_regex_patterns(
    df: pd.DataFrame,
    source_column: str,
    pattern: str,
    new_column_name: str,
    extract_all: bool = False,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 24: Text Regex Extraction.
    Extracts regex matches (e.g. emails, phone numbers, zip codes) into a new column.
    """
    if source_column not in df.columns:
        raise ValueError(f"Column '{source_column}' does not exist.")

    result = df.copy()
    str_series = result[source_column].astype(str)

    if extract_all:
        result[new_column_name] = str_series.apply(
            lambda x: ", ".join(re.findall(pattern, x)) if pd.notna(x) else ""
        )
    else:
        # If pattern has no capture group, enclose in parentheses
        compiled_pattern = pattern if "(" in pattern else f"({pattern})"
        extracted = str_series.str.extract(compiled_pattern, expand=False)
        result[new_column_name] = extracted

    if logger:
        logger.log(
            action="Text Regex Extraction",
            details=f"Extracted pattern '{pattern}' from '{source_column}' into '{new_column_name}'",
            rows_affected=int((result[new_column_name] != "").sum()),
            columns_affected=[new_column_name]
        )
    free_memory()
    return result


def pivot_dataframe(
    df: pd.DataFrame,
    index_cols: List[str],
    columns: str,
    values: str,
    aggfunc: str = "mean"
) -> pd.DataFrame:
    """
    Feature 25a: Reshaping - Long to Wide (Pivot).
    """
    pivoted = df.pivot_table(
        index=index_cols,
        columns=columns,
        values=values,
        aggfunc=aggfunc
    ).reset_index()

    # Flatten column names if index is a multi-index
    if isinstance(pivoted.columns, pd.MultiIndex):
        pivoted.columns = [f"{col}_{lvl}".strip("_") for col, lvl in pivoted.columns]

    free_memory()
    return pivoted


def unpivot_dataframe(
    df: pd.DataFrame,
    id_vars: List[str],
    value_vars: List[str],
    var_name: str = "variable",
    value_name: str = "value"
) -> pd.DataFrame:
    """
    Feature 25b: Reshaping - Wide to Long (Unpivot / Melt).
    """
    melted = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name
    )
    free_memory()
    return melted


def filter_rows(
    df: pd.DataFrame,
    query_expression: str,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 26: Custom Filtering & Querying.
    Row filtering using custom query expression syntax (e.g. `age > 30 and status == 'active'`).
    """
    initial_rows = len(df)
    try:
        filtered_df = df.query(query_expression).reset_index(drop=True)
    except Exception as e:
        raise ValueError(f"Invalid query expression '{query_expression}': {str(e)}")

    rows_retained = len(filtered_df)
    if logger:
        logger.log(
            action="Filter Rows",
            details=f"Applied query: '{query_expression}' (retained {rows_retained} of {initial_rows} rows)",
            rows_affected=initial_rows - rows_retained,
            columns_affected=[]
        )
    free_memory()
    return filtered_df
