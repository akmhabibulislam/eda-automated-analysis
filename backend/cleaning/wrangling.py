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

from backend.cleaning.cleaning import AuditLogger


def add_custom_formula_column(
    df: pd.DataFrame,
    new_column_name: str,
    expression: str,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 20: Custom Formula / Equation Builder.
    Robustly evaluates expressions even when column names contain spaces, dashes, or special characters.
    Supports standard syntax (e.g. `Revenue - Cost` or `df['Revenue'] - df['Cost']` or `np.log(Price)`).
    """
    result = df.copy()

    # Create safe mapping replacing special characters in column names for evaluation
    col_mapping = {}
    reverse_mapping = {}
    clean_expr = expression

    # Sort columns by length descending so longer column names get replaced first
    sorted_cols = sorted(list(result.columns), key=len, reverse=True)

    local_dict = {
        "np": np,
        "pd": pd,
        "df": result
    }

    # Map column references
    for idx, col in enumerate(sorted_cols):
        safe_alias = f"__col_{idx}__"
        col_mapping[col] = safe_alias
        reverse_mapping[safe_alias] = col
        local_dict[safe_alias] = result[col]

        # Support `column name` backtick notation or direct names with spaces
        pattern_backtick = rf"`{re.escape(col)}`"
        clean_expr = re.sub(pattern_backtick, safe_alias, clean_expr)

        pattern_bracket = rf"df\[['\"]{re.escape(col)}['\"]\]"
        clean_expr = re.sub(pattern_bracket, safe_alias, clean_expr)

        # Match standalone column name with word boundaries
        pattern_word = rf"(?<![\w'\"]){re.escape(col)}(?![\w'\"])"
        clean_expr = re.sub(pattern_word, safe_alias, clean_expr)

    # First attempt: pandas eval
    try:
        result[new_column_name] = pd.eval(clean_expr, local_dict=local_dict, engine="python")
    except Exception:
        # Second attempt: Python eval with isolated builtins
        try:
            safe_builtins = {
                "abs": abs, "min": min, "max": max, "round": round,
                "float": float, "int": int, "str": str, "bool": bool
            }
            res_series = eval(clean_expr, {"__builtins__": safe_builtins}, local_dict)
            if isinstance(res_series, (pd.Series, np.ndarray, list, int, float)):
                result[new_column_name] = res_series
            else:
                result[new_column_name] = pd.Series([res_series] * len(result))
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {str(e)}")

    if logger:
        logger.log(
            action="Custom Formula Column",
            details=f"Created '{new_column_name}' with expression: {expression}",
            rows_affected=len(result),
            columns_affected=[new_column_name]
        )
    return result


def bin_continuous_column(
    df: pd.DataFrame,
    column: str,
    new_column_name: Optional[str] = None,
    bins: Union[int, List[float]] = 5,
    labels: Optional[List[str]] = None,
    bin_type: str = "equal_width",
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
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]

    return grouped.reset_index()


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

    if isinstance(pivoted.columns, pd.MultiIndex):
        pivoted.columns = [f"{col}_{lvl}".strip("_") for col, lvl in pivoted.columns]

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
    return pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name
    )


def filter_rows(
    df: pd.DataFrame,
    query_expression: str,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 26: Custom Filtering & Querying.
    Row filtering using custom query expression syntax.
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
    return filtered_df
