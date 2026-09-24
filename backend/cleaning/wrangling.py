"""
Advanced Data Wrangling & Transformations module.
Features 20-26:
20. Secure Formula / Calculated Column Builder (AST-restricted parser)
21. Binning & Discretization (grouping continuous numerical data into discrete ranges)
22. SQL-like Groupby & Aggregations (grouping by categories and applying summary functions)
23. Merging & Joining (combining datasets using inner, left, right, or outer joins)
24. Text Regex Extraction (pulling patterns like emails, phone numbers, zip codes)
25. Data Pivoting & Unpivoting (reshaping between wide and long formats)
26. Structured Query Filtering (secure non-eval row selection)
"""

import ast
import re
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from backend.cleaning.cleaning import AuditLogger


# Restricted mathematical functions allowed in secure AST evaluation
ALLOWED_FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "exp": np.exp,
    "round": np.round,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan
}


class SecureExpressionEvaluator(ast.NodeVisitor):
    """
    AST-based expression validator and evaluator.
    Strictly forbids dangerous Python operations (__import__, open, eval, exec, lambda, etc.)
    and only permits safe arithmetic, unary operations, math calls, and dataset column lookups.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def evaluate(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body)

        elif isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):
            col_name = node.id
            if col_name in self.df.columns:
                return self.df[col_name]
            elif col_name in ALLOWED_FUNCTIONS:
                return ALLOWED_FUNCTIONS[col_name]
            else:
                raise ValueError(f"Unknown variable or column '{col_name}' in expression.")

        elif isinstance(node, ast.UnaryOp):
            operand = self.evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        elif isinstance(node, ast.BinOp):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right
            elif isinstance(node.op, ast.FloorDiv):
                return left // right
            elif isinstance(node.op, ast.Mod):
                return left % right
            elif isinstance(node.op, ast.Pow):
                return left ** right
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Dangerous dynamic function call blocked.")
            func_name = node.func.id
            if func_name not in ALLOWED_FUNCTIONS:
                raise ValueError(f"Function '{func_name}' is not in the allowed math function library.")

            args = [self.evaluate(arg) for arg in node.args]
            return ALLOWED_FUNCTIONS[func_name](*args)

        else:
            raise ValueError(f"Security Violation: AST node type '{type(node).__name__}' is strictly prohibited.")


def add_custom_formula_column(
    df: pd.DataFrame,
    new_column_name: str,
    expression: str,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 20: Secure AST-based Formula Builder.
    Evaluates arithmetic and math expressions safely without arbitrary code execution risk.
    Supports column aliases with spaces and backticks.
    """
    result = df.copy()

    # Pre-process expression to map column names with spaces into valid AST identifiers
    clean_expr = expression
    alias_map = {}
    reverse_map = {}

    sorted_cols = sorted(list(result.columns), key=len, reverse=True)
    temp_df = result.copy()

    for idx, col in enumerate(sorted_cols):
        alias = f"col_alias_{idx}"
        alias_map[col] = alias
        reverse_map[alias] = col
        temp_df[alias] = temp_df[col]

        pattern_backtick = rf"`{re.escape(col)}`"
        clean_expr = re.sub(pattern_backtick, alias, clean_expr)

        pattern_bracket = rf"df\[['\"]{re.escape(col)}['\"]\]"
        clean_expr = re.sub(pattern_bracket, alias, clean_expr)

        pattern_word = rf"(?<![\w'\"]){re.escape(col)}(?![\w'\"])"
        clean_expr = re.sub(pattern_word, alias, clean_expr)

    try:
        parsed_ast = ast.parse(clean_expr, mode="eval")
        evaluator = SecureExpressionEvaluator(temp_df)
        computed = evaluator.evaluate(parsed_ast)
        result[new_column_name] = computed
    except Exception as e:
        raise ValueError(f"Formula evaluation rejected: {str(e)}")

    if logger:
        logger.log(
            action="Custom Formula Column (Secure AST)",
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
    """
    for col in group_columns:
        if col not in df.columns:
            raise ValueError(f"Group column '{col}' not found.")

    grouped = df.groupby(group_columns, observed=False).agg(aggregations)
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
        aggfunc=aggfunc,
        observed=False
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


SUPPORTED_QUERY_OPERATORS = ["==", "!=", ">", ">=", "<", "<=", "contains", "in"]


def filter_rows_structured(
    df: pd.DataFrame,
    column: str,
    operator: str,
    comparison_value: Any,
    logger: Optional[AuditLogger] = None
) -> pd.DataFrame:
    """
    Feature 26: Secure, structured row filtering.
    Replaces unrestricted eval/query strings with typed parametric comparisons.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")

    initial_rows = len(df)
    s = df[column]

    if operator == "==":
        mask = s == comparison_value
    elif operator == "!=":
        mask = s != comparison_value
    elif operator == ">":
        mask = s > float(comparison_value)
    elif operator == ">=":
        mask = s >= float(comparison_value)
    elif operator == "<":
        mask = s < float(comparison_value)
    elif operator == "<=":
        mask = s <= float(comparison_value)
    elif operator == "contains":
        mask = s.astype(str).str.contains(str(comparison_value), case=False, na=False)
    elif operator == "in":
        val_list = [v.strip() for v in str(comparison_value).split(",")]
        mask = s.astype(str).isin(val_list)
    else:
        raise ValueError(f"Unsupported query operator '{operator}'. Allowed: {SUPPORTED_QUERY_OPERATORS}")

    filtered_df = df[mask].reset_index(drop=True)
    rows_retained = len(filtered_df)

    if logger:
        logger.log(
            action="Filter Rows (Structured)",
            details=f"{column} {operator} {comparison_value} (retained {rows_retained} of {initial_rows})",
            rows_affected=initial_rows - rows_retained,
            columns_affected=[column]
        )
    return filtered_df
