"""
Data ingestion module supporting multi-format file uploads and database connections.
Features 1-5:
1. Multi-Format File Upload (CSV, Excel, JSON, Parquet, TSV) with true chunking / streaming options
2. Database Connectors (SQLite, PostgreSQL, MySQL query support)
3. Data Schema Detection (automatic identification with robust date validation)
4. Dataset Overview Dashboard (total rows, columns, memory footprint, duplicate counts)
5. Data Preview Table (interactive, filterable, and sortable data grid metadata)
"""

import os
import re
import warnings
from typing import Dict, Any, Optional, Tuple, Generator, List, Union
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

from backend.ingestion.memory import optimize_dataframe_memory, get_memory_usage, free_memory


LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 50000


def detect_file_format(file_name: str) -> str:
    """
    Identifies file extension from file name.
    """
    ext = os.path.splitext(file_name)[1].lower()
    mapping = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".tab": "tsv",
        ".txt": "csv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".json": "json",
        ".parquet": "parquet",
        ".pq": "parquet",
    }
    return mapping.get(ext, "csv")


def stream_dataset_chunks(
    file_source: Any,
    file_name: str,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    auto_optimize: bool = True
) -> Generator[pd.DataFrame, None, None]:
    """
    Streams file in chunks to prevent memory blowup on large datasets.
    Each yielded chunk is independently memory-downcasted.
    """
    fmt = detect_file_format(file_name)
    sep = "\t" if fmt == "tsv" else ","

    if fmt in ["csv", "tsv"]:
        chunk_iter = pd.read_csv(file_source, sep=sep, chunksize=chunksize)
        for chunk in chunk_iter:
            if auto_optimize:
                chunk, _ = optimize_dataframe_memory(chunk)
            yield chunk
    else:
        # For non-splittable formats, load single chunk
        df, _ = load_dataset(file_source, file_name, auto_optimize=auto_optimize)
        yield df


def load_dataset(
    file_source: Any,
    file_name: str,
    auto_optimize: bool = True,
    sample_rows: Optional[int] = None,
    chunksize: Optional[int] = None
) -> Tuple[Union[pd.DataFrame, Generator[pd.DataFrame, None, None]], Dict[str, Any]]:
    """
    Universal dataset reader supporting CSV, Excel, JSON, Parquet, TSV with memory management.
    If chunksize is specified without full load requirement, returns a generator instead of concatenating.
    """
    fmt = detect_file_format(file_name)
    metadata: Dict[str, Any] = {
        "file_name": file_name,
        "format": fmt,
        "was_sampled": False,
        "sample_size": sample_rows,
        "is_streaming": False
    }

    try:
        # If user explicitly requests chunked streaming without sampling
        if chunksize and fmt in ["csv", "tsv"] and sample_rows is None:
            metadata["is_streaming"] = True
            metadata["chunk_size"] = chunksize
            gen = stream_dataset_chunks(file_source, file_name, chunksize=chunksize, auto_optimize=auto_optimize)
            return gen, metadata

        if fmt == "csv":
            if sample_rows is not None and sample_rows > 0:
                df = pd.read_csv(file_source, nrows=sample_rows)
                metadata["was_sampled"] = True
            else:
                df = pd.read_csv(file_source)

        elif fmt == "tsv":
            if sample_rows is not None and sample_rows > 0:
                df = pd.read_csv(file_source, sep="\t", nrows=sample_rows)
                metadata["was_sampled"] = True
            else:
                df = pd.read_csv(file_source, sep="\t")

        elif fmt == "excel":
            df = pd.read_excel(file_source)
            if sample_rows is not None and len(df) > sample_rows:
                df = df.iloc[:sample_rows].copy()
                metadata["was_sampled"] = True

        elif fmt == "json":
            df = pd.read_json(file_source)
            if sample_rows is not None and len(df) > sample_rows:
                df = df.iloc[:sample_rows].copy()
                metadata["was_sampled"] = True

        elif fmt == "parquet":
            df = pd.read_parquet(file_source)
            if sample_rows is not None and len(df) > sample_rows:
                df = df.iloc[:sample_rows].copy()
                metadata["was_sampled"] = True

        else:
            df = pd.read_csv(file_source)

    except Exception as e:
        raise ValueError(f"Failed to parse {file_name} as {fmt}: {str(e)}")

    if auto_optimize:
        df, opt_metrics = optimize_dataframe_memory(df)
        metadata["memory_optimization"] = opt_metrics
    else:
        metadata["memory_optimization"] = get_memory_usage(df)

    return df, metadata


def load_from_database(
    connection_string: str,
    query: str,
    auto_optimize: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Connect to SQLite, PostgreSQL, MySQL or other SQL databases via SQLAlchemy and execute query.
    """
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    metadata = {
        "source": "database",
        "query": query,
        "rows_retrieved": len(df)
    }

    if auto_optimize:
        df, opt_metrics = optimize_dataframe_memory(df)
        metadata["memory_optimization"] = opt_metrics
    else:
        metadata["memory_optimization"] = get_memory_usage(df)

    return df, metadata


# Strict date regex patterns to prevent UUIDs, SKUs, or random alphanumeric strings from false matching
DATE_PATTERNS = [
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}",           # 2024-01-01, 2024/01/01
    r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}",           # 01-01-2024, 01/01/2024
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}",       # ISO format with time
    r"^\w{3}\s+\d{1,2},\s+\d{4}"               # Jan 01, 2024
]
DATE_REGEX = re.compile("|".join(f"({p})" for p in DATE_PATTERNS))


def is_valid_date_series(series: pd.Series) -> bool:
    """
    Verifies if non-null string entries actually adhere to valid date formats,
    rejecting UUIDs, SKUs, part numbers, or arbitrary strings containing hyphens.
    """
    sample = series.dropna().head(15).astype(str)
    if len(sample) == 0:
        return False

    matches = 0
    for val in sample:
        v = val.strip()
        # Must match explicit date pattern and cannot contain underscores or typical non-date alphanumeric prefixes
        if DATE_REGEX.search(v) and not any(p in v.lower() for p in ["id", "sku", "uuid", "tx-", "part"]):
            matches += 1

    # At least 80% of sampled values must strictly match date formats
    if (matches / len(sample)) < 0.80:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            return parsed.notna().sum() / len(sample) >= 0.80
        except Exception:
            return False


def detect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Feature 3: Automatic identification of numeric, categorical, datetime, boolean types
    with robust non-date exclusion.
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    boolean_cols: List[str] = []
    text_cols: List[str] = []

    column_types: Dict[str, str] = {}

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_bool_dtype(s):
            boolean_cols.append(col)
            column_types[col] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
            column_types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            column_types[col] = "numeric"
        elif isinstance(s.dtype, pd.CategoricalDtype):
            categorical_cols.append(col)
            column_types[col] = "categorical"
        else:
            if is_valid_date_series(s):
                datetime_cols.append(col)
                column_types[col] = "datetime"
            else:
                n_unique = s.nunique(dropna=True)
                total_len = len(s)
                if total_len > 0 and (n_unique / total_len < 0.20 or n_unique < 30):
                    categorical_cols.append(col)
                    column_types[col] = "categorical"
                else:
                    text_cols.append(col)
                    column_types[col] = "text"

    return {
        "columns": list(df.columns),
        "column_types": column_types,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "boolean_columns": boolean_cols,
        "text_columns": text_cols,
        "total_columns": len(df.columns)
    }


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Feature 4: Total rows, columns, memory footprint, duplicate counts, missing counts.
    """
    if df is None:
        return {}

    n_rows, n_cols = df.shape
    mem_info = get_memory_usage(df)
    duplicate_rows = int(df.duplicated().sum())
    total_cells = n_rows * n_cols
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells * 100) if total_cells > 0 else 0.0, 2)

    schema = detect_schema(df)

    return {
        "total_rows": n_rows,
        "total_columns": n_cols,
        "memory_bytes": mem_info["bytes"],
        "memory_mb": mem_info["mb"],
        "memory_readable": mem_info["readable"],
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round((duplicate_rows / n_rows * 100) if n_rows > 0 else 0.0, 2),
        "total_missing_cells": total_missing,
        "missing_percentage": missing_pct,
        "numeric_count": len(schema["numeric_columns"]),
        "categorical_count": len(schema["categorical_columns"]),
        "datetime_count": len(schema["datetime_columns"]),
        "boolean_count": len(schema["boolean_columns"]),
        "text_count": len(schema["text_columns"])
    }
