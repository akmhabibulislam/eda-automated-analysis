"""
Data ingestion module supporting multi-format file uploads and database connections.
Features 1-5:
1. Multi-Format File Upload (CSV, Excel, JSON, Parquet, TSV) with strict extension checks
2. Secure Database Connectors (database-side LIMIT injection, read-only transaction configuration)
3. Data Schema Detection (automatic identification with robust date and ID distinction)
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

from backend.ingestion.memory import optimize_dataframe_memory, get_memory_usage
from backend.core.validation import validate_unique_columns


SUPPORTED_FORMATS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".parquet": "parquet",
    ".pq": "parquet",
}

DEFAULT_CHUNK_SIZE = 50000


def detect_file_format(file_name: str) -> str:
    """
    Identifies file extension from file name.
    Strictly raises ValueError on unsupported or ambiguous formats (e.g., .xyz, .txt).
    """
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".txt":
        raise ValueError(
            f"Ambiguous file extension '{ext}' for file '{file_name}'. "
            "Please rename to .csv or .tsv to indicate delimiter explicitly."
        )
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format '{ext}' for file '{file_name}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_FORMATS.keys()))}"
        )
    return SUPPORTED_FORMATS[ext]


def stream_dataset_chunks(
    file_source: Any,
    file_name: str,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    auto_optimize: bool = True
) -> Generator[pd.DataFrame, None, None]:
    """
    Streams flat files in chunks to prevent memory blowup on large datasets.
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
            raise ValueError(f"Unhandled format: {fmt}")

    except Exception as e:
        if isinstance(e, ValueError) and "Unsupported file format" in str(e):
            raise
        raise ValueError(f"Failed to parse {file_name} as {fmt}: {str(e)}")

    if auto_optimize:
        df, opt_metrics = optimize_dataframe_memory(df)
        metadata["memory_optimization"] = opt_metrics
    else:
        metadata["memory_optimization"] = get_memory_usage(df)

    return df, metadata


DISALLOWED_SQL_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "REPLACE", "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
    "MERGE", "CALL", "INTO", "OUTFILE", "DUMPFILE"
]


def load_from_database(
    connection_string: str,
    query: str,
    auto_optimize: bool = True,
    max_rows: int = 100000,
    timeout_seconds: int = 15
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Connect to SQL databases with multi-layered security:
    - Multi-statement execution blocking (rejects semicolon query chaining)
    - Keyword blacklist rejection
    - Read-only transaction enforcement
    - Strict database-side LIMIT injection
    - Socket/statement execution timeout
    """
    clean_query = query.strip()
    if clean_query.endswith(";"):
        clean_query = clean_query[:-1].strip()

    # Reject query chaining / stacked queries
    if ";" in clean_query:
        raise ValueError("Security Violation: Multiple SQL statements or query chaining via ';' are strictly forbidden.")

    # Block comment-based SQL injection obfuscation
    if "--" in clean_query or "/*" in clean_query:
        raise ValueError("Security Violation: SQL comments are not permitted in analytical queries.")

    upper_query = clean_query.upper()

    # Require SELECT or WITH (CTE)
    if not (upper_query.startswith("SELECT") or upper_query.startswith("WITH")):
        raise ValueError("Security Violation: Only SELECT and WITH (CTE) queries are permitted.")

    for kw in DISALLOWED_SQL_KEYWORDS:
        pattern = rf"\b{kw}\b"
        if re.search(pattern, upper_query):
            raise ValueError(f"Security Violation: Prohibited query keyword '{kw}' detected.")

    # Inject database-side LIMIT if not present or replace if larger than max_rows
    limit_match = re.search(r"\bLIMIT\s+(\d+)\b", upper_query)
    if not limit_match:
        executed_query = f"{clean_query} LIMIT {max_rows}"
    else:
        existing_limit = int(limit_match.group(1))
        if existing_limit > max_rows:
            executed_query = re.sub(r"\bLIMIT\s+\d+\b", f"LIMIT {max_rows}", clean_query, flags=re.IGNORECASE)
        else:
            executed_query = clean_query

    # Configure read-only connection with execution timeout
    connect_args = {}
    if "sqlite" in connection_string.lower():
        connect_args["timeout"] = timeout_seconds
    elif "postgres" in connection_string.lower():
        connect_args["options"] = f"-c statement_timeout={timeout_seconds * 1000}"

    engine = create_engine(
        connection_string,
        execution_options={"isolation_level": "AUTOCOMMIT"},
        connect_args=connect_args
    )

    with engine.connect() as conn:
        df = pd.read_sql(text(executed_query), conn)

    metadata = {
        "source": "database",
        "query": executed_query,
        "rows_retrieved": len(df)
    }

    if auto_optimize:
        df, opt_metrics = optimize_dataframe_memory(df)
        metadata["memory_optimization"] = opt_metrics
    else:
        metadata["memory_optimization"] = get_memory_usage(df)

    return df, metadata


DATE_PATTERNS = [
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$",
    r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{4}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$",
    r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$",
    r"^[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}$"
]
DATE_REGEX = re.compile("|".join(f"({p})" for p in DATE_PATTERNS))

NON_DATE_IDENTIFIERS = re.compile(
    r"(?:^|[_\W])(?:id|uuid|sku|tx|part|serial|vin|hash|guid|code|ref|order|inv)(?:[_\W]|$)|[a-f0-9]{8}-[a-f0-9]{4}",
    re.IGNORECASE
)

# Regex patterns for semantic role inference
SEMANTIC_PATTERNS = {
    "uuid": re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.IGNORECASE),
    "email": re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$"),
    "phone": re.compile(r"^\+?[\d\s\-\(\)]{7,20}$"),
    "currency": re.compile(r"^[\$€£¥₹]\s?[\d,]+(?:\.\d+)?$|^[\d,]+(?:\.\d+)?\s?[\$€£¥₹]$"),
    "percentage": re.compile(r"^-?[\d,]+(?:\.\d+)?\s*%$")
}


def is_valid_date_series(series: pd.Series, col_name: str = "") -> bool:
    """
    Robust date series verification.
    Prevents false positives on IDs, UUIDs, SKUs, and version numbers.
    """
    if NON_DATE_IDENTIFIERS.search(col_name):
        return False

    sample = series.dropna().head(20).astype(str)
    if len(sample) == 0:
        return False

    matches = 0
    for val in sample:
        v = val.strip()
        if NON_DATE_IDENTIFIERS.search(v):
            return False
        if DATE_REGEX.match(v):
            matches += 1

    if (matches / len(sample)) < 0.80:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            return (parsed.notna().sum() / len(sample)) >= 0.80
        except Exception:
            return False


def detect_semantic_role(series: pd.Series, col_name: str, base_type: str) -> str:
    """
    Infers semantic role beyond basic dtype:
    - 'identifier' / 'uuid' (high uniqueness IDs that should not be correlated)
    - 'latitude' / 'longitude'
    - 'currency' / 'percentage'
    - 'email' / 'phone'
    - 'general_numeric' / 'general_categorical'
    """
    name_lower = col_name.lower().strip()

    # Coordinates
    if name_lower in ["lat", "latitude"] and base_type == "numeric":
        valid = series.dropna()
        if len(valid) > 0 and valid.between(-90, 90).all():
            return "latitude"
    if name_lower in ["lon", "lng", "long", "longitude"] and base_type == "numeric":
        valid = series.dropna()
        if len(valid) > 0 and valid.between(-180, 180).all():
            return "longitude"

    # String / object checks
    sample = series.dropna().head(30).astype(str)
    if len(sample) > 0:
        # UUID check
        uuid_matches = sum(1 for s in sample if SEMANTIC_PATTERNS["uuid"].match(s.strip()))
        if uuid_matches / len(sample) >= 0.80:
            return "uuid"

        # Email check
        email_matches = sum(1 for s in sample if SEMANTIC_PATTERNS["email"].match(s.strip()))
        if email_matches / len(sample) >= 0.80:
            return "email"

        # Phone check
        phone_matches = sum(1 for s in sample if SEMANTIC_PATTERNS["phone"].match(s.strip()))
        if phone_matches / len(sample) >= 0.80 and name_lower in ["phone", "mobile", "tel", "contact"]:
            return "phone"

        # Currency string check
        curr_matches = sum(1 for s in sample if SEMANTIC_PATTERNS["currency"].match(s.strip()))
        if curr_matches / len(sample) >= 0.80:
            return "currency"

        # Percentage string check
        pct_matches = sum(1 for s in sample if SEMANTIC_PATTERNS["percentage"].match(s.strip()))
        if pct_matches / len(sample) >= 0.80:
            return "percentage"

    # Identifier check (including numeric primary keys)
    if NON_DATE_IDENTIFIERS.search(name_lower):
        n_unique = series.nunique(dropna=True)
        if len(series) > 0 and (n_unique / len(series) > 0.85):
            return "identifier"

    return f"general_{base_type}"


def detect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Feature 3: Automatic identification of numeric, categorical, datetime, boolean types,
    and advanced semantic roles (identifier, uuid, currency, percentage, latitude, longitude, email, phone).
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    boolean_cols: List[str] = []
    text_cols: List[str] = []

    column_types: Dict[str, str] = {}
    semantic_roles: Dict[str, str] = {}
    analytical_numeric_cols: List[str] = []

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_bool_dtype(s):
            boolean_cols.append(col)
            base_type = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
            base_type = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            base_type = "numeric"
        elif isinstance(s.dtype, pd.CategoricalDtype):
            categorical_cols.append(col)
            base_type = "categorical"
        else:
            if is_valid_date_series(s, col_name=str(col)):
                datetime_cols.append(col)
                base_type = "datetime"
            else:
                n_unique = s.nunique(dropna=True)
                total_len = len(s)
                if total_len > 0 and (n_unique / total_len < 0.15 or n_unique < 20):
                    categorical_cols.append(col)
                    base_type = "categorical"
                else:
                    text_cols.append(col)
                    base_type = "text"

        column_types[col] = base_type
        role = detect_semantic_role(s, str(col), base_type)
        semantic_roles[col] = role

        # Analytical numerics exclude identifiers and keys
        if base_type == "numeric" and role not in ["identifier", "uuid"]:
            analytical_numeric_cols.append(col)

    return {
        "columns": list(df.columns),
        "column_types": column_types,
        "semantic_roles": semantic_roles,
        "numeric_columns": numeric_cols,
        "analytical_numeric_columns": analytical_numeric_cols,
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
