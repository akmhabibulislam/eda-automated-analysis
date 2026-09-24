"""
DataSight Backend Service Router.
Exposes application architecture metadata and programmatic API access
for data ingestion, cleaning, statistics, and modeling pipelines.
"""

from typing import Dict, Any, List
import pandas as pd
from backend.ingestion.loaders import load_dataset, detect_schema, get_dataset_overview
from backend.ingestion.memory import get_system_memory, free_memory
from backend.cleaning.cleaning import run_automated_cleaning, AuditLogger
from backend.reporting.reports import generate_executive_summary


def get_service_status() -> Dict[str, Any]:
    """
    Returns platform health and resource utilization.
    """
    mem = get_system_memory()
    return {
        "status": "healthy",
        "service": "DataSight Analytics Engine",
        "system_memory": mem
    }


def analyze_dataset_programmatic(file_path: str) -> Dict[str, Any]:
    """
    Executes automated ingestion, auto-cleaning, and executive reporting programmatically.
    """
    df, meta = load_dataset(file_path, file_path, auto_optimize=True)
    overview_initial = get_dataset_overview(df)
    
    logger = AuditLogger()
    cleaned_df, clean_summary = run_automated_cleaning(df, logger=logger)
    overview_cleaned = get_dataset_overview(cleaned_df)
    
    summary_text = generate_executive_summary(cleaned_df, audit_logs=logger.get_logs())
    free_memory()
    
    return {
        "initial_overview": overview_initial,
        "clean_summary": clean_summary,
        "final_overview": overview_cleaned,
        "executive_summary": summary_text
    }


if __name__ == "__main__":
    status = get_service_status()
    print("DataSight Backend Initialized:", status)
