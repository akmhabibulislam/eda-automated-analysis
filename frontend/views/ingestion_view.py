"""
Workspace 1 View: Data Ingestion, Schema & Memory Profiling.
"""

import streamlit as st
import pandas as pd
import numpy as np

from backend.ingestion.memory import optimize_dataframe_memory, get_memory_usage
from backend.ingestion.loaders import load_dataset, load_from_database, detect_schema, get_dataset_overview
from backend.core.lineage import DatasetSessionManager


def render_ingestion_view(session_manager: DatasetSessionManager):
    st.subheader("Data Ingestion & Memory Profiling")
    st.caption("Load multi-format data files or execute secure database queries with integer downcasting and memory profiling.")

    src_tab, db_tab, demo_tab = st.tabs(["File Ingestion", "Database Connectivity", "Pre-Loaded Datasets"])

    with src_tab:
        up_file = st.file_uploader(
            "Select data file (CSV, TSV, Excel, JSON, Parquet)",
            type=["csv", "tsv", "tab", "xlsx", "xls", "json", "parquet"]
        )

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            downcast_int = st.checkbox("Integer Downcasting (Nullable Safe)", value=True)
        with col_c2:
            downcast_flt = st.checkbox("Float Downcasting (Precision Trade-off)", value=False)
        with col_c3:
            use_sample = st.checkbox("Limit Preview Rows", value=False)
            max_rows = st.number_input("Maximum Rows", min_value=100, max_value=2000000, value=50000) if use_sample else None

        if up_file is not None and st.button("Ingest and Optimize Dataset"):
            with st.spinner("Ingesting file and profiling memory footprint..."):
                try:
                    df, meta = load_dataset(up_file, up_file.name, auto_optimize=False, sample_rows=max_rows)
                    if downcast_int or downcast_flt:
                        df, _ = optimize_dataframe_memory(df, downcast_integers=downcast_int, downcast_floats=downcast_flt)
                    session_manager.load_dataset(df, dataset_name=up_file.name)
                    st.success(f"Successfully loaded '{up_file.name}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {str(e)}")

    with db_tab:
        st.markdown("##### Secure Relational Database Connector")
        st.caption("Read-only validation enforced (only SELECT queries permitted with database-side row limits).")
        db_conn = st.text_input("SQLAlchemy URI", value="sqlite:///example.db", placeholder="postgresql://user:pass@localhost:5432/dbname")
        db_sql = st.text_area("SQL Statement (Read-only)", value="SELECT * FROM dataset LIMIT 1000;")
        if st.button("Query Database"):
            with st.spinner("Executing query safely..."):
                try:
                    df, meta = load_from_database(db_conn, db_sql, auto_optimize=downcast_int)
                    session_manager.load_dataset(df, dataset_name="Database Query Result")
                    st.success(f"Loaded {len(df):,} records from database.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Database error: {str(e)}")

    with demo_tab:
        st.markdown("##### Pre-Configured Benchmark Datasets")
        demo_sel = st.selectbox("Benchmark Dataset", ["Enterprise Sales & Transactions", "Customer Behavioral Cohort", "Industrial Telemetry & Sensors"])
        if st.button("Load Benchmark"):
            np.random.seed(42)
            n = 1500
            if demo_sel == "Enterprise Sales & Transactions":
                dates = pd.date_range("2024-01-01", periods=n, freq="D")
                df_bench = pd.DataFrame({
                    "tx_id": [f"TX-{10000+i}" for i in range(n)],
                    "date": dates,
                    "region": np.random.choice(["North", "South", "East", "West"], n),
                    "category": np.random.choice(["Hardware", "Software", "Cloud", "Support"], n),
                    "units_sold": np.random.randint(1, 100, n),
                    "unit_price": np.random.uniform(50.0, 500.0, n).round(2),
                    "discount_rate": np.random.choice([0.0, 0.05, 0.10, 0.15], n),
                    "customer_rating": np.random.uniform(1.0, 5.0, n).round(1)
                })
                df_bench["revenue"] = (df_bench["units_sold"] * df_bench["unit_price"] * (1 - df_bench["discount_rate"])).round(2)
                df_bench.loc[np.random.choice(n, 25, replace=False), "customer_rating"] = np.nan
            elif demo_sel == "Customer Behavioral Cohort":
                df_bench = pd.DataFrame({
                    "cust_id": [f"CUST-{20000+i}" for i in range(n)],
                    "tier": np.random.choice(["Standard", "Business", "Enterprise"], n),
                    "age": np.random.randint(20, 70, n),
                    "tenure": np.random.randint(1, 60, n),
                    "monthly_spend": np.random.uniform(25.0, 300.0, n).round(2),
                    "total_spend": np.random.uniform(100.0, 10000.0, n).round(2),
                    "churn": np.random.choice(["Retained", "Churned"], n, p=[0.8, 0.2])
                })
                df_bench.loc[np.random.choice(n, 20, replace=False), "total_spend"] = np.nan
            else:
                timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
                t = np.linspace(0, 50, n)
                df_bench = pd.DataFrame({
                    "timestamp": timestamps,
                    "sensor_id": np.random.choice(["Sensor-A", "Sensor-B", "Sensor-C"], n),
                    "temperature": 25.0 + 5.0 * np.sin(t) + np.random.normal(0, 0.5, n),
                    "pressure": 101.3 + 1.5 * np.cos(t/2) + np.random.normal(0, 0.2, n),
                    "vibration": np.random.exponential(scale=1.2, size=n)
                })

            df_bench, _ = optimize_dataframe_memory(df_bench, downcast_integers=True)
            session_manager.load_dataset(df_bench, dataset_name=demo_sel)
            st.success(f"Loaded benchmark dataset '{demo_sel}'.")
            st.rerun()

    # Overview & Preview Grid
    df = session_manager.current_df
    if df is not None:
        st.markdown("---")
        st.markdown("#### Dataset Overview & Profile")

        ov = get_dataset_overview(df)
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Rows", f"{ov['total_rows']:,}")
        kpi2.metric("Columns", f"{ov['total_columns']:,}")
        kpi3.metric("RAM Footprint", ov["memory_readable"])
        kpi4.metric("Duplicates", f"{ov['duplicate_rows']:,} ({ov['duplicate_percentage']}%)")
        kpi5.metric("Missing Cells", f"{ov['total_missing_cells']:,} ({ov['missing_percentage']}%)")

        schema = detect_schema(df)
        st.markdown(
            f"**Schema Summary**: Numeric: `{ov['numeric_count']}` | "
            f"Categorical: `{ov['categorical_count']}` | "
            f"Datetime: `{ov['datetime_count']}` | "
            f"Boolean: `{ov['boolean_count']}` | "
            f"Text: `{ov['text_count']}`"
        )

        st.markdown("#### Interactive Data Grid")
        filter_col = st.selectbox("Search Column", ["All Columns"] + list(df.columns))
        search_kw = st.text_input("Filter Rows by Keyword", "")

        preview_grid = df
        if search_kw:
            if filter_col == "All Columns":
                mask = df.astype(str).apply(lambda r: r.str.contains(search_kw, case=False, na=False)).any(axis=1)
            else:
                mask = df[filter_col].astype(str).str.contains(search_kw, case=False, na=False)
            preview_grid = df[mask]

        st.dataframe(preview_grid.head(100), use_container_width=True)
        st.caption(f"Displaying top {min(100, len(preview_grid))} of {len(preview_grid):,} matching records.")
