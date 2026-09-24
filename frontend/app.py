"""
DataSight Enterprise Analytics Platform.
Main UI Controller with modern workspace layout and optimized execution pipelines.
Integrates all 52 end-to-end data analysis features across 6 structured workspaces:
- Workspace 1: Data Ingestion, Schema & Memory Profiling (Features 1-5 + Memory Optimization)
- Workspace 2: Data Quality, Recommendations & Controlled Clean (Features 6-13)
- Workspace 3: Advanced Wrangling & Reshaping (Features 20-26)
- Workspace 4: Statistics & Hypothesis Testing (Features 14-19, 27-31)
- Workspace 5: Time-Series & Mathematical Modeling (Features 32-41)
- Workspace 6: Dynamic Visual Studio & Executive Reports (Features 42-52)
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import streamlit as st

# Configure page layout and professional theme
st.set_page_config(
    page_title="DataSight Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project root import resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ingestion.memory import (
    optimize_dataframe_memory,
    get_memory_usage,
    get_system_memory,
    free_memory
)
from backend.ingestion.loaders import (
    load_dataset,
    load_from_database,
    detect_schema,
    get_dataset_overview
)
from backend.cleaning.cleaning import (
    AuditLogger,
    standardize_column_headers,
    impute_missing_values,
    drop_missing_values,
    remove_duplicates,
    treat_outliers,
    clean_text_columns,
    cast_data_types,
    analyze_cleaning_recommendations,
    run_automated_cleaning
)
from backend.cleaning.wrangling import (
    add_custom_formula_column,
    bin_continuous_column,
    aggregate_groupby,
    merge_datasets,
    extract_regex_patterns,
    pivot_dataframe,
    unpivot_dataframe,
    filter_rows_structured
)
from backend.analysis.statistics import (
    compute_univariate_summary,
    compute_categorical_distribution,
    compute_missingness_matrix,
    compute_correlation_matrix,
    detect_highly_correlated_pairs,
    compute_variance_inflation_factors,
    compute_skewness_kurtosis
)
from backend.analysis.hypothesis import (
    run_t_test,
    run_anova,
    run_chi_square,
    run_non_parametric_tests,
    fit_distributions
)
from backend.analysis.timeseries import (
    resample_temporal_data,
    compute_rolling_metrics,
    compute_period_over_period_growth,
    compute_cumulative_totals,
    decompose_seasonality_trend
)
from backend.modeling.modeling import (
    fit_curve_and_equation,
    run_symbolic_regression,
    run_kmeans_clustering,
    run_pca_reduction,
    run_isolation_forest_anomaly_detection
)
from frontend.components.visualizations import (
    create_auto_plot,
    create_distribution_chart,
    create_relationship_chart,
    create_categorical_chart,
    build_custom_chart
)
from backend.reporting.reports import (
    export_dataset_bytes,
    export_chart_html,
    generate_executive_summary,
    generate_html_report,
    generate_pdf_report
)

# Inject custom enterprise CSS
css_path = os.path.join(PROJECT_ROOT, "frontend", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "audit_logger" not in st.session_state:
    st.session_state.audit_logger = AuditLogger()
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "No dataset loaded"
if "recent_figures" not in st.session_state:
    st.session_state.recent_figures = []


def register_figure(fig):
    st.session_state.recent_figures.append(fig)
    if len(st.session_state.recent_figures) > 3:
        st.session_state.recent_figures.pop(0)


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### DataSight Platform")
    st.caption("Enterprise Exploratory Data Analysis & Analytics Engine")
    st.markdown("---")

    active_workspace = st.radio(
        "Analytical Workspace",
        [
            "1. Ingestion & Memory Profiling",
            "2. Data Quality & Controlled Clean",
            "3. Advanced Data Wrangling",
            "4. Statistics & Hypothesis Testing",
            "5. Time-Series & Math Modeling",
            "6. Visualization Studio & Reports"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### System Resource Health")
    sys_mem = get_system_memory()
    st.write(f"**RAM Utilization**: {sys_mem['used_mb']:,.0f} MB / {sys_mem['total_mb']:,.0f} MB ({sys_mem['percent']}%)")

    if st.session_state.df is not None:
        df_mem = get_memory_usage(st.session_state.df)
        st.write(f"**Active Dataset**: {df_mem['readable']}")

    if st.button("Explicit Garbage Collection"):
        free_memory(force=True)
        st.success("Freed unreferenced memory.")


# ==============================================================================
# TOP STATUS BANNER
# ==============================================================================
ds_status = "Active" if st.session_state.df is not None else "Awaiting Data"
badge_color = "#ecfdf5" if st.session_state.df is not None else "#fef2f2"
badge_text_color = "#047857" if st.session_state.df is not None else "#b91c1c"
rows_str = f"{len(st.session_state.df):,} Rows" if st.session_state.df is not None else "0 Rows"
cols_str = f"{len(st.session_state.df.columns)} Columns" if st.session_state.df is not None else "0 Columns"

st.markdown(f"""
<div class="app-header-banner">
    <div class="app-title-group">
        <h1>DataSight Analytics Engine</h1>
        <p>Current Dataset: <strong>{st.session_state.dataset_name}</strong> ({rows_str} &bull; {cols_str})</p>
    </div>
    <div class="app-status-badge" style="background-color: {badge_color}; color: {badge_text_color};">
        {ds_status}
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# WORKSPACE 1: Data Ingestion & Memory Profiling (Features 1-5 + Memory Optimization)
# ==============================================================================
if active_workspace == "1. Ingestion & Memory Profiling":
    st.subheader("Data Ingestion & Memory Profiling")
    st.caption("Load multi-format data files or execute secure database queries with integer downcasting and memory profiling.")

    src_tab, db_tab, demo_tab = st.tabs(["File Ingestion", "Database Connectivity", "Pre-Loaded Datasets"])

    with src_tab:
        up_file = st.file_uploader("Select data file (CSV, TSV, Excel, JSON, Parquet)", type=["csv", "tsv", "tab", "xlsx", "xls", "json", "parquet"])

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            downcast_int = st.checkbox("Integer Downcasting (Safe)", value=True)
        with col_c2:
            downcast_flt = st.checkbox("Float Downcasting (Precision trade-off)", value=False)
        with col_c3:
            use_sample = st.checkbox("Limit Preview Rows", value=False)
            max_rows = st.number_input("Maximum Rows", min_value=100, max_value=2000000, value=50000) if use_sample else None

        if up_file is not None and st.button("Ingest and Optimize Dataset"):
            with st.spinner("Ingesting file and profiling memory footprint..."):
                try:
                    df, meta = load_dataset(up_file, up_file.name, auto_optimize=False, sample_rows=max_rows)
                    if downcast_int or downcast_flt:
                        df, opt_m = optimize_dataframe_memory(df, downcast_integers=downcast_int, downcast_floats=downcast_flt)
                        meta["memory_optimization"] = opt_m
                    st.session_state.df = df
                    st.session_state.original_df = df.copy()
                    st.session_state.dataset_name = up_file.name
                    st.session_state.audit_logger.clear()
                    st.session_state.audit_logger.log("File Ingestion", f"Loaded {up_file.name} (Format: {meta['format']})", len(df), list(df.columns))
                    st.success(f"Successfully loaded {up_file.name}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {str(e)}")

    with db_tab:
        st.markdown("##### Secure Relational Database Connector")
        st.caption("Read-only validation enforced (only SELECT queries permitted with configurable timeouts).")
        db_conn = st.text_input("SQLAlchemy URI", value="sqlite:///example.db", placeholder="postgresql://user:pass@localhost:5432/dbname")
        db_sql = st.text_area("SQL Statement (Read-only)", value="SELECT * FROM dataset LIMIT 1000;")
        if st.button("Query Database"):
            with st.spinner("Executing query safely..."):
                try:
                    df, meta = load_from_database(db_conn, db_sql, auto_optimize=downcast_int)
                    st.session_state.df = df
                    st.session_state.original_df = df.copy()
                    st.session_state.dataset_name = "Database Query"
                    st.session_state.audit_logger.clear()
                    st.session_state.audit_logger.log("Database Query", f"Executed: {db_sql[:50]}...", len(df), list(df.columns))
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
            st.session_state.df = df_bench
            st.session_state.original_df = df_bench.copy()
            st.session_state.dataset_name = demo_sel
            st.session_state.audit_logger.clear()
            st.session_state.audit_logger.log("Benchmark Ingestion", f"Loaded {demo_sel}", len(df_bench), list(df_bench.columns))
            st.success(f"Loaded benchmark dataset '{demo_sel}'.")
            st.rerun()

    # Overview & Preview Grid
    if st.session_state.df is not None:
        df = st.session_state.df
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


# ==============================================================================
# WORKSPACE 2: Data Quality, Recommendations & Controlled Clean (Features 6-13)
# ==============================================================================
elif active_workspace == "2. Data Quality & Controlled Clean":
    st.subheader("Data Quality, Recommendations & Controlled Cleaning")
    st.caption("Review automated recommendations, configure controlled cleaning pipelines, and inspect data lineage without blind mutations.")

    if st.session_state.df is None:
        st.info("Please load a dataset in Workspace 1 first.")
    else:
        df = st.session_state.df
        logger = st.session_state.audit_logger

        # Automated Cleaning Recommendations Card
        st.markdown("#### Automated Quality Recommendations")
        recs = analyze_cleaning_recommendations(df)
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.write(f"**Duplicate Rows**: {recs['duplicate_rows']}")
            st.write(f"**Columns with Missing Values**: {len(recs['missing_columns'])}")
            if recs["missing_columns"]:
                st.caption(f"Missing Columns: {list(recs['missing_columns'].keys())}")
        with r_col2:
            st.write(f"**Header Standardization Needed**: {'Yes' if recs['header_standardization_needed'] else 'No'}")
            st.write(f"**Outlier Candidate Columns**: {len(recs['outlier_candidates'])}")
            if recs["outlier_candidates"]:
                st.caption(f"Outlier Columns: {list(recs['outlier_candidates'].keys())}")

        st.markdown("---")
        st.markdown("#### Controlled Auto-Clean Pipeline")
        st.write("Configure which operations to execute (imputation and outlier capping are strictly opt-in).")

        cfg_c1, cfg_c2, cfg_c3, cfg_c4, cfg_c5 = st.columns(5)
        with cfg_c1:
            opt_headers = st.checkbox("Standardize Headers", value=True)
        with cfg_c2:
            opt_dups = st.checkbox("Purge Duplicates", value=True)
        with cfg_c3:
            opt_strings = st.checkbox("Trim String Spaces", value=True)
        with cfg_c4:
            opt_impute = st.checkbox("Impute Missing (Median/Mode)", value=False)
        with cfg_c5:
            opt_outliers = st.checkbox("Cap Outliers (IQR 3.0)", value=False)

        if st.button("Execute Controlled Cleaning Pipeline", type="primary"):
            with st.spinner("Executing configured pipeline..."):
                cleaned_df, summary = run_automated_cleaning(
                    df,
                    logger=logger,
                    impute_missing=opt_impute,
                    cap_outliers=opt_outliers,
                    standardize_headers=opt_headers,
                    purge_duplicates=opt_dups,
                    clean_strings=opt_strings
                )
                st.session_state.df = cleaned_df
                st.success(f"Cleaning complete. {summary['steps_executed']} operations recorded.")
                st.rerun()

        st.markdown("---")
        q_tab1, q_tab2, q_tab3, q_tab4, q_tab5, q_tab6 = st.tabs([
            "Missing Imputation", "Duplicate Purge", "Outlier Treatment", "Header & Text Sanitization", "Data Type Cast", "Audit Trail"
        ])

        with q_tab1:
            st.markdown("##### Impute or Drop Missing Data")
            c_nulls = [c for c in df.columns if df[c].isna().sum() > 0]
            col_imp, col_drop = st.columns(2)
            with col_imp:
                st.markdown("**Imputation Engine**")
                sel_imp_cols = st.multiselect("Columns to Impute", df.columns, default=c_nulls)
                strat = st.selectbox("Strategy", ["mean", "median", "mode", "constant", "ffill", "bfill"])
                const_val = st.text_input("Constant Value", "0") if strat == "constant" else None

                if st.button("Execute Imputation"):
                    cv = float(const_val) if const_val and const_val.replace(".", "").isdigit() else const_val
                    st.session_state.df = impute_missing_values(df, columns=sel_imp_cols, strategy=strat, fill_value=cv, logger=logger)
                    st.success("Imputed missing values.")
                    st.rerun()

            with col_drop:
                st.markdown("**Threshold Removal**")
                c_th = st.slider("Drop column if missing > X%", 10, 100, 50)
                r_th = st.slider("Drop row if missing > X%", 10, 100, 50)
                if st.button("Drop Missing by Threshold"):
                    st.session_state.df = drop_missing_values(df, row_threshold_pct=float(r_th), col_threshold_pct=float(c_th), logger=logger)
                    st.success("Dropped missing rows/columns.")
                    st.rerun()

        with q_tab2:
            st.markdown("##### Duplicate Detection and Purging")
            sub_dup = st.multiselect("Subset columns for duplicate evaluation", df.columns)
            k_rule = st.selectbox("Retention Rule", ["first", "last", False], format_func=lambda x: "Drop all duplicates" if x is False else f"Keep {x}")
            if st.button("Purge Duplicates"):
                st.session_state.df = remove_duplicates(df, subset=sub_dup if sub_dup else None, keep=k_rule, logger=logger)
                st.success("Duplicates purged.")
                st.rerun()

        with q_tab3:
            st.markdown("##### Outlier Detection & Treatment")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                o_col = st.selectbox("Target Metric", num_cols)
                o_meth = st.selectbox("Method", ["iqr", "zscore"])
                o_th = st.number_input("Threshold Multiplier", min_value=1.0, max_value=5.0, value=1.5 if o_meth == "iqr" else 3.0, step=0.1)
                o_act = st.selectbox("Action", ["cap", "drop", "flag"])

                if st.button("Execute Outlier Treatment"):
                    treated_df, info = treat_outliers(df, column=o_col, method=o_meth, threshold=float(o_th), action=o_act, logger=logger)
                    st.session_state.df = treated_df
                    st.success(f"Processed {info['outliers_detected']} anomalies outside bounds.")
                    st.rerun()
            else:
                st.info("No numeric columns available.")

        with q_tab4:
            st.markdown("##### Header Standardization & Text Sanitization")
            h_col, t_col = st.columns(2)
            with h_col:
                st.markdown("**Header Formatting**")
                c_style = st.selectbox("Case Convention", ["snake_case", "lower_case", "upper_case", "camelCase"])
                if st.button("Standardize Headers"):
                    st.session_state.df = standardize_column_headers(df, case_style=c_style, logger=logger)
                    st.success("Headers formatted.")
                    st.rerun()

            with t_col:
                st.markdown("**Text String Cleaning (Nulls Preserved)**")
                str_cols = [c for c in df.columns if isinstance(df[c].dtype, (pd.CategoricalDtype, pd.StringDtype)) or df[c].dtype == "object"]
                t_cols_sel = st.multiselect("Text Columns to Clean", str_cols)
                tr_ws = st.checkbox("Trim Whitespace", value=True)
                tr_case = st.selectbox("Case Adjustment", ["None", "lower", "upper", "title"])
                tr_sp = st.checkbox("Strip Special Characters", value=False)

                if st.button("Sanitize Text Columns"):
                    st.session_state.df = clean_text_columns(
                        df,
                        columns=t_cols_sel,
                        strip_whitespace=tr_ws,
                        case_transformation=None if tr_case == "None" else tr_case,
                        remove_special_chars=tr_sp,
                        logger=logger
                    )
                    st.success("Text sanitized.")
                    st.rerun()

        with q_tab5:
            st.markdown("##### Data Type Casting (with Robust Boolean Parser)")
            cast_col = st.selectbox("Column to Cast", df.columns)
            target_t = st.selectbox("Target Datatype", ["int", "float", "string", "datetime", "boolean", "category"])
            if st.button("Apply Datatype Cast"):
                try:
                    st.session_state.df = cast_data_types(df, {cast_col: target_t}, logger=logger)
                    st.success(f"Converted '{cast_col}' to {target_t}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Cast failed: {str(e)}")

        with q_tab6:
            st.markdown("##### Audit Trail & Data Lineage")
            a_df = logger.to_dataframe()
            if a_df.empty:
                st.info("No transformation events recorded in this session.")
            else:
                st.dataframe(a_df, use_container_width=True)
                if st.button("Reset Dataset to Initial State"):
                    st.session_state.df = st.session_state.original_df.copy()
                    st.session_state.audit_logger.clear()
                    st.session_state.audit_logger.log("Dataset Reset", "Reverted to initial uploaded state")
                    st.success("Reset complete.")
                    st.rerun()


# ==============================================================================
# WORKSPACE 3: Advanced Wrangling & Reshaping (Features 20-26)
# ==============================================================================
elif active_workspace == "3. Advanced Data Wrangling":
    st.subheader("Advanced Data Wrangling & Reshaping")
    st.caption("Secure AST formula evaluation, continuous binning, SQL-like aggregations, regex pattern extraction, and structured filtering.")

    if st.session_state.df is None:
        st.info("Please load a dataset in Workspace 1 first.")
    else:
        df = st.session_state.df
        logger = st.session_state.audit_logger

        w_tab1, w_tab2, w_tab3, w_tab4, w_tab5, w_tab6 = st.tabs([
            "Secure Formula Builder", "Continuous Binning", "SQL Aggregation", "Regex Extraction", "Pivot / Reshape", "Structured Filter"
        ])

        with w_tab1:
            st.markdown("##### Secure AST Formula Column Builder")
            st.caption("Evaluates mathematical expressions safely using an abstract syntax tree parser without code injection risks.")
            new_col = st.text_input("New Column Name", "calculated_metric")
            form_expr = st.text_input("Formula Expression (e.g. `revenue - cost` or `sqrt(units_sold) * 10`)", "")
            st.write(f"Available columns: `{', '.join(df.columns)}`")

            if st.button("Evaluate Formula"):
                try:
                    st.session_state.df = add_custom_formula_column(df, new_col, form_expr, logger=logger)
                    st.success(f"Created '{new_col}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Formula evaluation error: {str(e)}")

        with w_tab2:
            st.markdown("##### Continuous Binning & Discretization")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                b_col = st.selectbox("Continuous Metric", num_cols)
                b_name = st.text_input("Binned Column Name", f"{b_col}_binned")
                b_mode = st.selectbox("Binning Method", ["equal_width", "quantile"])
                b_cnt = st.slider("Number of Bins", 2, 20, 5)

                if st.button("Apply Binning"):
                    st.session_state.df = bin_continuous_column(df, column=b_col, new_column_name=b_name, bins=b_cnt, bin_type=b_mode, logger=logger)
                    st.success(f"Binned '{b_col}' into '{b_name}'.")
                    st.rerun()
            else:
                st.info("No numeric columns available.")

        with w_tab3:
            st.markdown("##### SQL-like Groupby & Aggregations")
            grp_by = st.multiselect("Group Columns", df.columns)
            agg_metric = st.selectbox("Metric to Aggregate", df.select_dtypes(include=[np.number]).columns.tolist())
            agg_funcs = st.multiselect("Functions", ["mean", "sum", "count", "std", "min", "max", "median"], default=["mean", "sum"])

            if grp_by and agg_metric and agg_funcs and st.button("Run Aggregations"):
                res_agg = aggregate_groupby(df, group_columns=grp_by, aggregations={agg_metric: agg_funcs})
                st.dataframe(res_agg, use_container_width=True)

        with w_tab4:
            st.markdown("##### Text Regex Extraction")
            text_cols = [c for c in df.columns if isinstance(df[c].dtype, (pd.CategoricalDtype, pd.StringDtype)) or df[c].dtype == "object"]
            if text_cols:
                rx_src = st.selectbox("Source Column", text_cols)
                rx_pat = st.text_input("Regex Pattern", r"([A-Za-z0-9]+)")
                rx_dest = st.text_input("Destination Column", f"{rx_src}_extracted")

                if st.button("Extract Regex"):
                    try:
                        st.session_state.df = extract_regex_patterns(df, source_column=rx_src, pattern=rx_pat, new_column_name=rx_dest, logger=logger)
                        st.success(f"Pattern extracted into '{rx_dest}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Regex error: {str(e)}")
            else:
                st.info("No text columns found.")

        with w_tab5:
            st.markdown("##### Reshaping: Pivot & Unpivot")
            resh_mode = st.radio("Operation", ["Pivot (Long to Wide)", "Unpivot / Melt (Wide to Long)"])
            if resh_mode == "Pivot (Long to Wide)":
                p_idx = st.multiselect("Index Column(s)", df.columns)
                p_cols = st.selectbox("Pivot Column", df.columns)
                p_vals = st.selectbox("Values Column", df.select_dtypes(include=[np.number]).columns.tolist())
                p_agg = st.selectbox("Aggregation", ["mean", "sum", "count", "min", "max"])

                if p_idx and p_cols and p_vals and st.button("Execute Pivot"):
                    p_res = pivot_dataframe(df, index_cols=p_idx, columns=p_cols, values=p_vals, aggfunc=p_agg)
                    st.dataframe(p_res, use_container_width=True)
            else:
                u_id = st.multiselect("Fixed Identifiers", df.columns)
                u_val = st.multiselect("Value Columns to Unpivot", [c for c in df.columns if c not in u_id])
                if u_id and u_val and st.button("Execute Unpivot"):
                    u_res = unpivot_dataframe(df, id_vars=u_id, value_vars=u_val)
                    st.dataframe(u_res, use_container_width=True)

        with w_tab6:
            st.markdown("##### Structured Row Query Filter")
            st.caption("Applies typed, parametric comparisons safely without unrestricted expression evaluation.")
            q_col = st.selectbox("Filter Column", df.columns)
            q_op = st.selectbox("Operator", ["==", "!=", ">", ">=", "<", "<=", "contains", "in"])
            q_val = st.text_input("Comparison Value", "")

            if q_val and st.button("Apply Structured Filter"):
                try:
                    st.session_state.df = filter_rows_structured(df, column=q_col, operator=q_op, comparison_value=q_val, logger=logger)
                    st.success("Filter applied.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Filter error: {str(e)}")


# ==============================================================================
# WORKSPACE 4: Statistics & Hypothesis Testing (Features 14-19, 27-31)
# ==============================================================================
elif active_workspace == "4. Statistics & Hypothesis Testing":
    st.subheader("Statistical Analysis & Hypothesis Validation")
    st.caption("Descriptive statistics, pairwise correlations, VIF multicollinearity, parametric/non-parametric tests, and distribution fitting.")

    if st.session_state.df is None:
        st.info("Please load a dataset in Workspace 1 first.")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if isinstance(df[c].dtype, (pd.CategoricalDtype, pd.StringDtype)) or df[c].dtype == "object"]

        s_tab1, s_tab2, s_tab3, s_tab4, s_tab5 = st.tabs([
            "Descriptive Stats", "Correlations & VIF", "T-Tests & ANOVA", "Categorical Tests", "Distribution Fitting"
        ])

        with s_tab1:
            st.markdown("##### Univariate Descriptive Statistics")
            uni_df = compute_univariate_summary(df)
            if not uni_df.empty:
                st.dataframe(uni_df, use_container_width=True)
            st.markdown("##### Skewness & Kurtosis Distribution Symmetry")
            sk_df = compute_skewness_kurtosis(df)
            if not sk_df.empty:
                st.dataframe(sk_df, use_container_width=True)

        with s_tab2:
            st.markdown("##### Correlation Matrix")
            c_meth = st.selectbox("Method", ["pearson", "spearman", "kendall"])
            corr_m = compute_correlation_matrix(df, method=c_meth)
            if not corr_m.empty:
                st.dataframe(corr_m, use_container_width=True)

            st.markdown("##### Highly Correlated Feature Pairs (Pairwise Correlation >= 0.80)")
            corr_pairs = detect_highly_correlated_pairs(df, threshold=0.80)
            if corr_pairs:
                for w in corr_pairs:
                    st.warning(f"{w['severity']} Correlation: `{w['column_1']}` and `{w['column_2']}` (Coeff: {w['correlation']})")
            else:
                st.success("No feature pairs exceed the 0.80 correlation threshold.")

            st.markdown("##### Variance Inflation Factor (VIF) Multi-Collinearity Calculation")
            vif_df = compute_variance_inflation_factors(df)
            if not vif_df.empty:
                st.dataframe(vif_df, use_container_width=True)

        with s_tab3:
            st.markdown("##### T-Tests & One-Way ANOVA")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("**T-Test Analysis**")
                if len(num_cols) >= 2:
                    tt_s1 = st.selectbox("Sample 1", num_cols, index=0)
                    tt_s2 = st.selectbox("Sample 2", num_cols, index=1)
                    tt_type = st.selectbox("Type", ["independent", "paired"])
                    if st.button("Run T-Test"):
                        res_tt = run_t_test(df[tt_s1], df[tt_s2], test_type=tt_type)
                        st.json(res_tt)

            with t_col2:
                st.markdown("**ANOVA (F-Test)**")
                if num_cols and cat_cols:
                    an_val = st.selectbox("Metric", num_cols)
                    an_grp = st.selectbox("Group Factor", cat_cols)
                    if st.button("Run ANOVA"):
                        groups = [group[an_val].values for _, group in df.groupby(an_grp, observed=False)]
                        res_an = run_anova(groups)
                        st.json(res_an)

        with s_tab4:
            st.markdown("##### Chi-Square & Non-Parametric Tests")
            c1_test, c2_test = st.columns(2)
            with c1_test:
                st.markdown("**Chi-Square Independence**")
                if len(cat_cols) >= 2:
                    c_x1 = st.selectbox("Variable 1", cat_cols, index=0)
                    c_x2 = st.selectbox("Variable 2", cat_cols, index=1)
                    if st.button("Execute Chi-Square"):
                        ct = pd.crosstab(df[c_x1], df[c_x2])
                        res_chi = run_chi_square(ct)
                        st.json({k: v for k, v in res_chi.items() if k != "expected_frequencies"})
                        st.dataframe(res_chi["expected_frequencies"])

            with c2_test:
                st.markdown("**Non-Parametric Tests (Mann-Whitney / Kruskal-Wallis)**")
                if num_cols and cat_cols:
                    np_val = st.selectbox("Non-Parametric Metric", num_cols)
                    np_grp = st.selectbox("Grouping Factor", cat_cols)
                    if st.button("Run Non-Parametric"):
                        groups = [group[np_val].values for _, group in df.groupby(np_grp, observed=False)]
                        res_np = run_non_parametric_tests(groups)
                        st.json(res_np)

        with s_tab5:
            st.markdown("##### Distribution Fitting (Kolmogorov-Smirnov)")
            if num_cols:
                dist_col = st.selectbox("Metric to Fit", num_cols)
                if st.button("Evaluate Distribution Fits"):
                    fit_tbl = fit_distributions(df[dist_col].values)
                    st.dataframe(fit_tbl, use_container_width=True)


# ==============================================================================
# WORKSPACE 5: Time-Series & Mathematical Modeling (Features 32-41)
# ==============================================================================
elif active_workspace == "5. Time-Series & Math Modeling":
    st.subheader("Time-Series & Mathematical Modeling")
    st.caption("Temporal decomposition, curve fitting trendlines, parametric function-family search, clustering with inverse centers, and PCA loadings.")

    if st.session_state.df is None:
        st.info("Please load a dataset in Workspace 1 first.")
    else:
        df = st.session_state.df
        dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        m_tab1, m_tab2, m_tab3, m_tab4, m_tab5 = st.tabs([
            "Time-Series Analysis", "Curve Fitting", "Parametric Function Search", "K-Means & PCA", "Isolation Forest"
        ])

        with m_tab1:
            st.markdown("##### Temporal Aggregations & Seasonal Decomposition")
            if not dt_cols:
                st.info("No datetime column detected.")
                cand_dt = st.selectbox("Select column to cast as Datetime", df.columns)
                if st.button("Parse as Datetime"):
                    st.session_state.df[cand_dt] = pd.to_datetime(st.session_state.df[cand_dt], errors="coerce")
                    st.success("Parsed column as Datetime.")
                    st.rerun()
            else:
                ts_dt = st.selectbox("Timestamp Column", dt_cols)
                ts_val = st.selectbox("Series Metric", num_cols)

                col_res, col_roll = st.columns(2)
                with col_res:
                    st.markdown("**Resampling**")
                    f_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE", "Annual": "YE"}
                    f_choice = st.selectbox("Frequency", list(f_map.keys()))
                    f_agg = st.selectbox("Aggregation", ["sum", "mean", "max", "min"])
                    if st.button("Execute Resampling"):
                        res_df = resample_temporal_data(df, ts_dt, ts_val, frequency=f_map[f_choice], aggregation=f_agg)
                        fig_ts = create_relationship_chart(res_df, "date", f"{ts_val}_{f_agg}", chart_type="line")
                        st.plotly_chart(fig_ts, use_container_width=True)
                        register_figure(fig_ts)

                with col_roll:
                    st.markdown("**Moving Averages & Decomposition**")
                    w_size = st.slider("Moving Window", 2, 60, 7)
                    if st.button("Rolling Metrics"):
                        r_df = compute_rolling_metrics(df, ts_val, date_column=ts_dt, window_size=w_size)
                        fig_roll = create_relationship_chart(r_df, ts_dt, f"{ts_val}_rolling_mean_{w_size}", chart_type="line")
                        st.plotly_chart(fig_roll, use_container_width=True)

        with m_tab2:
            st.markdown("##### Curve Fitting & Trendline Equations")
            if len(num_cols) >= 2:
                cf_x = st.selectbox("Predictor (X)", num_cols, index=0)
                cf_y = st.selectbox("Response (Y)", num_cols, index=1)
                cf_mode = st.selectbox("Model", ["linear", "quadratic", "exponential", "logarithmic", "power"])

                if st.button("Fit Trendline"):
                    res_cf = fit_curve_and_equation(df, cf_x, cf_y, curve_type=cf_mode)
                    st.success(f"**Discovered Equation**: `{res_cf['formula']}` | R² Goodness: **{res_cf['r_squared']}**")
                    fig_cf = create_relationship_chart(df, cf_x, cf_y, chart_type="scatter")
                    st.plotly_chart(fig_cf, use_container_width=True)
                    register_figure(fig_cf)

        with m_tab3:
            st.markdown("##### Parametric Function-Family Search")
            st.caption("Evolves mathematical laws across functional families using bounded genetic optimization.")
            if len(num_cols) >= 2:
                sr_x = st.selectbox("Symbolic X", num_cols, index=0)
                sr_y = st.selectbox("Symbolic Y", num_cols, index=1)
                sr_gen = st.slider("Generations", 5, 30, 15)
                sr_pop = st.slider("Population", 20, 60, 40)

                if st.button("Discover Equation"):
                    with st.spinner("Executing function-family search..."):
                        sr_res = run_symbolic_regression(df, sr_x, sr_y, generations=sr_gen, population_size=sr_pop)
                        st.success(f"Discovered Equation: **{sr_res['equation']}** (Family: {sr_res['basis_family']})")
                        st.write(f"R²: **{sr_res['r_squared']}** | MSE: **{sr_res['mse']}**")

        with m_tab4:
            st.markdown("##### K-Means Clustering & PCA with Loadings")
            if len(num_cols) >= 2:
                col_km, col_pca = st.columns(2)
                with col_km:
                    st.markdown("**K-Means Clustering**")
                    km_feats = st.multiselect("Clustering Features", num_cols, default=num_cols[:min(3, len(num_cols))])
                    km_k = st.slider("Clusters (K)", 2, 8, 3)
                    if km_feats and st.button("Run K-Means"):
                        km_df, km_meta = run_kmeans_clustering(df, km_feats, n_clusters=km_k)
                        st.write(f"Inertia: {km_meta['inertia']}")
                        st.markdown("**Cluster Centers (Original Units)**")
                        st.dataframe(km_meta["centers_dataframe"], use_container_width=True)
                        fig_km = create_relationship_chart(km_df, km_feats[0], km_feats[1], chart_type="scatter", color_col="Cluster")
                        st.plotly_chart(fig_km, use_container_width=True)
                        register_figure(fig_km)

                with col_pca:
                    st.markdown("**PCA Dimensionality Reduction**")
                    pca_feats = st.multiselect("PCA Features", num_cols, default=num_cols[:min(4, len(num_cols))])
                    pca_c = st.selectbox("Components", [2, 3])
                    if len(pca_feats) >= pca_c and st.button("Run PCA"):
                        pca_df, pca_meta = run_pca_reduction(df, pca_feats, n_components=pca_c)
                        st.write(f"Explained Variance: {pca_meta['total_explained_variance']}%")
                        st.markdown("**Feature Loadings**")
                        st.dataframe(pca_meta["loadings"], use_container_width=True)
                        fig_pca = create_relationship_chart(pca_df, "PC1", "PC2", chart_type="scatter")
                        st.plotly_chart(fig_pca, use_container_width=True)

        with m_tab5:
            st.markdown("##### Multivariate Anomaly Scoring (Isolation Forest)")
            if len(num_cols) >= 2:
                iso_feats = st.multiselect("Anomaly Features", num_cols, default=num_cols[:min(3, len(num_cols))])
                iso_c = st.slider("Contamination", 0.01, 0.20, 0.05, step=0.01)
                if iso_feats and st.button("Score Anomalies"):
                    iso_df, iso_meta = run_isolation_forest_anomaly_detection(df, iso_feats, contamination=iso_c)
                    st.write(f"Detected {iso_meta['anomalies_detected']} anomalies ({iso_meta['anomaly_percentage']}%)")
                    fig_iso = create_relationship_chart(iso_df, iso_feats[0], iso_feats[1], chart_type="scatter", color_col="Anomaly_Flag")
                    st.plotly_chart(fig_iso, use_container_width=True)
                    register_figure(fig_iso)


# ==============================================================================
# WORKSPACE 6: Visualization Studio & Executive Reports (Features 42-52)
# ==============================================================================
elif active_workspace == "6. Visualization Studio & Reports":
    st.subheader("Dynamic Visualization Studio & Executive Reports")
    st.caption("Interactive Plotly chart builders, dataset downloads, standalone HTML reports, and dynamic printable PDFs.")

    if st.session_state.df is None:
        st.info("Please load a dataset in Workspace 1 first.")
    else:
        df = st.session_state.df
        logger = st.session_state.audit_logger

        v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs([
            "Visualization Studio", "Custom Chart Builder", "Automated Executive Summary", "Report & Dataset Export"
        ])

        with v_tab1:
            st.markdown("##### Auto-Plotting Recommendation")
            fig_auto = create_auto_plot(df)
            st.plotly_chart(fig_auto, use_container_width=True)
            register_figure(fig_auto)

        with v_tab2:
            st.markdown("##### Custom Multi-Axis Chart Builder")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            c_type = st.selectbox("Primitive", ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Density Heatmap"])
            c_x = st.selectbox("X Axis", df.columns)
            c_y = st.selectbox("Y Axis (Optional)", ["None"] + list(df.columns))
            c_col = st.selectbox("Color Grouping", ["None"] + list(df.columns))

            if st.button("Render Chart"):
                built_f = build_custom_chart(df, chart_type=c_type, x_col=c_x, y_col=None if c_y == "None" else c_y, color_col=None if c_col == "None" else c_col)
                st.plotly_chart(built_f, use_container_width=True)
                register_figure(built_f)

        with v_tab3:
            st.markdown("##### Automated Executive Summary")
            st.caption("Deterministic analytical narrative generated from dataset distributions and transformation records.")
            exec_text = generate_executive_summary(df, audit_logs=logger.get_logs())
            st.markdown(exec_text)

        with v_tab4:
            st.markdown("##### Comprehensive Analytical Report & Dataset Download")
            exec_text = generate_executive_summary(df, audit_logs=logger.get_logs())

            rep_c1, rep_c2, rep_c3 = st.columns(3)
            with rep_c1:
                st.markdown("**Standalone HTML Report**")
                html_out = generate_html_report(df, summary_text=exec_text, audit_logs=logger.get_logs(), figures=st.session_state.recent_figures)
                st.download_button("Download HTML Report", data=html_out, file_name="datasight_report.html", mime="text/html")

            with rep_c2:
                st.markdown("**Executive PDF Report**")
                pdf_out = generate_pdf_report(df, summary_text=exec_text, audit_logs=logger.get_logs())
                st.download_button("Download PDF Report", data=pdf_out, file_name="datasight_report.pdf", mime="application/pdf")

            with rep_c3:
                st.markdown("**Cleaned Dataset Export**")
                f_fmt = st.selectbox("Format", ["CSV", "Excel", "Parquet"])
                d_bytes, mime_t, f_ext = export_dataset_bytes(df, file_format=f_fmt.lower())
                st.download_button(f"Download .{f_ext}", data=d_bytes, file_name=f"datasight_clean.{f_ext}", mime=mime_t)
