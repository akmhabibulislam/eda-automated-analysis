"""
Automated Data Analysis Web Application.
Main frontend controller built with Streamlit.
Integrates all 52 end-to-end data analysis features across:
- Data Ingestion & Memory Management
- Automated Data Cleaning & Lineage
- Exploratory Data Analysis & Statistics
- Advanced Data Wrangling & Transformations
- Statistical Testing & Hypothesis Validation
- Time-Series Analysis
- Mathematical Modeling & Symbolic Regression
- Unsupervised Clustering & Segmentation
- Dynamic Visualizations
- Exporting & Reporting
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

# Add project root to sys.path for robust module imports
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
    filter_rows
)
from backend.analysis.statistics import (
    compute_univariate_summary,
    compute_categorical_distribution,
    compute_missingness_matrix,
    compute_correlation_matrix,
    detect_multicollinearity,
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
    st.session_state.dataset_name = "Untitled Dataset"
if "recent_figures" not in st.session_state:
    st.session_state.recent_figures = []


def register_figure(fig):
    """Keep up to 3 recent figures for report export inclusion."""
    st.session_state.recent_figures.append(fig)
    if len(st.session_state.recent_figures) > 3:
        st.session_state.recent_figures.pop(0)


# ==============================================================================
# SIDEBAR: Ingestion & Navigation
# ==============================================================================

with st.sidebar:
    st.title("DataSight Analytics")
    st.caption("Enterprise Exploratory Data Analysis & Modeling Platform")
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "Data Ingestion & Overview",
            "Auto & Custom Cleaning",
            "Exploratory Statistics",
            "Data Wrangling & Reshaping",
            "Hypothesis & Statistical Tests",
            "Time-Series Analysis",
            "Mathematical Modeling & Clusters",
            "Dynamic Visualizations",
            "Executive Reports & Export"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("System Memory & Health")
    sys_mem = get_system_memory()
    st.write(f"RAM Used: {sys_mem['used_mb']:,.0f} MB / {sys_mem['total_mb']:,.0f} MB ({sys_mem['percent']}%)")
    
    if st.button("Trigger Garbage Collection (gc.collect)"):
        free_memory()
        st.success("Freed unreferenced memory cache.")


# ==============================================================================
# PAGE 1: Data Ingestion & Setup (Features 1-5 & Memory Optimization)
# ==============================================================================
if nav_selection == "Data Ingestion & Overview":
    st.header("Data Ingestion & Dataset Overview")
    st.caption("Upload multi-format files or connect to databases with automatic memory optimization.")

    source_tab, db_tab, demo_tab = st.tabs(["File Upload", "Database Connector", "Sample Datasets"])

    with source_tab:
        uploaded_file = st.file_uploader(
            "Upload file (CSV, Excel, JSON, Parquet, TSV)",
            type=["csv", "tsv", "tab", "xlsx", "xls", "json", "parquet"]
        )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            auto_downcast = st.checkbox("Automatic Type Downcasting & Category Optimization", value=True)
        with col_opt2:
            use_sampling = st.checkbox("Limit Rows (Sampling Mode for Large Files)", value=False)
            sample_rows = st.number_input("Max Rows to Load", min_value=100, max_value=2000000, value=50000) if use_sampling else None

        if uploaded_file is not None and st.button("Load and Process Dataset"):
            with st.spinner("Ingesting and optimizing memory..."):
                try:
                    df, meta = load_dataset(
                        uploaded_file,
                        uploaded_file.name,
                        auto_optimize=auto_downcast,
                        sample_rows=sample_rows
                    )
                    st.session_state.df = df
                    st.session_state.original_df = df.copy()
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.audit_logger.clear()
                    st.session_state.audit_logger.log(
                        action="File Ingestion",
                        details=f"Loaded {uploaded_file.name} (Format: {meta['format']})",
                        rows_affected=len(df),
                        columns_affected=list(df.columns)
                    )
                    st.success(f"Successfully loaded {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")

    with db_tab:
        st.subheader("SQL Database Connection")
        st.caption("Supports SQLite, PostgreSQL, MySQL via SQLAlchemy connection strings.")
        db_url = st.text_input("Connection String", value="sqlite:///example.db", placeholder="postgresql://user:password@localhost:5432/dbname")
        db_query = st.text_area("SQL Query", value="SELECT * FROM my_table LIMIT 1000;")

        if st.button("Execute Query & Ingest"):
            with st.spinner("Executing database query..."):
                try:
                    df, meta = load_from_database(db_url, db_query, auto_optimize=auto_downcast)
                    st.session_state.df = df
                    st.session_state.original_df = df.copy()
                    st.session_state.dataset_name = "Database Query Result"
                    st.session_state.audit_logger.clear()
                    st.session_state.audit_logger.log(
                        action="Database Query Ingestion",
                        details=f"Executed: {db_query[:50]}...",
                        rows_affected=len(df),
                        columns_affected=list(df.columns)
                    )
                    st.success(f"Retrieved {len(df):,} records from database.")
                except Exception as e:
                    st.error(f"Database connection error: {str(e)}")

    with demo_tab:
        st.subheader("Load Built-in Demo Datasets")
        demo_choice = st.selectbox("Choose Sample Dataset", ["Sales & Financial Performance", "Customer Demographic & Churn", "Sensor Time-Series Telemetry"])
        if st.button("Load Selected Sample"):
            np.random.seed(42)
            n = 1500
            if demo_choice == "Sales & Financial Performance":
                dates = pd.date_range("2024-01-01", periods=n, freq="D")
                regions = ["North", "South", "East", "West"]
                categories = ["Hardware", "Software", "Cloud", "Consulting"]
                df_demo = pd.DataFrame({
                    "transaction_id": [f"TX-{10000+i}" for i in range(n)],
                    "date": dates,
                    "region": np.random.choice(regions, n),
                    "category": np.random.choice(categories, n),
                    "units_sold": np.random.randint(1, 100, n),
                    "unit_price": np.random.uniform(50.0, 500.0, n).round(2),
                    "discount_rate": np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], n),
                    "customer_satisfaction": np.random.uniform(1.0, 5.0, n).round(1)
                })
                # Add calculated revenue and introduce occasional missing values
                df_demo["revenue"] = (df_demo["units_sold"] * df_demo["unit_price"] * (1 - df_demo["discount_rate"])).round(2)
                df_demo.loc[np.random.choice(n, 30, replace=False), "customer_satisfaction"] = np.nan
            elif demo_choice == "Customer Demographic & Churn":
                genders = ["Male", "Female", "Non-Binary"]
                plans = ["Basic", "Standard", "Premium"]
                df_demo = pd.DataFrame({
                    "customer_id": [f"CUST-{20000+i}" for i in range(n)],
                    "gender": np.random.choice(genders, n),
                    "subscription_tier": np.random.choice(plans, n),
                    "age": np.random.randint(18, 70, n),
                    "tenure_months": np.random.randint(1, 72, n),
                    "monthly_charges": np.random.uniform(20.0, 150.0, n).round(2),
                    "total_charges": np.random.uniform(100.0, 8000.0, n).round(2),
                    "support_tickets": np.random.poisson(lam=2, size=n),
                    "churn": np.random.choice(["No", "Yes"], n, p=[0.75, 0.25])
                })
                df_demo.loc[np.random.choice(n, 25, replace=False), "total_charges"] = np.nan
            else:
                timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
                t = np.linspace(0, 50, n)
                df_demo = pd.DataFrame({
                    "timestamp": timestamps,
                    "sensor_id": np.random.choice(["S1", "S2", "S3"], n),
                    "temperature": 25.0 + 5.0 * np.sin(t) + np.random.normal(0, 0.8, n),
                    "pressure": 101.3 + 2.0 * np.cos(t / 2) + np.random.normal(0, 0.3, n),
                    "vibration": np.random.exponential(scale=1.5, size=n),
                    "voltage": 220.0 + np.random.normal(0, 1.2, n)
                })

            df_demo, opt_metrics = optimize_dataframe_memory(df_demo)
            st.session_state.df = df_demo
            st.session_state.original_df = df_demo.copy()
            st.session_state.dataset_name = demo_choice
            st.session_state.audit_logger.clear()
            st.session_state.audit_logger.log("Load Demo Dataset", f"Loaded {demo_choice}", len(df_demo), list(df_demo.columns))
            st.success(f"Loaded '{demo_choice}' successfully.")

    # Overview Dashboard & Preview Table
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.subheader(f"Dataset Overview: {st.session_state.dataset_name}")

        ov = get_dataset_overview(df)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Rows", f"{ov['total_rows']:,}")
        m2.metric("Total Columns", f"{ov['total_columns']:,}")
        m3.metric("Memory Footprint", ov["memory_readable"])
        m4.metric("Duplicates", f"{ov['duplicate_rows']:,} ({ov['duplicate_percentage']}%)")
        m5.metric("Missing Cells", f"{ov['total_missing_cells']:,} ({ov['missing_percentage']}%)")

        # Column Schema Detection breakdown
        schema = detect_schema(df)
        st.markdown("**Detected Column Types**")
        st.write(
            f"Numeric ({ov['numeric_count']}): `{', '.join(schema['numeric_columns']) if schema['numeric_columns'] else 'None'}` | "
            f"Categorical ({ov['categorical_count']}): `{', '.join(schema['categorical_columns']) if schema['categorical_columns'] else 'None'}` | "
            f"Datetime ({ov['datetime_count']}): `{', '.join(schema['datetime_columns']) if schema['datetime_columns'] else 'None'}` | "
            f"Boolean ({ov['boolean_count']}): `{', '.join(schema['boolean_columns']) if schema['boolean_columns'] else 'None'}`"
        )

        st.markdown("---")
        st.subheader("Data Preview Table")
        
        # Interactive filter/search control
        filter_col = st.selectbox("Search Filter Column", ["All Columns"] + list(df.columns))
        search_query = st.text_input("Filter rows by substring search", "")

        preview_df = df
        if search_query:
            if filter_col == "All Columns":
                mask = df.astype(str).apply(lambda row: row.str.contains(search_query, case=False, na=False)).any(axis=1)
            else:
                mask = df[filter_col].astype(str).str.contains(search_query, case=False, na=False)
            preview_df = df[mask]

        st.dataframe(preview_df.head(100), use_container_width=True)
        st.caption(f"Displaying top {min(100, len(preview_df))} of {len(preview_df):,} matching rows.")


# ==============================================================================
# PAGE 2: Automated Data Cleaning & Lineage (Features 6-13)
# ==============================================================================
elif nav_selection == "Auto & Custom Cleaning":
    st.header("Automated & Custom Data Cleaning")
    st.caption("Apply automated one-click pipelines or configure granular imputation, outlier clipping, and type conversions.")

    if st.session_state.df is None:
        st.info("Please load a dataset in 'Data Ingestion & Overview' first.")
    else:
        df = st.session_state.df
        logger = st.session_state.audit_logger

        # One-Click Automated Cleaning Banner
        st.subheader("Automated Auto-Clean Pipeline")
        st.write("Executes column standardization, duplicate removal, text trimming, median/mode imputation, and safe outlier capping.")
        if st.button("Run One-Click Automated Cleaning", type="primary"):
            with st.spinner("Executing auto-clean pipeline..."):
                mem_before = get_memory_usage(df)["mb"]
                cleaned_df, summary = run_automated_cleaning(df, logger=logger)
                cleaned_df, _ = optimize_dataframe_memory(cleaned_df)
                mem_after = get_memory_usage(cleaned_df)["mb"]

                st.session_state.df = cleaned_df
                st.success(
                    f"Auto-Clean complete. {summary['steps_executed']} operations executed. "
                    f"Memory changed from {mem_before:.2f} MB to {mem_after:.2f} MB."
                )
                st.rerun()

        st.markdown("---")
        st.subheader("Granular Cleaning Controls")

        c_tab1, c_tab2, c_tab3, c_tab4, c_tab5, c_tab6 = st.tabs([
            "Missing Values", "Duplicates", "Outliers", "Headers & Strings", "Type Casting", "Audit Lineage"
        ])

        with c_tab1:
            st.markdown("##### Imputation & Removal")
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown("**Impute Missing Values**")
                imp_cols = st.multiselect("Select Columns to Impute", df.columns, default=[c for c in df.columns if df[c].isna().sum() > 0])
                imp_strategy = st.selectbox("Strategy", ["mean", "median", "mode", "constant", "ffill", "bfill"])
                imp_const = st.text_input("Constant Value (if constant selected)", "0") if imp_strategy == "constant" else None

                if st.button("Apply Imputation"):
                    const_val = float(imp_const) if imp_const and imp_const.replace(".", "").isdigit() else imp_const
                    st.session_state.df = impute_missing_values(df, columns=imp_cols, strategy=imp_strategy, fill_value=const_val, logger=logger)
                    st.success("Imputation applied successfully.")
                    st.rerun()

            with sub_col2:
                st.markdown("**Drop Missing Entries**")
                drop_mode = st.radio("Removal Scope", ["Drop by threshold %", "Drop any rows with nulls in selected columns"])
                if drop_mode == "Drop by threshold %":
                    col_thresh = st.slider("Drop column if missing > X%", 10, 100, 50)
                    row_thresh = st.slider("Drop row if missing > X%", 10, 100, 50)
                    if st.button("Drop by Threshold"):
                        st.session_state.df = drop_missing_values(df, row_threshold_pct=float(row_thresh), col_threshold_pct=float(col_thresh), logger=logger)
                        st.success("Threshold-based missing drop executed.")
                        st.rerun()
                else:
                    target_null_cols = st.multiselect("Columns requiring non-nulls", df.columns)
                    if st.button("Drop Rows with Nulls"):
                        st.session_state.df = drop_missing_values(df, how="any", columns=target_null_cols, logger=logger)
                        st.success("Targeted row drop executed.")
                        st.rerun()

        with c_tab2:
            st.markdown("##### Duplicate Removal")
            dup_cols = st.multiselect("Subset columns for duplicate evaluation (leave empty for entire row)", df.columns)
            dup_keep = st.selectbox("Keep occurrence", ["first", "last", False], format_func=lambda x: "Drop all duplicates" if x is False else f"Keep {x}")
            if st.button("Purge Duplicates"):
                st.session_state.df = remove_duplicates(df, subset=dup_cols if dup_cols else None, keep=dup_keep, logger=logger)
                st.success("Duplicate removal complete.")
                st.rerun()

        with c_tab3:
            st.markdown("##### Outlier Detection & Treatment")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.info("No numeric columns available for outlier detection.")
            else:
                out_col = st.selectbox("Target Numeric Column", num_cols)
                out_method = st.selectbox("Detection Algorithm", ["iqr", "zscore"])
                out_thresh = st.number_input("Threshold multiplier", min_value=1.0, max_value=5.0, value=1.5 if out_method == "iqr" else 3.0, step=0.1)
                out_action = st.selectbox("Treatment Action", ["cap", "drop", "flag"])

                if st.button("Execute Outlier Treatment"):
                    treated_df, info = treat_outliers(df, column=out_col, method=out_method, threshold=float(out_thresh), action=out_action, logger=logger)
                    st.session_state.df = treated_df
                    st.success(f"Detected {info['outliers_detected']} anomalies outside bounds [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]. Action '{out_action}' applied.")
                    st.rerun()

        with c_tab4:
            st.markdown("##### Header Standardization & Text Sanitization")
            h_col, s_col = st.columns(2)
            with h_col:
                st.markdown("**Standardize Column Headers**")
                case_style = st.selectbox("Case Standard", ["snake_case", "lower_case", "upper_case", "camelCase"])
                if st.button("Standardize Headers"):
                    st.session_state.df = standardize_column_headers(df, case_style=case_style, logger=logger)
                    st.success("Headers formatted.")
                    st.rerun()

            with s_col:
                st.markdown("**Text & String Cleaning**")
                str_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                target_str_cols = st.multiselect("Select Text Columns", str_cols)
                strip_ws = st.checkbox("Trim Leading/Trailing Whitespace", value=True)
                case_trans = st.selectbox("Transform Case", ["None", "lower", "upper", "title"])
                strip_special = st.checkbox("Strip Special Characters", value=False)

                if st.button("Clean Text Columns"):
                    st.session_state.df = clean_text_columns(
                        df,
                        columns=target_str_cols,
                        strip_whitespace=strip_ws,
                        case_transformation=None if case_trans == "None" else case_trans,
                        remove_special_chars=strip_special,
                        logger=logger
                    )
                    st.success("Text cleaning applied.")
                    st.rerun()

        with c_tab5:
            st.markdown("##### Data Type Casting")
            cast_col = st.selectbox("Column to Cast", df.columns)
            target_dtype = st.selectbox("Target Type", ["int", "float", "string", "datetime", "boolean", "category"])
            if st.button("Apply Cast"):
                try:
                    st.session_state.df = cast_data_types(df, {cast_col: target_dtype}, logger=logger)
                    st.success(f"Successfully converted '{cast_col}' to {target_dtype}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Casting error: {str(e)}")

        with c_tab6:
            st.markdown("##### Chronological Audit Trail & Data Lineage")
            audit_df = logger.to_dataframe()
            if audit_df.empty:
                st.info("No transformation events logged in this session yet.")
            else:
                st.dataframe(audit_df, use_container_width=True)
                if st.button("Reset Dataset to Original"):
                    st.session_state.df = st.session_state.original_df.copy()
                    st.session_state.audit_logger.clear()
                    st.session_state.audit_logger.log("Dataset Reset", "Restored initial uploaded state")
                    st.success("Dataset reverted to original state.")
                    st.rerun()


# ==============================================================================
# PAGE 3: Exploratory Data Analysis & Statistics (Features 14-19)
# ==============================================================================
elif nav_selection == "Exploratory Statistics":
    st.header("Exploratory Data Analysis & Statistics")
    st.caption("Deep statistical breakdowns, missingness matrices, correlation coefficients, multicollinearity alerts, and skewness/kurtosis.")

    if st.session_state.df is None:
        st.info("Please load a dataset in 'Data Ingestion & Overview' first.")
    else:
        df = st.session_state.df

        eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
            "Univariate Summary", "Categorical Frequencies", "Missingness Matrix", "Correlations & Multicollinearity", "Skewness & Kurtosis"
        ])

        with eda_tab1:
            st.subheader("Univariate Summary Statistics")
            summary_stats = compute_univariate_summary(df)
            if summary_stats.empty:
                st.info("No numeric features found in dataset.")
            else:
                st.dataframe(summary_stats, use_container_width=True)

        with eda_tab2:
            st.subheader("Categorical Frequency Distributions")
            cat_dists = compute_categorical_distribution(df)
            if not cat_dists:
                st.info("No categorical features found.")
            else:
                selected_cat = st.selectbox("Select Categorical Column", list(cat_dists.keys()))
                dist_df = cat_dists[selected_cat]
                c_tbl, c_chart = st.columns([1, 1])
                with c_tbl:
                    st.dataframe(dist_df, use_container_width=True)
                with c_chart:
                    fig = create_categorical_chart(df, selected_cat, chart_type="bar")
                    st.plotly_chart(fig, use_container_width=True)
                    register_figure(fig)

        with eda_tab3:
            st.subheader("Missing Data Heatmap & Profile")
            miss_data = compute_missingness_matrix(df)
            st.write(f"Total Missing Cells: **{miss_data['total_missing_cells']:,}** across **{len(miss_data['columns_with_missing'])}** columns.")
            
            m_col1, m_col2 = st.columns([1, 2])
            with m_col1:
                st.dataframe(miss_data["summary"], use_container_width=True)
            with m_col2:
                if miss_data["total_missing_cells"] > 0:
                    matrix_fig = create_relationship_chart(
                        miss_data["summary"],
                        x_col="Column",
                        y_col="Missing_Percentage",
                        chart_type="line"
                    )
                    st.plotly_chart(matrix_fig, use_container_width=True)
                else:
                    st.success("Zero missing values across all columns.")

        with eda_tab4:
            st.subheader("Correlation Analysis & Multicollinearity Warnings")
            corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
            corr_df = compute_correlation_matrix(df, method=corr_method)

            if corr_df.empty:
                st.info("Insufficient numeric columns for correlation computation.")
            else:
                st.dataframe(corr_df, use_container_width=True)

                # Heatmap visualization
                corr_fig = build_custom_chart(df, chart_type="Scatter", x_col=corr_df.columns[0], y_col=corr_df.columns[1] if len(corr_df.columns) > 1 else corr_df.columns[0])
                # Show correlation warnings
                st.markdown("##### Automated Multi-Collinearity Warnings (Threshold > 0.80)")
                collin_warnings = detect_multicollinearity(df, threshold=0.80)
                if collin_warnings:
                    for w in collin_warnings:
                        st.warning(f"**{w['severity']} Collinearity Detected**: `{w['column_1']}` and `{w['column_2']}` have correlation coefficient **{w['correlation']}**.")
                else:
                    st.success("No critical multicollinearity detected among predictor variables.")

        with eda_tab5:
            st.subheader("Skewness & Kurtosis Tail Analysis")
            skew_kurt_df = compute_skewness_kurtosis(df)
            if skew_kurt_df.empty:
                st.info("Insufficient numeric data for shape analysis.")
            else:
                st.dataframe(skew_kurt_df, use_container_width=True)
                st.caption("Skewness measures distributional symmetry; Kurtosis evaluates tail heaviness and extreme value propensity.")


# ==============================================================================
# PAGE 4: Advanced Data Wrangling & Transformations (Features 20-26)
# ==============================================================================
elif nav_selection == "Data Wrangling & Reshaping":
    st.header("Advanced Data Wrangling & Transformations")
    st.caption("Construct calculated columns, continuous binning, aggregations, merges, regex extractions, and reshaping.")

    if st.session_state.df is None:
        st.info("Please load a dataset in 'Data Ingestion & Overview' first.")
    else:
        df = st.session_state.df
        logger = st.session_state.audit_logger

        w_tab1, w_tab2, w_tab3, w_tab4, w_tab5, w_tab6 = st.tabs([
            "Formula Builder", "Binning", "SQL Groupby", "Regex Extraction", "Pivot / Unpivot", "Row Querying"
        ])

        with w_tab1:
            st.subheader("Custom Formula / Calculated Column Builder")
            st.caption("Write vectorized mathematical expressions using existing columns (e.g., `units_sold * unit_price` or `(age - 18) / 10`).")
            new_col_name = st.text_input("New Column Name", "calculated_metric")
            formula_expr = st.text_input("Mathematical Formula", "")
            st.write(f"Available columns: `{', '.join(df.columns)}`")

            if st.button("Evaluate & Add Column"):
                try:
                    st.session_state.df = add_custom_formula_column(df, new_col_name, formula_expr, logger=logger)
                    st.success(f"Added new column '{new_col_name}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Formula evaluation error: {str(e)}")

        with w_tab2:
            st.subheader("Continuous Binning & Discretization")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.info("No numeric columns available for binning.")
            else:
                bin_col = st.selectbox("Select Continuous Column", num_cols)
                b_name = st.text_input("Binned Column Name", f"{bin_col}_binned")
                b_type = st.selectbox("Binning Strategy", ["equal_width", "quantile"])
                num_bins = st.slider("Number of Bins", 2, 20, 5)

                if st.button("Apply Binning"):
                    st.session_state.df = bin_continuous_column(df, column=bin_col, new_column_name=b_name, bins=num_bins, bin_type=b_type, logger=logger)
                    st.success(f"Binned '{bin_col}' into '{b_name}'.")
                    st.rerun()

        with w_tab3:
            st.subheader("SQL-like Groupby & Aggregations")
            grp_cols = st.multiselect("Group By Columns", df.columns)
            agg_col = st.selectbox("Metric Column to Aggregate", df.select_dtypes(include=[np.number]).columns.tolist())
            agg_funcs = st.multiselect("Aggregate Functions", ["mean", "sum", "count", "std", "min", "max", "median"], default=["mean", "sum"])

            if grp_cols and agg_col and agg_funcs and st.button("Run Groupby Aggregation"):
                agg_dict = {agg_col: agg_funcs}
                result_agg = aggregate_groupby(df, group_columns=grp_cols, aggregations=agg_dict)
                st.dataframe(result_agg, use_container_width=True)

        with w_tab4:
            st.subheader("Text Regex Pattern Extraction")
            text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if not text_cols:
                st.info("No text columns found.")
            else:
                regex_src = st.selectbox("Source Column", text_cols)
                regex_pattern = st.text_input("Regular Expression (e.g., `([A-Za-z0-9]+@[a-zA-Z0-9.-]+)` or `(\\d{3}-\\d{3})`)", "([A-Za-z0-9]+)")
                regex_dest = st.text_input("Destination Column Name", f"{regex_src}_extracted")

                if st.button("Extract Regex Pattern"):
                    try:
                        st.session_state.df = extract_regex_patterns(df, source_column=regex_src, pattern=regex_pattern, new_column_name=regex_dest, logger=logger)
                        st.success(f"Pattern extracted into column '{regex_dest}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Regex error: {str(e)}")

        with w_tab5:
            st.subheader("Data Reshaping: Pivot & Unpivot")
            res_mode = st.radio("Reshaping Mode", ["Pivot (Long to Wide)", "Unpivot / Melt (Wide to Long)"])

            if res_mode == "Pivot (Long to Wide)":
                p_idx = st.multiselect("Index Column(s)", df.columns)
                p_col = st.selectbox("Column for New Headers", df.columns)
                p_val = st.selectbox("Values Column", df.select_dtypes(include=[np.number]).columns.tolist())
                p_agg = st.selectbox("Aggregation Function", ["mean", "sum", "count", "min", "max"])

                if p_idx and p_col and p_val and st.button("Execute Pivot"):
                    pivoted_df = pivot_dataframe(df, index_cols=p_idx, columns=p_col, values=p_val, aggfunc=p_agg)
                    st.dataframe(pivoted_df, use_container_width=True)

            else:
                u_ids = st.multiselect("Identifier Variables (Keep fixed)", df.columns)
                u_vals = st.multiselect("Value Variables to Unpivot", [c for c in df.columns if c not in u_ids])

                if u_ids and u_vals and st.button("Execute Unpivot (Melt)"):
                    unpivoted_df = unpivot_dataframe(df, id_vars=u_ids, value_vars=u_vals)
                    st.dataframe(unpivoted_df, use_container_width=True)

        with w_tab6:
            st.subheader("Custom Row Filtering & Querying")
            st.caption("Query rows using pandas expression syntax (e.g., `age > 25 and units_sold > 10`).")
            query_str = st.text_input("Query Expression", "")

            if query_str and st.button("Apply Row Filter"):
                try:
                    st.session_state.df = filter_rows(df, query_str, logger=logger)
                    st.success("Query applied successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Query syntax error: {str(e)}")


# ==============================================================================
# PAGE 5: Statistical Testing & Hypothesis Validation (Features 27-31)
# ==============================================================================
elif nav_selection == "Hypothesis & Statistical Tests":
    st.header("Statistical Testing & Hypothesis Validation")
    st.caption("Parametric and non-parametric hypothesis tests, ANOVA, Chi-Square, and continuous distribution evaluation.")

    if st.session_state.df is None:
        st.info("Please load a dataset in 'Data Ingestion & Overview' first.")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        t_tab1, t_tab2, t_tab3, t_tab4, t_tab5 = st.tabs([
            "T-Tests", "ANOVA", "Chi-Square", "Non-Parametric", "Distribution Fitting"
        ])

        with t_tab1:
            st.subheader("Independent and Paired T-Tests")
            if len(num_cols) < 1:
                st.info("Requires numeric variables.")
            else:
                tt_mode = st.radio("T-Test Design", ["Two Numeric Columns", "Split Column by Binary Category"])
                if tt_mode == "Two Numeric Columns" and len(num_cols) >= 2:
                    c1 = st.selectbox("Sample 1", num_cols, index=0)
                    c2 = st.selectbox("Sample 2", num_cols, index=1)
                    t_type = st.selectbox("Type", ["independent", "paired"])
                    if st.button("Run Column T-Test"):
                        res = run_t_test(df[c1].values, df[c2].values, test_type=t_type)
                        st.json(res)
                elif tt_mode == "Split Column by Binary Category" and cat_cols:
                    num_target = st.selectbox("Metric Column", num_cols)
                    bin_cat = st.selectbox("Grouping Variable", cat_cols)
                    unique_groups = df[bin_cat].dropna().unique()
                    if len(unique_groups) >= 2:
                        g1_name = unique_groups[0]
                        g2_name = unique_groups[1]
                        s1 = df[df[bin_cat] == g1_name][num_target].values
                        s2 = df[df[bin_cat] == g2_name][num_target].values
                        if st.button("Run Group T-Test"):
                            res = run_t_test(s1, s2, test_type="independent")
                            st.json(res)
                    else:
                        st.warning("Selected category requires at least 2 distinct values.")

        with t_tab2:
            st.subheader("One-Way ANOVA (Analysis of Variance)")
            if num_cols and cat_cols:
                anova_val = st.selectbox("Measurement Metric", num_cols)
                anova_grp = st.selectbox("Grouping Factor", cat_cols)
                if st.button("Run ANOVA"):
                    groups = [group[anova_val].values for _, group in df.groupby(anova_grp)]
                    names = [str(k) for k, _ in df.groupby(anova_grp)]
                    res = run_anova(groups, group_names=names)
                    st.json(res)
            else:
                st.info("Requires at least one numeric and one categorical variable.")

        with t_tab3:
            st.subheader("Chi-Square Test of Independence")
            if len(cat_cols) >= 2:
                cat1 = st.selectbox("Categorical Variable 1", cat_cols, index=0)
                cat2 = st.selectbox("Categorical Variable 2", cat_cols, index=1)
                if st.button("Execute Chi-Square Test"):
                    ct = pd.crosstab(df[cat1], df[cat2])
                    st.write("**Observed Contingency Table**")
                    st.dataframe(ct)
                    res = run_chi_square(ct)
                    st.json({k: v for k, v in res.items() if k != "expected_frequencies"})
                    st.write("**Expected Frequencies**")
                    st.dataframe(res["expected_frequencies"])
            else:
                st.info("Requires at least two categorical columns.")

        with t_tab4:
            st.subheader("Non-Parametric Tests (Mann-Whitney U / Kruskal-Wallis)")
            if num_cols and cat_cols:
                np_metric = st.selectbox("Non-Parametric Metric", num_cols)
                np_grp = st.selectbox("Non-Parametric Factor", cat_cols)
                if st.button("Run Non-Parametric Test"):
                    groups = [group[np_metric].values for _, group in df.groupby(np_grp)]
                    res = run_non_parametric_tests(groups)
                    st.json(res)

        with t_tab5:
            st.subheader("Continuous Distribution Fitting")
            if num_cols:
                dist_target = st.selectbox("Target Column for Fitting", num_cols)
                if st.button("Evaluate Distribution Fits"):
                    fit_table = fit_distributions(df[dist_target].values)
                    st.dataframe(fit_table, use_container_width=True)
                    st.caption("Lower Kolmogorov-Smirnov (KS) statistic indicates a closer statistical fit.")


# ==============================================================================
# PAGE 6: Time-Series Analysis (Features 32-36)
# ==============================================================================
elif nav_selection == "Time-Series Analysis":
    st.header("Time-Series Analysis & Decomposition")
    st.caption("Resampling, rolling moving averages, period growth, cumulative sums, and seasonal-trend decomposition.")

    if st.session_state.df is None:
        st.info("Please load a dataset in 'Data Ingestion & Overview' first.")
    else:
        df = st.session_state.df
        dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # If no explicit datetime, allow user to specify a column to parse
        if not dt_cols:
            st.warning("No datetime column automatically detected.")
            candidate_dt = st.selectbox("Select column to parse as Datetime", df.columns)
            if st.button("Parse as Datetime"):
                try:
                    st.session_state.df[candidate_dt] = pd.to_datetime(st.session_state.df[candidate_dt])
                    st.success(f"Column '{candidate_dt}' parsed as Datetime.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Parse error: {str(e)}")
        else:
            time_col = st.selectbox("Temporal Timestamp Column", dt_cols)
            val_col = st.selectbox("Time Series Metric", num_cols)

            ts1, ts2, ts3, ts4, ts5 = st.tabs([
                "Temporal Resampling", "Rolling Averages", "Growth (MoM/YoY)", "Cumulative Sums", "Seasonal Decomposition"
            ])

            with ts1:
                st.subheader("Temporal Resampling")
                freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE", "Annual": "YE"}
                freq_label = st.selectbox("Resampling Frequency", list(freq_map.keys()))
                agg_type = st.selectbox("Aggregation", ["sum", "mean", "max", "min"])

                if st.button("Execute Resampling"):
                    resampled_df = resample_temporal_data(df, time_col, val_col, frequency=freq_map[freq_label], aggregation=agg_type)
                    st.dataframe(resampled_df.head(50), use_container_width=True)
                    fig = create_relationship_chart(resampled_df, "date", f"{val_col}_{agg_type}", chart_type="line")
                    st.plotly_chart(fig, use_container_width=True)
                    register_figure(fig)

            with ts2:
                st.subheader("Rolling & Moving Window Averages")
                window_val = st.slider("Sliding Window Size", 2, 60, 7)
                if st.button("Compute Rolling Metrics"):
                    rolling_df = compute_rolling_metrics(df, val_col, window_size=window_val)
                    st.dataframe(rolling_df[[time_col, val_col, f"{val_col}_rolling_mean_{window_val}"]].head(50), use_container_width=True)
                    fig = create_relationship_chart(rolling_df, time_col, f"{val_col}_rolling_mean_{window_val}", chart_type="line")
                    st.plotly_chart(fig, use_container_width=True)

            with ts3:
                st.subheader("Period-over-Period Growth Calculations")
                periods = st.slider("Growth Periods Lag", 1, 30, 1)
                if st.button("Compute Period Growth"):
                    growth_df = compute_period_over_period_growth(df, val_col, periods=periods)
                    st.dataframe(growth_df[[time_col, val_col, f"{val_col}_pct_change_{periods}"]].head(50), use_container_width=True)

            with ts4:
                st.subheader("Cumulative Totals & Running Sums")
                if st.button("Calculate Cumulative Sum"):
                    cumsum_df = compute_cumulative_totals(df, val_col)
                    fig = create_relationship_chart(cumsum_df, time_col, f"{val_col}_cumsum", chart_type="line")
                    st.plotly_chart(fig, use_container_width=True)

            with ts5:
                st.subheader("Seasonality & Trend Decomposition")
                decomp_model = st.selectbox("Decomposition Model", ["additive", "multiplicative"])
                decomp_period = st.number_input("Seasonal Cycle Period", min_value=2, max_value=365, value=12)

                if st.button("Decompose Time-Series"):
                    try:
                        decomp_res = decompose_seasonality_trend(df, time_col, val_col, model=decomp_model, period=int(decomp_period))
                        decomp_df = decomp_res["data"]
                        st.dataframe(decomp_df.head(20), use_container_width=True)

                        fig_trend = create_relationship_chart(decomp_df, "date", "trend", chart_type="line")
                        st.plotly_chart(fig_trend, use_container_width=True)
                        fig_season = create_relationship_chart(decomp_df, "date", "seasonal", chart_type="line")
                        st.plotly_chart(fig_season, use_container_width=True)
                    except Exception as e:
                        st.error(f"Decomposition error: {str(e)}")


# ==============================================================================
# PAGE 7: Mathematical Modeling & Equations (Features 37-41)
# ==============================================================================
elif nav_selection == "Mathematical Modeling & Clusters":
    st.header("Mathematical Modeling & Unsupervised Learning")
    st.caption("Curve fitting equations, genetic symbolic regression, K-Means clustering, PCA, and Isolation Forest anomaly scoring.")

    if st.session_state.df is None:
        st.info("Please load a dataset in 'Data Ingestion & Overview' first.")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        m_tab1, m_tab2, m_tab3, m_tab4, m_tab5 = st.tabs([
            "Curve Fitting", "Symbolic Regression", "K-Means Clustering", "PCA Reduction", "Isolation Forest"
        ])

        with m_tab1:
            st.subheader("Curve Fitting & Trendline Equations")
            if len(num_cols) >= 2:
                cf_x = st.selectbox("Predictor Variable (X)", num_cols, index=0)
                cf_y = st.selectbox("Response Variable (Y)", num_cols, index=1)
                cf_type = st.selectbox("Mathematical Model Function", ["linear", "quadratic", "exponential", "logarithmic", "power"])

                if st.button("Fit Mathematical Formula"):
                    fit_res = fit_curve_and_equation(df, cf_x, cf_y, curve_type=cf_type)
                    st.success(f"**Discovered Formula**: `{fit_res['formula']}` | R² Goodness-of-Fit: **{fit_res['r_squared']}**")

                    # Overlay plot
                    fig = create_relationship_chart(df, cf_x, cf_y, chart_type="scatter")
                    st.plotly_chart(fig, use_container_width=True)
                    register_figure(fig)
            else:
                st.info("Requires at least 2 numerical features.")

        with m_tab2:
            st.subheader("Symbolic Regression (Genetic Algorithm)")
            st.caption("Discovers unknown mathematical laws without pre-specified functional forms using genetic expression trees.")
            if len(num_cols) >= 2:
                sr_x = st.selectbox("Symbolic X Variable", num_cols, index=0)
                sr_y = st.selectbox("Symbolic Y Variable", num_cols, index=1)
                sr_gen = st.slider("Generations", 5, 50, 15)
                sr_pop = st.slider("Population Size", 20, 100, 40)

                if st.button("Execute Genetic Discovery"):
                    with st.spinner("Evolving mathematical formulas..."):
                        sr_res = run_symbolic_regression(df, sr_x, sr_y, generations=sr_gen, population_size=sr_pop)
                        st.success(f"Discovered Equation: **{sr_res['equation']}**")
                        st.write(f"Mean Squared Error: **{sr_res['mse']}** | R²: **{sr_res['r_squared']}**")
            else:
                st.info("Requires at least 2 numerical features.")

        with m_tab3:
            st.subheader("K-Means Feature Clustering")
            if len(num_cols) >= 2:
                km_features = st.multiselect("Clustering Features", num_cols, default=num_cols[:min(3, len(num_cols))])
                km_k = st.slider("Number of Clusters (K)", 2, 8, 3)

                if km_features and st.button("Execute K-Means"):
                    clustered_df, km_meta = run_kmeans_clustering(df, features=km_features, n_clusters=km_k)
                    st.write(f"Inertia: **{km_meta['inertia']}** | Distribution: `{km_meta['cluster_counts']}`")
                    st.dataframe(clustered_df.head(20), use_container_width=True)
                    fig = create_relationship_chart(clustered_df, km_features[0], km_features[1], chart_type="scatter", color_col="Cluster")
                    st.plotly_chart(fig, use_container_width=True)
                    register_figure(fig)

        with m_tab4:
            st.subheader("Dimensionality Reduction (PCA)")
            if len(num_cols) >= 2:
                pca_features = st.multiselect("PCA Features", num_cols, default=num_cols[:min(4, len(num_cols))])
                pca_comp = st.selectbox("Components", [2, 3])

                if len(pca_features) >= pca_comp and st.button("Run PCA"):
                    pca_df, p_meta = run_pca_reduction(df, features=pca_features, n_components=pca_comp)
                    st.write(f"Total Explained Variance: **{p_meta['total_explained_variance']}%**")
                    st.dataframe(pca_df.head(20), use_container_width=True)
                    fig = create_relationship_chart(pca_df, "PC1", "PC2", chart_type="scatter")
                    st.plotly_chart(fig, use_container_width=True)

        with m_tab5:
            st.subheader("Multivariate Anomaly Scoring (Isolation Forest)")
            if len(num_cols) >= 2:
                iso_features = st.multiselect("Anomaly Features", num_cols, default=num_cols[:min(3, len(num_cols))])
                iso_contam = st.slider("Expected Contamination Rate", 0.01, 0.20, 0.05, step=0.01)

                if iso_features and st.button("Compute Anomaly Scores"):
                    iso_df, i_meta = run_isolation_forest_anomaly_detection(df, features=iso_features, contamination=iso_contam)
                    st.write(f"Detected **{i_meta['anomalies_detected']}** anomalies ({i_meta['anomaly_percentage']}%)")
                    st.dataframe(iso_df.head(30), use_container_width=True)
                    fig = create_relationship_chart(iso_df, iso_features[0], iso_features[1], chart_type="scatter", color_col="Anomaly_Flag")
                    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 8: Dynamic Visualizations (Features 42-47)
# ==============================================================================
elif nav_selection == "Dynamic Visualizations":
    st.header("Dynamic Visualizations & Plot Studio")
    st.caption("Auto-plotting engine, distribution analysis, bivariate relationship graphs, and custom interactive chart builder.")

    if st.session_state.df is None:
        st.info("Please load a dataset in 'Data Ingestion & Overview' first.")
    else:
        df = st.session_state.df

        v_tab1, v_tab2, v_tab3, v_tab4, v_tab5 = st.tabs([
            "Auto-Plotting Engine", "Distribution Studio", "Relationship Studio", "Categorical Studio", "Custom Chart Builder"
        ])

        with v_tab1:
            st.subheader("Auto-Plotting Recommendation Engine")
            st.caption("Inspects column data types and generates recommended visualizations.")
            auto_fig = create_auto_plot(df)
            st.plotly_chart(auto_fig, use_container_width=True)
            register_figure(auto_fig)

        with v_tab2:
            st.subheader("Distribution Charts")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if num_cols:
                d_col = st.selectbox("Distribution Target", num_cols)
                d_type = st.selectbox("Chart Style", ["histogram", "box", "kde"])
                d_group = st.selectbox("Split by Category (Optional)", ["None"] + cat_cols)
                d_fig = create_distribution_chart(df, d_col, chart_type=d_type, color_col=None if d_group == "None" else d_group)
                st.plotly_chart(d_fig, use_container_width=True)
                register_figure(d_fig)

        with v_tab3:
            st.subheader("Relationship Charts")
            if len(num_cols) >= 2:
                r_x = st.selectbox("X Axis", num_cols, index=0)
                r_y = st.selectbox("Y Axis", num_cols, index=1)
                r_style = st.selectbox("Plot Type", ["scatter", "bubble", "line"])
                r_group = st.selectbox("Color Grouping (Optional)", ["None"] + cat_cols)
                r_size = st.selectbox("Bubble Size (Optional)", ["None"] + num_cols) if r_style == "bubble" else "None"

                r_fig = create_relationship_chart(
                    df,
                    x_col=r_x,
                    y_col=r_y,
                    chart_type=r_style,
                    color_col=None if r_group == "None" else r_group,
                    size_col=None if r_size == "None" else r_size
                )
                st.plotly_chart(r_fig, use_container_width=True)
                register_figure(r_fig)

        with v_tab4:
            st.subheader("Categorical Charts")
            if cat_cols:
                cat_var = st.selectbox("Category Dimension", cat_cols)
                cat_style = st.selectbox("Style", ["bar", "pie", "violin"])
                cat_metric = st.selectbox("Aggregate Metric (Optional)", ["None"] + num_cols) if cat_style in ["bar", "violin"] else "None"

                c_fig = create_categorical_chart(
                    df,
                    cat_col=cat_var,
                    val_col=None if cat_metric == "None" else cat_metric,
                    chart_type=cat_style
                )
                st.plotly_chart(c_fig, use_container_width=True)
                register_figure(c_fig)

        with v_tab5:
            st.subheader("Custom Multi-Axis Chart Builder")
            cb_type = st.selectbox("Select Visual Primitive", ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Density Heatmap"])
            c_x = st.selectbox("Custom X Axis", df.columns)
            c_y = st.selectbox("Custom Y Axis (Optional)", ["None"] + list(df.columns))
            c_col = st.selectbox("Custom Color Dimension", ["None"] + list(df.columns))

            if st.button("Generate Custom Chart"):
                built_fig = build_custom_chart(
                    df,
                    chart_type=cb_type,
                    x_col=c_x,
                    y_col=None if c_y == "None" else c_y,
                    color_col=None if c_col == "None" else c_col
                )
                st.plotly_chart(built_fig, use_container_width=True)
                register_figure(built_fig)


# ==============================================================================
# PAGE 9: Exporting & Reporting (Features 48-52)
# ==============================================================================
elif nav_selection == "Executive Reports & Export":
    st.header("Executive Reports & Dataset Exporting")
    st.caption("Generate executive summaries, downloadable sanitized datasets, standalone interactive HTML reports, and printable PDFs.")

    if st.session_state.df is None:
        st.info("Please load a dataset in 'Data Ingestion & Overview' first.")
    else:
        df = st.session_state.df
        logger = st.session_state.audit_logger

        rep_tab1, rep_tab2, rep_tab3 = st.tabs([
            "Automated Executive Narrative", "Report Compilation (HTML/PDF)", "Dataset Download"
        ])

        with rep_tab1:
            st.subheader("AI Executive Summary (Feature 50)")
            exec_summary = generate_executive_summary(df, audit_logs=logger.get_logs())
            st.markdown(exec_summary)

        with rep_tab2:
            st.subheader("Comprehensive Analytical Report Compilation")
            st.write("Compiles executive narrative, dataset structure, descriptive statistics, and active charts into unified reports.")

            col_rep1, col_rep2 = st.columns(2)
            with col_rep1:
                st.markdown("##### Standalone HTML Report (Feature 51)")
                html_report_str = generate_html_report(
                    df,
                    summary_text=exec_summary,
                    audit_logs=logger.get_logs(),
                    figures=st.session_state.recent_figures
                )
                st.download_button(
                    label="Download Full HTML Report",
                    data=html_report_str,
                    file_name="datasight_analytical_report.html",
                    mime="text/html"
                )

            with col_rep2:
                st.markdown("##### Executive PDF Report (Feature 52)")
                pdf_bytes = generate_pdf_report(
                    df,
                    summary_text=exec_summary,
                    audit_logs=logger.get_logs()
                )
                st.download_button(
                    label="Download Executive PDF Report",
                    data=pdf_bytes,
                    file_name="datasight_analytical_report.pdf",
                    mime="application/pdf"
                )

        with rep_tab3:
            st.subheader("Cleaned Dataset Export (Feature 48)")
            fmt_choice = st.selectbox("Export File Format", ["CSV", "Excel", "Parquet"])

            data_bytes, mime_type, ext = export_dataset_bytes(df, file_format=fmt_choice.lower())
            st.download_button(
                label=f"Download Dataset as .{ext}",
                data=data_bytes,
                file_name=f"datasight_cleaned_dataset.{ext}",
                mime=mime_type
            )
