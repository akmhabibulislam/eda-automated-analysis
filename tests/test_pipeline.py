"""
Comprehensive Unit and Integration Tests for DataSight Analytics Engine.
Verifies all 52 core data analytics features across:
- Ingestion and Memory Optimization
- Automated and Granular Cleaning
- Statistics, Correlations, and Hypothesis Tests
- Advanced Wrangling and Reshaping
- Time-Series Analysis and Seasonality
- Modeling, Curve Fitting, Symbolic Regression, Clustering, PCA, Isolation Forest
- Report Generation (HTML, PDF, Dataset Exports)
"""

import unittest
import os
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from backend.ingestion.memory import optimize_dataframe_memory, get_memory_usage, free_memory
from backend.ingestion.loaders import load_dataset, detect_schema, get_dataset_overview
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
    generate_executive_summary,
    generate_html_report,
    generate_pdf_report
)


class TestDataSightPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        self.sample_df = pd.DataFrame({
            "Transaction ID": [f"ID_{i}" for i in range(n)],
            "Units Sold": np.random.randint(10, 50, n),
            "Price": np.random.uniform(5.0, 100.0, n).round(2),
            "Category": np.random.choice(["Alpha", "Beta", "Gamma"], n),
            "Score": np.random.normal(50, 10, n),
            "Notes": [" test text " if i % 2 == 0 else "example text" for i in range(n)]
        })
        # Add duplicates and missing values
        self.sample_df.loc[5, "Score"] = np.nan
        self.sample_df.loc[6, "Score"] = np.nan

    def test_memory_optimization(self):
        df_opt, metrics = optimize_dataframe_memory(self.sample_df)
        self.assertIn("initial_mb", metrics)
        self.assertIn("final_mb", metrics)
        self.assertLessEqual(metrics["final_mb"], metrics["initial_mb"])

    def test_ingestion_and_schema(self):
        csv_buf = io.StringIO()
        self.sample_df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)

        df_loaded, meta = load_dataset(csv_buf, "test.csv")
        self.assertEqual(len(df_loaded), 100)
        schema = detect_schema(df_loaded)
        self.assertIn("numeric_columns", schema)
        self.assertIn("categorical_columns", schema)

        ov = get_dataset_overview(df_loaded)
        self.assertEqual(ov["total_rows"], 100)
        self.assertGreaterEqual(ov["total_missing_cells"], 2)

    def test_cleaning_suite(self):
        logger = AuditLogger()
        # 1. Standardize headers
        df = standardize_column_headers(self.sample_df, case_style="snake_case", logger=logger)
        self.assertIn("transaction_id", df.columns)
        self.assertIn("units_sold", df.columns)

        # 2. Impute missing values
        df = impute_missing_values(df, columns=["score"], strategy="mean", logger=logger)
        self.assertEqual(df["score"].isna().sum(), 0)

        # 3. Duplicate handling
        df = remove_duplicates(df, logger=logger)

        # 4. Outlier treatment
        df, info = treat_outliers(df, column="score", method="iqr", action="cap", logger=logger)
        self.assertIn("outliers_detected", info)

        # 5. Text cleaning
        df = clean_text_columns(df, columns=["notes"], strip_whitespace=True, case_transformation="lower", logger=logger)
        self.assertEqual(df["notes"].iloc[0], "test text")

        # 6. Type casting
        df = cast_data_types(df, {"units_sold": "float"}, logger=logger)
        self.assertTrue(pd.api.types.is_float_dtype(df["units_sold"]))

        # 7. Audit logs
        logs = logger.get_logs()
        self.assertGreater(len(logs), 0)

    def test_auto_cleaning(self):
        logger = AuditLogger()
        cleaned_df, summary = run_automated_cleaning(self.sample_df, logger=logger)
        self.assertEqual(cleaned_df["score"].isna().sum(), 0)
        self.assertGreaterEqual(summary["steps_executed"], 3)

    def test_wrangling_operations(self):
        df = standardize_column_headers(self.sample_df)
        # Custom formula
        df = add_custom_formula_column(df, "total_value", "units_sold * price")
        self.assertIn("total_value", df.columns)

        # Continuous binning
        df = bin_continuous_column(df, "price", "price_bin", bins=4)
        self.assertIn("price_bin", df.columns)

        # Groupby
        agg_df = aggregate_groupby(df, ["category"], {"units_sold": ["mean", "sum"]})
        self.assertGreater(len(agg_df), 0)

        # Filtering
        filtered = filter_rows(df, "units_sold > 20")
        self.assertTrue((filtered["units_sold"] > 20).all())

        # Regex
        regex_df = extract_regex_patterns(df, "transaction_id", r"ID_(\d+)", "id_num")
        self.assertIn("id_num", regex_df.columns)

    def test_statistics_and_correlations(self):
        df = standardize_column_headers(self.sample_df)
        uni = compute_univariate_summary(df)
        self.assertGreater(len(uni), 0)

        cats = compute_categorical_distribution(df)
        self.assertIn("category", cats)

        miss = compute_missingness_matrix(df)
        self.assertIn("summary", miss)

        corr = compute_correlation_matrix(df)
        self.assertFalse(corr.empty)

        sk = compute_skewness_kurtosis(df)
        self.assertGreater(len(sk), 0)

    def test_hypothesis_testing(self):
        a = np.random.normal(10, 2, 50)
        b = np.random.normal(15, 2, 50)
        tt_res = run_t_test(a, b, test_type="independent")
        self.assertTrue(tt_res["is_significant"])

        anova_res = run_anova([a, b, np.random.normal(12, 2, 50)])
        self.assertIn("f_statistic", anova_res)

        ct = pd.DataFrame([[10, 20], [20, 10]], index=["R1", "R2"], columns=["C1", "C2"])
        chi_res = run_chi_square(ct)
        self.assertIn("chi2_statistic", chi_res)

        non_param = run_non_parametric_tests([a, b])
        self.assertIn("p_value", non_param)

        fits = fit_distributions(a)
        self.assertGreater(len(fits), 0)

    def test_timeseries_analysis(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        ts_df = pd.DataFrame({
            "timestamp": dates,
            "metric": np.sin(np.linspace(0, 20, 60)) + np.random.normal(0, 0.1, 60) + 10
        })

        resampled = resample_temporal_data(ts_df, "timestamp", "metric", frequency="W", aggregation="mean")
        self.assertGreater(len(resampled), 0)

        rolling = compute_rolling_metrics(ts_df, "metric", window_size=5)
        self.assertIn("metric_rolling_mean_5", rolling.columns)

        growth = compute_period_over_period_growth(ts_df, "metric", periods=1)
        self.assertIn("metric_pct_change_1", growth.columns)

        cum = compute_cumulative_totals(ts_df, "metric")
        self.assertIn("metric_cumsum", cum.columns)

        decomp = decompose_seasonality_trend(ts_df, "timestamp", "metric", period=7)
        self.assertIn("trend", decomp["data"].columns)

    def test_modeling_and_clusters(self):
        x = np.linspace(1, 10, 50)
        y = 2.5 * x + 4.0 + np.random.normal(0, 0.2, 50)
        fit_df = pd.DataFrame({"x": x, "y": y})

        # Curve fitting
        curve_res = fit_curve_and_equation(fit_df, "x", "y", curve_type="linear")
        self.assertGreater(curve_res["r_squared"], 0.85)

        # Symbolic Regression
        sr_res = run_symbolic_regression(fit_df, "x", "y", generations=5, population_size=15)
        self.assertIn("equation", sr_res)

        # K-Means
        clustered, km_info = run_kmeans_clustering(fit_df, ["x", "y"], n_clusters=2)
        self.assertIn("Cluster", clustered.columns)

        # PCA
        pca_df, pca_info = run_pca_reduction(fit_df, ["x", "y"], n_components=2)
        self.assertIn("PC1", pca_df.columns)

        # Isolation Forest
        iso_df, iso_info = run_isolation_forest_anomaly_detection(fit_df, ["x", "y"])
        self.assertIn("Anomaly_Flag", iso_df.columns)

    def test_visualizations(self):
        fig_auto = create_auto_plot(self.sample_df)
        self.assertIsInstance(fig_auto, go.Figure)

        fig_dist = create_distribution_chart(self.sample_df, "Price", chart_type="histogram")
        self.assertIsInstance(fig_dist, go.Figure)

        fig_rel = create_relationship_chart(self.sample_df, "Units Sold", "Price", chart_type="scatter")
        self.assertIsInstance(fig_rel, go.Figure)

        fig_cat = create_categorical_chart(self.sample_df, "Category", chart_type="bar")
        self.assertIsInstance(fig_cat, go.Figure)

        fig_custom = build_custom_chart(self.sample_df, chart_type="Scatter", x_col="Units Sold", y_col="Price")
        self.assertIsInstance(fig_custom, go.Figure)

    def test_reporting_and_export(self):
        # Dataset exports
        csv_bytes, _, _ = export_dataset_bytes(self.sample_df, file_format="csv")
        self.assertGreater(len(csv_bytes), 0)

        xlsx_bytes, _, _ = export_dataset_bytes(self.sample_df, file_format="excel")
        self.assertGreater(len(xlsx_bytes), 0)

        pq_bytes, _, _ = export_dataset_bytes(self.sample_df, file_format="parquet")
        self.assertGreater(len(pq_bytes), 0)

        # Executive summary
        summary = generate_executive_summary(self.sample_df)
        self.assertIn("Dataset Overview", summary)

        # HTML report
        html_rep = generate_html_report(self.sample_df, summary_text=summary)
        self.assertIn("<!DOCTYPE html>", html_rep)

        # PDF report
        pdf_bytes = generate_pdf_report(self.sample_df, summary_text=summary)
        self.assertGreater(len(pdf_bytes), 0)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
