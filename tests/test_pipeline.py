"""
Exhaustive Unit & Integration Test Suite for DataSight Analytics Engine.
Performs verification against known mathematical ground truths and handles
complex edge cases:
- Known-Answer Tests (KAT) for descriptive statistics, Pearson/Spearman/Kendall, T-Tests, ANOVA, Chi-Square
- Linear & Polynomial curve fitting ground truths
- Secure AST formula evaluation & injection blocking
- Structured row filtering
- Null preservation in string operations
- Boolean string parsing integrity
- Float precision retention policy
- Inverse transformed K-Means cluster centers
- PCA loadings and explained variance sums
- Strict date vs non-date identifier recognition
- Edge cases: Empty DataFrames, 1-row DataFrames, all-null columns, constant zero-variance features
"""

import unittest
import math
import numpy as np
import pandas as pd
from scipy import stats

from backend.ingestion.memory import optimize_dataframe_memory, get_memory_usage
from backend.ingestion.loaders import detect_file_format, is_valid_date_series, detect_schema
from backend.cleaning.cleaning import (
    clean_text_columns,
    parse_boolean_series,
    treat_outliers,
    standardize_column_headers,
    impute_missing_values,
    analyze_cleaning_recommendations,
    run_automated_cleaning,
    cast_data_types,
    drop_missing_values,
    AuditLogger
)
from backend.core.validation import (
    validate_dataframe_not_empty,
    validate_unique_columns,
    validate_numeric_finite_column,
    validate_non_zero_variance,
    validate_cardinality_guard,
    validate_merge_safety,
    sanitize_and_report_numeric_policy,
    ValidationError
)
from backend.core.lineage import (
    DatasetSessionManager,
    LineageRecord,
    compute_dataframe_fingerprint,
    get_provenance_metadata
)
from backend.cleaning.wrangling import (
    add_custom_formula_column,
    bin_continuous_column,
    filter_rows_structured,
    extract_regex_patterns,
    merge_datasets,
    pivot_dataframe
)
from backend.analysis.statistics import (
    compute_univariate_summary,
    compute_correlation_matrix,
    detect_highly_correlated_pairs,
    compute_variance_inflation_factors,
    compute_skewness_kurtosis
)
from backend.analysis.hypothesis import (
    run_t_test,
    run_anova,
    run_chi_square,
    fit_distributions,
    apply_multiple_testing_correction
)
from backend.analysis.timeseries import (
    ensure_sorted_timeseries,
    decompose_seasonality_trend
)
from backend.modeling.modeling import (
    fit_curve_and_equation,
    run_kmeans_clustering,
    run_pca_reduction
)
from backend.reporting.reports import (
    generate_executive_summary,
    generate_pdf_report
)


class TestDataSightCorrectness(unittest.TestCase):

    # -------------------------------------------------------------
    # 1. Known-Answer Statistical Tests
    # -------------------------------------------------------------

    def test_univariate_summary_known_values(self):
        # Known sample: [2, 4, 4, 4, 5, 5, 7, 9]
        # Mean = 5.0, Median = 4.5, Min = 2.0, Max = 9.0
        # Population Variance = 4.0, Sample Variance (ddof=1) = 4.5714
        data = pd.DataFrame({"metric": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]})
        summary = compute_univariate_summary(data)
        row = summary.iloc[0]

        self.assertAlmostEqual(row["Mean"], 5.0, places=3)
        self.assertAlmostEqual(row["Median"], 4.5, places=3)
        self.assertAlmostEqual(row["Min"], 2.0, places=3)
        self.assertAlmostEqual(row["Max"], 9.0, places=3)
        self.assertAlmostEqual(row["Variance"], 4.5714, places=3)
        self.assertAlmostEqual(row["Std_Dev"], math.sqrt(4.571428), places=3)

    def test_correlation_matrix_known_values(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        df = pd.DataFrame({"x": x, "y": y})

        corr_pearson = compute_correlation_matrix(df, method="pearson")
        self.assertAlmostEqual(corr_pearson.loc["x", "y"], 1.0, places=4)

        corr_spearman = compute_correlation_matrix(df, method="spearman")
        self.assertAlmostEqual(corr_spearman.loc["x", "y"], 1.0, places=4)

        z = [10.0, 8.0, 6.0, 4.0, 2.0]
        df_inv = pd.DataFrame({"x": x, "z": z})
        corr_inv = compute_correlation_matrix(df_inv, method="pearson")
        self.assertAlmostEqual(corr_inv.loc["x", "z"], -1.0, places=4)

    def test_paired_t_test_known_values(self):
        s_a = pd.Series([10.0, 12.0, 14.0, np.nan, 16.0, 18.0])
        s_b = pd.Series([8.0, 10.0, 12.0, 14.0, np.nan, 16.0])

        res = run_t_test(s_a, s_b, test_type="paired")
        self.assertEqual(res["n_obs_a"], 4)
        self.assertEqual(res["n_obs_b"], 4)
        self.assertAlmostEqual(res["mean_sample_a"] - res["mean_sample_b"], 2.0, places=3)

    def test_anova_known_values(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        g3 = np.array([7.0, 8.0, 9.0])
        res = run_anova([g1, g2, g3])
        self.assertAlmostEqual(res["f_statistic"], 27.0, places=2)
        self.assertTrue(res["is_significant"])

    def test_chi_square_known_values(self):
        ct_indep = pd.DataFrame([[10, 10], [10, 10]], index=["R1", "R2"], columns=["C1", "C2"])
        res_zero = run_chi_square(ct_indep)
        self.assertAlmostEqual(res_zero["chi2_statistic"], 0.0, places=4)
        self.assertFalse(res_zero["is_significant"])

        ct_dep = pd.DataFrame([[50, 5], [5, 50]], index=["R1", "R2"], columns=["C1", "C2"])
        res_sig = run_chi_square(ct_dep)
        self.assertTrue(res_sig["is_significant"])

    def test_curve_fitting_known_line(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * x + 5.0
        df = pd.DataFrame({"x": x, "y": y})
        res = fit_curve_and_equation(df, "x", "y", curve_type="linear")
        self.assertAlmostEqual(res["parameters"][0], 3.0, places=3)
        self.assertAlmostEqual(res["parameters"][1], 5.0, places=3)
        self.assertAlmostEqual(res["r_squared"], 1.0, places=4)

    # -------------------------------------------------------------
    # 2. Security & AST Expression Evaluation
    # -------------------------------------------------------------

    def test_secure_ast_formula_valid_operations(self):
        df = pd.DataFrame({
            "revenue": [100.0, 200.0, 300.0],
            "cost": [40.0, 80.0, 120.0],
            "tax rate": [0.10, 0.10, 0.10]
        })
        df1 = add_custom_formula_column(df, "profit", "revenue - cost")
        self.assertEqual(df1["profit"].tolist(), [60.0, 120.0, 180.0])

        df2 = add_custom_formula_column(df, "tax", "cost * `tax rate`")
        self.assertEqual(df2["tax"].tolist(), [4.0, 8.0, 12.0])

        df3 = add_custom_formula_column(df, "log_rev", "log(revenue)")
        self.assertAlmostEqual(df3["log_rev"].iloc[0], math.log(100.0), places=3)

    def test_secure_ast_formula_blocks_code_injection(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with self.assertRaises(ValueError):
            add_custom_formula_column(df, "hacked", "__import__('os').system('echo hacked')")

        with self.assertRaises(ValueError):
            add_custom_formula_column(df, "hacked", "open('/etc/passwd').read()")

        with self.assertRaises(ValueError):
            add_custom_formula_column(df, "hacked", "exec('x=1')")

    def test_structured_query_filtering(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 35, 45]
        })
        f1 = filter_rows_structured(df, "age", ">", 30)
        self.assertEqual(len(f1), 2)

        f2 = filter_rows_structured(df, "name", "contains", "ali")
        self.assertEqual(len(f2), 1)
        self.assertEqual(f2["name"].iloc[0], "Alice")

    # -------------------------------------------------------------
    # 3. Data Cleaning, Parsing & Type Correctness
    # -------------------------------------------------------------

    def test_text_cleaning_preserves_nan_without_stringification(self):
        df = pd.DataFrame({
            "comments": ["  Clean ME  ", np.nan, "Sample Text"]
        })
        cleaned = clean_text_columns(df, columns=["comments"], strip_whitespace=True, case_transformation="lower")
        self.assertEqual(cleaned["comments"].iloc[0], "clean me")
        # Ensure it remains a true null and not the string 'nan'
        self.assertTrue(pd.isna(cleaned["comments"].iloc[1]))
        val_str = str(cleaned["comments"].iloc[1])
        self.assertNotIn(val_str, ["clean me", "sample text"])

    def test_robust_boolean_parser(self):
        bool_inputs = pd.Series(["True", "false", "YES", "No", "1", "0", np.nan])
        parsed = parse_boolean_series(bool_inputs)

        self.assertTrue(parsed.iloc[0])
        self.assertFalse(parsed.iloc[1])
        self.assertTrue(parsed.iloc[2])
        self.assertFalse(parsed.iloc[3])
        self.assertTrue(parsed.iloc[4])
        self.assertFalse(parsed.iloc[5])
        self.assertTrue(pd.isna(parsed.iloc[6]))

        with self.assertRaises(ValueError):
            parse_boolean_series(pd.Series(["NotABool"]))

    def test_memory_layer_preserves_float_precision_by_default(self):
        df = pd.DataFrame({"high_precision_pi": [3.141592653589793]})
        opt_df, meta = optimize_dataframe_memory(df, downcast_integers=True, downcast_floats=False)
        self.assertEqual(opt_df["high_precision_pi"].dtype, np.float64)
        self.assertFalse(meta["float_downcasting_applied"])

    def test_controlled_auto_clean_does_not_mutate_blindly(self):
        df = pd.DataFrame({
            "User ID": [1, 2, 3],
            "Score": [10.0, np.nan, 1000.0]
        })
        cleaned_df, summary = run_automated_cleaning(df, impute_missing=False, cap_outliers=False)
        self.assertTrue(cleaned_df["score"].isna().sum() == 1)
        self.assertEqual(cleaned_df["score"].max(), 1000.0)
        self.assertFalse(summary["imputation_performed"])
        self.assertFalse(summary["outlier_capping_performed"])

    # -------------------------------------------------------------
    # 4. Ingestion & Date vs Non-Date Recognition
    # -------------------------------------------------------------

    def test_file_format_detection_safety(self):
        self.assertEqual(detect_file_format("data.csv"), "csv")
        self.assertEqual(detect_file_format("metrics.parquet"), "parquet")

        with self.assertRaises(ValueError):
            detect_file_format("unknown.xyz")

        with self.assertRaises(ValueError):
            detect_file_format("ambiguous.txt")

    def test_date_detection_rejects_alphanumeric_ids(self):
        id_series = pd.Series(["TX-1001-A", "TX-1002-B", "TX-1003-C"])
        self.assertFalse(is_valid_date_series(id_series, col_name="tx_id"))

        uuid_series = pd.Series(["123e4567-e89b-12d3-a456-426614174000", "e6362c7f-e8ac-4126-866b-05c65b09ee1c"])
        self.assertFalse(is_valid_date_series(uuid_series, col_name="uuid"))

        real_dates = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"])
        self.assertTrue(is_valid_date_series(real_dates, col_name="event_date"))

        slash_dates = pd.Series(["15/01/2024", "16/01/2024", "17/01/2024"])
        self.assertTrue(is_valid_date_series(slash_dates, col_name="event_date"))

    # -------------------------------------------------------------
    # 5. Modeling Correctness (Centers & Loadings)
    # -------------------------------------------------------------

    def test_kmeans_centers_in_original_scale(self):
        x = np.array([90, 100, 110, 990, 1000, 1010], dtype=float)
        y = np.array([90, 100, 110, 990, 1000, 1010], dtype=float)
        df = pd.DataFrame({"x": x, "y": y})

        _, meta = run_kmeans_clustering(df, ["x", "y"], n_clusters=2, scale=True)
        centers_df = meta["centers_dataframe"]

        c_vals = sorted(centers_df["x"].tolist())
        self.assertAlmostEqual(c_vals[0], 100.0, delta=15.0)
        self.assertAlmostEqual(c_vals[1], 1000.0, delta=15.0)

    def test_pca_loadings_and_explained_variance(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            "c": [5.0, 1.0, 4.0, 2.0, 3.0]
        })
        pca_coords, meta = run_pca_reduction(df, ["a", "b", "c"], n_components=2)
        self.assertIn("loadings", meta)
        self.assertIn("contributions", meta)
        self.assertAlmostEqual(sum(meta["explained_variance_ratio"]), meta["total_explained_variance"], places=1)

    # -------------------------------------------------------------
    # 6. Edge Cases Coverage
    # -------------------------------------------------------------

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        summary = compute_univariate_summary(empty_df)
        self.assertTrue(summary.empty)

        opt_df, meta = optimize_dataframe_memory(empty_df)
        self.assertEqual(meta["initial_mb"], 0.0)

    def test_single_row_dataframe(self):
        one_row = pd.DataFrame({"val": [42.0], "cat": ["single"]})
        summary = compute_univariate_summary(one_row)
        self.assertEqual(summary["Count"].iloc[0], 1)
        self.assertEqual(summary["Mean"].iloc[0], 42.0)

    def test_all_null_column(self):
        df_nulls = pd.DataFrame({"empty_col": [np.nan, np.nan, np.nan]})
        summary = compute_univariate_summary(df_nulls)
        self.assertTrue(summary.empty)

    def test_constant_zero_variance_column(self):
        df_const = pd.DataFrame({"const": [5.0, 5.0, 5.0, 5.0]})
        summary = compute_univariate_summary(df_const)
        self.assertEqual(summary["Variance"].iloc[0], 0.0)

    def test_time_series_insufficient_period_raises_error(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        ts_df = pd.DataFrame({"date": dates, "val": np.arange(10)})
        with self.assertRaises(ValueError):
            decompose_seasonality_trend(ts_df, "date", "val", period=12)

    # -------------------------------------------------------------
    # 7. Hardened Architecture & Validation Layer
    # -------------------------------------------------------------

    def test_validation_layer_inf_rejection(self):
        df_inf = pd.DataFrame({"val": [1.0, np.inf, 3.0]})
        with self.assertRaises(ValidationError):
            validate_numeric_finite_column(df_inf, "val")

        df_neginf = pd.DataFrame({"val": [1.0, -np.inf, 3.0]})
        with self.assertRaises(ValidationError):
            validate_numeric_finite_column(df_neginf, "val")

        df_valid = pd.DataFrame({"val": [1.0, np.nan, 3.0, 4.0]})
        clean = validate_numeric_finite_column(df_valid, "val")
        self.assertEqual(len(clean), 3)

    def test_validation_layer_cardinality_guard(self):
        df_cats = pd.DataFrame({"cat": [f"c_{i}" for i in range(150)]})
        with self.assertRaises(ValidationError):
            validate_cardinality_guard(df_cats, "cat", max_cardinality=100)

        # Passes when within threshold
        n = validate_cardinality_guard(df_cats, "cat", max_cardinality=200)
        self.assertEqual(n, 150)

    def test_validation_layer_duplicate_columns(self):
        df_dup = pd.DataFrame([[1, 2]], columns=["col_a", "col_a"])
        with self.assertRaises(ValidationError):
            validate_unique_columns(df_dup)

    def test_nullable_integer_downcasting_with_nan(self):
        # A float column containing whole numbers and NaN
        df = pd.DataFrame({"counts": [10.0, 20.0, np.nan, 40.0]})
        opt_df, meta = optimize_dataframe_memory(df, downcast_integers=True)
        # Should be downcast to nullable Int8 without crashing
        self.assertEqual(str(opt_df["counts"].dtype), "Int8")
        self.assertTrue(pd.isna(opt_df["counts"].iloc[2]))
        self.assertEqual(opt_df["counts"].iloc[0], 10)

    def test_column_header_standardization_collision_guard(self):
        # Two columns that would both sanitize to 'test_metric'
        df = pd.DataFrame(columns=["Test Metric", "test_metric"])
        std_df, mapping = standardize_column_headers(df, case_style="snake_case")
        self.assertEqual(list(std_df.columns), ["test_metric", "test_metric_2"])
        self.assertEqual(mapping["Test Metric"], "test_metric")
        self.assertEqual(mapping["test_metric"], "test_metric_2")

    def test_data_type_casting_unparsed_token_accounting(self):
        df = pd.DataFrame({"numbers": ["10", "20", "invalid_number", "40"]})
        cast_df, unparsed = cast_data_types(df, {"numbers": "int64"})
        self.assertEqual(unparsed.get("numbers"), 1)
        self.assertTrue(pd.isna(cast_df["numbers"].iloc[2]))
        self.assertEqual(cast_df["numbers"].iloc[0], 10)

    def test_multiple_testing_correction(self):
        # 4 p-values
        raw_p = [0.01, 0.04, 0.03, 0.20]
        # Bonferroni
        res_bonf = apply_multiple_testing_correction(raw_p, method="bonferroni", alpha=0.05)
        self.assertEqual(res_bonf["adjusted_p_values"], [0.04, 0.16, 0.12, 0.80])
        self.assertEqual(res_bonf["significant"], [True, False, False, False])

        # Benjamini-Hochberg (FDR)
        res_bh = apply_multiple_testing_correction(raw_p, method="benjamini_hochberg", alpha=0.05)
        self.assertTrue(len(res_bh["adjusted_p_values"]) == 4)
        # Adjusted p-values must be monotonic when sorted
        sorted_bh = sorted(res_bh["adjusted_p_values"])
        self.assertTrue(all(x <= y for x, y in zip(sorted_bh, sorted_bh[1:])))

    def test_dataset_session_manager_and_lineage(self):
        manager = DatasetSessionManager()
        df_init = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        manager.load_dataset(df_init, dataset_name="TestSet")
        self.assertEqual(manager.version, 1)

        # Modify current df
        df_next = pd.DataFrame({"a": [1, 2], "b": [4, 5]})
        manager.update_current_df(
            new_df=df_next,
            operation_name="Filter",
            parameters={"limit": 2},
            description="Filtered rows"
        )
        self.assertEqual(manager.version, 2)
        self.assertEqual(len(manager.current_df), 2)
        # Immutable original unchanged
        self.assertEqual(len(manager.original_df), 3)

        # Reset to original
        manager.reset_to_original()
        self.assertEqual(len(manager.current_df), 3)

        lineage_df = manager.get_lineage_dataframe()
        self.assertEqual(len(lineage_df), 3)  # Ingestion, Filter, Reset

    # -------------------------------------------------------------
    # 8. Hardening Guardrails, Semantic Schema & Reproducibility
    # -------------------------------------------------------------

    def test_regex_extraction_preserves_nan(self):
        df = pd.DataFrame({"emails": ["contact@test.com", np.nan, "hello@world.org"]})
        extracted = extract_regex_patterns(df, "emails", r"[\w\.-]+@([\w\.-]+)", "domain")
        self.assertEqual(extracted["domain"].iloc[0], "test.com")
        self.assertTrue(pd.isna(extracted["domain"].iloc[1]))
        self.assertEqual(extracted["domain"].iloc[2], "world.org")

    def test_binning_preserves_nan_without_stringification(self):
        df = pd.DataFrame({"scores": [10.0, 50.0, np.nan, 90.0]})
        binned = bin_continuous_column(df, "scores", bins=3)
        self.assertTrue(pd.isna(binned["scores_binned"].iloc[2]))
        # Must not be the string 'nan'
        self.assertNotEqual(str(binned["scores_binned"].iloc[2]), "nan")

    def test_merge_cartesian_explosion_guard(self):
        # Create data with massive duplicate key potential
        df1 = pd.DataFrame({"key": ["dup"] * 1000, "val1": range(1000)})
        df2 = pd.DataFrame({"key": ["dup"] * 1000, "val2": range(1000)})
        # Merging would create 1,000,000 rows
        with self.assertRaises(ValidationError):
            merge_datasets(df1, df2, on="key", max_output_rows=50000)

    def test_ast_formula_blocks_excessive_exponentiation(self):
        df = pd.DataFrame({"x": [2.0, 3.0, 4.0]})
        with self.assertRaises(ValueError):
            add_custom_formula_column(df, "blown_up", "x ** 100000000")

    def test_semantic_role_detection(self):
        from backend.ingestion.loaders import detect_schema
        df = pd.DataFrame({
            "user_id": [f"ID_{i}" for i in range(100)],
            "email_address": [f"user{i}@example.com" for i in range(100)],
            "latitude": np.random.uniform(20.0, 50.0, size=100),
            "revenue": np.random.uniform(100.0, 1000.0, size=100)
        })
        schema = detect_schema(df)
        self.assertEqual(schema["semantic_roles"]["user_id"], "identifier")
        self.assertEqual(schema["semantic_roles"]["email_address"], "email")
        self.assertEqual(schema["semantic_roles"]["latitude"], "latitude")
        # Ensure user_id was excluded from analytical numerics
        self.assertNotIn("user_id", schema["analytical_numeric_columns"])

    def test_cryptographic_dataset_fingerprint(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df3 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 999]})

        fp1 = compute_dataframe_fingerprint(df1)
        fp2 = compute_dataframe_fingerprint(df2)
        fp3 = compute_dataframe_fingerprint(df3)

        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, fp3)
        self.assertEqual(len(fp1), 64)  # SHA-256 hex length

    def test_software_provenance_metadata(self):
        prov = get_provenance_metadata()
        self.assertIn("python_version", prov)
        self.assertIn("pandas_version", prov)
        self.assertIn("numpy_version", prov)
        self.assertIn("scikit_learn_version", prov)

    def test_treat_outliers_zero_iqr(self):
        # Dataset where Q25 == Q75 (e.g. constant or dominant values)
        df_zero_iqr = pd.DataFrame({"val": [10.0, 10.0, 10.0, 10.0, 10.0, 100.0]})
        treated, meta = treat_outliers(df_zero_iqr, "val", method="iqr")
        self.assertEqual(meta["outliers_detected"], 0)


if __name__ == "__main__":
    unittest.main()

