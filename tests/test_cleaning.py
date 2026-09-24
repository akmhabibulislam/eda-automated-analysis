"""
Tests for Data Cleaning, Wrangling, and Validation Guardrails.
"""

import unittest
import math
import numpy as np
import pandas as pd

from backend.cleaning.cleaning import (
    clean_text_columns,
    parse_boolean_series,
    treat_outliers,
    standardize_column_headers,
    impute_missing_values,
    analyze_cleaning_recommendations,
    run_automated_cleaning,
    cast_data_types,
    drop_missing_values
)
from backend.cleaning.wrangling import (
    add_custom_formula_column,
    bin_continuous_column,
    filter_rows_structured,
    extract_regex_patterns,
    merge_datasets,
    pivot_dataframe,
    aggregate_groupby
)
from backend.core.validation import (
    validate_dataframe_not_empty,
    validate_unique_columns,
    validate_numeric_finite_column,
    validate_cardinality_guard,
    validate_merge_safety,
    ValidationError,
    ResourceLimitError
)


class TestCleaningAndWranglingModule(unittest.TestCase):

    def test_text_cleaning_preserves_nan_without_stringification(self):
        df = pd.DataFrame({
            "comments": ["  Clean ME  ", np.nan, "Sample Text"]
        })
        cleaned = clean_text_columns(df, columns=["comments"], strip_whitespace=True, case_transformation="lower")
        self.assertEqual(cleaned["comments"].iloc[0], "clean me")
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

    def test_ast_formula_blocks_excessive_exponentiation(self):
        df = pd.DataFrame({"x": [2.0, 3.0, 4.0]})
        with self.assertRaises(ValueError):
            add_custom_formula_column(df, "blown_up", "x ** 100000000")

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

    def test_structured_query_contains_semantic_guard(self):
        df = pd.DataFrame({"age": [25, 35, 45]})
        with self.assertRaises(ValueError):
            filter_rows_structured(df, "age", "contains", "2")

    def test_structured_query_in_type_coercion(self):
        df = pd.DataFrame({"age": [25, 35, 45]})
        f = filter_rows_structured(df, "age", "in", "25, 45")
        self.assertEqual(len(f), 2)

    def test_column_header_standardization_collision_guard(self):
        df = pd.DataFrame(columns=["Test Metric", "test_metric"])
        std_df, mapping = standardize_column_headers(df, case_style="snake_case")
        self.assertEqual(list(std_df.columns), ["test_metric", "test_metric_2"])

    def test_data_type_casting_unparsed_token_accounting(self):
        df = pd.DataFrame({"numbers": ["10", "20", "invalid_number", "40"]})
        cast_df, unparsed = cast_data_types(df, {"numbers": "int64"})
        self.assertEqual(unparsed.get("numbers"), 1)
        self.assertTrue(pd.isna(cast_df["numbers"].iloc[2]))
        self.assertEqual(cast_df["numbers"].iloc[0], 10)

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
        self.assertNotEqual(str(binned["scores_binned"].iloc[2]), "nan")

    def test_merge_cartesian_explosion_guard(self):
        df1 = pd.DataFrame({"key": ["dup"] * 1000, "val1": range(1000)})
        df2 = pd.DataFrame({"key": ["dup"] * 1000, "val2": range(1000)})
        with self.assertRaises(ResourceLimitError):
            merge_datasets(df1, df2, on="key", max_output_rows=50000)

    def test_groupby_and_pivot_pre_execution_resource_guards(self):
        df = pd.DataFrame({
            f"col_{i}": [f"val_{i}_{j}" for j in range(200)] for i in range(5)
        })
        with self.assertRaises(ResourceLimitError):
            aggregate_groupby(df, [f"col_{i}" for i in range(5)], aggregations={"col_0": ["count"]}, max_groups_limit=10)

    def test_treat_outliers_zero_iqr(self):
        df_zero_iqr = pd.DataFrame({"val": [10.0, 10.0, 10.0, 10.0, 10.0, 100.0]})
        treated, meta = treat_outliers(df_zero_iqr, "val", method="iqr")
        self.assertEqual(meta["outliers_detected"], 0)


if __name__ == "__main__":
    unittest.main()
