"""
Tests for Data Ingestion, Schema Detection, and Memory Optimization.
"""

import unittest
import numpy as np
import pandas as pd

from backend.ingestion.memory import optimize_dataframe_memory, get_memory_usage
from backend.ingestion.loaders import (
    detect_file_format,
    is_valid_date_series,
    detect_schema,
    load_from_database
)


class TestIngestionModule(unittest.TestCase):

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

    def test_memory_layer_preserves_float_precision_by_default(self):
        df = pd.DataFrame({"high_precision_pi": [3.141592653589793]})
        opt_df, meta = optimize_dataframe_memory(df, downcast_integers=True, downcast_floats=False)
        self.assertEqual(opt_df["high_precision_pi"].dtype, np.float64)
        self.assertFalse(meta["float_downcasting_applied"])

    def test_nullable_integer_downcasting_with_nan(self):
        df = pd.DataFrame({"counts": [10.0, 20.0, np.nan, 40.0]})
        opt_df, meta = optimize_dataframe_memory(df, downcast_integers=True)
        self.assertEqual(str(opt_df["counts"].dtype), "Int8")
        self.assertTrue(pd.isna(opt_df["counts"].iloc[2]))
        self.assertEqual(opt_df["counts"].iloc[0], 10)

    def test_semantic_role_detection(self):
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
        self.assertNotIn("user_id", schema["analytical_numeric_columns"])


if __name__ == "__main__":
    unittest.main()
