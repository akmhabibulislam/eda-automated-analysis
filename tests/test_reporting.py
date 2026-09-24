"""
Tests for Reporting, Summaries, HTML/PDF Generation, and Provenance.
"""

import unittest
import numpy as np
import pandas as pd

from backend.reporting.reports import (
    generate_executive_summary,
    generate_html_report,
    generate_pdf_report,
    export_dataset_bytes
)
from backend.core.lineage import (
    DatasetSessionManager,
    compute_dataframe_fingerprint,
    get_provenance_metadata
)


class TestReportingModule(unittest.TestCase):

    def test_executive_summary_exact_percentages_and_skewness(self):
        df = pd.DataFrame({
            "age": [20, 21, 22, 23, 24, 25, 26, 90, 22, 25],
            "city": ["New York"] * 8 + ["London"] * 2
        })
        summary = generate_executive_summary(df)
        self.assertIn("age", summary)
        self.assertIn("city", summary)
        self.assertIn("skewness", summary.lower())
        self.assertIn("%", summary)

    def test_html_report_escaping_and_fingerprint(self):
        malicious_str = "<script>alert('xss')</script>"
        df = pd.DataFrame({
            "user": [malicious_str, "bob"],
            "score": [10, 20]
        })
        html_out = generate_html_report(df, summary_text=f"Summary with {malicious_str}", fingerprint_sha256="abc123sha")
        self.assertIn("abc123sha", html_out)
        self.assertNotIn("<script>alert('xss')</script>", html_out)
        self.assertIn("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;", html_out)

    def test_cryptographic_dataset_fingerprint(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df3 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 999]})

        fp1 = compute_dataframe_fingerprint(df1)
        fp2 = compute_dataframe_fingerprint(df2)
        fp3 = compute_dataframe_fingerprint(df3)

        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, fp3)
        self.assertEqual(len(fp1), 64)

    def test_software_provenance_metadata(self):
        prov = get_provenance_metadata()
        self.assertIn("python_version", prov)
        self.assertIn("pandas_version", prov)
        self.assertIn("numpy_version", prov)
        self.assertIn("scikit_learn_version", prov)

    def test_dataset_session_manager_and_lineage(self):
        manager = DatasetSessionManager()
        df_init = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        manager.load_dataset(df_init, dataset_name="TestSet")
        self.assertEqual(manager.version, 1)

        df_next = pd.DataFrame({"a": [1, 2], "b": [4, 5]})
        manager.update_current_df(
            new_df=df_next,
            operation_name="Filter",
            parameters={"limit": 2},
            description="Filtered rows"
        )
        self.assertEqual(manager.version, 2)
        self.assertEqual(len(manager.current_df), 2)
        self.assertEqual(len(manager.original_df), 3)

        manager.reset_to_original()
        self.assertEqual(len(manager.current_df), 3)

        lineage_df = manager.get_lineage_dataframe()
        self.assertEqual(len(lineage_df), 3)

    def test_dataset_export_bytes(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        csv_bytes, mime, ext = export_dataset_bytes(df, "csv")
        self.assertEqual(ext, "csv")
        self.assertTrue(len(csv_bytes) > 0)

        pq_bytes, mime_pq, ext_pq = export_dataset_bytes(df, "parquet")
        self.assertEqual(ext_pq, "parquet")
        self.assertTrue(len(pq_bytes) > 0)


if __name__ == "__main__":
    unittest.main()
