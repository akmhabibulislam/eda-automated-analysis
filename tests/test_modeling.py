"""
Tests for Mathematical Modeling, Clustering, Dimensionality Reduction, and Anomaly Detection.
"""

import unittest
import numpy as np
import pandas as pd

from backend.modeling.modeling import (
    fit_curve_and_equation,
    run_kmeans_clustering,
    run_pca_reduction,
    run_isolation_forest_anomaly_detection
)


class TestModelingModule(unittest.TestCase):

    def test_curve_fitting_known_line(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * x + 5.0
        df = pd.DataFrame({"x": x, "y": y})
        res = fit_curve_and_equation(df, "x", "y", curve_type="linear")
        self.assertAlmostEqual(res["parameters"][0], 3.0, places=3)
        self.assertAlmostEqual(res["parameters"][1], 5.0, places=3)
        self.assertAlmostEqual(res["r_squared"], 1.0, places=4)

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

    def test_modeling_blocks_raw_identifiers(self):
        df = pd.DataFrame({
            "customer_id": [101, 102, 103, 104, 105],
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        with self.assertRaises(ValueError):
            run_kmeans_clustering(df, ["customer_id", "feature1"])

        with self.assertRaises(ValueError):
            run_pca_reduction(df, ["customer_id", "feature1"])

        with self.assertRaises(ValueError):
            run_isolation_forest_anomaly_detection(df, ["customer_id", "feature1"])


if __name__ == "__main__":
    unittest.main()
