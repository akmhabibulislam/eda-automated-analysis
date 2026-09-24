"""
Tests for Descriptive Statistics, Correlation, Hypothesis Testing, and Time Series.
"""

import unittest
import math
import numpy as np
import pandas as pd

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
    decompose_seasonality_trend,
    compute_period_over_period_growth
)


class TestStatisticsModule(unittest.TestCase):

    def test_univariate_summary_known_values(self):
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

    def test_multiple_testing_correction(self):
        raw_p = [0.01, 0.04, 0.03, 0.20]
        res_bonf = apply_multiple_testing_correction(raw_p, method="bonferroni", alpha=0.05)
        self.assertEqual(res_bonf["adjusted_p_values"], [0.04, 0.16, 0.12, 0.80])
        self.assertEqual(res_bonf["significant"], [True, False, False, False])

        res_bh = apply_multiple_testing_correction(raw_p, method="benjamini_hochberg", alpha=0.05)
        self.assertTrue(len(res_bh["adjusted_p_values"]) == 4)
        sorted_bh = sorted(res_bh["adjusted_p_values"])
        self.assertTrue(all(x <= y for x, y in zip(sorted_bh, sorted_bh[1:])))

    def test_calendar_aware_mom_yoy_growth(self):
        dates = pd.to_datetime(["2023-01-15", "2023-02-14", "2023-03-20"])
        df = pd.DataFrame({"date": dates, "sales": [100.0, 150.0, 300.0]})
        mom = compute_period_over_period_growth(
            df,
            value_column="sales",
            date_column="date",
            periods=1,
            frequency="M"
        )
        self.assertIn("sales_growth_1ME", mom.columns)
        # February should show (150-100)/100 * 100 = 50.00%
        self.assertAlmostEqual(mom["sales_growth_1ME"].iloc[1], 50.0, places=1)


if __name__ == "__main__":
    unittest.main()
