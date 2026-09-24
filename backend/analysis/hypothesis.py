"""
Statistical Testing & Hypothesis Validation module.
Features 27-31:
27. T-Tests (independent with Welch/Student options and index-aligned Paired tests)
28. ANOVA (Analysis of Variance)
29. Chi-Square Test of Independence
30. Mann-Whitney U / Kruskal-Wallis Non-Parametric Tests
31. Distribution Fitting (Kolmogorov-Smirnov evaluation with transparent parameter disclosure)
Includes Multiple-Testing Corrections: Bonferroni and Benjamini-Hochberg (FDR).
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats

from backend.core.validation import (
    validate_numeric_finite_column,
    validate_non_zero_variance,
    validate_dataframe_not_empty
)


def apply_multiple_testing_correction(
    p_values: List[float],
    method: str = "benjamini_hochberg",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Applies multiple comparison adjustments to family-wise error rates.
    Supported: 'bonferroni', 'benjamini_hochberg' (FDR).
    """
    m = len(p_values)
    if m == 0:
        return {"adjusted_p_values": [], "significant": []}

    p_arr = np.array(p_values, dtype=float)

    if method == "bonferroni":
        adj_p = np.clip(p_arr * m, 0.0, 1.0)
        sig = adj_p < alpha
    elif method == "benjamini_hochberg":
        # Sort p-values ascending
        sorted_indices = np.argsort(p_arr)
        sorted_p = p_arr[sorted_indices]

        # FDR threshold: (k / m) * alpha
        adj_p_sorted = np.zeros(m)
        for k in range(m, 0, -1):
            adj_p_sorted[k - 1] = sorted_p[k - 1] * m / k

        # Enforce monotonicity: p_i <= p_{i+1}
        for k in range(m - 2, -1, -1):
            adj_p_sorted[k] = min(adj_p_sorted[k], adj_p_sorted[k + 1])

        adj_p_sorted = np.clip(adj_p_sorted, 0.0, 1.0)
        adj_p = np.zeros(m)
        adj_p[sorted_indices] = adj_p_sorted
        sig = adj_p < alpha
    else:
        raise ValueError(f"Unknown correction method '{method}'. Choose 'bonferroni' or 'benjamini_hochberg'.")

    return {
        "method": method,
        "nominal_alpha": alpha,
        "adjusted_p_values": [round(float(p), 4) for p in adj_p],
        "significant": [bool(s) for s in sig]
    }


def run_t_test(
    sample_a: Union[np.ndarray, pd.Series],
    sample_b: Union[np.ndarray, pd.Series],
    test_type: str = "independent",
    equal_var: bool = False
) -> Dict[str, Any]:
    """
    Feature 27: Independent and Paired T-Tests.
    For paired tests, index-aligns observations and drops missing pairs simultaneously.
    """
    if test_type == "paired":
        s_a = pd.Series(sample_a)
        s_b = pd.Series(sample_b)
        paired_df = pd.DataFrame({"a": s_a, "b": s_b}).dropna()

        clean_a = paired_df["a"].values.astype(float)
        clean_b = paired_df["b"].values.astype(float)

        if len(clean_a) < 2:
            raise ValueError("Paired T-Test requires at least 2 complete, non-null paired observations.")

        # Reject infinite values
        if np.isinf(clean_a).any() or np.isinf(clean_b).any():
            raise ValueError("Samples contain infinite values.")

        stat, p_val = stats.ttest_rel(clean_a, clean_b)
        test_name = "Paired Student's T-Test"

    elif test_type == "independent":
        clean_a = np.asarray(sample_a)[~pd.isna(sample_a)].astype(float)
        clean_b = np.asarray(sample_b)[~pd.isna(sample_b)].astype(float)

        if len(clean_a) < 2 or len(clean_b) < 2:
            raise ValueError("Each sample must have at least 2 non-null observations.")

        if np.isinf(clean_a).any() or np.isinf(clean_b).any():
            raise ValueError("Samples contain infinite values.")

        stat, p_val = stats.ttest_ind(clean_a, clean_b, equal_var=equal_var)
        test_name = "Welch's T-Test (Unequal Variance)" if not equal_var else "Student's T-Test (Equal Variance)"

    else:
        raise ValueError("test_type must be either 'independent' or 'paired'")

    alpha = 0.05
    significant = bool(p_val < alpha)

    return {
        "test_name": test_name,
        "statistic": round(float(stat), 4),
        "p_value": float(p_val),
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
        "alpha": alpha,
        "is_significant": significant,
        "n_obs_a": len(clean_a),
        "n_obs_b": len(clean_b),
        "mean_sample_a": round(float(np.mean(clean_a)), 4),
        "mean_sample_b": round(float(np.mean(clean_b)), 4),
        "conclusion": (
            "Reject null hypothesis: Significant difference detected between groups (p < 0.05)."
            if significant
            else "Fail to reject null hypothesis: No statistically significant difference detected (p >= 0.05)."
        )
    }


def run_anova(groups: List[np.ndarray], group_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Feature 28: One-Way ANOVA (Analysis of Variance).
    """
    clean_groups = []
    for g in groups:
        arr = np.asarray(g)[~pd.isna(g)].astype(float)
        if len(arr) >= 2 and not np.isinf(arr).any():
            clean_groups.append(arr)

    if len(clean_groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups with >= 2 finite observations each.")

    stat, p_val = stats.f_oneway(*clean_groups)
    alpha = 0.05
    significant = bool(p_val < alpha)

    group_summaries = []
    for idx, g in enumerate(clean_groups):
        name = group_names[idx] if group_names and idx < len(group_names) else f"Group_{idx+1}"
        group_summaries.append({
            "group": name,
            "count": len(g),
            "mean": round(float(np.mean(g)), 4),
            "std": round(float(np.std(g, ddof=1)), 4)
        })

    return {
        "test_name": "One-Way ANOVA (F-Test)",
        "f_statistic": round(float(stat), 4),
        "p_value": float(p_val),
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
        "alpha": alpha,
        "is_significant": significant,
        "groups": group_summaries,
        "conclusion": (
            "Reject null hypothesis: At least one group mean differs significantly from the others (p < 0.05)."
            if significant
            else "Fail to reject null hypothesis: No statistically significant difference among group means (p >= 0.05)."
        )
    }


def run_chi_square(contingency_table: pd.DataFrame) -> Dict[str, Any]:
    """
    Feature 29: Chi-Square Test of Independence.
    """
    stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    alpha = 0.05
    significant = bool(p_val < alpha)

    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns).round(2)

    return {
        "test_name": "Pearson's Chi-Square Test of Independence",
        "chi2_statistic": round(float(stat), 4),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_val),
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
        "alpha": alpha,
        "is_significant": significant,
        "expected_frequencies": expected_df,
        "conclusion": (
            "Reject null hypothesis: Significant association between the two categorical variables (p < 0.05)."
            if significant
            else "Fail to reject null hypothesis: Variables appear independent (p >= 0.05)."
        )
    }


def run_non_parametric_tests(
    groups: List[np.ndarray],
    group_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Feature 30: Mann-Whitney U or Kruskal-Wallis Non-Parametric Tests.
    """
    clean_groups = [np.asarray(g)[~pd.isna(g)].astype(float) for g in groups]
    clean_groups = [g for g in clean_groups if len(g) >= 2 and not np.isinf(g).any()]

    if len(clean_groups) < 2:
        raise ValueError("Non-parametric test requires at least 2 valid groups.")

    alpha = 0.05

    if len(clean_groups) == 2:
        stat, p_val = stats.mannwhitneyu(clean_groups[0], clean_groups[1], alternative="two-sided")
        test_name = "Mann-Whitney U Test"
        significant = bool(p_val < alpha)
        return {
            "test_name": test_name,
            "u_statistic": round(float(stat), 4),
            "p_value": float(p_val),
            "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
            "is_significant": significant,
            "conclusion": (
                "Reject null hypothesis: Distributions of the two groups differ significantly (p < 0.05)."
                if significant
                else "Fail to reject null hypothesis: No significant distributional difference detected (p >= 0.05)."
            )
        }
    else:
        stat, p_val = stats.kruskal(*clean_groups)
        test_name = "Kruskal-Wallis H Test"
        significant = bool(p_val < alpha)
        return {
            "test_name": test_name,
            "h_statistic": round(float(stat), 4),
            "p_value": float(p_val),
            "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
            "is_significant": significant,
            "conclusion": (
                "Reject null hypothesis: Significant difference among population medians across groups (p < 0.05)."
                if significant
                else "Fail to reject null hypothesis: No significant median differences detected (p >= 0.05)."
            )
        }


def fit_distributions(data: np.ndarray) -> pd.DataFrame:
    """
    Feature 31: Distribution Fitting via Kolmogorov-Smirnov Tests.
    """
    clean_data = np.asarray(data)[~pd.isna(data)].astype(float)
    clean_data = clean_data[np.isfinite(clean_data)]

    if len(clean_data) < 10:
        raise ValueError("Distribution fitting requires at least 10 finite observations.")

    results = []

    # 1. Normal Distribution
    norm_params = stats.norm.fit(clean_data)
    norm_dist = stats.norm(*norm_params)
    ks_stat, ks_pval = stats.kstest(clean_data, norm_dist.cdf)
    results.append({
        "Distribution": "Normal (Gaussian)",
        "KS_Statistic": round(float(ks_stat), 4),
        "P_Value": round(float(ks_pval), 4),
        "Fitted_Parameters": f"Mean={norm_params[0]:.2f}, Std={norm_params[1]:.2f}",
        "Statistical_Assessment": (
            "No strong evidence against this distribution under this test"
            if ks_pval > 0.05 else "Significant deviation from hypothesized distribution"
        )
    })

    # 2. Exponential Distribution (non-negative data)
    if (clean_data >= 0).all():
        exp_params = stats.expon.fit(clean_data)
        exp_dist = stats.expon(*exp_params)
        ks_stat, ks_pval = stats.kstest(clean_data, exp_dist.cdf)
        results.append({
            "Distribution": "Exponential",
            "KS_Statistic": round(float(ks_stat), 4),
            "P_Value": round(float(ks_pval), 4),
            "Fitted_Parameters": f"Loc={exp_params[0]:.2f}, Scale={exp_params[1]:.2f}",
            "Statistical_Assessment": (
                "No strong evidence against this distribution under this test"
                if ks_pval > 0.05 else "Significant deviation from hypothesized distribution"
            )
        })

    # 3. Uniform Distribution
    uni_params = stats.uniform.fit(clean_data)
    uni_dist = stats.uniform(*uni_params)
    ks_stat, ks_pval = stats.kstest(clean_data, uni_dist.cdf)
    results.append({
        "Distribution": "Uniform",
        "KS_Statistic": round(float(ks_stat), 4),
        "P_Value": round(float(ks_pval), 4),
        "Fitted_Parameters": f"Min={uni_params[0]:.2f}, Width={uni_params[1]:.2f}",
        "Statistical_Assessment": (
            "No strong evidence against this distribution under this test"
            if ks_pval > 0.05 else "Significant deviation from hypothesized distribution"
        )
    })

    # 4. Log-Normal Distribution
    if (clean_data > 0).all():
        try:
            lognorm_params = stats.lognorm.fit(clean_data)
            lognorm_dist = stats.lognorm(*lognorm_params)
            ks_stat, ks_pval = stats.kstest(clean_data, lognorm_dist.cdf)
            results.append({
                "Distribution": "Log-Normal",
                "KS_Statistic": round(float(ks_stat), 4),
                "P_Value": round(float(ks_pval), 4),
                "Fitted_Parameters": f"Shape={lognorm_params[0]:.2f}, Scale={lognorm_params[2]:.2f}",
                "Statistical_Assessment": (
                    "No strong evidence against this distribution under this test"
                    if ks_pval > 0.05 else "Significant deviation from hypothesized distribution"
                )
            })
        except Exception:
            pass

    return pd.DataFrame(results).sort_values(by="KS_Statistic").reset_index(drop=True)
