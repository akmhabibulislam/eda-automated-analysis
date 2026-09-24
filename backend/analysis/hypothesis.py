"""
Statistical Testing & Hypothesis Validation module.
Features 27-31:
27. T-Tests (independent and paired)
28. ANOVA (Analysis of Variance)
29. Chi-Square Test of Independence
30. Mann-Whitney U / Kruskal-Wallis Non-Parametric Tests
31. Distribution Fitting (Normal, Poisson, Exponential, Uniform evaluation)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats


def run_t_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    test_type: str = "independent",
    equal_var: bool = False
) -> Dict[str, Any]:
    """
    Feature 27: Independent and Paired T-Tests.
    """
    clean_a = sample_a[~np.isnan(sample_a)]
    clean_b = sample_b[~np.isnan(sample_b)]

    if len(clean_a) < 2 or len(clean_b) < 2:
        raise ValueError("Each sample must have at least 2 non-null observations.")

    if test_type == "independent":
        stat, p_val = stats.ttest_ind(clean_a, clean_b, equal_var=equal_var)
        test_name = "Welch's T-Test" if not equal_var else "Student's Independent T-Test"
    elif test_type == "paired":
        if len(clean_a) != len(clean_b):
            min_len = min(len(clean_a), len(clean_b))
            clean_a = clean_a[:min_len]
            clean_b = clean_b[:min_len]
        stat, p_val = stats.ttest_rel(clean_a, clean_b)
        test_name = "Paired Sample T-Test"
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
        "mean_sample_a": round(float(np.mean(clean_a)), 4),
        "mean_sample_b": round(float(np.mean(clean_b)), 4),
        "conclusion": (
            "Reject null hypothesis: Significant difference between groups (p < 0.05)."
            if significant
            else "Fail to reject null hypothesis: No statistically significant difference detected (p >= 0.05)."
        )
    }


def run_anova(groups: List[np.ndarray], group_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Feature 28: One-Way ANOVA (Analysis of Variance).
    """
    clean_groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) >= 2]
    if len(clean_groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups with >= 2 observations each.")

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
    Feature 30: Mann-Whitney U (2 groups) or Kruskal-Wallis (> 2 groups) Non-Parametric Tests.
    """
    clean_groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) >= 2]
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
    Feature 31: Distribution Fitting (Normal, Exponential, Uniform, Log-Normal).
    Computes Kolmogorov-Smirnov test statistics and p-values using explicit CDF instances.
    """
    clean_data = data[~np.isnan(data)]
    if len(clean_data) < 10:
        raise ValueError("Distribution fitting requires at least 10 observations.")

    results = []

    # 1. Normal Distribution
    norm_params = stats.norm.fit(clean_data)
    norm_cdf = stats.norm(*norm_params).cdf
    ks_stat, ks_pval = stats.kstest(clean_data, norm_cdf)
    results.append({
        "Distribution": "Normal (Gaussian)",
        "KS_Statistic": round(float(ks_stat), 4),
        "P_Value": round(float(ks_pval), 4),
        "Fitted_Parameters": f"μ={norm_params[0]:.2f}, σ={norm_params[1]:.2f}",
        "Fit_Quality": "Good" if ks_pval > 0.05 else "Poor"
    })

    # 2. Exponential Distribution (non-negative data)
    if (clean_data >= 0).all():
        exp_params = stats.expon.fit(clean_data)
        exp_cdf = stats.expon(*exp_params).cdf
        ks_stat, ks_pval = stats.kstest(clean_data, exp_cdf)
        results.append({
            "Distribution": "Exponential",
            "KS_Statistic": round(float(ks_stat), 4),
            "P_Value": round(float(ks_pval), 4),
            "Fitted_Parameters": f"loc={exp_params[0]:.2f}, scale={exp_params[1]:.2f}",
            "Fit_Quality": "Good" if ks_pval > 0.05 else "Poor"
        })

    # 3. Uniform Distribution
    uni_params = stats.uniform.fit(clean_data)
    uni_cdf = stats.uniform(*uni_params).cdf
    ks_stat, ks_pval = stats.kstest(clean_data, uni_cdf)
    results.append({
        "Distribution": "Uniform",
        "KS_Statistic": round(float(ks_stat), 4),
        "P_Value": round(float(ks_pval), 4),
        "Fitted_Parameters": f"min={uni_params[0]:.2f}, width={uni_params[1]:.2f}",
        "Fit_Quality": "Good" if ks_pval > 0.05 else "Poor"
    })

    # 4. Log-Normal Distribution (strictly positive)
    if (clean_data > 0).all():
        try:
            lognorm_params = stats.lognorm.fit(clean_data)
            lognorm_cdf = stats.lognorm(*lognorm_params).cdf
            ks_stat, ks_pval = stats.kstest(clean_data, lognorm_cdf)
            results.append({
                "Distribution": "Log-Normal",
                "KS_Statistic": round(float(ks_stat), 4),
                "P_Value": round(float(ks_pval), 4),
                "Fitted_Parameters": f"s={lognorm_params[0]:.2f}, scale={lognorm_params[2]:.2f}",
                "Fit_Quality": "Good" if ks_pval > 0.05 else "Poor"
            })
        except Exception:
            pass

    res_df = pd.DataFrame(results)
    return res_df.sort_values(by="KS_Statistic").reset_index(drop=True)
