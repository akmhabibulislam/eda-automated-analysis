"""
Workspace 4 View: Statistics & Hypothesis Testing.
"""

import streamlit as st
import pandas as pd
import numpy as np

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
    run_non_parametric_tests,
    fit_distributions,
    apply_multiple_testing_correction
)
from backend.core.lineage import DatasetSessionManager


def render_statistics_view(session_manager: DatasetSessionManager):
    st.subheader("Statistical Analysis & Hypothesis Validation")
    st.caption("Descriptive statistics, pairwise correlations, VIF multicollinearity, parametric/non-parametric tests, and distribution fitting.")

    df = session_manager.current_df
    if df is None:
        st.info("Please load a dataset in Workspace 1 first.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if isinstance(df[c].dtype, (pd.CategoricalDtype, pd.StringDtype)) or df[c].dtype == "object"]

    from backend.ingestion.loaders import detect_schema
    schema_info = detect_schema(df)
    semantic_roles = schema_info.get("semantic_roles", {})
    # Filter raw identifiers out of analytical numerical candidates
    analytical_num_cols = [
        c for c in num_cols if semantic_roles.get(c) not in ["identifier", "uuid"]
    ]
    if not analytical_num_cols:
        analytical_num_cols = num_cols

    s_tab1, s_tab2, s_tab3, s_tab4, s_tab5, s_tab6 = st.tabs([
        "Descriptive Stats", "Correlations & VIF", "T-Tests & ANOVA", "Categorical Tests", "Distribution Fitting", "Multiple Testing Correction"
    ])

    with s_tab1:
        st.markdown("##### Univariate Descriptive Statistics")
        uni_df = compute_univariate_summary(df)
        if not uni_df.empty:
            st.dataframe(uni_df, use_container_width=True)
        st.markdown("##### Skewness & Kurtosis Distribution Symmetry")
        sk_df = compute_skewness_kurtosis(df)
        if not sk_df.empty:
            st.dataframe(sk_df, use_container_width=True)

    with s_tab2:
        st.markdown("##### Correlation Matrix with Sample Coverage (N used)")
        st.caption("Displays pairwise coefficients alongside exact non-null observation counts to surface pairwise deletion bias.")
        c_meth = st.selectbox("Method", ["pearson", "spearman", "kendall"])
        from backend.analysis.statistics import compute_correlation_matrix_with_sample_sizes
        corr_m, n_m = compute_correlation_matrix_with_sample_sizes(df[analytical_num_cols], method=c_meth)
        if not corr_m.empty:
            st.markdown("**Correlation Coefficients**")
            st.dataframe(corr_m, use_container_width=True)
            st.markdown("**Pairwise Sample Size (N observations used)**")
            st.dataframe(n_m, use_container_width=True)

        st.markdown("##### Highly Correlated Feature Pairs (Pairwise Correlation >= 0.80)")
        corr_pairs = detect_highly_correlated_pairs(df[analytical_num_cols], threshold=0.80)
        if corr_pairs:
            for w in corr_pairs:
                st.warning(f"{w['severity']} Correlation: `{w['column_1']}` and `{w['column_2']}` (r={w['correlation']}, N={w.get('n_observations')})")
        else:
            st.success("No feature pairs exceed the 0.80 correlation threshold.")

        st.markdown("##### Variance Inflation Factor (VIF) Multi-Collinearity Calculation")
        vif_df = compute_variance_inflation_factors(df[analytical_num_cols])
        if not vif_df.empty:
            st.dataframe(vif_df, use_container_width=True)

    with s_tab3:
        st.markdown("##### Parametric Hypothesis Testing (T-Tests & One-Way ANOVA)")
        st.caption("Includes configurable significance threshold (alpha), effect sizes (Cohen's d, Eta-squared), confidence intervals, and Tukey HSD post-hoc testing.")
        alpha_val = st.selectbox("Significance Level (Alpha)", [0.05, 0.01, 0.10], index=0)

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown("**T-Test Analysis (Paired & Independent)**")
            if len(analytical_num_cols) >= 2:
                tt_s1 = st.selectbox("Sample 1", analytical_num_cols, index=0)
                tt_s2 = st.selectbox("Sample 2", analytical_num_cols, index=1)
                tt_type = st.selectbox("Design", ["independent", "paired"])
                if st.button("Run T-Test"):
                    try:
                        res_tt = run_t_test(df[tt_s1], df[tt_s2], test_type=tt_type, alpha=alpha_val)
                        st.json(res_tt)
                    except Exception as e:
                        st.error(f"T-Test error: {str(e)}")

        with t_col2:
            st.markdown("**ANOVA (F-Test) & Tukey HSD**")
            if analytical_num_cols and cat_cols:
                an_val = st.selectbox("Metric", analytical_num_cols)
                an_grp = st.selectbox("Group Factor", cat_cols)
                if st.button("Run ANOVA"):
                    try:
                        groups = [group[an_val].values for _, group in df.groupby(an_grp, observed=False)]
                        res_an = run_anova(groups, alpha=alpha_val)
                        st.json({k: v for k, v in res_an.items() if k != "post_hoc_tukey"})
                        if res_an.get("post_hoc_tukey") is not None:
                            st.markdown("**Tukey HSD Post-Hoc Pairwise Comparisons**")
                            st.dataframe(res_an["post_hoc_tukey"], use_container_width=True)
                    except Exception as e:
                        st.error(f"ANOVA error: {str(e)}")

    with s_tab4:
        st.markdown("##### Chi-Square & Non-Parametric Tests")
        c1_test, c2_test = st.columns(2)
        with c1_test:
            st.markdown("**Chi-Square Independence**")
            if len(cat_cols) >= 2:
                c_x1 = st.selectbox("Variable 1", cat_cols, index=0)
                c_x2 = st.selectbox("Variable 2", cat_cols, index=1)
                if st.button("Execute Chi-Square"):
                    ct = pd.crosstab(df[c_x1], df[c_x2])
                    res_chi = run_chi_square(ct)
                    st.json({k: v for k, v in res_chi.items() if k != "expected_frequencies"})
                    st.dataframe(res_chi["expected_frequencies"])

        with c2_test:
            st.markdown("**Non-Parametric Tests (Mann-Whitney / Kruskal-Wallis)**")
            if num_cols and cat_cols:
                np_val = st.selectbox("Non-Parametric Metric", num_cols)
                np_grp = st.selectbox("Grouping Factor", cat_cols)
                if st.button("Run Non-Parametric"):
                    try:
                        groups = [group[np_val].values for _, group in df.groupby(np_grp, observed=False)]
                        res_np = run_non_parametric_tests(groups)
                        st.json(res_np)
                    except Exception as e:
                        st.error(f"Non-parametric error: {str(e)}")

    with s_tab5:
        st.markdown("##### Distribution Fitting (Kolmogorov-Smirnov)")
        if num_cols:
            dist_col = st.selectbox("Metric to Fit", num_cols)
            if st.button("Evaluate Distribution Fits"):
                try:
                    fit_tbl = fit_distributions(df[dist_col].values)
                    st.dataframe(fit_tbl, use_container_width=True)
                except Exception as e:
                    st.error(f"Fitting error: {str(e)}")

    with s_tab6:
        st.markdown("##### Multiple Testing Corrections (FDR & Bonferroni)")
        st.caption("Enter a comma-separated list of raw p-values from multiple comparisons to calculate corrected significance thresholds.")
        p_raw_input = st.text_input("Raw P-Values (comma-separated)", "0.001, 0.012, 0.035, 0.048, 0.082, 0.15")
        corr_method = st.selectbox("Correction Method", ["benjamini_hochberg", "bonferroni"])
        alpha_val = st.number_input("Significance Threshold (Alpha)", min_value=0.001, max_value=0.20, value=0.05, step=0.01)

        if st.button("Apply Multiple Testing Correction"):
            try:
                p_list = [float(p.strip()) for p in p_raw_input.split(",") if p.strip()]
                adj_results = apply_multiple_testing_correction(p_list, method=corr_method, alpha=float(alpha_val))
                corr_df = pd.DataFrame({
                    "Comparison": [f"Test {i+1}" for i in range(len(p_list))],
                    "Raw_P_Value": p_list,
                    "Adjusted_P_Value": adj_results["adjusted_p_values"],
                    "Significant_After_Correction": adj_results["significant"]
                })
                st.dataframe(corr_df, use_container_width=True)
            except Exception as e:
                st.error(f"Correction error: {str(e)}")
