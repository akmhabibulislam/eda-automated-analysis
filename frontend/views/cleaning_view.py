"""
Workspace 2 View: Data Quality, Recommendations & Controlled Cleaning.
"""

import streamlit as st
import pandas as pd
import numpy as np

from backend.cleaning.cleaning import (
    standardize_column_headers,
    impute_missing_values,
    drop_missing_values,
    remove_duplicates,
    treat_outliers,
    clean_text_columns,
    cast_data_types,
    analyze_cleaning_recommendations,
    run_automated_cleaning
)
from backend.core.lineage import DatasetSessionManager


def render_cleaning_view(session_manager: DatasetSessionManager):
    st.subheader("Data Quality, Recommendations & Controlled Cleaning")
    st.caption("Review automated recommendations, configure controlled cleaning pipelines, and inspect data lineage without blind mutations.")

    df = session_manager.current_df
    if df is None:
        st.info("Please load a dataset in Workspace 1 first.")
        return

    # Quality Recommendations Card
    st.markdown("#### Automated Quality Recommendations")
    recs = analyze_cleaning_recommendations(df)
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.write(f"**Duplicate Rows**: {recs['duplicate_rows']}")
        st.write(f"**Columns with Missing Values**: {len(recs['missing_columns'])}")
        if recs["missing_columns"]:
            st.caption(f"Missing Columns: {list(recs['missing_columns'].keys())}")
    with r_col2:
        st.write(f"**Header Standardization Needed**: {'Yes' if recs['header_standardization_needed'] else 'No'}")
        st.write(f"**Outlier Candidate Columns**: {len(recs['outlier_candidates'])}")
        if recs["outlier_candidates"]:
            st.caption(f"Outlier Columns: {list(recs['outlier_candidates'].keys())}")

    st.markdown("---")
    st.markdown("#### Controlled Auto-Clean Pipeline")
    st.write("Configure which operations to execute (imputation and outlier capping are strictly opt-in).")

    cfg_c1, cfg_c2, cfg_c3, cfg_c4, cfg_c5 = st.columns(5)
    with cfg_c1:
        opt_headers = st.checkbox("Standardize Headers", value=True)
    with cfg_c2:
        opt_dups = st.checkbox("Purge Duplicates", value=True)
    with cfg_c3:
        opt_strings = st.checkbox("Trim String Spaces", value=True)
    with cfg_c4:
        opt_impute = st.checkbox("Impute Missing (Median/Mode)", value=False)
    with cfg_c5:
        opt_outliers = st.checkbox("Cap Outliers (IQR 3.0)", value=False)

    if st.button("Execute Controlled Cleaning Pipeline", type="primary"):
        with st.spinner("Executing configured pipeline..."):
            cleaned_df, summary = run_automated_cleaning(
                df,
                impute_missing=opt_impute,
                cap_outliers=opt_outliers,
                standardize_headers=opt_headers,
                purge_duplicates=opt_dups,
                clean_strings=opt_strings
            )
            session_manager.update_current_df(
                new_df=cleaned_df,
                operation_name="Controlled Auto-Clean",
                parameters={"impute": opt_impute, "outliers": opt_outliers, "headers": opt_headers, "duplicates": opt_dups},
                description=f"Executed controlled clean with {summary['steps_executed']} operations."
            )
            st.success(f"Cleaning complete. {summary['steps_executed']} operations recorded.")
            st.rerun()

    st.markdown("---")
    q_tab1, q_tab2, q_tab3, q_tab4, q_tab5, q_tab6 = st.tabs([
        "Missing Imputation", "Duplicate Purge", "Outlier Treatment", "Header & Text Sanitization", "Data Type Cast", "Lineage Audit Trail"
    ])

    with q_tab1:
        st.markdown("##### Impute or Drop Missing Data")
        c_nulls = [c for c in df.columns if df[c].isna().sum() > 0]
        col_imp, col_drop = st.columns(2)
        with col_imp:
            st.markdown("**Imputation Engine**")
            sel_imp_cols = st.multiselect("Columns to Impute", df.columns, default=c_nulls)
            strat = st.selectbox("Strategy", ["mean", "median", "mode", "constant", "ffill", "bfill"])
            const_val = st.text_input("Constant Value", "0") if strat == "constant" else None

            if st.button("Execute Imputation"):
                cv = float(const_val) if const_val and const_val.replace(".", "").isdigit() else const_val
                imp_df = impute_missing_values(df, columns=sel_imp_cols, strategy=strat, fill_value=cv)
                session_manager.update_current_df(
                    new_df=imp_df,
                    operation_name=f"Imputation ({strat})",
                    parameters={"strategy": strat, "columns": sel_imp_cols},
                    description=f"Imputed missing entries using strategy: {strat}",
                    columns_affected=sel_imp_cols
                )
                st.success("Imputed missing values.")
                st.rerun()

        with col_drop:
            st.markdown("**Threshold Removal (math.ceil safety)**")
            c_th = st.slider("Drop column if missing > X%", 10, 100, 50)
            r_th = st.slider("Drop row if missing > X%", 10, 100, 50)
            if st.button("Drop Missing by Threshold"):
                dropped_df = drop_missing_values(df, row_threshold_pct=float(r_th), col_threshold_pct=float(c_th))
                session_manager.update_current_df(
                    new_df=dropped_df,
                    operation_name="Drop Missing Threshold",
                    parameters={"row_threshold": r_th, "col_threshold": c_th},
                    description=f"Dropped missing data exceeding threshold row:{r_th}%, col:{c_th}%."
                )
                st.success("Dropped missing rows/columns.")
                st.rerun()

    with q_tab2:
        st.markdown("##### Duplicate Detection and Purging")
        sub_dup = st.multiselect("Subset columns for duplicate evaluation", df.columns)
        k_rule = st.selectbox("Retention Rule", ["first", "last", False], format_func=lambda x: "Drop all duplicates" if x is False else f"Keep {x}")
        if st.button("Purge Duplicates"):
            dedup_df = remove_duplicates(df, subset=sub_dup if sub_dup else None, keep=k_rule)
            session_manager.update_current_df(
                new_df=dedup_df,
                operation_name="Remove Duplicates",
                parameters={"subset": sub_dup, "keep": k_rule},
                description=f"Purged duplicate rows ({'subset ' + str(sub_dup) if sub_dup else 'entire row match'})."
            )
            st.success("Duplicates purged.")
            st.rerun()

    with q_tab3:
        st.markdown("##### Outlier Detection & Treatment")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            o_col = st.selectbox("Target Metric", num_cols)
            o_meth = st.selectbox("Method", ["iqr", "zscore"])
            o_th = st.number_input("Threshold Multiplier", min_value=1.0, max_value=5.0, value=1.5 if o_meth == "iqr" else 3.0, step=0.1)
            o_act = st.selectbox("Action", ["cap", "drop", "flag"])

            if st.button("Execute Outlier Treatment"):
                treated_df, info = treat_outliers(df, column=o_col, method=o_meth, threshold=float(o_th), action=o_act)
                session_manager.update_current_df(
                    new_df=treated_df,
                    operation_name=f"Outlier Treatment ({o_meth.upper()})",
                    parameters={"column": o_col, "threshold": o_th, "action": o_act},
                    description=f"Outlier action '{o_act}' on column '{o_col}' ({info['outliers_detected']} detected).",
                    columns_affected=[o_col]
                )
                st.success(f"Processed {info['outliers_detected']} anomalies outside bounds.")
                st.rerun()
        else:
            st.info("No numeric columns available.")

    with q_tab4:
        st.markdown("##### Header Standardization & Text Sanitization")
        h_col, t_col = st.columns(2)
        with h_col:
            st.markdown("**Collision-Guarded Header Formatting**")
            c_style = st.selectbox("Case Convention", ["snake_case", "lower_case", "upper_case", "camelCase"])
            if st.button("Standardize Headers"):
                std_df, mapping = standardize_column_headers(df, case_style=c_style)
                session_manager.update_current_df(
                    new_df=std_df,
                    operation_name="Standardize Headers",
                    parameters={"case_style": c_style},
                    description=f"Formatted columns to {c_style} with collision guards. Mapped {len(mapping)} columns."
                )
                st.success("Headers formatted safely.")
                st.json(mapping)
                st.rerun()

        with t_col:
            st.markdown("**Text String Cleaning (Nulls Strictly Preserved)**")
            str_cols = [c for c in df.columns if isinstance(df[c].dtype, (pd.CategoricalDtype, pd.StringDtype)) or df[c].dtype == "object"]
            t_cols_sel = st.multiselect("Text Columns to Clean", str_cols)
            tr_ws = st.checkbox("Trim Whitespace", value=True)
            tr_case = st.selectbox("Case Adjustment", ["None", "lower", "upper", "title"])
            tr_sp = st.checkbox("Strip Special Characters", value=False)

            if st.button("Sanitize Text Columns"):
                clean_txt_df = clean_text_columns(
                    df,
                    columns=t_cols_sel,
                    strip_whitespace=tr_ws,
                    case_transformation=None if tr_case == "None" else tr_case,
                    remove_special_chars=tr_sp
                )
                session_manager.update_current_df(
                    new_df=clean_txt_df,
                    operation_name="Clean Text Columns",
                    parameters={"columns": t_cols_sel, "trim": tr_ws, "case": tr_case, "strip_special": tr_sp},
                    description=f"Cleaned string formatting for {t_cols_sel} without stringifying nulls.",
                    columns_affected=t_cols_sel
                )
                st.success("Text sanitized.")
                st.rerun()

    with q_tab5:
        st.markdown("##### Data Type Casting (with Unparsed Token Reporting)")
        cast_col = st.selectbox("Column to Cast", df.columns)
        target_t = st.selectbox("Target Datatype", ["int", "float", "string", "datetime", "boolean", "category"])
        if st.button("Apply Datatype Cast"):
            try:
                cast_df, unparsed = cast_data_types(df, {cast_col: target_t})
                session_manager.update_current_df(
                    new_df=cast_df,
                    operation_name="Cast Data Type",
                    parameters={"column": cast_col, "target_type": target_t},
                    description=f"Cast column '{cast_col}' to {target_t}. Coerced unparsed tokens: {unparsed.get(cast_col, 0)}.",
                    columns_affected=[cast_col],
                    dropped_null_count=unparsed.get(cast_col, 0)
                )
                st.success(f"Converted '{cast_col}' to {target_t}.")
                if unparsed.get(cast_col, 0) > 0:
                    st.warning(f"Note: {unparsed[cast_col]} unparseable values were safely coerced to NaN/NaT.")
                st.rerun()
            except Exception as e:
                st.error(f"Cast failed: {str(e)}")

    with q_tab6:
        st.markdown("##### Structured Lineage & Audit History")
        lineage_df = session_manager.get_lineage_dataframe()
        st.dataframe(lineage_df, use_container_width=True)

        if st.button("Revert to Immutable Baseline Dataset"):
            session_manager.reset_to_original()
            st.success("Reverted to baseline original dataset.")
            st.rerun()
