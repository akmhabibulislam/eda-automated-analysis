"""
Workspace 3 View: Advanced Data Wrangling & Reshaping.
"""

import streamlit as st
import pandas as pd
import numpy as np

from backend.cleaning.wrangling import (
    add_custom_formula_column,
    bin_continuous_column,
    aggregate_groupby,
    extract_regex_patterns,
    pivot_dataframe,
    unpivot_dataframe,
    filter_rows_structured
)
from backend.core.lineage import DatasetSessionManager


def render_wrangling_view(session_manager: DatasetSessionManager):
    st.subheader("Advanced Data Wrangling & Reshaping")
    st.caption("Secure AST formula evaluation, continuous binning, SQL-like aggregations, regex pattern extraction, and structured filtering.")

    df = session_manager.current_df
    if df is None:
        st.info("Please load a dataset in Workspace 1 first.")
        return

    w_tab1, w_tab2, w_tab3, w_tab4, w_tab5, w_tab6 = st.tabs([
        "Secure Formula Builder", "Continuous Binning", "SQL Aggregation", "Regex Extraction", "Pivot / Reshape", "Structured Filter"
    ])

    with w_tab1:
        st.markdown("##### Secure AST Formula Column Builder")
        st.caption("Evaluates mathematical expressions safely using an abstract syntax tree parser without code injection risks.")
        new_col = st.text_input("New Column Name", "calculated_metric")
        form_expr = st.text_input("Formula Expression (e.g. `revenue - cost` or `sqrt(units_sold) * 10`)", "")
        st.write(f"Available columns: `{', '.join(df.columns)}`")

        if st.button("Evaluate Formula"):
            try:
                mod_df = add_custom_formula_column(df, new_col, form_expr)
                session_manager.update_current_df(
                    new_df=mod_df,
                    operation_name="Custom Formula (AST)",
                    parameters={"new_column": new_col, "expression": form_expr},
                    description=f"Added calculated column '{new_col}' via AST expression: {form_expr}",
                    columns_affected=[new_col]
                )
                st.success(f"Created '{new_col}'.")
                st.rerun()
            except Exception as e:
                st.error(f"Formula evaluation error: {str(e)}")

    with w_tab2:
        st.markdown("##### Continuous Binning & Discretization")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            b_col = st.selectbox("Continuous Metric", num_cols)
            b_name = st.text_input("Binned Column Name", f"{b_col}_binned")
            b_mode = st.selectbox("Binning Method", ["equal_width", "quantile"])
            b_cnt = st.slider("Number of Bins", 2, 20, 5)

            if st.button("Apply Binning"):
                bin_df = bin_continuous_column(df, column=b_col, new_column_name=b_name, bins=b_cnt, bin_type=b_mode)
                session_manager.update_current_df(
                    new_df=bin_df,
                    operation_name="Discretization / Binning",
                    parameters={"column": b_col, "bins": b_cnt, "mode": b_mode},
                    description=f"Binned '{b_col}' into {b_cnt} discrete ranges ({b_mode}).",
                    columns_affected=[b_name]
                )
                st.success(f"Binned '{b_col}' into '{b_name}'.")
                st.rerun()
        else:
            st.info("No numeric columns available.")

    with w_tab3:
        st.markdown("##### SQL-like Groupby & Aggregations")
        grp_by = st.multiselect("Group Columns", df.columns)
        agg_metric = st.selectbox("Metric to Aggregate", df.select_dtypes(include=[np.number]).columns.tolist())
        agg_funcs = st.multiselect("Functions", ["mean", "sum", "count", "std", "min", "max", "median"], default=["mean", "sum"])

        if grp_by and agg_metric and agg_funcs and st.button("Run Aggregations"):
            try:
                res_agg = aggregate_groupby(df, group_columns=grp_by, aggregations={agg_metric: agg_funcs})
                st.dataframe(res_agg, use_container_width=True)
            except Exception as e:
                st.error(f"Aggregation error: {str(e)}")

    with w_tab4:
        st.markdown("##### Text Regex Extraction")
        text_cols = [c for c in df.columns if isinstance(df[c].dtype, (pd.CategoricalDtype, pd.StringDtype)) or df[c].dtype == "object"]
        if text_cols:
            rx_src = st.selectbox("Source Column", text_cols)
            rx_pat = st.text_input("Regex Pattern", r"([A-Za-z0-9]+)")
            rx_dest = st.text_input("Destination Column", f"{rx_src}_extracted")

            if st.button("Extract Regex"):
                try:
                    rx_df = extract_regex_patterns(df, source_column=rx_src, pattern=rx_pat, new_column_name=rx_dest)
                    session_manager.update_current_df(
                        new_df=rx_df,
                        operation_name="Regex Extraction",
                        parameters={"source": rx_src, "pattern": rx_pat, "destination": rx_dest},
                        description=f"Extracted regex pattern '{rx_pat}' from '{rx_src}' into '{rx_dest}'.",
                        columns_affected=[rx_dest]
                    )
                    st.success(f"Pattern extracted into '{rx_dest}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Regex error: {str(e)}")
        else:
            st.info("No text columns found.")

    with w_tab5:
        st.markdown("##### Reshaping: Pivot & Unpivot (with Cardinality Protection)")
        resh_mode = st.radio("Operation", ["Pivot (Long to Wide)", "Unpivot / Melt (Wide to Long)"])
        if resh_mode == "Pivot (Long to Wide)":
            p_idx = st.multiselect("Index Column(s)", df.columns)
            p_cols = st.selectbox("Pivot Column (Headers)", df.columns)
            p_vals = st.selectbox("Values Column", df.select_dtypes(include=[np.number]).columns.tolist())
            p_agg = st.selectbox("Aggregation", ["mean", "sum", "count", "min", "max"])

            if p_idx and p_cols and p_vals and st.button("Execute Pivot"):
                try:
                    p_res = pivot_dataframe(df, index_cols=p_idx, columns=p_cols, values=p_vals, aggfunc=p_agg)
                    st.dataframe(p_res, use_container_width=True)
                except Exception as e:
                    st.error(f"Pivot error: {str(e)}")
        else:
            u_id = st.multiselect("Fixed Identifiers", df.columns)
            u_val = st.multiselect("Value Columns to Unpivot", [c for c in df.columns if c not in u_id])
            if u_id and u_val and st.button("Execute Unpivot"):
                u_res = unpivot_dataframe(df, id_vars=u_id, value_vars=u_val)
                st.dataframe(u_res, use_container_width=True)

    with w_tab6:
        st.markdown("##### Structured Row Query Filter")
        st.caption("Applies typed, parametric comparisons safely without unrestricted expression evaluation.")
        q_col = st.selectbox("Filter Column", df.columns)
        q_op = st.selectbox("Operator", ["==", "!=", ">", ">=", "<", "<=", "contains", "in"])
        q_val = st.text_input("Comparison Value", "")

        if q_val and st.button("Apply Structured Filter"):
            try:
                flt_df = filter_rows_structured(df, column=q_col, operator=q_op, comparison_value=q_val)
                session_manager.update_current_df(
                    new_df=flt_df,
                    operation_name="Filter Rows (Structured)",
                    parameters={"column": q_col, "operator": q_op, "value": q_val},
                    description=f"Filtered dataset where {q_col} {q_op} '{q_val}'.",
                    columns_affected=[q_col]
                )
                st.success("Filter applied.")
                st.rerun()
            except Exception as e:
                st.error(f"Filter error: {str(e)}")
