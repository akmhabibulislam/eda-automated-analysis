"""
Workspace 5 View: Time-Series & Mathematical Modeling.
"""

import streamlit as st
import pandas as pd
import numpy as np

from backend.analysis.timeseries import (
    resample_temporal_data,
    compute_rolling_metrics,
    decompose_seasonality_trend
)
from backend.modeling.modeling import (
    fit_curve_and_equation,
    run_symbolic_regression,
    run_kmeans_clustering,
    run_pca_reduction,
    run_isolation_forest_anomaly_detection
)
from frontend.components.visualizations import create_relationship_chart
from backend.core.lineage import DatasetSessionManager


def render_modeling_view(session_manager: DatasetSessionManager, register_fig_callback):
    st.subheader("Time-Series & Mathematical Modeling")
    st.caption("Temporal decomposition, curve fitting trendlines, parametric function-family search, clustering with inverse centers, and PCA loadings.")

    df = session_manager.current_df
    if df is None:
        st.info("Please load a dataset in Workspace 1 first.")
        return

    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    raw_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    from backend.ingestion.loaders import detect_schema
    schema_info = detect_schema(df)
    semantic_roles = schema_info.get("semantic_roles", {})
    # Exclude identifiers and UUIDs from modeling, clustering, and PCA
    num_cols = [c for c in raw_num_cols if semantic_roles.get(c) not in ["identifier", "uuid"]]
    if not num_cols:
        num_cols = raw_num_cols

    m_tab1, m_tab2, m_tab3, m_tab4, m_tab5 = st.tabs([
        "Time-Series Analysis", "Curve Fitting", "Parametric Function Search", "K-Means & PCA", "Isolation Forest"
    ])

    with m_tab1:
        st.markdown("##### Temporal Aggregations & Seasonal Decomposition")
        if not dt_cols:
            st.info("No datetime column detected.")
            cand_dt = st.selectbox("Select column to cast as Datetime", df.columns)
            if st.button("Parse as Datetime"):
                try:
                    ts_df = df.copy()
                    ts_df[cand_dt] = pd.to_datetime(ts_df[cand_dt], errors="coerce")
                    session_manager.update_current_df(
                        new_df=ts_df,
                        operation_name="Parse Datetime",
                        parameters={"column": cand_dt},
                        description=f"Parsed column '{cand_dt}' as Datetime."
                    )
                    st.success("Parsed column as Datetime.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Parse error: {str(e)}")
        else:
            ts_dt = st.selectbox("Timestamp Column", dt_cols)
            ts_val = st.selectbox("Series Metric", num_cols)

            col_res, col_roll = st.columns(2)
            with col_res:
                st.markdown("**Resampling**")
                f_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE", "Annual": "YE"}
                f_choice = st.selectbox("Frequency", list(f_map.keys()))
                f_agg = st.selectbox("Aggregation", ["sum", "mean", "max", "min"])
                if st.button("Execute Resampling"):
                    try:
                        res_df = resample_temporal_data(df, ts_dt, ts_val, frequency=f_map[f_choice], aggregation=f_agg)
                        fig_ts = create_relationship_chart(res_df, "date", f"{ts_val}_{f_agg}", chart_type="line")
                        st.plotly_chart(fig_ts, use_container_width=True)
                        register_fig_callback(fig_ts)
                    except Exception as e:
                        st.error(f"Resampling error: {str(e)}")

            with col_roll:
                st.markdown("**Moving Averages**")
                w_size = st.slider("Moving Window", 2, 60, 7)
                if st.button("Rolling Metrics"):
                    try:
                        r_df = compute_rolling_metrics(df, ts_val, date_column=ts_dt, window_size=w_size)
                        fig_roll = create_relationship_chart(r_df, ts_dt, f"{ts_val}_rolling_mean_{w_size}", chart_type="line")
                        st.plotly_chart(fig_roll, use_container_width=True)
                    except Exception as e:
                        st.error(f"Rolling error: {str(e)}")

    with m_tab2:
        st.markdown("##### Curve Fitting & Trendline Equations")
        if len(num_cols) >= 2:
            cf_x = st.selectbox("Predictor (X)", num_cols, index=0)
            cf_y = st.selectbox("Response (Y)", num_cols, index=1)
            cf_mode = st.selectbox("Model", ["linear", "quadratic", "exponential", "logarithmic", "power"])

            if st.button("Fit Trendline"):
                try:
                    res_cf = fit_curve_and_equation(df, cf_x, cf_y, curve_type=cf_mode)
                    st.success(f"**Discovered Equation**: `{res_cf['formula']}` | R² Goodness: **{res_cf['r_squared']}**")
                    fig_cf = create_relationship_chart(df, cf_x, cf_y, chart_type="scatter")
                    st.plotly_chart(fig_cf, use_container_width=True)
                    register_fig_callback(fig_cf)
                except Exception as e:
                    st.error(f"Fitting error: {str(e)}")

    with m_tab3:
        st.markdown("##### Parametric Function-Family Search")
        st.caption("Evolves mathematical laws across functional families using bounded genetic optimization.")
        if len(num_cols) >= 2:
            sr_x = st.selectbox("Symbolic X", num_cols, index=0)
            sr_y = st.selectbox("Symbolic Y", num_cols, index=1)
            sr_gen = st.slider("Generations", 5, 30, 15)
            sr_pop = st.slider("Population", 20, 60, 40)

            if st.button("Discover Equation"):
                with st.spinner("Executing function-family search..."):
                    try:
                        sr_res = run_symbolic_regression(df, sr_x, sr_y, generations=sr_gen, population_size=sr_pop)
                        st.success(f"Discovered Equation: **{sr_res['equation']}** (Family: {sr_res['basis_family']})")
                        st.write(f"R²: **{sr_res['r_squared']}** | MSE: **{sr_res['mse']}**")
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")

    with m_tab4:
        st.markdown("##### K-Means Clustering & PCA with Loadings")
        if len(num_cols) >= 2:
            col_km, col_pca = st.columns(2)
            with col_km:
                st.markdown("**K-Means Clustering**")
                km_feats = st.multiselect("Clustering Features", num_cols, default=num_cols[:min(3, len(num_cols))])
                km_k = st.slider("Clusters (K)", 2, 8, 3)
                if km_feats and st.button("Run K-Means"):
                    try:
                        km_df, km_meta = run_kmeans_clustering(df, km_feats, n_clusters=km_k)
                        st.write(f"Inertia: {km_meta['inertia']}")
                        st.markdown("**Cluster Centers (Original Units)**")
                        st.dataframe(km_meta["centers_dataframe"], use_container_width=True)
                        fig_km = create_relationship_chart(km_df, km_feats[0], km_feats[1], chart_type="scatter", color_col="Cluster")
                        st.plotly_chart(fig_km, use_container_width=True)
                        register_fig_callback(fig_km)
                    except Exception as e:
                        st.error(f"K-Means error: {str(e)}")

            with col_pca:
                st.markdown("**PCA Dimensionality Reduction**")
                pca_feats = st.multiselect("PCA Features", num_cols, default=num_cols[:min(4, len(num_cols))])
                pca_c = st.selectbox("Components", [2, 3])
                if len(pca_feats) >= pca_c and st.button("Run PCA"):
                    try:
                        pca_df, pca_meta = run_pca_reduction(df, pca_feats, n_components=pca_c)
                        st.write(f"Explained Variance: {pca_meta['total_explained_variance']}%")
                        st.markdown("**Feature Loadings**")
                        st.dataframe(pca_meta["loadings"], use_container_width=True)
                        fig_pca = create_relationship_chart(pca_df, "PC1", "PC2", chart_type="scatter")
                        st.plotly_chart(fig_pca, use_container_width=True)
                    except Exception as e:
                        st.error(f"PCA error: {str(e)}")

    with m_tab5:
        st.markdown("##### Multivariate Anomaly Scoring (Isolation Forest)")
        if len(num_cols) >= 2:
            iso_feats = st.multiselect("Anomaly Features", num_cols, default=num_cols[:min(3, len(num_cols))])
            iso_c = st.slider("Contamination", 0.01, 0.20, 0.05, step=0.01)
            if iso_feats and st.button("Score Anomalies"):
                try:
                    iso_df, iso_meta = run_isolation_forest_anomaly_detection(df, iso_feats, contamination=iso_c)
                    st.write(f"Detected {iso_meta['anomalies_detected']} anomalies ({iso_meta['anomaly_percentage']}%)")
                    fig_iso = create_relationship_chart(iso_df, iso_feats[0], iso_feats[1], chart_type="scatter", color_col="Anomaly_Flag")
                    st.plotly_chart(fig_iso, use_container_width=True)
                    register_fig_callback(fig_iso)
                except Exception as e:
                    st.error(f"Anomaly error: {str(e)}")
