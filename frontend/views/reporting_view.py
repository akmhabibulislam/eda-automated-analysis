"""
Workspace 6 View: Visualization Studio & Executive Reports.
"""

import streamlit as st
import pandas as pd
import numpy as np

from frontend.components.visualizations import (
    create_auto_plot,
    build_custom_chart
)
from backend.reporting.reports import (
    export_dataset_bytes,
    generate_executive_summary,
    generate_html_report,
    generate_pdf_report
)
from backend.core.lineage import DatasetSessionManager


def render_reporting_view(session_manager: DatasetSessionManager, recent_figures: list, register_fig_callback):
    st.subheader("Dynamic Visualization Studio & Executive Reports")
    st.caption("Interactive Plotly chart builders, dataset downloads, standalone HTML reports, and dynamic printable PDFs.")

    df = session_manager.current_df
    if df is None:
        st.info("Please load a dataset in Workspace 1 first.")
        return

    lineage_logs = session_manager.get_lineage_dicts()

    v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs([
        "Visualization Studio", "Custom Chart Builder", "Automated Executive Summary", "Report & Dataset Export"
    ])

    with v_tab1:
        st.markdown("##### Auto-Plotting Recommendation")
        try:
            fig_auto = create_auto_plot(df)
            st.plotly_chart(fig_auto, use_container_width=True)
            register_fig_callback(fig_auto)
        except Exception as e:
            st.error(f"Plotting error: {str(e)}")

    with v_tab2:
        st.markdown("##### Custom Multi-Axis Chart Builder")
        c_type = st.selectbox("Primitive", ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Density Heatmap"])
        c_x = st.selectbox("X Axis", df.columns)
        c_y = st.selectbox("Y Axis (Optional)", ["None"] + list(df.columns))
        c_col = st.selectbox("Color Grouping", ["None"] + list(df.columns))

        if st.button("Render Chart"):
            try:
                built_f = build_custom_chart(
                    df,
                    chart_type=c_type,
                    x_col=c_x,
                    y_col=None if c_y == "None" else c_y,
                    color_col=None if c_col == "None" else c_col
                )
                st.plotly_chart(built_f, use_container_width=True)
                register_fig_callback(built_f)
            except Exception as e:
                st.error(f"Render error: {str(e)}")

    with v_tab3:
        st.markdown("##### Automated Executive Summary")
        st.caption("Deterministic analytical narrative generated from dataset distributions and transformation records.")
        exec_text = generate_executive_summary(df, audit_logs=lineage_logs)
        st.markdown(exec_text)

    with v_tab4:
        st.markdown("##### Comprehensive Analytical Report & Dataset Download")
        exec_text = generate_executive_summary(df, audit_logs=lineage_logs)

        rep_c1, rep_c2, rep_c3 = st.columns(3)
        with rep_c1:
            st.markdown("**Standalone HTML Report**")
            html_out = generate_html_report(df, summary_text=exec_text, audit_logs=lineage_logs, figures=recent_figures)
            st.download_button("Download HTML Report", data=html_out, file_name="datasight_report.html", mime="text/html")

        with rep_c2:
            st.markdown("**Executive PDF Report**")
            pdf_out = generate_pdf_report(df, summary_text=exec_text, audit_logs=lineage_logs)
            st.download_button("Download PDF Report", data=pdf_out, file_name="datasight_report.pdf", mime="application/pdf")

        with rep_c3:
            st.markdown("**Cleaned Dataset Export**")
            f_fmt = st.selectbox("Format", ["CSV", "Excel", "Parquet"])
            d_bytes, mime_t, f_ext = export_dataset_bytes(df, file_format=f_fmt.lower())
            st.download_button(f"Download .{f_ext}", data=d_bytes, file_name=f"datasight_clean.{f_ext}", mime=mime_t)
