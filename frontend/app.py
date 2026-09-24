"""
DataSight Automated EDA Platform.
Modular UI Controller with immutable session management and structured lineage tracking.
"""

import os
import sys
import streamlit as st

# Configure page layout and clean UI theme
st.set_page_config(
    page_title="DataSight Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project root resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ingestion.memory import get_memory_usage, get_system_memory, free_memory
from backend.core.lineage import DatasetSessionManager
from frontend.views import (
    render_ingestion_view,
    render_cleaning_view,
    render_wrangling_view,
    render_statistics_view,
    render_modeling_view,
    render_reporting_view
)

# Inject custom clean styling
css_path = os.path.join(PROJECT_ROOT, "frontend", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize Session State
if "session_manager" not in st.session_state:
    st.session_state.session_manager = DatasetSessionManager()
if "recent_figures" not in st.session_state:
    st.session_state.recent_figures = []

session_manager: DatasetSessionManager = st.session_state.session_manager


def register_figure(fig):
    st.session_state.recent_figures.append(fig)
    if len(st.session_state.recent_figures) > 3:
        st.session_state.recent_figures.pop(0)


# ==============================================================================
# SIDEBAR NAVIGATION & HEALTH
# ==============================================================================
with st.sidebar:
    st.markdown("### DataSight Platform")
    st.caption("Automated Exploratory Data Analysis & Analytics Engine")
    st.markdown("---")

    active_workspace = st.radio(
        "Analytical Workspace",
        [
            "1. Ingestion & Memory Profiling",
            "2. Data Quality & Controlled Clean",
            "3. Advanced Data Wrangling",
            "4. Statistics & Hypothesis Testing",
            "5. Time-Series & Math Modeling",
            "6. Visualization Studio & Reports"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### System Resource Health")
    sys_mem = get_system_memory()
    st.write(f"**RAM Utilization**: {sys_mem['used_mb']:,.0f} MB / {sys_mem['total_mb']:,.0f} MB ({sys_mem['percent']}%)")

    current_df = session_manager.current_df
    if current_df is not None:
        df_mem = get_memory_usage(current_df)
        st.write(f"**Working Dataset**: {df_mem['readable']}")
        st.write(f"**Dataset Version**: v{session_manager.version}")

    if st.button("Explicit Garbage Collection"):
        free_memory(force=True)
        st.success("Freed unreferenced memory.")


# ==============================================================================
# TOP STATUS HEADER BANNER
# ==============================================================================
current_df = session_manager.current_df
ds_status = "Active" if current_df is not None else "Awaiting Data"
badge_color = "#ecfdf5" if current_df is not None else "#fef2f2"
badge_text_color = "#047857" if current_df is not None else "#b91c1c"
rows_str = f"{len(current_df):,} Rows" if current_df is not None else "0 Rows"
cols_str = f"{len(current_df.columns)} Columns" if current_df is not None else "0 Columns"

st.markdown(f"""
<div class="app-header-banner">
    <div class="app-title-group">
        <h1>DataSight Analytics Platform</h1>
        <p>Current Dataset: <strong>{session_manager.dataset_name}</strong> ({rows_str} &bull; {cols_str} &bull; Version: v{session_manager.version})</p>
    </div>
    <div class="app-status-badge" style="background-color: {badge_color}; color: {badge_text_color};">
        {ds_status}
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# ROUTING TO MODULAR WORKSPACE VIEWS
# ==============================================================================
if active_workspace == "1. Ingestion & Memory Profiling":
    render_ingestion_view(session_manager)

elif active_workspace == "2. Data Quality & Controlled Clean":
    render_cleaning_view(session_manager)

elif active_workspace == "3. Advanced Data Wrangling":
    render_wrangling_view(session_manager)

elif active_workspace == "4. Statistics & Hypothesis Testing":
    render_statistics_view(session_manager)

elif active_workspace == "5. Time-Series & Math Modeling":
    render_modeling_view(session_manager, register_figure)

elif active_workspace == "6. Visualization Studio & Reports":
    render_reporting_view(session_manager, st.session_state.recent_figures, register_figure)
