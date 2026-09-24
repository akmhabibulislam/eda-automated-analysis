"""
Dynamic Visualizations module using Plotly.
Features 42-47:
42. Auto-Plotting (recommended charts based on data types)
43. Distribution Charts (histograms, KDE plots, box plots)
44. Relationship Charts (scatter plots, bubble charts, line graphs)
45. Categorical Charts (bar charts, pie charts, violin plots)
46. Custom Chart Builder (manual axis selection and chart type customization)
47. Interactive Tooltips & Zooming (via Plotly)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


THEME_TEMPLATE = "plotly_white"


def create_auto_plot(df: pd.DataFrame) -> go.Figure:
    """
    Feature 42: Auto-Plotting.
    Recommends and builds charts based on detected column data types.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if isinstance(df[c].dtype, (pd.CategoricalDtype, pd.StringDtype)) or df[c].dtype == "object"]
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    if dt_cols and num_cols:
        fig = px.line(df, x=dt_cols[0], y=num_cols[0], title=f"Auto Trend: {num_cols[0]} over {dt_cols[0]}", template=THEME_TEMPLATE)
    elif len(num_cols) >= 2 and cat_cols:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1], color=cat_cols[0], title=f"Auto Relationship: {num_cols[0]} vs {num_cols[1]} by {cat_cols[0]}", template=THEME_TEMPLATE)
    elif len(num_cols) >= 2:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1], title=f"Auto Relationship: {num_cols[0]} vs {num_cols[1]}", template=THEME_TEMPLATE)
    elif cat_cols and num_cols:
        agg_df = df.groupby(cat_cols[0])[num_cols[0]].mean().reset_index().head(20)
        fig = px.bar(agg_df, x=cat_cols[0], y=num_cols[0], title=f"Auto Summary: Mean {num_cols[0]} by {cat_cols[0]}", template=THEME_TEMPLATE)
    elif num_cols:
        fig = px.histogram(df, x=num_cols[0], marginal="box", title=f"Auto Distribution: {num_cols[0]}", template=THEME_TEMPLATE)
    elif cat_cols:
        vc = df[cat_cols[0]].value_counts().head(20).reset_index()
        vc.columns = [cat_cols[0], "Count"]
        fig = px.bar(vc, x=cat_cols[0], y="Count", title=f"Auto Frequency: {cat_cols[0]}", template=THEME_TEMPLATE)
    else:
        fig = go.Figure()
        fig.update_layout(title="No plottable columns found")

    fig.update_layout(hovermode="closest", margin=dict(l=40, r=40, t=50, b=40))
    return fig


def create_distribution_chart(
    df: pd.DataFrame,
    column: str,
    chart_type: str = "histogram",
    color_col: Optional[str] = None,
    nbins: int = 30
) -> go.Figure:
    """
    Feature 43: Distribution Charts (histograms, KDE plots, box plots).
    """
    if chart_type == "histogram":
        fig = px.histogram(
            df,
            x=column,
            color=color_col,
            nbins=nbins,
            marginal="box",
            title=f"Distribution of {column}",
            template=THEME_TEMPLATE
        )
    elif chart_type == "box":
        fig = px.box(
            df,
            y=column,
            x=color_col,
            points="outliers",
            title=f"Box Plot of {column}",
            template=THEME_TEMPLATE
        )
    elif chart_type == "kde":
        fig = px.histogram(
            df,
            x=column,
            color=color_col,
            histnorm="probability density",
            marginal="violin",
            title=f"Density Estimation of {column}",
            template=THEME_TEMPLATE
        )
    else:
        fig = px.histogram(df, x=column, template=THEME_TEMPLATE)

    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def create_relationship_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    chart_type: str = "scatter",
    color_col: Optional[str] = None,
    size_col: Optional[str] = None
) -> go.Figure:
    """
    Feature 44: Relationship Charts (scatter plots, bubble charts, line graphs).
    """
    if chart_type == "scatter":
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"{y_col} vs {x_col}",
            template=THEME_TEMPLATE
        )
    elif chart_type == "bubble":
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col if size_col else None,
            title=f"Bubble Plot: {y_col} vs {x_col}",
            template=THEME_TEMPLATE
        )
    elif chart_type == "line":
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"Line Chart: {y_col} over {x_col}",
            template=THEME_TEMPLATE
        )
    else:
        fig = px.scatter(df, x=x_col, y=y_col, template=THEME_TEMPLATE)

    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def create_categorical_chart(
    df: pd.DataFrame,
    cat_col: str,
    val_col: Optional[str] = None,
    chart_type: str = "bar",
    aggregation: str = "mean"
) -> go.Figure:
    """
    Feature 45: Categorical Charts (bar charts, pie charts, violin plots).
    """
    if chart_type == "bar":
        if val_col:
            grouped = df.groupby(cat_col, observed=False)[val_col].agg(aggregation).reset_index().head(25)
            fig = px.bar(
                grouped,
                x=cat_col,
                y=val_col,
                title=f"{aggregation.title()} {val_col} by {cat_col}",
                template=THEME_TEMPLATE
            )
        else:
            vc = df[cat_col].value_counts().reset_index().head(25)
            vc.columns = [cat_col, "Count"]
            fig = px.bar(
                vc,
                x=cat_col,
                y="Count",
                title=f"Counts for {cat_col}",
                template=THEME_TEMPLATE
            )

    elif chart_type == "pie":
        vc = df[cat_col].value_counts().head(10).reset_index()
        vc.columns = [cat_col, "Count"]
        fig = px.pie(
            vc,
            names=cat_col,
            values="Count",
            title=f"Share of {cat_col} (Top 10)",
            template=THEME_TEMPLATE
        )

    elif chart_type == "violin":
        if not val_col:
            num_cols = df.select_dtypes(include=[np.number]).columns
            val_col = num_cols[0] if len(num_cols) > 0 else None

        if val_col:
            fig = px.violin(
                df,
                x=cat_col,
                y=val_col,
                box=True,
                points="all",
                title=f"Violin Plot: {val_col} by {cat_col}",
                template=THEME_TEMPLATE
            )
        else:
            fig = go.Figure()
            fig.update_layout(title="Violin plot requires a numeric value column")
    else:
        fig = go.Figure()

    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def build_custom_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    facet_col: Optional[str] = None
) -> go.Figure:
    """
    Feature 46: Custom Chart Builder.
    """
    if chart_type == "Scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col, template=THEME_TEMPLATE)
    elif chart_type == "Line":
        fig = px.line(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col, template=THEME_TEMPLATE)
    elif chart_type == "Bar":
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col, template=THEME_TEMPLATE)
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_col, color=color_col, facet_col=facet_col, template=THEME_TEMPLATE)
    elif chart_type == "Box":
        fig = px.box(df, x=x_col, y=y_col, color=color_col, template=THEME_TEMPLATE)
    elif chart_type == "Violin":
        fig = px.violin(df, x=x_col, y=y_col, color=color_col, box=True, template=THEME_TEMPLATE)
    elif chart_type == "Density Heatmap":
        fig = px.density_heatmap(df, x=x_col, y=y_col, template=THEME_TEMPLATE)
    else:
        fig = px.scatter(df, x=x_col, y=y_col, template=THEME_TEMPLATE)

    fig.update_layout(
        title=f"Custom {chart_type}: {x_col} vs {y_col if y_col else ''}",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
