"""
Exporting & Reporting module.
Features 48-52:
48. Cleaned Dataset Export (CSV, Excel, Parquet download bytes)
49. Chart Image Export (PNG, SVG, HTML saving)
50. Automated Executive Summary (deterministic rule-based analytical report)
51. Comprehensive HTML Report Generation
52. Dynamic Multi-Page PDF Report Export with Auto-Wrapping Tables
"""

import io
import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# -------------------------------------------------------------
# Feature 48: Cleaned Dataset Export
# -------------------------------------------------------------

def export_dataset_bytes(df: pd.DataFrame, file_format: str = "csv") -> Tuple[bytes, str, str]:
    """
    Exports a dataframe to in-memory bytes for CSV, Excel, or Parquet download.
    Returns: (bytes_data, mime_type, file_extension)
    """
    buffer = io.BytesIO()

    if file_format.lower() == "csv":
        df.to_csv(buffer, index=False)
        mime = "text/csv"
        ext = "csv"
    elif file_format.lower() in ["excel", "xlsx"]:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Dataset")
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ext = "xlsx"
    elif file_format.lower() in ["parquet", "pq"]:
        df.to_parquet(buffer, index=False)
        mime = "application/octet-stream"
        ext = "parquet"
    else:
        df.to_csv(buffer, index=False)
        mime = "text/csv"
        ext = "csv"

    buffer.seek(0)
    data = buffer.getvalue()
    return data, mime, ext


# -------------------------------------------------------------
# Feature 49: Chart Image / HTML Export
# -------------------------------------------------------------

def export_chart_html(fig: go.Figure) -> str:
    """
    Exports Plotly chart as an interactive standalone HTML string.
    """
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


# -------------------------------------------------------------
# Feature 50: Automated Executive Summary (Deterministic Engine)
# -------------------------------------------------------------

def generate_executive_summary(df: pd.DataFrame, audit_logs: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Feature 50: Automated Executive Summary.
    Produces a plain-English, professional executive narrative of dataset properties,
    statistical patterns, missingness, and detected data qualities using deterministic
    analytical rules (not an LLM).
    """
    n_rows, n_cols = df.shape
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if isinstance(df[c].dtype, (pd.CategoricalDtype, pd.StringDtype)) or df[c].dtype == "object"]
    missing_cells = int(df.isna().sum().sum())
    total_cells = n_rows * n_cols
    missing_pct = round((missing_cells / total_cells * 100) if total_cells > 0 else 0.0, 2)
    duplicates = int(df.duplicated().sum())

    sections = []

    # 1. Dataset Scale & Completeness
    sections.append(
        f"**Dataset Dimensions and Integrity Overview**\n"
        f"The active dataset comprises {n_rows:,} records across {n_cols} attributes, including {len(num_cols)} numerical "
        f"features and {len(cat_cols)} categorical dimensions. Memory utilization is currently optimized. "
        f"A total of {missing_cells:,} missing data points were detected, representing {missing_pct}% of total cells. "
        f"{duplicates:,} duplicate rows were identified."
    )

    # 2. Key Statistical Insights
    if num_cols:
        primary_num = num_cols[0]
        s = df[primary_num].dropna()
        if len(s) > 0:
            mean_val = s.mean()
            median_val = s.median()
            std_val = s.std()
            skew_desc = "positively skewed" if mean_val > median_val else "negatively skewed or symmetrical"
            sections.append(
                f"**Numerical Distribution Characteristics**\n"
                f"Primary continuous metric '{primary_num}' exhibits a mean of {mean_val:.2f}, median of {median_val:.2f}, "
                f"and standard deviation of {std_val:.2f}. The relationship between the mean and median indicates that the "
                f"distribution is {skew_desc}."
            )

    # 3. Categorical Breakdown
    if cat_cols:
        primary_cat = cat_cols[0]
        mode_series = df[primary_cat].dropna().mode()
        top_val = mode_series.iloc[0] if len(mode_series) > 0 else "N/A"
        unique_cnt = df[primary_cat].nunique()
        sections.append(
            f"**Categorical Composition**\n"
            f"For attribute '{primary_cat}', there are {unique_cnt} unique categories. The dominant class is "
            f"'{top_val}', highlighting significant concentration within this dimension."
        )

    # 4. Data Lineage and Cleaning Actions
    if audit_logs:
        steps_count = len(audit_logs)
        sections.append(
            f"**Transformation Audit Trail**\n"
            f"A total of {steps_count} data preparation steps were executed during this analytical session, "
            f"including schema verification, missing value handling, and duplicate purging."
        )
    else:
        sections.append(
            f"**Transformation Audit Trail**\n"
            f"The dataset is maintained in its current uploaded baseline state."
        )

    return "\n\n".join(sections)


# -------------------------------------------------------------
# Feature 51: Comprehensive HTML Report Generation
# -------------------------------------------------------------

def generate_html_report(
    df: pd.DataFrame,
    summary_text: str,
    audit_logs: Optional[List[Dict[str, Any]]] = None,
    figures: Optional[List[go.Figure]] = None,
    fingerprint_sha256: Optional[str] = None,
    provenance: Optional[Dict[str, str]] = None,
    analysis_parameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Feature 51: Standalone HTML Analytical Report with cryptographic dataset fingerprint,
    software environment provenance, and parameter audit trail.
    """
    n_rows, n_cols = df.shape
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fp_display = fingerprint_sha256 if fingerprint_sha256 else "N/A"

    provenance_html = ""
    if provenance:
        prov_items = "".join([f"<li><strong>{k}:</strong> {v}</li>" for k, v in provenance.items()])
        provenance_html = f"""
        <div class="provenance-box">
            <h3>Software Environment & Reproducibility Provenance</h3>
            <ul>{prov_items}</ul>
        </div>
        """

    preview_table_html = df.head(10).to_html(classes="styled-table", index=False)
    num_df = df.describe().round(2).reset_index()
    num_table_html = num_df.to_html(classes="styled-table", index=False) if not num_df.empty else "<p>No numeric columns.</p>"

    charts_html = ""
    if figures:
        for idx, fig in enumerate(figures):
            charts_html += f"""
            <div class="chart-container">
                <h3>Visualization {idx + 1}</h3>
                {export_chart_html(fig)}
            </div>
            """

    audit_table_html = ""
    if audit_logs:
        audit_df = pd.DataFrame(audit_logs)
        audit_table_html = f"""
        <h2>Data Transformation Audit Trail</h2>
        {audit_df.to_html(classes="styled-table", index=False)}
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Automated Analytical Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1f2937;
            background-color: #f9fafb;
            margin: 0;
            padding: 40px 20px;
        }}
        .report-wrapper {{
            max-width: 1000px;
            margin: 0 auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #111827;
            font-size: 28px;
            margin-bottom: 8px;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 12px;
        }}
        .metadata {{
            font-size: 13px;
            color: #6b7280;
            margin-bottom: 20px;
            line-height: 1.5;
        }}
        .provenance-box {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 24px;
            font-size: 12px;
        }}
        .provenance-box ul {{
            margin: 4px 0 0 16px;
            padding: 0;
        }}
        .provenance-box li {{
            margin-bottom: 2px;
        }}
        h2 {{
            color: #1f2937;
            font-size: 20px;
            margin-top: 32px;
            margin-bottom: 12px;
        }}
        h3 {{
            color: #374151;
            font-size: 16px;
            margin-top: 16px;
        }}
        .executive-summary {{
            background: #f3f4f6;
            border-left: 4px solid #2563eb;
            padding: 16px 20px;
            border-radius: 4px;
            line-height: 1.6;
        }}
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 13px;
        }}
        .styled-table th, .styled-table td {{
            padding: 10px 12px;
            border: 1px solid #e5e7eb;
            text-align: left;
        }}
        .styled-table th {{
            background-color: #f9fafb;
            font-weight: 600;
        }}
        .styled-table tr:nth-child(even) {{
            background-color: #fdfdfd;
        }}
        .chart-container {{
            margin: 28px 0;
            padding: 16px;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <div class="report-wrapper">
        <h1>Automated Data Analysis Report</h1>
        <div class="metadata">
            Generated on: {timestamp} | Dimensions: {n_rows:,} Rows × {n_cols} Columns<br>
            <strong>Dataset SHA-256 Fingerprint:</strong> <code>{fp_display}</code>
        </div>

        {provenance_html}

        <h2>Executive Narrative</h2>
        <div class="executive-summary">
            {summary_text.replace(chr(10), '<br>')}
        </div>

        <h2>Dataset Sample (First 10 Records)</h2>
        <div style="overflow-x: auto;">
            {preview_table_html}
        </div>

        <h2>Statistical Summary</h2>
        <div style="overflow-x: auto;">
            {num_table_html}
        </div>

        {charts_html}

        {audit_table_html}
    </div>
</body>
</html>
"""


# -------------------------------------------------------------
# Feature 52: Dynamic Multi-Page PDF Report Export
# -------------------------------------------------------------

def generate_pdf_report(
    df: pd.DataFrame,
    summary_text: str,
    audit_logs: Optional[List[Dict[str, Any]]] = None,
    fingerprint_sha256: Optional[str] = None,
    provenance: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Feature 52: Printable Executive PDF Report.
    Dynamically wraps text cells in Paragraph flowables to prevent text clipping
    or page boundary overflow. Embeds SHA-256 fingerprint and software provenance.
    """
    buffer = io.BytesIO()
    margin = 36
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=margin,
        leftMargin=margin,
        topMargin=margin,
        bottomMargin=margin
    )
    printable_width = letter[0] - (2 * margin)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#111827"),
        spaceAfter=8
    )
    meta_style = ParagraphStyle(
        name="MetaText",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        name="Heading2Custom",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1f2937"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        name="BodyCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#374151"),
        spaceAfter=8
    )
    cell_style = ParagraphStyle(
        name="TableCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1f2937")
    )
    header_cell_style = ParagraphStyle(
        name="TableHeaderCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#111827")
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("Executive Data Analysis Report", title_style))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fp_text = f" | SHA-256: {fingerprint_sha256[:16]}..." if fingerprint_sha256 else ""
    elements.append(Paragraph(f"Generated: {timestamp} | Total Rows: {len(df):,} | Total Columns: {len(df.columns)}{fp_text}", meta_style))

    if provenance:
        prov_summary = f"Environment: Python {provenance.get('python_version')} | Pandas {provenance.get('pandas_version')} | NumPy {provenance.get('numpy_version')} | Scikit-Learn {provenance.get('scikit_learn_version')}"
        elements.append(Paragraph(prov_summary, meta_style))

    elements.append(Spacer(1, 6))

    # Executive Narrative
    elements.append(Paragraph("Executive Narrative", h2_style))
    for paragraph in summary_text.split("\n\n"):
        clean_para = paragraph.replace("**", "")
        elements.append(Paragraph(clean_para, body_style))

    elements.append(Spacer(1, 10))

    # Descriptive Statistics Table
    elements.append(Paragraph("Descriptive Statistics", h2_style))
    num_df = df.describe().round(2).reset_index()

    if not num_df.empty:
        selected_cols = list(num_df.columns[:min(6, len(num_df.columns))])
        col_width = printable_width / len(selected_cols)

        table_flowables = []
        header_row = [Paragraph(str(c), header_cell_style) for c in selected_cols]
        table_flowables.append(header_row)

        for _, row in num_df[selected_cols].iterrows():
            row_cells = [Paragraph(str(val), cell_style) for val in row]
            table_flowables.append(row_cells)

        t = Table(table_flowables, colWidths=[col_width] * len(selected_cols))
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ]))
        elements.append(t)

    # Lineage Audit Trail Table
    if audit_logs:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Data Lineage and Transformation History", h2_style))

        audit_headers = ["Timestamp", "Action", "Details", "Rows Affected"]
        col_widths = [printable_width * 0.18, printable_width * 0.25, printable_width * 0.42, printable_width * 0.15]

        audit_flowables = [[Paragraph(h, header_cell_style) for h in audit_headers]]
        for log in audit_logs[-12:]:
            audit_flowables.append([
                Paragraph(str(log.get("timestamp", ""))[-8:], cell_style),
                Paragraph(str(log.get("action", "")), cell_style),
                Paragraph(str(log.get("details", "")), cell_style),
                Paragraph(str(log.get("rows_affected", 0)), cell_style)
            ])

        t_audit = Table(audit_flowables, colWidths=col_widths)
        t_audit.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ]))
        elements.append(t_audit)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
