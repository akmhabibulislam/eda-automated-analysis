# DataSight Analytics Platform

Production-ready, enterprise-grade Automated Data Analysis Web Application built with Python, Streamlit, Pandas, SciPy, Scikit-Learn, and Plotly.

The application serves as a standalone exploratory data analysis (EDA), hypothesis testing, mathematical modeling, time-series decomposition, and executive reporting engine for data analysts and data scientists without requiring machine learning model training.

---

## Architecture & Workspaces

The platform is organized into six functional analytical workspaces:

1. **Ingestion, Schema & Memory Profiling (Features 1–5 + Memory Layer)**
   - Strict file format detection (.csv, .tsv, .xlsx, .xls, .json, .parquet) rejecting ambiguous extensions (.txt, .xyz).
   - Safe relational database connector (SQLite, PostgreSQL, MySQL) enforcing read-only SELECT queries with connection timeouts and row ceilings.
   - Robust schema detection distinguishing between genuine dates and alphanumeric identifiers, UUIDs, or SKUs containing hyphens.
   - Precision-safe memory management: safe integer downcasting (`int64` to `int32`/`int16`/`int8`) with float downcasting strictly opt-in to avoid precision loss in financial and scientific applications.
   - Real-time RAM utilization tracking and explicit garbage collection (`gc.collect()`).

2. **Data Quality, Recommendations & Controlled Clean (Features 6–13)**
   - Interactive recommendation analysis identifying missing columns, duplicate rows, and outlier candidates without blind mutations.
   - Controlled cleaning pipeline where imputation and outlier capping (`threshold=3.0`) are opt-in.
   - Strict null preservation in text cleaning, ensuring missing values are not converted into string `'nan'`.
   - Robust boolean parser converting `'true'`, `'yes'`, `'1'` vs `'false'`, `'no'`, `'0'` safely without `astype(bool)` flaws.
   - Chronological audit lineage logging tracking all transformation events.

3. **Advanced Data Wrangling & Reshaping (Features 20–26)**
   - Secure Abstract Syntax Tree (AST) formula evaluation engine (`revenue - cost` or `sqrt(units_sold) * 10`) blocking code injection attempts (`__import__`, `open`, `eval`, `exec`).
   - Continuous binning & discretization (equal-width and quantile).
   - SQL-like groupby and multi-aggregations (`mean`, `sum`, `std`, `median`).
   - Structured parametric row query filtering replacing unrestricted expression evaluation.
   - Text regex pattern extraction and dataset pivoting/unpivoting (melt).

4. **Statistics & Hypothesis Testing (Features 14–19, 27–31)**
   - Univariate descriptive statistics, distribution skewness, and kurtosis.
   - Correlation matrices (Pearson, Spearman, Kendall).
   - Pairwise correlation warnings (features with $r \ge 0.80$) and Variance Inflation Factor (VIF) multicollinearity diagnostics.
   - Independent T-Tests and index-aligned paired T-Tests dropping missing pairs symmetrically.
   - One-Way ANOVA and Pearson's Chi-Square Test of Independence with expected frequency tables.
   - Non-parametric Mann-Whitney U and Kruskal-Wallis tests.
   - Kolmogorov-Smirnov distribution fitting (Normal, Exponential, Uniform, Log-Normal) with explicit parameter disclosure.

5. **Time-Series Analysis & Mathematical Modeling (Features 32–41)**
   - Temporal resampling and moving window metrics strictly sorted by datetime.
   - Period-over-period growth (MoM/YoY) and cumulative sums.
   - Seasonal-trend decomposition strictly enforcing observation requirements ($N \ge 2 \times \text{period}$) without silent period alterations.
   - Non-linear curve fitting with trendline equation derivation ($y = mx + b$, quadratic, exponential, logarithmic, power law).
   - Parametric function-family search powered by bounded genetic optimization across multiple functional families.
   - K-Means clustering with cluster centers inverse-transformed back to original feature scale.
   - PCA dimensionality reduction reporting feature loadings, variable contributions, and explained variance ratios.
   - Multivariate anomaly scoring via Isolation Forest.

6. **Visualization Studio & Executive Reports (Features 42–52)**
   - Plotly interactive auto-plotting, distribution histograms, relationship charts, and custom multi-axis chart builders.
   - Cleaned dataset exports (CSV, Excel, Parquet).
   - Automated Executive Summary narrative generated via deterministic analytical rules.
   - Standalone interactive HTML report generation.
   - Printable multi-page PDF report generation with dynamic ReportLab table cell wrapping to prevent margin overflow.

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested through Python 3.14)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/akmhabibulislam/eda-automated-analysis.git
cd eda-automated-analysis
```

### 2. Configure Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Exhaustive Test Suite
```bash
python -m unittest tests/test_pipeline.py
```

### 4. Launch the Web Application
```bash
streamlit run frontend/app.py
```
Open your browser to `http://localhost:8501`.

---

## License
MIT License.
