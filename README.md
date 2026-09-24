# DataSight Analytics Platform

Production-ready, enterprise-grade Automated Data Analysis Web Application built from scratch with Python, Streamlit, Pandas, SciPy, Scikit-Learn, and Plotly.

The application serves as a standalone exploratory data analysis (EDA), hypothesis testing, mathematical modeling, time-series decomposition, and executive reporting engine for data analysts and data scientists without requiring machine learning model training.

---

## Architectural Highlights

- **Multi-Format Ingestion**: Native streaming and chunked parsing for CSV, Excel (.xlsx/.xls), JSON, Apache Parquet, and TSV files.
- **SQL Database Integrations**: Unified SQLAlchemy connector for SQLite, PostgreSQL, and MySQL.
- **Dedicated Memory Management & Optimization Layer**:
  - Automatic numeric downcasting (converting standard `int64` and `float64` to safe minimal representations like `int32`, `int16`, `int8`, and `float32`).
  - Low-cardinality string/object detection with conversion to high-efficiency Pandas `category` dtype.
  - Automatic file size checks and configurable row sampling for previewing large files.
  - Real-time RAM utilization tracking and explicit garbage collection (`gc.collect()`) triggers.
- **Automated Data Cleaning (Auto-Clean)**:
  - Missing value imputation (mean, median, mode, constant, ffill, bfill).
  - Outlier detection and treatment (IQR and Z-score methods with capping, dropping, or flagging).
  - Column header standardization (snake_case, lower_case, upper_case, camelCase).
  - Full chronological audit trail and data lineage logging.
- **Deep Statistical & Hypothesis Testing**:
  - Univariate descriptive statistics, skewness, kurtosis, and missingness matrix.
  - Correlation matrices (Pearson, Spearman, Kendall) and automated multi-collinearity detection.
  - Independent & Paired T-Tests, One-Way ANOVA, Chi-Square Independence Test, Mann-Whitney U, Kruskal-Wallis, and Kolmogorov-Smirnov distribution fitting.
- **Time-Series Analysis**:
  - Temporal resampling, rolling averages, period-over-period growth (MoM/YoY), cumulative totals, and seasonal decomposition (trend, seasonality, residuals).
- **Mathematical Modeling & Unsupervised Learning**:
  - Non-linear curve fitting with trendline equation derivation ($y = mx + b$, quadratic, exponential, logarithmic, power law).
  - Symbolic regression powered by genetic programming algorithms to uncover mathematical equations.
  - K-Means clustering, Principal Component Analysis (PCA 2D/3D), and Isolation Forest anomaly scoring.
- **Dynamic Visualizations & Reporting**:
  - Plotly interactive auto-plotting, distribution histograms, KDEs, box plots, scatter/bubble charts, and custom multi-axis chart builder.
  - One-click dataset export (CSV, Excel, Parquet).
  - Automated executive narrative generation with standalone HTML and printable PDF report compilation.

---

## Directory Structure

```text
data-analysis-app/
├── backend/
│   ├── __init__.py
│   ├── main.py              # Core service router and programmatic analysis API
│   ├── ingestion/           # Multi-format loaders, schema detection, memory layer
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   └── memory.py
│   ├── cleaning/            # Auto-cleaning pipeline, wrangling, audit logging
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   └── wrangling.py
│   ├── analysis/            # Statistics, correlation, hypothesis tests, time-series
│   │   ├── __init__.py
│   │   ├── statistics.py
│   │   ├── hypothesis.py
│   │   └── timeseries.py
│   ├── modeling/            # Curve fitting, symbolic regression, clustering, PCA
│   │   ├── __init__.py
│   │   └── modeling.py
│   └── reporting/           # Dataset exports, HTML and PDF report generation
│       ├── __init__.py
│       └── reports.py
├── frontend/
│   ├── __init__.py
│   ├── app.py               # Main UI application controller (Streamlit)
│   ├── components/          # Reusable UI widgets and visualization charts
│   │   ├── __init__.py
│   │   └── visualizations.py
│   └── assets/              # Enterprise styling and CSS stylesheets
│       └── styles.css
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Comprehensive automated test suite
├── requirements.txt         # Pinned Python dependencies
├── README.md                # Technical documentation
└── .gitignore               # Python environment and cache exclusion
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested through Python 3.14)
- Git

### 1. Clone the Repository
```bash
git clone <repository_url>
cd eda
```

### 2. Configure Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite
```bash
python -m unittest tests/test_pipeline.py
```

### 4. Launch the Web Application
```bash
streamlit run frontend/app.py
```
Open your browser to `http://localhost:8501`.

---

## Feature Matrix (52 Implemented Features)

| ID | Feature | Module Location |
|---|---|---|
| 1 | Multi-Format File Upload | `backend/ingestion/loaders.py` |
| 2 | Database Connectors (SQLite, Postgres, MySQL) | `backend/ingestion/loaders.py` |
| 3 | Data Schema Detection | `backend/ingestion/loaders.py` |
| 4 | Dataset Overview Dashboard | `backend/ingestion/loaders.py` |
| 5 | Data Preview Table | `backend/ingestion/loaders.py` |
| 6 | Missing Value Imputation | `backend/cleaning/cleaning.py` |
| 7 | Missing Value Dropping | `backend/cleaning/cleaning.py` |
| 8 | Duplicate Removal | `backend/cleaning/cleaning.py` |
| 9 | Outlier Detection & Treatment | `backend/cleaning/cleaning.py` |
| 10 | Column Header Standardization | `backend/cleaning/cleaning.py` |
| 11 | Text & String Cleaning | `backend/cleaning/cleaning.py` |
| 12 | Data Type Casting | `backend/cleaning/cleaning.py` |
| 13 | Data Lineage / Audit Trail | `backend/cleaning/cleaning.py` |
| 14 | Univariate Summary Statistics | `backend/analysis/statistics.py` |
| 15 | Categorical Frequency Distribution | `backend/analysis/statistics.py` |
| 16 | Missingness Matrix | `backend/analysis/statistics.py` |
| 17 | Correlation Matrix | `backend/analysis/statistics.py` |
| 18 | Multi-Collinearity Detection | `backend/analysis/statistics.py` |
| 19 | Skewness & Kurtosis Analysis | `backend/analysis/statistics.py` |
| 20 | Custom Formula / Equation Builder | `backend/cleaning/wrangling.py` |
| 21 | Binning & Discretization | `backend/cleaning/wrangling.py` |
| 22 | SQL-like Groupby & Aggregations | `backend/cleaning/wrangling.py` |
| 23 | Merging & Joining | `backend/cleaning/wrangling.py` |
| 24 | Text Regex Extraction | `backend/cleaning/wrangling.py` |
| 25 | Data Pivoting & Unpivoting | `backend/cleaning/wrangling.py` |
| 26 | Custom Filtering & Querying | `backend/cleaning/wrangling.py` |
| 27 | T-Tests (Independent & Paired) | `backend/analysis/hypothesis.py` |
| 28 | One-Way ANOVA | `backend/analysis/hypothesis.py` |
| 29 | Chi-Square Test of Independence | `backend/analysis/hypothesis.py` |
| 30 | Mann-Whitney U / Kruskal-Wallis | `backend/analysis/hypothesis.py` |
| 31 | Distribution Fitting (KS-Test) | `backend/analysis/hypothesis.py` |
| 32 | Temporal Resampling | `backend/analysis/timeseries.py` |
| 33 | Rolling & Moving Averages | `backend/analysis/timeseries.py` |
| 34 | Period-over-Period Growth | `backend/analysis/timeseries.py` |
| 35 | Cumulative Sums & Running Totals | `backend/analysis/timeseries.py` |
| 36 | Seasonality & Trend Decomposition | `backend/analysis/timeseries.py` |
| 37 | Curve Fitting & Trendline Equations | `backend/modeling/modeling.py` |
| 38 | Symbolic Regression (Genetic Algorithm) | `backend/modeling/modeling.py` |
| 39 | K-Means Clustering | `backend/modeling/modeling.py` |
| 40 | Dimensionality Reduction (PCA) | `backend/modeling/modeling.py` |
| 41 | Anomaly / Outlier Scoring (Isolation Forest) | `backend/modeling/modeling.py` |
| 42 | Auto-Plotting Recommendations | `frontend/components/visualizations.py` |
| 43 | Distribution Charts (Histograms, KDE, Box) | `frontend/components/visualizations.py` |
| 44 | Relationship Charts (Scatter, Bubble, Line) | `frontend/components/visualizations.py` |
| 45 | Categorical Charts (Bar, Pie, Violin) | `frontend/components/visualizations.py` |
| 46 | Custom Multi-Axis Chart Builder | `frontend/components/visualizations.py` |
| 47 | Interactive Tooltips & Zooming (Plotly) | `frontend/components/visualizations.py` |
| 48 | Cleaned Dataset Export (CSV, Excel, Parquet) | `backend/reporting/reports.py` |
| 49 | Chart Image & HTML Export | `backend/reporting/reports.py` |
| 50 | Automated Executive Narrative | `backend/reporting/reports.py` |
| 51 | Standalone HTML Report Generation | `backend/reporting/reports.py` |
| 52 | Printable PDF Report Export | `backend/reporting/reports.py` |
| Opt | Memory Management & Auto-Downcasting | `backend/ingestion/memory.py` |

---

## License
MIT License.
