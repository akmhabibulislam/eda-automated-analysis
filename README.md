# DataSight Analytics Platform

A modular, automated exploratory data analysis (EDA) and analytical computing platform built with Python, Streamlit, Pandas, SciPy, Scikit-Learn, and Plotly.

The application serves as a standalone exploratory data analysis, hypothesis testing, mathematical modeling, time-series decomposition, and executive reporting engine for data analysts and data scientists without requiring machine learning model training.

---

## Architecture & Workspaces

The platform is organized into six functional analytical workspaces:

1. **Ingestion, Schema & Memory Profiling (Features 1–5 + Memory Layer)**
   - Strict file format detection (.csv, .tsv, .xlsx, .xls, .json, .parquet) rejecting ambiguous extensions (.txt, .xyz).
   - Multi-layered database security: multi-statement blocking, keyword blacklist, statement timeouts, and database-side LIMIT injection.
   - Semantic schema inference identifying roles (identifier, uuid, currency, percentage, latitude, longitude, email, phone) to exclude identifiers from numerical statistics.
   - Precision-safe memory downcasting with Pandas nullable integer types (`Int8` through `Int64`) preventing NaN crash conditions.
   - Cryptographic SHA-256 dataset fingerprinting and software environment provenance tracking (Python, Pandas, NumPy, Scikit-Learn).

2. **Data Quality, Recommendations & Controlled Clean (Features 6–13)**
   - Interactive recommendation analysis identifying missing columns, duplicate rows, and outlier candidates without blind mutations.
   - Controlled cleaning pipeline where imputation and outlier capping (`threshold=3.0`) are strictly opt-in.
   - Zero-IQR protection in outlier detection preventing truncation on repeated values.
   - Strict null preservation in text cleaning, ensuring missing values are not converted into string `'nan'`.
   - Chronological audit lineage logging tracking all transformation events with operation IDs.

3. **Advanced Data Wrangling & Reshaping (Features 20–26)**
   - Secure Abstract Syntax Tree (AST) formula builder with strict depth limits, input length caps, and exponentiation ceilings (blocking catastrophic exponentiation).
   - Text regex extraction and continuous binning with native string dtypes strictly preserving null values.
   - Cartesian explosion guard on merges and cell size estimation limits on pivots to prevent memory crashes.
   - Structured parametric row query filtering replacing unrestricted expression evaluation.

4. **Statistics & Hypothesis Testing (Features 14–19, 27–31)**
   - Transparent correlation matrices reporting pairwise sample coverage ($N$ used).
   - Multicollinearity diagnostics (VIF) and correlation calculations protected with feature bounds and constant-column elimination.
   - Configurable significance thresholds ($\alpha = 0.10, 0.05, 0.01$).
   - Effect sizes (Cohen's d, Eta-squared, Cramer's V), confidence intervals, and post-hoc Tukey HSD testing.
   - Multiple testing corrections: Bonferroni and Benjamini-Hochberg (FDR).

5. **Time-Series Analysis & Mathematical Modeling (Features 32–41)**
   - Temporal resampling and moving window metrics strictly sorted by datetime.
   - Period-over-period growth with calendar frequency validation and row-order preservation.
   - Seasonal-trend decomposition strictly enforcing observation requirements ($N \ge 2 \times \text{period}$).
   - Non-linear curve fitting with trendline equation derivation ($y = mx + b$, quadratic, exponential, logarithmic, power law).
   - Parametric function-family search powered by bounded genetic optimization across multiple functional families.
   - K-Means clustering with cluster centers inverse-transformed back to original feature scale.
   - PCA dimensionality reduction reporting feature loadings, variable contributions, and explained variance ratios.
   - Multivariate anomaly scoring via Isolation Forest.

6. **Visualization Studio & Executive Reports (Features 42–52)**
   - Plotly interactive auto-plotting, distribution histograms, relationship charts, and custom multi-axis chart builders.
   - Cleaned dataset exports (CSV, Excel, Parquet).
   - Automated Executive Summary narrative generated via deterministic analytical rules.
   - Standalone interactive HTML report generation embedding cryptographic dataset fingerprints and software provenance.
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
