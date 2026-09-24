"""
Mathematical Modeling, Equations & Unsupervised Clustering module.
Features 37-41:
37. Curve Fitting & Trendline Equations (computing and displaying mathematical formulas like y = mx + b)
38. Parametric Function-Family Search (formerly Symbolic Regression) with strict resource bounds
39. K-Means Clustering (with inverse-transformed cluster centers)
40. Dimensionality Reduction (PCA with feature loadings, contributions & explained variance)
41. Anomaly/Outlier Scoring (Isolation Forest scoring)
"""

import math
import random
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------
# Feature 37: Curve Fitting & Trendline Equations
# -------------------------------------------------------------

def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_model(x, a, b):
    return a * np.exp(np.clip(b * x, -20, 20))

def logarithmic_model(x, a, b):
    return a * np.log(np.maximum(x, 1e-9)) + b

def power_law_model(x, a, b):
    return a * np.power(np.maximum(x, 1e-9), b)


def fit_curve_and_equation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    curve_type: str = "linear"
) -> Dict[str, Any]:
    """
    Computes mathematical curve fit, trendline formula, and R-squared goodness of fit.
    """
    sub = df[[x_col, y_col]].dropna()
    x = sub[x_col].values.astype(float)
    y = sub[y_col].values.astype(float)

    if len(x) < 3:
        raise ValueError("Curve fitting requires at least 3 points.")

    model_funcs = {
        "linear": (linear_model, [1.0, 0.0]),
        "quadratic": (quadratic_model, [1.0, 1.0, 0.0]),
        "exponential": (exponential_model, [1.0, 0.01]),
        "logarithmic": (logarithmic_model, [1.0, 0.0]),
        "power": (power_law_model, [1.0, 1.0]),
    }

    if curve_type not in model_funcs:
        curve_type = "linear"

    func, p0 = model_funcs[curve_type]

    try:
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
    except Exception:
        func = linear_model
        popt, _ = curve_fit(func, x, y, p0=[1.0, 0.0], maxfev=5000)
        curve_type = "linear"

    y_pred = func(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    if curve_type == "linear":
        a, b = popt
        formula = f"y = {a:.4f}x + {b:.4f}"
    elif curve_type == "quadratic":
        a, b, c = popt
        formula = f"y = {a:.4f}x² + {b:.4f}x + {c:.4f}"
    elif curve_type == "exponential":
        a, b = popt
        formula = f"y = {a:.4f} * e^({b:.4f}x)"
    elif curve_type == "logarithmic":
        a, b = popt
        formula = f"y = {a:.4f} * ln(x) + {b:.4f}"
    elif curve_type == "power":
        a, b = popt
        formula = f"y = {a:.4f} * x^({b:.4f})"
    else:
        formula = "y = f(x)"

    x_dense = np.linspace(x.min(), x.max(), 100)
    y_dense = func(x_dense, *popt)

    return {
        "curve_type": curve_type,
        "formula": formula,
        "parameters": [round(float(p), 5) for p in popt],
        "r_squared": round(float(r_squared), 4),
        "x_dense": x_dense,
        "y_dense": y_dense,
        "x_original": x,
        "y_original": y
    }


# -------------------------------------------------------------
# Feature 38: Parametric Function-Family Search (Vectorized Genetic Algorithm)
# -------------------------------------------------------------

def run_symbolic_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    generations: int = 15,
    population_size: int = 40,
    subsample_size: int = 1000
) -> Dict[str, Any]:
    """
    Feature 38: Parametric Function-Family Search (formerly Symbolic Regression).
    Discovers mathematical equations across diverse non-linear function families
    (polynomial, sinusoidal, exponential, rational, logarithmic) using vectorized
    genetic optimization with strict population and generation limits.
    """
    sub = df[[x_col, y_col]].dropna()
    if len(sub) > subsample_size:
        sub = sub.sample(n=subsample_size, random_state=42)

    x = sub[x_col].values.astype(float)
    y = sub[y_col].values.astype(float)

    if len(x) < 5:
        raise ValueError("Parametric function search requires at least 5 data points.")

    basis_functions = [
        ("Linear Family", lambda x, c: c[0] * x + c[1], "c0*x + c1", 2),
        ("Quadratic Family", lambda x, c: c[0] * (x**2) + c[1] * x + c[2], "c0*x^2 + c1*x + c2", 3),
        ("Cubic Family", lambda x, c: c[0] * (x**3) + c[1] * (x**2) + c[2] * x + c[3], "c0*x^3 + c1*x^2 + c2*x + c3", 4),
        ("Sinusoidal Family", lambda x, c: c[0] * np.sin(c[1] * x) + c[2], "c0*sin(c1*x) + c2", 3),
        ("Exponential Family", lambda x, c: c[0] * np.exp(np.clip(c[1] * x, -15, 15)) + c[2], "c0*exp(c1*x) + c2", 3),
        ("Rational Family", lambda x, c: (c[0] * x) / (np.abs(x) + c[1] + 1e-6) + c[2], "(c0*x)/(|x|+c1) + c2", 3),
        ("Logarithmic Family", lambda x, c: c[0] * np.log(np.maximum(np.abs(x), 1e-5)) + c[1], "c0*ln(|x|) + c1", 2)
    ]

    best_fit = None
    best_fitness = float("inf")
    best_coeffs = None
    best_name = ""
    best_template = ""

    # Bound computational complexity
    gen_bounded = min(generations, 30)
    pop_bounded = min(population_size, 60)

    for name, func, template, n_coeffs in basis_functions:
        pop = [np.random.uniform(-5.0, 5.0, size=n_coeffs) for _ in range(pop_bounded)]

        for _ in range(gen_bounded):
            scores = []
            for indiv in pop:
                try:
                    preds = func(x, indiv)
                    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                        mse = 1e12
                    else:
                        mse = float(np.mean((y - preds) ** 2))
                except Exception:
                    mse = 1e12
                scores.append((mse, indiv))

            scores.sort(key=lambda item: item[0])
            if scores[0][0] < best_fitness:
                best_fitness = scores[0][0]
                best_coeffs = scores[0][1]
                best_fit = func
                best_name = name
                best_template = template

            survivors = [s[1] for s in scores[: max(2, pop_bounded // 4)]]
            new_pop = list(survivors)
            while len(new_pop) < pop_bounded:
                parent = random.choice(survivors)
                mutant = parent + np.random.normal(0, 0.20, size=n_coeffs)
                new_pop.append(mutant)
            pop = new_pop

    y_pred = best_fit(x, best_coeffs)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    clean_eq = best_template
    for idx, c in enumerate(best_coeffs):
        clean_eq = clean_eq.replace(f"c{idx}", f"{c:.3f}")

    return {
        "equation": f"y = {clean_eq}",
        "basis_family": best_name,
        "mse": round(float(best_fitness), 4),
        "r_squared": round(float(r_squared), 4),
        "generations_run": gen_bounded,
        "x": x,
        "y": y,
        "y_pred": y_pred
    }


# -------------------------------------------------------------
# Feature 39: K-Means Clustering (Inverse-Transformed Centers)
# -------------------------------------------------------------

def run_kmeans_clustering(
    df: pd.DataFrame,
    features: List[str],
    n_clusters: int = 3,
    scale: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Groups data points by feature similarity using K-Means.
    Inverse-transforms cluster centers to original data coordinates so centroids
    are interpretable in original feature units.
    """
    sub = df[features].dropna().copy()
    if len(sub) < n_clusters:
        raise ValueError("Number of samples must exceed number of clusters.")

    X = sub.values
    if scale:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
    else:
        scaler = None
        X_fit = X

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_fit)

    # Convert centers back to original feature scale
    if scaler is not None:
        raw_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    else:
        raw_centers = kmeans.cluster_centers_

    result_df = sub.copy()
    result_df["Cluster"] = [f"Cluster {c}" for c in clusters]
    cluster_counts = result_df["Cluster"].value_counts().to_dict()

    # Build center dataframe
    centers_df = pd.DataFrame(
        raw_centers.round(4),
        columns=features,
        index=[f"Cluster {c}" for c in range(n_clusters)]
    ).reset_index()
    centers_df.rename(columns={"index": "Cluster"}, inplace=True)

    return result_df, {
        "n_clusters": n_clusters,
        "inertia": round(float(kmeans.inertia_), 2),
        "cluster_counts": cluster_counts,
        "cluster_centers": centers_df.to_dict(orient="records"),
        "centers_dataframe": centers_df
    }


# -------------------------------------------------------------
# Feature 40: Dimensionality Reduction (PCA with Loadings & Contributions)
# -------------------------------------------------------------

def run_pca_reduction(
    df: pd.DataFrame,
    features: List[str],
    n_components: int = 2,
    scale: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Feature 40: PCA with feature loadings, variable contributions, and explained variance.
    """
    sub = df[features].dropna().copy()
    if len(sub) < n_components or len(features) < n_components:
        raise ValueError("Insufficient features or rows for requested PCA components.")

    X = sub.values
    if scale:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
    else:
        X_fit = X

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_fit)

    col_names = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(coords, columns=col_names, index=sub.index)

    explained_var = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    total_var = round(float(sum(explained_var)), 2)

    # Calculate feature loadings (eigenvectors)
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=col_names,
        index=features
    ).round(4).reset_index().rename(columns={"index": "Feature"})

    # Calculate relative feature contributions
    contributions = (pca.components_ ** 2) / np.sum(pca.components_ ** 2, axis=1, keepdims=True)
    contributions_df = pd.DataFrame(
        (contributions.T * 100).round(2),
        columns=[f"{col}_Contribution_Pct" for col in col_names],
        index=features
    ).reset_index().rename(columns={"index": "Feature"})

    return pca_df, {
        "components": n_components,
        "explained_variance_ratio": explained_var,
        "total_explained_variance": total_var,
        "loadings": loadings_df,
        "contributions": contributions_df
    }


# -------------------------------------------------------------
# Feature 41: Anomaly / Outlier Scoring (Isolation Forest)
# -------------------------------------------------------------

def run_isolation_forest_anomaly_detection(
    df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detects multivariate anomalies and computes anomaly scores using Isolation Forest.
    """
    sub = df[features].dropna().copy()
    if len(sub) < 10:
        raise ValueError("Anomaly detection requires at least 10 observations.")

    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(sub)
    scores = iso.decision_function(sub)

    result_df = sub.copy()
    result_df["Anomaly_Flag"] = np.where(preds == -1, "Anomaly", "Normal")
    result_df["Anomaly_Score"] = np.round(scores, 4)

    anomaly_count = int((result_df["Anomaly_Flag"] == "Anomaly").sum())

    return result_df, {
        "contamination": contamination,
        "total_analyzed": len(result_df),
        "anomalies_detected": anomaly_count,
        "anomaly_percentage": round((anomaly_count / len(result_df) * 100), 2)
    }
