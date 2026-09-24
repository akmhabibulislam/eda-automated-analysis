"""
Mathematical Modeling, Equations & Unsupervised Clustering module.
Features 37-41:
37. Curve Fitting & Trendline Equations (computing and displaying mathematical formulas like y = mx + b)
38. Symbolic Regression (genetic search for discovering mathematical equations)
39. K-Means Clustering (grouping data points by feature similarity)
40. Dimensionality Reduction (PCA for 2D/3D visualization)
41. Anomaly/Outlier Scoring (Isolation Forest scoring)
"""

import math
import random
from typing import Dict, Any, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from backend.ingestion.memory import free_memory


# -------------------------------------------------------------
# Feature 37: Curve Fitting & Trendline Equations
# -------------------------------------------------------------

def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_model(x, a, b):
    return a * np.exp(b * x)

def logarithmic_model(x, a, b):
    return a * np.log(np.maximum(x, 1e-9)) + b

def power_law_model(x, a, b):
    return a * np.power(np.maximum(x, 1e-9), b)


def fit_curve_and_equation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    curve_type: str = "linear"  # linear, quadratic, exponential, logarithmic, power
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
        # Fallback to linear if non-linear fails convergence
        func = linear_model
        popt, _ = curve_fit(func, x, y, p0=[1.0, 0.0], maxfev=5000)
        curve_type = "linear"

    y_pred = func(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Format human-readable formula
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

    # Generate smooth trendline line
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
# Feature 38: Symbolic Regression (Genetic Algorithm)
# -------------------------------------------------------------

class SymbolicNode:
    """Node in an expression tree for symbolic regression."""
    def __init__(self, node_type: str, value: Any, left=None, right=None):
        self.node_type = node_type  # 'op', 'var', 'const'
        self.value = value          # '+', '-', '*', '/', or variable name, or float
        self.left = left
        self.right = right

    def evaluate(self, x_data: np.ndarray) -> np.ndarray:
        if self.node_type == "const":
            return np.full_like(x_data, self.value)
        elif self.node_type == "var":
            return x_data
        elif self.node_type == "op":
            left_val = self.left.evaluate(x_data) if self.left else 0
            right_val = self.right.evaluate(x_data) if self.right else 0
            if self.value == "+":
                return left_val + right_val
            elif self.value == "-":
                return left_val - right_val
            elif self.value == "*":
                return left_val * right_val
            elif self.value == "/":
                return np.divide(left_val, right_val, out=np.zeros_like(left_val), where=np.abs(right_val) > 1e-6)
            elif self.value == "sin":
                return np.sin(left_val)
            elif self.value == "cos":
                return np.cos(left_val)
        return np.zeros_like(x_data)

    def to_string(self) -> str:
        if self.node_type == "const":
            return f"{self.value:.2f}"
        elif self.node_type == "var":
            return "x"
        elif self.node_type == "op":
            if self.value in ["sin", "cos"]:
                return f"{self.value}({self.left.to_string() if self.left else 'x'})"
            return f"({self.left.to_string() if self.left else '0'} {self.value} {self.right.to_string() if self.right else '0'})"
        return ""


def run_symbolic_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    generations: int = 15,
    population_size: int = 40
) -> Dict[str, Any]:
    """
    Discovers mathematical equations from data using a genetic programming algorithm.
    """
    sub = df[[x_col, y_col]].dropna()
    x = sub[x_col].values.astype(float)
    y = sub[y_col].values.astype(float)

    if len(x) < 5:
        raise ValueError("Symbolic regression requires at least 5 data points.")

    operators = ["+", "-", "*", "/", "sin", "cos"]

    def create_random_tree(depth=2):
        if depth == 0 or (depth < 2 and random.random() < 0.3):
            if random.random() < 0.6:
                return SymbolicNode("var", "x")
            else:
                return SymbolicNode("const", round(random.uniform(-5.0, 5.0), 2))
        op = random.choice(operators)
        if op in ["sin", "cos"]:
            return SymbolicNode("op", op, left=create_random_tree(depth - 1))
        return SymbolicNode("op", op, left=create_random_tree(depth - 1), right=create_random_tree(depth - 1))

    # Initialize population
    population = [create_random_tree(depth=random.randint(1, 3)) for _ in range(population_size)]

    def compute_fitness(tree: SymbolicNode) -> float:
        try:
            preds = tree.evaluate(x)
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return 1e12
            mse = float(np.mean((y - preds) ** 2))
            return mse if not math.isnan(mse) else 1e12
        except Exception:
            return 1e12

    best_tree = population[0]
    best_fitness = compute_fitness(best_tree)

    for _ in range(generations):
        scores = [(compute_fitness(t), t) for t in population]
        scores.sort(key=lambda item: item[0])
        if scores[0][0] < best_fitness:
            best_fitness = scores[0][0]
            best_tree = scores[0][1]

        # Top 20% survive
        survivors = [item[1] for item in scores[: max(2, population_size // 5)]]
        new_pop = list(survivors)
        while len(new_pop) < population_size:
            parent = random.choice(survivors)
            # Mutation: replace a subtree or generate fresh tree
            if random.random() < 0.5:
                child = create_random_tree(depth=random.randint(1, 3))
            else:
                child = parent
            new_pop.append(child)
        population = new_pop

    y_pred = best_tree.evaluate(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "equation": f"y = {best_tree.to_string()}",
        "mse": round(float(best_fitness), 4),
        "r_squared": round(float(r_squared), 4),
        "generations_run": generations,
        "x": x,
        "y": y,
        "y_pred": y_pred
    }


# -------------------------------------------------------------
# Feature 39: K-Means Clustering
# -------------------------------------------------------------

def run_kmeans_clustering(
    df: pd.DataFrame,
    features: List[str],
    n_clusters: int = 3,
    scale: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Groups data points by feature similarity using K-Means.
    """
    sub = df[features].dropna().copy()
    if len(sub) < n_clusters:
        raise ValueError("Number of samples must exceed number of clusters.")

    X = sub.values
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    result_df = sub.copy()
    result_df["Cluster"] = [f"Cluster {c}" for c in clusters]

    cluster_counts = result_df["Cluster"].value_counts().to_dict()

    free_memory()
    return result_df, {
        "n_clusters": n_clusters,
        "inertia": round(float(kmeans.inertia_), 2),
        "cluster_counts": cluster_counts,
        "cluster_centers": kmeans.cluster_centers_.tolist()
    }


# -------------------------------------------------------------
# Feature 40: Dimensionality Reduction (PCA)
# -------------------------------------------------------------

def run_pca_reduction(
    df: pd.DataFrame,
    features: List[str],
    n_components: int = 2,
    scale: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes PCA for 2D or 3D visualization.
    """
    sub = df[features].dropna().copy()
    if len(sub) < n_components or len(features) < n_components:
        raise ValueError("Insufficient features or rows for requested PCA components.")

    X = sub.values
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)

    col_names = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(coords, columns=col_names, index=sub.index)

    explained_var = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    total_var = round(float(sum(explained_var)), 2)

    free_memory()
    return pca_df, {
        "components": n_components,
        "explained_variance_ratio": explained_var,
        "total_explained_variance": total_var
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

    free_memory()
    return result_df, {
        "contamination": contamination,
        "total_analyzed": len(result_df),
        "anomalies_detected": anomaly_count,
        "anomaly_percentage": round((anomaly_count / len(result_df) * 100), 2)
    }
