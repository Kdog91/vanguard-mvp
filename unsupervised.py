"""
Unsupervised Model Selector — Clustering & Anomaly Detection
These techniques don't use a target variable (Y) -- they find structure in
the data on their own. This module implements two of the six model families
from the business plan's Universal Model Selector:

  3C. Clustering (K-Means)      -- grouping similar rows without labels
  3F. Anomaly Detection (Isolation Forest) -- flagging statistical outliers
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score


def get_numeric_feature_columns(df: pd.DataFrame, profile: dict) -> list:
    """Only numeric columns are used for clustering/anomaly detection in this MVP.
    Categorical encoding for unsupervised methods is a reasonable Phase 2+ extension,
    but numeric-only keeps the first version correct and easy to validate."""
    return [c for c in df.columns if profile.get(c) == "numeric"]


def run_clustering(df: pd.DataFrame, profile: dict, k_range=range(2, 7)):
    """
    Run K-Means across a range of k values, pick the best k by Silhouette Score,
    and return cluster assignments plus per-cluster summary statistics.
    """
    feature_cols = get_numeric_feature_columns(df, profile)
    if len(feature_cols) < 2:
        raise ValueError(
            f"Clustering needs at least 2 numeric columns; found {len(feature_cols)}: {feature_cols}"
        )

    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_rows = len(df)
    max_k = min(max(k_range), n_rows - 1)
    valid_k_range = [k for k in k_range if k < n_rows and k >= 2]
    if not valid_k_range:
        raise ValueError("Not enough rows to run clustering with the requested range of k.")

    results_by_k = {}
    for k in valid_k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        # Silhouette score needs at least 2 distinct labels and fewer clusters than samples
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels)
        results_by_k[k] = {"model": km, "labels": labels, "silhouette": score}

    if not results_by_k:
        raise ValueError("Could not compute a valid clustering for any k in the given range.")

    best_k = max(results_by_k, key=lambda k: results_by_k[k]["silhouette"])
    best = results_by_k[best_k]

    df_out = df.copy()
    df_out["cluster"] = best["labels"]

    # Per-cluster summary: mean of each numeric feature, and cluster size
    cluster_summary = df_out.groupby("cluster")[feature_cols].mean().round(2)
    cluster_summary["count"] = df_out.groupby("cluster").size()

    all_scores = {k: results_by_k[k]["silhouette"] for k in results_by_k}

    return {
        "best_k": best_k,
        "silhouette_score": best["silhouette"],
        "labels": best["labels"],
        "cluster_summary": cluster_summary,
        "feature_cols": feature_cols,
        "all_k_scores": all_scores,
        "df_with_clusters": df_out,
    }


def run_anomaly_detection(df: pd.DataFrame, profile: dict, contamination: float = 0.05):
    """
    Run Isolation Forest to flag statistical outliers across numeric columns.
    `contamination` is the expected proportion of anomalies (default 5%).
    """
    feature_cols = get_numeric_feature_columns(df, profile)
    if len(feature_cols) < 1:
        raise ValueError("Anomaly detection needs at least 1 numeric column.")

    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    # -1 = anomaly, 1 = normal (sklearn convention)
    raw_preds = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)  # higher = more normal, lower = more anomalous

    df_out = df.copy()
    df_out["is_anomaly"] = raw_preds == -1
    df_out["anomaly_score"] = scores

    n_flagged = int(df_out["is_anomaly"].sum())
    flagged_rows = df_out[df_out["is_anomaly"]].sort_values("anomaly_score")

    return {
        "n_flagged": n_flagged,
        "pct_flagged": n_flagged / len(df) if len(df) else 0,
        "feature_cols": feature_cols,
        "flagged_rows": flagged_rows,
        "df_with_scores": df_out,
        "contamination_setting": contamination,
    }
