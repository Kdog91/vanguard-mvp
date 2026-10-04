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


MAX_CATEGORIES = 15  # category columns with more distinct values than this are skipped


def get_numeric_feature_columns(df: pd.DataFrame, profile: dict) -> list:
    return [c for c in df.columns if profile.get(c) == "numeric"]


def build_features(df: pd.DataFrame, profile: dict, include_categorical: bool = False):
    """
    Build the matrix the unsupervised models see.
    Numeric columns are standardized (mean 0, spread 1). If include_categorical is on,
    each category column becomes a set of 0/1 columns (one-hot), scaled so that a whole
    category column carries about the same weight as one numeric column.
    Returns (matrix, numeric columns used, category columns used, category columns skipped).
    """
    num_cols = get_numeric_feature_columns(df, profile)
    parts = []
    if num_cols:
        parts.append(StandardScaler().fit_transform(df[num_cols]))
    cat_cols, skipped = [], []
    if include_categorical:
        for c in df.columns:
            if profile.get(c) != "categorical":
                continue
            n = df[c].nunique()
            if n < 2 or n > MAX_CATEGORIES:
                skipped.append(c)
                continue
            cat_cols.append(c)
            parts.append(pd.get_dummies(df[c].astype(str)).to_numpy(dtype=float) / np.sqrt(2))
    if not parts:
        raise ValueError("No usable columns were found.")
    return np.hstack(parts), num_cols, cat_cols, skipped


def run_clustering(df: pd.DataFrame, profile: dict, k_range=range(2, 7), include_categorical: bool = False):
    """
    Run K-Means across a range of k values, pick the best k by Silhouette Score,
    and return cluster assignments plus per-cluster summary statistics.
    """
    X_scaled, num_cols, cat_cols, skipped = build_features(df, profile, include_categorical)
    feature_cols = num_cols + cat_cols
    if len(feature_cols) < 2:
        raise ValueError(
            f"Clustering needs at least 2 usable columns; found {len(feature_cols)}: {feature_cols}"
        )

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
        # On big tables the score is computed on 10,000 random rows; the full calculation grows with rows squared
        score = silhouette_score(X_scaled, labels, sample_size=10_000 if len(X_scaled) > 10_000 else None, random_state=42)
        results_by_k[k] = {"model": km, "labels": labels, "silhouette": score}

    if not results_by_k:
        raise ValueError("Could not compute a valid clustering for any k in the given range.")

    best_k = max(results_by_k, key=lambda k: results_by_k[k]["silhouette"])
    best = results_by_k[best_k]

    df_out = df.copy()
    df_out["cluster"] = best["labels"]

    # Per-cluster summary: mean of each numeric feature, and cluster size
    cluster_summary = df_out.groupby("cluster")[num_cols].mean().round(2)
    for c in cat_cols:  # for category columns, show the most common value and its share of the cluster
        top = df_out.groupby("cluster")[c].agg(lambda v: v.astype(str).value_counts().index[0])
        share = df_out.groupby("cluster")[c].agg(lambda v: v.astype(str).value_counts(normalize=True).iloc[0])
        cluster_summary[f"most common {c}"] = [f"{t} ({p:.0%})" for t, p in zip(top, share)]
    cluster_summary["count"] = df_out.groupby("cluster").size()

    all_scores = {k: results_by_k[k]["silhouette"] for k in results_by_k}

    return {
        "best_k": best_k,
        "silhouette_score": best["silhouette"],
        "labels": best["labels"],
        "cluster_summary": cluster_summary,
        "feature_cols": feature_cols,
        "skipped_cols": skipped,
        "all_k_scores": all_scores,
        "df_with_clusters": df_out,
    }


def run_anomaly_detection(df: pd.DataFrame, profile: dict, contamination: float = 0.05, include_categorical: bool = False):
    """
    Run Isolation Forest to flag statistical outliers across numeric columns.
    `contamination` is the expected proportion of anomalies (default 5%).
    """
    X_scaled, num_cols, cat_cols, skipped = build_features(df, profile, include_categorical)
    feature_cols = num_cols + cat_cols

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
        "skipped_cols": skipped,
        "flagged_rows": flagged_rows,
        "df_with_scores": df_out,
        "contamination_setting": contamination,
    }
