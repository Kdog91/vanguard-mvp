"""
Descriptive analysis & data preprocessing.

Covers the descriptive-statistics and data-preparation toolkit:
  - Scale types (nominal / ordinal / interval / ratio)
  - Univariate: frequency tables (absolute, relative, cumulative) and statistical measures
  - Bivariate: cross-tabulations, grouped summaries, covariance and correlation
  - Multivariate: multivariate frequencies, correlation / covariance matrices
  - Data quality report
  - Scale conversion (binning, one-hot encoding, ordinal encoding)
  - Rescaling (min-max normalization, z-score standardization)
  - Transformations (log, square root, square, Box-Cox)
  - Dimensionality reduction (PCA)
  - Word frequencies for free-text columns
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Known ordered vocabularies -- a categorical column whose values all come from one
# of these is ORDINAL (its categories have a natural order), not just NOMINAL.
ORDINAL_VOCABS = [
    ["very low", "low", "medium", "high", "very high"],
    ["low", "moderate", "high"],
    ["small", "medium", "large", "extra large"],
    ["poor", "fair", "good", "very good", "excellent"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["bad", "good"],
    ["minor", "moderate", "major", "critical"],
]

SCALE_EXPLANATIONS = {
    "nominal": "Categories with no order (e.g. agency). Count them; don't average them.",
    "ordinal": "Ordered categories (e.g. Low < Medium < High). Rank them; gaps aren't equal.",
    "interval": "Numbers with equal gaps but no true zero (e.g. dates, temperature °F).",
    "ratio": "Numbers with a true zero (e.g. cost, days). All math is meaningful.",
    "identifier/text": "IDs or free text — not analyzed as a measurement.",
}


def ordinal_order(series: pd.Series):
    """Return the natural category order if the column's values match a known ordered vocabulary."""
    values = {str(v).strip().lower() for v in series.dropna().unique()}
    if not values:
        return None
    for vocab in ORDINAL_VOCABS:
        if values <= set(vocab):
            ordered = [v for v in vocab if v in values]
            # map back to the original spelling/casing used in the data
            original = {str(v).strip().lower(): v for v in series.dropna().unique()}
            return [original[v] for v in ordered]
    return None


def scale_type(series: pd.Series, kind: str) -> str:
    """Best-effort scale type for a column, given its profiled kind."""
    if kind in ("identifier", "text"):
        return "identifier/text"
    if kind == "datetime":
        return "interval"
    if kind == "categorical":
        return "ordinal" if ordinal_order(series) else "nominal"
    # numeric
    return "interval" if (series.dropna() < 0).any() else "ratio"


def scale_type_table(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        st_type = scale_type(df[col], profile.get(col, "text"))
        rows.append({"Column": col, "Detected type": profile.get(col, "?"),
                     "Scale type": st_type, "What it means": SCALE_EXPLANATIONS[st_type]})
    return pd.DataFrame(rows)


def _sorted_categories(series: pd.Series):
    order = ordinal_order(series)
    if order:
        return order
    return series.value_counts().index.tolist()  # most frequent first


def frequency_table(series: pd.Series, kind: str, bins: int | None = None) -> pd.DataFrame:
    """Absolute, relative, and cumulative frequencies.
    Numeric columns with many distinct values are grouped into bins (Sturges' rule by default)."""
    s = series.dropna()
    if kind == "numeric" and s.nunique() > 15:
        k = bins or int(np.ceil(np.log2(len(s)) + 1))  # Sturges' rule
        binned = pd.cut(s, bins=k)
        counts = binned.value_counts().sort_index()
        labels = [f"{iv.left:,.2f} – {iv.right:,.2f}" for iv in counts.index]
    elif kind == "numeric":
        counts = s.value_counts().sort_index()
        labels = [str(v) for v in counts.index]
    else:
        counts = s.value_counts().reindex(_sorted_categories(s))
        labels = [str(v) for v in counts.index]
    n = counts.sum()
    out = pd.DataFrame({
        "Value": labels,
        "Absolute frequency": counts.values.astype(int),
        "Relative frequency (%)": (counts.values / n * 100).round(2),
    })
    # cumulative frequencies only mean something when the values have an order
    if kind == "numeric" or ordinal_order(s):
        out["Cumulative absolute"] = out["Absolute frequency"].cumsum()
        out["Cumulative relative (%)"] = (out["Absolute frequency"].cumsum() / n * 100).round(2)
    return out


def summary_stats(series: pd.Series) -> pd.DataFrame:
    """Location, dispersion, and shape measures for a numeric column."""
    s = series.dropna().astype(float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    mode = s.mode()
    mean = s.mean()
    measures = [
        ("Count", len(s), "Number of values"),
        ("Mean", mean, "Average"),
        ("Median", s.median(), "Middle value — less affected by outliers than the mean"),
        ("Mode", mode.iloc[0] if len(mode) and len(mode) < len(s) else np.nan, "Most common value (blank if every value is unique)"),
        ("Minimum", s.min(), ""),
        ("Maximum", s.max(), ""),
        ("Range", s.max() - s.min(), "Maximum − minimum"),
        ("1st quartile (Q1)", q1, "25% of values are below this"),
        ("3rd quartile (Q3)", q3, "75% of values are below this"),
        ("Interquartile range (IQR)", q3 - q1, "Spread of the middle 50%"),
        ("Variance", s.var(), "Average squared distance from the mean"),
        ("Standard deviation", s.std(), "Typical distance from the mean"),
        ("Coefficient of variation (%)", (s.std() / mean * 100) if mean else np.nan, "Std. deviation relative to the mean"),
        ("Skewness", s.skew(), "0 = symmetric; > 0 = long tail to the right; < 0 = long tail to the left"),
        ("Kurtosis", s.kurt(), "0 = normal-like tails; > 0 = heavier tails / more outliers"),
    ]
    return pd.DataFrame(measures, columns=["Measure", "Value", "Meaning"])


def categorical_stats(series: pd.Series) -> pd.DataFrame:
    s = series.dropna()
    vc = s.value_counts()
    rows = [("Count", len(s)), ("Distinct categories", s.nunique()),
            ("Mode (most common)", vc.index[0] if len(vc) else None),
            ("Mode frequency", int(vc.iloc[0]) if len(vc) else None)]
    order = ordinal_order(s)
    if order:  # ordinal: the median category is meaningful
        ranks = s.map({v: i for i, v in enumerate(order)})
        rows.append(("Median category", order[int(np.floor(ranks.median()))]))
    return pd.DataFrame(rows, columns=["Measure", "Value"])


def crosstab(df: pd.DataFrame, a: str, b: str, normalize: str | None = None) -> pd.DataFrame:
    """Two-way frequency table. normalize: None (counts), 'index' (row %), 'columns' (column %), 'all' (% of total)."""
    row_order = [c for c in _sorted_categories(df[a].dropna())]
    col_order = [c for c in _sorted_categories(df[b].dropna())]
    if normalize:
        t = pd.crosstab(df[a], df[b], normalize=normalize) * 100
        t = t.round(2)
    else:
        t = pd.crosstab(df[a], df[b], margins=True, margins_name="Total")
        row_order, col_order = row_order + ["Total"], col_order + ["Total"]
    return t.reindex(index=row_order, columns=col_order)


def group_summary(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Summarize a numeric column within each category ('average cost by agency')."""
    g = df.groupby(group_col)[value_col]
    out = pd.DataFrame({
        "Count": g.count(), "Mean": g.mean(), "Median": g.median(),
        "Std. deviation": g.std(), "Minimum": g.min(), "Maximum": g.max(), "Total": g.sum(),
    }).round(2)
    order = [c for c in _sorted_categories(df[group_col].dropna()) if c in out.index]
    return out.reindex(order).reset_index()


def bivariate_numeric(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    pair = df[[x, y]].dropna()
    pr, pp = stats.pearsonr(pair[x], pair[y])
    sr, sp = stats.spearmanr(pair[x], pair[y])
    return pd.DataFrame([
        ("Covariance", pair[x].cov(pair[y]), "Direction of joint movement (scale-dependent)"),
        ("Pearson correlation (r)", pr, "Straight-line relationship, −1 to 1"),
        ("Pearson p-value", pp, "< 0.05 = unlikely to be random chance"),
        ("Spearman correlation (ρ)", sr, "Rank-based; works for curved or ordinal relationships"),
        ("Spearman p-value", sp, ""),
    ], columns=["Measure", "Value", "Meaning"])


def correlation_matrix(df: pd.DataFrame, cols: list, method: str = "pearson") -> pd.DataFrame:
    return df[cols].corr(method=method).round(3)


def covariance_matrix(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df[cols].cov().round(2)


def multivariate_frequency(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Frequency of each combination of 2+ categorical columns."""
    out = df.groupby(cols).size().reset_index(name="Absolute frequency")
    out["Relative frequency (%)"] = (out["Absolute frequency"] / out["Absolute frequency"].sum() * 100).round(2)
    return out.sort_values("Absolute frequency", ascending=False).reset_index(drop=True)


def iqr_outlier_count(series: pd.Series) -> int:
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def data_quality_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    rows = []
    for col in raw_df.columns:
        kind = profile.get(col, "removed (identifier)")
        in_clean = col in clean_df.columns
        rows.append({
            "Column": col,
            "Detected type": kind,
            "Scale type": scale_type(clean_df[col], kind) if in_clean else "identifier/text",
            "Missing in original": int(raw_df[col].isna().sum()),
            "Missing (%)": round(raw_df[col].isna().mean() * 100, 2),
            "Distinct values": int(raw_df[col].nunique()),
            "Outliers (IQR rule)": iqr_outlier_count(clean_df[col]) if in_clean and kind == "numeric" else None,
            "Kept for analysis": "yes" if in_clean else "no",
        })
    return pd.DataFrame(rows)


# ---------------------------- scale conversion ----------------------------
def bin_numeric(series: pd.Series, k: int = 4, method: str = "equal width", labels=None) -> pd.Series:
    """Numeric (ratio/interval) -> ordinal categories."""
    labels = labels or (["Low", "Medium", "High"] if k == 3 else [f"Bin {i+1}" for i in range(k)])
    if method == "equal frequency":
        return pd.qcut(series, q=k, labels=labels, duplicates="drop").astype(str)
    return pd.cut(series, bins=k, labels=labels).astype(str)


def one_hot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Nominal -> one 0/1 column per category."""
    return pd.get_dummies(df[col], prefix=col, dtype=int)


def ordinal_encode(series: pd.Series) -> pd.Series:
    """Ordinal -> rank numbers (uses the natural order if known, otherwise frequency order)."""
    order = _sorted_categories(series.dropna())
    return series.map({v: i + 1 for i, v in enumerate(order)})


# ---------------------------- rescaling & transforms ----------------------------
def rescale(series: pd.Series, method: str) -> pd.Series:
    s = series.astype(float)
    if method == "min-max":
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng else s * 0
    if method == "z-score":
        return (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else s * 0
    raise ValueError(method)


def transform(series: pd.Series, method: str) -> pd.Series:
    s = series.astype(float)
    if method == "log":
        if (s < 0).any():
            raise ValueError("Log transform needs values ≥ 0.")
        return np.log1p(s)  # log(1 + x) so zeros are allowed
    if method == "square root":
        if (s < 0).any():
            raise ValueError("Square root needs values ≥ 0.")
        return np.sqrt(s)
    if method == "square":
        return s ** 2
    if method == "Box-Cox":
        if (s <= 0).any():
            raise ValueError("Box-Cox needs strictly positive values.")
        out, _ = stats.boxcox(s.dropna())
        return pd.Series(out, index=s.dropna().index).reindex(s.index)
    raise ValueError(method)


# ---------------------------- dimensionality reduction ----------------------------
def run_pca(df: pd.DataFrame, cols: list, n_components: int):
    X = StandardScaler().fit_transform(df[cols].dropna())
    n_components = min(n_components, len(cols))
    pca = PCA(n_components=n_components).fit(X)
    names = [f"PC{i+1}" for i in range(n_components)]
    variance = pd.DataFrame({
        "Component": names,
        "Explained variance (%)": (pca.explained_variance_ratio_ * 100).round(2),
        "Cumulative (%)": (np.cumsum(pca.explained_variance_ratio_) * 100).round(2),
    })
    loadings = pd.DataFrame(pca.components_.T, index=cols, columns=names).round(3)
    scores = pd.DataFrame(pca.transform(X), columns=names, index=df[cols].dropna().index)
    return variance, loadings, scores


# ---------------------------- text ----------------------------
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the to was were will with this
these those not no but if then than so such into out up down over under about after before i you he she we they them our
your their his her my me us""".split())


def word_frequencies(series: pd.Series, top: int = 25) -> pd.DataFrame:
    words = Counter()
    for text in series.dropna().astype(str):
        for w in re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower()):
            if w not in STOPWORDS and len(w) > 2:
                words[w] += 1
    return pd.DataFrame(words.most_common(top), columns=["Word", "Frequency"])
