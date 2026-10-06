"""
Chart recommendations: look at the values, not just the column type, and say which
chart shows them best and why. Each function returns (chart name, one-line reason).
The user can always pick a different chart.
"""
import pandas as pd
from scipy import stats


def _outlier_count(x: pd.Series) -> int:
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    return int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())


def one_numeric(series: pd.Series):
    """Histogram, Box plot or Density (KDE) for one numeric column."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n < 8 or x.nunique() < 3:
        return "Histogram", "there are too few distinct values for anything more detailed"
    if x.nunique() <= 10:
        return "Histogram", f"the column takes only {x.nunique()} distinct values, so counting each one is clearest"
    out = _outlier_count(x)
    skew = float(stats.skew(x))
    if out / n >= 0.02:
        return "Box plot", f"{out:,} values ({out / n:.1%}) are outliers, and a box plot shows each one"
    if abs(skew) >= 1:
        side = "high" if skew > 0 else "low"
        return "Box plot", f"the column is strongly skewed toward {side} values (skewness {skew:.2f}), which a box plot summarizes well"
    sample = x.sample(5000, random_state=42) if n > 5000 else x
    p = float(stats.shapiro(sample)[1])
    if p >= 0.05:
        return "Density (KDE)", "the values are close to a bell curve, and the density view shows how close"
    return "Histogram", "there is no strong skew and few outliers, so a histogram shows the shape most directly"


def one_categorical(series: pd.Series):
    """Bar chart or Pie chart for one category column."""
    k = int(series.nunique())
    if k <= 1:
        return "Bar chart", "there is only one category"
    if k <= 4:
        return "Pie chart", f"with only {k} categories, a pie shows each one's share of the whole"
    if k <= 8:
        return "Bar chart", f"with {k} categories, bars are easier to compare than pie slices"
    return "Bar chart", f"with {k} categories a pie would be unreadable; bars are sorted so the largest stand out"


def two_numeric(df: pd.DataFrame, a: str, b: str):
    """Scatter plot, Line chart or Area chart for two numeric columns (a on the x axis)."""
    d = df[[a, b]].dropna()
    n, distinct = len(d), d[a].nunique()
    if n and distinct <= 30 and n / distinct >= 5:
        return "Line chart", (f"{a} has only {distinct} distinct values, so a line of the average {b} at each one "
                              "is clearer than stacked points")
    if n and distinct == n and (d[a].is_monotonic_increasing or d[a].is_monotonic_decreasing):
        return "Line chart", f"{a} runs in order with no repeats, like a sequence, so a line shows the path"
    r = d[a].corr(d[b]) if n > 2 else float("nan")
    strength = ("strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.3 else "weak") if pd.notna(r) else "unclear"
    return "Scatter plot", f"each row is one point, which shows the {strength} relationship (r = {r:.2f}) and any odd rows"


def numeric_by_category(df: pd.DataFrame, num_col: str, cat_col: str):
    """'Box plot by group' or 'Bar chart of average' for a number split by a category."""
    k = int(df[cat_col].nunique())
    if k > 12:
        return "Bar", f"{k} groups are too many for side-by-side box plots; bars of the averages stay readable"
    smallest = int(df.groupby(cat_col, observed=True)[num_col].count().min()) if k else 0
    if smallest < 5:
        return "Bar", f"the smallest group has only {smallest} rows, too few to draw a meaningful box"
    return "Box", "box plots show each group's spread and outliers, which an average alone would hide"
