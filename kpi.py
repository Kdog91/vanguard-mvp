"""
KPIs for a business problem: customer KPIs from transaction data (length of relationship,
transaction frequency, revenue per customer, churn rate and its trend), a KPI trend for any
number over time, and the "drivers" of a target (dependent variable, DV) among the other
columns (independent variables, IVs).
"""
import numpy as np
import pandas as pd
from scipy import stats

PERIOD_CODES = {"Week": "W", "Month": "M", "Quarter": "Q"}


# ---------------------------------------------------------------- customer KPIs
def customer_table(df: pd.DataFrame, customer_col: str, date_col: str, amount_col: str | None,
                   churn_days: int, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per customer. A customer counts as churned when their last transaction is more than
    `churn_days` before the end of the data (or `as_of`)."""
    d = df[[customer_col, date_col] + ([amount_col] if amount_col else [])].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[customer_col, date_col])
    end = as_of or d[date_col].max()
    g = d.groupby(customer_col)
    out = pd.DataFrame({
        "first_date": g[date_col].min(),
        "last_date": g[date_col].max(),
        "transactions": g[date_col].nunique(),
    })
    out["relationship_days"] = (out["last_date"] - out["first_date"]).dt.days
    span_months = ((end - out["first_date"]).dt.days / 30.44).clip(lower=1)
    out["transactions_per_month"] = (out["transactions"] / span_months).round(3)
    out["avg_days_between"] = np.where(out["transactions"] > 1,
                                       out["relationship_days"] / (out["transactions"] - 1), np.nan)
    if amount_col:
        out["total_amount"] = g[amount_col].sum().round(2)
        out["avg_amount_per_transaction"] = (out["total_amount"] / out["transactions"]).round(2)
    out["days_since_last"] = (end - out["last_date"]).dt.days
    out["churned"] = np.where(out["days_since_last"] > churn_days, "Yes", "No")
    return out.reset_index()


def customer_kpis(cust: pd.DataFrame) -> dict:
    k = {
        "Customers": len(cust),
        "Churn rate (%)": round((cust["churned"] == "Yes").mean() * 100, 1),
        "Avg length of relationship (days)": round(cust["relationship_days"].mean(), 1),
        "Avg transactions per customer": round(cust["transactions"].mean(), 2),
        "Avg transactions per month": round(cust["transactions_per_month"].mean(), 2),
        "Repeat customers (%)": round((cust["transactions"] > 1).mean() * 100, 1),
    }
    if "total_amount" in cust:
        k["Avg revenue per customer"] = round(cust["total_amount"].mean(), 2)
        active = cust[cust["churned"] == "No"]["total_amount"]
        lost = cust[cust["churned"] == "Yes"]["total_amount"]
        k["Revenue from churned customers (%)"] = round(lost.sum() / cust["total_amount"].sum() * 100, 1) \
            if cust["total_amount"].sum() else 0.0
        k["_active_avg"], k["_lost_avg"] = active.mean(), lost.mean()
    return k


def churn_trend(df: pd.DataFrame, customer_col: str, date_col: str, churn_days: int, period: str = "Month") -> pd.DataFrame:
    """Churn rate per period: of the customers who bought in the period, the share who did not buy again
    within `churn_days` after their last purchase in that period. Periods too recent to know are left out."""
    d = df[[customer_col, date_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna().sort_values(date_col)
    end = d[date_col].max()
    d["per"] = d[date_col].dt.to_period(PERIOD_CODES[period])
    last_in = d.groupby([customer_col, "per"])[date_col].max().reset_index(name="last_in_period")
    dates = d.groupby(customer_col)[date_col].apply(lambda s: s.values)
    rows = []
    for per, g in last_in.groupby("per"):
        if g["last_in_period"].max() + pd.Timedelta(days=churn_days) > end:
            continue                                         # too recent to know who came back
        lost = 0
        for cust, last in zip(g[customer_col], g["last_in_period"]):
            later = dates[cust]
            nxt = later[later > np.datetime64(last)]
            if len(nxt) == 0 or (pd.Timestamp(nxt[0]) - last).days > churn_days:
                lost += 1
        rows.append({"Period": per.start_time, "Active customers": len(g), "Customers lost": lost,
                     "Churn rate (%)": round(lost / len(g) * 100, 2)})
    return pd.DataFrame(rows, columns=["Period", "Active customers", "Customers lost", "Churn rate (%)"])


def rate_trend(df: pd.DataFrame, flag_col: str, positive: str, date_col: str, period: str = "Month") -> pd.DataFrame:
    """Share of rows with flag_col == positive in each period (for a churned Yes/No column)."""
    d = df[[flag_col, date_col]].dropna().copy()
    d["Period"] = pd.to_datetime(d[date_col]).dt.to_period(PERIOD_CODES[period]).dt.start_time
    g = d.groupby("Period")[flag_col]
    return pd.DataFrame({"Rows": g.size(), "Count": g.apply(lambda s: (s.astype(str) == positive).sum()),
                         "Rate (%)": (g.apply(lambda s: (s.astype(str) == positive).mean()) * 100).round(2)}).reset_index()


# ---------------------------------------------------------------- any KPI over time
def kpi_trend(df: pd.DataFrame, date_col: str, value_col: str | None, how: str, period: str = "Month") -> pd.DataFrame:
    d = df.copy()
    d["Period"] = pd.to_datetime(d[date_col], errors="coerce").dt.to_period(PERIOD_CODES[period]).dt.start_time
    d = d.dropna(subset=["Period"])
    g = d.groupby("Period")
    s = g.size() if how == "count" else g[value_col].agg(how)
    return s.rename("Value").reset_index()


def trend_summary(t: pd.DataFrame, col: str, last_n: int = 3) -> dict:
    """Direction of a KPI: straight-line slope per period, its p-value, and the change between the
    first and last few periods."""
    y = t[col].astype(float).dropna()
    if len(y) < 4:
        return {"direction": "too few periods", "slope": np.nan, "p": np.nan, "change_pct": np.nan}
    # leave out the final period if it is clearly partial
    x = np.arange(len(y))
    res = stats.linregress(x, y.values)
    n = min(last_n, len(y) // 2)
    a, b = y.iloc[:n].mean(), y.iloc[-n:].mean()
    change = (b - a) / abs(a) * 100 if a else np.nan
    if res.pvalue < 0.05:
        direction = "rising" if res.slope > 0 else "falling"
    else:
        direction = "no clear trend"
    return {"direction": direction, "slope": float(res.slope), "p": float(res.pvalue), "change_pct": float(change)}


# ---------------------------------------------------------------- drivers of the DV
def drivers(df: pd.DataFrame, dv: str, ivs: list, profile: dict, positive: str | None = None) -> pd.DataFrame:
    """How strongly each IV is linked to the DV, with a significance test.
    Two-outcome DV: numbers compared between the two groups (t-test), categories by rate per group (chi-square).
    Number DV: Spearman correlation for numbers, ANOVA for categories."""
    rows = []
    y = df[dv]
    binary = profile.get(dv) == "categorical" and y.nunique() == 2
    if binary:
        positive = positive or sorted(y.astype(str).unique())[-1]
        is_pos = y.astype(str) == positive
    for iv in ivs:
        if iv == dv or iv not in df.columns:
            continue
        t = profile.get(iv)
        x = df[iv]
        try:
            if binary and t == "numeric":
                a, b = x[is_pos].dropna(), x[~is_pos].dropna()
                if len(a) < 3 or len(b) < 3:
                    continue
                stat, p = stats.ttest_ind(a, b, equal_var=False)
                pooled = np.sqrt((a.var() + b.var()) / 2) or np.nan
                rows.append({"Column (IV)": iv, "Test": "t-test",
                             "What the data shows": f"average {a.mean():,.2f} when {dv} = {positive} vs {b.mean():,.2f} otherwise",
                             "Strength": abs(a.mean() - b.mean()) / pooled if pooled else np.nan, "p-value": p})
            elif binary and t == "categorical" and x.nunique() <= 50:
                tab = pd.crosstab(x, is_pos)
                if tab.shape[1] < 2:
                    continue
                chi2, p, _, _ = stats.chi2_contingency(tab)
                rate = (tab[True] / tab.sum(axis=1) * 100).sort_values()
                v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
                rows.append({"Column (IV)": iv, "Test": "chi-square",
                             "What the data shows": f"{dv} = {positive}: {rate.iloc[-1]:.0f}% for '{rate.index[-1]}' "
                                                    f"vs {rate.iloc[0]:.0f}% for '{rate.index[0]}'",
                             "Strength": v, "p-value": p})
            elif not binary and profile.get(dv) == "numeric" and t == "numeric":
                pair = df[[iv, dv]].dropna()
                r, p = stats.spearmanr(pair[iv], pair[dv])
                rows.append({"Column (IV)": iv, "Test": "Spearman correlation",
                             "What the data shows": f"{dv} tends to {'rise' if r > 0 else 'fall'} as {iv} rises (r = {r:.2f})",
                             "Strength": abs(r), "p-value": p})
            elif not binary and profile.get(dv) == "numeric" and t == "categorical" and x.nunique() <= 50:
                groups = [g[dv].dropna() for _, g in df.groupby(iv, observed=True) if len(g) >= 3]
                if len(groups) < 2:
                    continue
                f, p = stats.f_oneway(*groups)
                means = df.groupby(iv, observed=True)[dv].mean().sort_values()
                grand = df[dv].var() * (len(df) - 1)
                between = sum(len(g) * (g.mean() - df[dv].mean()) ** 2 for g in groups)
                rows.append({"Column (IV)": iv, "Test": "ANOVA",
                             "What the data shows": f"average {dv} from {means.iloc[0]:,.2f} ('{means.index[0]}') "
                                                    f"to {means.iloc[-1]:,.2f} ('{means.index[-1]}')",
                             "Strength": np.sqrt(between / grand) if grand else np.nan, "p-value": p})
        except Exception:
            continue
    out = pd.DataFrame(rows, columns=["Column (IV)", "Test", "What the data shows", "Strength", "p-value"])
    if out.empty:
        return out
    out["Significant (p < 0.05)"] = np.where(out["p-value"] < 0.05, "Yes", "No")
    out["Strength"] = out["Strength"].round(3)
    out["p-value"] = out["p-value"].map(lambda v: f"{v:.4f}" if v >= 0.0001 else "< 0.0001")
    return out.sort_values("Strength", ascending=False).reset_index(drop=True)
