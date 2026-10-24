"""
Group and summarize (pandas groupby / pivot_table): split rows into groups, apply a function
to each group, and combine the results, the same split-apply-combine idea used by Spark.
"""
import pandas as pd

FUNCS = {
    "Total (sum)": "sum",
    "Average (mean)": "mean",
    "Median": "median",
    "Count of rows": "count",
    "Smallest (min)": "min",
    "Largest (max)": "max",
    "Spread (standard deviation)": "std",
    "Number of different values": "nunique",
}
PERIODS = {"(use exact dates)": None, "Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}


def _keys(df: pd.DataFrame, by: list, date_col: str | None, period: str | None):
    keys = list(by)
    work = df
    if date_col and period:
        work = df.copy()
        name = f"{date_col} ({period.lower()})"
        work[name] = pd.to_datetime(work[date_col], errors="coerce").dt.to_period(PERIODS[period]).dt.start_time
        keys = [name] + [k for k in keys if k != date_col]
    return work, keys


def group_summary(df: pd.DataFrame, by: list, values: list, funcs: list,
                  date_col: str | None = None, period: str | None = None, sort_desc: bool = True) -> pd.DataFrame:
    """One row per group, one column per (value, function). Also adds the number of rows in each group."""
    if not by and not (date_col and period):
        raise ValueError("Pick at least one column to group by.")
    work, keys = _keys(df, by, date_col, period)
    g = work.groupby(keys, dropna=False, observed=True)
    out = g.size().rename("Rows in group").to_frame()
    for v in values:
        for f in funcs:
            out[f"{v} · {f.split(' (')[0].lower()}"] = g[v].agg(FUNCS[f])
    out = out.reset_index()
    if len(out.columns) > len(keys) + 1 and sort_desc and not (date_col and period):
        out = out.sort_values(out.columns[len(keys) + 1], ascending=False)
    elif date_col and period:
        out = out.sort_values(keys)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(3)
    return out.reset_index(drop=True)


def pivot(df: pd.DataFrame, rows: str, cols: str, value: str, func: str,
          date_col: str | None = None, period: str | None = None) -> pd.DataFrame:
    """Rows × columns table of one value (pivot_table), with totals."""
    work, keys = _keys(df, [rows], date_col, period) if rows == date_col else (df, [rows])
    agg = FUNCS[func]
    margins = agg in ("sum", "count", "mean", "median", "min", "max")
    t = pd.pivot_table(work, index=keys[0], columns=cols, values=value, aggfunc=agg,
                       margins=margins, margins_name="All", observed=True)
    return t.round(3)
