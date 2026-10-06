"""
Hands-on data preparation screens: missing-value methods, row filtering, column editing.
Each screen works on a copy and offers the result as a download; the main analysis keeps
using the automatically cleaned data.
"""
import pandas as pd
import streamlit as st

from data_cleaner import clean_currency_and_numbers

FILL_METHODS = [
    "Forward fill (copy the value above)",
    "Backward fill (copy the value below)",
    "Median",
    "Mean",
    "Most common value",
    "Drop rows with a gap",
]


def fill_missing(df: pd.DataFrame, col: str, method: str, sort_by: str | None = None) -> pd.DataFrame:
    """Return a copy of df with the gaps in `col` handled by `method`."""
    out = df.sort_values(sort_by, kind="stable") if sort_by else df.copy()
    s = out[col]
    if method.startswith("Forward"):
        out[col] = s.ffill()
    elif method.startswith("Backward"):
        out[col] = s.bfill()
    elif method == "Median":
        out[col] = s.fillna(s.median())
    elif method == "Mean":
        out[col] = s.fillna(s.mean())
    elif method.startswith("Most common"):
        mode = s.mode(dropna=True)
        if not mode.empty:
            out[col] = s.fillna(mode.iloc[0])
    elif method.startswith("Drop"):
        out = out[s.notna()]
    else:
        raise ValueError(method)
    return out


def _download(df, name, label="Download result as CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv", key=f"dl_{name}")


def render_missing(raw_df: pd.DataFrame):
    st.markdown("**Missing values: count them and try a different method**")
    df = raw_df.copy()
    for c in df.columns:                       # "$1,200" -> 1200 so numeric methods apply
        df[c] = clean_currency_and_numbers(df[c])
    counts = pd.DataFrame({"Column": df.columns, "Missing": df.isna().sum().values,
                           "Missing (%)": (df.isna().mean().values * 100).round(2)})
    st.dataframe(counts, hide_index=True, width="stretch")
    gaps = counts.loc[counts["Missing"] > 0, "Column"].tolist()
    if not gaps:
        st.info("This file has no missing values.")
        return
    st.caption("Automatic cleaning fills numeric gaps with the median and category gaps with the most common value. "
               "Here you can try another method on the original data and download the result.")
    c1, c2, c3 = st.columns(3)
    col = c1.selectbox("Column with gaps", gaps, key="mv_col")
    numeric = pd.api.types.is_numeric_dtype(df[col])
    methods = FILL_METHODS if numeric else [m for m in FILL_METHODS if m not in ("Median", "Mean")]
    method = c2.selectbox("Method", methods, key="mv_method")
    sort_by = None
    if method.startswith(("Forward", "Backward")):
        pick = c3.selectbox("Order rows by (optional)", ["File order"] + [c for c in df.columns if c != col], key="mv_sort",
                            help="Forward and backward fill copy a neighbouring row's value, so row order matters. "
                                 "For data over time, order by the date column.")
        sort_by = None if pick == "File order" else pick
    result = fill_missing(df, col, method, sort_by)
    was_missing = df[col].isna()
    left = int(result[col].isna().sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Gaps before", f"{int(was_missing.sum()):,}")
    m2.metric("Gaps after", f"{left:,}")
    m3.metric("Rows after", f"{len(result):,}")
    if left:
        st.caption(f"{left} gap(s) remain: forward fill cannot fill a gap in the first row, and backward fill cannot "
                   "fill one in the last row, because there is no neighbour to copy.")
    if not method.startswith("Drop"):
        shown = result.loc[was_missing.reindex(result.index)].head(10)
        st.write("First rows that had a gap, after filling:")
        st.dataframe(shown, width="stretch")
    _download(result, "missing_values_handled.csv")


def render_filter(df: pd.DataFrame, profile: dict):
    st.markdown("**Filter rows: keep only the rows that match a condition**")
    cols = [c for c, k in profile.items() if k in ("numeric", "categorical", "datetime") and c in df.columns]
    if not cols:
        st.warning("No columns to filter on.")
        return
    col = st.selectbox("Column", cols, key="flt_col")
    kind = profile[col]
    if kind == "numeric":
        lo, hi = float(df[col].min()), float(df[col].max())
        if lo == hi:
            st.info("Every row has the same value in this column.")
            return
        a, b = st.slider("Keep values between", lo, hi, (lo, hi), key="flt_num")
        mask = df[col].between(a, b)
    elif kind == "datetime":
        dates = pd.to_datetime(df[col], errors="coerce")
        lo, hi = dates.min().date(), dates.max().date()
        picked = st.date_input("Keep dates between", (lo, hi), min_value=lo, max_value=hi, key="flt_date")
        if not isinstance(picked, (tuple, list)) or len(picked) != 2:
            st.info("Pick a start and an end date.")
            return
        mask = dates.dt.date.between(picked[0], picked[1])
    else:
        options = sorted(df[col].astype(str).unique())
        keep = st.multiselect("Keep these values", options, default=options, key="flt_cat")
        mask = df[col].astype(str).isin(keep)
    result = df[mask]
    m1, m2 = st.columns(2)
    m1.metric("Rows kept", f"{len(result):,}", help=f"Out of {len(df):,}.")
    m2.metric("Rows filtered out", f"{len(df) - len(result):,}")
    st.dataframe(result.head(50), width="stretch")
    st.caption("Showing up to 50 rows. To analyze only these rows, download the result and upload it as a new file.")
    _download(result, "filtered_rows.csv")


def render_columns(df: pd.DataFrame, profile: dict):
    st.markdown("**Edit columns: remove, rename, or add a calculated column**")
    work = df.copy()
    drop = st.multiselect("Remove columns", list(work.columns), key="col_drop")
    work = work.drop(columns=drop)

    c1, c2 = st.columns(2)
    old = c1.selectbox("Rename a column", ["(none)"] + list(work.columns), key="col_old")
    new = c2.text_input("New name", key="col_new").strip()
    if old != "(none)" and new:
        if new in work.columns:
            st.error(f"A column named '{new}' already exists.")
        else:
            work = work.rename(columns={old: new})

    nums = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    if nums:
        st.write("Add a calculated column:")
        a1, a2, a3, a4 = st.columns([3, 2, 3, 3])
        left = a1.selectbox("First column", nums, key="calc_a")
        op = a2.selectbox("Operation", ["+", "−", "×", "÷"], key="calc_op")
        right_kind = a3.selectbox("With", ["A number"] + nums, key="calc_b")
        number = a3.number_input("Number", value=1.0, key="calc_n") if right_kind == "A number" else None
        name = a4.text_input("Name of the new column", key="calc_name").strip()
        if name:
            if name in work.columns:
                st.error(f"A column named '{name}' already exists.")
            else:
                rhs = number if right_kind == "A number" else work[right_kind]
                if op == "+":
                    work[name] = work[left] + rhs
                elif op == "−":
                    work[name] = work[left] - rhs
                elif op == "×":
                    work[name] = work[left] * rhs
                else:
                    work[name] = work[left] / (rhs if right_kind == "A number" and rhs != 0 else
                                               (float("nan") if right_kind == "A number" else rhs.replace(0, float("nan"))))
                    st.caption("Division by zero gives an empty value.")
    st.write(f"Result: {work.shape[0]:,} rows × {work.shape[1]} columns")
    st.dataframe(work.head(50), width="stretch")
    st.caption("Showing up to 50 rows. To analyze the edited table, download it and upload it as a new file.")
    _download(work, "edited_columns.csv")
