"""Streamlit screen: Combine & standardize sources (multi-site, multi-method measurements)."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import standardize as sz
from charts import forecast_chart
from forecasting import run_forecast
from data_cleaner import load_file
from explore_ui import _download

BLUE = "#2a78d6"
RED = "#e34948"


@st.cache_data(show_spinner=False)
def _run(df, source, date, value, split, weight, log, z_flag, gap_days, period, combine):
    return sz.standardize(df, source, date, value, list(split), weight, log=log, z_flag=z_flag,
                          gap_days=gap_days, period=period, combine=combine)


def _date_like(df: pd.DataFrame, profile: dict) -> list:
    cols = [c for c, t in profile.items() if t == "datetime"]
    for c in df.columns:
        if (c not in cols and not pd.api.types.is_numeric_dtype(df[c])
                and any(k in c.lower() for k in ("date", "time", "week", "period"))):
            if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.8:
                cols.append(c)
    return cols


def render_standardize(clean_df: pd.DataFrame, profile: dict, source_name: str):
    st.write("Make measurements from **different sites, labs or methods comparable**, check their quality, and "
             "combine them into **one index**. Each series is put on its own robust scale (median and median "
             "absolute deviation), then compared with what the other series did in the same week, so a real "
             "regional rise is kept while a bad reading or a lab change is flagged.")

    df = clean_df.copy()
    extra = st.file_uploader("Add more files with the same columns (optional)", accept_multiple_files=True,
                             type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"], key="sz_extra",
                             help="Each file's name is kept in a 'source_file' column, so you can use it as the source.")
    if extra:
        parts = [df.assign(source_file=source_name)]
        for f in extra:
            try:
                parts.append(load_file(f).assign(source_file=f.name))
            except Exception as e:
                st.warning(f"Skipped {f.name}: {e}")
        common = set(parts[0].columns)
        for p in parts[1:]:
            common &= set(p.columns)
        missing = sorted(set().union(*[set(p.columns) for p in parts]) - common)
        df = pd.concat([p[[c for c in parts[0].columns if c in common]] for p in parts], ignore_index=True)
        st.caption(f"Combined {len(parts)} files: {len(df):,} rows."
                   + (f" Columns not in every file were left out: {', '.join(missing[:8])}." if missing else ""))
        profile = {**profile, "source_file": "categorical"}

    dates = _date_like(df, profile)
    nums = [c for c, t in profile.items() if t == "numeric" and c in df.columns]
    cats = [c for c in df.columns if profile.get(c) in ("categorical", "text") or c == "source_file"]
    if not dates or not nums or not cats:
        st.warning("This needs a **date** column, a **number** column (the measurement) and a **source** column "
                   "(site, plant, lab or location). Try the sample **Multi-site measurements** to see how it works.")
        return

    c1, c2, c3 = st.columns(3)
    source = c1.selectbox("Source (site, plant, lab)", cats, key="sz_source",
                          index=next((i for i, c in enumerate(cats) if any(k in c.lower() for k in ("site", "plant", "key", "source", "location"))), 0))
    date = c2.selectbox("Date", dates, key="sz_date")
    value = c3.selectbox("Measurement", nums, key="sz_value",
                         index=next((i for i, c in enumerate(nums) if any(k in c.lower() for k in ("conc", "value", "level", "count", "rate"))), 0))
    c4, c5 = st.columns(2)
    split = c4.multiselect("Split each source by (optional)", [c for c in cats if c != source], key="sz_split",
                           help="Use for a lab or normalization method. Readings made different ways are never "
                                "put on the same scale.")
    w_opts = ["(equal weight)"] + [c for c in nums if c != value]
    weight = c5.selectbox("Weight sources by (optional)", w_opts, key="sz_weight",
                          index=next((i for i, c in enumerate(w_opts) if "pop" in c.lower()), 0), help="For example population served, so a large plant counts more.")
    weight = None if weight.startswith("(") else weight

    v = pd.to_numeric(df[value], errors="coerce")
    pos = v[v > 0]
    span = float(np.log10(pos.max() / pos.min())) if len(pos) > 1 else 0.0
    with st.expander("Settings", expanded=False):
        log = st.checkbox("Log scale (recommended when values span many orders of magnitude)", value=span >= 3,
                          key="sz_log", help=f"These values span about {span:.0f} orders of magnitude.")
        s1, s2, s3 = st.columns(3)
        z_flag = s1.slider("Outlier threshold (robust z)", 3.0, 8.0, 4.5, 0.5, key="sz_z")
        gap_days = s2.slider("Flag gaps longer than (days)", 7, 60, 21, 1, key="sz_gap")
        period = s3.selectbox("Index period", ["W", "M"], format_func=lambda p: {"W": "Weekly", "M": "Monthly"}[p],
                              key="sz_period")
        combine = st.radio("Combine series with", ["Weighted mean", "Mean", "Median"] if weight else ["Mean", "Median"],
                           horizontal=True, key="sz_combine")

    with st.spinner("Standardizing..."):
        r = _run(df, source, date, value, tuple(split), weight, log, z_flag, gap_days, period, combine)
    sm, qa, idx, rows = r["summary"], r["qa"], r["index"], r["rows"]

    m = st.columns(5)
    m[0].metric("Sources", f"{sm['sources']:,}")
    m[1].metric("Series", f"{sm['series']:,}", help="A series is one source measured one way.")
    m[2].metric("Outliers flagged", f"{sm['outliers']:,}")
    m[3].metric(f"Gaps over {gap_days} days", f"{sm['gaps']:,}")
    m[4].metric("Step changes", f"{sm['step_changes']:,}", help="A lasting jump against the other series, "
                                                                          "often a lab or method switch. Check these.")
    notes = [f"{sm['rows_used']:,} of {sm['rows_in']:,} readings used"]
    if sm["dropped_missing"]:
        notes.append(f"{sm['dropped_missing']:,} had no date or value")
    if sm["dropped_nonpositive"]:
        notes.append(f"{sm['dropped_nonpositive']:,} were zero or negative and cannot be logged")
    if sm["orders_of_magnitude"] >= 3:
        notes.append(f"raw values span about {sm['orders_of_magnitude']:.0f} orders of magnitude, which is why each "
                     "series gets its own scale")
    st.caption("; ".join(notes) + ". Flagged readings stay in the data; they are left out of the index only.")

    st.markdown("**Combined index**")
    base = alt.Chart(idx).encode(x=alt.X("period:T", title=None))
    line = base.mark_line(color=BLUE, strokeWidth=2).encode(
        y=alt.Y("index:Q", title="Index (0 = typical)"),
        tooltip=[alt.Tooltip("period:T", title="Period"), alt.Tooltip("index:Q", title="Index", format=".2f"),
                 alt.Tooltip("sources:Q", title="Series reporting")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#a3a29c", strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart((zero + line).properties(height=300), width="stretch")
    st.caption(f"Each period's value is the {sm['index_method']} of the series' averages. "
               "0 is the region's typical level; +1 is one typical spread above it. Hover to see how many "
               "series reported; fewer series means a less certain value.")

    _forecast_index(idx, period)

    st.markdown("**Quality check by series**")
    qa_show = qa.copy()
    qa_show["Step size (z)"] = qa["Step size (z)"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    st.dataframe(qa_show, hide_index=True, width="stretch")

    st.markdown("**Look at one series**")
    pick = st.selectbox("Series", qa["Series"].tolist(), key="sz_pick",
                        index=int(qa["Outliers flagged"].to_numpy().argmax()) if len(qa) else 0)
    one = rows[rows["series"] == pick][["date_parsed", "robust_z", "flag_outlier"]].copy()
    one["Reading"] = np.where(one["flag_outlier"], "Flagged", "Kept")
    pts = alt.Chart(one).mark_circle(size=40).encode(
        x=alt.X("date_parsed:T", title=None), y=alt.Y("robust_z:Q", title="Robust z"),
        color=alt.Color("Reading:N", scale=alt.Scale(domain=["Kept", "Flagged"], range=[BLUE, RED])),
        tooltip=[alt.Tooltip("date_parsed:T", title="Date"), alt.Tooltip("robust_z:Q", format=".2f"), "Reading"])
    layers = pts + alt.Chart(one).mark_line(color=BLUE, opacity=0.4).encode(x="date_parsed:T", y="robust_z:Q")
    cp = qa.loc[qa["Series"] == pick, "Possible step change"]
    if len(cp) and cp.iloc[0] != "":
        layers += alt.Chart(pd.DataFrame({"d": [pd.Timestamp(cp.iloc[0])]})).mark_rule(
            color="#eda100", strokeWidth=2).encode(x="d:T")
        st.caption(f"Orange line: possible step change on {cp.iloc[0]}. Check whether the lab or method changed then.")
    st.altair_chart(layers.properties(height=260), width="stretch")

    d1, d2, d3 = st.columns(3)
    with d1:
        _download(rows, "standardized_readings.csv", "Download standardized readings")
    with d2:
        _download(qa, "quality_by_series.csv", "Download quality table")
    with d3:
        _download(idx, "combined_index.csv", "Download index")

def _forecast_index(idx: pd.DataFrame, period: str):
    """Forecast the combined index a few periods ahead, tested against a naive guess."""
    st.markdown("**Forecast this index**")
    if len(idx) < 12:
        st.caption("Forecasting needs at least 12 periods of index values.")
        return
    unit = {"W": "week", "M": "month"}[period]
    f1, f2 = st.columns([1, 2])
    horizon = f1.number_input(f"{unit.capitalize()}s ahead", 1, 12, 4 if period == "W" else 3, key="sz_h")
    if not f2.button("Forecast this index", key="sz_fc", type="primary"):
        st.caption("Compares several forecasting methods on the most recent periods (held back, so the methods never "
                   "see them), then forecasts with the one that did best.")
        return
    s = idx.set_index("period")["index"].astype(float).sort_index()
    rule = "W-MON" if period == "W" else "MS"
    s = s.resample(rule).mean().interpolate(limit_area="inside")
    s.index.freq = None
    try:
        with st.spinner("Comparing forecasting methods..."):
            fc = run_forecast(s, int(horizon), "Week" if period == "W" else "Month")
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        return
    comp = fc["comparison"]
    best = comp.iloc[0]
    naive = comp[comp["Method"].str.startswith("Naive")]
    if len(naive) and best["Method"] != naive.iloc[0]["Method"]:
        gain = (1 - best["MAE"] / naive.iloc[0]["MAE"]) * 100 if naive.iloc[0]["MAE"] else 0
        verdict = (f"It beat the simple guess (next {unit} = this {unit}) by {gain:.0f}% on the held-back {unit}s."
                   if gain > 0 else "It was no better than the simple guess (next = this).")
    else:
        verdict = (f"No method beat the simple guess (next {unit} = this {unit}), so the forecast repeats the latest "
                   "value. That is an honest baseline, not a failure: it is what better models must beat.")
    st.success(f"**Best method: {best['Method']}**, typically off by {best['MAE']:.2f} index points on the last "
               f"{fc['holdout']} {unit}s. {verdict}")
    st.altair_chart(forecast_chart(fc["history"], fc["forecast"], "Index"), width="stretch")
    st.caption("Shaded band = rough range, from how far off the method was on the held-back periods. "
               "A guide, not a guarantee.")
    out = fc["forecast"].copy()
    out["Period"] = out["Period"].dt.strftime("%Y-%m-%d")
    st.dataframe(out.round(2), hide_index=True, width="stretch")
    with st.expander("Method comparison"):
        st.dataframe(comp.drop(columns=["MAPE (%)"], errors="ignore").round(3), hide_index=True, width="stretch")
    _download(out.round(3), "index_forecast.csv", "Download index forecast")
