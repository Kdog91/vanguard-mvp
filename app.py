"""
GovData Analytics — MVP
Universal data cleaner, descriptive analytics, model selector, forecasting,
clustering, anomaly detection and significance tests.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import tempfile

from data_cleaner import load_file, clean_dataframe
from model_selector import run_pipeline, detect_problem_type
from unsupervised import run_clustering, run_anomaly_detection
from explore_ui import render_explore, render_prepare
from forecasting import FREQS, suggest_frequency, prepare_series, series_from_daily, run_forecast
import bigdata
from charts import forecast_chart
from statistical_tests import (
    chi_square_independence, chi_square_goodness_of_fit,
    two_sample_ttest, one_way_anova, correlation_test,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ON_CLOUD = HERE.replace("\\", "/").startswith("/mount/src")   # Streamlit Community Cloud: no access to the visitor's disk
LOGO = os.path.join(HERE, "assets", "logo_mark.png")
SAMPLES = {
    "Contracts (305 rows)": "sample_contracts.csv",
    "Awards over time (1,383 rows)": "sample_awards_timeseries.csv",
}

st.set_page_config(page_title="Vanguard Data Analytics | GovData Analytics",
                   page_icon=LOGO if os.path.exists(LOGO) else None, layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  [data-testid="stMetric"] {background: #f2f5f9; border: 1px solid #e1e7ef; border-radius: 10px; padding: 14px 16px;}
  [data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #4a5b70;}
  [data-testid="stMetricValue"] {font-size: 1.55rem; color: #15365b;}
  .vg-brand {display: flex; align-items: center; gap: 14px; margin-bottom: 0.2rem;}
  .vg-brand .name {font-size: 1.9rem; font-weight: 700; color: #15365b; line-height: 1.1;}
  .vg-brand .by {font-size: 0.9rem; color: #4a5b70;}
  .vg-card {border: 1px solid #e1e7ef; border-radius: 10px; padding: 16px 18px; height: 100%; background: #ffffff;}
  .vg-card h4 {margin: 0 0 6px 0; color: #15365b; font-size: 1.02rem;}
  .vg-card p {margin: 0; color: #3c4d63; font-size: 0.92rem; line-height: 1.45;}
  .vg-section {font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #1f5fb0; margin: 1.4rem 0 0.2rem 0;}
</style>
""", unsafe_allow_html=True)

head_logo, head_text = st.columns([1, 14], vertical_alignment="center")
if os.path.exists(LOGO):
    head_logo.image(LOGO, width=64)
head_text.markdown('<div class="vg-brand"><div><div class="name">Vanguard Data Analytics</div>'
                   '<div class="by">GovData Analytics · clean, describe, predict and forecast from one file</div>'
                   '</div></div>', unsafe_allow_html=True)


def section(label):
    st.markdown(f'<div class="vg-section">{label}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _big_summary(path, stamp):
    return bigdata.summarize_file(path)


@st.cache_data(show_spinner=False)
def _big_sample(path, stamp, n):
    return bigdata.load_sample(path, n)


@st.cache_data(show_spinner=False)
def _big_daily(path, stamp, date_col, value_col, types):
    return bigdata.daily_totals(path, date_col, value_col, types)


def _spool_upload(uploaded):
    """Write a big upload to disk once, so it can be read in pieces instead of held in memory."""
    key = f"spool::{uploaded.name}::{uploaded.size}"
    if key not in st.session_state or not os.path.exists(st.session_state[key]):
        ext = os.path.splitext(uploaded.name)[1].lower()
        fd, path = tempfile.mkstemp(suffix=ext, prefix="vanguard_upload_")
        with os.fdopen(fd, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state[key] = path
    return st.session_state[key]


# --- Data source (sidebar) ---
with st.sidebar:
    st.markdown("### Data")
    uploaded_file = st.file_uploader("Upload a file",
                                     type=["csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl", "ndjson", "parquet"],
                                     help="CSV, TSV, TXT, Excel, JSON or Parquet, up to 200 MB.")
    sample_choice = st.selectbox("Or try sample data", ["None"] + list(SAMPLES), key="sample_choice",
                                 help="Made-up data that ships with the app, so you can see every feature work.")
    local_path, sample_rows = "", 20_000
    if ON_CLOUD:
        st.caption("Files with millions of rows: run the app on your own computer to read them straight from disk.")
    else:
        with st.expander("Very large file (millions of rows)"):
            local_path = st.text_input(
                "Full path to the file", placeholder=r"C:\Users\you\data\big_file.csv",
                help="For files over the 200 MB upload limit. Works when the app runs on your own computer; "
                     "CSV, TSV, TXT, Parquet or JSON Lines.").strip().strip('"')
            sample_rows = st.select_slider(
                "Rows to sample for charts and models", options=[10_000, 20_000, 50_000, 100_000], value=20_000,
                format_func=lambda v: f"{v:,}",
                help="Large files are cleaned and summarized in full. Charts, models and tests use a random sample "
                     "of this many rows so they finish in reasonable time. More rows = slower: comparing models "
                     "on 50,000 rows can take several minutes.")

sample_path = None
if uploaded_file is None and not local_path and sample_choice != "None":
    sample_path = os.path.join(HERE, SAMPLES[sample_choice])
    if not os.path.exists(sample_path):
        st.error(f"Sample file {SAMPLES[sample_choice]} is missing from the app folder.")
        st.stop()

big = None          # set when large-file mode is active: {"path", "stamp", "summary"}
big_path = None
if local_path:
    if not os.path.isfile(local_path):
        st.error("That file was not found. Check the path, or use the upload box above.")
        st.stop()
    if not local_path.lower().endswith(bigdata.SUPPORTED):
        st.error("Reading from a path supports CSV, TSV, TXT, Parquet and JSON Lines files.")
        st.stop()
    big_path = local_path
elif (uploaded_file is not None and uploaded_file.size > bigdata.LARGE_FILE_BYTES
      and uploaded_file.name.lower().endswith(bigdata.SUPPORTED)):
    big_path = _spool_upload(uploaded_file)

if big_path:
    stamp = (os.path.getsize(big_path), os.path.getmtime(big_path))
    try:
        with st.spinner("Scanning the whole file..."):
            summary = _big_summary(big_path, stamp)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
    if summary["rows"] > bigdata.LARGE_ROW_THRESHOLD or local_path:
        big = {"path": big_path, "stamp": stamp, "summary": summary}

if uploaded_file or big or sample_path:
    section("1 · Your data")
    if big:
        summary = big["summary"]
        st.success(f"Large-file mode: {summary['rows']:,} rows × {summary['columns']} columns "
                   f"({summary['size_mb']:,.0f} MB), scanned in {summary['seconds']:.1f} seconds.")
        with st.expander("Full-file summary (every row)", expanded=True):
            st.dataframe(summary["table"], hide_index=True)
        n_sample = min(sample_rows, summary["rows"])
        with st.spinner("Drawing a random sample..."):
            raw_df = _big_sample(big["path"], big["stamp"], n_sample)
        st.info(f"Cleaning preview, charts and models run on a random sample of **{len(raw_df):,}** of {summary['rows']:,} rows "
                f"({len(raw_df) / summary['rows']:.1%}). Results from a sample are estimates. "
                "Full-file cleaning and forecasting totals use every row.")
    else:
        try:
            if sample_path:
                with open(sample_path, "rb") as fh:
                    raw_df = load_file(fh)
            else:
                raw_df = load_file(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    if not big:
        src_name = os.path.basename(sample_path) if sample_path else uploaded_file.name
        st.markdown(f"**{src_name}** · loaded {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns"
                    + (" · sample data (synthetic)" if sample_path else ""))
    with st.expander("Preview raw data"):
        st.dataframe(raw_df.head(20))

    # --- Step 2: Clean ---
    section("2 · Automated cleaning")
    with st.spinner("Cleaning data..."):
        clean_df, profile, report, outlier_summary, dropped_identifiers = clean_dataframe(raw_df)

    import re as _re
    filled = sum(int(m.group(1)) for line in report for m in [_re.match(r"Filled (\d+) missing", line)] if m)
    flagged = sum(int(m.group(1)) for line in report for m in [_re.match(r"Flagged (\d+) statistical", line)] if m)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows after cleaning", f"{clean_df.shape[0]:,}")
    m2.metric("Columns kept", f"{clean_df.shape[1]} of {raw_df.shape[1]}")
    m3.metric("Rows removed", f"{len(raw_df) - len(clean_df):,}", help="Duplicate and completely empty rows.")
    m4.metric("Missing values filled", f"{filled:,}")
    m5.metric("Outliers flagged", f"{flagged:,}", help="Flagged with the 1.5 × IQR rule. They are kept, not removed.")
    st.write("")

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.expander("Cleaning report", expanded=True):
            for line in report:
                st.write(f"• {line}")
            if not report:
                st.write("No cleaning actions were necessary — data was already tidy.")
    with col2:
        st.write("**Column types detected:**")
        profile_df = pd.DataFrame(list(profile.items()), columns=["Column", "Type"])
        st.dataframe(profile_df, hide_index=True, height=250)

    with st.expander("Preview cleaned data"):
        st.dataframe(clean_df.head(20))

    if big:
        st.subheader("Clean the full file")
        st.write(f"The report above describes the sample. This applies the same rules to all "
                 f"{big['summary']['rows']:,} rows and saves the result as a Parquet file.")
        fc_key = f"full_clean::{big['path']}::{big['stamp']}"
        if st.button("Clean every row and save"):
            src = big["path"]
            folder = os.path.dirname(src) if local_path else tempfile.gettempdir()
            stem = os.path.splitext(os.path.basename(local_path or uploaded_file.name))[0]
            out_path = os.path.join(folder, f"{stem}_cleaned.parquet")
            with st.spinner("Cleaning every row..."):
                try:
                    full = bigdata.clean_full_file(src, profile, list(dropped_identifiers),
                                                   big["summary"]["types"], out_path)
                except Exception as e:
                    st.error(f"Full-file cleaning failed: {e}")
                    st.stop()
            st.session_state[fc_key] = full
        full = st.session_state.get(fc_key)
        if full and os.path.exists(full["out_path"]):
            st.success(f"Cleaned {full['rows_in']:,} rows in {full['seconds']:.1f} seconds. "
                       f"{full['rows_out']:,} rows × {full['columns_out']} columns saved "
                       f"({full['out_mb']:,.0f} MB).")
            for line in full["report"]:
                st.write(f"• {line}")
            st.code(full["out_path"])
            if full["out_mb"] <= 200:
                with open(full["out_path"], "rb") as f:
                    st.download_button("Download cleaned file (Parquet)", f,
                                       file_name=os.path.basename(full["out_path"]))

    # --- Step 3: Choose analysis mode (sidebar) ---
    with st.sidebar:
        st.markdown("### Analysis")
        mode = st.radio(
            "What do you want to do with this data?",
            ["Explore & describe", "Prepare & transform", "Predict a target variable", "Forecast future values",
             "Find hidden groups", "Detect anomalies", "Test statistical significance"],
            captions=["Frequency tables, statistics, charts", "Data quality, scale types, normalize, PCA",
                      "Regression or classification", "Time series; needs a date column",
                      "Clustering; no target needed", "Flag statistical outliers",
                      "Is this pattern real, or just noise?"],
            label_visibility="collapsed",
        )
    section("3 · Analysis")
    st.subheader(mode)

    # ===================== EXPLORE & DESCRIBE =====================
    if mode.startswith("Explore"):
        render_explore(clean_df, profile)

    # ===================== PREPARE & TRANSFORM =====================
    elif mode.startswith("Prepare"):
        render_prepare(raw_df, clean_df, profile)

    # ===================== MODE 1: SUPERVISED =====================
    elif mode.startswith("Predict a target"):
        numeric_or_cat_cols = [c for c, t in profile.items() if t in ("numeric", "categorical")]

        if not numeric_or_cat_cols:
            st.warning("No numeric or categorical columns available to predict. Upload a dataset with at least one such column.")
            st.stop()

        target_col = st.selectbox(
            "Select the column you want to predict (the target variable):",
            numeric_or_cat_cols
        )

        if target_col:
            problem_type = detect_problem_type(clean_df[target_col])
            st.info(f"Detected problem type: **{problem_type.upper()}** "
                    f"({'predicting a number' if problem_type == 'regression' else 'predicting a category'})")

            if st.button("Run Model Selector", type="primary"):
                with st.spinner("Splitting data, training models, and validating..."):
                    try:
                        output = run_pipeline(clean_df, target_col, profile)
                    except Exception as e:
                        st.error(f"Modeling failed: {e}")
                        st.stop()

                st.subheader("Results Dashboard")

                if output["dropped_cols"]:
                    st.caption(f"Note: columns excluded from modeling (text/datetime, not yet supported in Phase 1): "
                               f"{', '.join(output['dropped_cols'])}")

                results = output["results"]
                best = output["best_model"]

                if output["problem_type"] == "regression":
                    summary_rows = []
                    for name, r in results.items():
                        summary_rows.append({
                            "Model": name + (" ⭐ Best" if name == best else ""),
                            "RMSE (lower is better)": round(r["RMSE"], 3),
                            "MAE": round(r["MAE"], 3),
                            "R²": round(r["R2"], 3),
                            "CV RMSE (mean ± std)": f"{r['CV_RMSE_mean']:.3f} ± {r['CV_RMSE_std']:.3f}",
                        })
                    st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
                    st.success(f"**Best model: {best}** — lowest RMSE on held-out test data, "
                               f"confirmed stable across 5-fold cross-validation.")

                else:
                    summary_rows = []
                    for name, r in results.items():
                        summary_rows.append({
                            "Model": name + (" ⭐ Best" if name == best else ""),
                            "Accuracy": round(r["Accuracy"], 3),
                            "Precision": round(r["Precision"], 3),
                            "Recall": round(r["Recall"], 3),
                            "F1 Score": round(r["F1"], 3),
                            "CV Accuracy (mean ± std)": (f"{r['CV_Accuracy_mean']:.3f} ± {r['CV_Accuracy_std']:.3f}"
                                                       if r['CV_Accuracy_mean'] is not None else "N/A (too few samples per class)"),
                        })
                    st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
                    st.success(f"**Best model: {best}** — highest F1 score on held-out test data.")

                    with st.expander(f"Confusion Matrix — {best}"):
                        cm = results[best]["ConfusionMatrix"]
                        st.write(pd.DataFrame(cm,
                                               index=[f"Actual {i}" for i in range(cm.shape[0])],
                                               columns=[f"Predicted {i}" for i in range(cm.shape[1])]))

                st.caption(
                    "Model scoring definitions — RMSE/MAE/R²: numeric prediction accuracy (regression). "
                    "Accuracy/Precision/Recall/F1: category prediction accuracy (classification). "
                    "CV = 5-fold cross-validation, confirming results aren't a fluke of one train/test split."
                )

    # ===================== FORECASTING =====================
    elif mode.startswith("Forecast"):
        st.write("This mode projects a number forward in time, for example next quarter's spending or demand. "
                 "It compares several forecasting methods on your most recent periods (held back, so the methods "
                 "never see them) and uses the one that predicted them best.")
        date_cols = [c for c, t in profile.items() if t == "datetime"]
        num_cols = [c for c, t in profile.items() if t == "numeric"]
        if not date_cols:
            st.warning("No date column was found. Forecasting needs a column of dates "
                       "(with 'date' or 'time' in its name) plus a numeric column.")
        elif not num_cols:
            st.warning("No numeric column was found to forecast.")
        else:
            c1, c2 = st.columns(2)
            date_col = c1.selectbox("Date column", date_cols)
            value_col = c2.selectbox("Number to forecast", num_cols)
            c3, c4, c5 = st.columns(3)
            freq_names = list(FREQS)
            freq_label = c3.selectbox("Group dates by", freq_names,
                                      index=freq_names.index(suggest_frequency(clean_df[date_col])))
            agg_label = c4.selectbox("Combine rows in each period as",
                                     ["Total (sum)", "Average", "Number of rows (count)"])
            horizon = c5.number_input("Periods to forecast ahead", min_value=1, max_value=36, value=6)
            agg = {"Total (sum)": "sum", "Average": "mean", "Number of rows (count)": "count"}[agg_label]

            if st.button("Run Forecast", type="primary"):
                with st.spinner("Comparing forecasting methods..."):
                    try:
                        if big:   # totals come from every row, not the sample
                            daily = _big_daily(big["path"], big["stamp"], date_col, value_col,
                                               big["summary"]["types"])
                            series = series_from_daily(daily, value_col, freq_label, agg)
                        else:
                            series = prepare_series(clean_df, date_col, value_col, freq_label, agg)
                        fc = run_forecast(series, int(horizon), freq_label)
                    except Exception as e:
                        st.error(f"Forecast failed: {e}")
                        st.stop()

                unit = freq_label.lower()
                label = f"{agg_label.split(' (')[0]} {value_col} per {unit}"
                st.subheader("Forecast Results")
                if big:
                    st.caption(f"Computed from all {big['summary']['rows']:,} rows, not the sample.")
                best_row = fc["comparison"].iloc[0]
                mape_txt = f", off by {best_row['MAPE (%)']:.1f}% on average" if pd.notna(best_row["MAPE (%)"]) else ""
                st.success(f"**Best method: {fc['best_method']}** — tested on the last {fc['holdout']} "
                           f"{unit}s of your data{mape_txt}.")
                st.altair_chart(forecast_chart(fc["history"], fc["forecast"], label), use_container_width=True)
                st.caption("Shaded band = rough range, based on how far off the method was on the held-back periods. "
                           "It is a guide, not a guarantee.")

                st.write(f"**Forecast for the next {int(horizon)} {unit}s:**")
                out = fc["forecast"].copy()
                out["Period"] = out["Period"].dt.strftime("%Y-%m-%d")
                st.dataframe(out.round(2), hide_index=True)
                st.download_button("Download forecast as CSV", out.round(2).to_csv(index=False),
                                   file_name="forecast.csv", mime="text/csv")

                st.write("**Method comparison** (lower error = better; sorted best first):")
                st.dataframe(fc["comparison"].round(2), hide_index=True)
                with st.expander(f"How the best method did on the held-back {unit}s"):
                    bt = fc["backtest"].copy()
                    bt["Period"] = bt["Period"].dt.strftime("%Y-%m-%d")
                    st.dataframe(bt.round(2), hide_index=True)
                if not fc["seasonal_tested"] and FREQS[freq_label][1] > 1:
                    st.info("Seasonal methods were skipped: they need at least two full cycles of history "
                            "before the held-back periods.")
                st.caption("MAE = average miss. RMSE = like MAE but punishes big misses more. "
                           "MAPE = average miss as a percentage. The best method is the one with the lowest RMSE.")

    # ===================== MODE 2: CLUSTERING =====================
    elif mode.startswith("Find hidden groups"):
        st.write("This mode groups similar rows together automatically — no target column needed. "
                 "Useful for finding patterns like 'high-risk vendor clusters' without pre-labeling anything.")

        use_cats = st.checkbox("Also use category columns (not just numbers)", key="clu_cats",
                               help="Off: group rows by their numeric columns only. On: categories such as agency "
                                    "also count toward which rows are similar.")
        if st.button("Run Clustering", type="primary"):
            with st.spinner("Testing multiple cluster counts and scoring each with Silhouette Score..."):
                try:
                    result = run_clustering(clean_df, profile, include_categorical=use_cats)
                except Exception as e:
                    st.error(f"Clustering failed: {e}")
                    st.stop()

            st.subheader("Clustering Results")
            st.success(f"**Best number of clusters: {result['best_k']}** "
                       f"— Silhouette Score: {result['silhouette_score']:.3f} "
                       f"(range: -1 to 1, higher means better-separated clusters)")

            with st.expander("Silhouette scores tested across different cluster counts (k)"):
                scores_df = pd.DataFrame(
                    [{"k (number of clusters)": k, "Silhouette Score": round(v, 3)}
                     for k, v in sorted(result["all_k_scores"].items())]
                )
                st.dataframe(scores_df, hide_index=True)

            st.write("**Cluster profiles** (average value of each feature per cluster, plus cluster size):")
            st.dataframe(result["cluster_summary"])

            st.caption(
                f"Features used for clustering: {', '.join(result['feature_cols'])}. "
                + ("Category columns were converted to 0/1 columns so they could be compared."
                   if use_cats else "Tick the box above to include category columns as well.")
                + (f" Skipped (too many distinct values): {', '.join(result['skipped_cols'])}." if result["skipped_cols"] else "")
            )

    # ===================== MODE 3: ANOMALY DETECTION =====================
    elif mode.startswith("Detect anomalies"):
        st.write("This mode flags statistical outliers — rows that look very different from the rest of "
                 "the dataset. Useful for catching a single unusually large contract hidden among thousands "
                 "of normal ones.")

        contamination = st.slider(
            "Expected % of data that are anomalies", min_value=1, max_value=20, value=5,
            help="Isolation Forest needs a rough estimate of what fraction of rows are outliers. "
                 "5% is a reasonable default; lower it if you expect very few anomalies."
        ) / 100.0

        use_cats_an = st.checkbox("Also use category columns (not just numbers)", key="an_cats",
                                  help="On: a row can also be flagged for an unusual combination, such as a "
                                       "category that rarely goes with those amounts.")
        if st.button("Run Anomaly Detection", type="primary"):
            with st.spinner("Scoring every row for how anomalous it is..."):
                try:
                    result = run_anomaly_detection(clean_df, profile, contamination=contamination, include_categorical=use_cats_an)
                except Exception as e:
                    st.error(f"Anomaly detection failed: {e}")
                    st.stop()

            st.subheader("Anomaly Detection Results")
            st.success(f"**Flagged {result['n_flagged']} anomalies** "
                       f"out of {len(clean_df)} rows ({result['pct_flagged']*100:.1f}%).")

            st.write("**Most anomalous rows** (sorted most anomalous first — most negative score = most unusual):")
            st.dataframe(result["flagged_rows"], hide_index=True)

            st.caption(
                f"Features used for scoring: {', '.join(result['feature_cols'])}. "
                "Anomaly score: lower (more negative) means the row is more of a statistical outlier "
                "relative to the rest of the dataset."
            )

    # ===================== MODE 4: HYPOTHESIS TESTING =====================
    elif mode.startswith("Test statistical"):
        st.write("This mode checks whether a pattern is **statistically real or could be random chance** — "
                 "useful before acting on a finding, e.g. 'do agencies really differ in average contract cost?'")
        numeric_cols = [c for c, t in profile.items() if t == "numeric"]
        categorical_cols = [c for c, t in profile.items() if t == "categorical"]

        test = st.selectbox("Which test?", [
            "Correlation test (are two numbers related?)",
            "One-way ANOVA (does a number differ across 3+ groups?)",
            "Two-sample t-test (does a number differ between 2 groups?)",
            "Chi-Square test of independence (are two categories related?)",
            "Chi-Square goodness-of-fit (is one category evenly split?)",
        ])

        def show(result, detail):
            st.subheader("Test Results")
            (st.success if result["significant"] else st.info)(result["interpretation"])
            st.write(detail)

        try:
            if test.startswith("Correlation"):
                if len(numeric_cols) < 2:
                    st.warning("Needs at least 2 numeric columns."); st.stop()
                a = st.selectbox("First numeric column", numeric_cols, key="c_a")
                b = st.selectbox("Second numeric column", [c for c in numeric_cols if c != a], key="c_b")
                if st.button("Run Test", type="primary"):
                    r = correlation_test(clean_df, a, b)
                    show(r, f"Correlation r = {r['correlation_r']:.3f}, p-value = {r['p_value']:.4f}")

            elif test.startswith("One-way ANOVA"):
                if not numeric_cols or not categorical_cols:
                    st.warning("Needs 1 numeric and 1 categorical column."); st.stop()
                n = st.selectbox("Numeric column", numeric_cols, key="a_n")
                g = st.selectbox("Group column (3+ groups)", categorical_cols, key="a_g")
                if st.button("Run Test", type="primary"):
                    r = one_way_anova(clean_df, n, g)
                    show(r, f"F-statistic = {r['f_statistic']:.3f}, p-value = {r['p_value']:.4f}")
                    st.dataframe(pd.DataFrame(list(r["group_means"].items()), columns=["Group", "Mean"]), hide_index=True)

            elif test.startswith("Two-sample"):
                if not numeric_cols or not categorical_cols:
                    st.warning("Needs 1 numeric and 1 categorical column."); st.stop()
                n = st.selectbox("Numeric column", numeric_cols, key="t_n")
                g = st.selectbox("Group column (exactly 2 groups)", categorical_cols, key="t_g")
                if st.button("Run Test", type="primary"):
                    r = two_sample_ttest(clean_df, n, g)
                    show(r, f"t-statistic = {r['t_statistic']:.3f}, p-value = {r['p_value']:.4f}")

            elif test.startswith("Chi-Square test of independence"):
                if len(categorical_cols) < 2:
                    st.warning("Needs at least 2 categorical columns."); st.stop()
                a = st.selectbox("First categorical column", categorical_cols, key="x_a")
                b = st.selectbox("Second categorical column", [c for c in categorical_cols if c != a], key="x_b")
                if st.button("Run Test", type="primary"):
                    r = chi_square_independence(clean_df, a, b)
                    show(r, f"Chi-Square = {r['chi2_statistic']:.3f}, degrees of freedom = {r['degrees_of_freedom']}, p-value = {r['p_value']:.4f}")
                    st.dataframe(r["contingency_table"])

            else:
                if not categorical_cols:
                    st.warning("Needs at least 1 categorical column."); st.stop()
                c = st.selectbox("Categorical column", categorical_cols, key="g_c")
                if st.button("Run Test", type="primary"):
                    r = chi_square_goodness_of_fit(clean_df, c)
                    show(r, f"Chi-Square = {r['chi2_statistic']:.3f}, p-value = {r['p_value']:.4f}")
                    st.dataframe(r["observed"])
        except ValueError as e:
            st.error(str(e))

        st.caption("p-value below 0.05 = conventionally 'statistically significant' (unlikely to be random chance). "
                   "It says whether an effect is likely real, not how large or important it is.")

else:
    st.write("")
    st.markdown("#### Turn a raw data file into tested answers, without writing code.")
    st.write("Upload a file in the sidebar, or pick **Or try sample data** there to see it work straight away.")
    cards = [
        ("Clean", "Fixes currency text, removes duplicates and identifier columns, fills gaps, flags outliers, "
                  "and reports every change it made."),
        ("Describe", "Frequency tables, summary statistics, cross-tabulations, correlation, and charts chosen "
                     "to fit each column's type."),
        ("Predict", "Compares up to eight models on data they have not seen, with cross-validation, and "
                    "reports which one did best."),
        ("Forecast", "Projects a dated number forward, using whichever of seven methods best predicted your "
                     "most recent periods."),
        ("Find patterns", "Groups similar rows, flags unusual ones, and tests whether a difference is real "
                          "or just chance."),
        ("Large files", "Scans and cleans files with millions of rows from disk on a single machine; charts "
                        "and models run on a random sample."),
    ]
    for row in (cards[:3], cards[3:]):
        cols = st.columns(3)
        for col, (title, text) in zip(cols, row):
            col.markdown(f'<div class="vg-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
        st.write("")
    st.caption("Accepts CSV, TSV, TXT, Excel, JSON and Parquet. Runs on one computer; distributed processing, "
               "neural networks and association rules are on the roadmap, not built yet. "
               "Sample files are synthetic.")
