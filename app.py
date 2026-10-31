"""
Vanguard Analytics: clean, describe, predict and forecast any tabular data.
Universal data cleaner, descriptive analytics, model selector, forecasting,
clustering, anomaly detection and significance tests.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import tempfile

from data_cleaner import (load_file, load_url, clean_dataframe, set_header, rename_columns, missing_table,
                          MISSING_STRATEGIES, TYPE_CHOICES)
from model_selector import run_pipeline, detect_problem_type
from unsupervised import run_clustering, run_anomaly_detection
from explore_ui import render_explore, render_prepare
from forecasting import FREQS, suggest_frequency, prepare_series, series_from_daily, run_forecast
import bigdata
import usaspending
from charts import forecast_chart
from statistical_tests import (
    chi_square_independence, chi_square_goodness_of_fit,
    two_sample_ttest, one_way_anova, correlation_test,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ON_CLOUD = HERE.replace("\\", "/").startswith("/mount/src")   # Streamlit Community Cloud: no access to the visitor's disk
LOGO = os.path.join(HERE, "assets", "logo_mark.png")
SAMPLES = {
    "Retail sales by store (2,310 rows)": "sample_retail_sales.csv",
    "Clinic visits (1,641 rows)": "sample_clinic_visits.csv",
    "Multi-site measurements (401 rows)": "sample_multisite_measurements.csv",
    "Government contracts (305 rows)": "sample_contracts.csv",
    "Contract awards over time (1,383 rows)": "sample_awards_timeseries.csv",
    "Customer reviews (1,200 rows, text)": "sample_customer_reviews.csv",
    "Store orders (7,176 rows, items per order)": "sample_store_orders.csv",
    "Bank marketing (4,521 rows, subscribed yes/no)": "sample_bank_marketing.csv",
}

st.set_page_config(page_title="Vanguard Analytics",
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

import legal
if legal.route():
    st.stop()
import security
security.sweep_temp()          # delete leftover temporary files from sessions that ended hours ago

head_logo, head_text = st.columns([1, 14], vertical_alignment="center")
if os.path.exists(LOGO):
    head_logo.image(LOGO, width=64)
head_text.markdown('<div class="vg-brand"><div><div class="name">Vanguard Analytics</div>'
                   '<div class="by">by Vanguard Data Analytics LLC · clean, describe, predict and forecast any data file</div>'
                   '</div></div>', unsafe_allow_html=True)


def section(label):
    st.markdown(f'<div class="vg-section">{label}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)  # results leave server memory within an hour
def _big_summary(path, stamp):
    return bigdata.summarize_file(path)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)  # results leave server memory within an hour
def _big_sample(path, stamp, n):
    return bigdata.load_sample(path, n)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)  # results leave server memory within an hour
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
import auth
with st.sidebar:
    auth.render_sidebar()
    legal.footer_links()
    import projects_ui
    projects_ui.render_sidebar()
    _own = auth.can_use_own_data()
    st.markdown("### Data")
    uploaded_file = None
    url_go, url_in = False, ""
    if not _own:
        auth.render_locked()
        st.session_state.pop("url_data", None)
    if _own:
        uploaded_file = st.file_uploader("Upload a file", key=f"uploader_{st.session_state.get('up_ver', 0)}",
                                         type=["csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl", "ndjson", "parquet", "orc"],
                                         help="CSV, TSV, TXT, Excel, JSON, Parquet or ORC, up to 200 MB.")
        st.caption(":warning: Do not upload health records (PHI), controlled government data (CUI), card numbers "
                   "or other data you may not share with a cloud service. See the [Terms of Use](?page=terms).")
        if uploaded_file is not None:
            try:
                security.check_upload(uploaded_file.name, uploaded_file.size)
            except security.SecurityError as e:
                st.error(str(e))
                security.audit("upload_refused", file_type=os.path.splitext(uploaded_file.name)[1].lower(),
                               size_mb=round(uploaded_file.size / 1e6, 2))
                uploaded_file = None
        if uploaded_file is not None:
            _up_key = f"audited::{uploaded_file.name}::{uploaded_file.size}"
            if not st.session_state.get(_up_key):
                st.session_state[_up_key] = True
                security.audit("upload", file_type=os.path.splitext(uploaded_file.name)[1].lower(),
                               size_mb=round(uploaded_file.size / 1e6, 2))
        with st.form("url_form", border=False):
            url_in = st.text_input("Or load from a web address", key="url_in",
                                   placeholder="https://raw.githubusercontent.com/.../data.csv",
                                   help="Like pd.read_csv(url): a link straight to a CSV, Excel, JSON or Parquet file. "
                                        "GitHub page links are converted to the raw file automatically.")
            url_go = st.form_submit_button("Load from web")
        if url_go and url_in.strip():
            try:
                with st.spinner("Downloading..."):
                    st.session_state["url_data"] = {"df": load_url(url_in), "url": url_in.strip()}
                security.audit("url_load", host=security.host_of(url_in),
                               rows=len(st.session_state["url_data"]["df"]))
            except Exception as e:
                st.error(f"Could not load that address: {e}")
                security.audit("url_load_failed", host=security.host_of(url_in),
                               reason=type(e).__name__)
        from db_ui import render_sidebar_loader
        render_sidebar_loader()
    if "url_data" in st.session_state:
        st.caption(f"Loaded: {st.session_state['url_data']['url'][:70]}")
        if st.button("Clear loaded data", key="url_clear"):
            del st.session_state["url_data"]
            st.rerun()
    sample_choice = st.selectbox("Or try sample data", ["None"] + list(SAMPLES), key="sample_choice",
                                 help="Made-up data that ships with the app, so you can see every feature work.")
    with st.expander("Public data: U.S. federal contract awards"):
        st.caption("Real awards from the U.S. Treasury's public database. No account needed.")
        usa_agency = st.selectbox("Awarding agency", usaspending.AGENCIES, key="usa_agency")
        _today = pd.Timestamp.today().normalize()
        usa_dates = st.date_input("Dates", (_today - pd.DateOffset(years=1), _today),
                                  min_value=pd.Timestamp(usaspending.EARLIEST_DATE), max_value=_today, key="usa_dates")
        usa_n = st.select_slider("How many awards", options=[200, 500, 1000, 2000, 5000], value=1000, key="usa_n",
                                 help="Each 100 awards is one request to USAspending. 1,000 takes roughly 10 to 30 seconds.")
        usa_sort = st.radio("Which awards", list(usaspending.SORT_CHOICES), key="usa_sort",
                            captions=["Recommended: an equal share from each part of the period", "", ""],
                            help="'Spread' covers the whole period. The other two take the first awards in that "
                                 "order, so 'Largest first' gives only the biggest awards. None is a random sample.")
        usa_new = st.checkbox("Only new awards signed in these dates", value=True, key="usa_new",
                              help="Uses USAspending's 'new awards only' filter: awards first signed in the period. Work on some starts "
                                   "later. Off: also includes older awards with any activity in the period, such as a modification.")
        if st.button("Load awards", key="usa_load"):
            if not isinstance(usa_dates, (tuple, list)) or len(usa_dates) != 2:
                st.error("Pick a start and an end date.")
            else:
                bar = st.progress(0.0, text="Contacting USAspending.gov...")
                try:
                    _df, _info = usaspending.fetch_awards(
                        usa_agency, str(usa_dates[0]), str(usa_dates[1]), usa_n, usa_sort, new_only=usa_new,
                        progress=lambda done, total: bar.progress(min(done / total, 1.0),
                                                                  text=f"{done:,} of up to {total:,} awards"))
                    bar.empty()
                    if _df.empty:
                        st.warning("USAspending returned no awards for that agency and period.")
                    else:
                        st.session_state["usa_data"] = {"df": _df, "info": _info}
                except usaspending.USAspendingError as e:
                    bar.empty()
                    st.error(str(e))
        if "usa_data" in st.session_state:
            _i = st.session_state["usa_data"]["info"]
            st.success(f"{_i['rows']:,} awards loaded ({_i['agency']}).")
            if st.button("Clear loaded awards", key="usa_clear"):
                del st.session_state["usa_data"]
                st.rerun()
    local_path, sample_rows = "", 20_000
    if ON_CLOUD:
        st.caption("Files with millions of rows: run the app on your own computer to read them straight from disk.")
    elif _own:
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

web = st.session_state.get("url_data") if (uploaded_file is None and not local_path) else None
usa = st.session_state.get("usa_data") if (uploaded_file is None and not local_path and web is None) else None
sample_path = None
if uploaded_file is None and not local_path and usa is None and web is None and sample_choice != "None":
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

if uploaded_file or big or sample_path or usa or web:
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
            if web:
                raw_df = web["df"].copy()
                raw_df.attrs = dict(web["df"].attrs)
            elif usa:
                raw_df = usa["df"].copy()
            elif sample_path:
                with open(sample_path, "rb") as fh:
                    raw_df = load_file(fh)
            else:
                raw_df = load_file(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    # A short name for this dataset, used to name every downloaded file (e.g. Department_of_Defense_2025-10-12_to_2026-10-12)
    import re as _re_lbl
    if web and web.get("label"):
        _lbl = web["label"]
    elif web:
        _lbl = os.path.splitext(os.path.basename(web["url"].split("?")[0]))[0] or "web_data"
    elif usa:
        _lbl = f"{usa['info']['agency']}_{usa['info']['start_date']}_to_{usa['info']['end_date']}"
    elif sample_path:
        _lbl = os.path.splitext(os.path.basename(sample_path))[0]
        _lbl = _lbl if _lbl.startswith("sample") else "sample_" + _lbl
    elif big:
        _lbl = os.path.splitext(os.path.basename(big["path"]))[0]
    elif uploaded_file is not None:
        _lbl = os.path.splitext(uploaded_file.name)[0]
    else:
        _lbl = os.path.splitext(os.path.basename(local_path or "data"))[0]
    st.session_state["dataset_label"] = _re_lbl.sub(r"[^A-Za-z0-9._-]+", "_", _lbl).strip("_")[:80] or "data"

    if usa:
        _i = usa["info"]
        st.markdown(f"**USAspending.gov contract awards** · {_i['agency']} · {_i['start_date']} to {_i['end_date']} · "
                    f"{raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns · real federal data")
        _how = (f"an equal share from each of {_i.get('slices', 1)} parts of the period"
                if _i["sort_by"] == usaspending.SPREAD else f"the first awards, {_i['sort_by'].lower()}")
        _which = "new awards signed in the period" if _i.get("new_only") else "awards active in the period"
        st.caption(f"{_i['rows']:,} {_which}: {_how}"
                   + (" (more exist)" if _i["more_available"] else " (all there are)")
                   + f", fetched in {_i['seconds']:.0f} seconds. This is not a random sample of the agency's awards. "
                     "duration_days is calculated by the app from the start and end dates.")
        _d = _i.get("date_issues") or {}
        _notes = []
        if _d.get("end_before_start"):
            _notes.append(f"**{_d['end_before_start']:,}** awards have an end date before the start date, so their "
                          "duration is left blank (the cleaner then fills it with the median)")
        if _d.get("same_day"):
            _notes.append(f"**{_d['same_day']:,}** start and end on the same day (duration 0)")
        if _notes:
            st.warning("Date problems in the source records: " + "; ".join(_notes) + ".")
        if _d.get("future_start"):
            st.info(f"**{_d['future_start']:,}** awards are scheduled to start after today: signed now, with work "
                    "beginning later. That is normal, not an error.")
        for _m in _i.get("messages", []):
            if _m.startswith("USAspending did not accept"):
                st.info(_m)
        st.download_button("Download these awards as CSV", security.csv_bytes(raw_df),
                           file_name=f"{st.session_state['dataset_label']}_awards_raw.csv", mime="text/csv", key="usa_dl")
    elif not big:
        src_name = (web["url"] if web else os.path.basename(sample_path) if sample_path else uploaded_file.name)
        st.markdown(f"**{src_name}** · loaded {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns"
                    + (" · sample data (synthetic)" if sample_path else ""))
    if raw_df.shape[1] < 2 or len(raw_df) < 3:
        st.warning("This file doesn't look like a data table (it has "
                   f"{len(raw_df):,} rows and {raw_df.shape[1]} column{'s' if raw_df.shape[1] != 1 else ''}). "
                   "The app works best with one row per record and one column per field, such as a CSV or "
                   "spreadsheet export. Results below may not be meaningful.")
    # --- Column names: detect a missing header row, let the user switch it and type real names ---
    _ds = st.session_state["dataset_label"]
    for _note in raw_df.attrs.get("import_notes", []):
        st.info(_note)
    if not big:
        with st.expander("Fix column names", expanded=not raw_df.attrs.get("header_from_file", True)):
            _auto_header = raw_df.attrs.get("header_from_file", True)
            _first_is_data = st.checkbox("The first row is data, not column names", value=not _auto_header,
                                         key=f"hdr_{_ds}",
                                         help="Some files (for example many UCI datasets) have no header row.")
            if _first_is_data == _auto_header:      # the user switched it
                raw_df = set_header(raw_df, _first_is_data)
            _names = st.text_area("Column names, separated by commas (optional)", key=f"names_{_ds}",
                                  placeholder=", ".join(map(str, raw_df.columns[:6])) + ", ...",
                                  help=f"Type {raw_df.shape[1]} names in order to rename every column.")
            if _names.strip():
                raw_df, _msg = rename_columns(raw_df, _names)
                (st.success if _msg.startswith("Renamed") else st.warning)(_msg)
    with st.expander("Preview raw data"):
        st.dataframe(raw_df.head(20))

    # --- Step 2: Clean ---
    section("2 · Automated cleaning")
    _ov_key = f"overrides::{_ds}"
    _overrides = st.session_state.get(_ov_key, {})
    with st.expander("Cleaning choices: missing values and column types", expanded=bool(_overrides)):
        _miss_tbl = missing_table(raw_df)
        if _miss_tbl.empty:
            st.caption("No missing values in this file (after treating '?', 'N/A', 'null' and similar as missing).")
        else:
            st.markdown(f"**Missing values before cleaning** ({int(_miss_tbl['Missing'].sum()):,} gaps in "
                        f"{len(_miss_tbl)} columns; '?', 'N/A', 'null' and similar count as missing)")
            st.dataframe(_miss_tbl, hide_index=True, width="stretch", height=min(250, 38 + 35 * len(_miss_tbl)))
        missing_choice = st.selectbox("How to handle missing values", MISSING_STRATEGIES, key=f"miss_{_ds}",
                                      help="Automatic is a safe default. Dropping rows loses data and can skew results "
                                           "if gaps are not random. Forward fill suits readings taken over time.")
    try:
        with st.spinner("Cleaning data..."):
            _auto = clean_dataframe(raw_df) if _overrides else None
            clean_df, profile, report, outlier_summary, dropped_identifiers = clean_dataframe(
                raw_df, missing_choice, _overrides)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    _auto_profile = dict((_auto or (None, profile))[1])
    for _c in (_auto or (None, None, None, None, dropped_identifiers))[4]:
        _auto_profile.setdefault(_c, "identifier")

    import re as _re
    filled = sum(int(m.group(1)) for line in report for m in [_re.match(r"Filled (\d+) missing", line)] if m)
    flagged = sum(int(m.group(1)) for line in report for m in [_re.match(r"Flagged (\d+) statistical", line)] if m)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows after cleaning", f"{clean_df.shape[0]:,}")
    m2.metric("Columns kept", f"{clean_df.shape[1]} of {raw_df.shape[1]}")
    m3.metric("Rows removed", f"{len(raw_df) - len(clean_df):,}", help="Duplicate and completely empty rows, "
                                                                      "plus rows with gaps if you chose to drop them.")
    m4.metric("Missing values filled", f"{filled:,}")
    m5.metric("Outliers flagged", f"{flagged:,}", help="Flagged with the 1.5 × IQR rule. They are kept, not removed.")
    st.write("")

    col1, col2 = st.columns([3, 2])
    with col1:
        with st.expander("Cleaning report", expanded=True):
            for line in report:
                st.write(f"• {line}")
            if not report:
                st.write("No cleaning actions were necessary — data was already tidy.")
    with col2:
        st.write("**Column types** (change any that are wrong)")
        _shown = {c: _overrides.get(c, _auto_profile.get(c)) for c in raw_df.columns if c in _auto_profile}
        _shown = {c: ("identifier (removed)" if t == "identifier" else t) for c, t in _shown.items()}
        _types_df = pd.DataFrame({"Column": list(_shown), "Type": list(_shown.values()),
                                  "Example": [str(raw_df[c].dropna().iloc[0])[:30] if raw_df[c].notna().any() else ""
                                              for c in _shown]})
        _edited = st.data_editor(
            _types_df, hide_index=True, width="stretch", height=min(320, 38 + 35 * len(_types_df)),
            disabled=["Column", "Example"], key=f"types_{_ds}",
            column_config={"Type": st.column_config.SelectboxColumn("Type", options=TYPE_CHOICES, required=True)})
        _new_ov = {}
        for _c, _t in zip(_edited["Column"], _edited["Type"]):
            _auto_t = _auto_profile.get(_c)
            _auto_t = "identifier (removed)" if _auto_t == "identifier" else _auto_t
            if _t != _auto_t:
                _new_ov[_c] = _t
        if _new_ov != _overrides:
            st.session_state[_ov_key] = _new_ov
            st.rerun()
        st.caption("numeric = to_numeric · categorical = astype('category') · datetime = to_datetime · "
                   "text = free text · drop column = leave it out." + (" Changed by you: " + ", ".join(_overrides)
                                                                    if _overrides else ""))

    with st.expander("Preview cleaned data"):
        st.dataframe(clean_df.head(20))

    # --- What can this data answer? ---
    import use_cases
    with st.expander("What can this data answer?", expanded=False):
        st.caption("Common business questions, checked against the columns in this file.")
        st.dataframe(use_cases.suggest(raw_df, clean_df, profile), hide_index=True, width="stretch")

    # --- Export the cleaned data, and a record of how it was made ---
    import io as _io, json as _json, hashlib as _hl, sklearn as _skl
    _stem = st.session_state["dataset_label"]
    import pipeline as _pl
    _pii = {c: k for c, k in _pl.detect_pii(clean_df).items() if c in clean_df.columns}
    _export_df = clean_df
    with st.expander("Download cleaned data", expanded=False):
        if _pii:
            _mask_on = st.checkbox("Hide private details in downloads and the Word report (recommended)", value=True,
                                   key=f"mask_{_ds}",
                                   help="Columns that look like names, emails, phone or ID numbers are replaced with "
                                        "a code. The same value always gets the same code, so rows can still be "
                                        "counted and matched.")
            st.caption("Private details found: " + ", ".join(f"**{c}** ({k})" for c, k in _pii.items()))
            if _mask_on:
                _export_df = _pl.mask_pii(clean_df, _pii, "hash", salt=_stem)
        e1, e2, e3, e4, e5, e6 = st.columns(6)
        _dl_log = lambda fmt: (lambda: security.audit("export", what="cleaned_data", format=fmt,
                                                      rows=len(_export_df), masked=_export_df is not clean_df))
        e1.download_button("CSV", security.csv_bytes(_export_df), file_name=f"{_stem}_cleaned.csv",
                           mime="text/csv", key="exp_csv", on_click=_dl_log("csv"))
        e2.download_button("Excel", security.excel_bytes(_export_df), file_name=f"{_stem}_cleaned.xlsx", key="exp_xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           on_click=_dl_log("excel"))
        e3.download_button("JSON", _export_df.to_json(orient="records", date_format="iso", indent=1).encode("utf-8"),
                           file_name=f"{_stem}_cleaned.json", mime="application/json", key="exp_json",
                           on_click=_dl_log("json"))
        _pq = _io.BytesIO()
        _export_df.to_parquet(_pq, index=False)
        e4.download_button("Parquet", _pq.getvalue(), file_name=f"{_stem}_cleaned.parquet", key="exp_pq",
                           mime="application/octet-stream", on_click=_dl_log("parquet"))
        _record = {
            "app": "Vanguard Analytics (update 26f)", "created": pd.Timestamp.now().isoformat(timespec="seconds"),
            "source": (web["url"] if web else os.path.basename(sample_path) if sample_path
                       else uploaded_file.name if uploaded_file is not None else _stem),
            "input_rows": int(len(raw_df)), "input_columns": int(raw_df.shape[1]),
            "input_fingerprint": _hl.sha256(pd.util.hash_pandas_object(raw_df.astype(str), index=False)
                                            .values.tobytes()).hexdigest(),
            "column_names": [str(c) for c in raw_df.columns],
            "settings": {"missing_values": missing_choice, "column_type_changes": _overrides,
                         "outlier_rule": "1.5 x IQR, flagged not removed", "random_seed": 42},
            "cleaning_steps": report, "output_rows": int(len(clean_df)), "column_types": profile,
            "library_versions": {"pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": _skl.__version__},
        }
        try:
            _orc = _io.BytesIO()
            from pipeline import _safe_table as _st
            _st(_export_df).to_orc(_orc, index=False)
            e5.download_button("ORC", _orc.getvalue(), file_name=f"{_stem}_cleaned.orc", key="exp_orc",
                               mime="application/octet-stream")
        except Exception:
            e5.caption("ORC n/a")
        e6.download_button("Record (JSON)", _json.dumps(_record, indent=2, default=str).encode("utf-8"),
                           file_name=f"{_stem}_reproducibility_record.json", mime="application/json", key="exp_rec",
                           help="Everything needed to repeat this cleaning: source, a fingerprint of the input, "
                                "your settings, every step, the random seed and library versions.")
        st.caption("The record lets you (or a reviewer) repeat the exact same cleaning later and check the input "
                   "file has not changed: same fingerprint, same settings, same result.")

    # --- One-click Word report ---
    _rep_key = (f"report::{st.session_state['dataset_label']}::{raw_df.shape}::{clean_df.shape}"
                f"::{_export_df is not clean_df}")
    rc1, rc2 = st.columns([1, 2])
    with rc1:
        if st.button("Build report (Word)", help="A shareable summary: what was loaded, what the cleaning changed, "
                                                  "and a plain-language description of each column with charts."):
            from report import build_report
            if usa:
                _src = (f"Source: USAspending.gov contract awards, {usa['info']['agency']}, "
                        f"{usa['info']['start_date']} to {usa['info']['end_date']}. Not a random sample of the agency's awards.")
            elif big:
                _src = f"Source: {os.path.basename(big['path'])} (random sample of {len(raw_df):,} rows from a larger file)."
            elif sample_path:
                _src = f"Source: {os.path.basename(sample_path)} (synthetic sample data)."
            else:
                _src = f"Source: {uploaded_file.name if uploaded_file is not None else 'uploaded file'}."
            _notes = []
            _di = (usa or {}).get("info", {}).get("date_issues") or {}
            if _di.get("end_before_start"):
                _notes.append(f"{_di['end_before_start']:,} records have an end date before the start date in the source.")
            if _di.get("same_day"):
                _notes.append(f"{_di['same_day']:,} records start and end on the same day.")
            if _di.get("future_start"):
                _notes.append(f"{_di['future_start']:,} records are scheduled to start after today (normal for new awards).")
            with st.spinner("Building report..."):
                _rep_raw = _pl.mask_pii(raw_df, {c: k for c, k in _pii.items() if c in raw_df.columns}, "hash",
                                        salt=_stem) if _export_df is not clean_df else raw_df
                st.session_state[_rep_key] = build_report(
                    st.session_state["dataset_label"], _src, _rep_raw, _export_df, profile, report,
                    {"rows_after": len(clean_df), "cols_kept": clean_df.shape[1], "cols_raw": raw_df.shape[1],
                     "rows_removed": len(raw_df) - len(clean_df), "filled": filled, "flagged": flagged}, _notes)
    with rc2:
        if st.session_state.get(_rep_key):
            st.download_button("Download report (.docx)", st.session_state[_rep_key],
                               file_name=f"{st.session_state['dataset_label']}_report.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               key="report_dl", on_click=lambda: security.audit("export", what="word_report"))

    # --- Save this project (update 27) ---
    if web and web.get("kind") == "project":
        _psrc = web["url"]
    elif web and web.get("kind") == "database":
        _psrc = "Database: " + web["url"]
    elif web:
        _psrc = "Web: " + web["url"].split("?")[0][:150]
    elif usa:
        _psrc = f"USAspending.gov: {usa['info']['agency']}, {usa['info']['start_date']} to {usa['info']['end_date']}"
    elif sample_path:
        _psrc = "Sample: " + os.path.basename(sample_path)
    elif uploaded_file is not None:
        _psrc = "Upload: " + uploaded_file.name
    else:
        _psrc = _stem
    projects_ui.render_save(
        raw_df, {"label": _ds, "missing_choice": missing_choice, "overrides": _overrides,
                 "mode": st.session_state.get("mode"), "advanced": bool(st.session_state.get("show_advanced"))},
        _psrc, (web or {}).get("project_name") or _ds.replace("_", " "), bool(big))

    if big:
        st.subheader("Clean the full file")
        st.write(f"The report above describes the sample. This applies the same rules to all "
                 f"{big['summary']['rows']:,} rows and saves the result as a Parquet file.")
        fc_key = f"full_clean::{big['path']}::{big['stamp']}"
        part_opts = ["One file"] + [c for c, k in profile.items() if k == "categorical" and clean_df[c].nunique() <= 50]
        part_pick = st.selectbox("Save as", part_opts, format_func=lambda v: v if v == "One file" else f"A folder split by {v}",
                                 help="Splitting by a category (partitioned Parquet) writes one sub-folder per value, "
                                      "so later analysis of one agency or one year reads only that part.")
        partition_by = None if part_pick == "One file" else part_pick
        if st.button("Clean every row and save"):
            src = big["path"]
            folder = os.path.dirname(src) if local_path else tempfile.gettempdir()
            stem = os.path.splitext(os.path.basename(local_path or uploaded_file.name))[0]
            if not local_path:
                stem = security.TEMP_PREFIX + stem      # temporary copy on the server: removed by the sweep
            out_path = os.path.join(folder, f"{stem}_cleaned_by_{partition_by}" if partition_by
                                    else f"{stem}_cleaned.parquet")
            with st.spinner("Cleaning every row..."):
                try:
                    full = bigdata.clean_full_file(src, profile, list(dropped_identifiers),
                                                   big["summary"]["types"], out_path, partition_by)
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
            if full.get("partition_by"):
                st.caption(f"Partitioned Parquet: {full['n_parts']} sub-folders, one per value of "
                           f"'{full['partition_by']}'.")
            elif full["out_mb"] <= 200:
                with open(full["out_path"], "rb") as f:
                    st.download_button("Download cleaned file (Parquet)", f,
                                       file_name=os.path.basename(full["out_path"]))

    # --- Step 3: Choose analysis mode (sidebar), led by a plain-language goal picker ---
    import guide
    if st.session_state.get("mode") not in guide.MODES:
        st.session_state["mode"] = "Explore & describe"
    with st.sidebar:
        st.markdown("### Analysis")
        show_adv = st.toggle("Show advanced tools", key="show_advanced",
                             help="Clustering, data preparation, pipelines, combining sources and deep learning.")
        options = guide.everyday() + (guide.advanced() if show_adv else [])
        if st.session_state["mode"] not in options:
            options = options + [st.session_state["mode"]]
        mode = st.radio("What do you want to do with this data?", options, key="mode",
                        captions=[guide.MODES[m][0] for m in options], label_visibility="collapsed")
    section("3 · Analysis")
    guide.render_goal_picker(profile, clean_df)
    st.subheader(mode)
    guide.render_help(mode)

    # ===================== COMBINE & STANDARDIZE =====================
    if mode.startswith("Combine"):
        from standardize_ui import render_standardize
        render_standardize(clean_df, profile, st.session_state.get("dataset_label", "data"))

    # ===================== UPDATE 18 MODES =====================
    elif mode.startswith("Business problem"):
        from problem_ui import render_problem
        render_problem(raw_df, clean_df, profile, {
            "source": _record["source"], "rows_in": _record["input_rows"], "cols_in": _record["input_columns"],
            "missing": missing_choice, "overrides": _overrides, "rows_out": len(clean_df),
            "cols_out": clean_df.shape[1], "fingerprint": _record["input_fingerprint"], "report": report})

    elif mode.startswith("Data pipeline"):
        from pipeline_ui import render_pipeline
        render_pipeline(raw_df, clean_df, profile, {
            "source": _record["source"], "fingerprint": _record["input_fingerprint"], "loaded_at": _record["created"],
            "report": report, "missing": missing_choice, "overrides": _overrides, "label": _stem,
            "auto_profile": _auto_profile})

    elif mode.startswith("Group"):
        from analysis_ui import render_group
        render_group(clean_df, profile)

    elif mode.startswith("Analyze text"):
        from analysis_ui import render_text
        render_text(clean_df, profile)

    elif mode.startswith("Deep learning"):
        from deep_ui import render_deep
        render_deep(clean_df, profile, _stem)

    elif mode.startswith("Find items"):
        from analysis_ui import render_basket
        render_basket(raw_df)

    # ===================== EXPLORE & DESCRIBE =====================
    elif mode.startswith("Explore"):
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

            candidates = [c for c in numeric_or_cat_cols if c != target_col]
            default_feats = [c for c in candidates if clean_df[c].nunique() > 1]
            feature_cols = st.multiselect(
                "Columns the models may use to make the prediction", candidates, default=default_feats,
                key=f"feats_{target_col}",
                help="Remove any column that would not be known at the time you need the prediction, or that is "
                     "really the answer in another form (for example, money paid out when predicting the award "
                     "amount). Leaving such a column in makes the models look better than they are.")
            if not feature_cols:
                st.warning("Pick at least one column for the models to use.")
                st.stop()

            if st.button("Run Model Selector", type="primary"):
                with st.spinner("Splitting data, training models, and validating..."):
                    try:
                        output = run_pipeline(clean_df[feature_cols + [target_col]], target_col, profile)
                    except Exception as e:
                        st.error(f"Modeling failed: {e}")
                        st.stop()

                st.subheader("Results Dashboard")

                if output["dropped_cols"]:
                    st.caption(f"Note: columns excluded from modeling (text/datetime, not yet supported in Phase 1): "
                               f"{', '.join(output['dropped_cols'])}")

                results = output["results"]
                best = output["best_model"]
                st.session_state["model_done"] = {"target": target_col, "best": best,
                                                  "type": output["problem_type"], "models": len(results)}

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
                st.altair_chart(forecast_chart(fc["history"], fc["forecast"], label), width="stretch")
                st.caption("Shaded band = rough range, based on how far off the method was on the held-back periods. "
                           "It is a guide, not a guarantee.")

                st.write(f"**Forecast for the next {int(horizon)} {unit}s:**")
                out = fc["forecast"].copy()
                out["Period"] = out["Period"].dt.strftime("%Y-%m-%d")
                st.dataframe(out.round(2), hide_index=True)
                st.download_button("Download forecast as CSV", security.csv_bytes(out.round(2)),
                                   file_name=f"{st.session_state.get('dataset_label', 'data')}_forecast.csv", mime="text/csv")

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
                 "Useful for finding customer segments, similar sites or risk groups without labeling anything first.")

        use_cats = st.checkbox("Also use category columns (not just numbers)", key="clu_cats",
                               help="Off: group rows by their numeric columns only. On: categories such as region or store "
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
                 "the dataset. Useful for catching an unusual transaction, reading or record hidden among thousands "
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
                 "useful before acting on a finding, e.g. 'do stores really differ in average sales?'")
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
    import guide
    guide.render_start_steps()
    st.markdown("**What it can do**")
    cards = [
        ("Clean", "Fixes currency text, removes duplicates and identifier columns, fills gaps, flags outliers, "
                  "and reports every change it made."),
        ("Describe", "Frequency tables, summary statistics, cross-tabulations, correlation, and charts chosen "
                     "to fit each column's type."),
        ("Predict", "Compares up to ten models (including a neural network and SVM) on data they have not seen, with cross-validation, and "
                    "reports which one did best."),
        ("Forecast", "Projects a dated number forward, using whichever of seven methods best predicted your "
                     "most recent periods."),
        ("Find patterns", "Groups similar rows, flags unusual ones, and tests whether a difference is real "
                          "or just chance."),
        ("Combine sources", "Makes data from different sites, stores, labs or methods comparable, flags bad "
                            "readings and sudden shifts, and combines them into one index you can forecast."),
        ("Business problem & KPIs", "Write the problem statement and requirements, measure KPIs such as churn "
                                    "rate and its trend, and export a Word brief for the client."),
        ("Deep learning", "Train Keras neural networks on tables, CNNs on images and deep Q-learning agents, "
                          "with a fairness check and a FastAPI + Docker export."),
        ("Data pipeline", "Quality gates, bronze/silver/gold tables, private-data masking, joins, SQL, and a "
                          "saved pipeline you can re-run on a schedule."),
        ("Analyze text", "Scores comments and reviews as positive, neutral or negative and finds the words "
                         "and phrases people use most."),
        ("Bought together", "Finds which items sell together, with support, confidence and lift, to suggest "
                            "cross-sells and bundles."),
    ]
    for row in (cards[:3], cards[3:6], cards[6:9], cards[9:]):
        cols = st.columns(3)
        for col, (title, text) in zip(cols, row):
            col.markdown(f'<div class="vg-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
        st.write("")
    st.caption("Accepts CSV, TSV, TXT, Excel, JSON and Parquet from your computer or a web address, including files "
               "with millions of rows. Runs on one computer (DuckDB), not a Spark or Hadoop cluster; "
               "audio and video are not supported. "
               "Works with data from any field: sales, health, operations, finance, public records. Sample files are synthetic.")
