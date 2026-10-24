"""
Define a business problem, step by step, and document it:
  1. Problem identification   2. Requirement gathering   3. Data pipeline and workflow
  4. Measurable metrics (KPIs) and their trends            5. Documentation and presentation (Word brief)
"""
import io
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import charts
import importance as imp
import kpi
from explore_ui import _download

BLUE, RED = "#2a78d6", "#e34948"
USE_CASES = [
    "Demand, revenue or sales forecasts",
    "Customer conversion, churn and retention",
    "Fraud or pricing",
    "Customer, vendor or distributor scoring",
    "Cross-sell / up-sell recommendations",
    "Machine failures and maintenance",
    "Customer sentiment in text",
    "Other",
]
WHERE = {
    "Demand, revenue or sales forecasts": "Forecast future values",
    "Customer conversion, churn and retention": "Predict a target variable (the churn column), plus the KPIs on this screen",
    "Fraud or pricing": "Detect anomalies (fraud screen) or Predict a target variable (price)",
    "Customer, vendor or distributor scoring": "Find hidden groups, or Predict a target variable",
    "Cross-sell / up-sell recommendations": "Find items bought together",
    "Machine failures and maintenance": "Predict a target variable, or Combine & standardize sources",
    "Customer sentiment in text": "Analyze text",
    "Other": "Explore & describe first, then pick a mode",
}


def _line(t: pd.DataFrame, x: str, y: str, title: str, color: str = BLUE):
    d = pd.DataFrame({"x": pd.to_datetime(t[x]), "y": t[y]})
    return alt.Chart(d).mark_line(color=color, point=True).encode(
        x=alt.X("x:T", title=None), y=alt.Y("y:Q", title=y),
        tooltip=[alt.Tooltip("x:T", title="Period"), alt.Tooltip("y:Q", title=y, format=",.2f")]
    ).properties(height=260, title=title)


def _trend_text(name: str, s: dict, unit: str) -> str:
    if s["direction"] == "too few periods":
        return f"{name}: too few {unit}s to judge a trend."
    ch = "" if np.isnan(s["change_pct"]) else f" The last few {unit}s average {s['change_pct']:+.0f}% versus the first few."
    if s["direction"] == "no clear trend":
        return f"{name}: no clear trend (straight-line test p = {s['p']:.2f}).{ch}"
    return f"{name} is **{s['direction']}** by about {abs(s['slope']):,.2f} per {unit} (p = {s['p']:.3f}).{ch}"


def render_problem(raw_df: pd.DataFrame, df: pd.DataFrame, profile: dict, pipeline: dict):
    st.write("Turn a business question into numbers, in the five steps data teams use: **identify** the problem, "
             "**gather requirements**, set up the **data pipeline**, measure it with **KPIs** and their trends, and "
             "**document and present** it. Everything you enter goes into a Word brief in step 5.")
    S = st.session_state.setdefault("brief", {})
    _lifecycle(S, pipeline)
    num = [c for c, t in profile.items() if t == "numeric"]
    cat = [c for c, t in profile.items() if t == "categorical"]
    dates = [c for c, t in profile.items() if t == "datetime"]
    t1, t2, t3, t4, t45, t5 = st.tabs(["1 · Problem", "2 · Requirements", "3 · Data pipeline", "4 · KPIs and trends",
                                       "5 · Target & feature importance", "6 · Document"])

    # ---------------- 1. problem identification ----------------
    with t1:
        S["use_case"] = st.selectbox("Type of business problem", USE_CASES, key="bp_case")
        S["business_problem"] = st.text_area(
            "Business problem (the challenge in plain words)", key="bp_problem",
            placeholder="Example: we win new customers easily but too many stop buying within a year.")
        S["problem_statement"] = st.text_input(
            "Problem statement (one measurable question)", key="bp_statement",
            placeholder="How do we improve average customer revenue through customer retention analytics?")
        st.caption(f"Where to analyze it in this app: **{WHERE[S['use_case']]}**.")

    # ---------------- 2. requirement gathering ----------------
    with t2:
        c1, c2 = st.columns(2)
        S["client"] = c1.text_input("Client or stakeholder", key="bp_client")
        S["sme"] = c2.text_input("Subject matter expert (SME)", key="bp_sme")
        S["requirements"] = st.text_area("Requirements agreed with the client (one per line)", key="bp_reqs",
                                         placeholder="Monthly churn rate by region\nList of customers at risk\nResults by March 1")
        S["success"] = st.text_input("How success will be measured", key="bp_success",
                                     placeholder="Churn rate down from 12% to 9% within two quarters")
        st.markdown("**Target and inputs**")
        targets = num + cat
        if targets:
            c3, c4 = st.columns(2)
            S["dv"] = c3.selectbox("Target, the dependent variable (DV): what you want to explain or predict",
                                   targets, key="bp_dv", index=_guess_target(targets, df, profile))
            ivs_all = [c for c in targets if c != S["dv"]]
            S["ivs"] = c4.multiselect("Independent variables (IVs): the columns that may explain it", ivs_all,
                                      default=ivs_all[:8], key="bp_ivs")
            st.caption("Leave out any column that is the answer in another form, or that is only known after the "
                       "outcome (for example, a cancellation date when the target is churn).")

    # ---------------- 3. data pipeline ----------------
    with t3:
        st.markdown("**Where the data came from and what was done to it** (filled in by the app)")
        p = pipeline
        st.dataframe(pd.DataFrame({"Step": ["Source", "Rows and columns loaded", "Missing values", "Column type changes",
                                            "Rows and columns after cleaning", "Input fingerprint"],
                                   "Detail": [p["source"], f"{p['rows_in']:,} × {p['cols_in']}", p["missing"],
                                              ", ".join(f"{k} → {v}" for k, v in p["overrides"].items()) or "none",
                                              f"{p['rows_out']:,} × {p['cols_out']}", p["fingerprint"][:16] + "…"]}),
                     hide_index=True, width="stretch")
        with st.expander("Every cleaning step"):
            for line in p["report"]:
                st.write(f"• {line}")
        S["pipeline_notes"] = st.text_area("Other data sources or steps (optional)", key="bp_pipe",
                                           placeholder="Monthly export from the CRM; joined to the billing system by customer ID")

    # ---------------- 4. KPIs ----------------
    with t4:
        S["kpis"], S["charts"], S["drivers"] = {}, [], None
        if not dates:
            st.info("KPI trends need a date column. Drivers of the target are still shown below.")
        else:
            period = st.radio("Measure trends by", list(kpi.PERIOD_CODES), index=1, horizontal=True, key="bp_period")
            unit = period.lower()

            # customer KPIs from transaction-style data
            ids = [c for c in raw_df.columns if any(k in c.lower() for k in ("customer", "client", "account", "member", "user"))]
            with st.expander("Customer KPIs: length of relationship, transaction frequency, churn rate",
                             expanded=bool(ids)):
                if not ids:
                    st.caption("Needs a customer (or account) column with one row per transaction. "
                               "Try the sample **Store orders**.")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    cust_col = c1.selectbox("Customer column", ids, key="bp_cust")
                    rdates = [c for c in raw_df.columns if c in dates]
                    dcol = c2.selectbox("Transaction date", rdates or dates, key="bp_cdate")
                    amt_opts = ["(none)"] + [c for c in num if c in raw_df.columns]
                    amt = c3.selectbox("Amount (optional)", amt_opts, key="bp_amt",
                                       index=next((i for i, c in enumerate(amt_opts) if any(k in c.lower() for k in
                                                   ("revenue", "amount", "sales", "total", "price"))), 0))
                    days = c4.number_input("Churned after this many days without a purchase", 7, 730, 90, key="bp_days")
                    src = raw_df.copy()
                    src[dcol] = pd.to_datetime(src[dcol], errors="coerce")
                    if amt != "(none)":
                        src[amt] = pd.to_numeric(src[amt], errors="coerce")
                    cust = kpi.customer_table(src, cust_col, dcol, None if amt == "(none)" else amt, int(days))
                    if cust["transactions"].max() <= 1:
                        st.warning(f"Every value of '{cust_col}' appears on one date only, so there is no repeat "
                                   "behavior to measure. Pick the column that identifies the customer.")
                    else:
                        k = kpi.customer_kpis(cust)
                        shown = {a: b for a, b in k.items() if not a.startswith("_")}
                        cols = st.columns(min(4, len(shown)))
                        for i, (a, b) in enumerate(shown.items()):
                            cols[i % len(cols)].metric(a, f"{b:,}")
                        S["kpis"].update({f"{a}": b for a, b in shown.items()})
                        tr = kpi.churn_trend(src, cust_col, dcol, int(days), period)
                        tr_use = tr
                        s = kpi.trend_summary(tr_use, "Churn rate (%)")
                        if tr.empty:
                            st.caption(f"Not enough history yet: every {unit} is within {int(days)} days of the end of the data.")
                        else:
                            st.altair_chart(_line(tr, "Period", "Churn rate (%)", f"Churn rate by {unit}", RED), width="stretch")
                        msg = _trend_text("Churn rate", s, unit)
                        st.markdown(msg)
                        st.caption(f"Each {unit}'s churn rate: of the customers who bought that {unit}, the share who "
                                   f"did not buy again within {int(days)} days. Recent {unit}s are left off until "
                                   f"{int(days)} days have passed, because it is too soon to know who comes back. "
                                   f"The overall churn rate above counts customers with no purchase in the last {int(days)} days.")
                        S["charts"].append(("Churn rate by " + unit, tr, "Period", "Churn rate (%)"))
                        S["kpis"]["Churn rate trend"] = msg.replace("**", "")
                        if "_lost_avg" in k and pd.notna(k["_lost_avg"]):
                            st.caption(f"Churned customers spent {k['_lost_avg']:,.2f} on average; customers still "
                                       f"active spent {k['_active_avg']:,.2f}.")
                        _download(cust, "customer_kpis.csv", "Download one row per customer (with churned Yes/No)")
                        st.caption("Tip: download this table and load it as a new file to predict which customers "
                                   "churn, with 'churned' as the target.")

            # rate of a Yes/No column over time (e.g. churned)
            two = [c for c in cat if df[c].nunique() == 2]
            if two:
                with st.expander("Rate of a Yes/No outcome over time (for example churned)", expanded=not ids):
                    c1, c2, c3 = st.columns(3)
                    flag = c1.selectbox("Outcome column", two, key="bp_flag",
                                        index=next((i for i, c in enumerate(two) if "churn" in c.lower()), 0))
                    vals = sorted(df[flag].astype(str).unique())
                    pos = c2.selectbox("Count rows where it equals", vals, index=vals.index("Yes") if "Yes" in vals else 0,
                                       key="bp_pos")
                    fdate = c3.selectbox("Date", dates, key="bp_fdate")
                    rt = kpi.rate_trend(df, flag, pos, fdate, period)
                    s = kpi.trend_summary(rt, "Rate (%)")
                    st.metric(f"{flag} = {pos}, all periods", f"{(df[flag].astype(str) == pos).mean():.1%}")
                    st.altair_chart(_line(rt, "Period", "Rate (%)", f"Share of rows with {flag} = {pos}, by {unit}", RED),
                                    width="stretch")
                    msg = _trend_text(f"The {flag} rate", s, unit)
                    st.markdown(msg)
                    S["kpis"][f"{flag} = {pos} rate (%)"] = round((df[flag].astype(str) == pos).mean() * 100, 1)
                    S["kpis"][f"{flag} trend"] = msg.replace("**", "")
                    S["charts"].append((f"{flag} = {pos} rate by {unit}", rt, "Period", "Rate (%)"))

            # any number over time
            with st.expander("Any KPI over time (revenue, orders, average order size...)", expanded=not ids and not two):
                c1, c2, c3 = st.columns(3)
                how_lbl = c1.selectbox("Calculation", ["Total", "Average", "Count of rows"], key="bp_how")
                how = {"Total": "sum", "Average": "mean", "Count of rows": "count"}[how_lbl]
                val = None if how == "count" else c2.selectbox(
                    "Number", num, key="bp_val",
                    index=next((i for i, c in enumerate(num) if any(k in c.lower() for k in
                                ("revenue", "sales", "amount", "spend", "units", "visits", "obligat"))), 0)) if num else None
                kdate = c3.selectbox("Date", dates, key="bp_kdate")
                if how != "count" and val is None:
                    st.caption("No number columns.")
                else:
                    t = kpi.kpi_trend(df, kdate, val, how, period)
                    name = f"{how_lbl} {val or 'rows'} per {unit}"
                    s = kpi.trend_summary(t, "Value")
                    st.altair_chart(_line(t.rename(columns={"Value": name}), "Period", name, name), width="stretch")
                    msg = _trend_text(name, s, unit)
                    st.markdown(msg)
                    if st.checkbox("Add this KPI to the brief", key="bp_addkpi", value=True):
                        S["kpis"][name + " (latest)"] = round(float(t["Value"].iloc[-1]), 2)
                        S["kpis"][name + " trend"] = msg.replace("**", "")
                        S["charts"].append((name, t.rename(columns={"Value": name}), "Period", name))

        st.markdown("**What drives the target?** (the IVs most linked to the DV)")
        dv, ivs = S.get("dv"), S.get("ivs") or []
        if not dv or not ivs:
            st.caption("Pick a target and its inputs in step 2.")
        else:
            pos = None
            if profile.get(dv) == "categorical" and df[dv].nunique() == 2:
                vals = sorted(df[dv].astype(str).unique())
                pos = "Yes" if "Yes" in vals else vals[-1]
            drv = kpi.drivers(df, dv, ivs, profile, pos)
            if drv.empty:
                st.caption("No usable inputs to compare. A two-outcome or number target is needed.")
            else:
                st.dataframe(drv, hide_index=True, width="stretch")
                st.caption("Strength is the effect size for each test (Cohen's d, Cramér's V, |r| or eta); larger means "
                           "a stronger link. A link is not proof of cause. Use Predict a target variable to test how "
                           "well these columns predict the target together.")
                _download(drv, "drivers_of_target.csv")
                S["drivers"] = drv

    # ---------------- 5. target & feature importance ----------------
    with t45:
        _render_importance(df, profile, S)

    # ---------------- 6. documentation ----------------
    with t5:
        S["findings"] = st.text_area("Key findings and recommendations (one per line)", key="bp_findings",
                                     placeholder="Churn has risen three months in a row\nEastside store customers churn twice as often\n"
                                                 "Start a retention offer for customers 60 days without a purchase")
        S["author"] = st.text_input("Prepared by", value="Vanguard Data Analytics LLC", key="bp_author")
        missing = [n for n, v in (("problem statement", S.get("problem_statement")),
                                  ("requirements", S.get("requirements"))) if not (v or "").strip()]
        if missing:
            st.caption("Not filled in yet: " + ", ".join(missing) + ". The brief will mark them as to be agreed.")
        if st.button("Build business problem brief (Word)", type="primary", key="bp_build"):
            with st.spinner("Writing the brief..."):
                st.session_state["brief_doc"] = build_brief(S, pipeline, st.session_state.get("dataset_label", "data"))
        if st.session_state.get("brief_doc"):
            st.download_button("Download brief (.docx)", st.session_state["brief_doc"],
                               file_name=f"{st.session_state.get('dataset_label', 'data')}_business_problem_brief.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               key="bp_dl")


LIFECYCLE = [
    ("1. Understanding and defining the business problem", "Problem statement, SMEs and requirements (tabs 1–2 here)"),
    ("2. Data access and discovery", "Load files, web addresses or APIs; join sources into one analysis table (Data pipeline mode)"),
    ("3. Data engineering and pre-processing", "Cleaning, outliers, missing values, encoding, scaling, Gaussian mapping, "
                                               "binning (automatic cleaning + Prepare & transform)"),
    ("4. Modeling development and evaluation", "Train/test split, compare and tune models, measure accuracy or error "
                                               "(Predict a target variable)"),
]


def lifecycle_status(S: dict, pipeline: dict) -> list:
    md = st.session_state.get("model_done")
    return [
        "Done" if (S.get("problem_statement") or "").strip() and (S.get("requirements") or "").strip()
        else "In progress" if any((S.get(k) or "").strip() for k in ("problem_statement", "business_problem", "requirements"))
        else "Not started",
        "Done" if pipeline.get("rows_in") else "Not started",
        "Done" if pipeline.get("report") is not None else "Not started",
        f"Done: best of {md['models']} models for '{md['target']}' was {md['best']}" if md else "Not started",
    ]


def _lifecycle(S: dict, pipeline: dict):
    status = lifecycle_status(S, pipeline)
    icon = {"Done": "✅", "In progress": "🟡", "Not started": "⬜"}
    with st.expander("Data science project life cycle: where this project stands", expanded=False):
        st.dataframe(pd.DataFrame({"Phase": [p for p, _ in LIFECYCLE],
                                   "Status": [f"{icon[s.split(':')[0]]} {s}" for s in status],
                                   "Where in the app": [w for _, w in LIFECYCLE]}), hide_index=True, width="stretch")
        st.caption("Typical length for one model from problem to deployment: about three months, up to a year. "
                   "Real projects loop back (for example, modeling shows more data is needed).")


def _guess_target(targets: list, df: pd.DataFrame, profile: dict) -> int:
    exact = ("y", "target", "label", "outcome", "churn", "churned", "subscribed", "converted", "response", "class")
    for i, c in enumerate(targets):
        if c.lower() in exact:
            return i
    for words in (("churn", "cancel", "subscri", "convert", "default", "fraud", "fail"),
                  ("revenue", "sales", "amount", "spend")):
        for i, c in enumerate(targets):
            if any(k in c.lower() for k in words):
                return i
    return 0


# ======================================================================= target & feature importance
def _render_importance(df: pd.DataFrame, profile: dict, S: dict):
    dv, ivs = S.get("dv"), [c for c in (S.get("ivs") or []) if c in df.columns]
    S["balance"], S["importance"], S["boruta"] = None, None, None
    if not dv or not ivs:
        st.info("Pick the target (DV) and its inputs (IVs) in step 2.")
        return
    classify = imp.is_classification(df[dv], profile.get(dv))
    if classify and df[dv].nunique() > 20:
        st.warning(f"'{dv}' has {df[dv].nunique()} different values; pick a target with a few outcomes or a number.")
        return
    vals = sorted(df[dv].astype(str).unique())
    positive = ("Yes" if "Yes" in vals else vals[-1]) if classify and len(vals) == 2 else None

    # --- target distribution / imbalance
    st.markdown(f"**The target: {dv}**")
    if classify:
        b = imp.target_balance(df[dv])
        S["balance"] = b
        c1, c2 = st.columns([1, 2])
        c1.dataframe(b["table"], hide_index=True, width="stretch")
        c2.altair_chart(charts.bar_chart(b["table"], "Value", "Rows", f"Rows per value of {dv}"), width="stretch")
        if b["level"] == "balanced":
            st.caption("The outcomes are reasonably balanced.")
        else:
            st.warning(f"**Imbalanced target ({b['level']})**: '{b['minority']}' is only "
                       f"{b['table']['Share (%)'].iloc[-1]:.1f}% of rows (about 1 in {b['ratio'] + 1:.0f}). Accuracy can look "
                       "high while missing most of that group, so judge models by recall and F1 for it, and the "
                       "importance below uses balanced class weights.")
    else:
        st.altair_chart(charts.histogram(df, dv), width="stretch")

    # --- each IV against the target
    st.markdown("**Each input against the target**")
    num_ivs = [c for c in ivs if profile.get(c) == "numeric"]
    cat_ivs = [c for c in ivs if profile.get(c) == "categorical" and df[c].nunique() <= 30]
    c1, c2 = st.columns(2)
    if classify and num_ivs:
        with c1:
            x = st.selectbox("Number column", num_ivs, key="fi_num")
            d = pd.DataFrame({x: df[x], dv: df[dv].astype(str)}).dropna()
            ch = alt.Chart(d).transform_bin("bin", x, bin=alt.Bin(maxbins=30)).transform_aggregate(
                n="count()", groupby=["bin", dv]).transform_joinaggregate(total="sum(n)", groupby=[dv]
            ).transform_calculate(share="datum.n / datum.total").mark_bar(opacity=0.6).encode(
                x=alt.X("bin:Q", title=x), y=alt.Y("share:Q", title="Share of rows in each outcome", stack=None),
                color=alt.Color(f"{dv}:N", title=dv)).properties(height=260, title=f"{x} for each value of {dv}")
            st.altair_chart(ch, width="stretch")
            st.caption("Each outcome is scaled to 100%, so a small group can be compared with a large one.")
    if cat_ivs:
        with (c2 if classify and num_ivs else c1):
            g = st.selectbox("Category column", cat_ivs, key="fi_cat")
            if classify:
                ct = pd.crosstab(df[g], df[dv].astype(str))
                long = ct.reset_index().melt(id_vars=g, var_name=dv, value_name="Rows")
                long[g] = long[g].astype(str)
                order = ct.sum(axis=1).sort_values(ascending=False).index.astype(str).tolist()
                st.altair_chart(alt.Chart(long).mark_bar().encode(
                    x=alt.X(f"{g}:N", sort=order, title=g), y=alt.Y("Rows:Q"), color=alt.Color(f"{dv}:N"),
                    tooltip=[g, dv, "Rows"]).properties(height=260, title=f"{dv} by {g}"), width="stretch")
                if positive:
                    rate = (ct[positive] / ct.sum(axis=1) * 100).round(1).sort_values(ascending=False)
                    st.caption(f"Share with {dv} = {positive}: " + ", ".join(f"{k} {v}%" for k, v in rate.head(6).items()))
            else:
                st.altair_chart(charts.box_plot(df, dv, g), width="stretch")

    # --- correlation with the target
    corr = imp.target_correlation(df, dv, ivs, profile, positive)
    if corr.shape[0] >= 2:
        st.markdown("**Correlation matrix, including the target**" + (f" ({dv} = {positive} as 1, otherwise 0)" if positive else ""))
        corr.index.name = "variable"
        st.altair_chart(charts.heatmap(corr, "Correlation (−1 to 1)"), width="stretch")
        pairs = [(a, b2, corr.loc[a, b2]) for i, a in enumerate(corr.index) for b2 in corr.columns[i + 1:]
                 if dv not in a and dv not in b2 and abs(corr.loc[a, b2]) >= 0.8]
        if pairs:
            st.warning("Strongly related inputs (multicollinearity), which can confuse some models: "
                       + "; ".join(f"{a} and {b2} (r = {r:.2f})" for a, b2, r in pairs[:5]))

    # --- importance
    st.markdown("**Feature importance: which inputs explain the target**")
    X, y, skipped = imp.encode(df, dv, ivs, profile)
    if X.shape[1] < 1:
        st.caption("No usable inputs.")
        return
    if skipped:
        st.caption("Left out (free text or too many categories): " + ", ".join(skipped))
    run_boruta = st.checkbox("Also run Boruta (slower, gives a Confirmed / Tentative / Rejected decision for each column)",
                             value=True, key="fi_boruta")
    if st.button("Calculate feature importance", type="primary", key="fi_go"):
        with st.spinner("Training random forests..."):
            table, _ = imp.model_importance(X, y.astype(str) if classify else pd.to_numeric(y, errors="coerce"), classify)
            st.session_state["fi_table"] = (dv, tuple(ivs), table)
            if run_boruta:
                bar = st.progress(0.0, text="Boruta: comparing each column with shuffled copies...")
                bt = imp.boruta(X, y.astype(str) if classify else pd.to_numeric(y, errors="coerce"), classify,
                                progress=lambda f: bar.progress(f, text=f"Boruta round {int(round(f * 50))} of up to 50"))
                bar.empty()
                st.session_state["fi_boruta_t"] = (dv, tuple(ivs), bt)
    ft = st.session_state.get("fi_table")
    if ft and ft[0] == dv and ft[1] == tuple(ivs):
        table = ft[2]
        S["importance"] = table
        st.dataframe(table, hide_index=True, width="stretch")
        st.altair_chart(_bars_h(table, "Column", "Random forest (Gini) %", "Random forest importance (%)"), width="stretch")
        st.caption("Gini importance: how much each column helps the random forest split the data (adds to 100%). "
                   "Permutation: how much the score (ROC AUC for a yes/no target, R² for a number) drops on held-back rows when that column is shuffled; near 0 or "
                   "negative means the column adds nothing. Importance is not cause.")
        _download_df(table, "feature_importance.csv")
    bt = st.session_state.get("fi_boruta_t")
    if bt and bt[0] == dv and bt[1] == tuple(ivs):
        S["boruta"] = bt[2]
        st.markdown(f"**Boruta** ({bt[2].attrs.get('iterations', '?')} rounds)")
        st.dataframe(bt[2], hide_index=True, width="stretch")
        st.altair_chart(_bars_h(bt[2], "Column", "Rank", "Boruta rank (1 = confirmed important)"), width="stretch")
        st.caption("Each round adds a shuffled copy of every column. A real column that beats the best shuffled copy "
                   "far more often than chance is Confirmed; far less often, Rejected. Keep Confirmed columns for "
                   "models, review Tentative ones.")


def _bars_h(t: pd.DataFrame, label: str, value: str, title: str):
    d = pd.DataFrame({"Label": t[label].astype(str), "Value": t[value]})
    return alt.Chart(d).mark_bar(color=BLUE).encode(
        y=alt.Y("Label:N", sort=None, title=None), x=alt.X("Value:Q", title=value),
        tooltip=["Label", "Value"]).properties(height=max(150, 24 * len(d)), title=title)


def _download_df(t: pd.DataFrame, name: str):
    _download(t, name)


# ======================================================================= Word brief
def build_brief(S: dict, pipeline: dict, label: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(10)
    h = doc.add_heading("Business Problem Brief", level=0)
    h.runs[0].font.color.rgb = RGBColor(0x14, 0x26, 0x3D)
    doc.add_paragraph(f"{label.replace('_', ' ')}\nPrepared by {S.get('author') or 'Vanguard Data Analytics LLC'}, "
                      f"{date.today():%B %d, %Y}")

    def para(text, fallback="To be agreed with the client."):
        doc.add_paragraph((text or "").strip() or fallback)

    def bullets(text, fallback="To be agreed with the client."):
        lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
        if not lines:
            doc.add_paragraph(fallback)
        for l in lines:
            doc.add_paragraph(l, style="List Bullet")

    def table(header, rows):
        t = doc.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        for i, hd in enumerate(header):
            t.rows[0].cells[i].text = str(hd)
        for r in rows:
            cells = t.add_row().cells
            for i, v in enumerate(r):
                cells[i].text = str(v)
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(9)
        doc.add_paragraph()

    doc.add_heading("Project life cycle status", level=1)
    try:
        st_list = lifecycle_status(S, pipeline)
    except Exception:
        st_list = ["–"] * 4
    table(["Phase", "Status"], [[ph, s] for (ph, _), s in zip(LIFECYCLE, st_list)])

    doc.add_heading("1. Problem identification", level=1)
    doc.add_paragraph(f"Type of problem: {S.get('use_case', '')}")
    doc.add_heading("Business problem", level=2)
    para(S.get("business_problem"))
    doc.add_heading("Problem statement", level=2)
    para(S.get("problem_statement"))

    doc.add_heading("2. Requirements", level=1)
    table(["Item", "Detail"], [["Client or stakeholder", S.get("client") or "–"],
                               ["Subject matter expert", S.get("sme") or "–"],
                               ["Success measure", S.get("success") or "To be agreed"],
                               ["Target (dependent variable)", S.get("dv") or "–"],
                               ["Inputs (independent variables)", ", ".join(S.get("ivs") or []) or "–"]])
    doc.add_paragraph("Requirements agreed:")
    bullets(S.get("requirements"))

    doc.add_heading("3. Data pipeline and workflow", level=1)
    p = pipeline
    table(["Step", "Detail"], [["Source", p["source"]], ["Loaded", f"{p['rows_in']:,} rows × {p['cols_in']} columns"],
                               ["Missing values", p["missing"]],
                               ["Column type changes", ", ".join(f"{k} → {v}" for k, v in p["overrides"].items()) or "none"],
                               ["After cleaning", f"{p['rows_out']:,} rows × {p['cols_out']} columns"],
                               ["Input fingerprint (SHA-256)", p["fingerprint"]]])
    if (S.get("pipeline_notes") or "").strip():
        para(S["pipeline_notes"])
    doc.add_paragraph("Cleaning steps:")
    for line in p["report"][:40]:
        doc.add_paragraph(line, style="List Bullet")

    doc.add_heading("4. Measurable metrics (KPIs) and trends", level=1)
    k = S.get("kpis") or {}
    if k:
        table(["KPI", "Value"], [[a, f"{b:,}" if isinstance(b, (int, float)) else b] for a, b in k.items()])
    else:
        doc.add_paragraph("No KPIs were calculated (open step 4 in the app first).")
    for title, t, x, y in S.get("charts") or []:
        fig, ax = plt.subplots(figsize=(6.2, 2.4))
        ax.plot(pd.to_datetime(t[x]), t[y], color=BLUE, marker="o", markersize=3)
        ax.set_title(title, fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.autofmt_xdate()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        doc.add_picture(buf, width=Inches(6.2))
    drv = S.get("drivers")
    if drv is not None and not drv.empty:
        doc.add_heading(f"What is linked to {S.get('dv')}", level=2)
        table(list(drv.columns), drv.head(10).values.tolist())
        doc.add_paragraph("A link is not proof of cause; effect sizes show strength, p-values show whether the link "
                          "is likely to be more than chance.")

    doc.add_heading("5. Target and feature importance", level=1)
    b = S.get("balance")
    if b:
        table(["Value of " + str(S.get("dv")), "Rows", "Share (%)"], b["table"].values.tolist())
        if b["level"] != "balanced":
            doc.add_paragraph(f"The target is imbalanced ({b['level']}): '{b['minority']}' is the smaller group, so "
                              "models are judged by recall and F1 for it, not accuracy alone.")
    fi = S.get("importance")
    if fi is not None and not fi.empty:
        doc.add_paragraph("Feature importance (random forest Gini, decision tree, permutation on held-back rows):")
        table(list(fi.columns), fi.head(12).values.tolist())
    bo = S.get("boruta")
    if bo is not None and not bo.empty:
        doc.add_paragraph("Boruta decision for each input:")
        table(list(bo.columns), bo.values.tolist())
    if fi is None and bo is None and not b:
        doc.add_paragraph("Not calculated (open step 5 in the app and click Calculate feature importance).")

    doc.add_heading("6. Findings and recommendations", level=1)
    bullets(S.get("findings"), "To be completed after review with the client.")
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()
