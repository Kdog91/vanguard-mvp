"""Streamlit screen: Data pipeline (ingest → validate → bronze → silver → gold → orchestrate → govern)."""
import json
import time

import pandas as pd
import streamlit as st

import grouping
import parquet_tools as pt
import pipeline as pl
from data_cleaner import load_file

STATUS_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}


def render_pipeline(raw_df: pd.DataFrame, clean_df: pd.DataFrame, profile: dict, ctx: dict):
    st.write("Run this file through the seven stages of a data pipeline. The result is a **bronze** table (raw, "
             "untouched), a **silver** table (cleaned, joined, private details masked) and a **gold** table (summed up "
             "for reports), saved together as Parquet files and a DuckDB warehouse you can query with SQL. The same "
             "steps can be saved and re-run on a schedule.")
    t = st.tabs(["1 · Ingest", "2 · Validate", "3 · Bronze", "4 · Silver", "5 · Gold", "6 · Orchestrate",
                 "7 · Govern & monitor", "Parquet & partitions"])
    P = st.session_state.setdefault("pipe", {})

    # ---------------- 1. ingest ----------------
    with t[0]:
        st.markdown(f"**Source:** {ctx['source']}  \n**Loaded:** {len(raw_df):,} rows × {raw_df.shape[1]} columns  \n"
                    f"**Mode:** batch (a file or web address each run)")
        st.caption("Sources this app can ingest: file upload (CSV, Excel, JSON, Parquet), a web address, the "
                   "USAspending.gov API, and files with millions of rows from your own computer. Real-time streams "
                   "and change-data-capture from a live database are not supported.")

    # ---------------- 2. validate ----------------
    with t[1]:
        st.markdown("**Quality gates** (run on the raw data before anything is changed)")
        schema_file = st.file_uploader("Schema contract from an earlier run (optional, .json)", type=["json"], key="pl_schema",
                                       help="Download one in this tab after a good run; next time, any missing column or "
                                            "changed type is caught here.")
        schema = None
        if schema_file is not None:
            try:
                schema = json.loads(schema_file.read())
            except Exception as e:
                st.error(f"Could not read the schema file: {e}")
        c1, c2 = st.columns(2)
        warn_pct = c1.slider("Warn when a column is this % empty", 5, 80, 20, key="pl_warn") / 100
        fail_pct = c2.slider("Fail when a column is this % empty", 30, 100, 80, key="pl_fail") / 100
        checks = pl.validate(raw_df, profile, schema, warn_pct, max(fail_pct, warn_pct + 0.01))
        status = pl.gate(checks)
        P["checks"], P["gate"] = checks, status
        (st.error if status == "FAIL" else st.warning if status == "WARN" else st.success)(
            f"{STATUS_ICON[status]} Quality gate: **{status}** "
            f"({(checks['Result'] == 'FAIL').sum()} failed, {(checks['Result'] == 'WARN').sum()} warnings, "
            f"{(checks['Result'] == 'PASS').sum()} passed)")
        st.dataframe(checks.assign(Result=checks["Result"].map(lambda r: f"{STATUS_ICON[r]} {r}")),
                     hide_index=True, width="stretch")
        st.download_button("Download schema contract (.json)",
                           json.dumps(pl.schema_of(raw_df, {**ctx["auto_profile"], **profile}), indent=2),
                           file_name=f"{ctx['label']}_schema.json", mime="application/json", key="pl_schema_dl")
        if status == "FAIL":
            st.caption("A scheduled run stops here when a check fails, so bad data never reaches the silver and "
                       "gold tables. On screen you can still look at the later stages.")

    # ---------------- 3. bronze ----------------
    with t[2]:
        st.markdown("**Bronze: the raw data exactly as received.** It is never edited, so any later step can be "
                    "re-run from it.")
        P["bronze_fp"] = ctx["fingerprint"]
        st.dataframe(pd.DataFrame({"Item": ["Rows", "Columns", "Fingerprint (SHA-256)", "Loaded at"],
                                   "Value": [f"{len(raw_df):,}", raw_df.shape[1], ctx["fingerprint"], ctx["loaded_at"]]}),
                     hide_index=True, width="stretch")
        st.dataframe(raw_df.head(10), width="stretch")
        st.caption("If the same file is loaded again, the fingerprint is the same; any change to the data changes it.")

    # ---------------- 4. silver ----------------
    with t[3]:
        st.markdown("**Silver: cleaned, joined and with private details masked.**")
        st.caption(f"Cleaning already applied: {len(ctx['report'])} steps (see the cleaning report above), "
                   f"missing values: {ctx['missing']}.")
        silver = clean_df.copy()

        st.markdown("**Join another source** (build one analysis table from several files)")
        other = st.file_uploader("Second file to join (optional)", key="pl_join_file",
                                 type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet", "orc"])
        if other is not None:
            try:
                right = load_file(other)
                c1, c2, c3 = st.columns(3)
                lkey = c1.selectbox("Key in this file", list(raw_df.columns), key="pl_lkey",
                                    index=_guess_key(list(raw_df.columns), list(right.columns)))
                rkey = c2.selectbox(f"Key in {other.name}", list(right.columns), key="pl_rkey",
                                    index=list(right.columns).index(lkey) if lkey in right.columns else 0)
                how = c3.selectbox("Keep", ["left", "inner", "outer"], key="pl_how",
                                   format_func=lambda h: {"left": "all rows of this file (left join)",
                                                          "inner": "only matching rows (inner join)",
                                                          "outer": "all rows of both (outer join)"}[h])
                base = silver if lkey in silver.columns else silver.assign(**{lkey: raw_df.loc[silver.index, lkey]})
                silver, info = pl.join(base, right, [lkey], [rkey], how)
                st.success(f"Joined: {info['matched']:,} rows matched, {info['left_only']:,} rows of this file had no "
                           f"match" + (f", {info['right_only']:,} rows only in {other.name}" if how == "outer" else "")
                           + f". Result: {info['result_rows']:,} rows × {silver.shape[1]} columns.")
                if info["duplicate_keys_in_second_file"]:
                    st.warning(f"{info['duplicate_keys_in_second_file']:,} keys repeat in {other.name}, so some rows were "
                               "copied once per match. Check the key is unique there.")
                P["join"] = {"file": other.name, "left_on": lkey, "right_on": rkey, "how": how, **info}
            except Exception as e:
                st.error(f"Could not join: {e}")
        else:
            P.pop("join", None)

        st.markdown("**Private information (PII)**")
        pii = pl.detect_pii(raw_df)
        pii = {c: k for c, k in pii.items() if c in silver.columns or c in raw_df.columns}
        if not pii:
            st.caption("No columns look like names, emails, phone numbers, addresses or ID numbers.")
            P["pii"], P["pii_method"] = {}, None
        else:
            st.caption("Columns that look like personal information: "
                       + ", ".join(f"**{c}** ({k})" for c, k in pii.items()))
            c1, c2 = st.columns([2, 1])
            pick = c1.multiselect("Protect these columns", list(pii), default=list(pii), key="pl_pii")
            method = c2.selectbox("How", ["hash", "mask", "drop"], key="pl_pii_m",
                                  format_func=lambda m: {"hash": "Replace with a code (still countable)",
                                                         "mask": "Show last 4 characters only",
                                                         "drop": "Remove the column"}[m])
            present = {c: pii[c] for c in pick if c in silver.columns}
            silver = pl.mask_pii(silver, present, method, salt=ctx["label"])
            P["pii"], P["pii_method"] = present, method
        P["silver"] = silver
        st.dataframe(silver.head(10), width="stretch")
        st.caption(f"Silver table: {len(silver):,} rows × {silver.shape[1]} columns.")

    # ---------------- 5. gold ----------------
    with t[4]:
        st.markdown("**Gold: a summary table ready for reports and dashboards.**")
        silver = P.get("silver", clean_df)
        sp = {c: profile.get(c, "numeric" if pd.api.types.is_numeric_dtype(silver[c]) else "categorical")
              for c in silver.columns}
        num = [c for c, k in sp.items() if k == "numeric" and pd.api.types.is_numeric_dtype(silver[c])]
        cat = [c for c, k in sp.items() if k == "categorical" and silver[c].nunique() <= 200]
        dates = [c for c, k in sp.items() if k == "datetime"]
        c1, c2 = st.columns(2)
        by = c1.multiselect("Group by", cat, default=cat[:1], key="pl_by")
        dcol = c2.selectbox("And by period of", ["(none)"] + dates, key="pl_dcol", index=1 if dates else 0)
        period = c2.selectbox("Period", ["Month", "Week", "Quarter", "Year", "Day"], key="pl_period") if dcol != "(none)" else None
        c3, c4 = st.columns(2)
        vals = c3.multiselect("Numbers", num, default=num[:2], key="pl_vals")
        funcs = c4.multiselect("Calculations", list(grouping.FUNCS), default=["Total (sum)", "Average (mean)"], key="pl_funcs")
        gold = None
        if by or dcol != "(none)":
            try:
                gold = grouping.group_summary(silver, by, vals, funcs if vals else [],
                                              None if dcol == "(none)" else dcol, period)
                st.dataframe(gold.head(50), hide_index=True, width="stretch")
                st.caption(f"Gold table: {len(gold):,} rows.")
            except Exception as e:
                st.error(f"Could not build the gold table: {e}")
        P["gold"] = gold
        P["gold_spec"] = {"by": by, "values": vals, "funcs": funcs,
                          "date_col": None if dcol == "(none)" else dcol, "period": period}

    tables = {"bronze": raw_df, "silver": P.get("silver", clean_df), "gold": P.get("gold")}
    config = {
        "name": ctx["label"], "pipeline_version": pl.PIPELINE_VERSION, "missing": ctx["missing"],
        "overrides": ctx["overrides"], "pii": P.get("pii") or {}, "pii_method": P.get("pii_method") or "hash",
        "pii_salt": ctx["label"], "gold": P.get("gold_spec"), "stop_on_fail": True,
        "schema": pl.schema_of(raw_df, {**ctx["auto_profile"], **profile}),
    }
    if P.get("join"):
        config["join_note"] = (f"Joined {P['join']['file']} on {P['join']['left_on']} = {P['join']['right_on']} "
                               f"({P['join']['how']}). Scheduled runs do not repeat joins yet; join in the app.")

    # ---------------- 6. orchestrate ----------------
    with t[5]:
        st.markdown("**Save these steps and re-run them on a schedule.**")
        st.download_button("Download pipeline config (.json)", json.dumps(config, indent=2, default=str),
                           file_name=f"{ctx['label']}_pipeline.json", mime="application/json", key="pl_cfg")
        st.markdown("Run it on a new file from the app folder (the same steps, in order: ingest, validate, bronze, "
                    "silver, gold, warehouse, run log):")
        st.code(f'py pipeline_run.py --config {ctx["label"]}_pipeline.json --input "C:\\data\\new_file.csv" --out "C:\\data\\pipeline_output"',
                language="powershell")
        st.markdown("Schedule it every day at 6 AM with Windows Task Scheduler (one command, in PowerShell):")
        st.code(f'schtasks /Create /SC DAILY /ST 06:00 /TN "Vanguard {ctx["label"]}" /TR '
                f'"py C:\\Users\\profe\\vanguard-mvp\\pipeline_run.py --config C:\\Users\\profe\\vanguard-mvp\\{ctx["label"]}_pipeline.json '
                f'--input C:\\data\\new_file.csv --out C:\\data\\pipeline_output"', language="powershell")
        st.markdown("**Run the same pipeline on a Spark cluster**")
        _dates = [c for c, k in profile.items() if k == "datetime"]
        _parts = (["year", "month"] if _dates else []) or [c for c in (P.get("gold_spec") or {}).get("by", [])][:1]
        st.download_button("Download as a PySpark script (.py)",
                           pt.pyspark_script(ctx["source"], ctx["label"], _dates[0] if _dates else None, _parts,
                                             st.session_state.get("pq_comp", "snappy"), ctx["overrides"], P.get("gold_spec")),
                           file_name=f"{ctx['label']}_pyspark.py", mime="text/x-python", key="pl_pyspark")
        st.caption("Same steps in Spark: read CSV, clean, add year/month/day, write partitioned Parquet, read it back, "
                   "aggregate. Change the hdfs:/ paths to your cluster's storage. "
                   "Each run checks the quality gates first and stops if any fail, then writes a zip of bronze, "
                   "silver and gold tables and adds a line to run_log.csv. The same config works with cron on Mac or "
                   "Linux, or as one task in a larger orchestrator such as Airflow or Dagster.")

    # ---------------- 7. govern & monitor ----------------
    with t[6]:
        st.markdown("**Lineage: where each table came from**")
        lineage = pd.DataFrame([
            {"Table": "bronze", "Built from": ctx["source"], "Rows": len(raw_df), "Columns": raw_df.shape[1],
             "How": "loaded as received"},
            {"Table": "silver", "Built from": "bronze" + (f" + {P['join']['file']}" if P.get("join") else ""),
             "Rows": len(tables["silver"]), "Columns": tables["silver"].shape[1],
             "How": f"{len(ctx['report'])} cleaning steps" + (f"; joined on {P['join']['left_on']}" if P.get("join") else "")
                    + (f"; PII {P['pii_method']}ed: {', '.join(P['pii'])}" if P.get("pii") else "")},
            {"Table": "gold", "Built from": "silver", "Rows": len(tables["gold"]) if tables["gold"] is not None else 0,
             "Columns": tables["gold"].shape[1] if tables["gold"] is not None else 0,
             "How": ("grouped by " + ", ".join(filter(None, (P.get("gold_spec") or {}).get("by", []) +
                                                       [(P.get("gold_spec") or {}).get("date_col")])))
                    if tables["gold"] is not None else "not built"},
        ])
        st.dataframe(lineage, hide_index=True, width="stretch")
        st.markdown(f"**Quality gate:** {STATUS_ICON[P.get('gate', 'PASS')]} {P.get('gate', 'PASS')}  ·  "
                    f"**Private columns protected:** {len(P.get('pii') or {})}")

        st.markdown("**Query the warehouse with SQL** (tables: `bronze`, `silver`, `gold`)")
        default_sql = "SELECT * FROM gold LIMIT 20" if tables["gold"] is not None else "SELECT * FROM silver LIMIT 20"
        sql = st.text_area("SQL", default_sql, key="pl_sql", height=90,
                           help="Read-only. Column names with spaces or symbols are turned into underscores.")
        if st.button("Run query", key="pl_run_sql"):
            try:
                t0 = time.time()
                res = pl.query(tables, sql)
                st.caption(f"{len(res):,} rows in {time.time() - t0:.2f} s")
                st.dataframe(res, hide_index=True, width="stretch")
            except Exception as e:
                st.error(f"Query failed: {e}")

        manifest = {"pipeline": ctx["label"], "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                    "source": ctx["source"], "fingerprint_bronze": ctx["fingerprint"], "gate": P.get("gate"),
                    "lineage": lineage.to_dict("records"), "cleaning_steps": ctx["report"], "config": config}
        if st.button("Build pipeline bundle (zip)", type="primary", key="pl_bundle"):
            with st.spinner("Writing Parquet files and the DuckDB warehouse..."):
                st.session_state["pl_zip"] = pl.bundle(tables, manifest, P["checks"],
                                                       {"pipeline_config.json": json.dumps(config, indent=2, default=str)})
            st.session_state.setdefault("pl_runs", []).append(
                {"Run at": manifest["built_at"], "Gate": P.get("gate"), "Bronze rows": len(raw_df),
                 "Silver rows": len(tables["silver"]),
                 "Gold rows": len(tables["gold"]) if tables["gold"] is not None else 0})
        if st.session_state.get("pl_zip"):
            st.download_button("Download bundle: bronze, silver, gold, warehouse.duckdb, checks, manifest",
                               st.session_state["pl_zip"], file_name=f"{ctx['label']}_pipeline_bundle.zip",
                               mime="application/zip", key="pl_zip_dl")
        if st.session_state.get("pl_runs"):
            st.markdown("**Run log (this session)**")
            st.dataframe(pd.DataFrame(st.session_state["pl_runs"]), hide_index=True, width="stretch")
        st.caption("Scheduled runs write the same information to run_log.csv in the output folder, so you can see "
                   "every run, how long it took, and whether the quality gate passed.")

        st.divider()
        from db_ui import render_writer
        render_writer(tables, ctx["label"])

    with t[7]:
        render_parquet(P.get("silver", clean_df), profile, ctx["label"])


def render_parquet(df: pd.DataFrame, profile: dict, label: str):
    st.markdown("**Parquet: columnar, compressed storage** (the format Spark and Hadoop use)")
    if st.button("Compare CSV with Parquet compressions", key="pq_cmp"):
        with st.spinner("Writing the table in each format..."):
            st.session_state["pq_sizes"] = pt.size_comparison(df)
    if st.session_state.get("pq_sizes") is not None:
        st.dataframe(st.session_state["pq_sizes"], hide_index=True, width="stretch")
        st.caption("Parquet stores each column together and compresses it, so files are smaller and a query that needs "
                   "two columns reads only those two. Snappy is Spark's default: fast, moderate compression. "
                   "gzip and brotli are smaller but slower. (LZO is not available here.)")

    st.markdown("**Partitioned dataset** (Exercise 22)")
    dates = [c for c, k in profile.items() if k == "datetime" and c in df.columns]
    work = df
    c1, c2 = st.columns(2)
    if dates:
        dcol = c1.selectbox("Add year / month / day columns from", ["(none)"] + dates, index=1, key="pq_date")
        if dcol != "(none)":
            work = pt.add_date_parts(df, dcol)
    comp = c2.selectbox("Compression", pt.COMPRESSIONS, key="pq_comp")
    cand = [c for c in work.columns if c in ("year", "month", "day") or
            (profile.get(c) == "categorical" and work[c].nunique() <= 500)]
    default = [c for c in ("year", "month") if c in cand] or cand[:1]
    cols = st.multiselect("Partition by (nested in this order)", cand, default=default, key="pq_cols")
    if not cols:
        st.caption("Pick at least one partition column.")
        return
    adv = pt.partition_advice(work, cols)
    st.dataframe(adv, hide_index=True, width="stretch")
    st.caption(f"This makes {adv.attrs['folders']:,} folders. Good partition columns are ones you often filter on "
               "(like year), with a moderate number of values.")
    if st.button("Write partitioned Parquet", type="primary", key="pq_write"):
        old = st.session_state.get("pq_folder")
        if old:
            pt.cleanup(old)
        with st.spinner("Writing..."):
            folder, tree, n, kb = pt.write_partitioned(work, cols, comp)
        st.session_state.update(pq_folder=folder, pq_tree=tree, pq_n=n, pq_kb=kb, pq_cols_used=cols,
                                pq_zip=pt.zip_folder(folder), pq_vals=sorted(work[cols[0]].dropna().astype(str).unique()))
    if st.session_state.get("pq_folder") and st.session_state.get("pq_cols_used") == cols:
        st.success(f"Wrote {st.session_state['pq_n']:,} Parquet files ({st.session_state['pq_kb']:,} KB) in folders by "
                   + " / ".join(cols) + ".")
        st.code(st.session_state["pq_tree"], language="text")
        st.download_button("Download partitioned dataset (.zip)", st.session_state["pq_zip"],
                           file_name=f"{label}_partitioned_parquet.zip", mime="application/zip", key="pq_dl")
        st.markdown("**Partition pruning: read only what the filter needs**")
        val = st.selectbox(f"Filter {cols[0]} =", st.session_state["pq_vals"], key="pq_val")
        r = pt.pruning_demo(st.session_state["pq_folder"], pt._safe(work[[cols[0]]]).columns[0], val)
        st.dataframe(pd.DataFrame({"Query": ["All data", f"{cols[0]} = {val}"],
                                   "Files read": [r["files_all"], r["files_part"]],
                                   "Rows": [r["rows_all"], r["rows_part"]],
                                   "Seconds": [r["sec_all"], r["sec_part"]]}), hide_index=True, width="stretch")
        st.caption("With a filter on a partition column, only that folder's files are opened. On a small file the time "
                   "difference is tiny; on billions of rows it is the difference between minutes and seconds.")


def _guess_key(left: list, right: list) -> int:
    shared = [c for c in left if c in right]
    if shared:
        idish = [c for c in shared if any(k in c.lower() for k in ("id", "key", "code", "number"))]
        return left.index((idish or shared)[0])
    return 0
