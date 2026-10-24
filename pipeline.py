"""
Data pipeline stages, shared by the app (Data pipeline mode) and the command-line runner
(pipeline_run.py), so a pipeline built on screen can be re-run on a schedule.

  1. Ingest            load a file, web address or API (data_cleaner.load_file / load_url)
  2. Validate          quality gates: PASS / WARN / FAIL checks, optionally against a saved schema contract
  3. Bronze            the raw data exactly as received, with a fingerprint, never changed
  4. Silver            cleaned: types, gaps, duplicates (data_cleaner.clean_dataframe), PII masked, joins
  5. Gold              aggregated tables ready for reports (grouping.group_summary)
  6. Orchestrate       a pipeline config (JSON) that pipeline_run.py repeats; schedule it with
                       Windows Task Scheduler or cron
  7. Govern / monitor  lineage, run log, check results, and a DuckDB warehouse file to query with SQL
"""
import hashlib
import io
import json
import re
import time
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

PIPELINE_VERSION = 1


# ------------------------------------------------------------------ fingerprints
def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes()).hexdigest()


# ------------------------------------------------------------------ 2. validation
def schema_of(df: pd.DataFrame, profile: dict) -> dict:
    """A schema contract: the columns and types this pipeline expects next time."""
    return {"columns": {str(c): profile.get(c, "text") for c in df.columns},
            "rows_min": max(1, int(len(df) * 0.5))}


_FUTURE_OK = ("end", "expir", "due", "scheduled", "planned", "target")


def validate(raw: pd.DataFrame, profile: dict | None = None, schema: dict | None = None,
             max_missing_warn: float = 0.2, max_missing_fail: float = 0.8, max_dup_warn: float = 0.05) -> pd.DataFrame:
    """Quality gates run before processing. Each check is PASS, WARN or FAIL with a reason."""
    rows = []

    def add(check, status, detail):
        rows.append({"Check": check, "Result": status, "Detail": detail})

    add("File has data", "PASS" if len(raw) and raw.shape[1] else "FAIL", f"{len(raw):,} rows × {raw.shape[1]} columns")
    dupe_names = [c for c in raw.columns if list(raw.columns).count(c) > 1]
    add("Column names are unique", "FAIL" if dupe_names else "PASS",
        f"repeated: {', '.join(map(str, set(dupe_names)))}" if dupe_names else "all unique")
    blank = [c for c in raw.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
    add("Every column has a name", "WARN" if blank else "PASS", f"{len(blank)} unnamed columns" if blank else "ok")

    if schema:
        exp = schema.get("columns", {})
        missing = [c for c in exp if c not in raw.columns]
        extra = [c for c in raw.columns if c not in exp]
        add("Expected columns are present (schema)", "FAIL" if missing else "PASS",
            f"missing: {', '.join(missing[:10])}" if missing else f"all {len(exp)} present")
        add("No unexpected columns (schema)", "WARN" if extra else "PASS",
            f"new: {', '.join(map(str, extra[:10]))}" if extra else "none")
        if profile:
            changed = [f"{c} ({exp[c]} → {profile.get(c)})" for c in exp
                       if c in profile and profile.get(c) != exp[c] and not (exp[c] == "identifier")]
            add("Column types match the schema", "WARN" if changed else "PASS",
                "; ".join(changed[:8]) if changed else "all match")
        rmin = schema.get("rows_min")
        if rmin:
            add("Enough rows (schema)", "FAIL" if len(raw) < rmin else "PASS", f"{len(raw):,} rows; expected at least {rmin:,}")

    n = len(raw) or 1
    miss = raw.isna().mean()
    bad = miss[miss >= max_missing_fail]
    warn = miss[(miss >= max_missing_warn) & (miss < max_missing_fail)]
    add(f"No column is {max_missing_fail:.0%}+ empty", "FAIL" if len(bad) else "PASS",
        ", ".join(f"{c} {v:.0%}" for c, v in bad.items()) if len(bad) else "ok")
    add(f"Columns under {max_missing_warn:.0%} empty", "WARN" if len(warn) else "PASS",
        ", ".join(f"{c} {v:.0%}" for c, v in warn.items()) if len(warn) else "ok")
    dup = raw.duplicated().mean()
    add("Duplicate rows", "WARN" if dup > max_dup_warn else "PASS", f"{dup:.1%} of rows are exact duplicates")

    profile = profile or {}
    for c, t in profile.items():
        if c not in raw.columns:
            continue
        s = raw[c]
        if t == "numeric" and not pd.api.types.is_numeric_dtype(s):
            conv = pd.to_numeric(s.astype(str).str.replace(r"[\$,%]", "", regex=True), errors="coerce")
            lost = int(conv.isna().sum() - s.isna().sum())
            if lost > 0:
                add(f"'{c}' values are numbers", "WARN", f"{lost:,} values are not numbers (will become gaps)")
        if t == "numeric" and any(k in c.lower() for k in ("amount", "price", "cost", "revenue", "sales", "quantity", "qty", "age")):
            neg = int((pd.to_numeric(s, errors="coerce") < 0).sum())
            if neg:
                add(f"'{c}' has no negative values", "WARN", f"{neg:,} negative values; check for refunds or errors")
        if t == "datetime":
            d = pd.to_datetime(s, errors="coerce")
            bad_d = int(d.isna().sum() - s.isna().sum())
            if bad_d > 0:
                add(f"'{c}' values are dates", "WARN", f"{bad_d:,} values could not be read as dates")
            fut = int((d > pd.Timestamp.now() + pd.Timedelta(days=1)).sum())
            if fut and not any(k in c.lower() for k in _FUTURE_OK):
                add(f"'{c}' has no future dates", "WARN", f"{fut:,} dates are after today")
    out = pd.DataFrame(rows)
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    return out.sort_values("Result", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)


def gate(checks: pd.DataFrame) -> str:
    if (checks["Result"] == "FAIL").any():
        return "FAIL"
    return "WARN" if (checks["Result"] == "WARN").any() else "PASS"


# ------------------------------------------------------------------ 4. PII
PII_PATTERNS = {
    "email": re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$"),
    "phone": re.compile(r"^\+?1?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"),
    "SSN": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    "card number": re.compile(r"^(?:\d[ -]?){13,19}$"),
}
PII_NAME_HINTS = {"name": ("first_name", "last_name", "full_name", "customer_name", "patient_name", "contact_name", "name"),
                  "address": ("address", "street", "addr"), "date of birth": ("dob", "birth"),
                  "email": ("email", "e-mail"), "phone": ("phone", "mobile", "cell"), "SSN": ("ssn", "social_security")}


def detect_pii(df: pd.DataFrame) -> dict:
    """Columns that look like personal information, from their values or their names."""
    found = {}
    for c in df.columns:
        s = df[c].dropna().astype(str).str.strip()
        if s.empty:
            continue
        sample = s.head(300)
        for kind, rx in PII_PATTERNS.items():
            if kind == "card number" and pd.api.types.is_numeric_dtype(df[c]):
                continue
            if sample.map(lambda v: bool(rx.match(v))).mean() >= 0.8:
                found[c] = kind
                break
        if c in found:
            continue
        low = str(c).lower()
        for kind, hints in PII_NAME_HINTS.items():
            if any(low == h or low.startswith(h + "_") or low.endswith("_" + h) or h in low.split("_") for h in hints):
                if not pd.api.types.is_numeric_dtype(df[c]) or kind in ("phone", "SSN"):
                    found[c] = kind
                    break
    return found


def mask_pii(df: pd.DataFrame, cols: dict, method: str = "hash", salt: str = "") -> pd.DataFrame:
    """hash: same input gives the same code, so rows can still be counted and joined, but the value is hidden.
    mask: keep the last 4 characters.  drop: remove the column."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if method == "drop":
            out = out.drop(columns=[c])
        elif method == "mask":
            out[c] = out[c].map(lambda v: ("•" * max(0, len(str(v)) - 4) + str(v)[-4:]) if pd.notna(v) else v)
        else:
            out[c] = out[c].map(lambda v: "id_" + hashlib.sha256((salt + str(v)).encode()).hexdigest()[:12]
                                if pd.notna(v) else v)
    return out


# ------------------------------------------------------------------ 4. joins
def join(left: pd.DataFrame, right: pd.DataFrame, left_on: list, right_on: list, how: str = "left",
         suffix: str = "_2") -> tuple[pd.DataFrame, dict]:
    l, r = left.copy(), right.copy()
    for a, b in zip(left_on, right_on):            # match keys as text so 0123 and 123 do not silently mismatch
        l[a] = l[a].astype(str).str.strip()
        r[b] = r[b].astype(str).str.strip()
    dup_keys = int(r.duplicated(subset=right_on).sum())
    merged = l.merge(r, left_on=left_on, right_on=right_on, how=how, suffixes=("", suffix), indicator=True)
    info = {
        "left_rows": len(l), "right_rows": len(r), "result_rows": len(merged),
        "matched": int((merged["_merge"] == "both").sum()),
        "left_only": int((merged["_merge"] == "left_only").sum()),
        "right_only": int((merged["_merge"] == "right_only").sum()),
        "duplicate_keys_in_second_file": dup_keys,
    }
    return merged.drop(columns=["_merge"]), info


# ------------------------------------------------------------------ 5/7. warehouse + bundle
def _safe_table(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet and DuckDB need simple column names and no mixed-type object columns."""
    out = df.copy()
    out.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_") or f"col_{i}" for i, c in enumerate(out.columns)]
    seen = {}
    cols = []
    for c in out.columns:
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    out.columns = cols
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return out


def warehouse_bytes(tables: dict) -> bytes:
    """A DuckDB database file holding bronze, silver and gold tables, queryable with SQL."""
    import duckdb
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    os.remove(path)
    con = duckdb.connect(path)
    try:
        for name, df in tables.items():
            if df is None:
                continue
            t = _safe_table(df)
            con.register("tmp_df", t)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df")
            con.unregister("tmp_df")
    finally:
        con.close()
    data = open(path, "rb").read()
    os.remove(path)
    return data


def query(tables: dict, sql: str) -> pd.DataFrame:
    """Run read-only SQL against the bronze / silver / gold tables in memory."""
    import duckdb
    if not re.match(r"^\s*(select|with)\b", sql, re.I) or ";" in sql.strip().rstrip(";"):
        raise ValueError("Only one SELECT (or WITH ... SELECT) statement is allowed.")
    con = duckdb.connect()
    try:
        for name, df in tables.items():
            if df is not None:
                con.register(name, _safe_table(df))
        return con.execute(sql.strip().rstrip(";")).fetchdf()
    finally:
        con.close()


def bundle(tables: dict, manifest: dict, checks: pd.DataFrame, extras: dict | None = None) -> bytes:
    """One zip: bronze/silver/gold as Parquet, a DuckDB warehouse, the checks, and a manifest (lineage)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            if df is None:
                continue
            pq = io.BytesIO()
            _safe_table(df).to_parquet(pq, index=False)
            z.writestr(f"{name}.parquet", pq.getvalue())
        z.writestr("warehouse.duckdb", warehouse_bytes(tables))
        z.writestr("quality_checks.csv", checks.to_csv(index=False))
        z.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
        for name, data in (extras or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


# ------------------------------------------------------------------ 6. run from a config (scheduled)
def run_config(config: dict, input_path: str, out_dir: str, log=print) -> dict:
    """Repeat a saved pipeline on a new file: ingest → validate → bronze → silver → gold → warehouse + run log."""
    import os
    from data_cleaner import load_file, load_url, clean_dataframe
    import grouping

    t0 = time.time()
    started = datetime.now().isoformat(timespec="seconds")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. ingest
    if input_path.lower().startswith(("http://", "https://")):
        raw = load_url(input_path)
    else:
        class _F:
            def __init__(self, p):
                self.name, self._f = os.path.basename(p), open(p, "rb")

            def read(self, *a):
                return self._f.read(*a)
        raw = load_file(_F(input_path))
    if config.get("rename"):
        raw = raw.rename(columns=config["rename"])
    log(f"Ingested {len(raw):,} rows × {raw.shape[1]} columns from {input_path}")

    # 2. validate (profile from a dry clean so type checks work)
    _, prof, _, _, _ = clean_dataframe(raw, config.get("missing", "Automatic"), config.get("overrides"))
    checks = validate(raw, prof, config.get("schema"))
    status = gate(checks)
    log(f"Quality gate: {status}")
    for _, r in checks[checks["Result"] != "PASS"].iterrows():
        log(f"  {r['Result']}: {r['Check']} ({r['Detail']})")

    result = {"started": started, "input": input_path, "rows_in": len(raw), "gate": status,
              "fingerprint": fingerprint(raw)}
    if status == "FAIL" and config.get("stop_on_fail", True):
        result.update({"status": "stopped at quality gate", "seconds": round(time.time() - t0, 1)})
        _append_log(out_dir, result)
        return result

    # 3. bronze, 4. silver, 5. gold
    silver, profile, report, _, _ = clean_dataframe(raw, config.get("missing", "Automatic"), config.get("overrides"))
    if config.get("pii"):
        silver = mask_pii(silver, config["pii"], config.get("pii_method", "hash"), config.get("pii_salt", ""))
    gold = None
    g = config.get("gold")
    if g and (g.get("by") or g.get("date_col")):
        gold = grouping.group_summary(silver, g.get("by", []), g.get("values", []), g.get("funcs", []),
                                      g.get("date_col"), g.get("period"))
    tables = {"bronze": raw, "silver": silver, "gold": gold}
    manifest = {"pipeline": config.get("name", "pipeline"), "run_at": started, "source": input_path,
                "fingerprint_bronze": result["fingerprint"], "rows": {k: (len(v) if v is not None else 0) for k, v in tables.items()},
                "gate": status, "cleaning_steps": report, "config": config}
    path = os.path.join(out_dir, f"{config.get('name', 'pipeline')}_{stamp}.zip")
    with open(path, "wb") as f:
        f.write(bundle(tables, manifest, checks))
    result.update({"status": "ok", "rows_silver": len(silver), "rows_gold": len(gold) if gold is not None else 0,
                   "output": path, "seconds": round(time.time() - t0, 1)})
    _append_log(out_dir, result)
    log(f"Wrote {path} in {result['seconds']} s")
    return result


def _append_log(out_dir: str, result: dict):
    import os
    path = os.path.join(out_dir, "run_log.csv")
    row = pd.DataFrame([{k: v for k, v in result.items()}])
    row.to_csv(path, mode="a", index=False, header=not os.path.exists(path))
