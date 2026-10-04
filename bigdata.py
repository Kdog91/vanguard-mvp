"""
Large-file mode.

Files too big to load into memory are handled with DuckDB, which reads the file
from disk in pieces instead of all at once. Three things happen on the FULL file:

  1. summarize_file   - row count and per-column statistics for every row
  2. clean_full_file  - the same cleaning rules as the normal cleaner, applied to
                        every row and saved as a Parquet file
  3. daily_totals     - date totals for forecasting, computed from every row

Everything else in the app (charts, models, tests) runs on a random sample of the
file, drawn by load_sample. A sample is used because fitting eight models with
cross-validation on millions of rows would take a very long time on one machine.
"""
import os
import time
import tempfile
import duckdb
import pandas as pd

LARGE_FILE_BYTES = 25 * 1024 * 1024      # uploads bigger than this are checked for large-file mode
LARGE_ROW_THRESHOLD = 200_000            # files with more rows than this use large-file mode
SUPPORTED = (".csv", ".tsv", ".txt", ".parquet", ".jsonl", ".ndjson")


def _q(name: str) -> str:
    """Quote a column name for SQL."""
    return '"' + str(name).replace('"', '""') + '"'


def source_sql(path: str) -> str:
    """The SQL expression that reads the file."""
    p = str(path).replace("'", "''")
    low = p.lower()
    if low.endswith(".parquet"):
        return f"read_parquet('{p}')"
    if low.endswith((".jsonl", ".ndjson")):
        return f"read_json_auto('{p}')"
    if low.endswith((".csv", ".tsv", ".txt")):
        return f"read_csv_auto('{p}', sample_size=100000)"
    raise ValueError("Large-file mode supports CSV, TSV, TXT, Parquet and JSON Lines files.")


def _connect():
    """A DuckDB session that may spill to disk, so big sorts and de-duplication don't run out of memory."""
    con = duckdb.connect(os.path.join(tempfile.mkdtemp(prefix="vanguard_"), "work.duckdb"))
    con.execute("SET preserve_insertion_order=false")
    return con


def count_rows(path: str) -> int:
    con = duckdb.connect()
    try:
        return int(con.execute(f"SELECT count(*) FROM {source_sql(path)}").fetchone()[0])
    finally:
        con.close()


def summarize_file(path: str) -> dict:
    """Row count and per-column statistics over every row in the file."""
    t0 = time.time()
    con = duckdb.connect()
    try:
        s = con.execute(f"SUMMARIZE SELECT * FROM {source_sql(path)}").df()
    finally:
        con.close()
    rows = int(s["count"].max()) if len(s) else 0
    table = pd.DataFrame({
        "Column": s["column_name"],
        "Stored as": s["column_type"],
        "Missing (%)": pd.to_numeric(s["null_percentage"], errors="coerce").round(2),
        "Distinct values (approx.)": s["approx_unique"],
        "Min": s["min"].astype(str).str.slice(0, 24),
        "Max": s["max"].astype(str).str.slice(0, 24),
        "Mean": pd.to_numeric(s["avg"], errors="coerce").round(2),
    })
    return {"rows": rows, "columns": len(s), "table": table,
            "size_mb": os.path.getsize(path) / 1e6, "seconds": time.time() - t0,
            "types": dict(zip(s["column_name"], s["column_type"]))}


def load_sample(path: str, n: int, seed: int = 42) -> pd.DataFrame:
    """A random sample of n rows, as an ordinary table the rest of the app can use."""
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT * FROM {source_sql(path)} USING SAMPLE reservoir({int(n)} ROWS) REPEATABLE ({int(seed)})"
        ).df()
    finally:
        con.close()


def _number_expr(col: str, stored_type: str) -> str:
    """Turn a column into a number, stripping $ , % if it is stored as text."""
    if "VARCHAR" in stored_type.upper():
        return f"TRY_CAST(regexp_replace(trim({_q(col)}), '[$,%]', '', 'g') AS DOUBLE)"
    return f"TRY_CAST({_q(col)} AS DOUBLE)"


def clean_full_file(path: str, profile: dict, identifier_cols: list, types: dict, out_path: str) -> dict:
    """
    Apply the cleaner's rules to every row and save the result as Parquet.
    `profile` is the column-type decision made on the sample (numeric / categorical / datetime / text).
    Rules, same as the normal cleaner: text-to-number conversion, exact duplicates removed,
    identifier columns dropped, numeric gaps filled with the median, category gaps with the
    most common value, outliers counted (1.5 x IQR) but not removed.
    """
    t0 = time.time()
    report = []
    con = _connect()
    try:
        src = source_sql(path)
        cols = list(types)
        select = []
        for c in cols:
            if profile.get(c) == "numeric":
                select.append(f"{_number_expr(c, types[c])} AS {_q(c)}")
                if "VARCHAR" in types[c].upper():
                    report.append(f"Converted column '{c}' from text to numeric (stripped $ , % symbols).")
            elif profile.get(c) == "datetime":
                select.append(f"TRY_CAST({_q(c)} AS TIMESTAMP) AS {_q(c)}")
            else:
                select.append(_q(c))
        rows_in = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
        con.execute(f"CREATE TABLE t AS SELECT DISTINCT {', '.join(select)} FROM {src}")
        rows_out = con.execute("SELECT count(*) FROM t").fetchone()[0]
        if rows_in > rows_out:
            report.append(f"Removed {rows_in - rows_out:,} duplicate rows.")
        for c in identifier_cols:
            report.append(f"Removed identifier column '{c}'.")

        keep = [c for c in cols if c not in identifier_cols and c in profile]
        final, outliers = [], {}
        for c in keep:
            kind = profile[c]
            missing = con.execute(f"SELECT count(*) FROM t WHERE {_q(c)} IS NULL").fetchone()[0]
            if kind == "numeric":
                q1, med, q3 = con.execute(
                    f"SELECT quantile_cont({_q(c)}, [0.25, 0.5, 0.75]) FROM t").fetchone()[0]
                if missing:
                    report.append(f"Filled {missing:,} missing values in '{c}' with median ({med:.2f}).")
                final.append(f"COALESCE({_q(c)}, {med!r}) AS {_q(c)}")
                iqr = q3 - q1
                if iqr > 0:
                    n_out = con.execute(
                        f"SELECT count(*) FROM t WHERE {_q(c)} < {q1 - 1.5 * iqr!r} OR {_q(c)} > {q3 + 1.5 * iqr!r}"
                    ).fetchone()[0]
                    if n_out:
                        outliers[c] = n_out
                        report.append(f"Flagged {n_out:,} statistical outliers in '{c}' (IQR method) — not removed.")
            elif kind == "categorical" and missing:
                mode = con.execute(f"SELECT mode({_q(c)}) FROM t").fetchone()[0]
                lit = "'" + str(mode).replace("'", "''") + "'" if isinstance(mode, str) else repr(mode)
                report.append(f"Filled {missing:,} missing values in '{c}' with most common value ('{mode}').")
                final.append(f"COALESCE({_q(c)}, {lit}) AS {_q(c)}")
            else:
                final.append(_q(c))
        out_sql = out_path.replace("'", "''")
        con.execute(f"COPY (SELECT {', '.join(final)} FROM t) TO '{out_sql}' (FORMAT PARQUET)")
    finally:
        con.close()
    return {"rows_in": rows_in, "rows_out": rows_out, "columns_out": len(keep), "report": report,
            "outliers": outliers, "out_path": out_path, "out_mb": os.path.getsize(out_path) / 1e6,
            "seconds": time.time() - t0}


def daily_totals(path: str, date_col: str, value_col: str, types: dict) -> pd.DataFrame:
    """Per-day sum and count of a number over every row. Small result, however large the file."""
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT CAST(TRY_CAST({_q(date_col)} AS TIMESTAMP) AS DATE) AS d, "
            f"sum(v) AS total, count(v) AS n FROM "
            f"(SELECT {_q(date_col)}, {_number_expr(value_col, types[value_col])} AS v FROM {source_sql(path)}) "
            f"WHERE v IS NOT NULL GROUP BY 1 HAVING d IS NOT NULL ORDER BY 1"
        ).df()
    finally:
        con.close()
