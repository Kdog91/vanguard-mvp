"""
Parquet and partitions (DAT-260 lesson 4.4, Exercises 21-22):
  - convert a table to Parquet with a chosen compression (snappy, gzip, brotli, lz4, zstd, none)
    and compare sizes with CSV
  - add year / month / day columns from a date and write a partitioned dataset
    (folders such as year=2026/month=01/part-0.parquet)
  - advice on partition columns (too few or too many distinct values)
  - read the partitioned data back with a filter and show that only the matching folders are read
  - the same steps written as a PySpark script, to run on a Spark cluster (HDFS, Databricks...)
"""
import io
import os
import shutil
import tempfile
import time
import zipfile

import pandas as pd

COMPRESSIONS = ["snappy", "gzip", "brotli", "lz4", "zstd", "none"]   # lzo is not available in pyarrow


def _safe(df: pd.DataFrame) -> pd.DataFrame:
    from pipeline import _safe_table
    return _safe_table(df)


def add_date_parts(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[date_col], errors="coerce")
    out["year"] = d.dt.year.astype("Int64")
    out["month"] = d.dt.month.astype("Int64").map(lambda m: f"{m:02d}" if pd.notna(m) else None)
    out["day"] = d.dt.day.astype("Int64").map(lambda v: f"{v:02d}" if pd.notna(v) else None)
    return out


def size_comparison(df: pd.DataFrame) -> pd.DataFrame:
    t = _safe(df)
    rows = []
    t0 = time.time()
    csv = t.to_csv(index=False).encode()
    rows.append({"Format": "CSV", "Compression": "none", "Size (KB)": round(len(csv) / 1024, 1),
                 "Write time (s)": round(time.time() - t0, 3)})
    for c in COMPRESSIONS:
        buf = io.BytesIO()
        t0 = time.time()
        t.to_parquet(buf, index=False, compression=None if c == "none" else c)
        rows.append({"Format": "Parquet", "Compression": c, "Size (KB)": round(buf.tell() / 1024, 1),
                     "Write time (s)": round(time.time() - t0, 3)})
    out = pd.DataFrame(rows)
    out["% of CSV size"] = (out["Size (KB)"] / out["Size (KB)"].iloc[0] * 100).round(1)
    return out


def partition_advice(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for c in cols:
        k = int(df[c].nunique(dropna=True))
        per = len(df) / k if k else 0
        if k <= 1:
            note = "Only one value: partitioning by it gives no benefit."
        elif k > 500:
            note = "Too many values: thousands of tiny files will slow things down."
        elif per < 100:
            note = "Few rows per folder; fine for a demo, but on big data aim for much larger parts."
        else:
            note = "Good partition column."
        rows.append({"Column": c, "Distinct values": k, "Rows per value (avg)": round(per, 1), "Advice": note})
    out = pd.DataFrame(rows)
    combos = int(df[cols].drop_duplicates().shape[0]) if cols else 0
    out.attrs["folders"] = combos
    return out


def write_partitioned(df: pd.DataFrame, cols: list, compression: str = "snappy"):
    """Write a partitioned Parquet dataset to a temporary folder. Returns (folder, tree text, file count, size KB)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = _safe(df)
    cols = [_safe(df[[c]]).columns[0] for c in cols]
    folder = tempfile.mkdtemp(prefix="vanguard_parquet_")
    table = pa.Table.from_pandas(t, preserve_index=False)
    pq.write_to_dataset(table, root_path=folder, partition_cols=cols,
                        compression=None if compression == "none" else compression)
    lines, n_files, size = [], 0, 0
    for root, dirs, files in sorted(os.walk(folder)):
        dirs.sort()
        rel = os.path.relpath(root, folder)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if rel != ".":
            lines.append("    " * (depth - 1) + os.path.basename(root) + "/")
        for f in sorted(files):
            n_files += 1
            size += os.path.getsize(os.path.join(root, f))
            if len(lines) < 60:
                lines.append("    " * depth + f)
    if len(lines) >= 60:
        lines.append("... (more folders not shown)")
    return folder, "\n".join(lines), n_files, round(size / 1024, 1)


def zip_folder(folder: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, folder))
    return buf.getvalue()


def pruning_demo(folder: str, col: str, value) -> dict:
    """Read every file vs only the folder for one partition value, with DuckDB (hive partitioning)."""
    import duckdb
    pattern = os.path.join(folder, "**", "*.parquet").replace("\\", "/")
    con = duckdb.connect()
    try:
        t0 = time.time()
        full = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pattern}', hive_partitioning=true)").fetchone()[0]
        t_full = time.time() - t0
        all_files = con.execute(f"SELECT COUNT(*) FROM glob('{pattern}')").fetchone()[0]
        t0 = time.time()
        part = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pattern}', hive_partitioning=true) "
                           f"WHERE CAST({col} AS VARCHAR) = ?", [str(value)]).fetchone()[0]
        t_part = time.time() - t0
        sub = os.path.join(folder, f"{col}={value}", "**", "*.parquet").replace("\\", "/")
        files_read = con.execute(f"SELECT COUNT(*) FROM glob('{sub}')").fetchone()[0]
    finally:
        con.close()
    return {"rows_all": int(full), "rows_part": int(part), "files_all": int(all_files),
            "files_part": int(files_read), "sec_all": round(t_full, 4), "sec_part": round(t_part, 4)}


def cleanup(folder: str):
    shutil.rmtree(folder, ignore_errors=True)


def pyspark_script(source: str, label: str, date_col: str | None, part_cols: list, compression: str,
                   overrides: dict, gold: dict | None) -> str:
    """The same pipeline as a PySpark script for a Spark cluster."""
    import re
    label = re.sub(r"\W+", "_", label).strip("_") or "data"
    lines = [
        '"""',
        f"PySpark version of the Vanguard Analytics pipeline for '{label}'.",
        "Run on a Spark cluster (spark-submit, Databricks, EMR) or locally with: pip install pyspark",
        '"""',
        "from pyspark.sql import SparkSession, functions as F",
        "",
        f'spark = SparkSession.builder.appName("vanguard_{label}").getOrCreate()',
        "",
        "# 1. Ingest: read the raw CSV (header row, let Spark infer types)",
        f'df = spark.read.csv("hdfs:/data/{os.path.basename(str(source))}", header=True, inferSchema=True)',
        "df.printSchema()",
        "df.show(5)                                   # Spark is lazy: show() is the action that runs the read",
        "df.select(df.columns[0], df.columns[1]).show(5)   # select returns a new DataFrame",
        "",
        "# Other sources Spark reads the same way (Exercise 19):",
        '# df = spark.read.json("hdfs://namenode/data/myjsonfile.json")',
        '# df = spark.read.orc("hdfs:/data/myfile.orc")',
        '# df = spark.read.jdbc("jdbc:postgresql://server:5432/mydb", table="mytable",',
        '#                      properties={"user": "USER", "password": "PASSWORD", "driver": "org.postgresql.Driver"})',
        "",
        "# 2. Clean: drop exact duplicates and rows that are completely empty",
        "df = df.dropDuplicates().dropna(how='all')",
    ]
    cast = {"numeric": "double", "datetime": "date", "categorical": "string", "text": "string"}
    for c, t in (overrides or {}).items():
        if t in cast:
            lines.append(f'df = df.withColumn("{c}", F.col("{c}").cast("{cast[t]}"))')
        elif t in ("drop column", "identifier (removed)"):
            lines.append(f'df = df.drop("{c}")')
    if date_col:
        lines += ["", "# 3. Add partition columns from the date",
                  f'df = df.withColumn("{date_col}", F.to_date("{date_col}"))',
                  f'df = (df.withColumn("year", F.year("{date_col}"))',
                  f'        .withColumn("month", F.lpad(F.month("{date_col}").cast("string"), 2, "0"))',
                  f'        .withColumn("day", F.lpad(F.dayofmonth("{date_col}").cast("string"), 2, "0")))']
    lines += ["", f"# 4. Write Parquet ({compression}), partitioned by {', '.join(part_cols) or 'nothing'}",
              f'(df.write.mode("overwrite")' + (f'.partitionBy({", ".join(repr(c) for c in part_cols)})' if part_cols else "")
              + f'\n   .option("compression", "{compression if compression != "none" else "none"}")'
              + f'\n   .parquet("hdfs:/data/{label}_partitioned"))',
              "",
              "# 5. Read it back; Spark finds the partitions from the folder names",
              f'df_pq = spark.read.parquet("hdfs:/data/{label}_partitioned")']
    if part_cols:
        lines += [f"# filtering on a partition column reads only the matching folders (partition pruning)",
                  f'df_pq.filter(F.col("{part_cols[0]}") == "VALUE").count()']
    if date_col:
        lines += ["", "# 6. Minimum, mean and maximum per year, month and day (Exercise 22)",
                  "num_cols = [c for c, t in df_pq.dtypes if t in ('int', 'bigint', 'double', 'float') and c not in ('year',)]",
                  "aggs = [f(c) for c in num_cols for f in (F.min, F.mean, F.max)]",
                  'for level in (["year"], ["year", "month"], ["year", "month", "day"]):',
                  "    df_pq.groupBy(*level).agg(*aggs).orderBy(*level).show(20)"]
    if gold and (gold.get("by") or gold.get("date_col")) and gold.get("values"):
        fmap = {"Total (sum)": "sum", "Average (mean)": "mean", "Median": "median", "Count of rows": "count",
                "Smallest (min)": "min", "Largest (max)": "max", "Spread (standard deviation)": "stddev",
                "Number of different values": "countDistinct"}
        keys = list(gold.get("by") or [])
        aggs = ", ".join(f'F.{fmap[f]}("{v}").alias("{v}_{fmap[f]}")' for v in gold["values"]
                         for f in (gold.get("funcs") or []) if f in fmap)
        if keys and aggs:
            lines += ["", "# 7. Gold table (same as the app)",
                      f"gold = df_pq.groupBy({', '.join(repr(k) for k in keys)}).agg({aggs})",
                      f'gold.write.mode("overwrite").parquet("hdfs:/data/{label}_gold")']
    lines += ["", "# Write results back (Exercise 20)",
              f'df_pq.write.mode("overwrite").csv("hdfs:/data/{label}_results_csv", header=True)',
              f'# df_pq.write.jdbc("jdbc:postgresql://server:5432/mydb", table="{label}_results", mode="overwrite",',
              '#                  properties={"user": "USER", "password": "PASSWORD", "driver": "org.postgresql.Driver"})']
    lines += ["", "# SQL works too", f'df_pq.createOrReplaceTempView("{label[:30] or "data"}")',
              f'spark.sql("SELECT COUNT(*) AS n FROM {label[:30] or "data"}").show()', "", "spark.stop()", ""]
    return "\n".join(lines)
