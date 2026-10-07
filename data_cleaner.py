"""
Universal Data Cleaner
Handles automatic type detection, missing value handling, outlier detection,
and duplicate removal for any uploaded tabular dataset.
"""
import pandas as pd
import numpy as np


def _read_text_table(raw: bytes) -> pd.DataFrame:
    """Delimited text (comma, tab, semicolon, pipe). The separator and the text encoding are detected."""
    import io
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    # The separator is whichever candidate appears most in the header line
    header = next((ln for ln in text.splitlines() if ln.strip()), "")
    sep = max([",", "\t", ";", "|"], key=header.count)
    return pd.read_csv(io.StringIO(text), sep=sep)


def _read_json(raw: bytes) -> pd.DataFrame:
    """JSON in the usual shapes: a list of records, records nested under a key, or one record per line."""
    import json
    text = raw.decode("utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]  # JSON Lines
    if isinstance(data, dict):
        # e.g. {"results": [ {...}, {...} ]}: use the longest list of records inside
        lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if lists:
            data = max(lists, key=len)
        elif all(isinstance(v, (list, dict)) for v in data.values()):
            return pd.DataFrame(data)  # column-oriented JSON
        else:
            data = [data]
    return pd.json_normalize(data, sep=".")  # nested objects become columns like "vendor.name"


def load_file(uploaded_file):
    """Load CSV / TSV / TXT, Excel, JSON / JSON Lines, or Parquet into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith((".csv", ".tsv", ".txt")):
        df = _read_text_table(uploaded_file.read())
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif name.endswith((".json", ".jsonl", ".ndjson")):
        df = _read_json(uploaded_file.read())
    elif name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload CSV, TSV, TXT, Excel, JSON, or Parquet.")
    if df.empty or df.shape[1] == 0:
        raise ValueError("The file was read but contains no rows of data.")
    df.columns = [str(c).strip() for c in df.columns]
    # Lists / objects left inside cells (from nested JSON) become text so they can be counted and compared
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].map(lambda v: str(v) if isinstance(v, (list, dict)) else v)
    return df


def clean_currency_and_numbers(series: pd.Series) -> pd.Series:
    """Strip $ , % and convert to numeric where possible."""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        cleaned = series.astype(str).str.replace(r"[\$,%]", "", regex=True).str.strip()
        converted = pd.to_numeric(cleaned, errors="coerce")
        # Only replace if a meaningful fraction of values actually converted
        if converted.notna().mean() > 0.5:
            return converted
    return series


def auto_profile_columns(df: pd.DataFrame) -> dict:
    """Classify each column as numeric, categorical, datetime, identifier, or text."""
    profile = {}
    n_rows = len(df)
    for col in df.columns:
        series = df[col]

        # --- Identifier detection (runs first, before numeric/categorical checks) ---
        # A column is treated as an identifier if it's (a) named like one AND highly
        # unique, or (b) non-numeric with near-total uniqueness (e.g. a UUID/code
        # column). We deliberately do NOT apply the pure-uniqueness fallback to
        # numeric columns, since continuous measurements (cost, duration, etc.)
        # are expected to have high uniqueness too and are NOT identifiers.
        name_hints_id = any(k in col.lower() for k in ("_id", "id_", "contract_id", "record_id", "uuid", "guid")) or col.lower() == "id"
        nunique = series.nunique(dropna=True)
        uniqueness_ratio = nunique / n_rows if n_rows else 0
        is_numeric_col = pd.api.types.is_numeric_dtype(series)

        if name_hints_id and uniqueness_ratio > 0.9:
            profile[col] = "identifier"
            continue
        if (not is_numeric_col) and uniqueness_ratio > 0.98 and n_rows > 20:
            # Extremely high cardinality, non-numeric column (e.g. a hidden text ID)
            profile[col] = "identifier"
            continue

        # Numeric-looking CODE columns (NAICS, ZIP, PSC, FIPS...) are labels, not quantities --
        # doing math on them (averaging, correlating) is meaningless, so treat them as categories.
        code_hint = any(k in col.lower() for k in ("code", "naics", "zip", "psc", "fips", "sic"))
        if is_numeric_col and code_hint and (series.dropna() % 1 == 0).all():
            profile[col] = "categorical"
            continue

        if pd.api.types.is_numeric_dtype(series):
            profile[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile[col] = "datetime"
        else:
            # Try datetime parse only if column name/content hints at a date
            # (avoids false-positive datetime parsing of generic ID/text columns)
            looks_date_like = any(k in col.lower() for k in ("date", "time", "_dt", "timestamp"))
            if looks_date_like:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(series, errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        profile[col] = "datetime"
                        continue
                except Exception:
                    pass
            nunique = series.nunique(dropna=True)
            if nunique <= max(20, int(0.05 * len(series))):
                profile[col] = "categorical"
            else:
                profile[col] = "text"
    return profile


def clean_dataframe(df: pd.DataFrame):
    """
    Full cleaning pipeline:
    - strip currency/number formatting
    - convert dates
    - remove exact duplicates
    - handle missing values
    - flag outliers (does not remove, just flags)
    Returns cleaned df, profile dict, and a report of what was done.
    """
    report = []
    df = df.copy()

    # Drop fully empty rows/columns
    before_rows = len(df)
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    if len(df) < before_rows:
        report.append(f"Removed {before_rows - len(df)} completely empty rows.")

    # Attempt currency/number cleaning on object columns
    for col in df.columns:
        original_dtype = df[col].dtype
        df[col] = clean_currency_and_numbers(df[col])
        if df[col].dtype != original_dtype:
            report.append(f"Converted column '{col}' from text to numeric (stripped $ , % symbols).")

    # Remove duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        report.append(f"Removed {dup_count} duplicate rows.")

    # Profile columns
    profile = auto_profile_columns(df)

    # Identifier columns: report and drop from the cleaned dataset entirely.
    # These are things like contract_id, record_id, uuid -- useful for humans
    # to reference a row, but never useful as a modeling feature. Dropping them
    # here (rather than silently excluding them later at modeling time) means
    # the cleaning step itself demonstrably removes irrelevant columns.
    id_cols = [c for c, kind in profile.items() if kind == "identifier"]
    dropped_identifiers = {}
    if id_cols:
        for c in id_cols:
            dropped_identifiers[c] = df[c].copy()  # keep a reference in case caller wants it
        df = df.drop(columns=id_cols)
        for c in id_cols:
            report.append(f"Detected and removed identifier column '{c}' (nearly all values unique — not useful for modeling).")
        # Re-profile after dropping, so downstream code doesn't reference stale columns
        profile = {k: v for k, v in profile.items() if k not in id_cols}

    # Parse date columns into real dates (so they can be charted and forecast)
    for col, kind in profile.items():
        if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            report.append(f"Parsed column '{col}' as dates.")

    # Handle missing values per column type
    for col, kind in profile.items():
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        pct_missing = n_missing / len(df)
        if kind == "numeric":
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report.append(f"Filled {n_missing} missing values in '{col}' with median ({median_val:.2f}).")
            if pct_missing >= 0.3:
                report.append(f"Caution: '{col}' was {pct_missing:.0%} empty before filling. Treat results that "
                              "rely on it with care, or leave it out of models.")
        elif kind == "categorical":
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            report.append(f"Filled {n_missing} missing values in '{col}' with most common value ('{fill_val}').")
            if pct_missing >= 0.3:
                report.append(f"Caution: '{col}' was {pct_missing:.0%} empty before filling. Treat results that "
                              "rely on it with care, or leave it out of models.")
        elif kind == "datetime":
            report.append(f"Left {n_missing} missing dates in '{col}' as-is.")
        else:  # text
            df[col] = df[col].fillna("")
            report.append(f"Filled {n_missing} missing text values in '{col}' with empty string.")

    # Outlier flagging (IQR method) for numeric columns
    outlier_summary = {}
    for col, kind in profile.items():
        if kind != "numeric":
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            outlier_summary[col] = n_outliers
            report.append(f"Flagged {n_outliers} statistical outliers in '{col}' (IQR method) — not removed, review recommended.")

    return df, profile, report, outlier_summary, dropped_identifiers
