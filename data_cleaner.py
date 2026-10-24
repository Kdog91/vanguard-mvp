"""
Universal Data Cleaner
Handles automatic type detection, missing value handling, outlier detection,
and duplicate removal for any uploaded tabular dataset.
"""
import os
import re
import pandas as pd
import numpy as np


# Text that means "no value" in many public datasets (for example '?' in the UCI imports-85 file)
NA_MARKERS = ["?", "-", "--", "n/a", "N/A", "NA", "na", "null", "NULL", "None", "none", "missing", "Missing", "#N/A", "."]


def _looks_like_value(name: str) -> bool:
    s = str(name).strip()
    if s in NA_MARKERS or s == "" or s.lower().startswith("unnamed:"):
        return True
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def header_looks_like_data(columns) -> bool:
    """True when the 'column names' are really a row of data: many of them are numbers or missing markers."""
    cols = [str(c) for c in columns]
    if len(cols) < 2:
        return False
    hits = sum(_looks_like_value(c) for c in cols)
    return hits >= 2 and hits / len(cols) >= 0.4


def numbered_columns(n: int) -> list:
    return [f"column_{i + 1}" for i in range(n)]


def _read_text_table(raw: bytes) -> pd.DataFrame:
    """Delimited text (comma, tab, semicolon, pipe). The separator, the text encoding and whether the
    first row holds column names are detected."""
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
    df = pd.read_csv(io.StringIO(text), sep=sep, na_values=NA_MARKERS, keep_default_na=True)
    notes = []
    if header_looks_like_data(df.columns):
        df = pd.read_csv(io.StringIO(text), sep=sep, header=None, na_values=NA_MARKERS, keep_default_na=True)
        df.columns = numbered_columns(df.shape[1])
        notes.append("The first row looked like data, not column names, so it was kept as a row and the columns "
                     "were named column_1, column_2, ... You can type real names under 'Fix column names'.")
    df.attrs["import_notes"] = notes
    df.attrs["header_from_file"] = not notes
    return df


def set_header(df: pd.DataFrame, first_row_is_data: bool) -> pd.DataFrame:
    """Switch between 'the first row is column names' and 'the first row is data'."""
    if first_row_is_data:
        top = pd.DataFrame([list(df.columns)], columns=df.columns)
        top = top.replace({c: None for c in NA_MARKERS})
        out = pd.concat([top, df], ignore_index=True)
        out.columns = numbered_columns(out.shape[1])
    else:
        out = df.iloc[1:].reset_index(drop=True)
        out.columns = [str(v) if pd.notna(v) else f"column_{i + 1}" for i, v in enumerate(df.iloc[0])]
    # Values that moved between the header and the body are text; turn columns that are all numbers back into numbers
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            conv = pd.to_numeric(out[c], errors="coerce")
            if conv.notna().sum() == out[c].notna().sum():
                out[c] = conv
    return out


def rename_columns(df: pd.DataFrame, names_text: str):
    """Rename columns from a comma-separated list. Returns (df, message)."""
    names = [n.strip() for n in names_text.replace("\n", ",").split(",") if n.strip()]
    if not names:
        return df, ""
    if len(names) != df.shape[1]:
        return df, f"You typed {len(names)} names but the data has {df.shape[1]} columns, so nothing was renamed."
    if len(set(names)) != len(names):
        return df, "Two of the names are the same; each column needs its own name."
    out = df.copy()
    out.columns = names
    return out, f"Renamed all {len(names)} columns."


class _NamedBytes:
    """A small stand-in for an uploaded file (name + read), so web downloads use the same loader."""
    def __init__(self, name: str, data: bytes):
        import io
        self.name, self._buf = name, io.BytesIO(data)

    def read(self, *a):
        return self._buf.read(*a)

    def seek(self, *a):
        return self._buf.seek(*a)

    def tell(self):
        return self._buf.tell()


def load_url(url: str, max_mb: int = 200) -> pd.DataFrame:
    """Read a data file from a web address, like pd.read_csv(url). GitHub page links are turned into raw-file links."""
    import urllib.request
    from urllib.parse import urlparse
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("The address must start with http:// or https://")
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if m:
        url = f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    req = urllib.request.Request(url, headers={"User-Agent": "VanguardAnalytics/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read(max_mb * 1024 * 1024 + 1)
        ctype = (resp.headers.get("Content-Type") or "").lower()
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(f"That file is larger than {max_mb} MB.")
    name = os.path.basename(urlparse(url).path) or "download"
    if not name.lower().endswith((".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson", ".parquet", ".data")):
        if "json" in ctype:
            name += ".json"
        elif "spreadsheet" in ctype or "excel" in ctype:
            name += ".xlsx"
        elif "html" in ctype:
            raise ValueError("That address is a web page, not a data file. Use the link to the file itself "
                             "(on GitHub, open the file and click 'Raw').")
        else:
            name += ".csv"
    if name.lower().endswith(".data"):          # UCI-style comma-separated files
        name += ".csv"
    return load_file(_NamedBytes(name, data))


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
        import io
        raw = uploaded_file.read()
        df = pd.read_excel(io.BytesIO(raw), na_values=NA_MARKERS)
        if header_looks_like_data(df.columns):
            df = pd.read_excel(io.BytesIO(raw), header=None, na_values=NA_MARKERS)
            df.columns = numbered_columns(df.shape[1])
            df.attrs["import_notes"] = ["The first row looked like data, not column names, so it was kept as a "
                                        "row and the columns were named column_1, column_2, ..."]
            df.attrs["header_from_file"] = False
    elif name.endswith((".json", ".jsonl", ".ndjson")):
        df = _read_json(uploaded_file.read())
    elif name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    elif name.endswith(".orc"):
        import io
        df = pd.read_orc(io.BytesIO(uploaded_file.read()))
    else:
        raise ValueError("Unsupported file type. Please upload CSV, TSV, TXT, Excel, JSON, Parquet or ORC.")
    if df.empty or df.shape[1] == 0:
        raise ValueError("The file was read but contains no rows of data.")
    attrs = dict(df.attrs)
    df.columns = [str(c).strip() for c in df.columns]
    # Cells that only say "?", "N/A", "null" and so on mean "no value" (Excel and JSON files too)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            stripped = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = stripped.where(~stripped.isin(NA_MARKERS + [""]), None)
    df.attrs.update(attrs)
    # Lists / objects left inside cells (from nested JSON) become text so they can be counted and compared
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].map(lambda v: str(v) if isinstance(v, (list, dict)) else v)
    return df


NEAR_EMPTY = 0.80    # columns at least this empty are set aside instead of filled


def clean_currency_and_numbers(series: pd.Series) -> pd.Series:
    """Strip $ , % and convert to numeric where possible."""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        cleaned = series.astype(str).str.replace(r"[\$,%]", "", regex=True).str.strip()
        converted = pd.to_numeric(cleaned, errors="coerce")
        # Only replace if a meaningful fraction of values actually converted
        if converted.notna().mean() > 0.5:
            return converted
    return series


_CODE_WORDS = ("code", "naics", "zip", "psc", "fips", "sic")


def _is_code_name(col) -> bool:
    """True when a column name says it holds codes (NAICS, ZIP, PSC, FIPS...), which are labels, not amounts."""
    name = str(col).lower()
    return name.endswith("code") or any(w in re.split(r"[^a-z]+", name) for w in _CODE_WORDS)

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
        code_hint = _is_code_name(col)
        if is_numeric_col and code_hint and (series.dropna() % 1 == 0).all():
            profile[col] = "categorical"
            continue
        if (not is_numeric_col) and code_hint and uniqueness_ratio < 0.5:
            profile[col] = "categorical"
            continue

        if pd.api.types.is_numeric_dtype(series):
            profile[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile[col] = "datetime"
        else:
            # Try datetime parse only if column name/content hints at a date
            # (avoids false-positive datetime parsing of generic ID/text columns)
            looks_date_like = any(k in col.lower() for k in ("date", "time", "_dt", "timestamp", "week", "month", "period", "day"))
            if looks_date_like:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(series, errors="coerce")
                    # Real dates contain digits and a plausible year; month names alone ("may", "jul")
                    # are parsed by pandas as year 1, so they stay categories.
                    has_digits = series.dropna().astype(str).str.contains(r"\d").mean() > 0.8
                    years = parsed.dropna().dt.year
                    plausible = len(years) and years.between(1800, 2200).mean() > 0.95
                    if parsed.notna().mean() > 0.8 and has_digits and plausible:
                        profile[col] = "datetime"
                        continue
                except Exception:
                    pass
            nunique = series.nunique(dropna=True)
            # Short codes (state abbreviations, grades, flags) are categories even when there are
            # many of them, as long as they repeat.
            short_code = (nunique <= 80 and uniqueness_ratio < 0.5
                          and series.dropna().astype(str).str.len().max() <= 6)
            if nunique <= max(20, int(0.05 * len(series))) or short_code:
                profile[col] = "categorical"
            else:
                profile[col] = "text"
    return profile


MISSING_STRATEGIES = [
    "Automatic (median for numbers, most common value for categories)",
    "Mean for numbers, most common value for categories",
    "Forward fill (copy the previous row, in date order)",
    "Fill with 0 for numbers and 'Unknown' for categories",
    "Drop rows that have any gap",
]
TYPE_CHOICES = ["numeric", "categorical", "datetime", "text", "identifier (removed)", "drop column"]

_YES = {"yes", "y", "true", "t"}
_NO = {"no", "n", "false", "f"}


def _yes_no(series: pd.Series):
    """If a text column only holds yes/no style values (Yes, N, TRUE, f...), return it as Yes/No."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        if pd.api.types.is_bool_dtype(series):
            return series.map({True: "Yes", False: "No"})
        return None
    vals = series.dropna().astype(str).str.strip().str.lower()
    if vals.empty or not set(vals.unique()) <= (_YES | _NO) or vals.nunique() < 2:
        return None
    if set(series.dropna().astype(str).str.strip().unique()) <= {"Yes", "No"}:
        return None                                 # already tidy
    out = vals.map(lambda v: "Yes" if v in _YES else "No")
    return out.reindex(series.index)


def _apply_override(df: pd.DataFrame, col: str, kind: str):
    """Convert one column to the type the user picked. Returns (series or None to drop, note)."""
    s = df[col]
    if kind == "numeric":
        conv = pd.to_numeric(clean_currency_and_numbers(s.astype(str).where(s.notna(), None))
                             if not pd.api.types.is_numeric_dtype(s) else s, errors="coerce")
        lost = int(conv.isna().sum() - s.isna().sum())
        return conv, (f"Changed '{col}' to numbers (to_numeric)" +
                      (f"; {lost} values were not numbers and became gaps." if lost > 0 else "."))
    if kind == "datetime":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conv = pd.to_datetime(s, errors="coerce")
        lost = int(conv.isna().sum() - s.isna().sum())
        return conv, (f"Changed '{col}' to dates (to_datetime)" +
                      (f"; {lost} values were not dates and became gaps." if lost > 0 else "."))
    if kind in ("categorical", "text"):
        conv = s.map(lambda v: (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)) if pd.notna(v) else None)
        label = "categories (astype('category'))" if kind == "categorical" else "free text"
        return conv, f"Changed '{col}' to {label}."
    return None, f"Removed column '{col}' at your request."


def clean_dataframe(df: pd.DataFrame, missing: str = MISSING_STRATEGIES[0], overrides: dict | None = None):
    """
    Full cleaning pipeline:
    - strip currency/number formatting
    - standardize yes/no columns
    - convert dates
    - remove exact duplicates
    - apply any column types the user chose
    - handle missing values with the chosen strategy
    - flag outliers (does not remove, just flags)
    Returns cleaned df, profile dict, a report of what was done, outlier counts and removed identifier columns.
    """
    report = []
    overrides = {k: v for k, v in (overrides or {}).items() if k in df.columns}
    df = df.copy()

    # Drop fully empty rows/columns
    before_rows = len(df)
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    if len(df) < before_rows:
        report.append(f"Removed {before_rows - len(df)} completely empty rows.")

    # Attempt currency/number cleaning on object columns
    for col in df.columns:
        if col in overrides:
            continue                              # the user decided this column's type
        if _is_code_name(col) and not pd.api.types.is_numeric_dtype(df[col]):
            continue                              # NAICS, ZIP, PSC... are labels: keep the text (and leading zeros)
        original_dtype = df[col].dtype
        df[col] = clean_currency_and_numbers(df[col])
        if df[col].dtype != original_dtype:
            report.append(f"Converted column '{col}' from text to numeric (stripped $ , % symbols).")

    # Yes/No columns written different ways (Y, yes, TRUE, f...) become one tidy Yes/No category
    for col in df.columns:
        if col in overrides:
            continue
        yn = _yes_no(df[col])
        if yn is not None:
            df[col] = yn
            report.append(f"Recognized '{col}' as a yes/no column and wrote every value as Yes or No.")

    # Remove duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        report.append(f"Removed {dup_count} duplicate rows.")

    # Profile columns, then apply the types the user picked
    profile = auto_profile_columns(df)
    for col, kind in overrides.items():
        if col not in profile or kind == profile[col]:
            continue
        if kind == "identifier (removed)":
            profile[col] = "identifier"
            continue
        series, note = _apply_override(df, col, kind)
        if series is not None and df[col].notna().any() and series.notna().mean() < 1 - NEAR_EMPTY:
            report.append(f"Kept '{col}' as {profile[col]}: as {kind}, {series.isna().mean():.0%} of its values "
                          "would be gaps, so the change was not applied.")
            continue
        report.append(note)
        if series is None:
            df = df.drop(columns=[col])
            del profile[col]
        else:
            df[col] = series
            profile[col] = kind

    # Identifier columns: report and drop from the cleaned dataset entirely.
    # These are things like contract_id, record_id, uuid -- useful for humans
    # to reference a row, but never useful as a modeling feature.
    id_cols = [c for c, kind in profile.items() if kind == "identifier"]
    dropped_identifiers = {}
    if id_cols:
        for c in id_cols:
            dropped_identifiers[c] = df[c].copy()  # keep a reference in case caller wants it
        df = df.drop(columns=id_cols)
        for c in id_cols:
            how = "you marked it as an identifier" if overrides.get(c) == "identifier (removed)" else "nearly all values unique"
            report.append(f"Removed identifier column '{c}' ({how}; not useful for modeling).")
        profile = {k: v for k, v in profile.items() if k not in id_cols}

    # Set aside columns that are almost entirely empty. Filling them would invent most of the
    # column, and anything computed from it would describe the filler, not the data.
    for col in list(profile):
        pct_empty = df[col].isna().mean() if len(df) else 0
        if pct_empty >= NEAR_EMPTY and col not in overrides:
            df = df.drop(columns=[col])
            del profile[col]
            report.append(f"Set aside column '{col}': {pct_empty:.0%} empty, too little data to analyze. "
                          "It is still in the raw data.")

    # Parse date columns into real dates (so they can be charted and forecast)
    for col, kind in profile.items():
        if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_datetime(df[col], errors="coerce")
            report.append(f"Parsed column '{col}' as dates.")

    # ---- Missing values, using the chosen strategy ----
    gaps_before = {c: int(df[c].isna().sum()) for c in profile}
    if missing.startswith("Drop"):
        cols = [c for c, k in profile.items() if k in ("numeric", "categorical", "datetime")]
        before = len(df)
        df = df.dropna(subset=cols)
        if before - len(df):
            report.append(f"Dropped {before - len(df)} rows that had at least one gap ({(before - len(df)) / before:.0%} "
                          "of rows). Check that this does not remove one group more than others.")
        if len(df) == 0:
            raise ValueError("Every row has at least one gap, so dropping rows would leave nothing. "
                             "Pick a different way to handle missing values.")
    elif missing.startswith("Forward"):
        date_cols = [c for c, k in profile.items() if k == "datetime"]
        if date_cols:
            df = df.sort_values(date_cols[0], kind="stable")
        for col, kind in profile.items():
            n = gaps_before[col]
            if n and kind in ("numeric", "categorical"):
                df[col] = df[col].ffill().bfill()
                report.append(f"Filled {n} missing values in '{col}' by copying the previous row"
                              + (f" (rows in '{date_cols[0]}' order)." if date_cols else " (file order)."))

    for col, kind in profile.items():
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        pct_missing = gaps_before[col] / len(df) if len(df) else 0
        if kind == "numeric":
            if missing.startswith("Mean"):
                val, how = df[col].mean(), "mean"
            elif missing.startswith("Fill with 0"):
                val, how = 0.0, "0"
            else:
                val, how = df[col].median(), "median"
            df[col] = df[col].fillna(val)
            report.append(f"Filled {n_missing} missing values in '{col}' with {how} ({val:,.2f}).")
        elif kind == "categorical":
            if missing.startswith("Fill with 0"):
                fill_val = "Unknown"
            else:
                mode_val = df[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            report.append(f"Filled {n_missing} missing values in '{col}' with "
                          + ("'Unknown'." if fill_val == "Unknown" else f"most common value ('{fill_val}')."))
        elif kind == "datetime":
            report.append(f"Left {n_missing} missing dates in '{col}' as-is.")
            continue
        else:  # text
            df[col] = df[col].fillna("")
            report.append(f"Filled {n_missing} missing text values in '{col}' with empty string.")
            continue
        if pct_missing >= 0.3:
            report.append(f"Caution: '{col}' was {pct_missing:.0%} empty before filling. Treat results that "
                          "rely on it with care, or leave it out of models.")

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


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Gaps per column before cleaning (after '?', 'N/A' and similar are recognized as gaps)."""
    n = len(df)
    t = pd.DataFrame({"Column": df.columns,
                      "Missing": [int(df[c].isna().sum()) for c in df.columns]})
    t["Missing %"] = (t["Missing"] / n * 100).round(1) if n else 0.0
    return t[t["Missing"] > 0].sort_values("Missing", ascending=False).reset_index(drop=True)
