"""
Multi-source standardization: make measurements from different sites, labs or methods
comparable, check their quality, and combine them into one index.

Method (the same one used in the CDC wastewater work sample):
  1. Each series (a source, optionally split by method or lab) is treated on its own.
  2. Values can be log10-transformed, because sources often differ by orders of magnitude.
  3. Each series is converted to robust z-scores: (value - series median) / (1.4826 x MAD).
     Median and MAD (median absolute deviation) are not pulled around by a few extreme readings.
  4. Each reading is compared with the other series in the same period (z minus the period median).
     A real regional rise moves every series together; a bad reading or a lab change moves one alone.
     Quality flags: readings far from that comparison (outliers), reporting gaps longer than N days,
     and step changes (a lasting jump against the other series, often a lab or method switch).
  5. The index for each period is the weighted mean (or median) of the series' period averages.
Flagged readings are kept in the output with a flag; they are left out of the index only.
"""
import numpy as np
import pandas as pd

MAD_SCALE = 1.4826   # makes MAD comparable to a standard deviation for normal data


def series_key(df: pd.DataFrame, cols: list) -> pd.Series:
    return df[cols].astype(str).agg(" | ".join, axis=1) if len(cols) > 1 else df[cols[0]].astype(str)


def _step_change(dates: pd.Series, z: pd.Series, window: int, min_shift: float):
    """Largest shift in median level between the `window` readings before and after each point."""
    if len(z) < 2 * window + 2:
        return None, 0.0
    vals = z.to_numpy()
    best_i, best = None, 0.0
    for i in range(window, len(vals) - window + 1):
        before, after = vals[i - window:i], vals[i:i + window]
        if np.isnan(before).all() or np.isnan(after).all():
            continue
        shift = np.nanmedian(after) - np.nanmedian(before)
        if abs(shift) > abs(best):
            best_i, best = i, shift
    if best_i is None or abs(best) < min_shift:
        return None, 0.0
    return dates.iloc[best_i], float(best)


def standardize(df: pd.DataFrame, source_col: str, date_col: str, value_col: str,
                split_cols: list | None = None, weight_col: str | None = None,
                log: bool = True, z_flag: float = 4.5, gap_days: int = 21,
                step_window: int = 6, step_min: float = 3.5, period: str = "W",
                combine: str = "Weighted mean") -> dict:
    split_cols = [c for c in (split_cols or []) if c not in (source_col,)]
    work = df.copy()
    work["_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work["_value"] = pd.to_numeric(work[value_col], errors="coerce")
    n_in = len(work)
    work = work.dropna(subset=["_date", "_value"])
    dropped_missing = n_in - len(work)

    dropped_nonpos = 0
    if log:
        bad = work["_value"] <= 0
        dropped_nonpos = int(bad.sum())
        work = work[~bad]
        work["_x"] = np.log10(work["_value"])
    else:
        work["_x"] = work["_value"]

    work["_series"] = series_key(work, [source_col] + split_cols)
    work = work.sort_values(["_series", "_date"])

    grp = work.groupby("_series")["_x"]
    med = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median()) * MAD_SCALE
    work["robust_z"] = (work["_x"] - med) / mad.replace(0, np.nan)
    # Compare each reading with what the other series did in the same period. A real regional rise
    # moves every series together; a bad reading or a lab change moves one series alone.
    work["_period"] = work["_date"].dt.to_period(period).dt.start_time
    n_series = work["_series"].nunique()
    if n_series >= 3:
        common = work.groupby("_period")["robust_z"].transform("median")
        work["vs_others"] = work["robust_z"] - common
    else:
        work["vs_others"] = work["robust_z"]
    # Noise scale from reading-to-reading changes, so a lasting step does not inflate it
    rmad = work.groupby("_series")["vs_others"].transform(
        lambda s: s.diff().abs().median() * MAD_SCALE / np.sqrt(2))
    # Outliers: distance from the series' local level (rolling median of nearby readings)
    local = work.groupby("_series")["vs_others"].transform(
        lambda s: s.rolling(7, center=True, min_periods=3).median())
    work["vs_others_z"] = (work["vs_others"] - local) / rmad.replace(0, np.nan)
    work["_level_z"] = work["vs_others"] / rmad.replace(0, np.nan)
    work["flag_outlier"] = work["vs_others_z"].abs() > z_flag
    gap = work.groupby("_series")["_date"].diff().dt.days
    work["days_since_previous"] = gap
    work["flag_gap_before"] = gap > gap_days

    # ---- quality table, one row per series ----
    qa_rows = []
    for key, g in work.groupby("_series", sort=True):
        cp_date, cp_shift = _step_change(g["_date"], g["_level_z"], step_window, step_min)
        gaps = g["days_since_previous"]
        row = {
            "Series": key,
            "Readings": len(g),
            "First": str(g["_date"].min().date()),
            "Last": str(g["_date"].max().date()),
            "Median (original units)": float(np.median(g["_value"])),
            "Outliers flagged": int(g["flag_outlier"].sum()),
            f"Gaps over {gap_days} days": int((gaps > gap_days).sum()),
            "Longest gap (days)": int(gaps.max()) if gaps.notna().any() else 0,
            "Possible step change": str(cp_date.date()) if cp_date is not None else "",
            "Step size (z)": round(cp_shift, 2) if cp_date is not None else np.nan,
            "Note": "No spread: all values equal, z not computed" if g["robust_z"].isna().all() and len(g) > 1 else "",
        }
        if weight_col:
            row["Weight"] = pd.to_numeric(g[weight_col], errors="coerce").dropna().iloc[0] \
                if pd.to_numeric(g[weight_col], errors="coerce").notna().any() else np.nan
        qa_rows.append(row)
    qa = pd.DataFrame(qa_rows)

    # ---- combined index ----
    use = work[~work["flag_outlier"] & work["robust_z"].notna()].copy()
    per = use.groupby(["_period", "_series"], as_index=False)["robust_z"].mean()
    if weight_col and combine == "Weighted mean" and "Weight" in qa:
        wmap = qa.set_index("Series")["Weight"]
        per["_w"] = per["_series"].map(wmap).fillna(0)
        per["_wz"] = per["robust_z"] * per["_w"]
        agg = per.groupby("_period").agg(_wz=("_wz", "sum"), _w=("_w", "sum"), sources=("_series", "nunique"),
                                         median_z=("robust_z", "median"))
        agg["index"] = np.where(agg["_w"] > 0, agg["_wz"] / agg["_w"].replace(0, np.nan), agg["median_z"])
        method_used = f"weighted mean (weights from '{weight_col}')"
    elif combine == "Median":
        agg = per.groupby("_period").agg(index=("robust_z", "median"), sources=("_series", "nunique"))
        method_used = "median across series"
    else:
        agg = per.groupby("_period").agg(index=("robust_z", "mean"), sources=("_series", "nunique"))
        method_used = "mean across series"
    index = agg.reset_index().rename(columns={"_period": "period"})[["period", "index", "sources"]]
    index["index"] = index["index"].round(3)

    out = work.drop(columns=["_x", "_period", "_level_z"]).rename(columns={"_series": "series", "_date": "date_parsed",
                                                    "_value": "value_numeric"})
    for c in ("robust_z", "vs_others", "vs_others_z"):
        out[c] = out[c].round(3)
    summary = {
        "rows_in": n_in, "rows_used": len(work), "dropped_missing": dropped_missing,
        "dropped_nonpositive": dropped_nonpos, "series": work["_series"].nunique(),
        "sources": work[source_col].nunique(), "outliers": int(work["flag_outlier"].sum()),
        "gaps": int(work["flag_gap_before"].sum()), "step_changes": int((qa["Possible step change"] != "").sum()),
        "orders_of_magnitude": float(np.log10(work["_value"][work["_value"] > 0].max() /
                                              work["_value"][work["_value"] > 0].min()))
        if (work["_value"] > 0).sum() > 1 else 0.0,
        "index_method": method_used,
    }
    return {"rows": out, "qa": qa, "index": index, "summary": summary}
