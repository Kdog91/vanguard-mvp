"""
Creates a large made-up awards file for testing large-file mode.

    python make_big_sample.py              -> 5,000,000 rows (about 300 MB)
    python make_big_sample.py 20000000     -> 20,000,000 rows

The data is synthetic. It includes messy values on purpose: dollar signs,
missing values and duplicate rows.
"""
import sys
import numpy as np
import pandas as pd

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
out = f"big_awards_{rows}.csv"
rng = np.random.default_rng(11)
agencies = np.array(["DoD", "HHS", "GSA", "DHS", "VA"])
naics = np.array([541511, 541512, 518210, 541330, 561210])
risk = np.array(["Low", "Medium", "High"])
start = np.datetime64("2019-10-01")
days = (np.datetime64("2026-09-30") - start).astype(int)
chunk, written = 500_000, 0
while written < rows:
    n = min(chunk, rows - written)
    d = start + rng.integers(0, days, n).astype("timedelta64[D]")
    month = d.astype("datetime64[M]").astype(int) % 12 + 1
    duration = rng.integers(30, 400, n)
    materials = rng.lognormal(10.3, 0.6, n)
    season = np.where(month == 9, 1.5, 1.0)            # fiscal year-end bump
    total = (materials * 1.6 + duration * 120 + rng.normal(0, 9000, n)).clip(1000) * season
    r = np.where(total > 140_000, 2, np.where(total > 80_000, 1, 0))
    flip = rng.random(n) < 0.06
    r = np.where(flip, rng.integers(0, 3, n), r)
    df = pd.DataFrame({
        "award_id": [f"A{i:09d}" for i in range(written, written + n)],
        "award_date": d.astype(str),
        "agency": agencies[rng.choice(5, n, p=[.35, .2, .15, .15, .15])],
        "naics_code": naics[rng.integers(0, 5, n)],
        "duration_days": duration,
        "materials_cost": materials.round(2),
        "total_cost": [f"${v:,.2f}" for v in total],
        "risk_level": risk[r],
    })
    df.loc[rng.random(n) < 0.03, "materials_cost"] = np.nan
    df.loc[rng.random(n) < 0.02, "risk_level"] = None
    df = pd.concat([df, df.sample(frac=0.002, random_state=1)])   # duplicate rows
    df.to_csv(out, mode="a" if written else "w", header=not written, index=False)
    written += n
    print(f"{written:,} rows written", end="\r")
print(f"\nCreated {out}")
