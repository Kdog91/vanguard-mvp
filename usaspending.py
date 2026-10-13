"""
USAspending.gov connector.

Pulls real federal contract awards from the U.S. Treasury's public USAspending API
(no account or key needed) and returns them as an ordinary table the rest of the
app can clean and analyze.

Endpoint: POST /api/v2/search/spending_by_award/  (100 awards per request, paged)
"""
import json
import time
import urllib.error
import urllib.request

import pandas as pd

API_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
PAGE_SIZE = 100          # the API's maximum per request
MAX_AWARDS = 5000        # keep requests polite and inside the API's paging limit
EARLIEST_DATE = "2007-10-01"   # the search endpoint does not go back further

# Names must match USAspending's "top-tier agency" names exactly.
AGENCIES = [
    "Department of Defense",
    "Department of Veterans Affairs",
    "Department of Health and Human Services",
    "Department of Homeland Security",
    "General Services Administration",
    "Department of Energy",
    "National Aeronautics and Space Administration",
    "Department of Transportation",
    "Department of Agriculture",
    "Department of Justice",
    "Department of the Interior",
    "Department of Commerce",
    "Department of State",
    "Department of the Treasury",
    "Department of Labor",
    "Department of Education",
    "Environmental Protection Agency",
    "Small Business Administration",
]

# API field name -> column name used in the app
FIELDS = {
    "Award ID": "award_id",
    "Recipient Name": "recipient_name",
    "Start Date": "start_date",
    "End Date": "end_date",
    "Award Amount": "award_amount",
    "Total Outlays": "total_outlays",
    "Awarding Agency": "awarding_agency",
    "Awarding Sub Agency": "awarding_sub_agency",
    "Contract Award Type": "contract_award_type",
    "NAICS": "naics",
    "PSC": "psc",
    "Place of Performance State Code": "place_of_performance_state",
    "Description": "description",
}
CONTRACT_TYPE_CODES = ["A", "B", "C", "D"]   # purchase orders, delivery orders, definitive contracts, BPA calls
SPREAD = "Spread across the period"
SORT_CHOICES = {SPREAD: "Start Date", "Most recent first": "Start Date", "Largest first": "Award Amount"}
MAX_SLICES = 12          # "spread" mode takes an equal share of awards from up to this many time slices
NO_STATE = "Other"       # place of performance outside the 50 states, or not recorded


class USAspendingError(Exception):
    """A problem talking to USAspending.gov, with a message fit to show the user."""


def build_request(agency: str, start_date: str, end_date: str, page: int, sort_by: str,
                  min_amount: float | None = None, new_only: bool = False, limit: int = PAGE_SIZE) -> dict:
    period = {"start_date": start_date, "end_date": end_date}
    if new_only:
        # Only awards that BEGAN in the window. Without this the API also returns old awards
        # that merely had some activity (a modification, a payment) in the window.
        period["date_type"] = "new_awards_only"
    filters = {
        "award_type_codes": CONTRACT_TYPE_CODES,
        "agencies": [{"type": "awarding", "tier": "toptier", "name": agency}],
        "time_period": [period],
    }
    if min_amount:
        filters["award_amounts"] = [{"lower_bound": float(min_amount)}]
    return {"filters": filters, "fields": list(FIELDS), "page": page, "limit": int(max(1, min(limit, PAGE_SIZE))),
            "sort": SORT_CHOICES.get(sort_by, sort_by), "order": "desc", "subawards": False}


def _post(body: dict, url: str, timeout: int = 60, retries: int = 2) -> dict:
    data = json.dumps(body).encode("utf-8")
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "VanguardDataAnalytics-GovData/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            if e.code in (400, 422):      # our request was rejected: retrying will not help
                raise USAspendingError(f"USAspending rejected the request (HTTP {e.code}). {detail}") from e
            last = f"HTTP {e.code} {detail}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = str(getattr(e, "reason", e))
        except json.JSONDecodeError as e:
            last = f"unreadable reply ({e})"
        time.sleep(1.5 * (attempt + 1))
    raise USAspendingError(f"Could not reach USAspending.gov after {retries + 1} tries: {last}. "
                           "Check your internet connection and try again.")


def _code_and_text(value):
    """NAICS / PSC arrive either as plain text or as {'code': ..., 'description': ...}. Return (code, description)."""
    if isinstance(value, dict):
        return value.get("code"), value.get("description")
    return value, None


def to_dataframe(results: list) -> pd.DataFrame:
    """Turn the API's result rows into a tidy table with plain column names."""
    rows = []
    for r in results:
        row = {new: r.get(old) for old, new in FIELDS.items() if new not in ("naics", "psc")}
        row["naics_code"], row["naics_description"] = _code_and_text(r.get("NAICS"))
        row["psc_code"], row["psc_description"] = _code_and_text(r.get("PSC"))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("start_date", "end_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("award_amount", "total_outlays"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "place_of_performance_state" in df:
        df["place_of_performance_state"] = df["place_of_performance_state"].fillna(NO_STATE)
    # Planned length of the award in days: a number the models can use
    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days
    df = df.dropna(axis=1, how="all")          # e.g. descriptions, when the API sends plain codes
    return df


def check_dates(df: pd.DataFrame, today=None) -> tuple[pd.DataFrame, dict]:
    """
    Real award records contain date mistakes. Count them, and blank out any duration that is
    impossible (end before start) so it is not treated as a real number.
    """
    counts = {"end_before_start": 0, "future_start": 0, "same_day": 0}
    if df.empty or "duration_days" not in df:
        return df, counts
    df = df.copy()
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    bad = df["duration_days"] < 0
    counts["end_before_start"] = int(bad.sum())
    counts["future_start"] = int((df["start_date"] > today).sum())
    counts["same_day"] = int((df["duration_days"] == 0).sum())
    df["duration_days"] = df["duration_days"].astype("float64")
    df.loc[bad, "duration_days"] = float("nan")
    return df, counts


def _slices(start_date: str, end_date: str, n: int) -> list:
    """Split a date range into up to n back-to-back slices of (start, end) text dates."""
    a, b = pd.Timestamp(start_date), pd.Timestamp(end_date)
    days = (b - a).days + 1
    n = max(1, min(n, days))
    edges = [a + pd.Timedelta(days=round(days * i / n)) for i in range(n + 1)]
    return [(edges[i].strftime("%Y-%m-%d"), (edges[i + 1] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")) for i in range(n)]


def fetch_awards(agency: str, start_date: str, end_date: str, max_awards: int = 1000,
                 sort_by: str = SPREAD, min_amount: float | None = None,
                 progress=None, url: str = API_URL, new_only: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Download up to `max_awards` contract awards. Returns (table, info).
    `progress(done, total)` is called after each request so the app can show a progress bar.

    sort_by = SPREAD takes an equal share from up to 12 slices of the period, so the awards cover
    the whole period instead of bunching at one end. new_only keeps awards that began in the period.
    """
    if agency not in AGENCIES:
        raise USAspendingError(f"Unknown agency: {agency}")
    if str(start_date) < EARLIEST_DATE:
        raise USAspendingError(f"USAspending's search only goes back to {EARLIEST_DATE}.")
    if str(end_date) < str(start_date):
        raise USAspendingError("The end date is before the start date.")
    max_awards = int(max(1, min(max_awards, MAX_AWARDS)))

    t0 = time.time()
    state = {"new_only": bool(new_only), "requests": 0}
    results, messages, notes = [], [], []

    def ask(s, e, page, limit):
        """One request. If the 'began in the period' filter is refused, carry on without it and say so."""
        state["requests"] += 1
        try:
            return _post(build_request(agency, s, e, page, sort_by, min_amount, state["new_only"], limit), url)
        except USAspendingError as err:
            if not state["new_only"] or "rejected the request" not in str(err):
                raise
            state["new_only"] = False
            notes.append("USAspending did not accept the 'new awards only' filter, so awards that were "
                         "only active in the period are included.")
            state["requests"] += 1
            return _post(build_request(agency, s, e, page, sort_by, min_amount, False, limit), url)

    def pull(s, e, want):
        """Page through one date range until `want` awards are collected. Returns (rows, more_exist)."""
        got, page, has_next = [], 1, True
        while has_next and len(got) < want:
            reply = ask(s, e, page, min(PAGE_SIZE, want - len(got)) if page == 1 and want <= PAGE_SIZE else PAGE_SIZE)
            batch = reply.get("results") or []
            got.extend(batch)
            messages.extend(reply.get("messages") or [])
            has_next = bool((reply.get("page_metadata") or {}).get("hasNext")) and bool(batch)
            if progress:
                progress(min(len(results) + len(got), max_awards), max_awards)
            page += 1
            time.sleep(0.2)     # be polite to a free public service
        return got[:want], has_next or len(got) > want

    more = False
    if sort_by == SPREAD:
        parts = _slices(str(start_date), str(end_date), min(MAX_SLICES, max_awards))
        for k, (s, e) in enumerate(parts):
            want = -(-(max_awards - len(results)) // (len(parts) - k))     # share what is left evenly
            if want <= 0:
                break
            got, m = pull(s, e, want)
            results.extend(got)
            more = more or m
        n_slices = len(parts)
    else:
        results, more = pull(str(start_date), str(end_date), max_awards)
        n_slices = 1

    df = to_dataframe(results[:max_awards])
    if not df.empty:
        df = df.drop_duplicates(subset="award_id").reset_index(drop=True) if "award_id" in df else df
    df, date_issues = check_dates(df)
    info = {"agency": agency, "start_date": str(start_date), "end_date": str(end_date), "rows": len(df),
            "requests": state["requests"], "seconds": time.time() - t0, "more_available": more,
            "sort_by": sort_by, "new_only": state["new_only"], "slices": n_slices, "date_issues": date_issues,
            "messages": list(dict.fromkeys(notes + messages))}
    return df, info
