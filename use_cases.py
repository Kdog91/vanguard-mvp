"""
"What can this data answer?" Looks at the columns after cleaning and lists the business questions
the app can test with them, which mode to use, and what is missing for the ones it cannot.
Based on common data-science business problems: demand forecasts, churn, fraud and pricing,
scoring, cross-sell, machine failures, and sentiment in text.
"""
import pandas as pd

import basket
import text_analysis


def _yes_no_cols(df, profile):
    out = []
    for c, t in profile.items():
        if t == "categorical" and c in df.columns:
            vals = set(df[c].dropna().astype(str).unique())
            if 2 <= len(vals) <= 2:
                out.append(c)
    return out


def _named(cols, words):
    return [c for c in cols if any(w in c.lower() for w in words)]


def suggest(raw_df: pd.DataFrame, clean_df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    num = [c for c, t in profile.items() if t == "numeric"]
    cat = [c for c, t in profile.items() if t == "categorical"]
    dates = [c for c, t in profile.items() if t == "datetime"]
    two = _yes_no_cols(clean_df, profile)
    texts = text_analysis.text_columns(clean_df, profile)
    money = _named(num, ("price", "cost", "amount", "revenue", "sales", "spend", "value", "premium", "obligat"))
    rows = []

    def add(problem, ok, mode, how):
        rows.append({"Business question": problem, "Can this data answer it?": "Yes" if ok else "Not yet",
                     "Where in the app": mode if ok else "", "Why / what's needed": how})

    target = (money or num or ["a number"])[0]
    add("How much will we sell or spend next week, month or quarter? (demand, revenue forecasts)",
        bool(dates and num), "Forecast future values",
        f"Forecast '{target}' by '{dates[0]}'." if dates and num else "Needs a date column and a number column.")

    churn = _named(two, ("churn", "cancel", "left", "renew", "retain", "convert", "default", "fraud", "fail", "late",
                         "active", "subscri", "return", "bought", "purchase", "click", "respond"))
    add("Which customers will leave, convert or default? (churn, conversion, retention)",
        bool(churn), "Predict a target variable",
        f"Predict the yes/no column '{churn[0]}' from the other columns." if churn
        else "Needs a column with two outcomes, such as churned Yes/No."
        + (f" Two-value columns here ({', '.join(two[:3])}) can still be predicted in Predict." if two else ""))

    add("Which records look unusual or possibly fraudulent?",
        len(num) >= 1, "Detect anomalies",
        "Isolation Forest scores every row; the most unusual are listed first. A screen for review, not proof of fraud."
        if num else "Needs at least one number column.")

    add("What should this cost? (pricing, estimates)",
        bool(num) and (len(num) + len(cat)) >= 2, "Predict a target variable",
        f"Predict '{target}' from the other columns and compare models." if num and (len(num) + len(cat)) >= 2
        else "Needs a number to predict and other columns to predict it from.")

    add("Which customers, vendors or sites are alike? (segments, scoring)",
        len(num) >= 2, "Find hidden groups",
        "Clustering groups similar rows; the profile table shows what sets each group apart."
        if len(num) >= 2 else "Needs at least two number columns.")

    add("How do totals and averages compare across groups?",
        bool(cat or dates) and bool(num), "Group & summarize",
        "Group by a category or by month and total or average any number." if (cat or dates) and num
        else "Needs a category or date column and a number column.")

    order, item = basket.guess_columns(raw_df)
    basket_ok = False
    order_named = order and any(h in order.lower() for h in ("order", "invoice", "transaction", "basket", "receipt",
                                                             "cart", "ticket", "customer"))
    if order_named and item and order != item:
        try:
            sizes = raw_df[[order, item]].dropna().groupby(order)[item].nunique()
            basket_ok = len(sizes) >= 10 and (sizes >= 2).mean() >= 0.1
        except Exception:
            basket_ok = False
    add("What do customers buy together? (cross-sell, up-sell, recommendations)",
        basket_ok, "Find items bought together",
        f"Orders in '{order}', items in '{item}'." if basket_ok
        else "Needs one row per item with an order or customer column that repeats.")

    sensors = _named(num, ("temp", "pressure", "vibration", "reading", "sensor", "level", "conc"))
    fail = _named(two, ("fail", "fault", "breakdown", "downtime", "repair"))
    add("Which machines or sites are about to fail or drift? (maintenance)",
        bool(fail or (sensors and dates)), "Predict a target variable" if fail else "Combine & standardize sources",
        (f"Predict '{fail[0]}' from the readings." if fail else
         "Flag readings that jump or drift against the other sites.") if (fail or (sensors and dates))
        else "Needs sensor readings over time, or a failed Yes/No column.")

    add("How do customers feel? (sentiment in comments or reviews)",
        bool(texts), "Analyze text",
        f"Score the comments in '{texts[0]}' as positive, neutral or negative." if texts
        else "Needs a column of written comments.")

    add("Find patterns in images, audio or scanned documents",
        False, "", "Not supported: the app works on tables and text columns only.")
    return pd.DataFrame(rows)
