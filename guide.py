"""
Guided start: plain-language goals that lead to the right analysis mode, a short "how to use this"
panel for every mode, and the list of everyday vs advanced tools.
"""
import pandas as pd
import streamlit as st

MODES = {   # name: (caption, everyday?)
    "Explore & describe": ("Frequency tables, statistics, charts", True),
    "Business problem & KPIs": ("Problem, requirements, KPIs, feature importance, Word brief", True),
    "Group & summarize": ("Group by, totals, averages, pivot tables", True),
    "Predict a target variable": ("Regression or classification", True),
    "Forecast future values": ("Time series; needs a date column", True),
    "Test statistical significance": ("Is this pattern real, or just noise?", True),
    "Analyze text": ("Sentiment and common words in comments", True),
    "Find items bought together": ("Cross-sell: what sells with what", True),
    "Detect anomalies": ("Flag statistical outliers", True),
    "Find hidden groups": ("Clustering; no target needed", False),
    "Prepare & transform": ("Data quality, scale types, normalize, PCA", False),
    "Data pipeline": ("Validate, bronze/silver/gold, PII, joins, SQL, schedule", False),
    "Combine & standardize sources": ("Make sites, labs or methods comparable; one index", False),
    "Deep learning (Keras)": ("Neural networks, image CNN, deep Q-learning", False),
}

# goal shown to the user -> (mode, one-line reason)
GOALS = [
    ("Get to know this data", "Explore & describe", "Counts, averages, charts and how columns relate."),
    ("Totals or averages by group", "Group & summarize", "Like a pivot table in Excel."),
    ("Forecast future numbers", "Forecast future values", "Next week's, month's or quarter's numbers. Needs a date column and a number."),
    ("Predict an outcome", "Predict a target variable",
     "Yes/no, a category or an amount. For example who will churn, default or respond, or what something will cost."),
    ("Is a difference real?", "Test statistical significance",
     "For example: do two stores really differ?"),
    ("Find unusual records", "Detect anomalies", "A screen for review, such as possible errors or fraud."),
    ("Find similar groups", "Find hidden groups", "Segments without labeling anything first."),
    ("Read comments and reviews", "Analyze text", "Positive / negative and common words."),
    ("What sells together", "Find items bought together", "Cross-sell and bundle ideas."),
    ("Business problem and KPIs", "Business problem & KPIs",
     "Problem statement, KPIs and trends, what drives the target, and a Word brief."),
    ("Compare sites or labs", "Combine & standardize sources", "One fair index from many sources."),
    ("Build a data pipeline", "Data pipeline", "Quality checks, clean tables, SQL, scheduled runs."),
    ("Deep learning", "Deep learning (Keras)", "TensorFlow / Keras."),
    ("Prepare columns for modeling", "Prepare & transform", "Binning, encoding, scaling, PCA."),
]

HELP = {  # mode: (steps, what the results mean)
    "Explore & describe": (["Pick a column in **One variable** to see its counts, average and chart.",
                            "Use **Two variables** to compare a number across groups or see how two numbers move together.",
                            "Use **Many variables** for the correlation map of all number columns."],
                           "The median is the typical value; when the average is much higher, a few large values pull it up."),
    "Business problem & KPIs": (["Write the problem and requirements in tabs 1–2 and pick the target (what you want to explain).",
                                 "Tab 4 measures KPIs and whether they are rising or falling; tab 5 shows what drives the target.",
                                 "Tab 6 builds a Word brief you can hand to the client."],
                                "A trend is called rising or falling only when the straight-line test is significant (p < 0.05)."),
    "Group & summarize": (["Pick one or more columns to group by, or a date and a period (month, quarter...).",
                           "Pick the numbers and calculations (total, average...).",
                           "Use the Pivot table tab for a rows × columns table."],
                          "Each row is one group; 'Rows in group' shows how many records it is based on."),
    "Predict a target variable": (["Choose the column you want to predict.",
                                   "Remove inputs that would not be known in advance or are the answer in disguise.",
                                   "Click **Run Model Selector**; the app tests up to 10 models on rows they never saw."],
                                  "Number targets: lower RMSE/MAE and higher R² are better. Categories: look at F1 and "
                                  "recall, not just accuracy, when one outcome is rare."),
    "Forecast future values": (["Pick the date column and the number to forecast.",
                                "Choose the period (week, month...) and how to combine rows (total or average).",
                                "Click **Run Forecast**; the app tests 7 methods on your most recent periods."],
                               "The shaded band is a rough range from past errors, not a guarantee. If no method beats "
                               "'repeat the last value', the data has little predictable pattern."),
    "Test statistical significance": (["Pick the question: are two numbers related, do groups differ, are two categories linked?",
                                       "Pick the columns and click **Run Test**."],
                                      "p < 0.05 means the pattern is unlikely to be chance. It says nothing about how big "
                                      "or important the effect is."),
    "Analyze text": (["Pick the column of comments.", "Optionally compare by a group such as store or product."],
                     "Scores run from −1 (very negative) to +1 (very positive). Read a few examples: the method misses sarcasm."),
    "Find items bought together": (["Pick the order (or customer) column and the item column.",
                                    "Pairs with lift above 1 are bought together more often than chance."],
                                   "Confidence = of orders with item A, the share that also have B. It shows what happened "
                                   "together, not cause."),
    "Detect anomalies": (["Set roughly what share of rows you expect to be unusual (5% is a good start).",
                          "Click **Run Anomaly Detection** and review the top rows."],
                         "Flagged rows are unusual, not necessarily wrong. A person who knows the data should check them."),
    "Find hidden groups": (["Click **Run Clustering**; the app tries several group counts and keeps the best."],
                           "The profile table shows the average of each column per group: that is what sets the group apart."),
    "Prepare & transform": (["Use the tabs to check quality, fill gaps, filter, edit columns, rescale or transform.",
                             "Every result can be downloaded as a new file."],
                            "These tools make copies; the main analysis keeps using the automatically cleaned data."),
    "Data pipeline": (["Check the quality gates in tab 2, then review bronze (raw), silver (clean) and gold (summary).",
                       "Tab 6 saves the steps so they can be re-run on new files on a schedule.",
                       "Tab 7 lets you query the tables with SQL and download everything as one bundle."],
                      "FAIL checks stop a scheduled run before bad data reaches reports."),
    "Combine & standardize sources": (["Pick the source (site, lab), date and measurement columns.",
                                       "Review flagged readings, gaps and step changes per series."],
                                      "0 on the index is the typical level; +1 is one typical spread above it."),
    "Deep learning (Keras)": (["Choose a tab: table, images or agent.",
                               "Keep the default settings for a first run, then adjust one thing at a time.",
                               "Compare the neural network with the simpler models before using it."],
                              "On spreadsheets, simpler models often do as well; neural networks shine on images and text."),
}


def everyday() -> list:
    return [m for m, (_, e) in MODES.items() if e]


def advanced() -> list:
    return [m for m, (_, e) in MODES.items() if not e]


def _go(mode: str):
    st.session_state["mode"] = mode
    if mode in advanced():
        st.session_state["show_advanced"] = True


def render_goal_picker(profile: dict, df: pd.DataFrame):
    """'What do you want to know?' with goals that fit this data first."""
    num = [c for c, t in profile.items() if t == "numeric"]
    cat = [c for c, t in profile.items() if t == "categorical"]
    dates = [c for c, t in profile.items() if t == "datetime"]
    text = [c for c, t in profile.items() if t == "text"]
    fits = {
        "Forecast future values": bool(dates and num),
        "Predict a target variable": bool(num or cat) and df.shape[1] >= 2,
        "Test statistical significance": len(num) >= 2 or bool(num and cat) or len(cat) >= 2,
        "Group & summarize": bool((cat or dates) and num),
        "Find hidden groups": len(num) >= 2,
        "Detect anomalies": bool(num),
        "Analyze text": bool(text) or any(df[c].astype(str).str.split().str.len().mean() >= 3 for c in cat[:20]),
        "Combine & standardize sources": bool(dates and num and cat),
        "Find items bought together": bool(cat) and any(k in c.lower() for c in df.columns
                                                        for k in ("order", "invoice", "transaction", "basket")),
    }
    with st.container(border=True):
        st.markdown("**What do you want to know?** Pick a goal and the app opens the right tool.")
        shown = [g for g in GOALS if fits.get(g[1], True)]
        hidden = [g for g in GOALS if not fits.get(g[1], True)]
        cols = st.columns(2)
        for i, (label, mode, why) in enumerate(shown):
            cols[i % 2].button(label, key=f"goal_{i}", help=why, on_click=_go, args=(mode,),
                               use_container_width=True,
                               type="primary" if st.session_state.get("mode") == mode else "secondary")
        if hidden:
            st.caption("Not available for this file: " + "; ".join(f"{g[0].lower()} ({_why_not(g[1])})" for g in hidden))


def _why_not(mode: str) -> str:
    return {"Forecast future values": "needs a date column and a number",
            "Group & summarize": "needs a category or date column and a number",
            "Find hidden groups": "needs two number columns",
            "Detect anomalies": "needs a number column",
            "Analyze text": "needs a column of written comments",
            "Combine & standardize sources": "needs a date, a number and a source column",
            "Find items bought together": "needs an order or invoice column",
            "Test statistical significance": "needs two columns to compare",
            "Predict a target variable": "needs a number or category column"}.get(mode, "not suited to this file")


def render_help(mode: str):
    steps, meaning = HELP.get(mode, ([], ""))
    if not steps:
        return
    with st.expander("How to use this", expanded=st.session_state.get("first_visit_" + mode, True)):
        for i, s in enumerate(steps, 1):
            st.markdown(f"{i}. {s}")
        if meaning:
            st.caption("Reading the results: " + meaning)
    st.session_state["first_visit_" + mode] = False


def render_start_steps():
    c = st.columns(3)
    for col, (n, title, text) in zip(c, [
        ("1", "Load your data", "Upload a file in the sidebar (Excel, CSV and more), paste a web address, connect a "
                                "database, or pick sample data."),
        ("2", "Check the cleaning", "The app fixes types, gaps and duplicates and lists every change. Adjust anything "
                                    "that looks wrong."),
        ("3", "Ask your question", "Pick a goal such as 'predict next month' or 'see totals by region'; the app opens the "
                                   "right tool with instructions."),
    ]):
        col.markdown(f'<div class="vg-card"><h4>{n} · {title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
    st.write("")
