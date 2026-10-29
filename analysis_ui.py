"""Streamlit screens added in update 18: Group & summarize, Analyze text, Find items bought together."""
import altair as alt
import pandas as pd
import streamlit as st

import basket
import grouping
import text_analysis as ta
from explore_ui import _download

BLUE = "#2a78d6"
SENT_COLORS = alt.Scale(domain=["Negative", "Neutral", "Positive"], range=["#e34948", "#a3a29c", "#2a78d6"])


def _bars(df: pd.DataFrame, label: str, value: str, title: str, color: str = BLUE, label_title: str | None = None):
    d = pd.DataFrame({"Label": df[label].astype(str).values, "Value": df[value].values})
    name = label_title or label
    return alt.Chart(d).mark_bar(color=color, cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
        y=alt.Y("Label:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
        x=alt.X("Value:Q", title=value),
        tooltip=[alt.Tooltip("Label:N", title=name), alt.Tooltip("Value:Q", title=value, format=",.2f")],
    ).properties(height=alt.Step(24), title=title)


# =============================== GROUP & SUMMARIZE ===============================
def render_group(df: pd.DataFrame, profile: dict):
    st.write("Split the rows into groups, calculate something for each group, and put the results side by side "
             "(pandas **groupby** and **pivot_table**; Spark and SQL use the same split-apply-combine idea).")
    num = [c for c, t in profile.items() if t == "numeric"]
    cat = [c for c, t in profile.items() if t == "categorical"]
    dates = [c for c, t in profile.items() if t == "datetime"]
    if not (cat or dates):
        st.warning("Grouping needs a category column (such as region or product) or a date column.")
        return
    tab_g, tab_p = st.tabs(["Group by", "Pivot table (rows × columns)"])

    with tab_g:
        c1, c2 = st.columns(2)
        by = c1.multiselect("Group by", cat, default=cat[:1], key="gb_by")
        date_col, period = None, None
        if dates:
            date_col = c2.selectbox("And/or group dates by period", ["(no date grouping)"] + dates, key="gb_date")
            if date_col.startswith("("):
                date_col = None
            else:
                period = c2.selectbox("Period", [p for p in grouping.PERIODS if p != "(use exact dates)"],
                                      index=2, key="gb_period")
        c3, c4 = st.columns(2)
        money = [c for c in num if any(k in c.lower() for k in ("revenue", "sales", "amount", "cost", "price", "spend",
                                                                  "balance", "obligat", "units", "visits", "value"))]
        values = c3.multiselect("Numbers to summarize", num, default=(money or num)[:1], key="gb_vals")
        funcs = c4.multiselect("Calculations", list(grouping.FUNCS), default=["Average (mean)", "Total (sum)"],
                               key="gb_funcs")
        if not by and not date_col:
            st.info("Pick a column to group by, or a date column and period.")
            return
        try:
            out = grouping.group_summary(df, by, values, funcs if values else [], date_col, period)
        except Exception as e:
            st.error(f"Could not group: {e}")
            return
        st.caption(f"{len(out):,} groups.")
        st.dataframe(out, hide_index=True, width="stretch")
        _download(out, "group_summary.csv")
        if values and funcs:
            metric = out.columns[len(out.columns) - len(values) * len(funcs)]
            label_cols = [c for c in out.columns if c not in ("Rows in group",) and "·" not in c]
            plot = out.copy()
            plot["_label"] = plot[label_cols].astype(str).agg(" | ".join, axis=1)
            if date_col and period and len(label_cols) == 1:
                d = pd.DataFrame({"Period": pd.to_datetime(plot[label_cols[0]]), "Value": plot[metric]})
                st.altair_chart(alt.Chart(d).mark_line(color=BLUE, point=True).encode(
                    x=alt.X("Period:T", title=None), y=alt.Y("Value:Q", title=metric),
                    tooltip=[alt.Tooltip("Period:T"), alt.Tooltip("Value:Q", format=",.2f")]
                ).properties(height=300), width="stretch")
            else:
                top = plot.head(25)
                grp = " | ".join(label_cols)
                st.altair_chart(_bars(top, "_label", metric, f"{metric} by {grp}"
                                      + (" (top 25)" if len(plot) > 25 else ""), label_title=grp), width="stretch")

    with tab_p:
        if not num:
            st.warning("A pivot table needs a number column to fill the cells.")
            return
        row_opts = cat + dates
        c1, c2 = st.columns(2)
        rows = c1.selectbox("Rows", row_opts, key="pv_rows")
        col_opts = [c for c in cat if c != rows]
        if not col_opts:
            st.info("A pivot table needs a second category column for the columns.")
            return
        cols = c2.selectbox("Columns", col_opts, key="pv_cols")
        c3, c4 = st.columns(2)
        value = c3.selectbox("Value in each cell", num, key="pv_val")
        func = c4.selectbox("Calculation", [f for f in grouping.FUNCS if f != "Number of different values"], key="pv_func")
        period = None
        if rows in dates:
            period = st.selectbox("Group dates by", ["Month", "Quarter", "Year", "Week", "Day"], key="pv_period")
        if df[cols].nunique() > 30:
            st.warning(f"'{cols}' has {df[cols].nunique()} values; the table will be wide.")
        try:
            t = grouping.pivot(df, rows, cols, value, func, rows if rows in dates else None, period)
        except Exception as e:
            st.error(f"Could not build the pivot table: {e}")
            return
        st.dataframe(t, width="stretch")
        _download(t.reset_index(), "pivot_table.csv")
        st.caption("'All' is the row or column total, calculated with the same function.")


# =============================== ANALYZE TEXT ===============================
def render_text(df: pd.DataFrame, profile: dict):
    st.write("Count the words people use and score each comment as **positive, neutral or negative**, so thousands "
             "of comments can be sorted and compared by group.")
    cols = ta.text_columns(df, profile)
    if not cols:
        st.warning("No column of written comments was found. This needs a column of sentences, such as reviews, "
                   "survey answers or notes. Try the sample **Customer reviews**.")
        return
    cat = [c for c, t in profile.items() if t == "categorical" and c not in cols and df[c].nunique() <= 40]
    c1, c2 = st.columns(2)
    col = c1.selectbox("Text column", cols, key="tx_col")
    grp = c2.selectbox("Compare by (optional)", ["(none)"] + cat, key="tx_grp")
    series = df[col].astype("object").where(df[col].notna(), None)

    with st.spinner("Scoring comments..."):
        scored = _score(series)
    has = scored[scored["sentiment"] != "No text"]
    if has.empty:
        st.warning("That column has no text to score.")
        return
    shares = has["sentiment"].value_counts(normalize=True)
    m = st.columns(4)
    m[0].metric("Comments", f"{len(has):,}")
    m[1].metric("Positive", f"{shares.get('Positive', 0):.0%}")
    m[2].metric("Neutral", f"{shares.get('Neutral', 0):.0%}")
    m[3].metric("Negative", f"{shares.get('Negative', 0):.0%}")
    st.caption(f"Average score {has['sentiment_score'].mean():+.2f} on a scale from −1 (very negative) to +1 "
               f"(very positive). Typical comment length: {int(has['word_count'].median())} words. Scored with VADER, "
               "a rule-based method for short English text; it misses sarcasm and jargon, so read a few examples.")

    hist = alt.Chart(has).mark_bar().encode(
        x=alt.X("sentiment_score:Q", bin=alt.Bin(step=0.1), title="Sentiment score"),
        y=alt.Y("count():Q", title="Comments"),
        color=alt.Color("sentiment:N", scale=SENT_COLORS, title=None))
    st.altair_chart(hist.properties(height=240), width="stretch")

    if grp != "(none)":
        st.markdown(f"**Sentiment by {grp}** (lowest average first)")
        bg = ta.by_group(scored, df[grp])
        st.dataframe(bg, hide_index=True, width="stretch")
        st.altair_chart(_bars(bg, grp, "Average score", f"Average sentiment by {grp}"), width="stretch")
        _download(bg, "sentiment_by_group.csv")
        rating = [c for c, t in profile.items() if t == "numeric" and any(k in c.lower() for k in ("rating", "score", "star"))]
        if rating:
            r = pd.concat([scored["sentiment_score"], df[rating[0]]], axis=1).dropna().corr(method="spearman").iloc[0, 1]
            st.caption(f"Check: the sentiment score and '{rating[0]}' have a rank correlation of {r:.2f} "
                       "(closer to 1 means the text and the rating agree).")

    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Most common words**")
        wc = ta.word_counts(series)
        if not wc.empty:
            st.altair_chart(_bars(wc, "Word", "Count", ""), width="stretch")
    with w2:
        st.markdown("**Most common two-word phrases**")
        pc = ta.phrase_counts(series)
        if pc.empty:
            st.caption("No phrase appears more than once.")
        else:
            st.altair_chart(_bars(pc, "Phrase", "Count", ""), width="stretch")

    st.markdown("**Words used most in positive vs negative comments**")
    split = ta.words_by_sentiment(series, scored)
    p1, p2 = st.columns(2)
    for box, lab, color in ((p1, "Positive", BLUE), (p2, "Negative", "#e34948")):
        with box:
            t = split[lab]
            if t.empty:
                st.caption(f"No {lab.lower()} comments.")
            else:
                st.altair_chart(_bars(t, "Word", "Count", lab, color), width="stretch")

    st.markdown("**Examples**")
    show = pd.concat([df[[col] + ([grp] if grp != "(none)" else [])], scored[["sentiment_score", "sentiment"]]], axis=1)
    show = show[show["sentiment"] != "No text"]
    e1, e2 = st.columns(2)
    e1.caption("Most negative")
    e1.dataframe(show.nsmallest(5, "sentiment_score"), hide_index=True, width="stretch")
    e2.caption("Most positive")
    e2.dataframe(show.nlargest(5, "sentiment_score"), hide_index=True, width="stretch")
    _download(pd.concat([df, scored], axis=1), "with_sentiment.csv", "Download data with sentiment scores")


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)  # results leave server memory within an hour
def _score(series: pd.Series) -> pd.DataFrame:
    return ta.sentiment(series)


# =============================== ITEMS BOUGHT TOGETHER ===============================
def render_basket(raw_df: pd.DataFrame):
    st.write("Find which items are bought together, to suggest a second item (**cross-sell**) or a bundle. "
             "Needs one row per item, with a column that says which order or customer it belongs to.")
    cols = list(raw_df.columns)
    g_order, g_item = basket.guess_columns(raw_df)
    c1, c2, c3 = st.columns(3)
    order = c1.selectbox("Order (or customer) column", cols, index=cols.index(g_order) if g_order in cols else 0,
                         key="mb_order", help="The column that groups rows into one basket: order, invoice or customer.")
    item_opts = [c for c in cols if c != order]
    item = c2.selectbox("Item column", item_opts,
                        index=item_opts.index(g_item) if g_item in item_opts else 0, key="mb_item")
    min_orders = c3.number_input("Pair must appear in at least this many orders", 2, 1000, 5, key="mb_min",
                                 help="Pairs seen only a few times are often chance.")
    try:
        r = _rules(raw_df, order, item, int(min_orders))
    except ValueError as e:
        st.warning(str(e))
        return
    m = st.columns(4)
    m[0].metric("Orders", f"{r['orders']:,}")
    m[1].metric("Orders with 2+ items", f"{r['multi_item_orders']:,}")
    m[2].metric("Items per order (avg)", f"{r['avg_items']:.1f}")
    m[3].metric("Item pairs found", f"{len(r['rules']) // 2:,}")
    rules = r["rules"]
    if rules.empty:
        st.info("No pair of items appears together often enough. Lower the minimum number of orders.")
        return
    strong = rules[rules["Lift"] > 1]
    st.markdown("**Strongest pairs** (highest lift first)")
    st.dataframe(strong.head(30), hide_index=True, width="stretch")
    st.caption("Support: share of all orders with both items. Confidence: of orders with the first item, the share "
               "that also have the second. Lift: how many times more often they appear together than by chance "
               "(1 = no link). These show what happened together, not that one purchase causes the other.")
    _download(rules, "items_bought_together.csv")

    st.markdown("**Suggest items for a product**")
    _items = r["items"]["Item"].tolist()
    _first = strong["If an order has"].iloc[0] if not strong.empty else _items[0]
    pick = st.selectbox("Customer is buying", _items, index=_items.index(_first), key="mb_pick")
    rec = basket.recommend(rules, pick)
    if rec.empty:
        st.caption("No item is bought with this one clearly more often than chance (lift of 1.2 or more).")
    else:
        for _, row in rec.iterrows():
            st.write(f"• **{row['it also has']}**: in {row['Confidence (%)']:.0f}% of orders with {pick} "
                     f"({row['Lift']:.1f}× more often than chance, {int(row['Orders with both'])} orders)")
    with st.expander("How often each item is ordered"):
        st.dataframe(r["items"], hide_index=True, width="stretch")


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)  # results leave server memory within an hour
def _rules(df: pd.DataFrame, order: str, item: str, min_orders: int):
    return basket.rules(df, order, item, min_orders)
