"""
Streamlit screens for:
  - Explore & describe  (one variable / two variables / many variables)
  - Prepare & transform (data quality, scale types & conversion, rescaling,
                         transformations, dimensionality reduction)
"""
import pandas as pd
import streamlit as st

import charts
import descriptive as d


def _num(profile):
    return [c for c, t in profile.items() if t == "numeric"]


def _cat(profile):
    return [c for c, t in profile.items() if t == "categorical"]


def _txt(profile):
    return [c for c, t in profile.items() if t == "text"]


def _chart(c):
    st.altair_chart(c, use_container_width=True)


def _download(df: pd.DataFrame, filename: str, label: str = "Download as CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


# =============================== EXPLORE & DESCRIBE ===============================
def render_explore(df: pd.DataFrame, profile: dict):
    st.write("Describe what's in the data before modeling it: frequency tables, summary statistics, and charts.")
    one, two, many = st.tabs(["One variable (univariate)", "Two variables (bivariate)", "Many variables (multivariate)"])
    num, cat = _num(profile), _cat(profile)

    # ---------------- univariate ----------------
    with one:
        cols = num + cat
        if not cols:
            st.warning("No numeric or categorical columns to describe.")
        else:
            col = st.selectbox("Column", cols, key="uni_col")
            kind = profile[col]
            stype = d.scale_type(df[col], kind)
            st.caption(f"Scale type: **{stype}** — {d.SCALE_EXPLANATIONS[stype]}")

            left, right = st.columns(2)
            with left:
                st.markdown("**Frequency table**")
                freq = d.frequency_table(df[col], kind)
                st.dataframe(freq, hide_index=True, use_container_width=True)
                if kind == "numeric" and df[col].nunique() > 15:
                    st.caption("Values grouped into ranges (Sturges' rule) because there are too many distinct values to count one by one.")
            with right:
                st.markdown("**Statistical measures**")
                stats_df = d.summary_stats(df[col]) if kind == "numeric" else d.categorical_stats(df[col])
                if kind == "numeric":
                    stats_df["Value"] = stats_df["Value"].map(lambda v: f"{v:,.3f}" if pd.notna(v) else "")
                st.dataframe(stats_df, hide_index=True, use_container_width=True)

            st.markdown("**Charts**")
            if kind == "numeric":
                kinds = ["Histogram", "Box plot"]
            else:
                kinds = ["Bar chart", "Pie chart"]
            choice = st.radio("Chart type", kinds, horizontal=True, key="uni_chart")
            if choice == "Histogram":
                _chart(charts.histogram(df, col))
            elif choice == "Box plot":
                _chart(charts.box_plot(df, col))
            elif choice == "Bar chart":
                _chart(charts.bar_chart(freq, "Value", "Absolute frequency", f"Frequency of {col}"))
            else:
                if df[col].nunique() < 3:
                    st.info("With fewer than 3 categories a pie adds nothing — the frequency table above says it all.")
                _chart(charts.pie_chart(df[col], f"Share of {col}"))
                if df[col].nunique() > 6:
                    st.caption("Showing the 6 most common categories; the rest are grouped as 'Other'.")

    # ---------------- bivariate ----------------
    with two:
        cols = num + cat
        if len(cols) < 2:
            st.warning("Need at least 2 numeric or categorical columns.")
        else:
            a = st.selectbox("First column", cols, key="bi_a")
            b = st.selectbox("Second column", [c for c in cols if c != a], key="bi_b")
            ka, kb = profile[a], profile[b]

            if ka == "numeric" and kb == "numeric":
                st.markdown("**Relationship measures**")
                m = d.bivariate_numeric(df, a, b)
                m["Value"] = m["Value"].map(lambda v: f"{v:,.4f}")
                st.dataframe(m, hide_index=True, use_container_width=True)
                choice = st.radio("Chart type", ["Scatter plot", "Line chart", "Area chart"], horizontal=True, key="bi_chart")
                if choice == "Scatter plot":
                    color = st.selectbox("Color points by (optional)", ["(none)"] + cat, key="bi_color")
                    _chart(charts.scatter(df, a, b, None if color == "(none)" else color))
                else:
                    _chart(charts.line_or_area(df, a, b, "line" if choice == "Line chart" else "area"))
                    st.caption(f"Where several rows share the same {a}, their {b} values are averaged.")

            elif "numeric" in (ka, kb):
                num_col, cat_col = (a, b) if ka == "numeric" else (b, a)
                st.markdown(f"**Summary of {num_col} by {cat_col}**")
                gs = d.group_summary(df, cat_col, num_col)
                st.dataframe(gs, hide_index=True, use_container_width=True)
                _download(gs, f"{num_col}_by_{cat_col}.csv")
                choice = st.radio("Chart type", ["Box plot by group", f"Bar chart of average {num_col}"], horizontal=True, key="bi_chart2")
                if choice.startswith("Box"):
                    _chart(charts.box_plot(df, num_col, cat_col, d._sorted_categories(df[cat_col].dropna())))
                else:
                    _chart(charts.bar_chart(gs, cat_col, "Mean", f"Average {num_col} by {cat_col}"))

            else:
                st.markdown("**Cross-tabulation (two-way frequency table)**")
                view = st.radio("Show", ["Counts", "Row %", "Column %", "% of total"], horizontal=True, key="bi_ct")
                norm = {"Counts": None, "Row %": "index", "Column %": "columns", "% of total": "all"}[view]
                ct = d.crosstab(df, a, b, norm)
                st.dataframe(ct, use_container_width=True)
                counts = d.crosstab(df, a, b).drop(index="Total", columns="Total")
                counts.index.name = a
                _chart(charts.heatmap(counts, f"{a} × {b} (counts)", diverging=False, fmt=",.0f"))

    # ---------------- multivariate ----------------
    with many:
        if len(num) >= 2:
            st.markdown("**Correlation matrix**")
            method = st.radio("Method", ["Pearson (straight-line)", "Spearman (rank-based)"], horizontal=True, key="mv_m")
            corr = d.correlation_matrix(df, num, "pearson" if method.startswith("Pearson") else "spearman")
            corr.index.name = "variable"
            _chart(charts.heatmap(corr, "Correlation between numeric columns (−1 to 1)"))
            st.caption("Blue = move in opposite directions, red = move together, pale = little or no relationship.")
            with st.expander("Covariance matrix"):
                st.dataframe(d.covariance_matrix(df, num), use_container_width=True)
            with st.expander("Summary statistics for every numeric column"):
                st.dataframe(df[num].describe().T.round(2), use_container_width=True)
        else:
            st.info("Correlation matrix needs at least 2 numeric columns.")

        if len(cat) >= 2:
            st.markdown("**Multivariate frequencies (combinations of categories)**")
            chosen = st.multiselect("Categorical columns to combine", cat, default=cat[:2], key="mv_cats")
            if len(chosen) >= 2:
                mf = d.multivariate_frequency(df, chosen)
                st.dataframe(mf, hide_index=True, use_container_width=True)
                _download(mf, "multivariate_frequencies.csv")

        txt = _txt(profile)
        if txt:
            st.markdown("**Word frequencies (free-text columns)**")
            tcol = st.selectbox("Text column", txt, key="mv_txt")
            wf = d.word_frequencies(df[tcol])
            if wf.empty:
                st.info("No words found in that column.")
            else:
                _chart(charts.bar_chart(wf, "Word", "Frequency", f"Most common words in {tcol}", horizontal=True))
                st.caption("Shown as a ranked bar chart rather than a word cloud: bar length is far easier to compare than font size.")


# =============================== PREPARE & TRANSFORM ===============================
def render_prepare(raw_df: pd.DataFrame, df: pd.DataFrame, profile: dict):
    st.write("Check data quality and reshape columns before analysis. Every result can be downloaded as a CSV.")
    q, scale, resc, trans, dim = st.tabs(
        ["Data quality", "Scale types & conversion", "Rescale / normalize", "Transform", "Dimensionality reduction"])
    num, cat = _num(profile), _cat(profile)

    with q:
        st.markdown("**Data quality report**")
        rep = d.data_quality_report(raw_df, df, profile)
        st.dataframe(rep, hide_index=True, use_container_width=True)
        st.caption("'Missing in original' is before cleaning; the cleaned data has those gaps filled (see the cleaning report above). "
                   "Outliers use the same 1.5 × IQR rule as the box plots.")
        _download(rep, "data_quality_report.csv")

    with scale:
        st.markdown("**Scale types**")
        st.dataframe(d.scale_type_table(df, profile), hide_index=True, use_container_width=True)
        st.markdown("**Convert to a different scale type**")
        conv = st.radio("Conversion", [
            "Numeric → ordinal categories (binning)",
            "Nominal → binary columns (one-hot encoding)",
            "Ordinal → rank numbers",
        ], key="conv")
        if conv.startswith("Numeric"):
            if not num:
                st.warning("No numeric columns.")
            else:
                c = st.selectbox("Numeric column", num, key="bin_c")
                k = st.slider("Number of categories", 2, 10, 3, key="bin_k")
                method = st.radio("Bin method", ["equal width", "equal frequency"], horizontal=True, key="bin_m",
                                  help="Equal width: same-size value ranges. Equal frequency: about the same number of rows per category.")
                out = df[[c]].copy()
                out[f"{c}_category"] = d.bin_numeric(df[c], k, method)
                st.dataframe(out.head(20), use_container_width=True)
                st.dataframe(d.frequency_table(out[f"{c}_category"], "categorical"), hide_index=True)
                _download(df.assign(**{f"{c}_category": out[f"{c}_category"]}), f"{c}_binned.csv", "Download full data with new column")
        elif conv.startswith("Nominal"):
            if not cat:
                st.warning("No categorical columns.")
            else:
                c = st.selectbox("Categorical column", cat, key="oh_c")
                oh = d.one_hot(df, c)
                st.dataframe(pd.concat([df[[c]], oh], axis=1).head(20), use_container_width=True)
                _download(pd.concat([df, oh], axis=1), f"{c}_one_hot.csv", "Download full data with new columns")
        else:
            if not cat:
                st.warning("No categorical columns.")
            else:
                c = st.selectbox("Categorical column", cat, key="ord_c")
                order = d.ordinal_order(df[c])
                st.caption("Natural order detected: " + " < ".join(map(str, order)) if order
                           else "No natural order recognized — ranks follow frequency (most common = 1). Treat with care: this column may be nominal.")
                out = df[[c]].copy()
                out[f"{c}_rank"] = d.ordinal_encode(df[c])
                st.dataframe(out.drop_duplicates().sort_values(f"{c}_rank"), hide_index=True)
                _download(df.assign(**{f"{c}_rank": out[f"{c}_rank"]}), f"{c}_ranked.csv", "Download full data with new column")

    with resc:
        if not num:
            st.warning("No numeric columns.")
        else:
            st.markdown("**Rescale numeric columns to a common scale**")
            chosen = st.multiselect("Columns", num, default=num, key="rs_cols")
            method = st.radio("Method", ["min-max", "z-score"], horizontal=True, key="rs_m",
                              help="Min-max: squeezes values into 0–1. Z-score: mean 0, standard deviation 1.")
            if chosen:
                scaled = df.copy()
                for c in chosen:
                    scaled[c] = d.rescale(df[c], method).round(4)
                cmp = pd.DataFrame({c: {"Min before": df[c].min(), "Max before": df[c].max(), "Mean before": df[c].mean(),
                                        "Min after": scaled[c].min(), "Max after": scaled[c].max(), "Mean after": scaled[c].mean()}
                                    for c in chosen}).T.round(3)
                st.dataframe(cmp, use_container_width=True)
                show = st.selectbox("Compare distribution for", chosen, key="rs_show")
                _chart(charts.before_after_histograms(df[show], scaled[show], show))
                st.caption("Rescaling changes the units, not the shape — the two histograms should look the same.")
                _download(scaled, f"rescaled_{method}.csv")

    with trans:
        if not num:
            st.warning("No numeric columns.")
        else:
            st.markdown("**Transform a numeric column**")
            c = st.selectbox("Column", num, key="tr_c")
            method = st.radio("Transformation", ["log", "square root", "Box-Cox", "square"], horizontal=True, key="tr_m",
                              help="Log, square root and Box-Cox pull in a long right tail (reduce skew). Square stretches it.")
            try:
                out = d.transform(df[c], method)
                sk = pd.DataFrame({"Skewness": [df[c].skew(), out.skew()]}, index=["Before", "After"]).round(3)
                st.dataframe(sk, use_container_width=True)
                st.caption("Skewness closer to 0 means a more symmetric distribution.")
                _chart(charts.before_after_histograms(df[c], out, c))
                _download(df.assign(**{f"{c}_{method.replace(' ', '_')}": out.round(4)}), f"{c}_{method}.csv",
                          "Download full data with new column")
            except ValueError as e:
                st.error(str(e))

    with dim:
        if len(num) < 2:
            st.warning("Dimensionality reduction needs at least 2 numeric columns.")
        else:
            st.markdown("**Principal Component Analysis (PCA)**")
            st.caption("PCA combines correlated columns into fewer new ones ('components') that keep as much of the variation as possible.")
            chosen = st.multiselect("Numeric columns", num, default=num, key="pca_cols")
            if len(chosen) >= 2:
                k = st.slider("Number of components", 1, len(chosen), min(2, len(chosen)), key="pca_k")
                variance, loadings, scores = d.run_pca(df, chosen, k)
                left, right = st.columns(2)
                with left:
                    st.markdown("**Variance explained**")
                    st.dataframe(variance, hide_index=True, use_container_width=True)
                with right:
                    st.markdown("**Loadings** (how much each column feeds each component)")
                    st.dataframe(loadings, use_container_width=True)
                _chart(charts.bar_chart(variance, "Component", "Explained variance (%)", "Variance explained by each component"))
                if k >= 2:
                    color = st.selectbox("Color points by (optional)", ["(none)"] + cat, key="pca_color")
                    plot = scores.copy()
                    if color != "(none)":
                        plot[color] = df.loc[plot.index, color]
                    _chart(charts.scatter(plot, "PC1", "PC2", None if color == "(none)" else color))
                _download(pd.concat([df, scores], axis=1), "pca_components.csv", "Download data with component scores")
