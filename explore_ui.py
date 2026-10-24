"""
Streamlit screens for:
  - Explore & describe  (one variable / two variables / many variables)
  - Prepare & transform (data quality, scale types & conversion, rescaling,
                         transformations, dimensionality reduction)
"""
import numpy as np
import pandas as pd
import streamlit as st

import charts
import descriptive as d
import wrangle_ui
import recommend


def _pick_chart(options, rec, reason, key):
    """Chart-type picker that opens on the recommended chart and says why."""
    choice = st.radio("Chart type", options, index=options.index(rec), horizontal=True, key=key,
                      format_func=lambda o: f"{o} (recommended)" if o == rec else o)
    st.caption(f"Recommended: **{rec}**, because {reason}.")
    return choice


def _num(profile):
    return [c for c, t in profile.items() if t == "numeric"]


def _cat(profile):
    return [c for c, t in profile.items() if t == "categorical"]


def _txt(profile):
    return [c for c, t in profile.items() if t == "text"]


def _chart(c):
    st.altair_chart(c, width="stretch")


def _download(df: pd.DataFrame, filename: str, label: str = "Download as CSV"):
    """Download button whose file is named after the dataset and the table, e.g.
    Department_of_Defense_2025-10-12_to_2026-10-12_award_amount_statistics.csv"""
    name = f"{st.session_state.get('dataset_label', 'data')}_{filename}"
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv",
                       key=f"dl::{name}::{label}")


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
                grouped = kind == "numeric" and df[col].nunique() > 15
                method = "Equal-width ranges"
                if grouped:
                    auto_m, auto_why = d.choose_binning(df[col])
                    pick = st.selectbox("Group values by", d.BIN_METHODS, key="uni_bins",
                                        format_func=lambda m: f"Auto ({auto_m.lower()})" if m == "Auto" else m,
                                        help="Equal-width: ranges of the same size. Equal-count: each group holds "
                                             "about the same number of rows. Log-scale: ranges grow by powers of ten, "
                                             "best when a few values are huge.")
                    method = auto_m if pick == "Auto" else pick
                    if method.startswith("Log") and df[col].min() < 0:
                        st.caption("Log-scale ranges need values of zero or more; using equal-count groups instead.")
                        method = "Equal-count groups (quantiles)"
                freq = d.frequency_table(df[col], kind, method=method)
                st.dataframe(freq, hide_index=True, width="stretch")
                _download(freq, f"{col}_frequency_table.csv", "Download frequency table")
                if grouped:
                    if pick == "Auto":
                        st.caption(f"Grouped as **{auto_m.lower()}** because {auto_why}.")
                    elif method.startswith("Equal-width"):
                        st.caption("Values grouped into equal-width ranges (Sturges' rule for the number of ranges).")
                    elif method.startswith("Equal-count"):
                        st.caption("Each group holds about the same number of rows, so the ranges differ in width.")
                    else:
                        st.caption("Ranges grow by powers of ten (100, 1,000, 10,000...). Zeros get their own row.")
            with right:
                st.markdown("**Statistical measures**")
                stats_df = d.summary_stats(df[col]) if kind == "numeric" else d.categorical_stats(df[col])
                if kind == "numeric":
                    stats_df["Value"] = stats_df["Value"].map(lambda v: f"{v:,.3f}" if pd.notna(v) else "")
                else:
                    stats_df["Value"] = stats_df["Value"].astype(str)
                st.dataframe(stats_df, hide_index=True, width="stretch")
                _download(stats_df, f"{col}_statistics.csv", "Download statistics")

            st.markdown("**Charts**")
            if kind == "numeric":
                kinds = ["Histogram", "Box plot", "Density (KDE)"]
                rec, why = recommend.one_numeric(df[col])
            else:
                kinds = ["Bar chart", "Pie chart"]
                rec, why = recommend.one_categorical(df[col])
                if stype == "ordinal":
                    rec, why = "Bar chart", "the categories have a natural order, which bars keep and a pie loses"
            choice = _pick_chart(kinds, rec, why, f"uni_chart_{col}")
            plot_df, plot_col = df, col
            if kind == "numeric" and choice in ("Histogram", "Box plot") and df[col].min() >= 0 and df[col].nunique() > 2:
                skew = float(df[col].skew())
                use_log = st.checkbox("Log scale", value=skew > 3, key=f"uni_log_{col}",
                                      help="Plots log10(value + 1), so each step is 10 times the last: "
                                           "3 = about 1 thousand, 6 = about 1 million, 9 = about 1 billion.")
                if use_log:
                    plot_col = f"log10({col} + 1)"
                    plot_df = pd.DataFrame({plot_col: np.log10(df[col].astype(float) + 1)})
                    st.caption(f"Log scale is on because a few very large values (skewness {skew:.1f}) would otherwise "
                               "squash everything else flat. Read the axis as powers of ten: 3 is about 1 thousand, "
                               "6 about 1 million, 9 about 1 billion. The tables above are in the original units.")
            if choice == "Histogram":
                _chart(charts.histogram(plot_df, plot_col))
            elif choice == "Box plot":
                _chart(charts.box_plot(plot_df, plot_col))
            elif choice == "Density (KDE)":
                try:
                    nc = d.normality_check(df[col])
                    _chart(charts.kde_chart(df[col], col))
                    verdict = ("consistent with a normal distribution" if nc["looks_normal"]
                               else "not normally distributed")
                    st.write(f"**{verdict.capitalize()}** ({nc['test']} test, p {'< 0.0001' if nc['p_value'] < 0.0001 else '= ' + format(nc['p_value'], '.4f')}). "
                             f"The shape is {nc['shape']}: skewness {nc['skewness']:.2f}, "
                             f"excess kurtosis {nc['kurtosis']:.2f}.")
                    st.caption("A KDE is a smoothed histogram. The dashed line is a normal (bell) curve with the same "
                               "mean and spread. p below 0.05 means the data differs from a normal curve by more than "
                               "chance would explain; with thousands of rows even small differences give a low p.")
                except ValueError as e:
                    st.info(str(e))
            elif choice == "Bar chart":
                wide = df[col].nunique() > 8
                bars = freq.sort_values("Absolute frequency", ascending=False) if wide and stype == "nominal" else freq
                _chart(charts.bar_chart(bars, "Value", "Absolute frequency", f"Frequency of {col}", horizontal=wide))
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
                st.dataframe(m, hide_index=True, width="stretch")
                _download(m, f"{a}_vs_{b}_relationship.csv", "Download relationship measures")
                rec, why = recommend.two_numeric(df, a, b)
                choice = _pick_chart(["Scatter plot", "Line chart", "Area chart"], rec, why, f"bi_chart_{a}_{b}")
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
                st.dataframe(gs, hide_index=True, width="stretch")
                _download(gs, f"{num_col}_by_{cat_col}.csv")
                opts = ["Box plot by group", f"Bar chart of average {num_col}"]
                rec_kind, why = recommend.numeric_by_category(df, num_col, cat_col)
                choice = _pick_chart(opts, opts[0] if rec_kind == "Box" else opts[1], why, f"bi_chart2_{num_col}_{cat_col}")
                if choice.startswith("Box"):
                    _chart(charts.box_plot(df, num_col, cat_col, d._sorted_categories(df[cat_col].dropna())))
                else:
                    _chart(charts.bar_chart(gs, cat_col, "Mean", f"Average {num_col} by {cat_col}"))

            else:
                st.markdown("**Cross-tabulation (two-way frequency table)**")
                view = st.radio("Show", ["Counts", "Row %", "Column %", "% of total"], horizontal=True, key="bi_ct")
                norm = {"Counts": None, "Row %": "index", "Column %": "columns", "% of total": "all"}[view]
                ct = d.crosstab(df, a, b, norm)
                st.dataframe(ct, width="stretch")
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
                st.dataframe(d.covariance_matrix(df, num), width="stretch")
            with st.expander("Summary statistics for every numeric column"):
                st.dataframe(df[num].describe().T.round(2), width="stretch")
        else:
            st.info("Correlation matrix needs at least 2 numeric columns.")

        if len(cat) >= 2:
            st.markdown("**Multivariate frequencies (combinations of categories)**")
            chosen = st.multiselect("Categorical columns to combine", cat, default=cat[:2], key="mv_cats")
            if len(chosen) >= 2:
                mf = d.multivariate_frequency(df, chosen)
                st.dataframe(mf, hide_index=True, width="stretch")
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
    q, miss, flt, edit, scale, resc, trans, dim = st.tabs(
        ["Data quality", "Missing values", "Filter rows", "Edit columns", "Scale types & conversion",
         "Rescale / normalize", "Transform", "Dimensionality reduction"])
    with miss:
        wrangle_ui.render_missing(raw_df)
    with flt:
        wrangle_ui.render_filter(df, profile)
    with edit:
        wrangle_ui.render_columns(df, profile)
    num, cat = _num(profile), _cat(profile)

    with q:
        st.markdown("**Data quality report**")
        rep = d.data_quality_report(raw_df, df, profile)
        st.dataframe(rep, hide_index=True, width="stretch")
        st.caption("'Missing in original' is before cleaning; the cleaned data has those gaps filled (see the cleaning report above). "
                   "Outliers use the same 1.5 × IQR rule as the box plots.")
        _download(rep, "data_quality_report.csv")

    with scale:
        st.markdown("**Scale types**")
        st.dataframe(d.scale_type_table(df, profile), hide_index=True, width="stretch")
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
                st.dataframe(out.head(20), width="stretch")
                st.dataframe(d.frequency_table(out[f"{c}_category"], "categorical"), hide_index=True)
                _download(df.assign(**{f"{c}_category": out[f"{c}_category"]}), f"{c}_binned.csv", "Download full data with new column")
        elif conv.startswith("Nominal"):
            if not cat:
                st.warning("No categorical columns.")
            else:
                c = st.selectbox("Categorical column", cat, key="oh_c")
                oh = d.one_hot(df, c)
                st.dataframe(pd.concat([df[[c]], oh], axis=1).head(20), width="stretch")
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
                st.dataframe(cmp, width="stretch")
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
            method = st.radio("Transformation", ["log", "square root", "Box-Cox", "Yeo-Johnson", "Quantile → normal", "square"],
                              horizontal=True, key="tr_m",
                              help="Log, square root, Box-Cox and Yeo-Johnson pull in a long tail (reduce skew). "
                                   "Quantile → normal maps the values onto a bell curve. Square stretches the tail.")
            try:
                out = d.transform(df[c], method)
                sk = pd.DataFrame([d.normality(df[c]), d.normality(out)], index=["Before", "After"])
                st.dataframe(sk, width="stretch")
                st.caption("Skewness and excess kurtosis near 0, and a normality p-value above 0.05, mean the shape is "
                           "close to a Gaussian (normal) bell curve.")
                _chart(charts.before_after_histograms(df[c], out, c))
                _download(df.assign(**{f"{c}_{method.replace(' ', '_')}": out.round(4)}), f"{c}_{method}.csv",
                          "Download full data with new column")
            except ValueError as e:
                st.error(str(e))

        if num:
            st.markdown("**Map numeric columns to a Gaussian (normal) distribution**")
            st.caption("Many models (linear and logistic regression, SVM, neural networks) work best when inputs are "
                       "roughly bell-shaped. This maps every chosen column at once and shows the shape before and after.")
            g1, g2 = st.columns([2, 1])
            gcols = g1.multiselect("Columns", num, default=num, key="ga_cols")
            gm = g2.selectbox("Method", d.GAUSSIAN_METHODS, key="ga_m",
                              help="Yeo-Johnson works with any values. Quantile → normal forces a bell shape from the "
                                   "ranks. Box-Cox needs positive values (columns with 0 or less use Yeo-Johnson).")
            if gcols:
                # 1) normality test: which columns are not normal
                nt = pd.DataFrame([{"Column": c, **d.normality(df[c])} for c in gcols])
                nt["Normal? (p > 0.05)"] = nt["Normality p-value"].map(lambda p: "yes" if p > 0.05 else "no")
                st.markdown("**Normality test** (D'Agostino K²)")
                st.dataframe(nt, hide_index=True, width="stretch")
                # 2) fit on training rows, apply the same parameters to test rows
                test_pct = st.slider("Hold back this % of rows as evaluation (test) data", 10, 50, 30, key="ga_test")
                train = df.sample(frac=1 - test_pct / 100, random_state=42)
                test = df.drop(train.index)
                fitted = d.fit_gaussian(train, gcols, gm)
                tr_out, te_out = d.apply_gaussian(train, fitted), d.apply_gaussian(test, fitted)
                gtab = pd.DataFrame([{"Column": c, "Method": fitted[c][0],
                                      "Skew before": d.normality(df[c])["Skewness"],
                                      "Skew after (training)": d.normality(tr_out[c])["Skewness"],
                                      "Skew after (test, same parameters)": d.normality(te_out[c])["Skewness"],
                                      "Normality p after (test)": d.normality(te_out[c])["Normality p-value"]}
                                     for c in fitted])
                st.markdown(f"**Mapped with parameters learned on {len(train):,} training rows, then applied to "
                            f"{len(test):,} test rows**")
                st.dataframe(gtab, hide_index=True, width="stretch")
                with st.expander("Fitted parameters (λ for Box-Cox / Yeo-Johnson)"):
                    st.dataframe(d.fitted_params(fitted), hide_index=True, width="stretch")
                st.caption("Parameters come only from the training rows, so the test rows stay unseen, just as new data "
                           "would be when the model is used.")
                # 3) probability density before and after
                if fitted:
                    pc = st.selectbox("Probability density for", list(fitted), key="ga_plot")
                    dd = pd.concat([pd.DataFrame({"value": (df[pc] - df[pc].mean()) / (df[pc].std() or 1), "Data": "Before (standardized)"}),
                                    pd.DataFrame({"value": tr_out[pc], "Data": "After mapping"})]).dropna()
                    import altair as alt
                    st.altair_chart(alt.Chart(dd).transform_density("value", groupby=["Data"], as_=["value", "density"],
                                                                    extent=[-4, 4]).mark_line(strokeWidth=2).encode(
                        x=alt.X("value:Q", title=f"{pc} (standard units)"), y=alt.Y("density:Q", title="Density"),
                        color=alt.Color("Data:N", scale=alt.Scale(range=["#a3a29c", "#2a78d6"]))
                    ).properties(height=260), width="stretch")
                full = d.apply_gaussian(df, fitted)
                _download(full, f"gaussian_{gm.split()[0].lower()}.csv", "Download data with Gaussian-mapped columns")
                # 4) apply to a new evaluation file
                newf = st.file_uploader("Apply the same parameters to a new evaluation file (optional)", key="ga_new",
                                        type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet", "orc"])
                if newf is not None:
                    from data_cleaner import load_file
                    try:
                        nd = load_file(newf)
                        miss = [c for c in fitted if c not in nd.columns]
                        if miss:
                            st.warning("Missing in the new file: " + ", ".join(miss))
                        nout = d.apply_gaussian(nd, fitted)
                        st.dataframe(nout.head(10), width="stretch")
                        _download(nout, f"evaluation_gaussian_{gm.split()[0].lower()}.csv", "Download transformed evaluation file")
                    except Exception as ex:
                        st.error(f"Could not transform the file: {ex}")

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
                    st.dataframe(variance, hide_index=True, width="stretch")
                with right:
                    st.markdown("**Loadings** (how much each column feeds each component)")
                    st.dataframe(loadings, width="stretch")
                _chart(charts.bar_chart(variance, "Component", "Explained variance (%)", "Variance explained by each component"))
                if k >= 2:
                    color = st.selectbox("Color points by (optional)", ["(none)"] + cat, key="pca_color")
                    plot = scores.copy()
                    if color != "(none)":
                        plot[color] = df.loc[plot.index, color]
                    _chart(charts.scatter(plot, "PC1", "PC2", None if color == "(none)" else color))
                _download(pd.concat([df, scores], axis=1), "pca_components.csv", "Download data with component scores")
