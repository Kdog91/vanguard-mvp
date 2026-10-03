"""
Chart builders (Altair, rendered by Streamlit).

Color rules:
  - Single series: one blue. Category colors come in a fixed order and follow the
    category, never its rank.
  - Scatter colors: at most 3 categories (the set that stays colorblind-safe for
    every pair); anything beyond folds into gray "Other".
  - Pie: at most 6 slices, the rest folds into "Other", and every slice is labeled.
  - Correlation heatmap: diverging blue <-> red with a gray midpoint at 0.
  - Every chart has hover tooltips, and every chart sits next to its data table.
"""
import altair as alt
import pandas as pd

BLUE = "#2a78d6"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
OTHER_GRAY = "#a3a29c"
DIVERGING = ["#184f95", "#6da7ec", "#f0efec", "#ef8f8e", "#b52a2a"]  # blue - gray midpoint - red
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
HEIGHT = 320


def _fold_categories(series: pd.Series, max_n: int):
    """Keep the most common categories, fold the rest into 'Other'. Returns (folded series, domain, colors)."""
    top = series.value_counts().index[:max_n].tolist()
    folded = series.where(series.isin(top), "Other").astype(str)
    domain = [str(c) for c in top] + (["Other"] if (folded == "Other").any() else [])
    colors = CATEGORICAL[:len(top)] + ([OTHER_GRAY] if "Other" in domain else [])
    return folded, domain, colors


def histogram(df: pd.DataFrame, col: str, bins: int = 20):
    return alt.Chart(df).mark_bar(color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3, binSpacing=2).encode(
        x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=col),
        y=alt.Y("count():Q", title="Frequency"),
        tooltip=[alt.Tooltip(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=col), alt.Tooltip("count():Q", title="Frequency")],
    ).properties(height=HEIGHT, title=f"Histogram of {col}")


def box_plot(df: pd.DataFrame, value_col: str, group_col: str | None = None, order: list | None = None):
    """Box = middle 50% (Q1-Q3), line = median, whiskers = 1.5 x IQR, dots = outliers (same rule as the cleaner)."""
    enc = {"y": alt.Y(f"{value_col}:Q", title=value_col)}
    if group_col:
        enc["x"] = alt.X(f"{group_col}:N", title=group_col, sort=[str(o) for o in order] if order else "ascending", axis=alt.Axis(labelAngle=0))
    return alt.Chart(df).mark_boxplot(
        extent=1.5, size=40, color=BLUE,
        median={"color": "#0b0b0b", "strokeWidth": 2},
        outliers={"color": "#e34948", "size": 50, "filled": True},
        rule={"color": "#52514e"}, ticks=False,
    ).encode(**enc).properties(
        height=HEIGHT, title=f"Box plot of {value_col}" + (f" by {group_col}" if group_col else "") + " (red dots = outliers)")


def bar_chart(freq: pd.DataFrame, label_col: str, value_col: str, title: str, horizontal: bool = False):
    sort = freq[label_col].tolist()  # keep the table's order (ordinal order or most-frequent first)
    cat = alt.X(f"{label_col}:N", sort=sort, title=label_col) if not horizontal else alt.Y(f"{label_col}:N", sort=sort, title=None)
    val = alt.Y(f"{value_col}:Q", title=value_col) if not horizontal else alt.X(f"{value_col}:Q", title=value_col)
    radius = {"cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3} if not horizontal else {"cornerRadiusTopRight": 3, "cornerRadiusBottomRight": 3}
    bars = alt.Chart(freq).mark_bar(color=BLUE, **radius).encode(
        x=cat if not horizontal else val, y=val if not horizontal else cat,
        tooltip=[alt.Tooltip(f"{label_col}:N"), alt.Tooltip(f"{value_col}:Q", format=",.2f")],
    )
    return bars.properties(height=HEIGHT, title=title)


def pie_chart(series: pd.Series, title: str):
    folded, domain, colors = _fold_categories(series.dropna().astype(str), max_n=6)
    counts = folded.value_counts().reindex(domain).reset_index()
    counts.columns = ["Category", "Count"]
    counts["Percent"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
    counts["Label"] = counts["Category"] + " " + counts["Percent"].astype(str) + "%"
    base = alt.Chart(counts).encode(
        theta=alt.Theta("Count:Q", stack=True),
        color=alt.Color("Category:N", scale=alt.Scale(domain=domain, range=colors), sort=domain, legend=alt.Legend(title=None)),
        order=alt.Order("Count:Q", sort="descending"),
        tooltip=["Category:N", "Count:Q", alt.Tooltip("Percent:Q", title="Percent (%)")],
    )
    arcs = base.mark_arc(outerRadius=120, stroke="#ffffff", strokeWidth=2)
    labels = base.mark_text(radius=155, fontSize=13).encode(text="Label:N", color=alt.value("#0b0b0b"))
    return (arcs + labels).properties(height=HEIGHT + 40, title=title)


def scatter(df: pd.DataFrame, x: str, y: str, color_col: str | None = None):
    data = df.copy()
    enc = {
        "x": alt.X(f"{x}:Q", title=x, scale=alt.Scale(zero=False)),
        "y": alt.Y(f"{y}:Q", title=y, scale=alt.Scale(zero=False)),
        "tooltip": [alt.Tooltip(f"{x}:Q", format=",.2f"), alt.Tooltip(f"{y}:Q", format=",.2f")],
    }
    mark = {"size": 64, "opacity": 0.75, "stroke": "#ffffff", "strokeWidth": 1}
    if color_col:
        data["_group"], domain, colors = _fold_categories(data[color_col].astype(str), max_n=3)
        enc["color"] = alt.Color("_group:N", title=color_col, scale=alt.Scale(domain=domain, range=colors), sort=domain)
        enc["shape"] = alt.Shape("_group:N", title=color_col, sort=domain)  # second cue, so identity isn't color-only
        enc["tooltip"].append(alt.Tooltip(f"{color_col}:N"))
        chart = alt.Chart(data).mark_point(filled=True, **mark).encode(**enc)
    else:
        chart = alt.Chart(data).mark_circle(color=BLUE, **mark).encode(**enc)
    return chart.properties(height=HEIGHT, title=f"{y} vs. {x}").interactive()


def line_or_area(df: pd.DataFrame, x: str, y: str, kind: str = "line", x_type: str = "Q"):
    """Line/area of y across x. If x has repeated values, y is averaged per x so the line is readable."""
    data = df.groupby(x, as_index=False)[y].mean().sort_values(x)
    base = alt.Chart(data).encode(
        x=alt.X(f"{x}:{x_type}", title=x),
        y=alt.Y(f"{y}:Q", title=f"Average {y}" if df[x].duplicated().any() else y),
        tooltip=[alt.Tooltip(f"{x}:{x_type}"), alt.Tooltip(f"{y}:Q", format=",.2f")],
    )
    hover = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False)
    if kind == "area":
        layer = base.mark_area(color=BLUE, opacity=0.25, line={"color": BLUE, "strokeWidth": 2})
    else:
        layer = base.mark_line(color=BLUE, strokeWidth=2)
    points = base.mark_circle(color=BLUE, size=70).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))).add_params(hover)
    rule = alt.Chart(data).mark_rule(color="#a3a29c").encode(x=f"{x}:{x_type}").transform_filter(hover)
    return (layer + points + rule).properties(height=HEIGHT, title=f"{y} across {x}")


def heatmap(matrix: pd.DataFrame, title: str, diverging: bool = True, fmt: str = ".2f"):
    """Correlation (diverging, -1..1) or counts (sequential) as a colored grid with the number in every cell."""
    m = matrix.copy()
    m.index.name, m.columns.name = "Row", None
    long = m.reset_index().melt(id_vars="Row", var_name="Column", value_name="Value")
    order_r, order_c = [str(i) for i in matrix.index], [str(c) for c in matrix.columns]
    long["Row"], long["Column"] = long["Row"].astype(str), long["Column"].astype(str)
    if diverging:
        scale = alt.Scale(domain=[-1, -0.5, 0, 0.5, 1], range=DIVERGING, interpolate="lab")
    else:
        scale = alt.Scale(range=SEQUENTIAL, interpolate="lab")
    cells = alt.Chart(long).mark_rect(stroke="#ffffff", strokeWidth=2, cornerRadius=3).encode(
        x=alt.X("Column:N", sort=order_c, title=None, axis=alt.Axis(labelAngle=0, orient="top", labelLimit=200)),
        y=alt.Y("Row:N", sort=order_r, title=None),
        color=alt.Color("Value:Q", scale=scale, title=None),
        tooltip=["Row:N", "Column:N", alt.Tooltip("Value:Q", format=fmt)],
    )
    # text ink flips to white on the darkest cells so every number stays readable
    dark = "abs(datum.Value) >= 0.6" if diverging else f"datum.Value >= {long['Value'].quantile(0.75)}"
    text = alt.Chart(long).mark_text(fontSize=13).encode(
        x=alt.X("Column:N", sort=order_c), y=alt.Y("Row:N", sort=order_r),
        text=alt.Text("Value:Q", format=fmt),
        color=alt.condition(dark, alt.value("#ffffff"), alt.value("#0b0b0b")),
    )
    return (cells + text).properties(height=max(180, 60 * len(order_r)), title=title)


def before_after_histograms(before: pd.Series, after: pd.Series, name: str):
    a = pd.DataFrame({"value": before.values, "Version": "Before"})
    b = pd.DataFrame({"value": after.values, "Version": "After"})

    def h(d, t):
        return alt.Chart(d).mark_bar(color=BLUE, binSpacing=2, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=20), title=name),
            y=alt.Y("count():Q", title="Frequency"),
            tooltip=[alt.Tooltip("value:Q", bin=alt.Bin(maxbins=20)), "count():Q"],
        ).properties(height=260, width=320, title=t)
    return alt.hconcat(h(a, "Before"), h(b, "After"))
