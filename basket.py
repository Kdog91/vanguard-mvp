"""
Items bought together (market basket analysis, association rules) for cross-sell and up-sell.

Data: one row per item in an order, with an order (or customer) column and an item column.
For each pair of items:
  support    = share of orders containing both
  confidence = of the orders with A, the share that also have B   (P(B | A))
  lift       = confidence / share of all orders with B. Lift above 1 means A and B appear together
               more often than chance; lift near 1 means no link.
These are counts of what happened together, not proof that one purchase causes the other.
"""
from itertools import combinations

import pandas as pd


def baskets(df: pd.DataFrame, order_col: str, item_col: str) -> pd.Series:
    d = df[[order_col, item_col]].dropna()
    return d.groupby(order_col)[item_col].agg(lambda s: frozenset(map(str, s)))


def rules(df: pd.DataFrame, order_col: str, item_col: str, min_orders: int = 5, max_items: int = 300) -> dict:
    b = baskets(df, order_col, item_col)
    n = len(b)
    if n < 10:
        raise ValueError(f"Only {n} orders found. This needs at least 10 orders, each with one or more items.")
    item_n = pd.Series([i for s in b for i in s]).value_counts()
    multi = int((b.map(len) >= 2).sum())
    if multi == 0:
        raise ValueError(f"No order in '{order_col}' has two or more different values of '{item_col}'. "
                         "Pick the column that groups items into one order or basket.")
    keep = set(item_n.head(max_items).index)
    pairs = {}
    for s in b:
        items = sorted(s & keep)
        for a, c in combinations(items, 2):
            pairs[(a, c)] = pairs.get((a, c), 0) + 1
    rows = []
    for (a, c), together in pairs.items():
        if together < min_orders:
            continue
        for x, y in ((a, c), (c, a)):
            conf = together / item_n[x]
            rows.append({"If an order has": x, "it also has": y, "Orders with both": together,
                         "Support (% of orders)": round(together / n * 100, 2),
                         "Confidence (%)": round(conf * 100, 1),
                         "Lift": round(conf / (item_n[y] / n), 2)})
    table = pd.DataFrame(rows, columns=["If an order has", "it also has", "Orders with both", "Support (% of orders)",
                                        "Confidence (%)", "Lift"])
    if not table.empty:
        table = table.sort_values(["Lift", "Orders with both"], ascending=False).reset_index(drop=True)
    items = pd.DataFrame({"Item": item_n.index, "Orders": item_n.values,
                          "% of orders": (item_n.values / n * 100).round(1)})
    return {"rules": table, "items": items, "orders": n, "multi_item_orders": multi,
            "avg_items": round(float(b.map(len).mean()), 2)}


def recommend(rules_table: pd.DataFrame, item: str, top: int = 5, min_lift: float = 1.2) -> pd.DataFrame:
    """Items to suggest with `item`: bought with it clearly more often than chance (lift >= min_lift)."""
    r = rules_table[(rules_table["If an order has"] == item) & (rules_table["Lift"] >= min_lift)]
    return r.sort_values(["Confidence (%)", "Lift"], ascending=False).head(top).reset_index(drop=True)


def guess_columns(df: pd.DataFrame):
    """Best guess for the order column (repeats, many groups) and the item column (fewer values)."""
    order_hint = ("order", "invoice", "transaction", "basket", "receipt", "cart", "ticket")
    item_hint = ("product", "item", "sku", "description", "service", "category")
    cols = [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
    order = next((c for c in cols if any(h in c.lower() for h in order_hint)), None)
    if order is None:
        order = next((c for c in cols if "customer" in c.lower()), cols[0] if cols else None)
    item = next((c for c in cols if c != order and any(h in c.lower() for h in item_hint)), None)
    if item is None:
        item = next((c for c in cols if c != order), None)
    return order, item
