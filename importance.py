"""
Feature importance for a target variable (DV): which inputs (IVs) explain it.
  - target balance (class imbalance)
  - correlation with the target (two-outcome target turned into 1/0, as in the lesson)
  - random forest Gini importance and permutation importance (measured on held-out rows)
  - decision tree importance
  - Boruta: compares each real column with shuffled 'shadow' copies of all columns over many
    random-forest runs; a column is Confirmed if it beats the best shadow far more often than
    chance, Rejected if far less often, Tentative otherwise. (Written here directly, because the
    boruta package does not work with current NumPy.)
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42


def is_classification(y: pd.Series, kind: str) -> bool:
    return kind == "categorical" or y.nunique() <= 10 and not pd.api.types.is_float_dtype(y)


def target_balance(y: pd.Series) -> dict:
    counts = y.astype(str).value_counts()
    share = counts / counts.sum()
    ratio = counts.iloc[0] / counts.iloc[-1] if counts.iloc[-1] else np.inf
    level = "severe" if share.iloc[-1] < 0.1 else "moderate" if share.iloc[-1] < 0.3 else "balanced"
    return {"table": pd.DataFrame({"Value": counts.index, "Rows": counts.values, "Share (%)": (share.values * 100).round(1)}),
            "ratio": float(ratio), "level": level, "minority": counts.index[-1]}


def encode(df: pd.DataFrame, target: str, ivs: list, profile: dict, max_levels: int = 30):
    """Numbers stay numbers; categories become integer codes (like .astype('category').cat.codes)."""
    X = pd.DataFrame(index=df.index)
    skipped = []
    for c in ivs:
        t = profile.get(c)
        if t == "numeric":
            X[c] = pd.to_numeric(df[c], errors="coerce")
        elif t == "categorical" and df[c].nunique() <= max_levels:
            X[c] = df[c].astype("category").cat.codes
        elif t == "datetime":
            d = pd.to_datetime(df[c], errors="coerce")
            X[f"{c} (month)"] = d.dt.month
            X[f"{c} (day of week)"] = d.dt.dayofweek
        else:
            skipped.append(c)
    X = X.fillna(X.median(numeric_only=True))
    y = df[target]
    return X, y, skipped


def target_correlation(df: pd.DataFrame, target: str, ivs: list, profile: dict, positive: str | None) -> pd.DataFrame:
    """Correlation matrix of the number columns plus the target (two-outcome target as 1/0)."""
    d = pd.DataFrame(index=df.index)
    for c in ivs:
        if profile.get(c) == "numeric":
            d[c] = pd.to_numeric(df[c], errors="coerce")
        elif profile.get(c) == "categorical" and df[c].nunique() == 2:
            vals = sorted(df[c].astype(str).unique())
            pos = "Yes" if "Yes" in vals else vals[-1]
            d[f"{c} (= {pos})"] = (df[c].astype(str) == pos).astype(int)
    if positive is not None:
        d[f"{target} (= {positive})"] = (df[target].astype(str) == positive).astype(int)
    elif profile.get(target) == "numeric":
        d[target] = pd.to_numeric(df[target], errors="coerce")
    return d.corr().round(3)


def model_importance(X: pd.DataFrame, y: pd.Series, classify: bool) -> pd.DataFrame:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED,
                                          stratify=y if classify and y.value_counts().min() >= 2 else None)
    if classify:
        rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced")
        tree = DecisionTreeClassifier(max_depth=5, random_state=SEED, class_weight="balanced")
        scoring = "roc_auc" if y.nunique() == 2 else "balanced_accuracy"
    else:
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        tree = DecisionTreeRegressor(max_depth=5, random_state=SEED)
        scoring = "r2"
    rf.fit(Xtr, ytr)
    tree.fit(Xtr, ytr)
    perm = permutation_importance(rf, Xte, yte, n_repeats=8, random_state=SEED, scoring=scoring, n_jobs=-1)
    out = pd.DataFrame({
        "Column": X.columns,
        "Random forest (Gini) %": (rf.feature_importances_ * 100).round(1),
        "Decision tree %": (tree.feature_importances_ * 100).round(1),
        "Permutation (drop in score)": perm.importances_mean.round(4),
    })
    score = rf.score(Xte, yte) if not classify else None
    return out.sort_values("Random forest (Gini) %", ascending=False).reset_index(drop=True), score


def boruta(X: pd.DataFrame, y: pd.Series, classify: bool, max_iter: int = 50, alpha: float = 0.05,
           progress=None) -> pd.DataFrame:
    """Boruta feature selection with shadow features and a two-sided binomial test."""
    rng = np.random.default_rng(SEED)
    n_feat = X.shape[1]
    hits = np.zeros(n_feat)
    decided = np.zeros(n_feat, dtype=int)            # 1 confirmed, -1 rejected, 0 tentative
    first_decided = np.full(n_feat, -1)
    Model = RandomForestClassifier if classify else RandomForestRegressor
    Xv = X.values.astype(float)
    sample = None
    if len(X) > 5000:                               # keep it quick on large files
        sample = rng.choice(len(X), 5000, replace=False)
        Xv, yv = Xv[sample], np.asarray(y)[sample]
    else:
        yv = np.asarray(y)
    it = 0
    for it in range(1, max_iter + 1):
        active = np.where(decided != -1)[0]
        shadow = Xv[:, active].copy()
        for j in range(shadow.shape[1]):
            rng.shuffle(shadow[:, j])
        model = Model(n_estimators=100, max_depth=5, random_state=int(rng.integers(1e9)), n_jobs=-1,
                      **({"class_weight": "balanced"} if classify else {}))
        model.fit(np.hstack([Xv[:, active], shadow]), yv)
        imp = model.feature_importances_
        real, shad = imp[:len(active)], imp[len(active):]
        hits[active[real > shad.max()]] += 1
        for k in np.where(decided == 0)[0]:
            p_hi = stats.binom.sf(hits[k] - 1, it, 0.5)       # P(at least this many hits by chance)
            p_lo = stats.binom.cdf(hits[k], it, 0.5)
            corr = alpha / n_feat                               # Bonferroni, as in the original method
            if p_hi < corr:
                decided[k], first_decided[k] = 1, it
            elif p_lo < corr:
                decided[k], first_decided[k] = -1, it
        if progress:
            progress(it / max_iter)
        if (decided != 0).all():
            break
    status = np.where(decided == 1, "Confirmed", np.where(decided == -1, "Rejected", "Tentative"))
    out = pd.DataFrame({"Column": X.columns, "Decision": status, "Hits": hits.astype(int),
                        "Hit rate (%)": (hits / max(it, 1) * 100).round(0)})
    order = {"Confirmed": 0, "Tentative": 1, "Rejected": 2}
    out = out.sort_values(["Decision", "Hits"], key=lambda s: s.map(order) if s.name == "Decision" else -s).reset_index(drop=True)
    # rank like BorutaPy: confirmed = 1, then tentative = 2, then rejected ranked by hits
    ranks, r = [], 1
    rej_rank = 3
    for _, row in out.iterrows():
        if row["Decision"] == "Confirmed":
            ranks.append(1)
        elif row["Decision"] == "Tentative":
            ranks.append(2)
        else:
            ranks.append(rej_rank)
            rej_rank += 1
    out.insert(1, "Rank", ranks)
    out.attrs["iterations"] = it
    return out
