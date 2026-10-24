"""
Text analysis for free-text columns (comments, reviews, survey answers, notes).
  - word and two-word phrase counts (common filler words removed)
  - sentiment: VADER, a rule-based scorer built for short informal text. Each comment gets a
    'compound' score from -1 (very negative) to +1 (very positive); the usual cut-offs are
    >= 0.05 positive, <= -0.05 negative, otherwise neutral.
VADER does not understand sarcasm, domain jargon or languages other than English, so scores are
a first pass to sort and count comments, not a final judgment.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

STOP = set("""a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when where which while who whom
why will with would you your yours yourself yourselves im ive dont didnt its it's i'm i've don't didn't also get got
one would really much even still us""".split())

_analyzer = None


def _vader():
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def text_columns(df: pd.DataFrame, profile: dict) -> list:
    """Columns that hold sentences or phrases: profiled as text, or categories whose values average 3+ words."""
    out = []
    for c, t in profile.items():
        if t not in ("text", "categorical") or c not in df.columns:
            continue
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        words = s.head(500).str.split().str.len().mean()
        if words >= 3:
            out.append(c)
    return out


def tokens(text: str) -> list:
    return [w for w in re.findall(r"[a-z][a-z']+", str(text).lower()) if w not in STOP and len(w) > 2]


def word_counts(series: pd.Series, top: int = 20) -> pd.DataFrame:
    c = Counter()
    for t in series.dropna():
        c.update(tokens(t))
    return pd.DataFrame(c.most_common(top), columns=["Word", "Count"])


def phrase_counts(series: pd.Series, top: int = 15) -> pd.DataFrame:
    c = Counter()
    for t in series.dropna():
        w = tokens(t)
        c.update(" ".join(p) for p in zip(w, w[1:]))
    rows = [(p, n) for p, n in c.most_common(top) if n > 1]
    return pd.DataFrame(rows, columns=["Phrase", "Count"])


def label(score: float) -> str:
    return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"


def sentiment(series: pd.Series) -> pd.DataFrame:
    """One row per comment: compound score and Positive / Neutral / Negative label."""
    an = _vader()
    text = series.fillna("").astype(str)
    scores = [an.polarity_scores(t)["compound"] if t.strip() else np.nan for t in text]
    out = pd.DataFrame({"sentiment_score": scores}, index=series.index)
    out["sentiment"] = out["sentiment_score"].map(lambda s: label(s) if pd.notna(s) else "No text")
    out["word_count"] = text.str.split().str.len()
    return out


def by_group(scored: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    d = scored.assign(_g=groups.values)
    d = d[d["sentiment"] != "No text"]
    g = d.groupby("_g", observed=True)
    out = pd.DataFrame({
        "Comments": g.size(),
        "Average score": g["sentiment_score"].mean().round(3),
        "% positive": (g["sentiment"].apply(lambda s: (s == "Positive").mean()) * 100).round(1),
        "% negative": (g["sentiment"].apply(lambda s: (s == "Negative").mean()) * 100).round(1),
    })
    out.index.name = groups.name
    return out.sort_values("Average score").reset_index()


def words_by_sentiment(series: pd.Series, scored: pd.DataFrame, top: int = 12) -> dict:
    return {lab: word_counts(series[scored["sentiment"] == lab], top) for lab in ("Positive", "Negative")}
