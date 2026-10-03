"""
Statistical Hypothesis Testing (model family 3G).
Answers "is this pattern statistically real, or just random noise?"
  - Chi-Square Test of Independence (two categorical columns)
  - Chi-Square Goodness-of-Fit (one categorical column vs. uniform)
  - Two-sample Welch's t-test (numeric column, exactly 2 groups)
  - One-way ANOVA (numeric column, 3+ groups)
  - Pearson correlation test (two numeric columns)
"""
import numpy as np
import pandas as pd
from scipy import stats


def _sig(p):
    return p < 0.05


def chi_square_independence(df, col_a, col_b):
    table = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {
        "chi2_statistic": chi2, "p_value": p, "degrees_of_freedom": dof,
        "contingency_table": table, "significant": _sig(p),
        "interpretation": (f"There {'IS' if _sig(p) else 'is NO'} statistically significant relationship "
                           f"between '{col_a}' and '{col_b}' (p={p:.4f})."),
    }


def chi_square_goodness_of_fit(df, col):
    observed = df[col].value_counts().sort_index()
    expected = np.full(len(observed), observed.sum() / len(observed))
    chi2, p = stats.chisquare(observed.values, f_exp=expected)
    return {
        "chi2_statistic": chi2, "p_value": p, "observed": observed, "significant": _sig(p),
        "interpretation": (f"The distribution of '{col}' {'DOES' if _sig(p) else 'does NOT'} differ "
                           f"significantly from an even split across categories (p={p:.4f})."),
    }


def two_sample_ttest(df, numeric_col, group_col):
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"A t-test needs exactly 2 groups in '{group_col}', but found {len(groups)}. "
                         f"Use ANOVA for 3 or more groups.")
    a = df.loc[df[group_col] == groups[0], numeric_col].dropna()
    b = df.loc[df[group_col] == groups[1], numeric_col].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "t_statistic": t, "p_value": p, "significant": _sig(p),
        "group_means": {str(groups[0]): a.mean(), str(groups[1]): b.mean()},
        "interpretation": (f"Average '{numeric_col}' {'IS' if _sig(p) else 'is NOT'} significantly different between "
                           f"'{groups[0]}' ({a.mean():,.2f}) and '{groups[1]}' ({b.mean():,.2f}), p={p:.4f}."),
    }


def one_way_anova(df, numeric_col, group_col):
    groups = df[group_col].dropna().unique()
    if len(groups) < 3:
        raise ValueError(f"ANOVA needs 3 or more groups in '{group_col}', but found {len(groups)}. "
                         f"Use a t-test for 2 groups.")
    samples = [df.loc[df[group_col] == g, numeric_col].dropna() for g in groups]
    f, p = stats.f_oneway(*samples)
    return {
        "f_statistic": f, "p_value": p, "significant": _sig(p),
        "group_means": {str(g): s.mean() for g, s in zip(groups, samples)},
        "interpretation": (f"Average '{numeric_col}' {'DOES' if _sig(p) else 'does NOT'} differ significantly "
                           f"across the {len(groups)} groups in '{group_col}' (p={p:.4f})."),
    }


def correlation_test(df, col_a, col_b):
    pair = df[[col_a, col_b]].dropna()
    r, p = stats.pearsonr(pair[col_a], pair[col_b])
    return {
        "correlation_r": r, "p_value": p, "significant": _sig(p),
        "interpretation": (f"'{col_a}' and '{col_b}' {'ARE' if _sig(p) else 'are NOT'} significantly "
                           f"correlated (r={r:.3f}, p={p:.4f})."),
    }
