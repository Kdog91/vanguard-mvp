"""
Time-series forecasting.

Turns a date column + a numeric column into a regular series (one value per
day / week / month / quarter / year), then compares several forecasting methods
on the most recent periods, which are held back and never shown to the models.
The method with the lowest error on that held-back stretch is refit on all the
data and used to forecast forward.

Methods compared: Naive (repeat last value), Seasonal naive (repeat the same
period last year), Moving average, Linear trend, Holt (level + trend),
Holt-Winters (level + trend + seasonality), ARIMA.
"""
import warnings
import numpy as np
import pandas as pd

FREQS = {
    "Day": ("D", 7),        # label: (pandas frequency, periods in one seasonal cycle)
    "Week": ("W", 52),
    "Month": ("MS", 12),
    "Quarter": ("QS", 4),
    "Year": ("YS", 1),
}
MIN_PERIODS = 12


def suggest_frequency(dates: pd.Series) -> str:
    """Pick the finest period that still leaves a readable number of points."""
    d = pd.to_datetime(dates, errors="coerce").dropna()
    if d.empty:
        return "Month"
    span_days = (d.max() - d.min()).days
    if span_days <= 120:
        return "Day"
    if span_days <= 730:
        return "Week"
    if span_days <= 365 * 15:
        return "Month"
    return "Year"


def prepare_series(df: pd.DataFrame, date_col: str, value_col: str, freq_label: str, agg: str = "sum") -> pd.Series:
    """One value per period. Periods with no rows become 0 for totals/counts, or are interpolated for averages."""
    freq, _ = FREQS[freq_label]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    data = pd.DataFrame({"d": dates, "v": values}).dropna()
    if data.empty:
        raise ValueError("No rows have both a valid date and a valid number.")
    grouped = data.set_index("d")["v"].resample(freq)
    if agg == "sum":
        s = grouped.sum()
    elif agg == "count":
        s = grouped.count().astype(float)
    else:
        s = grouped.mean().interpolate(limit_direction="both")
    s.name = value_col
    if len(s) < MIN_PERIODS:
        raise ValueError(
            f"Only {len(s)} {freq_label.lower()} periods of data. Forecasting needs at least {MIN_PERIODS}. "
            "Try a shorter period (for example Week instead of Month).")
    return s


def series_from_daily(daily: pd.DataFrame, value_col: str, freq_label: str, agg: str = "sum") -> pd.Series:
    """Same result as prepare_series, but built from per-day totals (date, total, n) computed over a large file."""
    freq, _ = FREQS[freq_label]
    d = daily.set_index(pd.to_datetime(daily["d"]))
    total, n = d["total"].resample(freq).sum(), d["n"].resample(freq).sum()
    if agg == "sum":
        s = total
    elif agg == "count":
        s = n.astype(float)
    else:
        s = (total / n.where(n > 0)).interpolate(limit_direction="both")
    s.name = value_col
    if len(s) < MIN_PERIODS:
        raise ValueError(
            f"Only {len(s)} {freq_label.lower()} periods of data. Forecasting needs at least {MIN_PERIODS}. "
            "Try a shorter period (for example Week instead of Month).")
    return s


def _fit_predict(name: str, train: pd.Series, h: int, m: int) -> np.ndarray:
    """Fit one method on `train` and return h future values."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    y = train.values.astype(float)
    n = len(y)
    if name == "Naive (repeat last value)":
        return np.repeat(y[-1], h)
    if name == "Seasonal naive (same period last cycle)":
        return np.array([y[n - m + (i % m)] for i in range(h)])
    if name == "Moving average (last 3 periods)":
        return np.repeat(y[-3:].mean(), h)
    if name == "Linear trend":
        slope, intercept = np.polyfit(np.arange(n), y, 1)
        return intercept + slope * np.arange(n, n + h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "Holt (level + trend)":
            return ExponentialSmoothing(y, trend="add", damped_trend=True).fit().forecast(h)
        if name == "Holt-Winters (trend + seasonality)":
            return ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=m).fit().forecast(h)
        if name == "ARIMA":
            best, best_aic = None, np.inf
            for order in [(1, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 1), (1, 0, 0), (2, 1, 2)]:
                try:
                    fit = ARIMA(y, order=order).fit()
                    if fit.aic < best_aic:
                        best, best_aic = fit, fit.aic
                except Exception:
                    continue
            if best is None:
                raise RuntimeError("ARIMA could not be fit")
            return best.forecast(h)
    raise ValueError(name)


def run_forecast(series: pd.Series, horizon: int, freq_label: str) -> dict:
    """Compare methods on a held-back recent stretch, then forecast `horizon` periods with the best one."""
    freq, m = FREQS[freq_label]
    n = len(series)
    holdout = int(min(max(horizon, 3), max(3, n // 5)))
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]

    names = ["Naive (repeat last value)", "Moving average (last 3 periods)", "Linear trend",
             "Holt (level + trend)", "ARIMA"]
    seasonal_ok = m > 1 and len(train) >= 2 * m
    if seasonal_ok:
        names += ["Seasonal naive (same period last cycle)", "Holt-Winters (trend + seasonality)"]

    rows, preds = [], {}
    for name in names:
        try:
            p = np.asarray(_fit_predict(name, train, holdout, m), dtype=float)
        except Exception:
            continue
        if not np.isfinite(p).all():
            continue
        err = test.values - p
        nonzero = test.values != 0
        mape = float(np.mean(np.abs(err[nonzero] / test.values[nonzero])) * 100) if nonzero.any() else np.nan
        rows.append({"Method": name, "MAE": float(np.mean(np.abs(err))),
                     "RMSE": float(np.sqrt(np.mean(err ** 2))), "MAPE (%)": mape})
        preds[name] = p
    if not rows:
        raise ValueError("No forecasting method could be fit to this series.")

    comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best = comparison.iloc[0]["Method"]
    best_rmse = float(comparison.iloc[0]["RMSE"])

    future_vals = np.asarray(_fit_predict(best, series, horizon, m), dtype=float)
    future_idx = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
    # Rough range: typical miss on the held-back stretch, widening the further out we go.
    spread = 1.96 * best_rmse * np.sqrt(np.arange(1, horizon + 1) / holdout).clip(min=1.0)
    forecast = pd.DataFrame({"Period": future_idx, "Forecast": future_vals,
                             "Low": future_vals - spread, "High": future_vals + spread})
    if (series.values >= 0).all():
        forecast[["Forecast", "Low", "High"]] = forecast[["Forecast", "Low", "High"]].clip(lower=0)

    backtest = pd.DataFrame({"Period": test.index, "Actual": test.values, "Predicted": preds[best]})
    return {"comparison": comparison, "best_method": best, "holdout": holdout,
            "seasonal_tested": seasonal_ok, "forecast": forecast, "backtest": backtest,
            "history": series, "n_periods": n}
