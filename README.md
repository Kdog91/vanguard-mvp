# Vanguard Analytics

Turn a raw data file into tested answers, without writing code. Upload a table from any field
(sales, health, operations, finance, public records) and the app cleans it, describes it,
builds tested predictions and forecasts, and writes a shareable report.

**Live demo:** https://vanguard-mvp-32gx8uhtqp3yelxnwahjdu.streamlit.app
Built by Kevin Simmons, Vanguard Data Analytics LLC.

## What it does

| Step | What happens |
|---|---|
| **Clean** | Detects column types, strips currency and percent text, removes duplicates and ID columns, fills or sets aside missing values, flags (never deletes) outliers, and keeps codes such as ZIP or NAICS as categories. Every change is listed in a cleaning report. |
| **Describe** | Frequency tables (equal-width, equal-count or log-scale grouping, picked automatically for lopsided data), summary statistics including mean absolute deviation, cross-tabulations, correlation, and charts chosen to fit each column. |
| **Prepare** | Data quality report, scale types, binning, one-hot encoding, normalization, transforms and PCA. |
| **Predict** | Compares regression or classification models on held-out data with 5-fold cross-validation and reports the measured error. |
| **Forecast** | Tests seven forecasting methods on your most recent periods and forecasts with the best, with a rough range. |
| **Find patterns** | Clustering, anomaly detection and significance tests (t-test, ANOVA, chi-square, correlation). |
| **Combine sources** | Makes measurements from different sites, stores, labs or methods comparable (robust z-scores per series), compares each reading with the other series in the same period, flags outliers, reporting gaps and step changes (such as a lab switch), combines them into one weighted index, and forecasts that index against a naive baseline. |
| **Report** | One click builds a Word report of what was loaded, what the cleaning changed, and a plain-language description of every column with charts. |

**Inputs:** CSV, TSV, TXT, Excel, JSON and Parquet. Large-file mode (DuckDB) scans files with millions
of rows on one computer. A built-in connector also pulls public U.S. federal contract awards from USAspending.gov.

## Sample data (synthetic)

| Sample | Try |
|---|---|
| Retail sales by store | Forecast weekly revenue; compare stores; Combine sources by store |
| Clinic visits | Forecast visits; test whether wait time relates to satisfaction |
| Multi-site measurements | Combine & standardize sources (finds the planted bad readings, gaps and lab change) |
| Government contracts | Predict total cost or risk level |
| Contract awards over time | Forecast monthly awards |

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Limits (honest)

- Works on **tables** (one row per record, one column per field). Free text is only counted, not understood;
  images, audio and scanned PDFs are not supported.
- One table at a time; joining related tables is on the roadmap.
- No user accounts or security controls yet, so do not upload sensitive or protected data to the public demo.
- Results are decision support; a person who knows the field should check them.

## Roadmap

Messy-spreadsheet import (title rows, merged cells), joining related tables, database connections,
tables inside PDFs, text analysis, maps, and user accounts.

## Code layout

- `app.py`: Streamlit app
- `data_cleaner.py`: loading, type detection and cleaning
- `descriptive.py`, `explore_ui.py`, `wrangle_ui.py`: description and preparation
- `model_selector.py`, `forecasting.py`, `unsupervised.py`, `statistical_tests.py`: models
- `standardize.py`, `standardize_ui.py`: multi-source standardization and index forecast
- `report.py`: Word report
- `bigdata.py`: large-file mode (DuckDB)
- `usaspending.py`: USAspending.gov connector
