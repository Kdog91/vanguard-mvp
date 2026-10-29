# Vanguard Analytics

Turn a raw data file into tested answers, without writing code. Upload a table from any field
(sales, health, operations, finance, public records) and the app cleans it, describes it,
builds tested predictions and forecasts, and writes a shareable report.

**Live demo:** https://vanguard-mvp-32gx8uhtqp3yelxnwahjdu.streamlit.app
Built by Kevin Simmons, Vanguard Data Analytics LLC.

## Getting started

After a file loads, **What do you want to know?** offers plain-language goals (for example "Predict next month's numbers" or "See totals by group") that open the right tool, and hides goals the file cannot support. Each tool has a short **How to use this** panel. Everyday tools are listed first; **Show advanced tools** reveals clustering, preparation, pipelines, combining sources and deep learning.

## Sign-in

Sample data and public USAspending data work for everyone. Loading your own data (upload, web address,
database, large file) requires a free account (**Sign in / Sign up** through Auth0: email and password, or Google). Optional `[access]` rules in Secrets limit it to
approved emails or domains. Setup is described at the top of `auth.py`. Without an `[auth]` section the app
runs open, for local development.

## Security

Controls from the Vanguard Analytics Security Outline (update 26f), most in `security.py`:

| Area | Control |
|---|---|
| Web-address imports | http/https only; the host must resolve to public addresses only; every redirect is re-checked; the real peer address is checked after connecting and before anything is sent (blocks 127.0.0.1, 10.x, 172.16-31.x, 192.168.x, 169.254.x cloud metadata, IPv6 local, DNS rebinding) |
| Databases (public app) | Private and internal hosts refused; SQLite files on the server refused |
| Uploads | Allowed types only; 200 MB; 1,000,000 rows; 2,000 columns |
| Sensitive data | Warning on the upload screen and in the Terms of Use: no PHI, CUI, card numbers or SSNs on the public app |
| Downloads | CSV and Excel cells starting with = + - @ are escaped (formula injection); names, emails, phone and ID columns are replaced with codes in downloads and Word reports by default |
| Temporary files | All named `vanguard_*`; removed after 3 hours; cached results leave memory within an hour |
| Audit log | One `AUDIT {...}` JSON line per sign-in, sign-out, load, export and refused request, with the account as a one-way code; no data, file contents, passwords or full addresses. View in Streamlit Cloud under Manage app → logs |
| Model exports | FastAPI service requires an `X-API-Key` header (from `VANGUARD_API_KEY`), caps request size, has `/health`; Dockerfile runs as a non-root user; `SECURITY.txt` deployment notes in every bundle |

Account settings to keep on: Auth0 multi-factor authentication, brute-force protection and breached-password
detection; GitHub secret scanning, push protection and Dependabot alerts. Keys live only in Streamlit Secrets.

## What it does

| Step | What happens |
|---|---|
| **Clean** | Detects column types, strips currency and percent text, removes duplicates and ID columns, fills or sets aside missing values, flags (never deletes) outliers, and keeps codes such as ZIP or NAICS as categories. Every change is listed in a cleaning report. |
| **Describe** | Frequency tables (equal-width, equal-count or log-scale grouping, picked automatically for lopsided data), summary statistics including mean absolute deviation, cross-tabulations, correlation, and charts chosen to fit each column. |
| **Prepare** | Data quality report, scale types, binning, one-hot encoding, normalization, transforms (log, Box-Cox, Yeo-Johnson, quantile), mapping numeric columns to a Gaussian distribution with normality tests, and PCA. |
| **Predict** | Compares regression or classification models (linear/logistic, k-NN, naive Bayes, decision tree, random forest, gradient boosting, neural network (MLP), support vector machine) on held-out data with 5-fold cross-validation and reports the measured error. |
| **Forecast** | Tests seven forecasting methods on your most recent periods and forecasts with the best, with a rough range. |
| **Find patterns** | Clustering, anomaly detection and significance tests (t-test, ANOVA, chi-square, correlation). |
| **Combine sources** | Makes measurements from different sites, stores, labs or methods comparable (robust z-scores per series), compares each reading with the other series in the same period, flags outliers, reporting gaps and step changes (such as a lab switch), combines them into one weighted index, and forecasts that index against a naive baseline. |
| **Business problem & KPIs** | A project life cycle tracker (understand the problem, data access, data engineering, modeling) plus five steps: problem statement, requirements (target/DV and inputs/IVs), data pipeline (filled in from the cleaning log), target balance (class imbalance), each input against the target, a correlation matrix including the target, feature importance (random forest Gini, decision tree, permutation, Boruta), KPIs with trend tests (customer churn rate by month, length of relationship, transaction frequency, revenue per customer, any KPI over time), the columns most linked to the target, and a Word business problem brief. |
| **Data pipeline** | Seven stages: ingest; validate (PASS/WARN/FAIL quality gates, optional schema contract); bronze (raw copy with fingerprint); silver (cleaned, joined with a second file, private details hashed or masked); gold (summary table); orchestrate (save the steps as a config and re-run them with `pipeline_run.py` on a schedule); govern and monitor (lineage, run log, SQL on a DuckDB warehouse). Downloads a bundle of Parquet files plus `warehouse.duckdb`. A Parquet & partitions tab compares CSV with Parquet compressions (snappy, gzip, brotli, lz4, zstd), adds year/month/day columns, writes a partitioned dataset (year=2026/month=01/...), checks partition columns, and shows partition pruning. Tables can be written back to a database (fail / overwrite / append), and the whole pipeline can be downloaded as a PySpark script for a Spark cluster. |
| **Group & summarize** | Group by any category or by day, week, month, quarter or year, with totals, averages, medians, counts, min, max and spread; pivot tables with totals. |
| **Analyze text** | Scores each comment as positive, neutral or negative (VADER), compares sentiment by group, and lists the most common words and two-word phrases. |
| **Items bought together** | Market basket analysis: support, confidence and lift for every pair of items, plus "customers who buy X also buy" suggestions. |
| **Deep learning (Keras)** | TensorFlow/Keras: a dense neural network on any table (layers, dropout, optimizer, learning rate, early stopping, live training curves) compared with logistic/linear regression and random forest, plus a fairness check by group (four-fifths rule); a CNN image classifier (Conv2D, MaxPooling, Dropout, data augmentation) from a zip of class folders or built-in handwritten digits; and deep Q-learning agents (treasure maze, Gymnasium CartPole) with epsilon-greedy exploration, experience replay and a discount factor. Trained models download with a FastAPI service and Dockerfile. |
| **Report** | One click builds a Word report of what was loaded, what the cleaning changed, and a plain-language description of every column with charts. |

**Inputs:** CSV, TSV, TXT, Excel, JSON and Parquet, uploaded, loaded from a web address (like `pd.read_csv(url)`), or read from a PostgreSQL, MySQL or SQLite database (a table or a SELECT query, like Spark's `read.jdbc`). ORC files are supported too.
Files with no header row are detected and can be given column names; '?', 'N/A', 'null' and similar count as missing.
You can change any column's type, choose how missing values are handled, see which business questions the data can
answer, download the cleaned data as CSV, Excel, JSON or Parquet, and download a reproducibility record (input
fingerprint, settings, every cleaning step, random seed and library versions). Large-file mode (DuckDB) scans files with millions
of rows on one computer. A built-in connector also pulls public U.S. federal contract awards from USAspending.gov.

## Sample data (synthetic)

| Sample | Try |
|---|---|
| Retail sales by store | Forecast weekly revenue; compare stores; Combine sources by store |
| Clinic visits | Forecast visits; test whether wait time relates to satisfaction |
| Multi-site measurements | Combine & standardize sources (finds the planted bad readings, gaps and lab change) |
| Government contracts | Predict total cost or risk level |
| Contract awards over time | Forecast monthly awards |
| Customer reviews | Analyze text (sentiment by store); predict churn |
| Store orders | Find items bought together; customer KPIs and churn; data pipeline |
| Bank marketing | Business problem & KPIs: target y, imbalance, feature importance and Boruta |

## Run a saved pipeline on a schedule

```bash
py pipeline_run.py --config my_pipeline.json --input "C:\data\new_file.csv" --out "C:\data\pipeline_output"
```

Exit code 0 = success, 2 = stopped at a quality gate, 1 = error. Schedule it with Windows Task Scheduler
(`schtasks`), cron, or an orchestrator such as Airflow.

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
tables inside PDFs, maps, and user accounts. Distributed processing (Spark, Hadoop) is not planned for the
hosted app; DuckDB covers large files on one computer.

## Code layout

- `app.py`: Streamlit app
- `data_cleaner.py`: loading, type detection and cleaning
- `descriptive.py`, `explore_ui.py`, `wrangle_ui.py`: description and preparation
- `model_selector.py`, `forecasting.py`, `unsupervised.py`, `statistical_tests.py`: models
- `standardize.py`, `standardize_ui.py`: multi-source standardization and index forecast
- `grouping.py`, `text_analysis.py`, `basket.py`, `analysis_ui.py`: group & summarize, text, items bought together
- `pipeline.py`, `pipeline_ui.py`, `pipeline_run.py`: data pipeline stages, screen and scheduled runner
- `deep_learning.py`, `deep_ui.py`: Keras neural networks, CNN, deep Q-learning, fairness, FastAPI/Docker export
- `dbio.py`, `db_ui.py`: database read and write
- `parquet_tools.py`: Parquet compression, partitioned datasets, partition pruning, PySpark export
- `importance.py`: target balance, correlation with the target, feature importance, Boruta
- `kpi.py`, `problem_ui.py`: business problem steps, KPIs and the Word brief
- `auth.py`: Google sign-in and access rules
- `guide.py`: goal picker, help panels, everyday vs advanced tools
- `use_cases.py`: "What can this data answer?"
- `report.py`: Word report
- `bigdata.py`: large-file mode (DuckDB)
- `usaspending.py`: USAspending.gov connector
