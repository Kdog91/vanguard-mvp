"""
Re-run a pipeline saved from the app (Data pipeline → Orchestrate → Download pipeline config) on a new file.

    py pipeline_run.py --config my_pipeline.json --input "C:\\data\\new_file.csv" --out "C:\\data\\pipeline_output"

Steps, in order: ingest → validate (stops if a quality gate fails) → bronze → silver → gold →
a zip with Parquet files and a DuckDB warehouse → one line added to run_log.csv.
Exit code 0 = success, 2 = stopped at the quality gate, 1 = error. A scheduler (Windows Task
Scheduler, cron, Airflow) can use the exit code to alert you.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="Run a saved Vanguard Analytics pipeline")
    ap.add_argument("--config", required=True, help="pipeline .json downloaded from the app")
    ap.add_argument("--input", required=True, help="file path or web address of the new data")
    ap.add_argument("--out", default="pipeline_output", help="folder for outputs and run_log.csv")
    a = ap.parse_args()
    import pipeline
    with open(a.config, encoding="utf-8") as f:
        config = json.load(f)
    try:
        result = pipeline.run_config(config, a.input, a.out)
    except Exception as e:
        print(f"ERROR: {e}")
        pipeline._append_log(a.out, {"input": a.input, "status": f"error: {e}"})
        sys.exit(1)
    sys.exit(0 if result.get("status") == "ok" else 2)


if __name__ == "__main__":
    main()
