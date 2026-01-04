from __future__ import annotations

import json
import math
import pendulum
import pandas as pd
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

from src.config import DATA_DIR, MODELS_DIR
from src.monitoring.alerts import send_alert
from src.monitoring.drift import maybe_auto_retrain

TZ = "UTC"

BASELINE_FILE = f"{MODELS_DIR}/metrics_baseline.json"
PERF_LOG = f"{DATA_DIR}/logs/model_performance.jsonl"

# tuneables (keep simple + deterministic)
WINDOW_HOURS = 24
ALERT_DELTA_RMSE = 0.10
RETRAIN_DELTA_RMSE = 0.25


def _load_baseline_rmse() -> float:
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        return float(json.load(f)["rmse"])


@dag(
    dag_id="monitor_model",
    schedule="12 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz=TZ),
    catchup=False,
    tags=["monitoring", "model"],
)
def monitor_model():

    @task
    def run_model_check():
        baseline_rmse = _load_baseline_rmse()

        df = pd.read_json(PERF_LOG, lines=True)
        if df.empty or "prediction_made_at_utc" not in df.columns or "error" not in df.columns:
            raise AirflowSkipException("No performance data yet.")

        df["prediction_made_at_utc"] = pd.to_datetime(df["prediction_made_at_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["prediction_made_at_utc", "error"])

        now = pd.Timestamp.now(tz="UTC")
        window_start = now - pd.Timedelta(hours=WINDOW_HOURS)
        recent = df[df["prediction_made_at_utc"] >= window_start]

        if len(recent) < 30:
            raise AirflowSkipException("Not enough recent performance rows yet.")

        rmse = math.sqrt(float((recent["error"].astype(float) ** 2).mean()))
        delta = rmse - baseline_rmse

        if delta > ALERT_DELTA_RMSE:
            send_alert(
                "MODEL DRIFT",
                {
                    "window_hours": WINDOW_HOURS,
                    "n_rows": int(len(recent)),
                    "baseline_rmse": round(baseline_rmse, 4),
                    "current_rmse": round(rmse, 4),
                    "delta_rmse": round(delta, 4),
                },
            )

        maybe_auto_retrain(
            trigger="model_drift",
            condition=delta > RETRAIN_DELTA_RMSE,
            extra_details={
                "window_hours": WINDOW_HOURS,
                "n_rows": int(len(recent)),
                "baseline_rmse": round(baseline_rmse, 4),
                "current_rmse": round(rmse, 4),
                "delta_rmse": round(delta, 4),
            },
        )

    run_model_check()

monitor_model()
