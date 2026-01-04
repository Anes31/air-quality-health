from __future__ import annotations

import pendulum
import pandas as pd
from airflow.decorators import dag, task

from src.config import DATA_DIR
from src.monitoring.drift import (
    summarize_data_drift,
    maybe_alert_data_drift,
    maybe_auto_retrain,
)
from src.monitoring.schema import EXPECTED_FEATURE_COLS

TZ = "UTC"

@dag(
    dag_id="monitor_data_drift",
    schedule="11 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz=TZ),
    catchup=False,
    tags=["monitoring", "drift"],
)
def monitor_data_drift():

    @task
    def run_drift_check():
        df = pd.read_parquet(f"{DATA_DIR}/aq_clean.parquet")

        df = df.sort_values("timestamp_utc").dropna()
        features = list(EXPECTED_FEATURE_COLS)

        result = summarize_data_drift(df, features)

        maybe_alert_data_drift(
            overall_status=result["overall_status"],
            report=result["report"],
            n_logs=result["n_logs"],
        )

        maybe_auto_retrain(
            trigger="data_drift",
            condition=result["overall_status"] == "significant_drift",
            extra_details={"n_logs": result["n_logs"]},
        )

    run_drift_check()

monitor_data_drift()
