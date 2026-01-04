from __future__ import annotations

import pendulum
import pandas as pd
from airflow.decorators import dag, task

from src.config import DATA_DIR
from src.monitoring.schema import compute_schema_status
from src.monitoring.alerts import send_alert
from src.monitoring.schema import build_features


TZ = "UTC"

@dag(
    dag_id="monitor_schema",
    schedule="10 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz=TZ),
    catchup=False,
    tags=["monitoring", "schema"],
)
def monitor_schema():

    @task
    def run_schema_check():
        df = pd.read_parquet(f"{DATA_DIR}/aq_clean.parquet")
        X = build_features(df)

        status = compute_schema_status(X)

        if status["status"] != "ok":
            send_alert(
                "SCHEMA MISMATCH",
                {
                    "missing": list(status["missing"]),
                    "extra": list(status["extra"]),
                },
            )

    run_schema_check()

monitor_schema()