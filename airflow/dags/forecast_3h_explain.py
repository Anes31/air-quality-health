from __future__ import annotations

import os
import pendulum
import requests
from airflow.decorators import dag, task

from src.monitoring.latency import timed_call, maybe_alert_slow_call

TZ = "UTC"
API_BASE_URL = os.environ["AIR_QUALITY_API_BASE_URL"].rstrip("/")

LATENCY_THRESHOLD_S = float(os.getenv("FORECAST_EXPLAIN_LATENCY_THRESHOLD_S", "5.0"))

@dag(
    dag_id="forecast_3h_explain",
    schedule="*/30 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz=TZ),
    catchup=False,
    tags=["http", "forecast"],
)
def forecast_3h_explain():
    @task
    def run():
        url = f"{API_BASE_URL}/forecast/3h/explain"
        (resp, latency_s) = timed_call(requests.get, url, timeout=30)
        resp.raise_for_status()
        maybe_alert_slow_call(endpoint="/forecast/3h/explain", latency_s=latency_s, threshold_s=LATENCY_THRESHOLD_S)

    run()

forecast_3h_explain()
