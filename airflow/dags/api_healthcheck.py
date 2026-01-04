from __future__ import annotations

import os
import pendulum
import requests
from airflow.decorators import dag, task

TZ = "UTC"
API_BASE_URL = os.environ["AIR_QUALITY_API_BASE_URL"].rstrip("/")


@dag(
    dag_id="api_healthcheck",
    schedule="* * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz=TZ),
    catchup=False,
    tags=["http", "health"],
)
def api_healthcheck():
    @task(retries=0)
    def check() -> None:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        r.raise_for_status()  # task FAILS only when down

    check()

api_healthcheck()