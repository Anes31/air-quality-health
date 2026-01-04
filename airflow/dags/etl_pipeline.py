from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

with DAG(
    dag_id="etl_pipeline",
    default_args=default_args,
    schedule_interval="*/5 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:

    ingest = BashOperator(
        task_id="ingest",
        bash_command="set -e; python -u /opt/airflow/src/ingest_air_quality.py",
    )

    parse = BashOperator(
        task_id="parse",
        bash_command="set -e; python -u /opt/airflow/src/parse_air_quality.py",
    )

    ingest >> parse