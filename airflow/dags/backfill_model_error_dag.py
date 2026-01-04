from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="backfill_model_error",
    start_date=datetime(2026, 1, 1),
    schedule="5 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "airflow",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["air-quality", "monitoring"],
) as dag:

    backfill = BashOperator(
        task_id="backfill_model_error",
        bash_command=(
            "python -u -m monitoring.backfill_model_error "
            "--data_dir /opt/airflow/data "
            "--models_dir /opt/airflow/models"
        ),
        env={"PYTHONPATH": "/opt/airflow/src"},
    )
