FROM apache/airflow:2.10.2

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir lightgbm mlflow joblib scikit-learn pandas numpy pyarrow
