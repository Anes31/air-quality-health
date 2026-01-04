# Air Quality Health Risk Pipeline

This project is an **end-to-end MLOps system** that ingests live air-quality data, trains and evaluates forecasting models, serves predictions via an API, tracks drift and performance, and runs fully automated scheduling and monitoring with **Airflow**.

Everything runs in **Docker**, with CI/CD handling build and deployment.

---

## 🚀 What the System Does

* Ingests live air-quality data every 5 minutes
* Runs ETL and stores clean data as Parquet
* Trains a LightGBM forecasting model
* Serves predictions and explanations via FastAPI
* Logs predictions and model performance
* Tracks data drift and model drift
* Triggers alerts and optional retraining
* Tracks experiments with MLflow
* Orchestrates everything with Airflow

---

## 🧱 Tech Stack

* Python / FastAPI
* LightGBM
* MLflow
* Airflow
* Docker / Docker Compose
* GitHub Actions (CI/CD)
* DigitalOcean VM

---

## 📁 Project Structure

```
data/           # raw + clean data (host-mounted)
logs/           # API logs (host-mounted)
models/         # trained models + metrics (host-mounted)
src/            # shared code used by API + Airflow DAGs

airflow/
  ├─ dags/      # Airflow DAGs
  ├─ logs/      # Airflow logs (host-mounted)
  ├─ plugins/   # Airflow plugins (optional)
  └─ docker-compose.yml

docker-compose.yml   # API + MLflow stack
Dockerfile
.github/workflows/
```

---

## ⏱ Automation (Airflow)

All automation is handled by **Airflow**.

### Core DAGs

* `etl_pipeline` — ingest + parse (every 5 minutes)
* `train_model` — daily retraining (03:00)
* `backfill_model_error` — hourly performance backfill
* `monitor_schema`
* `monitor_data_drift`
* `monitor_model`
* `forecast_3h_explain`
* `api_healthcheck`

No cron jobs. No `docker exec`. No curl-based scheduling.

---

## 🌐 API

### Endpoints

* `GET /health` — liveness / readiness
* `GET /forecast/3h/explain` — predictions + explanations

Monitoring endpoints are **not exposed**; monitoring runs inside Airflow.

---

## 🔍 Monitoring & Alerts

* Schema drift
* Data drift
* Model performance drift
* Latency checks
* API health checks

Alerts are sent via `ALERT_WEBHOOK_URL`.

Health alerts trigger **only on failure**, not on success.

---

## 🖥️ Running Locally

### Start API (dev only)

```
uvicorn src.api:app --reload
```

### Start Airflow

Windows: use a local compose override file with Windows paths (example: `docker-compose.local.yml`).
Linux: run the Airflow compose file directly.

```
cd airflow
docker-compose up -d
```

---

## ☁️ Running on the Server

### Start services

```
# API + MLflow stack
cd /root/air-quality-health
docker-compose up -d

# Airflow stack
cd /root/air-quality-health/airflow
docker-compose up -d
```

### View logs

```
docker-compose logs -f
```

### Check container health

```
docker ps
```

---

## 🔒 Environment Variables

### API

```
MODEL_FILE=/app/models/risk_model.pkl
DATA_FILE=/app/data/aq_clean.parquet
ALERT_WEBHOOK_URL=...
```

### Airflow

```
AIR_QUALITY_API_BASE_URL=http://api:8000
```

`.env` is used by Docker Compose to set container environment (API + Airflow).

---

## 📌 Notes

* Airflow is the **only scheduler**
* First-time Airflow setup requires initializing the metadata DB (`airflow db migrate` / `airflow db init`)
* Airflow imports project code from `/opt/airflow/src` (mounted from repo `src/`)
* All paths inside containers are absolute and fixed
* Monitoring logic lives in `src/monitoring/`
* CI/CD builds and deploys clean images
* Local and production differ only by configuration, not code