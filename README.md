# Air Quality Health Risk Pipeline

## Overview

This project is a complete end-to-end Machine Learning Engineering system that:

* Ingests **live air quality + weather data** from OpenWeatherMap
* Stores raw data as an **append-only JSONL stream**
* Parses raw data into a **clean Parquet dataset**
* Trains a **3-hour AQI forecasting model** (LightGBM)
* Serves predictions through a **FastAPI API**
* Generates **LLM explanations** (Ollama or OpenAI-compatible)
* Logs predictions + errors for **drift monitoring**
* Monitors **schema drift**, **data drift**, and **model drift**
* Runs inside **Docker** on a cloud VM (DigitalOcean)
* Uses **cron jobs** to automate ingestion, ETL, training, drift jobs, and synthetic API traffic

A full MLOps-ready project suitable for a professional portfolio.

---

## Tech Stack

* Python
* FastAPI + Uvicorn
* LightGBM / scikit-learn
* Pandas + PyArrow
* MLflow
* Docker
* DigitalOcean (Ubuntu)
* Cron automation
* OpenWeatherMap Air Pollution + Weather APIs
* Ollama (local LLM for explanations)

---

## Project Structure

```text
src/
  ingest_air_quality.py     # live ingestion (raw JSON stream)
  parse_air_quality.py      # ETL: raw → clean Parquet
  train_risk_model.py       # model training + MLflow logging
  backfill_model_error.py   # model drift tracking
  monitoring_utils.py       # PSI + drift utilities
  api.py                    # FastAPI: predictions + monitoring
  llm_explainer.py          # local or OpenAI-compatible LLM
  risk_labels.py            # AQI category mapping
  run_hourly_etl.py         # for cron: ETL wrapper
  run_daily_train.py        # for cron: daily retraining wrapper
data/
  aq_raw.jsonl              # append-only raw logs
  aq_clean.parquet          # cleaned dataset
models/
  risk_model.pkl            # trained forecasting model
  model_metadata.json       # RMSE + timestamp
logs/
  predictions.jsonl         # API logs
  model_performance.jsonl   # backfilled model drift logs
Dockerfile
requirements.txt
.env                        # (not committed)
README.md
```

---

## Running the Pipeline Locally

### 1) Ingest live data

```bash
python src/ingest_air_quality.py
```

### 2) ETL: raw → clean

```bash
python src/parse_air_quality.py
```

### 3) Train model + log to MLflow

```bash
python src/train_risk_model.py
```

### 4) Start API

```bash
uvicorn src.api:app --reload
```

### 5) (Optional) MLflow UI

```bash
mlflow ui
```

---

## Docker Usage

### Build image

```bash
docker build -t air-quality-api .
```

### Run container (Windows)

```bash
docker run ^
  --env-file .env ^
  -p 8000:8000 ^
  -v "%cd%/data:/app/data" ^
  -v "%cd%/models:/app/models" ^
  -v "%cd%/logs:/app/logs" ^
  air-quality-api
```

### Run container (Linux/macOS)

```bash
docker run \
  --env-file .env \
  -p 8000:8000 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/logs:/app/logs" \
  air-quality-api
```

API docs:
[http://localhost:8000/docs](http://localhost:8000/docs)

---

## Deploying on DigitalOcean (Ubuntu VM)

### Clone repo + add `.env`

```bash
git clone https://github.com/YOUR_USERNAME/air-quality-health.git
cd air-quality-health
nano .env
```

### Build container

```bash
docker build -t air-quality-api .
```

### Run container

```bash
docker run -d \
  --name air-quality-api \
  --env-file .env \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/logs:/app/logs" \
  air-quality-api
```

API URL:
http://YOUR_SERVER_IP:8000/docs

---

## Cron Automation (Production)

Edit crontab:

```bash
crontab -e
```

### Ingest + ETL (every 5 min)

```bash
*/5 * * * * docker exec air-quality-api python src/ingest_air_quality.py
*/5 * * * * docker exec air-quality-api python src/parse_air_quality.py
```

### Daily retraining (3 AM)

```bash
0 3 * * * docker exec air-quality-api python src/train_risk_model.py
```

### Model drift logging (hourly)

```bash
5 * * * * docker exec air-quality-api python src/backfill_model_error.py
```

### Simulated API traffic (LLM explanations)

```bash
*/30 * * * * curl -s http://localhost:8000/forecast/3h/explain > /dev/null
```

### Automated monitoring

```bash
10 * * * * curl -s http://localhost:8000/monitor/schema > /dev/null
11 * * * * curl -s http://localhost:8000/monitor/data_drift > /dev/null
12 * * * * curl -s http://localhost:8000/monitor/model > /dev/null
```

---

## Log Rotation (Ubuntu)

Create:

```bash
sudo nano /etc/logrotate.d/airq
```

Add:

```text
/root/air-quality-health/logs/*.jsonl {
    daily
    rotate 7
    compress
    missingok
    notifempty
    copytruncate
}
```

---

## Notes

* `.env` is **not committed** to GitHub
* Drift detection (schema, data, model) is automated
* Uses **local LLM** for explanations (no API cost)
* Dockerized + cron-driven → production-like MLOps
* Cloud VM stores data, models, and logs persistently

---