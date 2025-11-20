# Air Quality Health Risk Pipeline (Updated)

This project is a **full end-to-end MLOps system** that continuously ingests live air‑quality data, cleans it, retrains forecasting models, serves predictions through an API, monitors operational & ML drift, and automatically triggers alerts and retraining — all running inside **Docker** with **cron‑driven automation**.

---

## 🚀 System Overview

The pipeline performs the following:

- **Ingest live air‑quality + weather data** from OpenWeatherMap every 5 minutes
- Store raw unmodified responses in an **append‑only JSONL log**
- Convert raw logs into a **clean Parquet dataset** for training & inference
- Train a **3‑hour AQI forecasting model** (LightGBM)
- Serve predictions and explanations via a **FastAPI microservice**
- Generate **LLM explanations** using Ollama (local model)
- Log predictions for **latency tracking, drift detection, and traffic analysis**
- Detect **schema drift**, **data drift**, **model drift**, and **traffic anomalies**
- Perform **auto‑retraining** when drift thresholds are exceeded
- Send **alerts** for failures, drift events, or API performance issues
- Run inside **Docker** and fully orchestrated with **cron jobs**

---

## 🧱 Tech Stack

- **Python**, Pandas, NumPy
- **FastAPI** + Uvicorn
- **LightGBM / scikit-learn**
- **MLflow**
- **Docker / Docker Compose**
- **Cron** for automation
- **OpenWeatherMap** (Air Pollution + Weather)
- **Ollama** for LLM explanations
- **Ubuntu (DigitalOcean / local VM)**

---

## 📁 Project Structure

```
├── data/
│   ├── aq_raw.jsonl            # append-only raw logs
│   └── aq_clean.parquet        # cleaned feature dataset
│
├── logs/
│   ├── predictions.jsonl       # API prediction + latency logs
│   └── model_performance.jsonl # backfilled error logs (model drift)
│
├── models/
│   └── risk_model.pkl          # trained LightGBM model
│
├── scripts/
│   ├── backfill_model_error.py # hourly backfill for model drift
│   ├── run_daily_train.py      # cron: daily retraining
│   ├── run_hourly_etl.py       # cron: ETL wrapper
│   └── quick_forecast.py       # dev-only
│
├── src/
│   ├── api.py                  # FastAPI application
│   ├── ingest_air_quality.py   # live ingestion (raw → JSONL)
│   ├── parse_air_quality.py    # ETL to Parquet
│   ├── train_risk_model.py     # model training
│   ├── llm_explainer.py        # Ollama explanation generation
│   ├── risk_labels.py          # AQI → health risk category
│   │
│   └── monitoring/             # full monitoring suite
│       ├── alerts.py
│       ├── drift.py
│       ├── latency.py
│       ├── logging.py
│       ├── schema.py
│       ├── traffic.py
│       └── utils.py
│
├── docker-compose.yml          # API + MLflow services
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🖥️ Running Locally (Development)

### 1. Ingest live data
```bash
python src/ingest_air_quality.py
```

### 2. ETL (raw → clean)
```bash
python src/parse_air_quality.py
```

### 3. Train the forecasting model
```bash
python src/train_risk_model.py
```

### 4. Start the API
```bash
uvicorn src.api:app --reload
```
Docs: http://localhost:8000/docs

### 5. Start MLflow locally (no Docker)
```bash
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000
```

### ⚠️ Local vs Server: LLM Behavior
- **Locally:** Ollama runs normally and provides natural-language explanations.
- **On the server:** If Ollama is not installed (or RAM is limited), the API automatically falls back to a lightweight string‑only explanation function (no LLM cost, no RAM overhead).

---

## ☁️ Running on the Server (DigitalOcean VM)

On the VM, **you only use `docker-compose`** — no building images manually.
Everything is preconfigured: API service, MLflow, volumes.

### Start all services
```bash
docker-compose up -d --build
```

### MLflow UI on the server
Accessible at:
```
http://YOUR_SERVER_IP:5000
```

`docker-compose.yml` manages:
- API service
- MLflow tracking server
- Shared volumes for MLruns, logs, data, models

---

## ⏱ Cron Automation (Production)

Open crontab:
```bash
crontab -e
```

### Ingestion + ETL (every 5 minutes)
```bash
*/5 * * * * docker exec air-quality-api python src/ingest_air_quality.py
*/5 * * * * docker exec air-quality-api python src/parse_air_quality.py
```

### Daily retraining (3 AM)
```bash
0 3 * * * docker exec air-quality-api python src/train_risk_model.py
```

### Model drift backfill (hourly)
```bash
5 * * * * docker exec air-quality-api python scripts/backfill_model_error.py
```

### Simulated API traffic
```bash
*/30 * * * * curl -s http://localhost:8000/forecast/3h/explain > /dev/null
```

### Monitoring suite
```bash
10 * * * * curl -s http://localhost:8000/monitor/schema > /dev/null
11 * * * * curl -s http://localhost:8000/monitor/data_drift > /dev/null
12 * * * * curl -s http://localhost:8000/monitor/model > /dev/null
*  * * * * curl -sf http://localhost:8000/health || curl -H "Content-Type: application/json" -d '{"alert": "API down"}' YOUR_ALERT_ENDPOINT
```

---

## 🔍 Monitoring Endpoints

### `/monitor/schema`
- Detects schema mismatches between live data and model features

### `/monitor/data_drift`
- Tracks distribution shift using recent prediction logs
- Includes auto-drift alerts and optional auto-retraining

### `/monitor/model`
- Checks degradation over time via RMSE comparison

### `/monitor/traffic`
- Detects spikes/drops in API usage

### `/forecast/3h/explain`
- Returns prediction
- Health label
- Latency
- LLM explanation of AQI risks

---

## 🔔 Alerts & Auto-Retraining
The monitoring suite uses the following rules:

- **Schema Drift:** missing/extra columns → alert + fail prediction
- **Data Drift:** moderate/significant drift → alert + optional auto‑retrain
- **Model Drift:** RMSE shift ≥ 0.25 → alert + auto‑retrain
- **Latency:** slow prediction → alert
- **Traffic:** large spike/drop → alert
- **API down:** health check fallback curl fires alert

---

## 🔒 Environment Variables (`.env`)

```
OWM_API_KEY=...
OLLAMA_BASE_URL=...
OLLAMA_MODEL=...
ALERT_WEBHOOK_URL=...
```

`.env` **must not be committed**.

---

## 📝 Useful Commands
```bash
git pull origin main
docker-compose up -d --build
docker-compose logs -f
docker logs --tail 50 air-quality-api
docker-compose down
docker-compose restart api
```

---

## 📌 Notes
- Everything is designed to run continuously with minimal supervision
- Auto‑drift detection + retraining makes this a production‑style MLOps system
- Local LLM explanations avoid external API cost
- Docker + cron create a stable, repeatable runtime

---