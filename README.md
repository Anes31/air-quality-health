# Air Quality Health Risk Pipeline (Updated)

This project is a **full end-to-end MLOps system** that ingests live air-quality data, retrains forecasting models, serves predictions through a FastAPI service, tracks drift, logs metrics, runs MLflow, and auto-deploys through **GitHub Actions (CI/CD)** to a **DigitalOcean VM**.

Everything runs inside **Docker**, with **cron-based automation** and **GitHub CI/CD** for builds, tests, validation, and deployment.

---

# 🚀 System Overview

* Live ingestion every 5 minutes
* ETL → Parquet
* LightGBM forecasting model
* FastAPI prediction + explanation
* MLflow tracking server
* Drift detection + alerts
* Cron-based retraining
* Dockerized end-to-end
* CI/CD auto-build + auto-deploy

---

# 🧱 Tech Stack

* Python / FastAPI
* LightGBM
* MLflow
* Docker / Docker Compose
* GitHub Actions (CI + CD)
* DigitalOcean VM
* Cron automation

---

# 📁 Project Structure (short)

```
data/          # raw + clean data
logs/          # API + model logs
models/        # model.pkl
scripts/       # retrain, ETL, drift jobs
src/           # API + training + monitoring
docker-compose.yml
Dockerfile
.github/workflows/ci.yml
.github/workflows/deploy.yml
```

---

# ⚙️ CI/CD PIPELINE (IMPORTANT)

## CI (Continuous Integration)

Triggered on **every push to main**.

Performs:

1. Install dependencies
2. Run unit tests (`pytest`)
3. Run model validation (`validate_model.py`)
4. Build Docker image
5. Push image to Docker Hub
6. Tag as `latest`

If any step fails, build stops.

---

## CD (Continuous Deployment)

Triggered **manually** from GitHub Actions (workflow_dispatch).

Performs on the VM:

```
cd /root/air-quality-health
git reset --hard origin/main
docker-compose down
docker system prune -f
docker-compose pull
docker-compose up -d
```

This guarantees:

* VM always matches GitHub repo
* Old containers removed
* New image pulled
* New healthchecks applied cleanly
* System restarts with zero stale config

---

# ❤️ Healthchecks (Docker)

### API container

```
test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
```

### MLflow container

```
test: ["CMD-SHELL", "curl -f http://localhost:5000/ || exit 1"]
```

**Important:**
Your Dockerfile MUST include `curl`:

```dockerfile
RUN apt-get update && apt-get install -y curl
```

Without this, both containers will stay **unhealthy**.

---

# 🖥️ Running Locally

### Start API

```
uvicorn src.api:app --reload
```

### Start MLflow

```
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000
```

---

# ☁️ Running on the Server (VM)

### Start services

```
docker-compose up -d
```

### View logs

```
docker-compose logs -f
```

### Check health

```
docker ps
```

---

# ⏱ Automation (Cron on VM)

* ETL + ingest every 5 min
* Drift backfill hourly
* Retrain daily
* Monitoring checks
* API uptime check

All run via:

```
docker exec air-quality-api python ...
```

---

# 🔍 Monitoring Endpoints

* `/health`
* `/ready`
* `/monitor/schema`
* `/monitor/data_drift`
* `/monitor/model`
* `/monitor/traffic`

---

# 🔒 Environment Variables

`.env` contains:

```
OWM_API_KEY=...
OLLAMA_BASE_URL=...
OLLAMA_MODEL=...
ALERT_WEBHOOK_URL=...
```

Do NOT commit `.env`.

---

# 📝 Useful Commands

```
docker-compose up -d
docker-compose down
docker-compose pull
docker-compose restart api
docker-compose logs -f
git reset --hard origin/main
```

---

# 📌 Notes

* CI/CD now handles image builds, pushes, and clean redeploys
* Healthchecks require `curl` inside the image
* Deploy workflow uses `reset --hard` to guarantee correct state
* Docker Compose health reflects container-internal checks
* Containers run cleanly after automatic rebuilds

---