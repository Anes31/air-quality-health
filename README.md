# Air Quality Health Risk Pipeline

## Overview

This project is an end-to-end Machine Learning Engineering pipeline that:
- Ingests live air quality + weather data from OpenWeatherMap
- Stores raw JSON as a streaming log
- Parses raw data into a clean Parquet dataset
- Trains a classification model to predict AQI-based health risk
- Serves predictions via a FastAPI endpoint
- Tracks experiments with MLflow
- Runs as a Dockerized API service

## Tech Stack

- Python
- FastAPI + Uvicorn
- scikit-learn
- MLflow
- Docker
- OpenWeatherMap Air Pollution + Weather APIs
- Parquet (pyarrow/pandas)

## Project Structure

```text
src/
  ingest_air_quality.py   # streaming ingestion (raw JSON)
  parse_air_quality.py    # ETL: raw → clean Parquet
  train_risk_model.py     # model training + MLflow logging
  infer_latest_risk.py    # local inference helper
  api.py                  # FastAPI service for predictions
  risk_labels.py          # AQI → label mapping
  run_hourly_etl.py       # wrapper for scheduled ETL
  run_daily_train.py      # wrapper for scheduled retraining
models/
  risk_model.pkl          # current deployed model (local)
data/
  aq_raw.jsonl            # raw streaming data (append-only)
  aq_clean.parquet        # cleaned data for training/inference
Dockerfile
requirements.txt
README.md



# 1) Ingest live data
python src/ingest_air_quality.py

# 2) ETL: raw → clean
python src/parse_air_quality.py

# 3) Train model + log to MLflow
python src/train_risk_model.py

# 4) Start API
uvicorn src.api:app --reload

# 5) (Optional) MLflow UI
mlflow ui

# Build image
docker build -t air-quality-api .

# Run container
docker run ^
  --env-file .env ^
  -p 8000:8000 ^
  -v "%cd%/data:/app/data" ^
  -v "%cd%/models:/app/models" ^
  -v "%cd%/logs:/app/logs" ^
  air-quality-api
