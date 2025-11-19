import os
import pandas as pd
import numpy as np
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import json
import logging
from datetime import datetime, UTC
from .risk_labels import aqi_to_label
from .llm_explainer import explain_forecast
from pathlib import Path
from src.monitoring_utils import compute_drift_metrics, psi_severity
from src.auto_retrain import auto_retrain_model, should_retrain
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

EXPECTED_FEATURE_COLS = {
    "co", "no", "no2", "o3", "so2",
    "pm2_5", "pm10", "nh3", "temp_c", "humidity",
    "aqi_lag1", "aqi_lag2", "aqi_lag3",
    "pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3",
    "pm10_lag1", "pm10_lag2", "pm10_lag3",
    "o3_lag1", "o3_lag2", "o3_lag3",
    "temp_c_lag1", "temp_c_lag2", "temp_c_lag3",
    "humidity_lag1", "humidity_lag2", "humidity_lag3",
}


DATA_FILE = os.path.join("data", "aq_clean.parquet")
MODEL_FILE = os.path.join("models", "risk_model.pkl")

logger = logging.getLogger("aq_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def log_predictions(results, endpoint: str):
    """Append predictions to a JSONL log file + log a short summary."""
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "predictions.jsonl")
    now = datetime.now(UTC).isoformat()

    with open(log_path, "a", encoding="utf-8") as f:
        for r in results:
            record = {
                "logged_at_utc": now,
                "endpoint": endpoint,
                **r,
            }
            f.write(json.dumps(record, default=str) + "\n")

    logger.info(f"{endpoint}: logged {len(results)} predictions → {log_path}")

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")

def send_alert(message: str):
    if not ALERT_WEBHOOK_URL:
        logger.warning(f"Alert requested but ALERT_WEBHOOK_URL not set. Message: {message}")
        return
    try:
        resp = requests.post(
            ALERT_WEBHOOK_URL,
            json={"content": message},
            timeout=5,
        )
        if resp.status_code != 204:     # Discord success = 204 No Content
            logger.error(f"Discord webhook error {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")

def load_latest_per_city():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("No clean data file found.")

    df = pd.read_parquet(DATA_FILE)
    if df.empty:
        raise ValueError("Clean data is empty.")

    df = df.sort_values("timestamp_utc")
    latest = df.groupby("city").tail(1)
    return latest


def build_features(df: pd.DataFrame):
    feature_cols = list(EXPECTED_FEATURE_COLS)
    return df[feature_cols]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not os.path.exists(MODEL_FILE):
        raise RuntimeError("Model file not found. Train the model first.")
    app.state.model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
    
    yield  # <-- the app runs here
    
    # Shutdown (nothing to clean up yet)
    print("Shutting down.")


app = FastAPI(
    title="Air Quality Health Risk API",
    lifespan=lifespan
)

@app.get("/forecast/3h/explain")
def forecast_3h_explain():
    try:
        df = load_latest_per_city()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    X = build_features(df)

    # Schema drift check
    missing = EXPECTED_FEATURE_COLS - set(X.columns)
    extra = set(X.columns) - EXPECTED_FEATURE_COLS

    if missing or extra:
        logger.error(f"Schema drift detected. Missing: {missing}, Extra: {extra}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Schema mismatch",
                "missing_columns": list(missing),
                "extra_columns": list(extra),
            },
        )

    preds = app.state.model.predict(X)


    results = []
    for (_, row), future_aqi in zip(df.iterrows(), preds):
        future_aqi = float(future_aqi)
        label = aqi_to_label(future_aqi)

        try:
            explanation = explain_forecast(
                city=row["city"],
                aqi_3h=future_aqi,
                label_3h=label,
                pm25=row["pm2_5"],
                pm10=row["pm10"],
                temp_c=row["temp_c"],
                humidity=row["humidity"],
            )
        except Exception as e:
            explanation = f"LLM explanation unavailable: {e}"

        results.append({
            "timestamp_utc": row["timestamp_utc"],
            "city": row["city"],
            "predicted_aqi_3h": future_aqi,
            "predicted_label_3h": label,
            "pm2_5": row["pm2_5"],
            "pm10": row["pm10"],
            "temp_c": row["temp_c"],
            "humidity": row["humidity"],
            "explanation": explanation,
        })

    log_predictions(results, endpoint="/forecast/3h/explain")
    return {"results": results}

@app.get("/monitor/data_drift")
def monitor_drift():
    log_path = Path("logs") / "predictions.jsonl"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No prediction logs found yet")

    df = pd.read_json(log_path, lines=True)

    # Hard cap to last N rows to avoid RAM spikes
    N = 1000
    if len(df) > N:
        df = df.tail(N).copy()

    if len(df) < 100:
        return {
            "status": "insufficient_data",
            "n_logs": int(len(df)),
            "detail": "Need at least 100 logged predictions for drift analysis.",
        }

    # Ensure sorted by time
    df["logged_at_utc"] = pd.to_datetime(df["logged_at_utc"], format="ISO8601", utc=True, errors="coerce")
    df = df.sort_values("logged_at_utc")

    n = len(df)
    baseline = df.iloc[: int(0.2 * n)]
    recent = df.iloc[int(0.8 * n):]

    features = ["pm2_5", "pm10", "temp_c", "humidity", "predicted_aqi_3h"]

    report = {}
    severity_map = {"no_drift": 0, "moderate_drift": 1, "significant_drift": 2}
    overall_level = 0

    for col in features:
        metrics = compute_drift_metrics(baseline, recent, col)
        sev = psi_severity(metrics["PSI"])
        metrics["severity"] = sev
        report[col] = metrics
        if sev in severity_map:
            overall_level = max(overall_level, severity_map[sev])

    inv_severity_map = {0: "no_drift", 1: "moderate_drift", 2: "significant_drift"}
    overall_status = inv_severity_map.get(overall_level, "insufficient_data")

    # 🔔 send alert if overall drift is moderate or significant
    if overall_status in ("moderate_drift", "significant_drift"):
        # list only features with drift
        drifted_features = [
            f"{name} (PSI={metrics['PSI']:.3f}, sev={metrics['severity']})"
            for name, metrics in report.items()
            if metrics["severity"] in ("moderate_drift", "significant_drift")
        ]
        msg = (
            f"[DATA DRIFT] Overall status={overall_status.upper()}, "
            f"n_logs={n}. Drifted features: " + ", ".join(drifted_features)
        )
        send_alert(msg)

    if should_retrain() and overall_status in ("moderate_drift", "significant_drift"):
        auto_retrain_model()

    return {
        "status": overall_status,
        "n_logs": int(n),
        "features": report,
    }


@app.get("/monitor/schema")
def monitor_schema():
    if not os.path.exists(DATA_FILE):
        raise HTTPException(status_code=404, detail="Clean data file not found")

    try:
        df_raw = pd.read_parquet(DATA_FILE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read data: {e}")

    if df_raw.empty:
        return {
            "status": "empty_data",
            "detail": "Clean data file is empty.",
        }

    # Use a small sample to infer available feature columns
    sample = df_raw.tail(10).copy()
    X = build_features(sample)

    current_cols = set(X.columns)
    missing = EXPECTED_FEATURE_COLS - current_cols
    extra = current_cols - EXPECTED_FEATURE_COLS

    status = "ok"
    if missing or extra:
        status = "mismatch"

    col_types = {col: str(dtype) for col, dtype in X.dtypes.items()}

    if status == "mismatch":  # for schema
        send_alert(f"[SCHEMA DRIFT] Missing: {missing}, Extra: {extra}")

    return {
        "status": status,
        "n_rows_checked": int(len(sample)),
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "column_types": col_types,
    }


@app.get("/monitor/model_drift")
def monitor_model():
    perf_file = os.path.join("logs", "model_performance.jsonl")
    if not os.path.exists(perf_file):
        raise HTTPException(status_code=404, detail="No model performance log found")

    df = pd.read_json(perf_file, lines=True)
    if df.empty:
        return {
            "status": "insufficient_data",
            "n_records": 0,
            "detail": "No performance records yet.",
        }

    df = df.sort_values("prediction_made_at_utc")
    n = len(df)
    if n < 50:
        return {
            "status": "insufficient_data",
            "n_records": int(n),
            "detail": "Need at least 50 records to assess model drift.",
        }

    baseline = df.iloc[: int(0.2 * n)]
    recent = df.iloc[int(0.8 * n):]

    def rmse(e):
        return float(np.sqrt(np.mean(np.square(e))))

    baseline_rmse = rmse(baseline["error"])
    recent_rmse = rmse(recent["error"])
    delta = recent_rmse - baseline_rmse

    # simple threshold: drift if RMSE worsens by >= 0.25
    threshold = 0.25
    if delta >= threshold:
        status = "drift_detected"
        send_alert(
            f"[MODEL DRIFT] RMSE worsened from {baseline_rmse:.3f} to {recent_rmse:.3f} "
            f"(Δ={delta:.3f}, threshold={threshold})"
        )
    else:
        status = "no_drift"        

    if should_retrain() and status=="drift_detected":
        auto_retrain_model()

    return {
        "status": status,
        "n_records": int(n),
        "baseline_rmse": baseline_rmse,
        "recent_rmse": recent_rmse,
        "delta_rmse": delta,
        "threshold": threshold,
    }