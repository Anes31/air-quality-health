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
import time


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

LATENCY_THRESHOLD_SECONDS = float(os.getenv("LATENCY_THRESHOLD_SECONDS", "0.5"))

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

def send_alert(title: str, details: dict | None = None):
    if not ALERT_WEBHOOK_URL:
        logger.warning(f"Alert requested but ALERT_WEBHOOK_URL not set. Title: {title}, details: {details}")
        return

    lines = [f"**{title}**"]
    if details:
        for k, v in details.items():
            lines.append(f"- **{k}**: {v}")
    payload = {"content": "\n".join(lines)}

    try:
        resp = requests.post(ALERT_WEBHOOK_URL, json=payload, timeout=5)
        if resp.status_code != 204:
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
    logger.info("Model loaded successfully.")
    send_alert(
        "MODEL LOADED",
        {
            "model_file": MODEL_FILE,
            "loaded_at_utc": datetime.now(UTC).isoformat(),
        },
    )

    yield  # <-- the app runs here

    # Shutdown
    logger.info("Shutting down.")
    send_alert(
        "API SHUTDOWN",
        {
            "time_utc": datetime.now(UTC).isoformat(),
        },
    )

app = FastAPI(
    title="Air Quality Health Risk API",
    lifespan=lifespan
)

@app.get("/health")
def health():
    """Simple liveness + readiness check."""
    model_loaded = hasattr(app.state, "model")
    model_exists = os.path.exists(MODEL_FILE)

    status = "ok" if (model_loaded and model_exists) else "degraded"

    return {
        "status": status,
        "time_utc": datetime.now(UTC).isoformat(),
        "model_loaded": bool(model_loaded),
        "model_file_exists": bool(model_exists),
    }


@app.get("/forecast/3h/explain")
def forecast_3h_explain():
    try:
        df = load_latest_per_city()
    except Exception as e:
        # alert + clean error to client
        send_alert(
            "PREDICTION ERROR",
            {
                "step": "load_latest_per_city",
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail="Failed to load latest data.")
    
    try:
        X = build_features(df)

        # Schema drift check
        missing = EXPECTED_FEATURE_COLS - set(X.columns)
        extra = set(X.columns) - EXPECTED_FEATURE_COLS

        if missing or extra:
            logger.error(f"Schema drift detected. Missing: {missing}, Extra: {extra}")
            send_alert(
                "PREDICTION ERROR",
                {
                    "step": "schema_check",
                    "missing": list(missing),
                    "extra": list(extra),
                },
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Schema mismatch",
                    "missing_columns": list(missing),
                    "extra_columns": list(extra),
                },
            )

        # Measure prediction latency
        t0 = time.perf_counter()
        preds = app.state.model.predict(X)
        latency_s = time.perf_counter() - t0

        # alert on slow calls
        if latency_s > LATENCY_THRESHOLD_SECONDS:
            send_alert(
                "HIGH LATENCY",
                {
                    "endpoint": "/forecast/3h/explain",
                    "latency_seconds": round(latency_s, 3),
                    "n_rows": int(len(X)),
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
                "latency_seconds": latency_s,
            })

        log_predictions(results, endpoint="/forecast/3h/explain")
        return {"results": results}

    except HTTPException:
        # already handled above (schema / data issues)
        raise
    except Exception as e:
        logger.exception("Unexpected error in /forecast/3h/explain")
        send_alert(
            "PREDICTION ERROR",
            {
                "step": "forecast_3h_explain",
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail="Internal error during forecast.")

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

    # Send alert if overall drift is moderate or significant
    if overall_status in ("moderate_drift", "significant_drift"):
        drifted_features = [
            f"{name} (PSI={metrics['PSI']:.3f}, sev={metrics['severity']})"
            for name, metrics in report.items()
            if metrics["severity"] in ("moderate_drift", "significant_drift")
        ]

    send_alert(
        "DATA DRIFT",
        {
            "severity": overall_status,
            "n_logs": n,
            "drifted_features": ", ".join(drifted_features),
        },
    )

    if should_retrain() and overall_status in ("moderate_drift", "significant_drift"):
        send_alert(
            "RETRAIN STARTED",
            {
                "trigger": "data_drift",
                "overall_status": overall_status,
                "n_logs": int(n),
            },
        )
        try:
            res = auto_retrain_model()
            send_alert(
                "RETRAIN RESULT",
                {
                    "trigger": "data_drift",
                    "status": res.get("status"),
                    "message": res.get("message"),
                    "model_path": str(res.get("model_path", "")),
                },
            )
        except Exception as e:
            send_alert(
                "RETRAIN FAILED",
                {
                    "trigger": "data_drift",
                    "error": str(e),
                },
            )

    return {
        "status": overall_status,
        "n_logs": int(n),
        "features": report,
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
            "MODEL DRIFT",
            {
                "baseline_rmse": round(baseline_rmse, 3),
                "recent_rmse": round(recent_rmse, 3),
                "delta_rmse": round(delta, 3),
                "threshold": threshold,
                "n_records": n,
            },
        )
    else:
        status = "no_drift"


    if should_retrain() and status == "drift_detected":
        send_alert(
            "RETRAIN STARTED",
            {
                "trigger": "model_drift",
                "delta_rmse": round(delta, 3),
                "threshold": threshold,
                "n_records": int(n),
            },
        )
        try:
            res = auto_retrain_model()
            send_alert(
                "RETRAIN RESULT",
                {
                    "trigger": "model_drift",
                    "status": res.get("status"),
                    "message": res.get("message"),
                    "model_path": str(res.get("model_path", "")),
                },
            )
        except Exception as e:
            send_alert(
                "RETRAIN FAILED",
                {
                    "trigger": "model_drift",
                    "error": str(e),
                },
            )

    return {
        "status": status,
        "n_records": int(n),
        "baseline_rmse": baseline_rmse,
        "recent_rmse": recent_rmse,
        "delta_rmse": delta,
        "threshold": threshold,
    }

@app.get("/monitor/traffic")
def monitor_traffic(window_minutes: int = 10):
    """
    Simple traffic monitor: compares last `window_minutes` to the previous window.
    Sends alert on big drop or spike.
    """
    log_path = Path("logs") / "predictions.jsonl"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No prediction logs found yet")

    df = pd.read_json(log_path, lines=True)
    if df.empty:
        return {
            "status": "no_data",
            "detail": "No prediction logs to analyze.",
        }

    df["logged_at_utc"] = pd.to_datetime(
        df["logged_at_utc"], utc=True, errors="coerce"
    )

    now = datetime.now(UTC)
    window = pd.Timedelta(minutes=window_minutes)

    recent = df[df["logged_at_utc"] >= now - window]
    previous = df[
        (df["logged_at_utc"] < now - window) &
        (df["logged_at_utc"] >= now - 2 * window)
    ]

    recent_rpm = len(recent) / window_minutes
    prev_rpm = len(previous) / window_minutes if len(previous) > 0 else None

    status = "ok"
    info = {
        "window_minutes": window_minutes,
        "recent_count": int(len(recent)),
        "recent_rpm": recent_rpm,
        "previous_count": int(len(previous)),
        "previous_rpm": prev_rpm,
    }

    # only compare if we have a previous window
    if prev_rpm and prev_rpm > 0:
        ratio = recent_rpm / prev_rpm
        if ratio <= 0.2:
            status = "volume_drop"
            send_alert(
                "TRAFFIC DROP",
                {
                    **info,
                    "ratio": round(ratio, 3),
                },
            )
        elif ratio >= 5.0:
            status = "volume_spike"
            send_alert(
                "TRAFFIC SPIKE",
                {
                    **info,
                    "ratio": round(ratio, 3),
                },
            )

    info["status"] = status
    return info
