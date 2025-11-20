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
from src.monitoring.alerts import send_alert
from src.monitoring.latency import timed_call, maybe_alert_slow_call
from src.monitoring.schema import build_features, compute_schema_status
from src.monitoring.traffic import compute_traffic_info, maybe_alert_traffic
from src.monitoring.drift import summarize_data_drift, maybe_alert_data_drift, maybe_auto_retrain
from src.monitoring.logging import log_predictions
from src.config import DATA_FILE, MODEL_FILE, LATENCY_THRESHOLD_SECONDS, PREDICTIONS_LOG_FILE, MODEL_PERF_LOG_FILE

logger = logging.getLogger("aq_api")

def load_latest_per_city():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("No clean data file found.")

    df = pd.read_parquet(DATA_FILE)
    if df.empty:
        raise ValueError("Clean data is empty.")

    df = df.sort_values("timestamp_utc")
    latest = df.groupby("city").tail(1)
    return latest


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
        schema_info = compute_schema_status(X)
        if schema_info["status"] == "mismatch":
            missing = schema_info["missing"]
            extra = schema_info["extra"]

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
        preds, latency_s = timed_call(app.state.model.predict, X)

        # alert on slow calls
        maybe_alert_slow_call(
            endpoint="/forecast/3h/explain",
            latency_s=latency_s,
            threshold_s=LATENCY_THRESHOLD_SECONDS,
            n_rows=len(X),
        )

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

    schema_info = compute_schema_status(X)
    status = schema_info["status"]
    missing = schema_info["missing"]
    extra = schema_info["extra"]
    col_types = schema_info["column_types"]

    if status == "mismatch":
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
    log_path = Path(PREDICTIONS_LOG_FILE)

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
    df["logged_at_utc"] = pd.to_datetime(
        df["logged_at_utc"],
        format="ISO8601",
        utc=True,
        errors="coerce",
    )
    df = df.sort_values("logged_at_utc")

    features = ["pm2_5", "pm10", "temp_c", "humidity", "predicted_aqi_3h"]

    summary = summarize_data_drift(df=df, features=features)
    overall_status = summary["overall_status"]
    report = summary["report"]
    n = summary["n_logs"]

    maybe_alert_data_drift(overall_status, report, n_logs=n)

    maybe_auto_retrain(
        trigger="data_drift",
        condition=overall_status in ("moderate_drift", "significant_drift"),
        extra_details={
            "overall_status": overall_status,
            "n_logs": int(n),
        },
    )

    return {
        "status": overall_status,
        "n_logs": int(n),
        "features": report,
    }



@app.get("/monitor/model_drift")
def monitor_model():
    perf_file = MODEL_PERF_LOG_FILE

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


    maybe_auto_retrain(
        trigger="model_drift",
        condition=status == "drift_detected",
        extra_details={
            "delta_rmse": round(delta, 3),
            "threshold": threshold,
            "n_records": int(n),
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
    log_path = Path(PREDICTIONS_LOG_FILE)

    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No prediction logs found yet")

    df = pd.read_json(log_path, lines=True)
    if df.empty:
        return {
            "status": "no_data",
            "detail": "No prediction logs to analyze.",
        }

    df["logged_at_utc"] = pd.to_datetime(df["logged_at_utc"], utc=True, errors="coerce")

    info = compute_traffic_info(df, window_minutes=window_minutes)
    maybe_alert_traffic(info)
    return info
