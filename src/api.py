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