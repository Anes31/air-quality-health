import os
from os import PathLike
from pathlib import Path
from dotenv import load_dotenv

# -------------------------
# Base paths / environment
# -------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
ENV_PATH: PathLike = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

# -------------------------
# Weather service config
# -------------------------
OWM_API_KEY = os.getenv("OWM_API_KEY")

CITIES = {
    "atlanta": (33.7490, -84.3880),
    "los_angeles": (34.0522, -118.2437),
    "new_york": (40.7128, -74.0060),
}

# -------------------------
# API data/model paths
# -------------------------
DATA_FILE = os.path.join("data", "aq_clean.parquet")
MODEL_FILE = os.path.join("models", "risk_model.pkl")

# -------------------------
# Monitoring config
# -------------------------
LOG_DIR = os.path.join("logs")
PREDICTIONS_LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")
MODEL_PERF_LOG_FILE = os.path.join(LOG_DIR, "model_performance.jsonl")

# latency alert threshold
LATENCY_THRESHOLD_SECONDS = float(
    os.getenv("LATENCY_THRESHOLD_SECONDS", "0.5")
)