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
# Paths (LOCAL DEFAULT, OVERRIDABLE IN DOCKER/AIRFLOW)
# -------------------------
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
MODELS_DIR = os.getenv("MODELS_DIR", str(BASE_DIR / "models"))
LOGS_DIR = os.getenv("LOGS_DIR", str(BASE_DIR / "data" / "logs"))

DATA_FILE = os.path.join(DATA_DIR, "aq_clean.parquet")
MODEL_FILE = os.path.join(MODELS_DIR, "risk_model.pkl")

PREDICTIONS_LOG_FILE = os.path.join(LOGS_DIR, "predictions.jsonl")
MODEL_PERF_LOG_FILE = os.path.join(LOGS_DIR, "model_performance.jsonl")

LATENCY_THRESHOLD_SECONDS = float(os.getenv("LATENCY_THRESHOLD_SECONDS", "0.5"))
