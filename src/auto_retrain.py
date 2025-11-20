import os
import joblib
import time
from pathlib import Path
from src.train_risk_model import train_and_maybe_save

BASE_DIR = Path(__file__).resolve().parent.parent   # /app
MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"


LAST_RETRAIN_TS = 0
RETRAIN_COOLDOWN = 60 * 60  # 1 hour
RETRAIN_IN_PROGRESS = False


def should_retrain():
    global LAST_RETRAIN_TS
    now = time.time()

    if now - LAST_RETRAIN_TS < RETRAIN_COOLDOWN:
        return False

    LAST_RETRAIN_TS = now
    return True


def auto_retrain_model():
    """
    Trains the model and returns RMSE info + save status.
    """
    res = train_and_maybe_save()

    if res["did_save"] and res.get("model_path") and os.path.exists(res["model_path"]):
        _ = joblib.load(res["model_path"])
        status = "success"
        message = "Model retrained and saved."
    else:
        status = "no_save"
        message = "Model trained but not saved (RMSE worse than previous)."

    return {
        "status": status,
        "model_path": res.get("model_path"),
        "message": message,
        "prev_rmse": res.get("prev_rmse"),
        "new_rmse": res.get("rmse"),
        "n_rows": res.get("n_rows"),
    }