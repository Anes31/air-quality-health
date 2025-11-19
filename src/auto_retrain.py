import subprocess
import joblib
import os
import time

MODEL_PATH = os.path.join("models", "risk_model.pkl")

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
    Calls train_risk_model.py, which:
      - loads data
      - builds features
      - trains LightGBM
      - logs to MLflow
      - saves models/risk_model.pkl
    """
    # Run training script
    subprocess.run(["python", "train_risk_model.py"], check=True)

    # Load the newly trained model for hot-reload
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        return {
            "status": "success",
            "model_path": MODEL_PATH,
            "message": "Model retrained and saved."
        }
    else:
        return {
            "status": "error",
            "message": "Training script ran but model file was not created."
        }
