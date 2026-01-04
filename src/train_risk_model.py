import json
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import mlflow
from pathlib import Path


DATA_FILE  = Path("/opt/airflow/data/aq_clean.parquet")
MODEL_FILE = Path("/opt/airflow/models/risk_model.pkl")
META_FILE  = Path("/opt/airflow/models/metrics_baseline.json")

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"No clean data file at {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)

    # Drop rows with missing target or key features
    df = df.dropna(subset=["aqi_future_3h", "pm2_5", "pm10", "temp_c", "humidity"])
    return df


def build_features(df: pd.DataFrame):
    feature_cols = [
        "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3", "temp_c", "humidity",
        # Lag features:
        "aqi_lag1", "aqi_lag2", "aqi_lag3",
        "pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3",
        "pm10_lag1", "pm10_lag2", "pm10_lag3",
        "o3_lag1", "o3_lag2", "o3_lag3",
        "temp_c_lag1", "temp_c_lag2", "temp_c_lag3",
        "humidity_lag1", "humidity_lag2", "humidity_lag3",
    ]

    X = df[feature_cols]
    y = df["aqi_future_3h"]
    return X, y

def load_prev_rmse():
    if not META_FILE.exists():
        return None
    try:
        with open(META_FILE, "r") as f:
            meta = json.load(f)
        val = meta.get("rmse")
        return float(val) if val is not None else None
    except Exception:
        return None


def save_meta(rmse: float, n_rows: int):
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump({"rmse": rmse, "n_rows": n_rows}, f)


def train_and_maybe_save():

    mlflow.set_tracking_uri("file:/app/mlruns")
    mlflow.set_experiment("airflow_docker")

    df = load_data()
    n_rows = len(df)
    print(f"Loaded {n_rows} rows for training.")

    X, y = build_features(df)

    # Time-based split (not random)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    

    with mlflow.start_run():
        mlflow.log_param("model_type", "LGBMRegressor")
        mlflow.log_param("n_rows", len(df))

        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        print(f"LightGBM RMSE: {rmse:.4f}")

        mlflow.log_metric("rmse", rmse)
        mlflow.lightgbm.log_model(model, name="model")

        prev_rmse = load_prev_rmse()
        did_save = False
        saved_model_path = None

        if (prev_rmse is None) or (rmse <= prev_rmse):
            MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODEL_FILE)
            save_meta(rmse, n_rows)
            did_save = True
            saved_model_path = MODEL_FILE

            if prev_rmse is None:
                print(
                    f"✅ Saved model to {MODEL_FILE} "
                    f"(no previous RMSE, new_rmse={rmse:.4f})"
                )
            else:
                print(
                    f"✅ Saved model to {MODEL_FILE} "
                    f"(prev_rmse={prev_rmse:.4f}, new_rmse={rmse:.4f})"
                )
        else:
            print(
                f"⚠️ Not saving model: new_rmse={rmse:.4f} "
                f"is worse than prev_rmse={prev_rmse:.4f}"
            )

    return {
        "rmse": rmse,
        "prev_rmse": prev_rmse,
        "n_rows": n_rows,
        "did_save": did_save,
        "model_path": str(saved_model_path) if saved_model_path else None,
    }


def main():
    # Keep CLI behavior working
    train_and_maybe_save()


if __name__ == "__main__":
    main()