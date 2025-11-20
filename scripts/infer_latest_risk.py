import os
import pandas as pd
import joblib

from src.risk_labels import aqi_to_label

DATA_FILE = os.path.join("data", "aq_clean.parquet")
MODEL_FILE = os.path.join("models", "risk_model.pkl")


def load_latest_samples(n=5):
    df = pd.read_parquet(DATA_FILE)
    df = df.sort_values("timestamp_utc")
    return df.tail(n)


def build_features(df: pd.DataFrame):
    feature_cols = [
        "co", "no", "no2", "o3", "so2",
        "pm2_5", "pm10", "nh3", "temp_c", "humidity"
    ]
    return df[feature_cols]


def main():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"No model found at {MODEL_FILE}. Train it first.")

    df = load_latest_samples(n=5)
    X = build_features(df)

    model = joblib.load(MODEL_FILE)
    preds = model.predict(X)

    for (_, row), pred in zip(df.iterrows(), preds):
        label = aqi_to_label(int(pred))
        print(
            f"{row['timestamp_utc']} | {row['city']:12} | "
            f"predicted_aqi={int(pred)} ({label}), "
            f"pm2_5={row['pm2_5']}, pm10={row['pm10']}, temp_c={row['temp_c']}, humidity={row['humidity']}"
        )


if __name__ == "__main__":
    main()