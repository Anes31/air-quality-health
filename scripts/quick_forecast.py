import pandas as pd
import joblib

from src.config import DATA_FILE, MODEL_FILE
from src.monitoring.schema import build_features
from src.risk_labels import aqi_to_label


def load_latest_samples(n=5) -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    df = df.sort_values("timestamp_utc")
    return df.tail(n)


def main(n=5):
    df = load_latest_samples(n)

    # Build features using the same logic the API uses
    X = build_features(df)

    model = joblib.load(MODEL_FILE)
    preds = model.predict(X)

    print("\n=== QUICK FORECAST CHECK ===\n")
    for (_, row), pred in zip(df.iterrows(), preds):
        pred = float(pred)
        label = aqi_to_label(pred)

        print(
            f"{row['timestamp_utc']}  |  {row['city']:12}  |  "
            f"AQI={pred:5.1f} ({label})  |  "
            f"pm2_5={row['pm2_5']:6.1f}  pm10={row['pm10']:6.1f}  "
            f"temp={row['temp_c']:5.1f}°C  hum={row['humidity']:5.1f}%"
        )
    print("\n================================\n")


if __name__ == "__main__":
    main()