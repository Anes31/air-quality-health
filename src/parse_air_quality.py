import json
import os
import pandas as pd

RAW_FILE = os.path.join("data", "aq_raw.jsonl")
OUT_FILE = os.path.join("data", "aq_clean.parquet")


def parse_record(rec):
    """Extracts useful AQI fields from raw API record."""
    raw_data = rec["raw"]["list"][0]
    components = raw_data["components"]

    return {
        "timestamp_utc": rec["timestamp_utc"],
        "city": rec["city"],
        "aqi": raw_data["main"]["aqi"],
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "nh3": components.get("nh3"),
        "temp_c": rec.get("temp_c"),
        "humidity": rec.get("humidity"),
    }


def main():
    records = []

    if not os.path.exists(RAW_FILE):
        print(f"No raw file found at {RAW_FILE}")
        return

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            clean = parse_record(rec)
            records.append(clean)

    if not records:
        print("No records to parse.")
        return

    df = pd.DataFrame(records)
    # Sort by city + time
    df = df.sort_values(["city", "timestamp_utc"])
    # Create 3h future target (shift backwards: future value aligned with past row)
    df["aqi_future_3h"] = df.groupby("city")["aqi"].shift(-180)
    # Create lag features (1, 2, 3 past steps)
    lags = [1, 2, 3]
    for lag in lags:
        for col in ["aqi", "pm2_5", "pm10", "o3", "temp_c", "humidity"]:
            df[f"{col}_lag{lag}"] = df.groupby("city")[col].shift(lag)
    df = df.dropna()
    df.to_parquet(OUT_FILE, index=False)
    print(df['aqi_future_3h'])
    print(f"Saved clean data to {OUT_FILE}")
    # print(df.head())
    print(df.shape)


if __name__ == "__main__":
    main()