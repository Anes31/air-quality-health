import os
import json
import pandas as pd

PRED_LOG = os.path.join("logs", "predictions.jsonl")
CLEAN_DATA = os.path.join("data", "aq_clean.parquet")
OUT_FILE = os.path.join("logs", "model_performance.jsonl")


def main():
    if not os.path.exists(PRED_LOG):
        print("No prediction log found.")
        return
    if not os.path.exists(CLEAN_DATA):
        print("No clean data file found.")
        return

    preds = pd.read_json(PRED_LOG, lines=True)
    df = pd.read_parquet(CLEAN_DATA)

    preds["timestamp_utc"] = pd.to_datetime(
        preds["timestamp_utc"], format="ISO8601", utc=True, errors="coerce"
    )
    preds["logged_at_utc"] = pd.to_datetime(
        preds["logged_at_utc"], format="ISO8601", utc=True, errors="coerce"
    )
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], format="ISO8601", utc=True, errors="coerce"
    )

    # Drop any rows that failed to parse
    preds = preds.dropna(subset=["timestamp_utc", "logged_at_utc"])
    df = df.dropna(subset=["timestamp_utc"])

    
    # Time we’re trying to hit with actuals (T + 3h)
    preds["target_time_utc"] = preds["timestamp_utc"] + pd.Timedelta(hours=3)

    records = []

    for city, p_sub in preds.groupby("city"):
        d_sub = df[df["city"] == city].sort_values("timestamp_utc")
        if d_sub.empty:
            continue

        # merge_asof: nearest timestamp <= target_time_utc
        merged = pd.merge_asof(
            p_sub.sort_values("target_time_utc"),
            d_sub[["timestamp_utc", "aqi"]],
            left_on="target_time_utc",
            right_on="timestamp_utc",
            direction="backward",
        )

        for _, row in merged.iterrows():
            if pd.isna(row.get("aqi")):
                continue
            pred = float(row["predicted_aqi_3h"])
            actual = float(row["aqi"])
            err = pred - actual
            abs_err = abs(err)
            record = {
                "city": row["city"],
                "prediction_made_at_utc": row["logged_at_utc"].isoformat(),
                "forecast_for_utc": row["timestamp_utc_x"].isoformat(),
                "actual_time_utc": row["timestamp_utc_y"].isoformat(),
                "predicted_aqi_3h": pred,
                "actual_aqi": actual,
                "error": err,
                "abs_error": abs_err,
            }
            records.append(record)

    if not records:
        print("No matches found between predictions and actuals yet.")
        return

    os.makedirs("logs", exist_ok=True)
    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} model performance records to {OUT_FILE}")


if __name__ == "__main__":
    main()
