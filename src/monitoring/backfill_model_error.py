import os
import json
import argparse
import sys
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--models_dir", required=True)  # kept for symmetry / future use
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir
    logs_dir = os.path.join(data_dir, "logs")

    pred_log = os.path.join(logs_dir, "predictions.jsonl")
    clean_data = os.path.join(data_dir, "aq_clean.parquet")
    out_file = os.path.join(logs_dir, "model_performance.jsonl")

    if not os.path.exists(pred_log):
        print(f"Missing prediction log: {pred_log}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(clean_data):
        print(f"Missing clean data file: {clean_data}", file=sys.stderr)
        sys.exit(1)

    preds = pd.read_json(pred_log, lines=True)
    df = pd.read_parquet(clean_data)

    preds["timestamp_utc"] = pd.to_datetime(
        preds["timestamp_utc"], utc=True, errors="coerce"
    )
    preds["logged_at_utc"] = pd.to_datetime(
        preds["logged_at_utc"], utc=True, errors="coerce"
    )
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )

    preds = preds.dropna(subset=["timestamp_utc", "logged_at_utc"])
    df = df.dropna(subset=["timestamp_utc"])

    preds["target_time_utc"] = preds["timestamp_utc"] + pd.Timedelta(hours=3)

    records = []

    for city, p_sub in preds.groupby("city"):
        d_sub = df[df["city"] == city].sort_values("timestamp_utc")
        if d_sub.empty:
            continue

        merged = pd.merge_asof(
            p_sub.sort_values("target_time_utc"),
            d_sub[["timestamp_utc", "aqi"]],
            left_on="target_time_utc",
            right_on="timestamp_utc",
            direction="backward",
        )

        for _, row in merged.iterrows():
            if pd.isna(row["aqi"]):
                continue

            pred = float(row["predicted_aqi_3h"])
            actual = float(row["aqi"])
            err = pred - actual

            records.append(
                {
                    "city": row["city"],
                    "prediction_made_at_utc": row["logged_at_utc"].isoformat(),
                    "forecast_for_utc": row["timestamp_utc_x"].isoformat(),
                    "actual_time_utc": row["timestamp_utc_y"].isoformat(),
                    "predicted_aqi_3h": pred,
                    "actual_aqi": actual,
                    "error": err,
                    "abs_error": abs(err),
                }
            )

    if not records:
        print("No matches found yet.")
        return

    os.makedirs(logs_dir, exist_ok=True)

    with open(out_file, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} records to {out_file}")


if __name__ == "__main__":
    main()