from typing import Dict, Set

import pandas as pd

# Single source of truth for expected features
EXPECTED_FEATURE_COLS: Set[str] = {
    "co", "no", "no2", "o3", "so2",
    "pm2_5", "pm10", "nh3", "temp_c", "humidity",
    "aqi_lag1", "aqi_lag2", "aqi_lag3",
    "pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3",
    "pm10_lag1", "pm10_lag2", "pm10_lag3",
    "o3_lag1", "o3_lag2", "o3_lag3",
    "temp_c_lag1", "temp_c_lag2", "temp_c_lag3",
    "humidity_lag1", "humidity_lag2", "humidity_lag3",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = list(EXPECTED_FEATURE_COLS)
    return df[feature_cols]


def compute_schema_status(X: pd.DataFrame) -> Dict:
    current_cols = set(X.columns)
    missing = EXPECTED_FEATURE_COLS - current_cols
    extra = current_cols - EXPECTED_FEATURE_COLS

    status = "ok"
    if missing or extra:
        status = "mismatch"

    col_types = {col: str(dtype) for col, dtype in X.dtypes.items()}

    return {
        "status": status,
        "missing": missing,
        "extra": extra,
        "column_types": col_types,
    }
