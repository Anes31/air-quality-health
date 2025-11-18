import os
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import mlflow

DATA_FILE = os.path.join("data", "aq_clean.parquet")
MODEL_FILE = os.path.join("models", "risk_model.pkl")


def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"No clean data file at {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)

    # Drop rows with missing target or key features
    df = df.dropna(subset=["aqi_future_3h", "pm2_5", "pm10", "temp_c", "humidity"])
    return df

def build_features(df: pd.DataFrame):
    
    feature_cols = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3", "temp_c", "humidity",
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


def main():
    df = load_data()
    n_rows = len(df)
    print(f"Loaded {n_rows} rows for training.")

    X, y = build_features(df)

    # Time-based split (not random)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    mlflow.set_experiment("air_quality_risk_model")

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
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"LightGBM RMSE: {rmse:.4f}")

        mlflow.log_metric("rmse", rmse)

        # Log model to MLflow
        mlflow.lightgbm.log_model(model, name="model")

        # Save locally for the API
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        joblib.dump(model, MODEL_FILE)
        print(f"Saved model to {MODEL_FILE}")



if __name__ == "__main__":
    main()