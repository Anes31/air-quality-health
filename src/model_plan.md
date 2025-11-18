 # Air Quality Health Pipeline — Modeling Plan

## Model 1: AQI Health Risk Classifier
- Goal: convert pollutant data into a health-friendly risk class
- Type: Multi-class classification
- Input features: co, no, no2, o3, so2, pm2_5, pm10, nh3, temp, humidity
- Target: aqi (converted to categories 0–5)
- Model candidates: XGBoost, LightGBM, RandomForest
- Output: risk_score (0–5), risk_label ("Good", "Moderate", ...)

## Model 2: Short-Term AQI Forecast (3 hours)
- Goal: predict future AQI
- Type: Regression
- Input: past N pollutant values + temp + humidity + timestamp features
- Target: future aqi (3h ahead)
- Model candidates: GradientBoostingRegressor, ARIMA (simple), small LSTM (optional)
- Output: predicted_aqi_3h