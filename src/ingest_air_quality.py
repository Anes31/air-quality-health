import time
import json
import requests
from datetime import datetime, timezone
import os

from config import OWM_API_KEY, CITIES

OUTPUT_FILE = os.path.join("data", "aq_raw.jsonl")

AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"


def fetch_air_quality(lat, lon):
    resp = requests.get(AIR_POLLUTION_URL, params={
        "lat": lat,
        "lon": lon,
        "appid": OWM_API_KEY
    })
    resp.raise_for_status()
    return resp.json()


def fetch_weather(lat, lon):
    resp = requests.get(WEATHER_URL, params={
        "lat": lat,
        "lon": lon,
        "appid": OWM_API_KEY
    })
    resp.raise_for_status()
    return resp.json()


def main(poll_interval=60):

    if not OWM_API_KEY:
        raise ValueError("Missing OWM_API_KEY. Add it to your .env file.")

    print(f"Starting air quality ingestion. Saving to: {OUTPUT_FILE}")

    while True:
        ts = datetime.now(timezone.utc).isoformat()

        for city, (lat, lon) in CITIES.items():
            try:
                aq_data = fetch_air_quality(lat, lon)
                weather_data = fetch_weather(lat, lon)

                main_weather = weather_data.get("main", {})
                temp_k = main_weather.get("temp")
                humidity = main_weather.get("humidity")

                temp_c = temp_k - 273.15 if temp_k is not None else None

                record = {
                    "timestamp_utc": ts,
                    "city": city,
                    "lat": lat,
                    "lon": lon,
                    "temp_c": temp_c,
                    "humidity": humidity,
                    "raw": aq_data,
                }

                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                print(f"[{ts}] {city} → saved")

            except Exception as e:
                print(f"[{ts}] Error for {city}: {e}")

        time.sleep(poll_interval)


if __name__ == "__main__":
    main()