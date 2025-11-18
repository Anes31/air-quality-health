import os
import requests

SYSTEM_PROMPT = (
    "You explain short-term air quality forecasts to non-technical people. "
    "Respond in 2 sentences maximum. Be clear, direct, and avoid filler. "
    "Mention health impact briefly, especially for sensitive groups."
)


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")


def explain_forecast(
    city: str,
    aqi_3h: float,
    label_3h: str,
    pm25: float,
    pm10: float,
    temp_c: float,
    humidity: float,
) -> str:
    user_prompt = (
        f"City: {city}\n"
        f"Forecast AQI in 3 hours: {aqi_3h:.2f} ({label_3h})\n"
        f"PM2.5: {pm25:.2f}, PM10: {pm10:.2f}\n"
        f"Temperature: {temp_c:.1f} °C, Humidity: {humidity:.0f}%\n\n"
        "Explain in 2–3 sentences what this means for people in the city, "
        "especially sensitive groups. Be specific but not alarmist."
    )

    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "max_tokens": 120,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()
