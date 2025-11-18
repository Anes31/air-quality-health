import os
import requests

SYSTEM_PROMPT = (
    "You explain short-term air quality forecasts to non-technical people. "
    "Respond in 2 sentences maximum. Be clear, direct, and avoid filler. "
    "Mention health impact briefly, especially for sensitive groups."
)

# IMPORTANT:
# Docker container connects to host Ollama via this env var:
# export OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
        },
        timeout=30,
    )

    resp.raise_for_status()
    data = resp.json()

    # Ollama /api/chat format:
    # { "message": { "role": "assistant", "content": "..." } }
    return data.get("message", {}).get("content", "").strip()