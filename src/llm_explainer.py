import os
import requests

SYSTEM_PROMPT = (
    "You explain short-term air quality forecasts to non-technical people. "
    "Respond in 2 sentences maximum. Be clear, direct, and avoid filler. "
    "Mention health impact briefly, especially for sensitive groups."
)

# If these are NOT set, we will fall back to a simple built-in explanation.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")  # no default → disabled on server
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")        # no default → disabled on server


def _fallback_explanation(
    city: str,
    aqi_3h: float,
    label_3h: str,
    pm25: float,
    pm10: float,
    temp_c: float,
    humidity: float,
) -> str:
    # Very simple, deterministic explanation if LLM is not configured or fails.
    if label_3h.lower().startswith(("unhealthy", "very unhealthy", "hazardous")):
        advice = (
            "People with asthma, heart, or lung conditions and children "
            "should limit long or intense outdoor activity."
        )
    elif "moderate" in label_3h.lower():
        advice = (
            "Most people can go about normal activities, but very sensitive "
            "groups should watch for symptoms."
        )
    else:
        advice = "Air quality is good and safe for normal outdoor activities for most people."

    return (
        f"In {city}, air quality in about 3 hours is expected to be {label_3h} "
        f"(AQI around {aqi_3h:.0f}). {advice}"
    )


def explain_forecast(
    city: str,
    aqi_3h: float,
    label_3h: str,
    pm25: float,
    pm10: float,
    temp_c: float,
    humidity: float,
) -> str:

    # If Ollama is not configured (e.g., on the server), use fallback.
    if not OLLAMA_BASE_URL or not OLLAMA_MODEL:
        return _fallback_explanation(
            city, aqi_3h, label_3h, pm25, pm10, temp_c, humidity
        )

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

    try:
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
        content = data.get("message", {}).get("content", "") or ""
        content = content.strip()
        if not content:
            # If LLM returns empty, still fall back.
            return _fallback_explanation(
                city, aqi_3h, label_3h, pm25, pm10, temp_c, humidity
            )
        return content
    except requests.RequestException:
        # If Ollama is down / unreachable, fall back gracefully.
        return _fallback_explanation(
            city, aqi_3h, label_3h, pm25, pm10, temp_c, humidity
        )