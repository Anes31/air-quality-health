def aqi_to_label(aqi: int) -> str:
    aqi = round(aqi)
    mapping = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor",
    }
    return mapping.get(aqi, "Unknown")