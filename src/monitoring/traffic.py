from datetime import datetime, UTC
from typing import Dict

import pandas as pd

from src.monitoring.alerts import send_alert


def compute_traffic_info(df: pd.DataFrame, window_minutes: int) -> Dict:
    """
    Compute recent/previous traffic, rpm, and status for a given window.
    """
    now = datetime.now(UTC)
    window = pd.Timedelta(minutes=window_minutes)

    recent = df[df["logged_at_utc"] >= now - window]
    previous = df[
        (df["logged_at_utc"] < now - window)
        & (df["logged_at_utc"] >= now - 2 * window)
    ]

    recent_rpm = len(recent) / window_minutes
    prev_rpm = len(previous) / window_minutes if len(previous) > 0 else None

    status = "ok"
    ratio = None

    if prev_rpm and prev_rpm > 0:
        ratio = recent_rpm / prev_rpm
        if ratio <= 0.2:
            status = "volume_drop"
        elif ratio >= 5.0:
            status = "volume_spike"

    info: Dict = {
        "window_minutes": window_minutes,
        "recent_count": int(len(recent)),
        "recent_rpm": recent_rpm,
        "previous_count": int(len(previous)),
        "previous_rpm": prev_rpm,
        "status": status,
    }
    if ratio is not None:
        info["ratio"] = round(ratio, 3)

    return info


def maybe_alert_traffic(info: Dict) -> None:
    """
    Send alert on traffic drop/spike based on computed info.
    """
    status = info.get("status")
    if status not in ("volume_drop", "volume_spike"):
        return

    payload = dict(info)  # shallow copy
    send_alert(
        "TRAFFIC DROP" if status == "volume_drop" else "TRAFFIC SPIKE",
        payload,
    )
