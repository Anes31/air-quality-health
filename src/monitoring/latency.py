import time
from typing import Any, Callable, Tuple, Optional

from src.monitoring.alerts import send_alert


def timed_call(func: Callable[..., Any], *args, **kwargs) -> Tuple[Any, float]:
    """
    Run func(*args, **kwargs) and return (result, latency_seconds).
    """
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    latency_s = time.perf_counter() - t0
    return result, latency_s


def maybe_alert_slow_call(
    endpoint: str,
    latency_s: float,
    threshold_s: float,
    n_rows: Optional[int] = None,
) -> None:
    """
    Send 'HIGH LATENCY' alert if latency_s > threshold_s.
    """
    if latency_s <= threshold_s:
        return

    details = {
        "endpoint": endpoint,
        "latency_seconds": round(latency_s, 3),
    }
    if n_rows is not None:
        details["n_rows"] = int(n_rows)

    send_alert("HIGH LATENCY", details)