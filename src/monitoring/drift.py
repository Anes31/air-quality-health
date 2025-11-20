from typing import Dict, List
import pandas as pd
from src.monitoring.utils import compute_drift_metrics, psi_severity
from src.monitoring.alerts import send_alert
from src.auto_retrain import auto_retrain_model, should_retrain


def summarize_data_drift(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Given a sorted df, compute feature-level drift metrics + overall status.
    Uses first 20% as baseline and last 20% as recent.
    """
    n = len(df)
    baseline = df.iloc[: int(0.2 * n)]
    recent = df.iloc[int(0.8 * n):]

    report: Dict[str, Dict] = {}
    severity_map = {"no_drift": 0, "moderate_drift": 1, "significant_drift": 2}
    overall_level = 0

    for col in features:
        metrics = compute_drift_metrics(baseline, recent, col)
        sev = psi_severity(metrics["PSI"])
        metrics["severity"] = sev
        report[col] = metrics
        if sev in severity_map:
            overall_level = max(overall_level, severity_map[sev])

    inv_severity_map = {0: "no_drift", 1: "moderate_drift", 2: "significant_drift"}
    overall_status = inv_severity_map.get(overall_level, "no_drift")

    return {
        "overall_status": overall_status,
        "report": report,
        "n_logs": n,
    }


def maybe_alert_data_drift(overall_status: str, report: Dict, n_logs: int) -> None:
    """
    Send DATA DRIFT alert only when moderate or significant drift.
    """
    if overall_status not in ("moderate_drift", "significant_drift"):
        return

    drifted_features = [
        f"{name} (PSI={m['PSI']:.3f}, sev={m['severity']})"
        for name, m in report.items()
        if m.get("severity") in ("moderate_drift", "significant_drift")
    ]

    send_alert(
        "DATA DRIFT",
        {
            "severity": overall_status,
            "n_logs": int(n_logs),
            "drifted_features": ", ".join(drifted_features),
        },
    )


def maybe_auto_retrain(
    trigger: str,
    condition: bool,
    extra_details: Dict | None = None,
) -> None:
    """
    Shared retrain wrapper for data drift + model drift.
    """
    if not condition or not should_retrain():
        return

    base = {"trigger": trigger}
    if extra_details:
        base.update(extra_details)

    send_alert("RETRAIN STARTED", base)

    try:
        res = auto_retrain_model()
        send_alert(
            "RETRAIN RESULT",
            {
                **base,
                "status": res.get("status"),
                "message": res.get("message"),
                "model_path": str(res.get("model_path", "")),
            },
        )
    except Exception as e:
        send_alert(
            "RETRAIN FAILED",
            {
                **base,
                "error": str(e),
            },
        )
