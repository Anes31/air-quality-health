import os
import logging
from typing import Optional, Dict

import requests

logger = logging.getLogger("aq_api")


def format_alert_payload(title: str, details: Optional[Dict] = None) -> dict:
    lines = [f"**{title}**"]
    if details:
        for k, v in details.items():
            lines.append(f"- **{k}**: {v}")
    return {"content": "\n".join(lines)}


def send_alert(title: str, details: Optional[Dict] = None) -> None:
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if not webhook_url:
        logger.warning(
            f"Alert requested but ALERT_WEBHOOK_URL not set. "
            f"Title: {title}, details: {details}"
        )
        return

    payload = format_alert_payload(title, details)

    try:
        resp = requests.post(webhook_url, json=payload, timeout=5)
        if resp.status_code != 204:
            logger.error(f"Discord webhook error {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")