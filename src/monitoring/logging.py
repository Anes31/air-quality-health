import json
import logging
import os
from datetime import datetime, UTC
from typing import List, Dict, Any

from src.config import PREDICTIONS_LOG_FILE

logger = logging.getLogger("aq_api")


def log_predictions(results: List[Dict[str, Any]], endpoint: str) -> None:
    """Append predictions to a JSONL log file + log a short summary."""
    os.makedirs(os.path.dirname(PREDICTIONS_LOG_FILE), exist_ok=True)
    now = datetime.now(UTC).isoformat()

    with open(PREDICTIONS_LOG_FILE, "a", encoding="utf-8") as f:
        for r in results:
            record = {
                "logged_at_utc": now,
                "endpoint": endpoint,
                **r,
            }
            f.write(json.dumps(record, default=str) + "\n")

    logger.info(
        f"{endpoint}: logged {len(results)} predictions → {PREDICTIONS_LOG_FILE}"
    )
