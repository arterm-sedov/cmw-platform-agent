"""Append one NDJSON line for debug session 392eaf (tab build timing)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

_SESSION = "392eaf"
_LOG = "debug-392eaf.log"
_logger = logging.getLogger(__name__)


def debug_ndjson(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
) -> None:
    path = Path(__file__).resolve().parents[1] / _LOG
    payload = {
        "sessionId": _SESSION,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        _logger.warning("debug_ndjson write failed: %s", exc)
