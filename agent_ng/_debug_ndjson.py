"""Append one NDJSON line for debug session 392eaf (Gradio 6 stall bisect).

Hypothesis ids (grep ``debug-392eaf.log``):

- **H1** — per-tab ``create_tab`` wall ms.
- **H1S** — sidebar ``create_tab`` enter/exit ms.
- **H2** — whole ``gr.Tabs()`` block ms (often >> sum of H1: Gradio internals).
- **H3** — markdown export total ms / errors.
- **H4** — download button visibility branch (``md_only`` / ``md_html`` / …).
- **H5** — UI manager download handler enter/exit (queue vs handler wall).
- **H6** — export sub-phases: preamble, body loop, MD disk, HTML section.
- **H7** — HTML export phases (md→HTML, CSS, template, write) + ``html_bytes``.
- **H8** — merged ``refresh_sidebar_after_turn``: UI block timing + ``has_request``.
- **H9** — same merge path: token budget timing + ``has_request``.
- **H10** — download visibility enter (hist len, flag).
- **HC** — concurrency snapshot on first ``get_concurrency_config()`` load.
- **HQ** — ``queue_manager.configure_queue``: branch + kwargs applied to ``demo.queue``.
- **HL** — ``create_interface`` queue snapshot; ``main`` launch_enter (port / server).
- **H11** — export path session id from ``session_manager`` (not ``DebugStreamer``).
- **H13** — ``re_enable_textbox_and_hide_stop`` enter/exit (multimodal input freeze bisect).
"""

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
