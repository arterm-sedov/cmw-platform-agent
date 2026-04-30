"""
File-attachment helpers for inline chat rendering.

These functions are the single source of truth for converting a tool's
``file_reference`` result into the Gradio 5 ``{"path": …, "alt_text": …}``
dict-content format that renders images (and other files) inline in the chat.

Used by:
- ``agent_ng/native_langchain_streaming.py`` — enriches ``tool_end`` event
  metadata with a resolved ``file_attachment`` dict.
- ``agent_ng/app_ng_modular.py`` — appends inline file bubble + caption to
  ``working_history`` when the attachment is present.
- ``agent_ng/tabs/chat_tab.py`` — symmetric user-side rendering for uploaded
  files (prepends a preview bubble before the ``[Files: …]`` text).

No vendor names, no model slugs, no infrastructure references are surfaced.
The LLM's view of tool results (the stringified dict) is not changed here.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
# Formatting helpers                                                         #
# ---------------------------------------------------------------------------#


def _fmt_size(num_bytes: int) -> str:
    """Human-readable file size, matching the style used for user uploads."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} GB"


# ---------------------------------------------------------------------------#
# Core: streaming layer — resolve tool result → attachment dict             #
# ---------------------------------------------------------------------------#


def build_file_attachment(
    tool_result: Any,
    agent: Any,
) -> dict[str, Any] | None:
    """Try to resolve a tool result into a file-attachment descriptor.

    Returns a dict ``{"path": <abs_str>, "display_name": <str>, "size_bytes":
    <int>}`` when the result carries a resolvable, on-disk ``file_reference``.
    Returns ``None`` in all other cases — callers must treat ``None`` as "no
    attachment" and behave exactly as before.

    Args:
        tool_result: The raw return value of a LangChain tool invocation.
        agent: The ``CmwAgent`` instance (or ``None``) that was injected into
            the tool call. Must have a ``get_file_path(name) -> str | None``
            method.
    """
    # Only dicts with success=True and a non-empty file_reference qualify.
    if not isinstance(tool_result, dict):
        return None
    if not tool_result.get("success"):
        return None
    ref = tool_result.get("file_reference")
    if not ref or not isinstance(ref, str):
        return None

    # Resolve the logical name to an on-disk path via the agent registry.
    if agent is None or not callable(getattr(agent, "get_file_path", None)):
        return None
    abs_path_str = agent.get_file_path(ref)
    if not abs_path_str:
        return None

    # Guard: file must actually exist (could have been cleaned up).
    path = Path(abs_path_str)
    if not path.is_file():
        logger.debug("file_attachment: resolved path does not exist: %s", abs_path_str)
        return None

    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    return {
        "path": str(path.resolve()),
        "display_name": ref,
        "size_bytes": size,
    }


# ---------------------------------------------------------------------------#
# Core: app layer — attachment dict → Gradio chat messages                  #
# ---------------------------------------------------------------------------#


def build_file_bubbles(attachment: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Convert a file-attachment descriptor into assistant-side chat messages.

    Returns a list of zero, one or two messages ready to extend
    ``working_history`` in ``app_ng_modular.py``:

    - Empty list when ``attachment`` is None or the path no longer exists.
    - Two messages when the file is present:
      1. Inline file bubble — ``{"role": "assistant", "content": {"path": …,
         "alt_text": …}}`` — Gradio 5 renders this as an image preview or
         file-chip depending on extension.
      2. Caption line — ``{"role": "assistant", "content": "📎 name — size"}``
         — visible text for accessibility / copy-paste / LLM reference.
    """
    if not attachment:
        return []
    abs_path = attachment.get("path", "")
    if not abs_path or not Path(abs_path).is_file():
        return []
    display_name = attachment.get("display_name", Path(abs_path).name)
    size_bytes = attachment.get("size_bytes") or 0
    caption = f"📎 {display_name} — {_fmt_size(size_bytes)}"
    return [
        {
            "role": "assistant",
            "content": {"path": abs_path, "alt_text": display_name},
        },
        {
            "role": "assistant",
            "content": caption,
        },
    ]


def build_file_bubbles_for_role(
    attachment: dict[str, Any] | None,
    role: str = "user",
) -> list[dict[str, Any]]:
    """Convert a file-attachment descriptor into a single inline file bubble.

    Used for the **user side** (uploaded files shown as previews in the user
    message bubble). Unlike the assistant side, there is no separate caption
    line — the existing ``[Files: name (size)]`` text already provides that.

    Returns an empty list when the attachment is None or the file is missing.
    """
    if not attachment:
        return []
    abs_path = attachment.get("path", "")
    if not abs_path or not Path(abs_path).is_file():
        return []
    display_name = attachment.get("display_name", Path(abs_path).name)
    return [
        {
            "role": role,
            "content": {"path": abs_path, "alt_text": display_name},
        }
    ]


def is_file_bubble(message: dict[str, Any]) -> bool:
    """Return True when *message* is a Gradio file-rendering bubble.

    File bubbles have ``content`` as a dict (``{"path": …, "alt_text": …}``)
    rather than a string. They are Gradio UI-only instructions and must be
    stripped out before any code iterates history for LLM input, token
    counting, or download-as-text serialization.

    Usage::

        history_for_llm = [m for m in working_history if not is_file_bubble(m)]
    """
    return isinstance(message.get("content"), dict)


__all__ = [
    "build_file_attachment",
    "build_file_bubbles",
    "build_file_bubbles_for_role",
    "is_file_bubble",
]
