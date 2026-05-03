"""Plain text from LangChain messages (multimodal list content).

Streaming/UI may use :func:`visible_plain_text_from_message` to extract readable
text without mutating stored :class:`~langchain_core.messages.BaseMessage`
instances.

Memory deduplication uses :func:`memory_dedupe_fingerprint` only to decide
whether an equivalent message already exists in persistence — it must **not**
collapse distinct tool results or tool-call turns (regression fixed vs LC 1.x
empty ``content_blocks`` on ``ToolMessage``).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def visible_plain_text_from_message(msg: BaseMessage) -> str:
    """Join text-type content blocks; fall back to plain string body.

    ``ToolMessage`` and similar often use string ``content`` without multimodal
    ``content_blocks``; block-only extraction would wrongly yield empty text.
    """
    parts: list[str] = []
    for block in msg.content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    if parts:
        return "\n".join(parts)
    raw = getattr(msg, "content", None)
    if isinstance(raw, str) and raw.strip():
        return raw
    return ""


def _tool_call_ids_tuple(msg: AIMessage) -> tuple[str, ...]:
    tc = getattr(msg, "tool_calls", None) or ()
    ids: list[str] = []
    for item in tc:
        if isinstance(item, dict):
            ids.append(str(item.get("id") or ""))
        else:
            ids.append(str(getattr(item, "id", "") or ""))
    return tuple(ids)


def memory_dedupe_fingerprint(msg: BaseMessage) -> tuple[Any, ...]:
    """Hashable key for memory dedupe only — does not alter stored messages.

    Distinct ``ToolMessage`` rows must not share one key when visible block text
    is empty. Assistant turns that only contain ``tool_calls`` must not all read
    as duplicates when textual content is empty.
    """
    cls_name = type(msg).__name__
    text = visible_plain_text_from_message(msg)
    if isinstance(msg, ToolMessage):
        tid = str(getattr(msg, "tool_call_id", "") or "")
        return (cls_name, tid, text)
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        return (cls_name, text, _tool_call_ids_tuple(msg))
    return (cls_name, text)
