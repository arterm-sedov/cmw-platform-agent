"""UI-only Gradio chat bubbles for streaming tool and reasoning progress."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any
import uuid

UI_ONLY_TYPES = frozenset(
    {
        "tool_call",
        "reasoning",
        "thinking",
        "generating_answer",
    }
)


def short_uid() -> str:
    """Return a compact id suitable for per-turn UI metadata."""
    return str(uuid.uuid4())[:8]


def is_ui_only_message(message: Any) -> bool:
    """Return True when a Gradio message is a render-only progress bubble."""
    if not isinstance(message, MutableMapping):
        return False
    metadata = message.get("metadata")
    return (
        isinstance(metadata, MutableMapping)
        and metadata.get("ui_type") in UI_ONLY_TYPES
    )


def make_tool_call_bubble(
    tool_name: str,
    tool_call_id: str | None = None,
    *,
    title: str | None = None,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a pending Gradio thought bubble for a tool call."""
    bubble_id = tool_call_id or short_uid()
    safe_tool_name = tool_name or "unknown"
    return {
        "role": "assistant",
        "content": content or f"Calling {safe_tool_name}...",
        "metadata": {
            "title": title or f"Tool called: {safe_tool_name}",
            "ui_type": "tool_call",
            "status": "pending",
            "id": bubble_id,
            "tool_name": safe_tool_name,
            **(metadata or {}),
        },
    }


def upsert_tool_call_bubble(
    history: list[dict[str, Any]],
    tool_name: str,
    tool_call_id: str | None = None,
    *,
    title: str | None = None,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Ensure a pending tool bubble exists and return its id."""
    existing = _find_tool_message(history, tool_name, tool_call_id)
    bubble_id = (
        tool_call_id
        or _metadata_value(existing, "id")
        or short_uid()
    )
    if existing is not None:
        existing["content"] = content or existing.get("content", "")
        existing_metadata = existing.setdefault("metadata", {})
        if isinstance(existing_metadata, MutableMapping):
            existing_metadata["status"] = "pending"
            existing_metadata["tool_name"] = tool_name or existing_metadata.get(
                "tool_name", "unknown"
            )
            if title:
                existing_metadata["title"] = title
            if metadata:
                existing_metadata.update(metadata)
        return bubble_id

    history.append(
        make_tool_call_bubble(
            tool_name,
            bubble_id,
            title=title,
            content=content,
            metadata=metadata,
        )
    )
    return bubble_id


def complete_tool_call_bubble(
    history: list[dict[str, Any]],
    *,
    tool_name: str,
    tool_call_id: str | None = None,
    content: str = "",
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Mark a tool bubble done, appending a completed fallback if start was absent."""
    msg = _find_tool_message(history, tool_name, tool_call_id)
    bubble_id = tool_call_id or _metadata_value(msg, "id") or short_uid()
    found = msg is not None
    if msg is None:
        msg = make_tool_call_bubble(
            tool_name,
            bubble_id,
            title=title,
            content=content,
            metadata=metadata,
        )
        history.append(msg)

    msg["content"] = content
    msg_metadata = msg.setdefault("metadata", {})
    if isinstance(msg_metadata, MutableMapping):
        msg_metadata["status"] = "done"
        msg_metadata["tool_name"] = tool_name or msg_metadata.get(
            "tool_name", "unknown"
        )
        if title:
            msg_metadata["title"] = title
        if metadata:
            msg_metadata.update(metadata)
    return found


def make_generating_answer_bubble(
    *,
    bubble_id: str | None = None,
    title: str = "Generating answer",
    content: str = "Preparing the final response...",
) -> dict[str, Any]:
    """Create a pending answer-generation progress bubble."""
    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            "ui_type": "generating_answer",
            "status": "pending",
            "id": bubble_id or short_uid(),
        },
    }


def begin_turn_with_generating_answer(
    history: list[dict[str, Any]],
    user_message: str,
    *,
    bubble_id: str | None = None,
    title: str = "Generating answer",
    content: str = "Preparing the final response...",
) -> tuple[list[dict[str, Any]], str]:
    """Append user message and the first pending generating-answer bubble."""
    turn_history = [*history, {"role": "user", "content": user_message}]
    resolved_id = upsert_generating_answer_bubble(
        turn_history,
        bubble_id=bubble_id,
        title=title,
        content=content,
    )
    return turn_history, resolved_id


def upsert_generating_answer_bubble(
    history: list[dict[str, Any]],
    *,
    bubble_id: str | None = None,
    title: str = "Generating answer",
    content: str = "Preparing the final response...",
) -> str:
    """Ensure a pending answer-generation bubble exists."""
    resolved_id = bubble_id or short_uid()
    existing = _find_message_by_id(history, "generating_answer", resolved_id)
    if existing is not None:
        existing["content"] = content
        metadata = existing.setdefault("metadata", {})
        if isinstance(metadata, MutableMapping):
            metadata["title"] = title
            metadata["status"] = "pending"
        return resolved_id
    history.append(
        make_generating_answer_bubble(
            bubble_id=resolved_id,
            title=title,
            content=content,
        )
    )
    return resolved_id


def complete_generating_answer_bubble(
    history: list[dict[str, Any]],
    bubble_id: str | None,
) -> bool:
    """Remove transient answer-generation bubble(s), including stale duplicates."""
    removed = False
    if bubble_id is not None:
        index = _find_message_index_by_id(history, "generating_answer", bubble_id)
        if index is not None:
            del history[index]
            removed = True
    for index in range(len(history) - 1, -1, -1):
        msg = history[index]
        if not isinstance(msg, MutableMapping):
            continue
        metadata = msg.get("metadata")
        if not isinstance(metadata, MutableMapping):
            continue
        if metadata.get("ui_type") == "generating_answer":
            del history[index]
            removed = True
    return removed


def make_reasoning_bubble(
    content: str,
    *,
    bubble_id: str | None = None,
    title: str = "Reasoning",
) -> dict[str, Any]:
    """Create a pending reasoning bubble."""
    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            "ui_type": "reasoning",
            "status": "pending",
            "id": bubble_id or short_uid(),
        },
    }


def update_reasoning_bubble(
    history: list[dict[str, Any]],
    *,
    bubble_id: str,
    content: str,
    title: str = "Reasoning",
    max_lines: int = 12,
    max_chars: int = 4000,
) -> bool:
    """Update a reasoning bubble with a bounded tail of streamed text."""
    msg = _find_message_by_id(history, "reasoning", bubble_id)
    if msg is None:
        history.append(
            make_reasoning_bubble(
                _tail_text(content, max_lines=max_lines, max_chars=max_chars),
                bubble_id=bubble_id,
                title=title,
            )
        )
        return False
    msg["content"] = _tail_text(content, max_lines=max_lines, max_chars=max_chars)
    metadata = msg.setdefault("metadata", {})
    if isinstance(metadata, MutableMapping):
        metadata["status"] = "pending"
        metadata["title"] = title
    return True


def complete_reasoning_bubble(
    history: list[dict[str, Any]],
    bubble_id: str | None,
) -> bool:
    """Collapse a reasoning bubble."""
    msg = _find_message_by_id(history, "reasoning", bubble_id)
    if msg is None:
        return False
    metadata = msg.get("metadata")
    if isinstance(metadata, MutableMapping):
        metadata["status"] = "done"
    return True


def _find_message_by_id(
    history: list[dict[str, Any]],
    ui_type: str,
    message_id: str | None,
) -> dict[str, Any] | None:
    if not message_id:
        return None
    for msg in reversed(history):
        if not isinstance(msg, MutableMapping):
            continue
        metadata = msg.get("metadata")
        if not isinstance(metadata, MutableMapping):
            continue
        if metadata.get("ui_type") == ui_type and metadata.get("id") == message_id:
            return msg
    return None


def _find_message_index_by_id(
    history: list[dict[str, Any]],
    ui_type: str,
    message_id: str | None,
) -> int | None:
    if not message_id:
        return None
    for index in range(len(history) - 1, -1, -1):
        msg = history[index]
        if not isinstance(msg, MutableMapping):
            continue
        metadata = msg.get("metadata")
        if not isinstance(metadata, MutableMapping):
            continue
        if metadata.get("ui_type") == ui_type and metadata.get("id") == message_id:
            return index
    return None


def _find_tool_message(
    history: list[dict[str, Any]],
    tool_name: str,
    tool_call_id: str | None,
) -> dict[str, Any] | None:
    if tool_call_id:
        return _find_message_by_id(history, "tool_call", tool_call_id)
    for msg in reversed(history):
        if not isinstance(msg, MutableMapping):
            continue
        metadata = msg.get("metadata")
        if not isinstance(metadata, MutableMapping):
            continue
        if metadata.get("ui_type") != "tool_call":
            continue
        if metadata.get("status") != "pending":
            continue
        if metadata.get("tool_name") == (tool_name or "unknown"):
            return msg
    return None


def _metadata_value(message: dict[str, Any] | None, key: str) -> str | None:
    if not isinstance(message, MutableMapping):
        return None
    metadata = message.get("metadata")
    if not isinstance(metadata, MutableMapping):
        return None
    value = metadata.get(key)
    return str(value) if value else None


def _tail_text(text: str, *, max_lines: int, max_chars: int) -> str:
    clean = (text or "").strip()
    if not clean:
        return ""
    if max_lines > 0:
        lines = clean.splitlines()
        if len(lines) > max_lines:
            clean = "...\n" + "\n".join(lines[-max_lines:])
    if len(clean) > max_chars:
        clean = "..." + clean[-max_chars:]
    return clean
