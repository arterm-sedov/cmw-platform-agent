"""Plain text from LangChain messages (multimodal list content)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def visible_plain_text_from_message(msg: BaseMessage) -> str:
    """Join text-type content blocks; empty if none."""
    parts: list[str] = []
    for block in msg.content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts) if parts else ""


def memory_dedupe_fingerprint(msg: BaseMessage) -> tuple[str, str]:
    """Hashable key for memory dedup (never embed raw list/dict content)."""
    return (type(msg).__name__, visible_plain_text_from_message(msg))
