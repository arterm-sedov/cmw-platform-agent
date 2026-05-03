"""Multimodal message content: plain text and hashable memory fingerprints."""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from agent_ng.message_content_text import (
    memory_dedupe_fingerprint,
    visible_plain_text_from_message,
)


def test_visible_text_from_gemini_like_list_content() -> None:
    content = [
        {"type": "text", "text": "Hello", "extras": {"foo": 1}},
        {"type": "text", "text": "world"},
    ]
    msg = AIMessage(content=content)
    assert visible_plain_text_from_message(msg) == "Hello\nworld"


def test_visible_text_string_content_unchanged_via_blocks() -> None:
    msg = HumanMessage(content="plain")
    assert visible_plain_text_from_message(msg) == "plain"


def test_visible_text_skips_nontext_blocks() -> None:
    msg = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "think"},
            {"type": "text", "text": "answer"},
        ]
    )
    assert visible_plain_text_from_message(msg) == "answer"


def test_memory_fingerprint_set_no_unhashable_error() -> None:
    msgs = [
        AIMessage(content=[{"type": "text", "text": "a"}]),
        AIMessageChunk(content=[{"type": "text", "text": "b"}]),
        HumanMessage(content="c"),
    ]
    keys = {memory_dedupe_fingerprint(m) for m in msgs}
    assert len(keys) == 3
