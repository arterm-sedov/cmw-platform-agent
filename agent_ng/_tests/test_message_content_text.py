"""Multimodal message content: plain text and hashable memory fingerprints."""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

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


def test_visible_text_tool_message_string_body() -> None:
    """LC 1.x ToolMessage payloads are usually plain strings, not blocks."""
    msg = ToolMessage(
        content='{"success": true, "file_reference": "out.png"}',
        tool_call_id="call_abc",
        name="generate_ai_image",
    )
    assert "file_reference" in visible_plain_text_from_message(msg)


def test_tool_message_fingerprints_differ_by_tool_call_id() -> None:
    body = '{"success": true}'
    a = ToolMessage(content=body, tool_call_id="id-1", name="t")
    b = ToolMessage(content=body, tool_call_id="id-2", name="t")
    assert memory_dedupe_fingerprint(a) != memory_dedupe_fingerprint(b)


def test_aimessage_tool_only_turns_differ_by_tool_call_ids() -> None:
    """Empty assistant text but distinct tool_calls must not dedupe together."""
    m1 = AIMessage(
        content="",
        tool_calls=[{"name": "x", "id": "tc-a", "args": {}}],
    )
    m2 = AIMessage(
        content="",
        tool_calls=[{"name": "x", "id": "tc-b", "args": {}}],
    )
    assert memory_dedupe_fingerprint(m1) != memory_dedupe_fingerprint(m2)
