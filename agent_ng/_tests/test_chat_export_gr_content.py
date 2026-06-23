"""Gradio Chatbot history shapes for Markdown export (Gradio 6 multimodal)."""

from __future__ import annotations

from agent_ng.tabs.chat_tab import _chatbot_message_content_to_export_text


def test_export_text_plain_string() -> None:
    assert _chatbot_message_content_to_export_text("  hello  ") == "hello"


def test_export_text_multimodal_list_typed_blocks() -> None:
    content = [
        {"type": "text", "text": "Part A"},
        {"type": "text", "text": "Part B"},
    ]
    assert _chatbot_message_content_to_export_text(content) == "Part A\n\nPart B"


def test_export_text_multimodal_list_text_only_dicts() -> None:
    """Some payloads omit ``type`` but include ``text``."""
    assert _chatbot_message_content_to_export_text([{"text": "x"}]) == "x"


def test_export_text_file_bubble_dict() -> None:
    out = _chatbot_message_content_to_export_text(
        {"path": "preview.png", "alt_text": "shot"}
    )
    assert "attachment" in out
    assert "shot" in out


def test_export_text_empty_list_returns_none() -> None:
    assert _chatbot_message_content_to_export_text([]) is None
