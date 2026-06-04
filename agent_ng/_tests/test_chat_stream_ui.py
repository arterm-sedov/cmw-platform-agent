"""Contracts for Gradio-native UI-only chat stream bubbles."""

from __future__ import annotations

from agent_ng.chat_stream_ui import (
    begin_turn_with_generating_answer,
    complete_generating_answer_bubble,
    complete_reasoning_bubble,
    complete_tool_call_bubble,
    is_ui_only_message,
    make_generating_answer_bubble,
    make_reasoning_bubble,
    make_tool_call_bubble,
    update_reasoning_bubble,
    upsert_generating_answer_bubble,
    upsert_tool_call_bubble,
)
from agent_ng.i18n_translations import format_translation


def test_tool_bubble_starts_pending_with_stable_metadata() -> None:
    msg = make_tool_call_bubble(
        tool_name="list_applications",
        tool_call_id="call-1",
        title="Tool called: list_applications",
    )

    assert msg["role"] == "assistant"
    assert msg["metadata"]["ui_type"] == "tool_call"
    assert msg["metadata"]["status"] == "pending"
    assert msg["metadata"]["id"] == "call-1"
    assert msg["metadata"]["tool_name"] == "list_applications"
    assert is_ui_only_message(msg) is True


def test_tool_completion_updates_existing_bubble_in_place() -> None:
    history = [
        make_tool_call_bubble("list_applications", "call-1"),
    ]

    updated = complete_tool_call_bubble(
        history,
        tool_name="list_applications",
        tool_call_id="call-1",
        content="Found 3 applications",
        metadata={"duplicate_count": 2, "tool_cost": 0.01},
    )

    assert updated is True
    assert len(history) == 1
    assert history[0]["content"] == "Found 3 applications"
    assert history[0]["metadata"]["status"] == "done"
    assert history[0]["metadata"]["duplicate_count"] == 2
    assert history[0]["metadata"]["tool_cost"] == 0.01


def test_tool_completion_appends_done_bubble_when_start_was_missing() -> None:
    history: list[dict] = []

    updated = complete_tool_call_bubble(
        history,
        tool_name="unknown_tool",
        tool_call_id="call-missing",
        content="Tool completed",
    )

    assert updated is False
    assert len(history) == 1
    assert history[0]["metadata"]["status"] == "done"
    assert history[0]["metadata"]["id"] == "call-missing"


def test_tool_upsert_does_not_duplicate_same_pending_bubble() -> None:
    history: list[dict] = []

    first_id = upsert_tool_call_bubble(history, "read_file", "call-2")
    second_id = upsert_tool_call_bubble(history, "read_file", "call-2")

    assert first_id == second_id == "call-2"
    assert len(history) == 1
    assert history[0]["metadata"]["status"] == "pending"


def test_tool_completion_matches_latest_pending_tool_when_id_is_missing() -> None:
    history: list[dict] = []

    bubble_id = upsert_tool_call_bubble(history, "read_file", None)
    found = complete_tool_call_bubble(
        history,
        tool_name="read_file",
        tool_call_id=None,
        content="Read complete",
    )

    assert found is True
    assert len(history) == 1
    assert history[0]["metadata"]["id"] == bubble_id
    assert history[0]["metadata"]["status"] == "done"


def test_begin_turn_puts_generating_answer_first_status_bubble() -> None:
    title = format_translation("generating_answer", "en")
    subtitle = format_translation("generating_answer_subtitle", "en")

    history, bubble_id = begin_turn_with_generating_answer(
        [],
        "hello",
        bubble_id="gen-turn",
        title=title,
        content=subtitle,
    )

    assert bubble_id == "gen-turn"
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1]["metadata"]["ui_type"] == "generating_answer"
    assert history[1]["metadata"]["status"] == "pending"
    assert history[1]["metadata"]["title"] == title
    assert history[1]["content"] == subtitle


def test_generating_answer_bubble_starts_and_completes() -> None:
    history: list[dict] = []

    bubble_id = upsert_generating_answer_bubble(history, bubble_id="gen-1")
    assert bubble_id == "gen-1"
    assert history[0]["metadata"]["ui_type"] == "generating_answer"
    assert history[0]["metadata"]["status"] == "pending"

    assert complete_generating_answer_bubble(history, bubble_id) is True
    assert history == []


def test_generating_answer_survives_simulated_first_content() -> None:
    """Answer tokens must not remove the phase bubble; only turn-end cleanup does."""
    title = format_translation("generating_answer", "en")
    subtitle = format_translation("generating_answer_subtitle", "en")
    history, bubble_id = begin_turn_with_generating_answer(
        [],
        "hello",
        bubble_id="gen-stream",
        title=title,
        content=subtitle,
    )

    history.append({"role": "assistant", "content": "Hel"})

    assert len(history) == 3
    assert history[1]["metadata"]["ui_type"] == "generating_answer"
    assert history[1]["metadata"]["status"] == "pending"
    assert history[1]["metadata"]["id"] == bubble_id
    assert history[2]["content"] == "Hel"

    assert complete_generating_answer_bubble(history, bubble_id) is True
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hel"},
    ]


def test_generating_answer_completion_is_noop_without_bubble() -> None:
    history: list[dict] = [{"role": "assistant", "content": "answer"}]

    assert complete_generating_answer_bubble(history, "gen-missing") is False
    assert history == [{"role": "assistant", "content": "answer"}]


def test_reasoning_bubble_updates_content_and_caps_tail() -> None:
    history = [make_reasoning_bubble("initial", bubble_id="reason-1")]
    long_text = "\n".join(f"line {i}" for i in range(20))

    assert update_reasoning_bubble(
        history,
        bubble_id="reason-1",
        content=long_text,
        max_lines=3,
    )

    assert history[0]["metadata"]["status"] == "pending"
    assert "line 19" in history[0]["content"]
    assert "line 0" not in history[0]["content"]

    assert complete_reasoning_bubble(history, "reason-1") is True
    assert history[0]["metadata"]["status"] == "done"


def test_plain_assistant_message_is_not_ui_only() -> None:
    assert is_ui_only_message({"role": "assistant", "content": "real answer"}) is False


def test_make_generating_answer_bubble_contract() -> None:
    title = format_translation("generating_answer", "en")
    subtitle = format_translation("generating_answer_subtitle", "en")
    msg = make_generating_answer_bubble(
        bubble_id="gen-2",
        title=title,
        content=subtitle,
    )

    assert msg["role"] == "assistant"
    assert msg["content"] == subtitle
    assert msg["metadata"] == {
        "title": title,
        "ui_type": "generating_answer",
        "status": "pending",
        "id": "gen-2",
    }


def test_generating_answer_completion_removes_stale_bubbles_without_id() -> None:
    history = [
        make_generating_answer_bubble(bubble_id="stale-1"),
        {"role": "assistant", "content": "Final answer"},
        make_generating_answer_bubble(bubble_id="stale-2"),
    ]

    assert complete_generating_answer_bubble(history, None) is True
    assert history == [{"role": "assistant", "content": "Final answer"}]


def test_generating_answer_i18n_russian_labels() -> None:
    assert format_translation("generating_answer", "ru") == "✨ Формирую ответ"
    assert (
        format_translation("generating_answer_subtitle", "ru")
        == "Готовлю финальный ответ..."
    )
