"""Chat tab streaming event contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from agent_ng.tabs.chat_tab import ChatTab


def test_stream_message_wrapper_yields_declared_output_count() -> None:
    """The stream event has chatbot and textbox outputs, so yields must be pairs."""

    def stream_message(
        _message: str,
        history: list[dict[str, Any]],
        _request: Any,
    ) -> Iterator[tuple[list[dict[str, Any]], str]]:
        updated = list(history)
        updated.append({"role": "assistant", "content": "done"})
        yield updated, ""

    chat_tab = ChatTab({"stream_message": stream_message})

    chunks = list(
        chat_tab._stream_message_wrapper(
            {"text": "hello", "files": []},
            [],
            {"cancelled": False},
            None,
        )
    )

    assert chunks
    assert all(len(chunk) == 2 for chunk in chunks)
