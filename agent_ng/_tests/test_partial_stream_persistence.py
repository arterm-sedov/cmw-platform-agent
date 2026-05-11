"""Tests for partial stream text extraction and persistence during cancellation."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from agent_ng.native_langchain_streaming import NativeLangChainStreaming


class TestExtractPartialText:
    """Unit tests for _extract_partial_text."""

    def test_none_chunk_returns_none(self):
        assert NativeLangChainStreaming._extract_partial_text(None) is None

    def test_chunk_without_content_attr_returns_none(self):
        assert (
            NativeLangChainStreaming._extract_partial_text(object()) is None
        )

    def test_aimessage_with_none_content_returns_none(self):
        assert (
            NativeLangChainStreaming._extract_partial_text(
                AIMessage(content="")
            )
            is None
        )

    def test_aimessage_with_whitespace_content_returns_none(self):
        assert (
            NativeLangChainStreaming._extract_partial_text(
                AIMessage(content="   ")
            )
            is None
        )

    def test_aimessage_with_plain_text_returns_rstrip(self):
        result = NativeLangChainStreaming._extract_partial_text(
            AIMessage(content="  hello world\n  ")
        )
        assert result == "  hello world"

    def test_aimessage_with_markdown(self):
        result = NativeLangChainStreaming._extract_partial_text(
            AIMessage(content="## Title\n\nSome text\n")
        )
        assert result == "## Title\n\nSome text"


class TestPersistPartialReturnValue:
    """Integration-style tests for _persist_partial return value."""

    def test_returns_none_when_no_accumulated_chunk(self):
        streaming = NativeLangChainStreaming()
        agent = MagicMock()
        agent.memory_manager.get_conversation_history.return_value = []
        result = streaming._persist_partial(
            agent, "conv1", [], accumulated_chunk=None
        )
        assert result is None

    def test_returns_content_string_when_text_found(self):
        streaming = NativeLangChainStreaming()
        agent = MagicMock()
        agent.memory_manager.get_conversation_history.return_value = []
        chunk = AIMessage(content="partial response")
        result = streaming._persist_partial(
            agent,
            "conv1",
            [],
            accumulated_chunk=chunk,
            suffix=" \u26a1",
        )
        assert result == "partial response \u26a1"
        agent.memory_manager.add_message.assert_called()

    def test_returns_none_when_chunk_is_empty(self):
        streaming = NativeLangChainStreaming()
        agent = MagicMock()
        agent.memory_manager.get_conversation_history.return_value = []
        chunk = AIMessage(content="")
        result = streaming._persist_partial(
            agent, "conv1", [], accumulated_chunk=chunk
        )
        assert result is None
        agent.memory_manager.add_message.assert_not_called()

    def test_default_suffix_is_truncated(self):
        streaming = NativeLangChainStreaming()
        agent = MagicMock()
        agent.memory_manager.get_conversation_history.return_value = []
        chunk = AIMessage(content="hello")
        result = streaming._persist_partial(
            agent, "conv1", [], accumulated_chunk=chunk
        )
        assert result == "hello [truncated]"

    def test_skips_duplicate_message(self):
        streaming = NativeLangChainStreaming()
        agent = MagicMock()
        existing = AIMessage(content="hello [truncated]")
        agent.memory_manager.get_conversation_history.return_value = [existing]
        chunk = AIMessage(content="hello")
        result = streaming._persist_partial(
            agent, "conv1", [], accumulated_chunk=chunk
        )
        assert result == "hello [truncated]"
        agent.memory_manager.add_message.assert_not_called()

    def test_saves_tool_messages_from_working_list(self):
        streaming = NativeLangChainStreaming()
        agent = MagicMock()
        agent.memory_manager.get_conversation_history.return_value = []
        tool_msg = ToolMessage(content="result", tool_call_id="t1", name="foo")
        result = streaming._persist_partial(
            agent,
            "conv1",
            [tool_msg],
            accumulated_chunk=None,
        )
        assert result is None
        agent.memory_manager.add_message.assert_called_once()
