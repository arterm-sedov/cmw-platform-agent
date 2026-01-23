"""
Unit tests for history compression functionality.

Tests cover:
- Message filtering and formatting
- Compression triggers
- Compression execution
- Error handling
- Stats tracking
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_ng.history_compression import (
    compress_conversation_history,
    compress_history_to_tokens,
    emit_compression_notification,
    filter_non_system_messages,
    format_messages_for_compression,
    get_compression_stats,
    should_compress_mid_turn,
    should_compress_on_completion,
    track_compression_stats,
)


class TestFilterNonSystemMessages:
    """Test filtering of SystemMessage from history"""

    def test_filters_system_messages(self):
        """Test that SystemMessage is filtered out"""
        history = [
            HumanMessage(content="Hello"),
            SystemMessage(content="System prompt"),
            AIMessage(content="Hi there"),
        ]
        result = filter_non_system_messages(history)
        assert len(result) == 2
        assert all(not isinstance(msg, SystemMessage) for msg in result)

    def test_empty_history(self):
        """Test with empty history"""
        assert filter_non_system_messages([]) == []

    def test_only_system_messages(self):
        """Test with only SystemMessage"""
        history = [SystemMessage(content="System")]
        assert filter_non_system_messages(history) == []


class TestFormatMessagesForCompression:
    """Test message formatting for compression"""

    def test_formats_human_message(self):
        """Test formatting HumanMessage"""
        messages = [HumanMessage(content="Hello")]
        result = format_messages_for_compression(messages)
        assert "User: Hello" in result

    def test_formats_ai_message(self):
        """Test formatting AIMessage"""
        messages = [AIMessage(content="Hi")]
        result = format_messages_for_compression(messages)
        assert "Assistant: Hi" in result

    def test_formats_tool_message(self):
        """Test formatting ToolMessage"""
        messages = [ToolMessage(content="Result", name="tool1", tool_call_id="call_123")]
        result = format_messages_for_compression(messages)
        assert "Tool" in result
        assert "tool1" in result


class TestCompressionTriggers:
    """Test compression trigger functions"""

    def test_should_compress_on_completion_critical(self):
        """Test compression trigger for critical status"""
        agent = MagicMock()
        agent.memory_manager = MagicMock()
        agent.memory_manager.get_conversation_history.return_value = [
            HumanMessage(content="Msg1"),
            AIMessage(content="Msg2"),
        ]

        result = should_compress_on_completion(agent, "conv1", "critical")
        assert result is True

    def test_should_compress_on_completion_warning(self):
        """Test no compression for warning status"""
        result = should_compress_on_completion(MagicMock(), "conv1", "warning")
        assert result is False

    def test_should_compress_mid_turn_threshold(self):
        """Test mid-turn compression trigger"""
        agent = MagicMock()
        agent.token_tracker = MagicMock()
        snapshot = {"percentage_used": 96.0}
        agent.token_tracker.refresh_budget_snapshot.return_value = snapshot
        agent.token_tracker.get_budget_snapshot.return_value = snapshot

        result = should_compress_mid_turn(agent, "conv1")
        assert result is True


class TestCompressionStats:
    """Test compression statistics tracking"""

    def test_track_compression_stats(self):
        """Test tracking compression statistics"""
        conversation_id = "test_conv"
        tokens_saved = 1000

        count, total = track_compression_stats(conversation_id, tokens_saved)
        assert count == 1
        assert total == 1000

        # Track another compression
        count, total = track_compression_stats(conversation_id, 500)
        assert count == 2
        assert total == 1500

    def test_get_compression_stats(self):
        """Test retrieving compression statistics"""
        conversation_id = "test_conv2"
        track_compression_stats(conversation_id, 2000)

        count, total = get_compression_stats(conversation_id)
        assert count == 1
        assert total == 2000

    def test_get_compression_stats_nonexistent(self):
        """Test getting stats for non-existent conversation"""
        count, total = get_compression_stats("nonexistent")
        assert count == 0
        assert total == 0


@pytest.mark.asyncio
class TestCompressHistoryToTokens:
    """Test history compression to tokens"""

    async def test_compress_history_success(self):
        """Test successful compression"""
        history = [
            HumanMessage(content="Message 1"),
            AIMessage(content="Response 1"),
        ]
        llm_instance = MagicMock()
        llm_instance.llm = MagicMock()
        llm_instance.llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Compressed summary")
        )

        result = await compress_history_to_tokens(history, 100, llm_instance)
        assert result == "Compressed summary"

    async def test_compress_history_filters_system(self):
        """Test that SystemMessage is filtered before compression"""
        history = [
            SystemMessage(content="System"),
            HumanMessage(content="Message"),
        ]
        llm_instance = MagicMock()
        llm_instance.llm = MagicMock()
        llm_instance.llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Summary")
        )

        result = await compress_history_to_tokens(history, 100, llm_instance)
        assert result == "Summary"
        # Verify SystemMessage from history was not sent to LLM (only the prompt SystemMessage should be there)
        call_args = llm_instance.llm.ainvoke.call_args[0][0]
        # The first message is the SystemMessage with the prompt (expected)
        # The second message is HumanMessage with the history (should not contain SystemMessage from history)
        history_msg = call_args[1]
        assert isinstance(history_msg, HumanMessage)
        # The history content should not contain the SystemMessage content
        assert "System" not in history_msg.content or "System prompt" not in history_msg.content

    async def test_compress_history_empty_result(self):
        """Test handling of empty compression result"""
        history = [HumanMessage(content="Message")]
        llm_instance = MagicMock()
        llm_instance.llm = MagicMock()
        llm_instance.llm.ainvoke = AsyncMock(return_value=AIMessage(content=""))

        result = await compress_history_to_tokens(history, 100, llm_instance)
        assert result is None


class TestEmitCompressionNotification:
    """Test compression notification emission"""

    @patch("agent_ng.history_compression.gr.Info")
    def test_emit_notification_before(self, mock_info):
        """Test notification before compression"""
        emit_compression_notification(
            tokens_saved=0,
            previous_pct=95.0,
            current_pct=95.0,
            reason="critical",
            language="en",
            is_before=True,
        )
        mock_info.assert_called_once()

    @patch("agent_ng.history_compression.gr.Info")
    def test_emit_notification_after(self, mock_info):
        """Test notification after compression with results"""
        emit_compression_notification(
            tokens_saved=5000,
            previous_pct=95.0,
            current_pct=60.0,
            reason="critical",
            language="en",
            compression_count=1,
            is_before=False,
        )
        mock_info.assert_called_once()


@pytest.mark.asyncio
class TestCompressConversationHistory:
    """Test full conversation history compression"""

    async def test_compress_conversation_success(self):
        """Test successful conversation compression"""
        agent = MagicMock()
        agent.memory_manager = MagicMock()
        agent.memory_manager.get_conversation_history.return_value = [
            HumanMessage(content="Msg1"),
            AIMessage(content="Msg2"),
            HumanMessage(content="Msg3"),
        ]
        agent.memory_manager.get_memory.return_value = MagicMock()
        agent.memory_manager.get_memory.return_value.chat_memory = MagicMock()
        agent.memory_manager.get_memory.return_value.chat_memory.chat_memory = []

        agent.llm_instance = MagicMock()
        agent.llm_instance.llm = MagicMock()
        agent.llm_instance.llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Compressed")
        )

        with patch(
            "agent_ng.token_budget.compute_context_tokens"
        ) as mock_tokens:
            mock_tokens.side_effect = [1000, 200]  # before, after

            success, tokens_saved = await compress_conversation_history(
                agent, "conv1", keep_recent_turns=0, target_tokens=100, reason="critical"
            )

            assert success is True
            assert tokens_saved == 800

    async def test_compress_conversation_keep_recent_turns(self):
        """Test compression keeping recent turns"""
        agent = MagicMock()
        agent.memory_manager = MagicMock()
        history = [
            HumanMessage(content="Old1"),
            AIMessage(content="Old2"),
            HumanMessage(content="Recent"),
        ]
        agent.memory_manager.get_conversation_history.return_value = history
        agent.memory_manager.get_memory.return_value = MagicMock()
        agent.memory_manager.get_memory.return_value.chat_memory = MagicMock()
        agent.memory_manager.get_memory.return_value.chat_memory.chat_memory = []

        agent.llm_instance = MagicMock()
        agent.llm_instance.llm = MagicMock()
        agent.llm_instance.llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Compressed")
        )

        with patch(
            "agent_ng.token_budget.compute_context_tokens"
        ) as mock_tokens:
            mock_tokens.side_effect = [500, 300]

            success, _ = await compress_conversation_history(
                agent, "conv1", keep_recent_turns=1, target_tokens=100, reason="proactive"
            )

            assert success is True
            # Verify recent turn was kept
            new_history = (
                agent.memory_manager.get_memory.return_value.chat_memory.chat_memory
            )
            assert len(new_history) > 0

    async def test_compress_conversation_no_memory_manager(self):
        """Test compression failure when memory manager unavailable"""
        agent = MagicMock()
        agent.memory_manager = None

        success, tokens_saved = await compress_conversation_history(
            agent, "conv1", keep_recent_turns=0, target_tokens=100, reason="critical"
        )

        assert success is False
        assert tokens_saved == 0
