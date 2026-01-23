"""
Test Token Counter Module
========================

Tests for the lean, modular token counting system.
"""

import pytest
from agent_ng.token_counter import (
    TokenCount, TiktokenCounter, ApiTokenCounter, 
    ConversationTokenTracker, get_token_tracker, convert_chat_history_to_messages
)
from langchain_core.messages import HumanMessage, SystemMessage


class TestTokenCount:
    """Test TokenCount dataclass"""

    def test_token_count_creation(self):
        """Test basic token count creation"""
        count = TokenCount(100, 50, 150, False, "tiktoken")
        assert count.input_tokens == 100
        assert count.output_tokens == 50
        assert count.total_tokens == 150
        assert not count.is_estimated
        assert count.source == "tiktoken"

    def test_formatted_display(self):
        """Test formatted display strings"""
        # Actual tokens
        actual = TokenCount(100, 50, 150, False, "api")
        assert "150 total (100 input + 50 output)" in actual.formatted

        # Estimated tokens
        estimated = TokenCount(100, 50, 150, True, "tiktoken")
        assert "~150 total (estimated via tiktoken)" in estimated.formatted


class TestTiktokenCounter:
    """Test TiktokenCounter implementation"""

    def test_counter_initialization(self):
        """Test counter initializes properly"""
        counter = TiktokenCounter()
        assert counter.get_source_name() == "tiktoken"

    def test_count_string_tokens(self):
        """Test counting tokens in string"""
        counter = TiktokenCounter()
        result = counter.count_tokens("Hello world")
        assert isinstance(result, TokenCount)
        assert result.total_tokens > 0
        assert result.source == "tiktoken"

    def test_count_message_tokens(self):
        """Test counting tokens in LangChain messages"""
        counter = TiktokenCounter()
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello, how are you?")
        ]
        result = counter.count_tokens(messages)
        assert isinstance(result, TokenCount)
        assert result.total_tokens > 0


class TestApiTokenCounter:
    """Test ApiTokenCounter implementation"""

    def test_counter_initialization(self):
        """Test counter initializes with tiktoken fallback"""
        tiktoken_counter = TiktokenCounter()
        api_counter = ApiTokenCounter(tiktoken_counter)
        assert api_counter.get_source_name() == "api_with_tiktoken_fallback"

    def test_api_tokens_setting(self):
        """Test setting API tokens"""
        tiktoken_counter = TiktokenCounter()
        api_counter = ApiTokenCounter(tiktoken_counter)

        api_counter.set_api_tokens(100, 50, 150)
        result = api_counter.count_tokens("test")

        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        assert not result.is_estimated
        assert result.source == "api"

    def test_fallback_to_tiktoken(self):
        """Test fallback to tiktoken when no API tokens"""
        tiktoken_counter = TiktokenCounter()
        api_counter = ApiTokenCounter(tiktoken_counter)

        result = api_counter.count_tokens("Hello world")
        assert isinstance(result, TokenCount)
        assert result.total_tokens > 0
        assert result.is_estimated
        assert "tiktoken" in result.source or "langchain" in result.source


class TestConversationTokenTracker:
    """Test ConversationTokenTracker implementation"""

    def test_tracker_initialization(self):
        """Test tracker initializes properly"""
        tracker = ConversationTokenTracker()
        assert tracker.conversation_tokens == 0
        assert tracker.session_tokens == 0
        assert tracker.message_count == 0

    def test_count_prompt_tokens(self):
        """Test counting prompt tokens"""
        tracker = ConversationTokenTracker()
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello!")
        ]
        result = tracker.count_prompt_tokens(messages)
        assert isinstance(result, TokenCount)
        assert result.total_tokens > 0

    def test_track_llm_response(self):
        """Test tracking LLM response"""
        tracker = ConversationTokenTracker()
        messages = [HumanMessage(content="Test message")]

        # Mock response with usage metadata
        class MockResponse:
            def __init__(self):
                self.usage_metadata = type('UsageMetadata', (), {
                    'input_tokens': 10,
                    'output_tokens': 5,
                    'total_tokens': 15
                })()

        response = MockResponse()
        result = tracker.track_llm_response(response, messages)

        assert isinstance(result, TokenCount)
        assert tracker.conversation_tokens > 0
        assert tracker.session_tokens > 0
        assert tracker.message_count == 1

    def test_cumulative_stats(self):
        """Test cumulative statistics"""
        tracker = ConversationTokenTracker()

        # Add some tokens
        tracker.conversation_tokens = 1000
        tracker.session_tokens = 500
        tracker.message_count = 10

        stats = tracker.get_cumulative_stats()
        assert stats['conversation_tokens'] == 1000
        assert stats['session_tokens'] == 500
        assert stats['message_count'] == 10
        assert stats['avg_tokens_per_message'] == 100.0

    def test_reset_session(self):
        """Test session reset"""
        tracker = ConversationTokenTracker()
        tracker.conversation_tokens = 1000
        tracker.session_tokens = 500
        tracker.message_count = 10

        tracker.reset_session()
        assert tracker.conversation_tokens == 1000  # Not reset
        assert tracker.session_tokens == 0  # Reset
        assert tracker.message_count == 10  # Not reset


class TestGlobalFunctions:
    """Test global functions"""

    def test_get_token_tracker(self):
        """Test getting global token tracker"""
        tracker1 = get_token_tracker()
        tracker2 = get_token_tracker()
        assert tracker1 is tracker2  # Should be same instance

    def test_tracker_singleton(self):
        """Test tracker is singleton"""
        from agent_ng.token_counter import reset_token_tracker, get_token_tracker

        tracker1 = get_token_tracker()
        reset_token_tracker()
        tracker2 = get_token_tracker()

        # Should be different instances after reset
        assert tracker1 is not tracker2


class TestUtilityFunctions:
    """Test utility functions"""

    def test_convert_chat_history_to_messages(self):
        """Test converting chat history to BaseMessage format"""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        messages = convert_chat_history_to_messages(history)
        assert len(messages) == 3
        assert all(isinstance(msg, HumanMessage) for msg in messages)
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there!"
        assert messages[2].content == "How are you?"

    def test_convert_chat_history_with_current_message(self):
        """Test converting chat history with current message"""
        history = [{"role": "user", "content": "Hello"}]
        current_message = "How are you?"

        messages = convert_chat_history_to_messages(history, current_message)
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "How are you?"

    def test_convert_chat_history_with_system_message(self):
        """Test converting chat history with system message"""
        history = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]

        messages = convert_chat_history_to_messages(history)
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[0].content == "You are helpful."
        assert messages[1].content == "Hello"


if __name__ == "__main__":
    pytest.main([__file__])
