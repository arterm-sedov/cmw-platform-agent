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
        count = TokenCount(100, 50, 150, False, "tiktoken", 0.0)
        assert count.input_tokens == 100
        assert count.output_tokens == 50
        assert count.total_tokens == 150
        assert not count.is_estimated
        assert count.source == "tiktoken"
        assert count.cost == 0.0

    def test_formatted_display(self):
        """Test formatted display strings"""
        # Actual tokens without cost
        actual = TokenCount(100, 50, 150, False, "api", 0.0)
        assert "150 total (100 input + 50 output)" in actual.formatted

        # Actual tokens with cost
        actual_with_cost = TokenCount(100, 50, 150, False, "api", 0.0015)
        assert "150 total (100 input + 50 output)" in actual_with_cost.formatted
        assert "$0.0015" in actual_with_cost.formatted

        # Estimated tokens
        estimated = TokenCount(100, 50, 150, True, "tiktoken", 0.0)
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

        api_counter.set_api_tokens(100, 50, 150, 0.0015)
        result = api_counter.count_tokens("test")

        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        assert not result.is_estimated
        assert result.source == "api"
        assert result.cost == 0.0015

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
        """Test tracking LLM response with API tokens and cost"""
        tracker = ConversationTokenTracker()
        messages = [HumanMessage(content="Test message")]

        # Mock response with usage metadata including cost
        class MockUsageMetadata:
            def __init__(self):
                self.input_tokens = 10
                self.output_tokens = 5
                self.total_tokens = 15

        class MockResponse:
            def __init__(self):
                self.usage_metadata = MockUsageMetadata()

        response = MockResponse()
        result = tracker.track_llm_response(response, messages)

        assert isinstance(result, TokenCount)
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.total_tokens == 15
        assert result.cost == 0.0  # No cost in usage_metadata object format
        assert tracker.conversation_tokens == 15
        assert tracker.session_tokens == 15
        assert tracker.message_count == 1

    def test_track_llm_response_with_cost(self):
        """Test tracking LLM response with cost from OpenRouter API"""
        tracker = ConversationTokenTracker()
        messages = [HumanMessage(content="Test message")]

        # Mock response with cost in usage dict (OpenRouter format)
        class MockResponse:
            def __init__(self):
                self.response_metadata = {
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                        "cost": 0.0025  # OpenRouter uses "cost" field, not "total_cost"
                    }
                }

        response = MockResponse()
        result = tracker.track_llm_response(response, messages)

        assert isinstance(result, TokenCount)
        assert result.input_tokens == 20
        assert result.output_tokens == 10
        assert result.total_tokens == 30
        assert result.cost == 0.0025
        assert tracker.conversation_tokens == 30
        assert tracker.session_tokens == 30
        assert tracker.conversation_cost == 0.0025
        assert tracker.session_cost == 0.0025

    def test_track_llm_response_with_prompt_cache_details(self):
        """Parse OpenRouter prompt_tokens_details.{cached_tokens,cache_write_tokens}."""
        tracker = ConversationTokenTracker()
        messages = [HumanMessage(content="Test message")]

        class MockResponse:
            def __init__(self):
                self.response_metadata = {
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                        "cost": 0.0,
                        "prompt_tokens_details": {
                            "cached_tokens": 7,
                            "cache_write_tokens": 11,
                        },
                    }
                }

        response = MockResponse()
        result = tracker.track_llm_response(response, messages)

        assert isinstance(result, TokenCount)
        assert result.input_tokens == 20
        assert result.output_tokens == 10
        assert result.total_tokens == 30
        assert result.cached_tokens == 7
        assert result.cache_write_tokens == 11

        stats = tracker.get_cumulative_stats()
        assert stats["last_cached_tokens"] == 7
        assert stats["last_cache_write_tokens"] == 11

    def test_cumulative_stats(self):
        """Test cumulative statistics including cost"""
        tracker = ConversationTokenTracker()

        # Set up token and cost data
        tracker.conversation_tokens = 1000
        tracker.session_tokens = 500
        tracker.message_count = 10
        tracker.conversation_cost = 0.05
        tracker.session_cost = 0.025
        tracker._last_turn_cost = 0.0025
        tracker._last_api_tokens = TokenCount(100, 50, 150, False, "api", 0.0025)

        stats = tracker.get_cumulative_stats()
        assert stats['conversation_tokens'] == 1000
        assert stats['session_tokens'] == 500
        assert stats['message_count'] == 10
        # Average is per-conversation (session_tokens / message_count)
        assert stats['avg_tokens_per_message'] == 50
        assert stats['turn_cost'] == 0.0025
        assert stats['conversation_cost'] == 0.025
        assert stats['total_cost'] == 0.05
        assert stats['last_input_tokens'] == 100
        assert stats['last_output_tokens'] == 50

    def test_reset_session(self):
        """Test session reset preserves conversation totals but resets session"""
        tracker = ConversationTokenTracker()
        tracker.conversation_tokens = 1000
        tracker.session_tokens = 500
        tracker.message_count = 10
        tracker.conversation_cost = 0.05
        tracker.session_cost = 0.025

        tracker.reset_session()
        assert tracker.conversation_tokens == 1000  # Not reset
        assert tracker.session_tokens == 0  # Reset
        assert tracker.session_cost == 0.0  # Reset
        assert tracker.conversation_cost == 0.05  # Not reset
        assert tracker.message_count == 10  # Not reset

    def test_turn_based_cost_tracking(self):
        """Test turn-based cost accumulation and finalization"""
        tracker = ConversationTokenTracker()

        # Begin a turn
        tracker.begin_turn()
        assert tracker._turn_active
        assert tracker._turn_cost == 0.0

        # Update turn usage with cost
        class MockResponse:
            def __init__(self):
                self.response_metadata = {
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 25,
                        "total_tokens": 75,
                        "cost": 0.0015  # OpenRouter uses "cost" field, not "total_cost"
                    }
                }

        response = MockResponse()
        success = tracker.update_turn_usage_from_api(response)
        assert success
        assert tracker._turn_input_tokens == 50
        assert tracker._turn_output_tokens == 25
        assert tracker._turn_total_tokens == 75
        assert tracker._turn_cost == 0.0015

        # Finalize turn
        messages = [HumanMessage(content="Test")]
        result = tracker.finalize_turn_usage(response, messages)
        assert result is not None
        assert result.cost == 0.0015
        assert tracker.conversation_cost == 0.0015
        assert tracker.session_cost == 0.0015
        assert tracker._last_turn_cost == 0.0015
        assert not tracker._turn_active

    def test_update_turn_usage_parses_prompt_cache_details(self):
        """Ensure per-turn accumulation captures prompt cache details when present."""
        tracker = ConversationTokenTracker()
        tracker.begin_turn()

        class MockResponse:
            def __init__(self):
                self.response_metadata = {
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 25,
                        "total_tokens": 75,
                        "cost": 0.0,
                        "prompt_tokens_details": {
                            "cached_tokens": 3,
                            "cache_write_tokens": 5,
                        },
                    }
                }

        response = MockResponse()
        assert tracker.update_turn_usage_from_api(response) is True
        last = tracker.get_last_api_tokens()
        assert last is not None
        assert last.cached_tokens == 3
        assert last.cache_write_tokens == 5


class TestGlobalFunctions:
    """Test global functions"""

    def test_get_token_tracker_default_session(self):
        """Test getting token tracker for default session"""
        tracker1 = get_token_tracker()
        tracker2 = get_token_tracker()
        assert tracker1 is tracker2  # Same session should return same instance

    def test_tracker_session_isolation(self):
        """Test tracker provides session-specific instances"""
        tracker1 = get_token_tracker("session1")
        tracker2 = get_token_tracker("session2")
        tracker1_again = get_token_tracker("session1")

        # Different sessions should get different instances
        assert tracker1 is not tracker2
        # Same session should get same instance
        assert tracker1 is tracker1_again


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
