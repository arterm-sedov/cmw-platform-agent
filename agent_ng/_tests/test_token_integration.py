"""
Test Token Integration
=====================

Test that token counting is properly integrated with the agent.
"""

import pytest
from agent_ng.langchain_agent import CmwAgent
from agent_ng.token_counter import TokenCount


def test_agent_has_token_tracker():
    """Test that agent has token tracker initialized"""
    agent = CmwAgent()
    assert hasattr(agent, 'token_tracker')
    assert agent.token_tracker is not None


def test_agent_token_methods():
    """Test that agent has token counting methods"""
    agent = CmwAgent()
    
    # Test token display info
    token_info = agent.get_token_display_info()
    assert isinstance(token_info, dict)
    assert 'prompt_tokens' in token_info
    assert 'api_tokens' in token_info
    assert 'cumulative_stats' in token_info
    
    # Test prompt token counting
    history = [{"role": "user", "content": "Hello"}]
    current_message = "How are you?"
    
    prompt_tokens = agent.count_prompt_tokens_for_chat(history, current_message)
    assert prompt_tokens is not None
    assert isinstance(prompt_tokens, TokenCount)
    assert prompt_tokens.total_tokens > 0
    
    # Test API token counting
    api_tokens = agent.get_last_api_tokens()
    # This might be None initially, which is fine
    assert api_tokens is None or isinstance(api_tokens, TokenCount)


def test_token_tracker_integration():
    """Test that token tracker is properly integrated"""
    agent = CmwAgent()
    
    # Test that token tracker has the expected methods
    tracker = agent.token_tracker
    assert hasattr(tracker, 'count_prompt_tokens')
    assert hasattr(tracker, 'track_llm_response')
    assert hasattr(tracker, 'get_cumulative_stats')
    assert hasattr(tracker, 'get_last_prompt_tokens')
    assert hasattr(tracker, 'get_last_api_tokens')


if __name__ == "__main__":
    pytest.main([__file__])
