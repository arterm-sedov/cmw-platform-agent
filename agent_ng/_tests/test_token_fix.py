#!/usr/bin/env python3
"""
Test script to verify token tracking fix
========================================

This script tests that tokens are only counted once per conversation turn,
not multiple times during tool calling loops.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ng.token_counter import ConversationTokenTracker, TokenCount
from agent_ng.langchain_memory import LangChainConversationChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# Mock LLM for testing
class MockLLM:
    def __init__(self):
        self.call_count = 0
    
    def invoke(self, messages, **kwargs):
        self.call_count += 1
        print(f"🔍 DEBUG: MockLLM invoke called #{self.call_count}")
        
        # Simulate a response with tool calls
        if self.call_count == 1:
            # First call - return tool call
            response = AIMessage(
                content="",
                tool_calls=[{
                    'name': 'test_tool',
                    'args': {'test': 'value'},
                    'id': 'call_1'
                }],
                usage_metadata={
                    'input_tokens': 100,
                    'output_tokens': 50,
                    'total_tokens': 150
                }
            )
        else:
            # Second call - return final response
            response = AIMessage(
                content="Final response after tool calls",
                usage_metadata={
                    'input_tokens': 200,
                    'output_tokens': 100,
                    'total_tokens': 300
                }
            )
        
        return response
    
    def bind_tools(self, tools):
        """Mock bind_tools method"""
        return self

# Mock LLM instance
class MockLLMInstance:
    def __init__(self):
        self.llm = MockLLM()

# Test tool
@tool
def test_tool(test: str) -> str:
    """Test tool for token tracking"""
    return f"Tool result: {test}"

def test_token_tracking_fix():
    """Test that tokens are only counted once per conversation turn"""
    print("🧪 Testing token tracking fix...")
    
    # Create token tracker
    token_tracker = ConversationTokenTracker()
    
    # Create mock LLM instance
    llm_instance = MockLLMInstance()
    
    # Create conversation chain
    chain = LangChainConversationChain(
        llm_instance=llm_instance,
        tools=[test_tool],
        system_prompt="You are a test agent.",
        agent=None  # No agent for this test
    )
    
    # Set up the chain with a mock agent that has our token tracker
    class MockAgent:
        def __init__(self, token_tracker):
            self.token_tracker = token_tracker
    
    mock_agent = MockAgent(token_tracker)
    chain.agent = mock_agent
    
    # Get initial token count
    initial_tokens = token_tracker.conversation_tokens
    print(f"📊 Initial tokens: {initial_tokens}")
    
    # Process a message that will trigger tool calls
    messages = [HumanMessage(content="Test message that requires tools")]
    
    # Run the tool calling loop
    result = chain._run_tool_calling_loop(messages, "test_conversation")
    
    # Check final token count
    final_tokens = token_tracker.conversation_tokens
    print(f"📊 Final tokens: {final_tokens}")
    print(f"📊 Tokens added: {final_tokens - initial_tokens}")
    print(f"📊 LLM calls made: {llm_instance.llm.call_count}")
    
    # Verify that tokens were only counted once (for the first LLM call)
    # Expected: 150 tokens (from first call only)
    expected_tokens = 150
    actual_tokens = final_tokens - initial_tokens
    
    print(f"\n✅ Expected tokens: {expected_tokens}")
    print(f"✅ Actual tokens: {actual_tokens}")
    
    if actual_tokens == expected_tokens:
        print("🎉 SUCCESS: Tokens were counted correctly (only once)!")
        return True
    else:
        print(f"❌ FAILURE: Expected {expected_tokens} tokens, got {actual_tokens}")
        return False

if __name__ == "__main__":
    success = test_token_tracking_fix()
    sys.exit(0 if success else 1)
