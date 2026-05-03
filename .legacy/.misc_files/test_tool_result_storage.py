#!/usr/bin/env python3
"""
Test Tool Result Storage in Chat History
========================================

This script tests that tool call results are properly stored and retrieved
from chat history, ensuring the LLM has access to complete conversation context.

Test Cases:
1. Tool calls are stored as separate ToolMessage objects in chat history
2. Tool results are properly formatted with tool_call_id and tool_name
3. MessageProcessor correctly handles ToolMessage objects from history
4. Multi-turn conversations preserve tool call context
"""

import sys
import os
import json
from typing import Dict, List, Any

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from agent_ng.core_agent import CoreAgent
    from agent_ng.langchain_memory import ConversationMemoryManager
    from agent_ng.message_processor import MessageProcessor
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def test_core_agent_tool_storage():
    """Test that CoreAgent properly stores tool call results in chat history"""
    print("Testing CoreAgent tool result storage...")

    agent = CoreAgent()
    conversation_id = "test_conversation"

    # Clear any existing conversation
    agent.clear_conversation(conversation_id)

    # Simulate a conversation with tool calls
    tool_calls = [
        {
            'name': 'test_tool',
            'args': {'param1': 'value1'},
            'result': 'Tool execution result: Success',
            'id': 'call_123'
        },
        {
            'name': 'another_tool',
            'args': {'param2': 'value2'},
            'result': 'Another tool result: Data processed',
            'id': 'call_456'
        }
    ]

    # Add a user message
    agent._add_to_conversation(conversation_id, "user", "Hello, can you help me?")

    # Add an AI response with tool calls
    agent._add_to_conversation(conversation_id, "assistant", "I'll help you by running some tools.", tool_calls=tool_calls)

    # Get conversation history
    history = agent.get_conversation_history(conversation_id)

    print(f"Conversation history length: {len(history)}")

    # Verify we have user, assistant, and tool messages
    roles = [msg['role'] for msg in history]
    print(f"Message roles: {roles}")

    # Check for tool messages
    tool_messages = [msg for msg in history if msg['role'] == 'tool']
    print(f"Tool messages found: {len(tool_messages)}")

    for i, tool_msg in enumerate(tool_messages):
        print(f"Tool message {i+1}:")
        print(f"  Content: {tool_msg['content']}")
        print(f"  Tool call ID: {tool_msg.get('tool_call_id', 'N/A')}")
        print(f"  Tool name: {tool_msg.get('tool_name', 'N/A')}")
        print(f"  Tool args: {tool_msg.get('tool_args', {})}")

    # Verify tool messages match the original tool calls
    assert len(tool_messages) == len(tool_calls), f"Expected {len(tool_calls)} tool messages, got {len(tool_messages)}"

    for i, (tool_msg, original_call) in enumerate(zip(tool_messages, tool_calls)):
        assert tool_msg['content'] == original_call['result'], f"Tool message {i} content mismatch"
        assert tool_msg['tool_call_id'] == original_call['id'], f"Tool message {i} ID mismatch"
        assert tool_msg['tool_name'] == original_call['name'], f"Tool message {i} name mismatch"

    print("✓ CoreAgent tool storage test passed!")
    return True


def test_langchain_memory_tool_storage():
    """Test that LangChain memory properly stores tool call results"""
    print("\nTesting LangChain memory tool result storage...")

    memory_manager = ConversationMemoryManager()
    conversation_id = "test_langchain_conversation"

    # Clear any existing memory
    memory_manager.clear_memory(conversation_id)

    # Add a user message
    memory_manager.add_message(conversation_id, HumanMessage(content="Hello"))

    # Add an AI message
    memory_manager.add_message(conversation_id, AIMessage(content="I'll help you"))

    # Add tool messages
    tool_messages = [
        ToolMessage(content="Tool result 1", tool_call_id="call_1", name="tool1"),
        ToolMessage(content="Tool result 2", tool_call_id="call_2", name="tool2")
    ]

    for tool_msg in tool_messages:
        memory_manager.add_message(conversation_id, tool_msg)

    # Get conversation history
    history = memory_manager.get_conversation_history(conversation_id)

    print(f"LangChain conversation history length: {len(history)}")

    # Check message types
    message_types = [type(msg).__name__ for msg in history]
    print(f"Message types: {message_types}")

    # Verify tool messages are present
    tool_messages_in_history = [msg for msg in history if isinstance(msg, ToolMessage)]
    print(f"Tool messages in history: {len(tool_messages_in_history)}")

    for i, tool_msg in enumerate(tool_messages_in_history):
        print(f"Tool message {i+1}:")
        print(f"  Content: {tool_msg.content}")
        print(f"  Tool call ID: {tool_msg.tool_call_id}")
        print(f"  Name: {tool_msg.name}")

    assert len(tool_messages_in_history) == len(tool_messages), f"Expected {len(tool_messages)} tool messages, got {len(tool_messages_in_history)}"

    print("✓ LangChain memory tool storage test passed!")
    return True


def test_message_processor_tool_handling():
    """Test that MessageProcessor correctly handles tool messages from history"""
    print("\nTesting MessageProcessor tool message handling...")

    processor = MessageProcessor()

    # Create a mock chat history with tool messages
    chat_history = [
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant", 
            "content": "I'll help you",
            "tool_calls": [
                {
                    "name": "test_tool",
                    "args": {"param": "value"},
                    "id": "call_123"
                }
            ]
        },
        {
            "role": "tool",
            "content": "Tool execution result",
            "tool_call_id": "call_123",
            "tool_name": "test_tool"
        }
    ]

    # Format messages using the processor
    messages = processor.format_messages_simple(
        question="What did the tool return?",
        chat_history=chat_history
    )

    print(f"Formatted messages length: {len(messages)}")

    # Check message types
    message_types = [type(msg).__name__ for msg in messages]
    print(f"Message types: {message_types}")

    # Verify tool message is present
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    print(f"Tool messages in formatted messages: {len(tool_messages)}")

    if tool_messages:
        tool_msg = tool_messages[0]
        print(f"Tool message content: {tool_msg.content}")
        print(f"Tool call ID: {tool_msg.tool_call_id}")
        print(f"Tool name: {tool_msg.name}")

    assert len(tool_messages) == 1, f"Expected 1 tool message, got {len(tool_messages)}"
    assert tool_messages[0].content == "Tool execution result"
    assert tool_messages[0].tool_call_id == "call_123"
    assert tool_messages[0].name == "test_tool"

    print("✓ MessageProcessor tool handling test passed!")
    return True


def test_multi_turn_conversation():
    """Test that tool call results persist across multiple conversation turns"""
    print("\nTesting multi-turn conversation with tool results...")

    agent = CoreAgent()
    conversation_id = "multi_turn_test"

    # Clear any existing conversation
    agent.clear_conversation(conversation_id)

    # First turn with tool calls
    agent._add_to_conversation(conversation_id, "user", "What's the weather like?")

    tool_calls_turn1 = [
        {
            'name': 'get_weather',
            'args': {'location': 'New York'},
            'result': 'Weather in New York: 72°F, sunny',
            'id': 'call_weather_1'
        }
    ]

    agent._add_to_conversation(conversation_id, "assistant", "Let me check the weather for you.", tool_calls=tool_calls_turn1)

    # Second turn - user asks follow-up
    agent._add_to_conversation(conversation_id, "user", "What about tomorrow?")

    tool_calls_turn2 = [
        {
            'name': 'get_weather_forecast',
            'args': {'location': 'New York', 'days': 1},
            'result': 'Tomorrow in New York: 75°F, partly cloudy',
            'id': 'call_weather_2'
        }
    ]

    agent._add_to_conversation(conversation_id, "assistant", "Here's tomorrow's forecast.", tool_calls=tool_calls_turn2)

    # Get full conversation history
    history = agent.get_conversation_history(conversation_id)

    print(f"Multi-turn conversation length: {len(history)}")

    # Count different message types
    user_messages = [msg for msg in history if msg['role'] == 'user']
    assistant_messages = [msg for msg in history if msg['role'] == 'assistant']
    tool_messages = [msg for msg in history if msg['role'] == 'tool']

    print(f"User messages: {len(user_messages)}")
    print(f"Assistant messages: {len(assistant_messages)}")
    print(f"Tool messages: {len(tool_messages)}")

    # Verify we have tool results from both turns
    assert len(tool_messages) == 2, f"Expected 2 tool messages, got {len(tool_messages)}"

    # Check that both tool results are preserved
    tool_contents = [msg['content'] for msg in tool_messages]
    assert 'Weather in New York: 72°F, sunny' in tool_contents
    assert 'Tomorrow in New York: 75°F, partly cloudy' in tool_contents

    print("✓ Multi-turn conversation test passed!")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING TOOL RESULT STORAGE IN CHAT HISTORY")
    print("=" * 60)

    tests = [
        test_core_agent_tool_storage,
        test_langchain_memory_tool_storage,
        test_message_processor_tool_handling,
        test_multi_turn_conversation
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("🎉 All tests passed! Tool call results are properly stored in chat history.")
    else:
        print("❌ Some tests failed. Check the implementation.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
