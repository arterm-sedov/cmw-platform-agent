"""
Simple Streaming Test
====================

Test the simple streaming implementation to ensure it works correctly.
"""

import asyncio
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ng.simple_streaming import get_simple_streaming_manager


async def test_simple_streaming():
    """Test the simple streaming implementation"""
    print("Testing simple streaming implementation...")
    
    # Get streaming manager
    streaming_manager = get_simple_streaming_manager()
    
    # Test 1: Stream text
    print("\n1. Testing text streaming:")
    async for event in streaming_manager.stream_text("Hello, this is a test of the streaming functionality!"):
        print(f"Event: {event.event_type} - {event.content}")
    
    # Test 2: Stream thinking
    print("\n2. Testing thinking streaming:")
    async for event in streaming_manager.stream_thinking("Test message"):
        print(f"Event: {event.event_type} - {event.content}")
    
    # Test 3: Stream response with tools
    print("\n3. Testing response with tools streaming:")
    tool_calls = [
        {"name": "multiply", "args": {"a": 15, "b": 23}},
        {"name": "add", "args": {"a": 345, "b": 7}}
    ]
    
    async for event in streaming_manager.stream_response_with_tools(
        "The result is 352.", 
        tool_calls
    ):
        print(f"Event: {event.event_type} - {event.content}")
    
    # Test 4: Stream error
    print("\n4. Testing error streaming:")
    async for event in streaming_manager.stream_error("Test error message"):
        print(f"Event: {event.event_type} - {event.content}")
    
    print("\n✅ All streaming tests completed!")


if __name__ == "__main__":
    asyncio.run(test_simple_streaming())
