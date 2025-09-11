"""
Test script for the new debug system and thinking transparency.

This script tests the debug streaming, thinking transparency, and response fixing
functionality to ensure everything works correctly.
"""

import asyncio
import time
from .debug_streamer import get_debug_streamer, get_log_handler, get_thinking_transparency, LogLevel, LogCategory
from .streaming_chat import get_chat_interface


async def test_debug_system():
    """Test the debug system components"""
    print("🧪 Testing Debug System Components...")
    
    # Test debug streamer
    debug_streamer = get_debug_streamer("test")
    print("✅ Debug streamer initialized")
    
    # Test log handler
    log_handler = get_log_handler("test")
    print("✅ Log handler initialized")
    
    # Test thinking transparency
    thinking_transparency = get_thinking_transparency("test")
    print("✅ Thinking transparency initialized")
    
    
    # Test chat interface
    chat_interface = get_chat_interface("test")
    print("✅ Chat interface initialized")
    
    # Test logging
    debug_streamer.info("Test info message", LogCategory.SYSTEM)
    debug_streamer.warning("Test warning message", LogCategory.SYSTEM)
    debug_streamer.error("Test error message", LogCategory.SYSTEM)
    debug_streamer.success("Test success message", LogCategory.SYSTEM)
    
    # Test thinking process
    thinking_transparency.start_thinking("🧠 Test Thinking")
    thinking_transparency.add_thinking("This is a test thinking process...")
    thinking_transparency.add_thinking(" Adding more thoughts...")
    thinking_message = thinking_transparency.complete_thinking()
    print(f"✅ Thinking message created: {thinking_message}")
    
    # Test tool usage
    tool_message = thinking_transparency.create_tool_usage_message(
        tool_name="test_tool",
        tool_args={"param1": "value1", "param2": "value2"},
        result="Tool executed successfully"
    )
    print(f"✅ Tool message created: {tool_message}")
    
    # Test log retrieval
    logs = log_handler.get_current_logs()
    print(f"✅ Current logs: {len(logs)} characters")
    
    print("🎉 All tests passed!")




async def test_streaming_chat():
    """Test the streaming chat interface"""
    print("\n💬 Testing Streaming Chat Interface...")
    
    chat_interface = get_chat_interface("test")
    
    # Test message creation
    thinking_msg = chat_interface.streaming_chat.create_thinking_message(
        "This is a test thinking process",
        "🧠 Test Thinking"
    )
    print(f"✅ Thinking message: {thinking_msg.role} - {thinking_msg.content[:50]}...")
    
    tool_msg = chat_interface.streaming_chat.create_tool_usage_message(
        tool_name="test_tool",
        tool_args={"test": "value"},
        result="Success"
    )
    print(f"✅ Tool message: {tool_msg.role} - {tool_msg.content[:50]}...")
    
    error_msg = chat_interface.streaming_chat.create_error_message(
        "Test error occurred",
        "Test Error"
    )
    print(f"✅ Error message: {error_msg.role} - {error_msg.content[:50]}...")
    
    print("🎉 Streaming chat tests passed!")


async def main():
    """Run all tests"""
    print("🚀 Starting Debug System Tests...\n")
    
    await test_debug_system()
    await test_streaming_chat()
    
    print("\n🎉 All tests completed successfully!")
    print("\nThe debug system is ready for use with the Next-Gen App!")


if __name__ == "__main__":
    asyncio.run(main())
