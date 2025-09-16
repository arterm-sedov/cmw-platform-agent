#!/usr/bin/env python3
"""
Test the merged LangChain-native app
===================================

This script tests the merged app_ng.py with LangChain-native features.
"""

import asyncio
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Set environment for OpenRouter
os.environ.setdefault("AGENT_PROVIDER", "openrouter")

try:
    from agent_ng.app_ng import NextGenApp
    print("✅ Successfully imported merged NextGenApp")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


async def test_app_creation():
    """Test app creation and basic functionality"""
    print("\n🧪 Testing App Creation")
    print("=" * 40)
    
    try:
        app = NextGenApp()
        print("✅ App created successfully")
        
        # Wait for initialization
        import time
        max_wait = 15
        wait_time = 0
        while not app.is_ready() and wait_time < max_wait:
            await asyncio.sleep(0.5)
            wait_time += 0.5
            print(f"⏳ Waiting for app... ({wait_time:.1f}s)")
        
        if app.is_ready():
            print("✅ App is ready")
            return app
        else:
            print("❌ App not ready after waiting")
            return None
            
    except Exception as e:
        print(f"❌ Error creating app: {e}")
        return None


async def test_chat_functionality(app: NextGenApp):
    """Test chat functionality"""
    print("\n💬 Testing Chat Functionality")
    print("=" * 40)
    
    try:
        # Test basic chat
        history = []
        message = "Hello, can you calculate 5 + 3?"
        
        print(f"👤 User: {message}")
        updated_history, _ = await app.chat_with_agent(message, history)
        
        if updated_history and len(updated_history) >= 2:
            user_msg = updated_history[-2]
            assistant_msg = updated_history[-1]
            print(f"🤖 Assistant: {assistant_msg.get('content', 'No response')}")
            print("✅ Basic chat works")
            return True
        else:
            print("❌ Chat failed - no response")
            return False
            
    except Exception as e:
        print(f"❌ Error in chat test: {e}")
        return False


async def test_streaming_functionality(app: NextGenApp):
    """Test streaming functionality"""
    print("\n🌊 Testing Streaming Functionality")
    print("=" * 40)
    
    try:
        history = []
        message = "What is 10 * 5?"
        
        print(f"👤 User: {message}")
        print("🤖 Assistant: ", end="", flush=True)
        
        event_count = 0
        async for updated_history, _ in app.stream_chat_with_agent(message, history):
            event_count += 1
            if updated_history and len(updated_history) >= 2:
                assistant_msg = updated_history[-1]
                content = assistant_msg.get('content', '')
                if content and not content.startswith('❌'):
                    print(content, end="", flush=True)
        
        print(f"\n✅ Streaming completed with {event_count} events")
        return event_count > 0
        
    except Exception as e:
        print(f"❌ Error in streaming test: {e}")
        return False


async def test_conversation_management(app: NextGenApp):
    """Test conversation management"""
    print("\n📚 Testing Conversation Management")
    print("=" * 40)
    
    try:
        # Test conversation history
        history = app.get_conversation_history()
        print(f"📖 Conversation history length: {len(history)}")
        
        # Test clear conversation
        cleared_history, _ = app.clear_conversation()
        print(f"🗑️ Cleared conversation: {len(cleared_history)} messages")
        
        # Test new conversation
        history = []
        message = "What is 2 + 2?"
        updated_history, _ = await app.chat_with_agent(message, history)
        
        if len(updated_history) >= 2:
            print("✅ New conversation works")
            return True
        else:
            print("❌ New conversation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in conversation management test: {e}")
        return False


async def test_ui_creation():
    """Test UI creation"""
    print("\n🎨 Testing UI Creation")
    print("=" * 40)
    
    try:
        app = NextGenApp()
        interface = app.create_interface()
        
        if interface:
            print("✅ UI interface created successfully")
            return True
        else:
            print("❌ UI creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in UI test: {e}")
        return False


async def main():
    """Main test function"""
    print("🧪 LangChain-Native Merged App Test")
    print("=" * 60)
    
    # Test app creation
    app = await test_app_creation()
    if not app:
        print("\n❌ App creation failed, cannot continue")
        return False
    
    # Run tests
    tests = [
        ("Chat Functionality", test_chat_functionality(app)),
        ("Streaming Functionality", test_streaming_functionality(app)),
        ("Conversation Management", test_conversation_management(app)),
        ("UI Creation", test_ui_creation())
    ]
    
    results = []
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results.append((test_name, result))
            print(f"\n{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"\n❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Merged app is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
