"""
Simple Integration Test
======================

A simple test to verify that CmwAgent works correctly
without the full app dependencies.

Usage:
    python test_simple_integration.py
"""

import asyncio
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Set environment before importing
os.environ.setdefault("AGENT_PROVIDER", "openrouter")

try:
    from agent_ng.langchain_agent import CmwAgent, ChatMessage, get_agent_ng
    print("✅ Successfully imported CmwAgent")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


async def test_basic_functionality():
    """Test basic functionality of CmwAgent"""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 40)
    
    try:
        # Initialize agent
        print("Initializing agent...")
        agent = await get_agent_ng()
        
        if not agent:
            print("❌ Failed to initialize agent")
            return False
        
        print(f"✅ Agent initialized: {type(agent).__name__}")
        
        # Wait for agent to be ready (async initialization)
        import time
        max_wait = 10  # seconds
        wait_time = 0
        while not agent.is_ready() and wait_time < max_wait:
            await asyncio.sleep(0.5)
            wait_time += 0.5
        
        # Test readiness
        if agent.is_ready():
            print("✅ Agent is ready")
        else:
            print("❌ Agent is not ready after waiting")
            return False
        
        # Test status methods
        print("\n📊 Testing Status Methods:")
        status = agent.get_status()
        print(f"   Status: {status}")
        
        llm_info = agent.get_llm_info()
        print(f"   LLM Info: {llm_info}")
        
        stats = agent.get_stats()
        print(f"   Stats keys: {list(stats.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic functionality test: {e}")
        return False


async def test_multi_turn_conversation():
    """Test multi-turn conversation"""
    print("\n🔄 Testing Multi-Turn Conversation")
    print("=" * 40)
    
    try:
        agent = await get_agent_ng()
        
        if not agent:
            print("❌ Agent not initialized")
            return False
        
        # Wait for agent to be ready
        import time
        max_wait = 10  # seconds
        wait_time = 0
        while not agent.is_ready() and wait_time < max_wait:
            await asyncio.sleep(0.5)
            wait_time += 0.5
        
        if not agent.is_ready():
            print("❌ Agent not ready after waiting")
            return False
        
        # Clear any existing conversation
        agent.clear_conversation("test")
        
        # First message
        print("\n👤 User: Calculate 5 + 3")
        response1 = agent.process_message("Calculate 5 + 3", "test")
        print(f"🤖 Assistant: {response1.answer}")
        print(f"   Success: {response1.success}")
        print(f"   Tool calls: {len(response1.tool_calls)}")
        
        if not response1.success:
            print("❌ First message failed")
            return False
        
        # Second message (should remember context)
        print("\n👤 User: Now multiply that by 2")
        response2 = agent.process_message("Now multiply that by 2", "test")
        print(f"🤖 Assistant: {response2.answer}")
        print(f"   Success: {response2.success}")
        print(f"   Tool calls: {len(response2.tool_calls)}")
        
        if not response2.success:
            print("❌ Second message failed")
            return False
        
        # Test conversation history
        history = agent.get_conversation_history("test")
        print(f"\n📚 Conversation history length: {len(history)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in multi-turn conversation test: {e}")
        return False


async def test_streaming():
    """Test streaming functionality"""
    print("\n🌊 Testing Streaming")
    print("=" * 40)
    
    try:
        agent = await get_agent_ng()
        
        if not agent:
            print("❌ Agent not initialized")
            return False
        
        # Wait for agent to be ready
        import time
        max_wait = 10  # seconds
        wait_time = 0
        while not agent.is_ready() and wait_time < max_wait:
            await asyncio.sleep(0.5)
            wait_time += 0.5
        
        if not agent.is_ready():
            print("❌ Agent not ready after waiting")
            return False
        
        print("👤 User: Calculate 10 * 5")
        print("🤖 Assistant: ", end="", flush=True)
        
        event_count = 0
        async for event in agent.stream_chat("Calculate 10 * 5"):
            event_count += 1
            if event["type"] == "content":
                print(event["content"], end="", flush=True)
            elif event["type"] == "tool_start":
                print(f"\n🔧 {event['content']}")
            elif event["type"] == "tool_end":
                print(f"✅ {event['content']}")
            elif event["type"] == "answer":
                print(f"\n\nFinal answer: {event['content']}")
        
        print(f"\n✅ Streaming completed with {event_count} events")
        return event_count > 0
        
    except Exception as e:
        print(f"❌ Error in streaming test: {e}")
        return False


async def main():
    """Main test function"""
    print("🧪 CmwAgent Simple Integration Test")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Multi-Turn Conversation", test_multi_turn_conversation),
        ("Streaming", test_streaming)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
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
        print("🎉 All tests passed! CmwAgent is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
