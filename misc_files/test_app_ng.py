"""
Test script for the next-generation app
======================================

This script tests the basic functionality of the new agent and app.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from agent_ng import NextGenAgent, AgentConfig
from app_ng import NextGenApp


async def test_agent():
    """Test the next-generation agent"""
    print("🧪 Testing NextGen Agent...")
    
    # Create agent with custom config
    config = AgentConfig(
        enable_vector_similarity=False,  # Disable for testing
        max_conversation_history=10,
        enable_tool_calling=True
    )
    
    agent = NextGenAgent(config)
    
    # Wait for initialization
    print("⏳ Waiting for agent initialization...")
    max_wait = 30
    wait_time = 0
    while not agent.is_ready() and wait_time < max_wait:
        await asyncio.sleep(0.5)
        wait_time += 0.5
        print(f"   Waiting... ({wait_time:.1f}s)")
    
    if not agent.is_ready():
        print("❌ Agent initialization failed")
        return False
    
    print("✅ Agent initialized successfully")
    
    # Test status
    status = agent.get_status()
    print(f"📊 Status: {status}")
    
    # Test simple chat
    print("\n💬 Testing chat functionality...")
    try:
        async for event in agent.stream_chat("Hello, how are you?"):
            print(f"   Event: {event['type']} - {event['content'][:50]}...")
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False
    
    print("✅ Chat test passed")
    
    # Test conversation history
    history = agent.get_conversation_history()
    print(f"📚 Conversation history: {len(history)} messages")
    
    # Test stats
    stats = agent.get_stats()
    print(f"📈 Stats keys: {list(stats.keys())}")
    
    print("✅ All agent tests passed")
    return True


async def test_app():
    """Test the next-generation app"""
    print("\n🧪 Testing NextGen App...")
    
    try:
        app = NextGenApp()
        
        # Wait for initialization
        print("⏳ Waiting for app initialization...")
        max_wait = 30
        wait_time = 0
        while not app.initialization_complete and wait_time < max_wait:
            await asyncio.sleep(0.5)
            wait_time += 0.5
            print(f"   Waiting... ({wait_time:.1f}s)")
        
        if not app.initialization_complete:
            print("❌ App initialization failed")
            return False
        
        print("✅ App initialized successfully")
        
        # Test status
        status = app.get_agent_status()
        print(f"📊 App status: {status}")
        
        # Test logs
        logs = app.get_initialization_logs()
        print(f"📜 Logs length: {len(logs)} characters")
        
        print("✅ All app tests passed")
        return True
        
    except Exception as e:
        print(f"❌ App test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Starting Next-Gen App Tests\n")
    
    # Test agent
    agent_success = await test_agent()
    
    # Test app
    app_success = await test_app()
    
    # Summary
    print(f"\n📋 Test Summary:")
    print(f"   Agent: {'✅ PASS' if agent_success else '❌ FAIL'}")
    print(f"   App: {'✅ PASS' if app_success else '❌ FAIL'}")
    
    if agent_success and app_success:
        print("\n🎉 All tests passed! The next-gen app is ready to use.")
        print("\nTo run the app, execute: python app_ng.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
