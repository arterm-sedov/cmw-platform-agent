"""
Test Integration: CmwAgent with app_ng.py
==============================================

This script tests that the enhanced CmwAgent works correctly
with app_ng.py and provides all the functionality of NextGenAgent.

Usage:
    python test_integration.py
"""

import asyncio
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from agent_ng.langchain_agent import CmwAgent, ChatMessage, get_agent_ng
    from agent_ng.app_ng import NextGenApp
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class IntegrationTester:
    """Test the integration between CmwAgent and app_ng.py"""
    
    def __init__(self):
        self.agent = None
        self.app = None
    
    async def test_agent_functionality(self):
        """Test that CmwAgent has all NextGenAgent functionality"""
        print("🧪 Testing CmwAgent Functionality")
        print("=" * 50)
        
        # Initialize agent
        self.agent = await get_agent_ng()
        
        if not self.agent:
            print("❌ Failed to initialize agent")
            return False
        
        print(f"✅ Agent initialized: {type(self.agent).__name__}")
        
        # Test all required methods
        required_methods = [
            'is_ready', 'get_status', 'get_llm_info', 'get_stats',
            'process_message', 'stream_message', 'stream_chat',
            'get_conversation_history', 'clear_conversation'
        ]
        
        for method_name in required_methods:
            if hasattr(self.agent, method_name):
                print(f"✅ Method {method_name} exists")
            else:
                print(f"❌ Method {method_name} missing")
                return False
        
        # Test status methods
        print("\n📊 Testing Status Methods:")
        status = self.agent.get_status()
        print(f"   Status: {status}")
        
        llm_info = self.agent.get_llm_info()
        print(f"   LLM Info: {llm_info}")
        
        stats = self.agent.get_stats()
        print(f"   Stats keys: {list(stats.keys())}")
        
        return True
    
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with tool calls"""
        print("\n🔄 Testing Multi-Turn Conversation")
        print("=" * 50)
        
        if not self.agent:
            print("❌ Agent not initialized")
            return False
        
        # Clear any existing conversation
        self.agent.clear_conversation("test")
        
        # First message
        print("\n👤 User: Calculate 5 + 3")
        response1 = self.agent.process_message("Calculate 5 + 3", "test")
        print(f"🤖 Assistant: {response1.answer}")
        print(f"   Success: {response1.success}")
        print(f"   Tool calls: {len(response1.tool_calls)}")
        
        # Second message (should remember context)
        print("\n👤 User: Now multiply that by 2")
        response2 = self.agent.process_message("Now multiply that by 2", "test")
        print(f"🤖 Assistant: {response2.answer}")
        print(f"   Success: {response2.success}")
        print(f"   Tool calls: {len(response2.tool_calls)}")
        
        # Test conversation history
        history = self.agent.get_conversation_history("test")
        print(f"\n📚 Conversation history length: {len(history)}")
        
        return response1.success and response2.success
    
    async def test_app_integration(self):
        """Test that app_ng.py works with CmwAgent"""
        print("\n🖥️ Testing App Integration")
        print("=" * 50)
        
        try:
            # Initialize app
            self.app = NextGenApp()
            
            # Wait for initialization
            max_wait = 10  # 10 seconds
            wait_time = 0
            while not self.app.is_ready() and wait_time < max_wait:
                await asyncio.sleep(0.5)
                wait_time += 0.5
            
            if not self.app.is_ready():
                print("❌ App not ready after 10 seconds")
                return False
            
            print("✅ App initialized and ready")
            
            # Test app methods
            app_methods = [
                'is_ready', 'get_agent_stats', 'get_debug_logs',
                'chat_with_agent', 'stream_chat_with_agent',
                'clear_conversation', 'get_conversation_history'
            ]
            
            for method_name in app_methods:
                if hasattr(self.app, method_name):
                    print(f"✅ App method {method_name} exists")
                else:
                    print(f"❌ App method {method_name} missing")
                    return False
            
            # Test agent stats
            stats = self.app.get_agent_stats()
            print(f"✅ App agent stats: {list(stats.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ App integration test failed: {e}")
            return False
    
    async def test_streaming(self):
        """Test streaming functionality"""
        print("\n🌊 Testing Streaming")
        print("=" * 50)
        
        if not self.agent:
            print("❌ Agent not initialized")
            return False
        
        print("👤 User: Calculate 10 * 5")
        print("🤖 Assistant: ", end="", flush=True)
        
        event_count = 0
        async for event in self.agent.stream_chat("Calculate 10 * 5"):
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
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🧪 CmwAgent Integration Tests")
        print("=" * 60)
        
        # Set up environment
        os.environ.setdefault("AGENT_PROVIDER", "mistral")
        
        # Run tests
        tests = [
            ("Agent Functionality", self.test_agent_functionality),
            ("Multi-Turn Conversation", self.test_multi_turn_conversation),
            ("App Integration", self.test_app_integration),
            ("Streaming", self.test_streaming)
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
            print("🎉 All tests passed! CmwAgent is ready to replace NextGenAgent.")
        else:
            print("⚠️ Some tests failed. Check the implementation.")
        
        return passed == total


async def main():
    """Main test function"""
    tester = IntegrationTester()
    success = await tester.run_all_tests()
    return success


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
