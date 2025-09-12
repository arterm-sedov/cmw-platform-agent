"""
Test LangChain Multi-Turn Conversations with Tool Calls
======================================================

This script demonstrates the new LangChain-based agent's ability to handle
multi-turn conversations with tool calls using pure LangChain patterns.

Features tested:
- Multi-turn conversations
- Tool calling in conversations
- Memory persistence
- Streaming responses
- Error handling

Usage:
    python test_langchain_multi_turn.py
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
    from agent_ng.langchain_agent import get_langchain_agent, LangChainAgent
    from agent_ng.langchain_memory import get_memory_manager
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class MultiTurnTester:
    """Test multi-turn conversations with tool calls"""
    
    def __init__(self):
        self.agent = None
        self.conversation_id = "test_conversation"
    
    async def initialize(self):
        """Initialize the agent"""
        print("🚀 Initializing LangChain Agent...")
        self.agent = await get_langchain_agent()
        
        if not self.agent.llm_instance:
            print("❌ Failed to initialize agent")
            return False
        
        print(f"✅ Agent initialized with {self.agent.llm_instance.provider.value}")
        return True
    
    async def test_basic_conversation(self):
        """Test basic multi-turn conversation"""
        print("\n" + "="*60)
        print("🧪 Testing Basic Multi-Turn Conversation")
        print("="*60)
        
        # Clear any existing conversation
        self.agent.clear_conversation(self.conversation_id)
        
        # First message
        print("\n👤 User: Hello! What's 5 + 3?")
        response1 = self.agent.process_message("Hello! What's 5 + 3?", self.conversation_id)
        print(f"🤖 Assistant: {response1.answer}")
        print(f"   Tool calls: {len(response1.tool_calls)}")
        
        # Second message (should remember context)
        print("\n👤 User: What about 10 + 7?")
        response2 = self.agent.process_message("What about 10 + 7?", self.conversation_id)
        print(f"🤖 Assistant: {response2.answer}")
        print(f"   Tool calls: {len(response2.tool_calls)}")
        
        # Third message (should remember both previous calculations)
        print("\n👤 User: What's the sum of all the numbers I asked about?")
        response3 = self.agent.process_message("What's the sum of all the numbers I asked about?", self.conversation_id)
        print(f"🤖 Assistant: {response3.answer}")
        print(f"   Tool calls: {len(response3.tool_calls)}")
        
        # Show conversation history
        print("\n📚 Conversation History:")
        history = self.agent.get_conversation_history(self.conversation_id)
        for i, msg in enumerate(history):
            role = "👤 User" if isinstance(msg, HumanMessage) else "🤖 Assistant"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"   {i+1}. {role}: {content}")
    
    async def test_tool_calling_conversation(self):
        """Test conversation with tool calls"""
        print("\n" + "="*60)
        print("🔧 Testing Tool Calling in Multi-Turn Conversation")
        print("="*60)
        
        # Clear conversation
        self.agent.clear_conversation(self.conversation_id)
        
        # First message with tool call
        print("\n👤 User: Calculate 15 * 8")
        response1 = self.agent.process_message("Calculate 15 * 8", self.conversation_id)
        print(f"🤖 Assistant: {response1.answer}")
        if response1.tool_calls:
            print(f"   Tool calls: {[call['name'] for call in response1.tool_calls]}")
        
        # Second message referencing previous calculation
        print("\n👤 User: Now divide that result by 4")
        response2 = self.agent.process_message("Now divide that result by 4", self.conversation_id)
        print(f"🤖 Assistant: {response2.answer}")
        if response2.tool_calls:
            print(f"   Tool calls: {[call['name'] for call in response2.tool_calls]}")
        
        # Third message asking for verification
        print("\n👤 User: Can you verify this calculation step by step?")
        response3 = self.agent.process_message("Can you verify this calculation step by step?", self.conversation_id)
        print(f"🤖 Assistant: {response3.answer}")
        if response3.tool_calls:
            print(f"   Tool calls: {[call['name'] for call in response3.tool_calls]}")
    
    async def test_streaming_conversation(self):
        """Test streaming conversation"""
        print("\n" + "="*60)
        print("🌊 Testing Streaming Multi-Turn Conversation")
        print("="*60)
        
        # Clear conversation
        self.agent.clear_conversation(self.conversation_id)
        
        print("\n👤 User: Calculate 25 * 12 and explain the steps")
        print("🤖 Assistant: ", end="", flush=True)
        
        async for event in self.agent.stream_message("Calculate 25 * 12 and explain the steps", self.conversation_id):
            if event["type"] == "content":
                print(event["content"], end="", flush=True)
            elif event["type"] == "tool_start":
                print(f"\n🔧 {event['content']}")
            elif event["type"] == "tool_end":
                print(f"✅ {event['content']}")
            elif event["type"] == "answer":
                print(f"\n\nFinal answer: {event['content']}")
        
        print("\n\n👤 User: What's 300 + 50?")
        print("🤖 Assistant: ", end="", flush=True)
        
        async for event in self.agent.stream_message("What's 300 + 50?", self.conversation_id):
            if event["type"] == "content":
                print(event["content"], end="", flush=True)
            elif event["type"] == "tool_start":
                print(f"\n🔧 {event['content']}")
            elif event["type"] == "tool_end":
                print(f"✅ {event['content']}")
            elif event["type"] == "answer":
                print(f"\n\nFinal answer: {event['content']}")
    
    async def test_memory_persistence(self):
        """Test memory persistence across conversations"""
        print("\n" + "="*60)
        print("🧠 Testing Memory Persistence")
        print("="*60)
        
        # First conversation
        conv1_id = "conversation_1"
        self.agent.clear_conversation(conv1_id)
        
        print(f"\n📝 Conversation 1 (ID: {conv1_id})")
        print("👤 User: My name is Alice and I like math")
        response1 = self.agent.process_message("My name is Alice and I like math", conv1_id)
        print(f"🤖 Assistant: {response1.answer}")
        
        print("\n👤 User: What's my name?")
        response2 = self.agent.process_message("What's my name?", conv1_id)
        print(f"🤖 Assistant: {response2.answer}")
        
        # Second conversation (should be separate)
        conv2_id = "conversation_2"
        self.agent.clear_conversation(conv2_id)
        
        print(f"\n📝 Conversation 2 (ID: {conv2_id})")
        print("👤 User: What's my name?")
        response3 = self.agent.process_message("What's my name?", conv2_id)
        print(f"🤖 Assistant: {response3.answer}")
        
        # Back to first conversation (should remember Alice)
        print(f"\n📝 Back to Conversation 1 (ID: {conv1_id})")
        print("👤 User: What do I like?")
        response4 = self.agent.process_message("What do I like?", conv1_id)
        print(f"🤖 Assistant: {response4.answer}")
    
    async def test_error_handling(self):
        """Test error handling in conversations"""
        print("\n" + "="*60)
        print("⚠️ Testing Error Handling")
        print("="*60)
        
        # Clear conversation
        self.agent.clear_conversation(self.conversation_id)
        
        # Test with invalid tool call
        print("\n👤 User: Use a non-existent tool called 'fake_tool'")
        response1 = self.agent.process_message("Use a non-existent tool called 'fake_tool'", self.conversation_id)
        print(f"🤖 Assistant: {response1.answer}")
        print(f"   Success: {response1.success}")
        print(f"   Error: {response1.error}")
        
        # Test recovery
        print("\n👤 User: Now calculate 5 + 5")
        response2 = self.agent.process_message("Now calculate 5 + 5", self.conversation_id)
        print(f"🤖 Assistant: {response2.answer}")
        print(f"   Success: {response2.success}")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🧪 LangChain Multi-Turn Conversation Tests")
        print("=" * 60)
        
        # Initialize agent
        if not await self.initialize():
            return
        
        # Run tests
        await self.test_basic_conversation()
        await self.test_tool_calling_conversation()
        await self.test_streaming_conversation()
        await self.test_memory_persistence()
        await self.test_error_handling()
        
        # Show final stats
        print("\n" + "="*60)
        print("📊 Final Agent Statistics")
        print("="*60)
        stats = self.agent.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✅ All tests completed!")


async def main():
    """Main test function"""
    tester = MultiTurnTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("AGENT_PROVIDER", "mistral")
    
    # Run tests
    asyncio.run(main())
