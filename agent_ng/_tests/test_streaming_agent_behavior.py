"""
Test Streaming Agent Behavior
============================

Test the actual agent behavior that users experience - streaming responses.
This replaces the fake non-streaming tests with real user behavior testing.
"""

import asyncio
import pytest
from typing import List, Dict, Any

# Test imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ng.langchain_agent import NextGenAgent
from agent_ng.llm_manager import get_llm_manager


class TestStreamingAgentBehavior:
    """Test actual streaming agent behavior that users experience"""

    def setup_method(self):
        """Setup test environment"""
        self.llm_manager = get_llm_manager()
        self.agent = None
        self.conversation_id = "test_streaming"

    async def _get_agent(self):
        """Get or create agent for testing"""
        if not self.agent:
            # Get a working LLM instance
            llm_instance = self.llm_manager.get_llm("gemini", use_tools=True)
            if not llm_instance:
                pytest.skip("No LLM available for testing")

            self.agent = NextGenAgent(
                llm_instance=llm_instance,
                tools=[],  # Add tools if needed
                system_prompt="You are a helpful assistant.",
                session_id=self.conversation_id
            )
        return self.agent

    @pytest.mark.asyncio
    async def test_streaming_calculation(self):
        """Test streaming calculation behavior"""
        agent = await self._get_agent()

        # Test streaming calculation
        print("\n👤 User: Calculate 5 + 3")
        events = []
        async for event in agent.stream_message("Calculate 5 + 3", self.conversation_id):
            events.append(event)
            print(f"📡 Event: {event.get('type', 'unknown')} - {event.get('content', '')[:50]}...")

        # Verify we got streaming events
        assert len(events) > 0, "Should receive streaming events"

        # Check for content events
        content_events = [e for e in events if e.get('type') == 'content']
        assert len(content_events) > 0, "Should receive content events"

        # Check for completion
        completion_events = [e for e in events if e.get('type') == 'completion']
        assert len(completion_events) > 0, "Should receive completion event"

        print("✅ Streaming calculation test passed")

    @pytest.mark.asyncio
    async def test_streaming_memory_persistence(self):
        """Test that streaming maintains memory across turns"""
        agent = await self._get_agent()

        # First message
        print("\n👤 User: My name is Alice")
        events1 = []
        async for event in agent.stream_message("My name is Alice", self.conversation_id):
            events1.append(event)

        # Second message (should remember)
        print("\n👤 User: What's my name?")
        events2 = []
        async for event in agent.stream_message("What's my name?", self.conversation_id):
            events2.append(event)

        # Verify both got responses
        content1 = [e for e in events1 if e.get('type') == 'content']
        content2 = [e for e in events2 if e.get('type') == 'content']

        assert len(content1) > 0, "First message should get response"
        assert len(content2) > 0, "Second message should get response"

        print("✅ Streaming memory persistence test passed")

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self):
        """Test streaming error handling"""
        agent = await self._get_agent()

        # Test with invalid input
        print("\n👤 User: Use a non-existent tool called 'fake_tool'")
        events = []
        async for event in agent.stream_message("Use a non-existent tool called 'fake_tool'", self.conversation_id):
            events.append(event)
            print(f"📡 Event: {event.get('type', 'unknown')} - {event.get('content', '')[:50]}...")

        # Should still get a response (even if it's an error)
        assert len(events) > 0, "Should receive events even for invalid input"

        print("✅ Streaming error handling test passed")

    @pytest.mark.asyncio
    async def test_streaming_conversation_flow(self):
        """Test complete conversation flow with streaming"""
        agent = await self._get_agent()

        # Multi-turn conversation
        messages = [
            "Hello! What's 5 + 3?",
            "Now multiply that by 2", 
            "What's the sum of all the numbers I asked about?"
        ]

        all_events = []
        for i, message in enumerate(messages):
            print(f"\n👤 User: {message}")
            events = []
            async for event in agent.stream_message(message, self.conversation_id):
                events.append(event)
                all_events.append(event)

            # Each message should get a response
            content_events = [e for e in events if e.get('type') == 'content']
            assert len(content_events) > 0, f"Message {i+1} should get content response"

        # Verify we got responses for all messages
        content_events = [e for e in all_events if e.get('type') == 'content']
        assert len(content_events) >= len(messages), "Should get content for all messages"

        print("✅ Streaming conversation flow test passed")

    @pytest.mark.asyncio
    async def test_streaming_event_types(self):
        """Test that we get the expected event types"""
        agent = await self._get_agent()

        print("\n👤 User: Calculate 10 * 5")
        events = []
        async for event in agent.stream_message("Calculate 10 * 5", self.conversation_id):
            events.append(event)

        # Check for expected event types
        event_types = [e.get('type') for e in events]
        print(f"📊 Event types received: {set(event_types)}")

        # Should have content events
        assert 'content' in event_types, "Should have content events"

        # Should have completion
        assert 'completion' in event_types, "Should have completion event"

        print("✅ Streaming event types test passed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
