"""
Debug Streaming Test
===================

Test the streaming implementation to debug any issues.
"""

import asyncio
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ng.langchain_agent import get_agent_ng


async def test_agent_streaming():
    """Test the agent streaming directly"""
    print("Testing agent streaming...")
    
    try:
        # Get agent
        agent = await get_agent_ng()
        
        if not agent:
            print("❌ Agent not available")
            return
        
        print(f"✅ Agent ready: {agent.is_ready()}")
        
        # Wait for agent to be ready
        if not agent.is_ready():
            print("⏳ Waiting for agent to be ready...")
            max_wait = 30  # 30 seconds timeout
            wait_time = 0
            while not agent.is_ready() and wait_time < max_wait:
                await asyncio.sleep(0.5)
                wait_time += 0.5
                print(f"⏳ Waiting... ({wait_time:.1f}s)")
            
            if not agent.is_ready():
                print("❌ Agent initialization timeout")
                return
        
        print(f"✅ Agent is ready: {agent.is_ready()}")
        
        # Test streaming
        print("\nTesting stream_message...")
        message = "Hello! What is 5 + 3?"
        
        async for event in agent.stream_message(message, "test"):
            print(f"Event: {event}")
        
        print("\n✅ Streaming test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_agent_streaming())
