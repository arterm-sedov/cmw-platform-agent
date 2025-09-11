#!/usr/bin/env python3
"""
Test that agent_ng.py switches providers correctly
"""

import os
import sys
import asyncio

# Set environment variable BEFORE importing
os.environ["AGENT_PROVIDER"] = "gemini"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ng.agent_ng import NextGenAgent

async def test_provider_switching():
    """Test that different providers can be selected"""
    print("🔄 Testing Provider Switching in NextGen Agent")
    print("=" * 50)
    
    print(f"AGENT_PROVIDER set to: {os.environ.get('AGENT_PROVIDER')}")
    
    # Create agent
    agent = NextGenAgent()
    
    # Wait for initialization
    print("⏳ Waiting for agent initialization...")
    max_wait = 10
    wait_time = 0
    while not agent.is_ready() and wait_time < max_wait:
        await asyncio.sleep(0.5)
        wait_time += 0.5
        print(f"   Waiting... ({wait_time:.1f}s)")
    
    if agent.is_ready():
        status = agent.get_status()
        print(f"✅ Agent ready!")
        print(f"   Provider: {status['current_provider']}")
        print(f"   Model: {status['current_llm']}")
        print(f"   Tools: {status['tools_count']}")
        
        # Check if it's using the right provider
        expected_provider = os.environ.get('AGENT_PROVIDER', 'mistral')
        if status['current_provider'] == expected_provider:
            print(f"✅ Correct provider selected: {expected_provider}")
            return True
        else:
            print(f"❌ Wrong provider selected. Expected: {expected_provider}, Got: {status['current_provider']}")
            return False
    else:
        print("❌ Agent initialization failed or timed out")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_provider_switching())
    if result:
        print("\n🎉 Provider switching test passed!")
    else:
        print("\n💥 Provider switching test failed!")
        sys.exit(1)
