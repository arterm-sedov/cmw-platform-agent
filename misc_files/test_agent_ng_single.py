#!/usr/bin/env python3
"""
Test that agent_ng.py only initializes a single provider
"""

import os
import sys
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ng.langchain_agent import CmwAgent as NextGenAgent

async def test_single_provider():
    """Test that only one provider is initialized"""
    print("🧪 Testing NextGen Agent Single Provider")
    print("=" * 50)
    
    # Set provider
    os.environ["AGENT_PROVIDER"] = "mistral"
    
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
        print(f"   Initialized: {status['is_initialized']}")
        return True
    else:
        print("❌ Agent initialization failed or timed out")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_single_provider())
    if result:
        print("\n🎉 Single provider test passed!")
    else:
        print("\n💥 Single provider test failed!")
        sys.exit(1)
