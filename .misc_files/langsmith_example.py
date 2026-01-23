#!/usr/bin/env python3
"""
LangSmith Integration Example
============================

This example demonstrates how to use LangSmith observability with the CMW Platform Agent.
Run this script to see LangSmith tracing in action.

Prerequisites:
1. Set LANGSMITH_TRACING=true in your environment
2. Set LANGSMITH_API_KEY=your_api_key in your environment
3. Install langsmith: pip install langsmith
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_langsmith_config():
    """Check if LangSmith is properly configured"""
    from agent_ng.langsmith_config import get_langsmith_config, setup_langsmith_environment

    print("🔍 Checking LangSmith configuration...")

    config = get_langsmith_config()
    print(f"   Tracing enabled: {config.tracing_enabled}")
    print(f"   API key set: {config.api_key is not None}")
    print(f"   Workspace ID: {config.workspace_id}")
    print(f"   Project name: {config.project_name}")
    print(f"   Is configured: {config.is_configured()}")

    if setup_langsmith_environment():
        print("✅ LangSmith environment setup successful")
    else:
        print("ℹ️ LangSmith tracing disabled")

    return config.is_configured()

async def test_agent_with_tracing():
    """Test the agent with LangSmith tracing"""
    from agent_ng.app_ng_modular import NextGenApp

    print("\n🤖 Testing agent with LangSmith tracing...")

    # Create app instance
    app = NextGenApp(language="en")

    # Wait for initialization
    import time
    max_wait = 10  # seconds
    start_time = time.time()

    while not app.is_ready() and (time.time() - start_time) < max_wait:
        print("   Waiting for agent initialization...")
        await asyncio.sleep(1)

    if not app.is_ready():
        print("❌ Agent initialization timeout")
        return

    print("✅ Agent ready")

    # Test message
    test_message = "Hello! Can you help me with a simple math problem: What is 2 + 2?"
    test_history = []

    print(f"📝 Sending test message: {test_message}")

    # Stream the response
    response_parts = []
    async for history, message in app.stream_chat_with_agent(test_message, test_history):
        if history:
            last_message = history[-1] if history else None
            if last_message and last_message.get("role") == "assistant":
                content = last_message.get("content", "")
                if content and content not in response_parts:
                    response_parts.append(content)
                    print(f"   Response: {content[:100]}...")

    print("✅ Test completed")
    print(f"   Response parts received: {len(response_parts)}")

    if response_parts:
        print(f"   Full response: {''.join(response_parts)}")

def main():
    """Main function"""
    print("🚀 LangSmith Integration Example")
    print("=" * 40)

    # Check configuration
    if not check_langsmith_config():
        print("\n⚠️ LangSmith not configured. To enable tracing:")
        print("   1. Set LANGSMITH_TRACING=true")
        print("   2. Set LANGSMITH_API_KEY=your_api_key")
        print("   3. Install langsmith: pip install langsmith")
        print("\nContinuing without tracing...")

    # Test the agent
    try:
        asyncio.run(test_agent_with_tracing())
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Example completed!")
    print("\nTo view traces:")
    print("   1. Go to https://smith.langchain.com")
    print("   2. Navigate to your project")
    print("   3. View the traces from this test")

if __name__ == "__main__":
    main()
