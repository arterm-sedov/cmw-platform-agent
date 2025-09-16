"""
Test script for LangChain Wrapper Module
========================================

This script tests the LangChain Wrapper functionality to ensure it works correctly.
"""

import time
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_wrapper import get_langchain_wrapper, reset_langchain_wrapper, ResponseType


def test_langchain_wrapper():
    """Test the LangChain Wrapper functionality"""
    print("Testing LangChain Wrapper...")
    
    # Reset wrapper for clean test
    reset_langchain_wrapper()
    wrapper = get_langchain_wrapper()
    
    # Test 1: Basic invoke with string message
    print("\n1. Testing basic invoke with string message...")
    try:
        response = wrapper.invoke("What is 2 + 2?")
        print(f"  Content: {response.content[:100]}...")
        print(f"  Provider: {response.provider_used}")
        print(f"  Model: {response.model_name}")
        print(f"  Response Type: {response.response_type.value}")
        print(f"  Execution Time: {response.execution_time:.2f}s")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Invoke with message list
    print("\n2. Testing invoke with message list...")
    try:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of France?")
        ]
        response = wrapper.invoke(messages)
        print(f"  Content: {response.content[:100]}...")
        print(f"  Provider: {response.provider_used}")
        print(f"  Tool Calls: {len(response.tool_calls) if response.tool_calls else 0}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 3: Invoke with specific provider
    print("\n3. Testing invoke with specific provider...")
    try:
        available_providers = wrapper.get_available_providers()
        print(f"  Available providers: {available_providers}")
        
        if available_providers:
            provider = available_providers[0]
            response = wrapper.invoke("Hello, how are you?", provider=provider)
            print(f"  Using provider: {provider}")
            print(f"  Content: {response.content[:100]}...")
        else:
            print("  No providers available")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 4: Streaming responses
    print("\n4. Testing streaming responses...")
    try:
        print("  Streaming response:")
        for chunk in wrapper.astream("Tell me a short story about a robot"):
            if chunk["type"] == "content":
                print(f"    Content: {chunk['content'][:50]}...")
            elif chunk["type"] in ["start", "complete", "error", "warning"]:
                print(f"    {chunk['type'].upper()}: {chunk.get('provider', chunk.get('error', 'Unknown'))}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 5: Provider information
    print("\n5. Testing provider information...")
    try:
        available_providers = wrapper.get_available_providers()
        for provider in available_providers[:2]:  # Test first 2 providers
            info = wrapper.get_provider_info(provider)
            print(f"  Provider: {provider}")
            print(f"    Name: {info.get('name', 'Unknown')}")
            print(f"    Supports Tools: {info.get('supports_tools', False)}")
            print(f"    Models: {info.get('models', [])}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 6: Statistics
    print("\n6. Testing statistics...")
    try:
        stats = wrapper.get_stats()
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Successful Requests: {stats['successful_requests']}")
        print(f"  Failed Requests: {stats['failed_requests']}")
        print(f"  Provider Usage: {stats['provider_usage']}")
        print(f"  Average Response Time: {stats['average_response_time']:.2f}s")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 7: Health check
    print("\n7. Testing health check...")
    try:
        health = wrapper.health_check()
        print(f"  Available Providers: {health['available_providers']}")
        print(f"  LLM Manager Health: {health['llm_manager_health']['status']}")
        print(f"  Wrapper Stats: {health['wrapper_stats']['total_requests']} requests")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 8: Error handling
    print("\n8. Testing error handling...")
    try:
        # Try with invalid provider
        response = wrapper.invoke("Test message", provider="invalid_provider")
        print(f"  Invalid provider response: {response.response_type.value}")
        print(f"  Error info: {response.error_info.description if response.error_info else 'None'}")
    except Exception as e:
        print(f"  Error (expected): {e}")
    
    # Test 9: Tool calling (if supported)
    print("\n9. Testing tool calling...")
    try:
        # This would test tool calling if the provider supports it
        response = wrapper.invoke("What is 5 * 7?", use_tools=True)
        print(f"  Tool calls: {len(response.tool_calls) if response.tool_calls else 0}")
        if response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"    Tool: {tool_call['name']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 10: Reset statistics
    print("\n10. Testing statistics reset...")
    try:
        wrapper.reset_stats()
        stats_after_reset = wrapper.get_stats()
        print(f"  Stats after reset: {stats_after_reset['total_requests']} requests")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n✓ LangChain Wrapper test completed!")


if __name__ == "__main__":
    test_langchain_wrapper()
