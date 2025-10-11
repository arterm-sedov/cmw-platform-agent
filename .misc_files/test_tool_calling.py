"""
Test script specifically for tool calling functionality
====================================================

This script tests whether the LLM wrapper can properly use tools.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_wrapper import get_langchain_wrapper, reset_langchain_wrapper


def test_tool_calling():
    """Test tool calling functionality specifically"""
    print("Testing Tool Calling Functionality...")
    
    # Reset wrapper for clean test
    reset_langchain_wrapper()
    wrapper = get_langchain_wrapper()
    
    # Test 1: Simple math that should trigger tools
    print("\n1. Testing simple math (should use multiply tool)...")
    try:
        response = wrapper.invoke("What is 5 multiplied by 7?", use_tools=True)
        print(f"  Content: {response.content}")
        print(f"  Provider: {response.provider_used}")
        print(f"  Tool Calls: {len(response.tool_calls) if response.tool_calls else 0}")
        if response.tool_calls:
            for i, tool_call in enumerate(response.tool_calls):
                print(f"    Tool {i+1}: {tool_call['name']} with args {tool_call['args']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Addition that should trigger tools
    print("\n2. Testing addition (should use add tool)...")
    try:
        response = wrapper.invoke("Add 15 and 25 together", use_tools=True)
        print(f"  Content: {response.content}")
        print(f"  Tool Calls: {len(response.tool_calls) if response.tool_calls else 0}")
        if response.tool_calls:
            for i, tool_call in enumerate(response.tool_calls):
                print(f"    Tool {i+1}: {tool_call['name']} with args {tool_call['args']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 3: Complex calculation that should use multiple tools
    print("\n3. Testing complex calculation (should use multiple tools)...")
    try:
        response = wrapper.invoke("Calculate (10 + 5) * 3 - 2", use_tools=True)
        print(f"  Content: {response.content}")
        print(f"  Tool Calls: {len(response.tool_calls) if response.tool_calls else 0}")
        if response.tool_calls:
            for i, tool_call in enumerate(response.tool_calls):
                print(f"    Tool {i+1}: {tool_call['name']} with args {tool_call['args']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 4: Check available tools
    print("\n4. Checking available tools...")
    try:
        from llm_manager import get_llm_manager
        llm_manager = get_llm_manager()
        tools = llm_manager.get_tools()
        print(f"  Available tools: {len(tools)}")
        for i, tool in enumerate(tools[:5]):  # Show first 5 tools
            print(f"    Tool {i+1}: {getattr(tool, 'name', 'Unknown')} - {getattr(tool, 'description', 'No description')}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 5: Test with specific provider that supports tools
    print("\n5. Testing with specific tool-supporting provider...")
    try:
        available_providers = wrapper.get_available_providers()
        tool_providers = [p for p in available_providers if wrapper.get_provider_info(p).get('supports_tools', False)]
        print(f"  Tool-supporting providers: {tool_providers}")
        
        if tool_providers:
            provider = tool_providers[0]
            response = wrapper.invoke("What is 8 * 9?", provider=provider, use_tools=True)
            print(f"  Using provider: {provider}")
            print(f"  Content: {response.content}")
            print(f"  Tool Calls: {len(response.tool_calls) if response.tool_calls else 0}")
            if response.tool_calls:
                for i, tool_call in enumerate(response.tool_calls):
                    print(f"    Tool {i+1}: {tool_call['name']} with args {tool_call['args']}")
        else:
            print("  No tool-supporting providers available")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n✓ Tool calling test completed!")


if __name__ == "__main__":
    test_tool_calling()
