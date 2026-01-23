#!/usr/bin/env python3
"""
Test script to verify that agent_ng fixes are working correctly.
This script tests:
1. System prompt loading
2. Tool loading and binding
3. Basic agent functionality
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ["AGENT_PROVIDER"] = "mistral"  # Use mistral as default for testing

def test_system_prompt_loading():
    """Test that system prompt is loaded correctly"""
    print("🧪 Testing system prompt loading...")

    try:
        from agent_ng.core_agent import CoreAgent

        agent = CoreAgent()

        # Check if system prompt is loaded
        if hasattr(agent, 'system_prompt') and agent.system_prompt:
            print(f"✅ System prompt loaded: {len(agent.system_prompt)} characters")
            print(f"   First 100 chars: {agent.system_prompt[:100]}...")
            return True
        else:
            print("❌ System prompt not loaded")
            return False

    except Exception as e:
        print(f"❌ Error testing system prompt: {e}")
        return False

def test_tool_loading():
    """Test that tools are loaded correctly"""
    print("\n🧪 Testing tool loading...")

    try:
        from agent_ng.core_agent import CoreAgent

        agent = CoreAgent()

        # Check if tools are loaded
        if hasattr(agent, 'tools') and agent.tools:
            print(f"✅ Tools loaded: {len(agent.tools)} tools")

            # Show first few tool names
            tool_names = []
            for tool in agent.tools[:5]:  # Show first 5 tools
                if hasattr(tool, 'name'):
                    tool_names.append(tool.name)
                elif hasattr(tool, '__name__'):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(type(tool).__name__))

            print(f"   Sample tools: {', '.join(tool_names)}")
            return True
        else:
            print("❌ No tools loaded")
            return False

    except Exception as e:
        print(f"❌ Error testing tool loading: {e}")
        return False

def test_llm_manager_tools():
    """Test that LLM manager can load tools"""
    print("\n🧪 Testing LLM manager tool loading...")

    try:
        from agent_ng.llm_manager import get_llm_manager

        llm_manager = get_llm_manager()
        tools = llm_manager.get_tools()

        if tools:
            print(f"✅ LLM manager loaded {len(tools)} tools")

            # Show first few tool names
            tool_names = []
            for tool in tools[:5]:  # Show first 5 tools
                if hasattr(tool, 'name'):
                    tool_names.append(tool.name)
                elif hasattr(tool, '__name__'):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(type(tool).__name__))

            print(f"   Sample tools: {', '.join(tool_names)}")
            return True
        else:
            print("❌ LLM manager loaded no tools")
            return False

    except Exception as e:
        print(f"❌ Error testing LLM manager tools: {e}")
        return False

def test_llm_initialization():
    """Test that LLM can be initialized with tools"""
    print("\n🧪 Testing LLM initialization with tools...")

    try:
        from agent_ng.llm_manager import get_llm_manager

        llm_manager = get_llm_manager()
        llm_instance = llm_manager.get_agent_llm()

        if llm_instance:
            print(f"✅ LLM initialized: {llm_instance.provider.value} ({llm_instance.model_name})")
            print(f"   Tools bound: {llm_instance.bound_tools}")
            print(f"   Is healthy: {llm_instance.is_healthy}")
            return True
        else:
            print("❌ Failed to initialize LLM")
            return False

    except Exception as e:
        print(f"❌ Error testing LLM initialization: {e}")
        return False

def test_basic_agent_functionality():
    """Test basic agent functionality"""
    print("\n🧪 Testing basic agent functionality...")

    try:
        from agent_ng.core_agent import CoreAgent

        agent = CoreAgent()

        # Test a simple question
        response = agent.process_question("Hello, can you tell me what tools you have available?")

        if response and hasattr(response, 'answer'):
            print(f"✅ Agent responded: {response.answer[:100]}...")
            print(f"   Tool calls made: {len(response.tool_calls)}")
            print(f"   LLM used: {response.llm_used}")
            return True
        else:
            print("❌ Agent did not respond properly")
            return False

    except Exception as e:
        print(f"❌ Error testing agent functionality: {e}")
        return False

async def test_langchain_agent():
    """Test LangChain agent functionality"""
    print("\n🧪 Testing LangChain agent...")

    try:
        from agent_ng.langchain_agent import get_agent_ng

        agent = await get_agent_ng()

        if agent and agent.is_ready():
            print(f"✅ LangChain agent ready")
            print(f"   Tools count: {len(agent.tools)}")
            print(f"   LLM info: {agent.get_llm_info()}")
            return True
        else:
            print("❌ LangChain agent not ready")
            return False

    except Exception as e:
        print(f"❌ Error testing LangChain agent: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing agent_ng fixes...")
    print("=" * 50)

    tests = [
        test_system_prompt_loading,
        test_tool_loading,
        test_llm_manager_tools,
        test_llm_initialization,
        test_basic_agent_functionality,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    # Test async function
    try:
        result = asyncio.run(test_langchain_agent())
        results.append(result)
    except Exception as e:
        print(f"❌ Test test_langchain_agent failed with exception: {e}")
        results.append(False)

    print("\n" + "=" * 50)
    print("📊 Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
