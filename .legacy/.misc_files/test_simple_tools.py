"""
Test script for simple tool binding
==================================
"""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from llm_manager import get_llm_manager


@tool
def simple_add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


@tool
def simple_multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b


def test_simple_tool_binding():
    """Test binding simple tools"""
    print("Testing simple tool binding...")

    # Create a simple tools list
    simple_tools = [simple_add, simple_multiply]

    # Test binding directly
    try:
        from langchain_wrapper import get_langchain_wrapper
        wrapper = get_langchain_wrapper()

        # Get an LLM instance
        llm_manager = get_llm_manager()
        available_providers = wrapper.get_available_providers()
        print(f"Available providers: {available_providers}")

        if available_providers:
            provider = available_providers[0]
            print(f"Testing with provider: {provider}")

            # Try to bind tools manually
            llm_instance = llm_manager.get_llm(provider, use_tools=False)  # Get without tools first
            if llm_instance:
                print(f"LLM type: {type(llm_instance.llm)}")

                # Try to bind simple tools
                try:
                    bound_llm = llm_instance.llm.bind_tools(simple_tools)
                    print("✓ Simple tools bound successfully!")

                    # Test invoke
                    response = bound_llm.invoke([HumanMessage(content="What is 5 + 3?")])
                    print(f"Response: {response.content}")
                    if hasattr(response, 'tool_calls'):
                        print(f"Tool calls: {response.tool_calls}")
                    else:
                        print("No tool_calls attribute")

                except Exception as e:
                    print(f"✗ Failed to bind simple tools: {e}")
            else:
                print("Could not get LLM instance")
        else:
            print("No providers available")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_simple_tool_binding()
