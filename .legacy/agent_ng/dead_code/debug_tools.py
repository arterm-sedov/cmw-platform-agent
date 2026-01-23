"""
Debug script to check tool detection and binding
===============================================
"""

from .llm_manager import get_llm_manager
import tools.tools


def debug_tools():
    """Debug tool detection and binding"""
    print("Debugging Tool Detection...")

    # Check what's in the tools module
    print("\n1. Tools module contents:")
    for name, obj in tools.__dict__.items():
        if callable(obj) and not name.startswith("_") and not isinstance(obj, type):
            print(f"  {name}: {type(obj)}")
            if hasattr(obj, 'name'):
                print(f"    - name: {obj.name}")
            if hasattr(obj, 'description'):
                print(f"    - description: {obj.description}")
            if hasattr(obj, 'args_schema'):
                print(f"    - args_schema: {obj.args_schema}")

    # Check LLM manager tool detection
    print("\n2. LLM Manager tool detection:")
    llm_manager = get_llm_manager()
    detected_tools = llm_manager.get_tools()
    print(f"  Detected {len(detected_tools)} tools")

    for i, tool in enumerate(detected_tools):
        print(f"  Tool {i+1}: {type(tool)}")
        if hasattr(tool, 'name'):
            print(f"    - name: {tool.name}")
        if hasattr(tool, 'description'):
            print(f"    - description: {tool.description}")
        if hasattr(tool, 'args_schema'):
            print(f"    - args_schema: {tool.args_schema}")

    # Test tool binding directly
    print("\n3. Testing tool binding directly:")
    try:
        from langchain_core.messages import HumanMessage
        from langchain_wrapper import get_langchain_wrapper

        wrapper = get_langchain_wrapper()
        available_providers = wrapper.get_available_providers()
        print(f"  Available providers: {available_providers}")

        if available_providers:
            # Try to get an LLM instance and check if tools are bound
            provider = available_providers[0]
            llm_instance = llm_manager.get_llm(provider, use_tools=True)
            if llm_instance:
                print(f"  LLM instance for {provider}: {type(llm_instance.llm)}")
                print(f"  LLM has bound_tools: {hasattr(llm_instance.llm, 'bound_tools')}")
                if hasattr(llm_instance.llm, 'bound_tools'):
                    print(f"  Bound tools: {llm_instance.llm.bound_tools}")

                # Try a simple invoke to see if tools are used
                print(f"\n4. Testing simple invoke with {provider}:")
                try:
                    response = llm_instance.llm.invoke([HumanMessage(content="What is 5 * 7?")])
                    print(f"  Response type: {type(response)}")
                    print(f"  Response content: {response.content}")
                    if hasattr(response, 'tool_calls'):
                        print(f"  Tool calls: {response.tool_calls}")
                    else:
                        print("  No tool_calls attribute")
                except Exception as e:
                    print(f"  Error during invoke: {e}")
            else:
                print(f"  Could not get LLM instance for {provider}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    debug_tools()
