#!/usr/bin/env python3
"""
Debug test to check what's happening with the LLM response
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_manager import LLMManager
from langchain_wrapper import LangChainWrapper

def debug_test():
    """Debug the LLM response issue"""
    print("🔍 Debug Test")
    print("=" * 30)

    # Set test provider
    os.environ["AGENT_PROVIDER"] = "mistral"

    # Test LLM Manager directly
    print("\n1. Testing LLMManager directly...")
    llm_manager = LLMManager()
    llm_instance = llm_manager.get_agent_llm()

    if llm_instance:
        print(f"✅ Got LLM instance: {llm_instance.provider} ({llm_instance.model_name})")

        # Test direct LLM call
        print("\n2. Testing direct LLM call...")
        try:
            response = llm_instance.llm.invoke("Hello, how are you?")
            print(f"Response type: {type(response)}")
            print(f"Response content: {response}")
            print(f"Has content attr: {hasattr(response, 'content')}")
            if hasattr(response, 'content'):
                print(f"Content: '{response.content}'")
        except Exception as e:
            print(f"❌ Direct LLM call failed: {e}")
    else:
        print("❌ No LLM instance available")
        return

    # Test LangChain Wrapper
    print("\n3. Testing LangChainWrapper...")
    wrapper = LangChainWrapper()

    try:
        response = wrapper.invoke("Hello, how are you?")
        print(f"Wrapper response type: {type(response)}")
        print(f"Wrapper response content: '{response.content}'")
        print(f"Response type: {response.response_type}")
        print(f"Provider used: {response.provider_used}")
    except Exception as e:
        print(f"❌ Wrapper call failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_test()
