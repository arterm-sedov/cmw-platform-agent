#!/usr/bin/env python3
"""
Test script for the simplified single-provider LLM system.
This tests that the new lean implementation works correctly.
"""

import os
import sys
sys.path.append('..')

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_manager import LLMManager
from langchain_wrapper import LangChainWrapper
from core_agent import CoreAgent

def test_single_provider():
    """Test the single provider system"""
    print("🧪 Testing Single Provider System")
    print("=" * 50)
    
    # Set test provider
    os.environ["AGENT_PROVIDER"] = "mistral"
    
    # Test LLM Manager
    print("\n1. Testing LLMManager...")
    llm_manager = LLMManager()
    
    # Test get_agent_llm method
    llm_instance = llm_manager.get_agent_llm()
    if llm_instance:
        print(f"✅ LLM Manager: Got {llm_instance.provider} ({llm_instance.model_name})")
    else:
        print("❌ LLM Manager: No LLM instance available")
        return False
    
    # Test LangChain Wrapper
    print("\n2. Testing LangChainWrapper...")
    wrapper = LangChainWrapper()
    
    # Test simple invoke
    try:
        response = wrapper.invoke("Hello, how are you?")
        if response.content and response.content.strip():
            print(f"✅ LangChain Wrapper: Got response from {response.provider_used}")
            print(f"   Response: {response.content[:100]}...")
        else:
            print(f"⚠️ LangChain Wrapper: Empty content (might be tool calls)")
            print(f"   Response type: {response.response_type}")
            print(f"   Tool calls: {len(response.tool_calls) if response.tool_calls else 0}")
            # This is actually OK if it's making tool calls
            if response.tool_calls:
                print("✅ LangChain Wrapper: Making tool calls (expected behavior)")
            else:
                print("❌ LangChain Wrapper: No content and no tool calls")
                return False
    except Exception as e:
        print(f"❌ LangChain Wrapper: Error - {e}")
        return False
    
    # Test Core Agent
    print("\n3. Testing CoreAgent...")
    agent = CoreAgent()
    
    try:
        # Test simple question
        response = agent.process_question("What is 2+2?", "test_conversation")
        if response.answer:
            print(f"✅ Core Agent: Got answer using {response.llm_used}")
        else:
            print("❌ Core Agent: No answer")
            return False
    except Exception as e:
        print(f"❌ Core Agent: Error - {e}")
        return False
    
    print("\n🎉 All tests passed! Single provider system is working correctly.")
    return True

def test_different_providers():
    """Test different providers work individually"""
    print("\n🔄 Testing Different Providers")
    print("=" * 50)
    
    providers = ["mistral", "openrouter", "gemini", "groq"]
    llm_manager = LLMManager()
    
    for provider in providers:
        print(f"\nTesting {provider}...")
        os.environ["AGENT_PROVIDER"] = provider
        
        llm_instance = llm_manager.get_agent_llm()
        if llm_instance:
            print(f"✅ {provider}: Available ({llm_instance.model_name})")
        else:
            print(f"⚠️ {provider}: Not available (expected if no API key)")

if __name__ == "__main__":
    print("🚀 Starting Single Provider System Tests")
    
    # Test main functionality
    success = test_single_provider()
    
    if success:
        # Test different providers
        test_different_providers()
        print("\n✨ All tests completed successfully!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
