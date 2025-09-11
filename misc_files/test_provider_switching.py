#!/usr/bin/env python3
"""
Test that provider switching works correctly with AGENT_PROVIDER
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_manager import LLMManager

def test_provider_switching():
    """Test that different providers can be selected via environment variable"""
    print("🔄 Testing Provider Switching")
    print("=" * 40)
    
    llm_manager = LLMManager()
    
    # Test different providers
    providers_to_test = ["mistral", "openrouter", "gemini", "groq"]
    
    for provider in providers_to_test:
        print(f"\nTesting {provider}...")
        os.environ["AGENT_PROVIDER"] = provider
        
        llm_instance = llm_manager.get_agent_llm()
        if llm_instance:
            print(f"✅ {provider}: {llm_instance.provider} ({llm_instance.model_name})")
        else:
            print(f"❌ {provider}: Not available")
    
    print("\n🎉 Provider switching test completed!")

if __name__ == "__main__":
    test_provider_switching()
