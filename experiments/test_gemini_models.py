"""
Test all Gemini models (direct API and OpenRouter)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_ng.llm_manager import LLMManager, LLMProvider

def test_gemini_models():
    """Test all Gemini models"""

    print("="*80)
    print("Testing Gemini Models")
    print("="*80)

    # Initialize LLM Manager
    manager = LLMManager()

    # Test Gemini Direct API models
    print("\n1. Testing Gemini Direct API Models")
    print("-"*80)

    gemini_models = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview"
    ]

    for model in gemini_models:
        print(f"\nTesting: {model}")
        try:
            llm = manager.get_llm(LLMProvider.GEMINI, model_name=model)
            if llm:
                print(f"  ✅ Model loaded successfully")
                print(f"  Type: {type(llm).__name__}")
            else:
                print(f"  ❌ Failed to load model")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Test OpenRouter Gemini models
    print("\n\n2. Testing OpenRouter Gemini Models")
    print("-"*80)

    openrouter_models = [
        "google/gemini-2.5-flash",
        "google/gemini-3.1-flash-lite-preview",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3-flash-preview"
    ]

    for model in openrouter_models:
        print(f"\nTesting: {model}")
        try:
            llm = manager.get_llm(LLMProvider.OPENROUTER, model_name=model)
            if llm:
                print(f"  ✅ Model loaded successfully")
                print(f"  Type: {type(llm).__name__}")
            else:
                print(f"  ❌ Failed to load model")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "="*80)
    print("Testing Complete")
    print("="*80)

if __name__ == "__main__":
    test_gemini_models()
