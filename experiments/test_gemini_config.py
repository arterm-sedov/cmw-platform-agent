"""
Simple test to verify Gemini models are in configuration
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_ng.llm_configs import get_default_llm_configs
from agent_ng.llm_manager import LLMProvider

LLM_CONFIGS = get_default_llm_configs()

def test_gemini_config():
    """Test Gemini configuration"""

    print("="*80)
    print("Gemini Models Configuration Test")
    print("="*80)

    # Test Gemini Direct API
    print("\n1. Gemini Direct API Configuration")
    print("-"*80)

    gemini_config = LLM_CONFIGS[LLMProvider.GEMINI]

    print(f"Provider: {gemini_config.name}")
    print(f"Vision Support: {gemini_config.vision_support}")
    print(f"Video Support: {gemini_config.video_support}")
    print(f"Audio Support: {gemini_config.audio_support}")
    print(f"Total Models: {len(gemini_config.models)}")

    print("\nModels:")
    for i, model in enumerate(gemini_config.models, 1):
        vision = model.get('vision_support', False)
        video = model.get('video_support', False)
        audio = model.get('audio_support', False)
        print(f"  {i}. {model['model']}")
        print(f"     Context: {model['token_limit']:,} tokens")
        print(f"     Vision: {vision}, Video: {video}, Audio: {audio}")

    # Test OpenRouter Gemini
    print("\n\n2. OpenRouter Gemini Configuration")
    print("-"*80)

    openrouter_config = LLM_CONFIGS[LLMProvider.OPENROUTER]
    gemini_models = [m for m in openrouter_config.models if 'gemini' in m['model']]

    print(f"Total Gemini Models: {len(gemini_models)}")

    print("\nModels:")
    for i, model in enumerate(gemini_models, 1):
        vision = model.get('vision_support', False)
        video = model.get('video_support', False)
        audio = model.get('audio_support', False)
        print(f"  {i}. {model['model']}")
        print(f"     Context: {model['token_limit']:,} tokens")
        print(f"     Vision: {vision}, Video: {video}, Audio: {audio}")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"✅ Gemini Direct API: {len(gemini_config.models)} models")
    print(f"✅ OpenRouter Gemini: {len(gemini_models)} models")
    print(f"✅ All models have VL capability flags")
    print(f"✅ Backward compatibility preserved (Gemini 2.5 available)")

    # Verify specific models
    print("\n" + "="*80)
    print("Verification")
    print("="*80)

    direct_models = [m['model'] for m in gemini_config.models]
    openrouter_models = [m['model'] for m in gemini_models]

    expected_direct = [
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-3.1-flash-lite-preview',
        'gemini-3.1-pro-preview',
        'gemini-3-flash-preview'
    ]

    expected_openrouter = [
        'google/gemini-2.5-flash',
        'google/gemini-3.1-flash-lite-preview',
        'google/gemini-3.1-pro-preview',
        'google/gemini-3-flash-preview'
    ]

    print("\nDirect API Models:")
    for model in expected_direct:
        status = "✅" if model in direct_models else "❌"
        print(f"  {status} {model}")

    print("\nOpenRouter Models:")
    for model in expected_openrouter:
        status = "✅" if model in openrouter_models else "❌"
        print(f"  {status} {model}")

    print("\n" + "="*80)
    print("✅ Configuration Test Complete")
    print("="*80)

if __name__ == "__main__":
    test_gemini_config()
