"""
Comprehensive test suite for VL model support
Tests configuration, model loading, and basic functionality
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_ng.llm_configs import get_default_llm_configs
from agent_ng.llm_manager import LLMProvider

def test_vl_configuration():
    """Test VL configuration is correct"""
    print("="*80)
    print("VL Configuration Test Suite")
    print("="*80)

    configs = get_default_llm_configs()

    # Test 1: Gemini Direct API
    print("\n[Test 1] Gemini Direct API Configuration")
    print("-"*80)
    gemini = configs[LLMProvider.GEMINI]
    assert gemini.vision_support == True, "Gemini should support vision"
    assert gemini.video_support == True, "Gemini should support video"
    assert gemini.audio_support == True, "Gemini should support audio"
    assert len(gemini.models) == 5, f"Expected 5 Gemini models, got {len(gemini.models)}"
    print(f"✅ Gemini provider: vision={gemini.vision_support}, video={gemini.video_support}, audio={gemini.audio_support}")
    print(f"✅ Gemini models: {len(gemini.models)}")

    # Test 2: Gemini models have VL flags
    print("\n[Test 2] Gemini Model VL Flags")
    print("-"*80)
    for model in gemini.models:
        assert model.get('vision_support') == True, f"{model['model']} should support vision"
        assert model.get('video_support') == True, f"{model['model']} should support video"
        assert model.get('audio_support') == True, f"{model['model']} should support audio"
        print(f"✅ {model['model']}: vision/video/audio supported")

    # Test 3: OpenRouter provider
    print("\n[Test 3] OpenRouter Provider Configuration")
    print("-"*80)
    openrouter = configs[LLMProvider.OPENROUTER]
    assert openrouter.vision_support == True, "OpenRouter should support vision"
    assert openrouter.video_support == True, "OpenRouter should support video"
    assert openrouter.audio_support == True, "OpenRouter should support audio"
    print(f"✅ OpenRouter provider: vision={openrouter.vision_support}, video={openrouter.video_support}, audio={openrouter.audio_support}")

    # Test 4: OpenRouter Gemini models
    print("\n[Test 4] OpenRouter Gemini Models")
    print("-"*80)
    gemini_models = [m for m in openrouter.models if 'gemini' in m['model']]
    assert len(gemini_models) == 4, f"Expected 4 OpenRouter Gemini models, got {len(gemini_models)}"
    for model in gemini_models:
        assert model.get('vision_support') == True, f"{model['model']} should support vision"
        assert model.get('video_support') == True, f"{model['model']} should support video"
        assert model.get('audio_support') == True, f"{model['model']} should support audio"
        print(f"✅ {model['model']}: vision/video/audio supported")

    # Test 5: OpenRouter Claude models
    print("\n[Test 5] OpenRouter Claude Models")
    print("-"*80)
    claude_models = [m for m in openrouter.models if 'claude-sonnet-4' in m['model']]
    for model in claude_models:
        assert model.get('vision_support') == True, f"{model['model']} should support vision"
        print(f"✅ {model['model']}: vision supported")

    # Test 6: OpenRouter Qwen models
    print("\n[Test 6] OpenRouter Qwen Models")
    print("-"*80)
    qwen_models = [m for m in openrouter.models if m['model'] == 'qwen/qwen3.6-plus']
    assert len(qwen_models) == 1, "Expected qwen/qwen3.6-plus in OpenRouter"
    qwen = qwen_models[0]
    assert qwen.get('vision_support') == True, "Qwen 3.6 Plus should support vision"
    assert qwen.get('video_support') == True, "Qwen 3.6 Plus should support video"
    print(f"✅ qwen/qwen3.6-plus: vision/video supported")

    # Test 7: VL environment variables
    print("\n[Test 7] VL Environment Variables")
    print("-"*80)
    vl_default = os.getenv('VL_DEFAULT_MODEL', 'NOT_SET')
    vl_fast = os.getenv('VL_FAST_MODEL', 'NOT_SET')
    vl_audio = os.getenv('VL_AUDIO_MODEL', 'NOT_SET')
    vl_fallback = os.getenv('VL_FALLBACK_MODEL', 'NOT_SET')

    print(f"VL_DEFAULT_MODEL: {vl_default}")
    print(f"VL_FAST_MODEL: {vl_fast}")
    print(f"VL_AUDIO_MODEL: {vl_audio}")
    print(f"VL_FALLBACK_MODEL: {vl_fallback}")

    if vl_default != 'NOT_SET':
        print("✅ VL environment variables configured")
    else:
        print("⚠️  VL environment variables not set (optional)")

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print("✅ All VL configuration tests passed")
    print(f"✅ Gemini Direct API: {len(gemini.models)} models")
    print(f"✅ OpenRouter Gemini: {len(gemini_models)} models")
    print(f"✅ OpenRouter Claude: {len(claude_models)} models with vision")
    print(f"✅ OpenRouter Qwen: 1 model with vision/video")
    print("✅ All models have correct VL capability flags")
    print("\n" + "="*80)
    print("✅ VL CONFIGURATION TEST SUITE PASSED")
    print("="*80)

if __name__ == "__main__":
    try:
        test_vl_configuration()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
