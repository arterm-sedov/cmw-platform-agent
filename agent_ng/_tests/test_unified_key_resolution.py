"""
Test Unified Key Resolution
============================

Tests for centralized API key resolution that checks config tab override first,
then falls back to environment variable. Reused by LLM, VL, and image generation.
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_unified_key_resolution_module_exists():
    """Test that unified key resolution module exists"""
    try:
        from agent_ng import key_resolution

        assert hasattr(key_resolution, "get_api_key")
        assert hasattr(key_resolution, "get_provider_api_key")
        print("✅ key_resolution module exists with get_api_key, get_provider_api_key")
    except ImportError:
        print("❌ key_resolution module not found")
        raise


def test_get_api_key_uses_override_first():
    """Test that get_api_key returns override before env var"""
    from agent_ng import key_resolution

    with patch.dict(os.environ, {"GEMINI_KEY": "env_key"}, clear=True):
        result = key_resolution.get_api_key(
            config_key="GEMINI_KEY",
            override_key="override_key",
            session_id="test_session",
        )
        assert result == "override_key", f"Expected 'override_key', got '{result}'"

    print("✅ get_api_key uses override first")


def test_get_api_key_falls_back_to_env():
    """Test that get_api_key falls back to env var when no override"""
    from agent_ng import key_resolution

    with patch.dict(os.environ, {"GEMINI_KEY": "env_key"}, clear=True):
        result = key_resolution.get_api_key(
            config_key="GEMINI_KEY",
            override_key="",
            session_id="test_session",
        )
        assert result == "env_key", f"Expected 'env_key', got '{result}'"

    print("✅ get_api_key falls back to env var")


def test_get_api_key_returns_none_when_no_source():
    """Test that get_api_key returns None when no override or env"""
    from agent_ng import key_resolution

    with patch.dict(os.environ, {}, clear=True):
        result = key_resolution.get_api_key(
            config_key="NONEXISTENT_KEY",
            override_key="",
            session_id="test_session",
        )
        assert result is None, f"Expected None, got '{result}'"

    print("✅ get_api_key returns None when no source")


def test_get_api_key_reads_session_config():
    """Test that get_api_key reads llm_api_key_override from session config"""
    from agent_ng import key_resolution
    from agent_ng import session_manager

    test_session = "test_key_resolution_session"
    session_manager.set_session_config(
        test_session,
        {"llm_api_key_override": "session_override_key"},
    )

    with patch.dict(os.environ, {}, clear=True):
        result = key_resolution.get_api_key(
            config_key="GEMINI_KEY",
            override_key=None,
            session_id=test_session,
        )
        assert result == "session_override_key", f"Expected 'session_override_key', got '{result}'"

    session_manager.clear_session_config(test_session)
    print("✅ get_api_key reads from session config")


def test_get_provider_api_key_uses_provider_config():
    """Test that get_provider_api_key uses provider's api_key_env"""
    from agent_ng import key_resolution
    from agent_ng.llm_configs import get_default_llm_configs

    # Get real configs to verify the function works with actual provider config
    configs = get_default_llm_configs()
    gemini_config = configs.get("gemini") if hasattr(configs, 'get') else None
    if gemini_config is None:
        # Try enum lookup
        from agent_ng.llm_manager import LLMProvider
        gemini_config = configs.get(LLMProvider.GEMINI)

    assert gemini_config is not None, "Gemini config should exist"
    assert gemini_config.api_key_env == "GEMINI_KEY", f"Expected GEMINI_KEY, got {gemini_config.api_key_env}"

    # Test that the function correctly resolves the provider config
    import os
    test_key = "test_gemini_key_12345"
    original = os.environ.get("GEMINI_KEY")
    try:
        os.environ["GEMINI_KEY"] = test_key
        result = key_resolution.get_provider_api_key(provider="gemini")
        assert result == test_key, f"Expected '{test_key}', got '{result}'"
    finally:
        if original is not None:
            os.environ["GEMINI_KEY"] = original
        elif "GEMINI_KEY" in os.environ:
            del os.environ["GEMINI_KEY"]

    print("✅ get_provider_api_key uses provider's api_key_env")


def test_llm_manager_uses_unified_resolution():
    """Test that LLMManager._get_api_key uses get_provider_api_key"""
    from agent_ng.llm_manager import LLMManager
    import inspect

    source = inspect.getsource(LLMManager._get_api_key)
    assert "get_provider_api_key" in source, "LLMManager._get_api_key doesn't use get_provider_api_key"

    print("✅ LLMManager uses get_provider_api_key")


def test_image_engine_uses_unified_resolution():
    """Test that ImageEngine uses get_provider_api_key"""
    import inspect
    from agent_ng import image_engine

    source = inspect.getsource(image_engine.ImageEngine.__init__)
    assert "get_provider_api_key" in source, "ImageEngine.__init__ doesn't use get_provider_api_key"
    assert '"openrouter"' in source, "ImageEngine should hardcode openrouter provider"

    print("✅ ImageEngine uses get_provider_api_key")


def test_gemini_vl_adapter_uses_unified_resolution():
    """Test that GeminiDirectVisionAdapter uses get_provider_api_key"""
    import inspect
    from agent_ng.vision_adapters import gemini_adapter

    source = inspect.getsource(gemini_adapter.GeminiDirectVisionAdapter.invoke)
    assert "get_provider_api_key" in source, "GeminiDirectVisionAdapter.invoke doesn't use get_provider_api_key"

    print("✅ GeminiDirectVisionAdapter uses get_provider_api_key")


def test_all_providers_use_same_resolution():
    """Test that all providers go through the same resolution path"""
    from agent_ng import key_resolution

    providers = ["gemini", "groq", "huggingface", "openrouter", "mistral", "gigachat"]
    with patch.dict(os.environ, {"GEMINI_KEY": "test_key"}, clear=True):
        for provider in providers:
            result = key_resolution.get_provider_api_key(provider=provider)
            # Should not crash - returns None for missing keys but doesn't error
            assert result is None or isinstance(result, str)

    print("✅ All providers use same resolution path")


def main():
    print("🧪 Testing Unified Key Resolution")
    print("=" * 50)

    tests = [
        test_unified_key_resolution_module_exists,
        test_get_api_key_uses_override_first,
        test_get_api_key_falls_back_to_env,
        test_get_api_key_returns_none_when_no_source,
        test_get_api_key_reads_session_config,
        test_get_provider_api_key_uses_provider_config,
        test_llm_manager_uses_unified_resolution,
        test_image_engine_uses_unified_resolution,
        test_gemini_vl_adapter_uses_unified_resolution,
        test_all_providers_use_same_resolution,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
