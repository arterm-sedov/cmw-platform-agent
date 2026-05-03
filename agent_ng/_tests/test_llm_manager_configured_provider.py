"""Tests for _get_configured_provider_and_model_index provider override behavior."""

from unittest.mock import patch

from agent_ng.llm_manager import LLMManager, LLMProvider


def test_provider_override_still_resolves_default_model_index() -> None:
    """Session API-key bucket must not force model_index=0 (first catalog slot)."""
    mgr = LLMManager()
    grok_idx = mgr._find_model_index(LLMProvider.POLZA, "x-ai/grok-4.20")
    assert grok_idx is not None and grok_idx > 0, "fixture expects grok after first polza slot"

    fake_settings = {
        "default_provider": "polza",
        "default_model": "x-ai/grok-4.20",
    }

    with patch("agent_ng.agent_config.get_llm_settings", return_value=fake_settings):
        pe, idx = mgr._get_configured_provider_and_model_index("polza")

    assert pe == LLMProvider.POLZA
    assert idx == grok_idx
