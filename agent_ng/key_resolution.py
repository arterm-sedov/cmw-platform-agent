"""
Unified API Key Resolution
=========================

Centralized key resolution reused by LLM, VL, and image generation.
Checks config tab override first, then falls back to environment variable.
"""

import os

from agent_ng.session_manager import get_session_config


def get_api_key(
    config_key: str,
    override_key: str | None = None,
    session_id: str | None = None,
) -> str | None:
    """
    Get API key with unified resolution.

    Resolution order:
    1. Direct override_key parameter (passed from config tab)
    2. Session config llm_api_key_override (if session_id provided)
    3. Environment variable (config_key)

    Args:
        config_key: Environment variable name (e.g., "GEMINI_KEY")
        override_key: Direct override from caller (highest priority)
        session_id: Session ID to read session config from

    Returns:
        API key string or None if not found
    """
    # 1. Direct override (highest priority)
    if override_key:
        return override_key

    # 2. Session config override
    if session_id:
        session_config = get_session_config(session_id)
        if session_config:
            llm_api_key_override = session_config.get("llm_api_key_override", "")
            if llm_api_key_override:
                return llm_api_key_override

    # 3. Environment variable (lowest priority)
    return os.getenv(config_key) or None


def get_provider_api_key(
    provider: str | None = None,
    override_key: str | None = None,
    session_id: str | None = None,
) -> str | None:
    """
    Get API key for a specific provider using its config env var.

    Resolution order for override_key:
    1. Direct override_key parameter (highest priority)
    2. Session config llm_api_key_override (if session_id provided)

    Resolution order for provider:
    1. Explicit provider parameter
    2. Session config llm_provider_override (if session_id provided)

    Args:
        provider: Provider name (e.g., "gemini", "groq", "openrouter")
        override_key: Direct override from caller (highest priority)
        session_id: Session ID to read session config from

    Returns:
        API key string or None if not found
    """
    # Import configs directly (avoid circular dependency via LLMManager instantiation)
    try:
        from .llm_configs import get_default_llm_configs
        from .llm_manager import LLMProvider
    except ImportError:
        from agent_ng.llm_configs import get_default_llm_configs
        from agent_ng.llm_manager import LLMProvider

    # Resolve provider: explicit param → session config → None
    resolved_provider = provider
    if not resolved_provider and session_id:
        session_config = get_session_config(session_id)
        if session_config:
            provider_override = session_config.get("llm_provider_override", "")
            if provider_override:
                resolved_provider = provider_override

    if not resolved_provider:
        return None

    try:
        provider_enum = LLMProvider(resolved_provider.lower())
    except ValueError:
        return None

    # Get configs directly without instantiating LLMManager
    configs = get_default_llm_configs()
    config = configs.get(provider_enum)
    if not config:
        return None

    config_key = config.api_key_env

    # Resolve API key: override → session config → env var
    final_key: str | None = None

    # 1. Direct override (highest priority)
    if override_key:
        final_key = override_key
    # 2. Session config llm_api_key_override
    elif session_id:
        session_config = get_session_config(session_id)
        if session_config:
            final_key = session_config.get("llm_api_key_override", "") or None
    # 3. Environment variable (lowest priority)
    if not final_key:
        final_key = os.getenv(config_key) or None

    return final_key
