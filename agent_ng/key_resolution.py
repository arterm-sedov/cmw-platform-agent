"""
Unified API Key Resolution
=========================

Centralized key resolution reused by LLM, VL, and image generation.
Session keys live in ``llm_provider_api_keys`` (provider id -> key).
"""

from __future__ import annotations

import os
from typing import Any

from .session_manager import get_session_config


def _session_keys_map(session_config: dict[str, Any] | None) -> dict[str, str]:
    if not session_config:
        return {}
    raw = session_config.get("llm_provider_api_keys")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for pk, vk in raw.items():
        if not isinstance(pk, str):
            continue
        k = pk.strip()
        if not k:
            continue
        out[k] = (vk or "").strip() if isinstance(vk, str) else ""
    return out


def _default_llm_provider_str() -> str:
    try:
        try:
            from .agent_config import get_llm_settings
        except ImportError:
            from agent_ng.agent_config import get_llm_settings

        return str(get_llm_settings().get("default_provider", "openrouter")).strip()
    except Exception:
        return str(os.environ.get("AGENT_PROVIDER", "openrouter")).strip()


def _provider_for_config_env(config_key: str) -> str | None:
    try:
        from .llm_configs import get_default_llm_configs
        from .llm_manager import LLMProvider
    except ImportError:
        from agent_ng.llm_configs import get_default_llm_configs
        from agent_ng.llm_manager import LLMProvider

    for pe, cfg in get_default_llm_configs().items():
        if isinstance(pe, LLMProvider) and cfg.api_key_env == config_key:
            return pe.value
    return None


def get_api_key(
    config_key: str,
    override_key: str | None = None,
    session_id: str | None = None,
) -> str | None:
    """
    Get API key with unified resolution.

    Resolution order:
    1. Direct ``override_key`` parameter
    2. Session ``llm_provider_api_keys`` for the provider that owns ``config_key``
    3. Environment variable ``config_key``
    """
    if override_key:
        return override_key

    if session_id:
        session_config = get_session_config(session_id)
        keys_map = _session_keys_map(session_config)
        if keys_map:
            prov = _provider_for_config_env(config_key)
            if prov:
                hit = (keys_map.get(prov) or "").strip()
                if hit:
                    return hit

    return os.getenv(config_key) or None


def get_provider_api_key(
    provider: str | None = None,
    override_key: str | None = None,
    session_id: str | None = None,
) -> str | None:
    """
    Get API key for a specific provider using its config env var.

    Resolution order for key:
    1. ``override_key``
    2. Session ``llm_provider_api_keys[provider]``
    3. Environment variable from provider config

    When ``provider`` is omitted, ``default_provider`` from settings/env is used.
    """
    try:
        from agent_ng.llm_configs import get_default_llm_configs
        from agent_ng.llm_manager import LLMProvider
    except ImportError:
        from agent_ng.llm_configs import get_default_llm_configs
        from agent_ng.llm_manager import LLMProvider

    resolved_provider = (provider or "").strip() if provider else ""
    if not resolved_provider:
        resolved_provider = _default_llm_provider_str()

    try:
        provider_enum = LLMProvider(resolved_provider.lower())
    except ValueError:
        return None

    configs = get_default_llm_configs()
    config = configs.get(provider_enum)
    if not config:
        return None

    config_key = config.api_key_env

    final_key: str | None = None
    if override_key:
        final_key = override_key
    elif session_id:
        session_config = get_session_config(session_id)
        hit = (_session_keys_map(session_config).get(provider_enum.value) or "").strip()
        if hit:
            final_key = hit

    if not final_key:
        final_key = os.getenv(config_key) or None

    return final_key
