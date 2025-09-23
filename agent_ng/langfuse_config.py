"""
Langfuse Integration (Lean)
===========================

Provides an optional Langfuse CallbackHandler for LangChain callbacks,
configured via environment variables. Designed to be imported and used
at the call site near existing LangSmith @traceable usage.

Env vars:
- LANGFUSE_ENABLED=true|false
- LANGFUSE_PUBLIC_KEY=...
- LANGFUSE_SECRET_KEY=...
- LANGFUSE_HOST=https://cloud.langfuse.com (optional)
"""

from __future__ import annotations

import os


class LangfuseConfig:
    """Simple configuration holder for Langfuse."""

    def __init__(self) -> None:
        enabled_val = os.getenv("LANGFUSE_ENABLED", "false").lower()
        self.enabled: bool = enabled_val in {"1", "true", "yes"}
        self.public_key: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key: str | None = os.getenv("LANGFUSE_SECRET_KEY")
        self.host: str | None = os.getenv(
            "LANGFUSE_HOST",
            "https://cloud.langfuse.com",
        )

    def is_configured(self) -> bool:
        return bool(self.enabled and self.public_key and self.secret_key)


def get_langfuse_config() -> LangfuseConfig:
    """Load .env and return current Langfuse configuration."""
    try:
        # Lazy import to avoid hard dependency if not present
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass
    return LangfuseConfig()


def get_langfuse_callback_handler():
    """Return a Langfuse CallbackHandler if configured, else None.

    Import langfuse lazily to avoid hard dependency when disabled.
    """
    config = get_langfuse_config()
    if not config.is_configured():
        return None

    try:
        # Lazy import to avoid import errors when not installed
        from langfuse import Langfuse  # type: ignore
        from langfuse.langchain import CallbackHandler  # type: ignore
    except Exception:
        # If package missing or runtime import fails, do not break the app
        return None

    try:
        # Initialize client (kept local; handler uses global client internally)
        Langfuse(
            public_key=config.public_key, secret_key=config.secret_key, host=config.host
        )
        return CallbackHandler()
    except Exception:
        # Fail-closed to None to keep core app resilient
        return None
