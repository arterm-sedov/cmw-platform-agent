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
import logging

# Global Langfuse client - initialized once per application
_global_langfuse_client = None
_logger = logging.getLogger(__name__)


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


def _ensure_global_langfuse_client():
    """Ensure global Langfuse client is initialized."""
    global _global_langfuse_client
    
    if _global_langfuse_client is not None:
        return _global_langfuse_client
    config = get_langfuse_config()
    if not config.is_configured():
        return None
    
    try:
        from langfuse import Langfuse
        _logger.debug(f"🔍 Langfuse Config: Initializing global client with public_key={config.public_key[:10]}..., secret_key={config.secret_key[:10]}..., host={config.host}")
        _global_langfuse_client = Langfuse(
            public_key=config.public_key, 
            secret_key=config.secret_key, 
            host=config.host
        )
        _logger.debug(f"🔍 Langfuse Config: Global client initialized: {_global_langfuse_client}")
        return _global_langfuse_client
    except Exception as e:
        _logger.debug(f"❌ Langfuse Config: Failed to initialize global client: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_langfuse_callback_handler(session_id: str | None = None):
    """Return a Langfuse CallbackHandler if configured, else None.

    Import langfuse lazily to avoid hard dependency when disabled.
    
    Args:
        session_id: Optional session ID to associate with the handler
    """
    # Ensure global client is initialized
    client = _ensure_global_langfuse_client()
    if client is None:
        return None

    try:
        from langfuse.langchain import CallbackHandler
        
        # Create handler with session_id if provided
        if session_id:
            _logger.debug(f"🔍 Langfuse Config: Created CallbackHandler with session_id: {session_id}")
            return CallbackHandler(session_id=session_id)
        else:
            _logger.debug(f"🔍 Langfuse Config: Created CallbackHandler (session_id will be passed via metadata)")
            return CallbackHandler()
    except Exception as e:
        _logger.debug(f"❌ Langfuse Config: Failed to create handler: {e}")
        import traceback
        traceback.print_exc()
        return None
