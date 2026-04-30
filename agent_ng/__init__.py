"""
Next-Generation Agent Package
============================

This package contains the next-generation modular agent implementation
with clean separation of concerns and modern async/streaming patterns.

Key Modules:
- langchain_agent: LangChain-native agent implementation
- app_ng: Gradio application interface
- llm_manager: LLM provider management and configuration
- error_handler: Error handling and fallback mechanisms
- streaming_manager: Real-time streaming capabilities
- message_processor: Message processing and formatting
- response_processor: Response processing and validation
- stats_manager: Statistics tracking and monitoring
- debug_streamer: Debug system and logging
- streaming_chat: Chat interface components
- langchain_wrapper: LangChain integration utilities
"""

# LangChain agent classes and functions
# Error handling
from .error_handler import ErrorHandler, ErrorInfo, ErrorType, get_error_handler

# Unified key resolution
from .key_resolution import get_api_key, get_provider_api_key
from .langchain_agent import ChatMessage
from .langchain_agent import CmwAgent as NextGenAgent

# App interface - import only when needed to avoid circular imports
# from .app_ng import NextGenApp, get_demo, main
# LLM management
from .llm_manager import (
    LLMConfig,
    LLMInstance,
    LLMManager,
    LLMProvider,
    get_llm_manager,
)

# Note: Global agent instances have been removed in favor of session-specific agents
# Use SessionManager.get_session_agent(session_id) to get agent instances

__all__ = [
    "ChatMessage",
    # Error handling
    "ErrorHandler",
    "ErrorInfo",
    "ErrorType",
    "LLMConfig",
    "LLMInstance",
    # App interface - commented out to avoid circular imports
    # 'NextGenApp',
    # 'get_demo',
    # 'main',
    # LLM management
    "LLMManager",
    "LLMProvider",
    # LangChain agent
    "NextGenAgent",
    "get_error_handler",
    "get_llm_manager",
]
