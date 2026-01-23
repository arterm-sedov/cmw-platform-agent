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
from .langchain_agent import CmwAgent as NextGenAgent, ChatMessage

# App interface - import only when needed to avoid circular imports
# from .app_ng import NextGenApp, get_demo, main

# LLM management
from .llm_manager import LLMManager, LLMProvider, LLMConfig, LLMInstance, get_llm_manager

# Error handling
from .error_handler import ErrorHandler, ErrorInfo, ErrorType, get_error_handler

# Note: Global agent instances have been removed in favor of session-specific agents
# Use SessionManager.get_session_agent(session_id) to get agent instances

__all__ = [

    # LangChain agent
    'NextGenAgent',
    'ChatMessage',

    # App interface - commented out to avoid circular imports
    # 'NextGenApp',
    # 'get_demo',
    # 'main',

    # LLM management
    'LLMManager',
    'LLMProvider', 
    'LLMConfig',
    'LLMInstance',
    'get_llm_manager',

    # Error handling
    'ErrorHandler',
    'ErrorInfo',
    'ErrorType',
    'get_error_handler'
]
