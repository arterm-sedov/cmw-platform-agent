"""
Next-Generation Agent Package
============================

This package contains the next-generation modular agent implementation
with clean separation of concerns and modern async/streaming patterns.

Key Modules:
- agent_ng: Main agent class
- app_ng: Gradio application
- core_agent: Core agent functionality
- llm_manager: LLM provider management
- error_handler: Error handling and fallback
- streaming_manager: Real-time streaming
- message_processor: Message processing
- response_processor: Response processing
- stats_manager: Statistics tracking
- trace_manager: Trace logging
- debug_streamer: Debug system
- streaming_chat: Chat interface
- langchain_wrapper: LangChain integration
"""

from .agent_ng import NextGenAgent, ChatMessage, get_agent_ng
from .app_ng import NextGenApp

__all__ = [
    'NextGenAgent',
    'ChatMessage', 
    'get_agent_ng',
    'NextGenApp'
]
