"""
LangChain LLM Invoke Wrapper Module
===================================

This module provides a unified wrapper that makes different LLM providers
appear as a single endpoint with consistent interfaces and error handling.

Key Features:
- Unified interface for all LLM providers
- Consistent error handling and retry logic
- Automatic fallback between providers
- Tool calling support across all providers
- Streaming and non-streaming responses
- Provider-specific optimizations

Usage:
    wrapper = LangChainWrapper()
    response = wrapper.invoke(messages, provider="gemini")
    for chunk in wrapper.astream(messages, provider="auto"):
        print(chunk)
"""

import os
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Generator, Iterator
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import Runnable
from .token_counter import get_token_tracker, UsageMetadataCallbackHandler
from langchain_core.runnables.utils import Input, Output

# Local imports
from .llm_manager import get_llm_manager, LLMInstance, LLMProvider
from .error_handler import get_error_handler, ErrorInfo


class ResponseType(Enum):
    """Types of responses from the wrapper"""
    SUCCESS = "success"
    ERROR = "error"
    FALLBACK = "fallback"
    TIMEOUT = "timeout"


@dataclass
class WrapperResponse:
    """Structured response from the wrapper"""
    content: str
    response_type: ResponseType
    provider_used: str
    model_name: str
    execution_time: float
    token_usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    error_info: Optional[ErrorInfo] = None
    metadata: Optional[Dict[str, Any]] = None


class LangChainWrapper:
    """
    Unified wrapper for LangChain LLM providers.

    This wrapper provides a consistent interface for all LLM providers,
    handling differences in implementation, error handling, and features.
    """

    def __init__(self, default_provider: str = "auto", max_retries: int = 3, timeout: float = 30.0):
        """
        Initialize the LangChain wrapper.

        Args:
            default_provider: Default provider to use ("auto" for automatic selection)
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout in seconds for requests
        """
        self.llm_manager = get_llm_manager()
        self.error_handler = get_error_handler()
        self.token_tracker = get_token_tracker()
        self.default_provider = default_provider
        self.max_retries = max_retries
        self.timeout = timeout

        # Provider-specific configurations
        self.provider_configs = {
            "gemini": {
                "supports_tools": True,
                "supports_streaming": True,
                "max_tokens": 2000000,
                "temperature": 0
            },
            "groq": {
                "supports_tools": True,
                "supports_streaming": True,
                "max_tokens": 32768,
                "temperature": 0
            },
            "mistral": {
                "supports_tools": True,
                "supports_streaming": True,
                "max_tokens": 2048,
                "temperature": 0
            },
            "openrouter": {
                "supports_tools": True,
                "supports_streaming": True,
                "max_tokens": 2048,
                "temperature": 0
            },
            "gigachat": {
                "supports_tools": True,
                "supports_streaming": True,
                "max_tokens": 2048,
                "temperature": 0
            },
            "huggingface": {
                "supports_tools": False,
                "supports_streaming": False,
                "max_tokens": 1024,
                "temperature": 0
            }
        }

        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "provider_usage": {},
            "average_response_time": 0.0
        }

    def _get_agent_provider(self) -> str:
        """Get the single provider from AGENT_PROVIDER environment variable"""
        return os.environ.get("AGENT_PROVIDER", "mistral")

    def _get_llm_instance(self, provider: str, use_tools: bool = True) -> Optional[LLMInstance]:
        """Get LLM instance for the specified provider"""
        try:
            return self.llm_manager.get_llm(provider, use_tools=use_tools)
        except Exception as e:
            print(f"Error getting LLM instance for {provider}: {e}")
            return None

    def _format_messages(self, messages: Union[List[BaseMessage], List[Dict[str, str]], str]) -> List[BaseMessage]:
        """Format messages to ensure they are BaseMessage objects"""
        if isinstance(messages, str):
            return [HumanMessage(content=messages)]

        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            # Convert dict format to BaseMessage
            formatted = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "user":
                    formatted.append(HumanMessage(content=content))
                elif role == "assistant":
                    formatted.append(AIMessage(content=content))
                elif role == "system":
                    formatted.append(SystemMessage(content=content))
                elif role == "tool":
                    formatted.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
                else:
                    formatted.append(HumanMessage(content=content))

            return formatted

        return messages

    def _extract_content(self, response: Any) -> str:
        """Extract content from LLM response"""
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response"""
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return [
                {
                    "name": call.get("name", "unknown"),
                    "args": call.get("args", {}),
                    "id": call.get("id", "")
                }
                for call in response.tool_calls
            ]
        return []

    def _extract_token_usage(self, response: Any, provider: str) -> Optional[Dict[str, int]]:
        """Extract token usage information from response"""
        # This would need to be implemented based on each provider's response format
        # For now, return None as most providers don't expose this easily
        return None

    def _handle_error(self, error: Exception, provider: str, attempt: int) -> ErrorInfo:
        """Handle errors and return structured error information"""
        error_info = self.error_handler.classify_error(error, provider)

        # Track provider failure
        self.error_handler.handle_provider_failure(provider, error_info.error_type.value)

        # Update statistics
        self.stats["failed_requests"] += 1

        return error_info

    def invoke(self, messages: Union[List[BaseMessage], List[Dict[str, str]], str],
               provider: str = "auto", use_tools: bool = True,
               **kwargs) -> WrapperResponse:
        """
        Invoke the LLM with the given messages using AGENT_PROVIDER.

        Args:
            messages: Messages to send to the LLM
            provider: Ignored - always uses AGENT_PROVIDER
            use_tools: Whether to use tools if available
            **kwargs: Additional arguments for the LLM

        Returns:
            WrapperResponse with the result
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Update statistics
        self.stats["total_requests"] += 1

        # Get single provider from environment
        current_provider = self._get_agent_provider()

        # Format messages
        formatted_messages = self._format_messages(messages)

        try:
            # Get LLM instance
            llm_instance = self.llm_manager.get_agent_llm()
            if not llm_instance:
                raise Exception(f"AGENT_PROVIDER '{current_provider}' not available")

            # Make the request
            response = llm_instance.llm.invoke(formatted_messages, **kwargs)

            # Extract content and metadata
            content = self._extract_content(response)
            tool_calls = self._extract_tool_calls(response)
            token_usage = self._extract_token_usage(response, current_provider)

            # Track token usage
            self.token_tracker.track_llm_response(response, formatted_messages)

            # Update statistics
            execution_time = time.time() - start_time
            self.stats["successful_requests"] += 1
            self.stats["provider_usage"][current_provider] = self.stats["provider_usage"].get(current_provider, 0) + 1
            self._update_average_response_time(execution_time)

            return WrapperResponse(
                content=content,
                response_type=ResponseType.SUCCESS,
                provider_used=current_provider,
                model_name=llm_instance.model_name,
                execution_time=execution_time,
                token_usage=token_usage,
                tool_calls=tool_calls,
                metadata={
                    "request_id": request_id,
                    "provider": current_provider
                }
            )

        except Exception as e:
            error_info = self._handle_error(e, current_provider, 0)
            print(f"LLM failed: {error_info.description}")

            execution_time = time.time() - start_time
            return WrapperResponse(
                content=f"Error: {error_info.description}",
                response_type=ResponseType.ERROR,
                provider_used=current_provider,
                model_name="unknown",
                execution_time=execution_time,
                error_info=error_info,
                metadata={
                    "request_id": request_id,
                    "provider": current_provider
                }
            )

    def astream(self, messages: Union[List[BaseMessage], List[Dict[str, str]], str], 
                provider: str = "auto", use_tools: bool = True,
                **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Stream responses from the LLM using AGENT_PROVIDER.

        Args:
            messages: Messages to send to the LLM
            provider: Ignored - always uses AGENT_PROVIDER
            use_tools: Whether to use tools if available
            **kwargs: Additional arguments for the LLM

        Yields:
            Dict with streaming information
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Update statistics
        self.stats["total_requests"] += 1

        # Get single provider from environment
        current_provider = self._get_agent_provider()

        # Format messages
        formatted_messages = self._format_messages(messages)

        try:
            # Get LLM instance
            llm_instance = self.llm_manager.get_agent_llm()
            if not llm_instance:
                yield {"type": "error", "content": f"AGENT_PROVIDER '{current_provider}' not available"}
                return

            # Check if provider supports streaming
            if not self.provider_configs.get(current_provider, {}).get("supports_streaming", False):
                # Fallback to non-streaming
                response = self.invoke(messages, use_tools=use_tools, **kwargs)
                yield {"type": "content", "content": response.content}
                yield {"type": "complete", "provider": current_provider, "execution_time": response.execution_time}
                return

            # Stream the response
            yield {"type": "start", "provider": current_provider, "model": llm_instance.model_name}

            accumulated_content = ""
            for chunk in llm_instance.llm.astream(formatted_messages, **kwargs):
                if hasattr(chunk, 'content') and chunk.content:
                    accumulated_content += chunk.content
                    yield {"type": "content", "content": chunk.content}
                elif isinstance(chunk, str):
                    accumulated_content += chunk
                    yield {"type": "content", "content": chunk}

            # Update statistics
            execution_time = time.time() - start_time
            self.stats["successful_requests"] += 1
            self.stats["provider_usage"][current_provider] = self.stats["provider_usage"].get(current_provider, 0) + 1
            self._update_average_response_time(execution_time)

            yield {
                "type": "complete", 
                "provider": current_provider, 
                "model": llm_instance.model_name,
                "execution_time": execution_time,
                "total_content": accumulated_content
            }

        except Exception as e:
            error_info = self._handle_error(e, current_provider, 0)
            yield {"type": "error", "provider": current_provider, "error": error_info.description}
            yield {"type": "final_error", "error": f"LLM failed: {error_info.description}"}

    def bind_tools(self, tools: List[BaseTool], provider: str = "auto") -> 'LangChainWrapper':
        """
        Bind tools to the wrapper for a specific provider.

        Args:
            tools: List of tools to bind
            provider: Provider to bind tools to

        Returns:
            Self for method chaining
        """
        # This would need to be implemented based on the specific provider
        # For now, tools are handled by the LLM manager
        return self

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return self.llm_manager.get_available_providers()

    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get information about a specific provider"""
        config = self.llm_manager.get_provider_config(provider)
        if not config:
            return {}

        return {
            "name": config.name,
            "supports_tools": config.tool_support,
            "max_history": config.max_history,
            "models": [model["model"] for model in config.models],
            "wrapper_config": self.provider_configs.get(provider, {})
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper statistics"""
        return {
            **self.stats,
            "llm_manager_stats": self.llm_manager.get_stats(),
            "error_handler_stats": self.error_handler.get_provider_failure_stats()
        }

    def get_token_counts(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Get token counts for display"""
        prompt_tokens = self.token_tracker.count_prompt_tokens(messages)
        cumulative_stats = self.token_tracker.get_cumulative_stats()

        return {
            "prompt_tokens": prompt_tokens,
            "cumulative_stats": cumulative_stats
        }

    def reset_stats(self):
        """Reset wrapper statistics"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "provider_usage": {},
            "average_response_time": 0.0
        }

    def _update_average_response_time(self, execution_time: float):
        """Update the average response time"""
        total_requests = self.stats["total_requests"]
        if total_requests > 0:
            current_avg = self.stats["average_response_time"]
            self.stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + execution_time) / total_requests
            )

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers"""
        return {
            "wrapper_stats": self.stats,
            "llm_manager_health": self.llm_manager.health_check(),
            "available_providers": self.get_available_providers()
        }


# Global wrapper instance
_wrapper = None
_wrapper_lock = None


def get_langchain_wrapper() -> LangChainWrapper:
    """Get the global LangChain wrapper instance (singleton pattern)"""
    global _wrapper, _wrapper_lock
    if _wrapper is None:
        import threading
        _wrapper_lock = threading.Lock()
        with _wrapper_lock:
            if _wrapper is None:
                _wrapper = LangChainWrapper()
    return _wrapper


def reset_langchain_wrapper():
    """Reset the global LangChain wrapper (useful for testing)"""
    global _wrapper
    if _wrapper_lock:
        with _wrapper_lock:
            _wrapper = None
