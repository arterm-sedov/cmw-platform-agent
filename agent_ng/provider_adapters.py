"""
Provider-Specific Adapters
==========================

This module contains provider-specific message formatting and handling utilities.
It centralizes all provider-specific logic to make the codebase more maintainable
and extensible for future providers.

Key Features:
- Centralized provider detection
- Provider-specific message conversion
- Extensible architecture for new providers
- Clean separation of concerns
"""

import re
from typing import List, Dict, Any, Optional, Union, Callable
from langchain_core.messages import BaseMessage

# Import ChatMistralAI only when needed to avoid import errors
try:
    from langchain_mistralai.chat_models import ChatMistralAI
except ImportError:
    ChatMistralAI = None


class ProviderDetector:
    """Detects provider-specific models by name patterns."""
    
    # Provider detection patterns
    PROVIDER_PATTERNS = {
        'mistral': [
            r'mistral',           # Contains "mistral"
            r'codestral',         # Codestral models
            r'pixtral',           # Pixtral models
        ],
        'openai': [
            r'gpt-',
            r'openai/',
        ],
        'anthropic': [
            r'claude-',
            r'anthropic/',
        ],
        'google': [
            r'gemini-',
            r'google/',
        ],
        'deepseek': [
            r'deepseek/',
        ],
        'qwen': [
            r'qwen/',
        ],
    }
    
    @classmethod
    def detect_provider(cls, model_name: str) -> Optional[str]:
        """
        Detect the provider type based on model name.
        
        Args:
            model_name: The model name to check
            
        Returns:
            Provider name or None if not detected
        """
        if not model_name:
            return None
        
        model_lower = model_name.lower()
        
        for provider, patterns in cls.PROVIDER_PATTERNS.items():
            if any(re.search(pattern, model_lower) for pattern in patterns):
                return provider
        
        return None
    
    @classmethod
    def is_mistral_model(cls, model_name: str) -> bool:
        """Check if a model is a Mistral model."""
        return cls.detect_provider(model_name) == 'mistral'


class MistralMessageConverter:
    """Handles Mistral-specific message conversion requirements."""
    
    @staticmethod
    def convert_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """
        Convert LangChain messages to Mistral AI compatible format.
        
        Mistral AI requires specific message formatting for tool calls:
        - Tool messages must immediately follow the assistant message that made the tool calls
        - Cannot have user messages directly after tool messages
        - Must insert assistant response after tool messages
        
        Args:
            messages: List of LangChain message objects
            
        Returns:
            List of messages in Mistral AI compatible format
        """
        converted_messages = []
        i = 0
        orphaned_count = 0
        
        while i < len(messages):
            msg = messages[i]
            
            if hasattr(msg, 'type'):
                if msg.type == 'system':
                    converted_messages.append({
                        "role": "system",
                        "content": msg.content
                    })
                elif msg.type == 'human':
                    converted_messages.append({
                        "role": "user",
                        "content": msg.content
                    })
                elif msg.type == 'ai':
                    # For AI messages with tool calls, handle them specially
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        # Add the assistant message with tool calls
                        converted_messages.append({
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": msg.tool_calls
                        })
                        
                        # Look ahead for tool messages that should immediately follow
                        j = i + 1
                        while j < len(messages) and hasattr(messages[j], 'type') and messages[j].type == 'tool':
                            tool_msg = messages[j]
                            converted_messages.append({
                                "role": "tool",
                                "name": getattr(tool_msg, 'name', 'unknown'),
                                "content": tool_msg.content,
                                "tool_call_id": getattr(tool_msg, 'tool_call_id', getattr(tool_msg, 'name', 'unknown'))
                            })
                            j += 1
                        
                        # If the next message is a user/system or we're at the end,
                        # insert an assistant turn to properly close the tool-call block
                        if j >= len(messages) or (
                            hasattr(messages[j], 'type') and messages[j].type in ('human', 'system')
                        ):
                            # Mistral requires assistant messages to have non-empty content
                            converted_messages.append({
                                "role": "assistant",
                                "content": "[continue]"
                            })
                        
                        # Skip the tool messages we've already processed
                        i = j - 1
                    else:
                        converted_messages.append({
                            "role": "assistant",
                            "content": msg.content or ""
                        })
                elif msg.type == 'tool':
                    # Tool messages should only appear after assistant messages with tool calls
                    # If we encounter one here, it might be orphaned - skip it
                    if orphaned_count < 2:
                        print(f"[Mistral Conversion] Warning: Orphaned tool message detected: {getattr(msg, 'name', 'unknown')}")
                        orphaned_count += 1
                    # Skip this message and continue to next
                    i += 1
                    continue
            else:
                # Handle raw message objects (fallback)
                converted_messages.append(msg)
            
            i += 1
        
        return converted_messages


class ProviderAdapter:
    """Main adapter class that handles provider-specific logic."""
    
    # Registry of provider-specific converters
    _converters: Dict[str, Callable] = {
        'mistral': MistralMessageConverter.convert_messages,
    }
    
    @classmethod
    def get_converter(cls, provider: str) -> Optional[Callable]:
        """Get the message converter for a specific provider."""
        return cls._converters.get(provider)
    
    @classmethod
    def register_converter(cls, provider: str, converter: Callable) -> None:
        """Register a new provider converter."""
        cls._converters[provider] = converter
    
    @classmethod
    def convert_messages_for_provider(cls, messages: List[BaseMessage], model_name: str) -> List[Union[BaseMessage, Dict[str, Any]]]:
        """
        Convert messages for the appropriate provider based on model name.
        
        Args:
            messages: List of LangChain message objects
            model_name: The model name to determine provider
            
        Returns:
            Converted messages (either LangChain objects or provider-specific format)
        """
        provider = ProviderDetector.detect_provider(model_name)
        
        if provider and provider in cls._converters:
            converter = cls._converters[provider]
            return converter(messages)
        
        # Return original messages if no converter found
        return messages


class MistralWrapper:
    """
    Wrapper for Mistral AI LLM instances that handles message conversion.
    
    This wrapper ensures that messages are properly formatted for Mistral AI's
    strict requirements regarding tool call message ordering.
    """
    
    def __init__(self, mistral_llm: Union[ChatMistralAI, Any]):
        """
        Initialize the wrapper with a Mistral AI LLM instance.
        
        Args:
            mistral_llm: The underlying Mistral AI LLM instance
        """
        self.llm = mistral_llm
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Invoke the LLM with message conversion for Mistral compatibility.
        
        Args:
            messages: List of LangChain message objects
            **kwargs: Additional arguments for the LLM
            
        Returns:
            The LLM response
        """
        try:
            # Convert messages to Mistral format
            converted_messages = MistralMessageConverter.convert_messages(messages)
            
            # Call the original invoke method with converted messages
            return self.llm.invoke(converted_messages, **kwargs)
        except Exception as e:
            # If conversion fails, try with original messages as fallback
            print(f"[Mistral Wrapper] Message conversion failed, using original format: {e}")
            return self.llm.invoke(messages, **kwargs)
    
    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Async invoke the LLM with message conversion for Mistral compatibility.
        
        Args:
            messages: List of LangChain message objects
            **kwargs: Additional arguments for the LLM
            
        Returns:
            The LLM response
        """
        try:
            # Convert messages to Mistral format
            converted_messages = MistralMessageConverter.convert_messages(messages)
            
            # Call the original ainvoke method with converted messages
            return await self.llm.ainvoke(converted_messages, **kwargs)
        except Exception as e:
            # If conversion fails, try with original messages as fallback
            print(f"[Mistral Wrapper] Message conversion failed, using original format: {e}")
            return await self.llm.ainvoke(messages, **kwargs)
    
    def __getattr__(self, name):
        """
        Delegate all other attributes to the wrapped LLM instance.
        
        Args:
            name: Attribute name
            
        Returns:
            The attribute value from the wrapped LLM
        """
        return getattr(self.llm, name)


# Convenience functions for backward compatibility
def is_mistral_model(model_name: str) -> bool:
    """Check if a model is a Mistral model."""
    return ProviderDetector.is_mistral_model(model_name)


def convert_messages_for_mistral(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Convert messages for Mistral AI compatibility."""
    return MistralMessageConverter.convert_messages(messages)
