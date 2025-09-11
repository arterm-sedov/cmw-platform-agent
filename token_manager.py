"""
Token Management Module
======================

This module handles all token counting, chunking, and limit management functionality.
It provides accurate token estimation and intelligent chunking for large inputs.

Key Features:
- Accurate token counting using tiktoken
- Intelligent message chunking
- Token limit error handling
- Provider-specific token limits
- Chunk processing and reassembly

Usage:
    from token_manager import TokenManager
    
    token_mgr = TokenManager()
    tokens = token_mgr.estimate_tokens("Hello world")
    chunks = token_mgr.create_chunks(messages, max_tokens=1000)
"""

import tiktoken
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class TokenChunk:
    """Represents a chunk of content with token information"""
    content: str
    token_count: int
    chunk_index: int
    total_chunks: int


class TokenManager:
    """Manages token counting, chunking, and limit handling"""
    
    def __init__(self, default_token_limit: int = 4000):
        self.default_token_limit = default_token_limit
        self.provider_limits: Dict[str, int] = {}
        self.encoding = None
        self._init_encoding()
    
    def _init_encoding(self):
        """Initialize tiktoken encoding"""
        try:
            # Use GPT-4 encoding as a reasonable approximation for most models
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            print(f"[TokenManager] Warning: Failed to initialize tiktoken encoding: {e}")
            self.encoding = None
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using tiktoken for accurate counting.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Estimated token count
        """
        if not text:
            return 0
            
        try:
            if self.encoding:
                tokens = self.encoding.encode(text)
                return len(tokens)
            else:
                # Fallback to character-based estimation if tiktoken fails
                # Rough approximation: 1 token ≈ 4 characters for English text
                return len(text) // 4
        except Exception as e:
            print(f"[TokenManager] Warning: Token estimation failed: {e}")
            # Fallback to character-based estimation
            return len(text) // 4
    
    def estimate_message_tokens(self, message: Union[str, Dict[str, Any]]) -> int:
        """
        Estimate tokens for a message object.
        
        Args:
            message: Message object or string
            
        Returns:
            int: Estimated token count
        """
        if isinstance(message, str):
            return self.estimate_tokens(message)
        elif isinstance(message, dict):
            # Handle different message formats
            content = message.get('content', '')
            if isinstance(content, list):
                # Handle content as list of text blocks
                total_tokens = 0
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        total_tokens += self.estimate_tokens(block['text'])
                    elif isinstance(block, str):
                        total_tokens += self.estimate_tokens(block)
                return total_tokens
            elif isinstance(content, str):
                return self.estimate_tokens(content)
            else:
                return self.estimate_tokens(str(content))
        else:
            return self.estimate_tokens(str(message))
    
    def estimate_messages_tokens(self, messages: List[Union[str, Dict[str, Any]]]) -> int:
        """
        Estimate total tokens for a list of messages.
        
        Args:
            messages: List of message objects or strings
            
        Returns:
            int: Total estimated token count
        """
        total_tokens = 0
        for message in messages:
            total_tokens += self.estimate_message_tokens(message)
        return total_tokens
    
    def set_provider_limit(self, provider: str, limit: int):
        """Set token limit for a specific provider"""
        self.provider_limits[provider] = limit
    
    def get_provider_limit(self, provider: str) -> int:
        """Get token limit for a specific provider"""
        return self.provider_limits.get(provider, self.default_token_limit)
    
    def create_chunks(self, content: List[str], max_tokens_per_chunk: int) -> List[TokenChunk]:
        """
        Create chunks of content that fit within the token limit.
        
        Args:
            content: List of content strings to chunk
            max_tokens_per_chunk: Maximum tokens per chunk
            
        Returns:
            List[TokenChunk]: List of token chunks
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for item in content:
            item_tokens = self.estimate_tokens(item)
            
            # If single item exceeds limit, add it as a separate chunk
            if item_tokens > max_tokens_per_chunk:
                # Add current chunk if it has content
                if current_chunk:
                    chunk_content = "\n\n".join(current_chunk)
                    chunks.append(TokenChunk(
                        content=chunk_content,
                        token_count=current_tokens,
                        chunk_index=len(chunks),
                        total_chunks=0  # Will be updated later
                    ))
                    current_chunk = []
                    current_tokens = 0
                
                # Add oversized item as separate chunk
                chunks.append(TokenChunk(
                    content=item,
                    token_count=item_tokens,
                    chunk_index=len(chunks),
                    total_chunks=0  # Will be updated later
                ))
                continue
            
            # Check if adding this item would exceed the limit
            if current_tokens + item_tokens > max_tokens_per_chunk and current_chunk:
                # Create chunk from current content
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(TokenChunk(
                    content=chunk_content,
                    token_count=current_tokens,
                    chunk_index=len(chunks),
                    total_chunks=0  # Will be updated later
                ))
                
                # Start new chunk
                current_chunk = [item]
                current_tokens = item_tokens
            else:
                # Add to current chunk
                current_chunk.append(item)
                current_tokens += item_tokens
        
        # Add final chunk if it has content
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(TokenChunk(
                content=chunk_content,
                token_count=current_tokens,
                chunk_index=len(chunks),
                total_chunks=0  # Will be updated later
            ))
        
        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
        
        return chunks
    
    def create_message_chunks(self, messages: List[Union[str, Dict[str, Any]]], 
                            max_tokens_per_chunk: int) -> List[List[Union[str, Dict[str, Any]]]]:
        """
        Create chunks of messages that fit within the token limit.
        
        Args:
            messages: List of message objects
            max_tokens_per_chunk: Maximum tokens per chunk
            
        Returns:
            List[List]: List of message chunks
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for message in messages:
            message_tokens = self.estimate_message_tokens(message)
            
            # If single message exceeds limit, add it as a separate chunk
            if message_tokens > max_tokens_per_chunk:
                # Add current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                
                # Add oversized message as separate chunk
                chunks.append([message])
                continue
            
            # Check if adding this message would exceed the limit
            if current_tokens + message_tokens > max_tokens_per_chunk and current_chunk:
                # Create chunk from current messages
                chunks.append(current_chunk)
                
                # Start new chunk
                current_chunk = [message]
                current_tokens = message_tokens
            else:
                # Add to current chunk
                current_chunk.append(message)
                current_tokens += message_tokens
        
        # Add final chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def is_token_limit_error(self, error: Exception, llm_type: str = "unknown") -> bool:
        """
        Check if the error is a token limit error.
        
        Args:
            error: The exception object
            llm_type: Type of LLM for specific error patterns
            
        Returns:
            bool: True if it's a token limit error
        """
        error_str = str(error).lower()
        
        # Common token limit error patterns
        token_patterns = [
            "token", "limit", "413", "too long", "maximum length",
            "context length", "input too long", "exceeds maximum"
        ]
        
        # Provider-specific patterns
        provider_patterns = {
            "huggingface": ["router.huggingface.co", "500 server error"],
            "groq": ["context_length_exceeded", "input too long"],
            "gemini": ["content_filter", "safety"],
            "openrouter": ["context_length_exceeded", "too many tokens"],
            "mistral": ["context_length_exceeded", "input too long"],
            "gigachat": ["token limit", "too long"]
        }
        
        # Check general patterns
        for pattern in token_patterns:
            if pattern in error_str:
                return True
        
        # Check provider-specific patterns
        if llm_type in provider_patterns:
            for pattern in provider_patterns[llm_type]:
                if pattern in error_str:
                    return True
        
        return False
    
    def get_safe_token_limit(self, provider: str, safety_factor: float = 0.6) -> int:
        """
        Get a safe token limit for a provider (reduced by safety factor).
        
        Args:
            provider: Provider name
            safety_factor: Safety factor (0.0 to 1.0)
            
        Returns:
            int: Safe token limit
        """
        limit = self.get_provider_limit(provider)
        return int(limit * safety_factor)
    
    def calculate_chunk_overhead(self, messages: List[Union[str, Dict[str, Any]]]) -> int:
        """
        Calculate overhead tokens for chunking (system prompts, formatting, etc.).
        
        Args:
            messages: List of messages
            
        Returns:
            int: Estimated overhead tokens
        """
        # Estimate overhead for system prompts, formatting, etc.
        # This is a rough approximation
        base_overhead = 100  # Base overhead for system prompts
        message_overhead = len(messages) * 10  # Overhead per message
        return base_overhead + message_overhead
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token manager statistics"""
        return {
            "default_token_limit": self.default_token_limit,
            "provider_limits": self.provider_limits,
            "encoding_available": self.encoding is not None,
            "total_providers": len(self.provider_limits)
        }


# Global token manager instance
_token_manager = None

def get_token_manager() -> TokenManager:
    """Get the global token manager instance"""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager

def reset_token_manager():
    """Reset the global token manager instance"""
    global _token_manager
    _token_manager = None
