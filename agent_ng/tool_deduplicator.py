"""
Tool Call Deduplication
======================

This module provides tool call deduplication functionality to prevent
identical tool calls from being executed multiple times in the same conversation.

Key Features:
- Hash-based tool call comparison
- Lean deduplication logic
- Integration with LangChain tool calling patterns
- Simple continuation prompts for LLM
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ToolCallHash:
    """Represents a hashed tool call for deduplication"""
    tool_name: str
    tool_args: Dict[str, Any]
    hash_value: str
    
    def __str__(self) -> str:
        return f"ToolCallHash({self.tool_name}, {self.hash_value[:8]}...)"


class ToolCallDeduplicator:
    """
    Handles deduplication of identical tool calls within a conversation.
    
    This class provides lean and efficient deduplication by:
    1. Creating deterministic hashes of tool calls
    2. Storing results for identical calls
    3. Providing simple continuation prompts
    """
    
    def __init__(self):
        self._tool_call_cache: Dict[str, Dict[str, Any]] = {}
        self._conversation_hashes: Dict[str, List[ToolCallHash]] = {}
    
    def _create_tool_call_hash(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Create a deterministic hash for a tool call.
        
        Args:
            tool_name: Name of the tool
            tool_args: Arguments passed to the tool
            
        Returns:
            Hash string representing the tool call
        """
        # Sort args to ensure consistent hashing regardless of order
        sorted_args = json.dumps(tool_args, sort_keys=True, separators=(',', ':'))
        hash_input = f"{tool_name}:{sorted_args}"
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    
    def _normalize_tool_args(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize tool arguments for consistent comparison.
        
        Args:
            tool_args: Raw tool arguments
            
        Returns:
            Normalized tool arguments
        """
        if not isinstance(tool_args, dict):
            return {}
        
        # Create a deep copy and normalize
        normalized = {}
        for key, value in tool_args.items():
            if isinstance(value, str):
                # Normalize string values (trim whitespace only, preserve case)
                normalized[key] = value.strip()
            elif isinstance(value, (int, float, bool)):
                normalized[key] = value
            elif isinstance(value, list):
                # Sort lists for consistent comparison
                normalized[key] = sorted(value) if value else []
            elif isinstance(value, dict):
                # Recursively normalize nested dicts
                normalized[key] = self._normalize_tool_args(value)
            else:
                # Convert other types to string
                normalized[key] = str(value)
        
        return normalized
    
    
    def store_tool_call(self, tool_name: str, tool_args: Dict[str, Any], 
                       tool_result: Any, conversation_id: str = "default") -> None:
        """
        Store a tool call and its result for future deduplication.
        
        Args:
            tool_name: Name of the tool
            tool_args: Arguments passed to the tool
            tool_result: Result returned by the tool
            conversation_id: Conversation identifier
        """
        # Normalize arguments
        normalized_args = self._normalize_tool_args(tool_args)
        hash_value = self._create_tool_call_hash(tool_name, normalized_args)
        
        # Store the result
        cache_key = f"{conversation_id}:{hash_value}"
        self._tool_call_cache[cache_key] = {
            'tool_name': tool_name,
            'tool_args': normalized_args,
            'result': tool_result,
            'hash': hash_value
        }
        
        # Track this hash for this conversation
        if conversation_id not in self._conversation_hashes:
            self._conversation_hashes[conversation_id] = []
        
        tool_call_hash = ToolCallHash(
            tool_name=tool_name,
            tool_args=normalized_args,
            hash_value=hash_value
        )
        
        # Only add if not already present
        if not any(h.tool_name == tool_name and h.hash_value == hash_value 
                  for h in self._conversation_hashes[conversation_id]):
            self._conversation_hashes[conversation_id].append(tool_call_hash)
        
    
    
    
    
    
    
    
    def clear_conversation(self, conversation_id: str = "default") -> None:
        """
        Clear deduplication cache for a specific conversation.
        
        Args:
            conversation_id: Conversation identifier
        """
        if conversation_id in self._conversation_hashes:
            del self._conversation_hashes[conversation_id]
        
        
        # Remove cached results for this conversation
        keys_to_remove = [k for k in self._tool_call_cache.keys() 
                         if k.startswith(f"{conversation_id}:")]
        for key in keys_to_remove:
            del self._tool_call_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get deduplication statistics.
        
        Returns:
            Dictionary with deduplication stats
        """
        total_cached = len(self._tool_call_cache)
        total_conversations = len(self._conversation_hashes)
        total_hashes = sum(len(hashes) for hashes in self._conversation_hashes.values())
        
        return {
            'total_cached_results': total_cached,
            'total_conversations': total_conversations,
            'total_unique_tool_calls': total_hashes
        }


# Global deduplicator instance
_deduplicator = ToolCallDeduplicator()


def get_deduplicator() -> ToolCallDeduplicator:
    """Get the global tool call deduplicator instance."""
    return _deduplicator
