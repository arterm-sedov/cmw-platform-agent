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
        self._tool_call_traces: Dict[str, List[Dict[str, Any]]] = {}  # Track tool call sequences
        self._duplicate_counts: Dict[str, Dict[str, int]] = {}  # Track duplicate counts per conversation
    
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
    
    def is_duplicate(self, tool_name: str, tool_args: Dict[str, Any], 
                    conversation_id: str = "default") -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if a tool call is a duplicate and return cached result if available.
        
        Args:
            tool_name: Name of the tool
            tool_args: Arguments passed to the tool
            conversation_id: Conversation identifier
            
        Returns:
            Tuple of (is_duplicate, cached_result)
        """
        # Normalize arguments for consistent comparison
        normalized_args = self._normalize_tool_args(tool_args)
        hash_value = self._create_tool_call_hash(tool_name, normalized_args)
        
        # Check if we have this exact tool call in this conversation
        if conversation_id in self._conversation_hashes:
            for cached_hash in self._conversation_hashes[conversation_id]:
                if (cached_hash.tool_name == tool_name and 
                    cached_hash.hash_value == hash_value):
                    # Found duplicate, increment count and return cached result
                    self._increment_duplicate_count(tool_name, hash_value, conversation_id)
                    cache_key = f"{conversation_id}:{hash_value}"
                    cached_result = self._tool_call_cache.get(cache_key)
                    return True, cached_result
        
        return False, None
    
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
        
        # Add to trace
        self._add_to_trace(tool_name, normalized_args, tool_result, conversation_id)
        
        # Initialize duplicate count for this tool call
        self._initialize_duplicate_count(tool_name, hash_value, conversation_id)
    
    def _add_to_trace(self, tool_name: str, tool_args: Dict[str, Any], 
                     tool_result: Any, conversation_id: str) -> None:
        """Add a tool call to the trace for a conversation"""
        if conversation_id not in self._tool_call_traces:
            self._tool_call_traces[conversation_id] = []
        
        self._tool_call_traces[conversation_id].append({
            'tool_name': tool_name,
            'tool_args': tool_args,
            'result': tool_result,
            'timestamp': time.time()
        })
    
    def get_trace_after_duplicate(self, tool_name: str, tool_args: Dict[str, Any], 
                                 conversation_id: str, max_traces: int = 5) -> List[Dict[str, Any]]:
        """
        Get tool calls executed after a duplicate was detected.
        
        Args:
            tool_name: Name of the duplicated tool
            tool_args: Arguments of the duplicated tool
            conversation_id: Conversation identifier
            max_traces: Maximum number of traces to return
            
        Returns:
            List of tool calls executed after the duplicate
        """
        if conversation_id not in self._tool_call_traces:
            return []
        
        # Find the duplicate tool call in the trace
        duplicate_hash = self._create_tool_call_hash(tool_name, self._normalize_tool_args(tool_args))
        duplicate_index = -1
        
        for i, trace in enumerate(self._tool_call_traces[conversation_id]):
            trace_hash = self._create_tool_call_hash(trace['tool_name'], trace['tool_args'])
            if trace_hash == duplicate_hash:
                duplicate_index = i
                break
        
        if duplicate_index == -1:
            return []
        
        # Return tool calls after the duplicate
        return self._tool_call_traces[conversation_id][duplicate_index + 1:duplicate_index + 1 + max_traces]
    
    def get_continuation_prompt(self, tool_name: str, tool_args: Dict[str, Any], 
                               conversation_id: str = "default") -> str:
        """
        Get a lean continuation prompt for the LLM when a duplicate is detected.
        Simple 50-character reminder without complex traces.
        
        Args:
            tool_name: Name of the tool that was duplicated
            tool_args: Arguments that were duplicated
            conversation_id: Conversation identifier
            
        Returns:
            Simple continuation prompt
        """
        # Create a lean, non-complex prompt with 50-char limit
        args_summary = ", ".join([f"{k}={v}" for k, v in tool_args.items()][:2])
        if len(tool_args) > 2:
            args_summary += "..."
        
        # Keep it under 50 characters
        prompt = f"{tool_name}({args_summary}) already called. Continue."
        if len(prompt) > 50:
            prompt = f"{tool_name} already called. Continue."
        
        return prompt
    
    def _initialize_duplicate_count(self, tool_name: str, hash_value: str, conversation_id: str) -> None:
        """Initialize duplicate count for a tool call"""
        if conversation_id not in self._duplicate_counts:
            self._duplicate_counts[conversation_id] = {}
        
        count_key = f"{tool_name}:{hash_value}"
        if count_key not in self._duplicate_counts[conversation_id]:
            self._duplicate_counts[conversation_id][count_key] = 0
    
    def _increment_duplicate_count(self, tool_name: str, hash_value: str, conversation_id: str) -> None:
        """Increment duplicate count for a tool call"""
        if conversation_id not in self._duplicate_counts:
            self._duplicate_counts[conversation_id] = {}
        
        count_key = f"{tool_name}:{hash_value}"
        if count_key not in self._duplicate_counts[conversation_id]:
            self._duplicate_counts[conversation_id][count_key] = 0
        
        self._duplicate_counts[conversation_id][count_key] += 1
    
    def get_duplicate_count(self, tool_name: str, tool_args: Dict[str, Any], conversation_id: str = "default") -> int:
        """Get the number of times a tool call has been duplicated"""
        normalized_args = self._normalize_tool_args(tool_args)
        hash_value = self._create_tool_call_hash(tool_name, normalized_args)
        count_key = f"{tool_name}:{hash_value}"
        
        if conversation_id in self._duplicate_counts:
            return self._duplicate_counts[conversation_id].get(count_key, 0)
        return 0
    
    def clear_conversation(self, conversation_id: str = "default") -> None:
        """
        Clear deduplication cache for a specific conversation.
        
        Args:
            conversation_id: Conversation identifier
        """
        if conversation_id in self._conversation_hashes:
            del self._conversation_hashes[conversation_id]
        
        if conversation_id in self._tool_call_traces:
            del self._tool_call_traces[conversation_id]
        
        if conversation_id in self._duplicate_counts:
            del self._duplicate_counts[conversation_id]
        
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
