"""
Tool Call Manager Module

This module handles tool call deduplication and history management
completely independently of vector storage or Supabase.

Features:
- Tool call deduplication using similarity analysis
- In-memory tool call history
- Exact and semantic duplicate detection
- Tool usage tracking and limits
- Integration with similarity manager for advanced matching

Usage:
    from tool_call_manager import tool_call_manager
    is_duplicate = tool_call_manager.is_duplicate_tool_call("tool_name", args, called_tools)
    tool_call_manager.add_tool_call_to_history("tool_name", args, called_tools)
"""

import json
from typing import List, Dict, Any, Optional

# Conditional import for similarity_manager
try:
    from similarity_manager import similarity_manager
except ImportError:
    similarity_manager = None


class ToolCallManager:
    """
    Manages tool call deduplication and history tracking.
    Completely independent of vector storage systems.
    """

    def __init__(self):
        """Initialize the tool call manager."""
        # Use lazy initialization - similarity_manager will be set when needed
        self.similarity_manager = None

    def is_duplicate_tool_call(self, tool_name: str, tool_args: dict, called_tools: list, threshold: float = 0.9) -> bool:
        """
        Check if a tool call is a duplicate.

        This method provides comprehensive duplicate detection by:
        1. First checking for exact argument matches (fast)
        2. Then using semantic similarity if available (accurate)

        Args:
            tool_name (str): Name of the tool
            tool_args (dict): Arguments for the tool
            called_tools (list): List of previously called tools
            threshold (float): Similarity threshold for semantic duplicates

        Returns:
            bool: True if this is a duplicate tool call
        """
        if not called_tools:
            return False

        # First, check for exact duplicates (fast and reliable)
        normalized_args = self._normalize_args(tool_args)

        for called_tool in called_tools:
            if called_tool.get('name') == tool_name:
                called_args = called_tool.get('args', {})
                normalized_called_args = self._normalize_args(called_args)

                # Exact argument match - definitive duplicate
                if normalized_args == normalized_called_args:
                    print(f"🔄 Exact duplicate tool call detected: {tool_name}")
                    return True

        # If no exact matches, use semantic similarity if available
        if self.similarity_manager is None and similarity_manager is not None:
            try:
                from similarity_manager import get_similarity_manager
                self.similarity_manager = get_similarity_manager()
            except ImportError:
                self.similarity_manager = None
        
        if self.similarity_manager is not None and self.similarity_manager.is_enabled():
            try:
                # Convert current args to text for similarity comparison
                current_args_text = json.dumps(tool_args, sort_keys=True) if isinstance(tool_args, dict) else str(tool_args)

                for called_tool in called_tools:
                    if called_tool.get('name') == tool_name:
                        called_args = called_tool.get('args', {})
                        called_args_text = json.dumps(called_args, sort_keys=True) if isinstance(called_args, dict) else str(called_args)

                        # Use similarity manager for semantic comparison
                        similarity = self.similarity_manager.calculate_similarity(current_args_text, called_args_text)
                        if similarity >= threshold:
                            print(f"🔄 Semantic duplicate tool call detected: {tool_name} (similarity: {similarity:.3f})")
                            return True
            except Exception as e:
                print(f"⚠️ Failed to check semantic duplicates: {e}")

        return False

    def add_tool_call_to_history(self, tool_name: str, tool_args: dict, called_tools: list) -> None:
        """
        Add a tool call to the history for future deduplication.

        Args:
            tool_name (str): Name of the tool
            tool_args (dict): Arguments for the tool
            called_tools (list): List to append to
        """
        tool_call_info = {
            'name': tool_name,
            'args': tool_args.copy() if isinstance(tool_args, dict) else tool_args,
            'timestamp': self._get_timestamp()
        }

        # Add embedding if similarity manager supports it (for future semantic analysis)
        # Note: We don't need embeddings for basic deduplication, but storing them
        # allows for more advanced analysis later if needed
        try:
            # Store the args as-is for now - embeddings can be generated on-demand
            pass  # Placeholder for future embedding functionality
        except Exception as e:
            print(f"⚠️ Failed to prepare tool call info: {e}")

        called_tools.append(tool_call_info)

    def check_tool_usage_limit(self, tool_name: str, tool_usage_count: dict, tool_usage_limits: dict) -> bool:
        """
        Check if a tool has exceeded its usage limit.

        Args:
            tool_name (str): Name of the tool
            tool_usage_count (dict): Current usage counts
            tool_usage_limits (dict): Usage limits per tool

        Returns:
            bool: True if limit exceeded
        """
        current_count = tool_usage_count.get(tool_name, 0)
        limit = tool_usage_limits.get(tool_name, tool_usage_limits.get('default', 3))

        if current_count >= limit:
            print(f"⚠️ {tool_name} usage limit reached ({current_count}/{limit})")
            return True

        return False

    def get_duplicate_tool_reminder(self, tool_name: str, tool_args: dict = None) -> str:
        """
        Generate a reminder message for duplicate tool usage.

        Args:
            tool_name (str): Name of the tool
            tool_args (dict): Arguments that were duplicated

        Returns:
            str: Reminder message
        """
        args_str = f" with args {tool_args}" if tool_args else ""
        return f"⚠️ Duplicate tool call detected: {tool_name}{args_str}. Consider using different parameters or checking if this information is already available."

    def get_tool_limit_reminder(self, tool_name: str, current_count: int, limit: int) -> str:
        """
        Generate a reminder message for tool usage limits.

        Args:
            tool_name (str): Name of the tool
            current_count (int): Current usage count
            limit (int): Usage limit

        Returns:
            str: Reminder message
        """
        return f"⚠️ {tool_name} has been used {current_count} times (limit: {limit}). Consider using alternative tools or checking if this information is already available."

    def get_tool_call_stats(self, called_tools: list) -> Dict[str, Any]:
        """
        Get statistics about tool call history.

        Args:
            called_tools (list): List of called tools

        Returns:
            dict: Statistics about tool usage
        """
        if not called_tools:
            return {"total_calls": 0, "unique_tools": 0, "tool_breakdown": {}}

        tool_counts = {}
        for call in called_tools:
            tool_name = call.get('name', 'unknown')
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        return {
            "total_calls": len(called_tools),
            "unique_tools": len(tool_counts),
            "tool_breakdown": tool_counts
        }

    def clear_tool_history(self, called_tools: list) -> None:
        """
        Clear the tool call history.

        Args:
            called_tools (list): List of called tools to clear
        """
        called_tools.clear()
        print("🧹 Tool call history cleared")

    def _normalize_args(self, args: dict) -> str:
        """
        Normalize tool arguments for comparison.

        Args:
            args (dict): Tool arguments

        Returns:
            str: Normalized string representation
        """
        if isinstance(args, dict):
            return json.dumps(args, sort_keys=True)
        return str(args)

    def _get_timestamp(self) -> str:
        """
        Get current timestamp for tool call history.

        Returns:
            str: ISO format timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()


# Global instance for easy access
tool_call_manager = ToolCallManager()

# Convenience functions for backward compatibility
def is_duplicate_tool_call(tool_name: str, tool_args: dict, called_tools: list, threshold: float = 0.9) -> bool:
    """Check if a tool call is a duplicate."""
    return tool_call_manager.is_duplicate_tool_call(tool_name, tool_args, called_tools, threshold)

def add_tool_call_to_history(tool_name: str, tool_args: dict, called_tools: list) -> None:
    """Add a tool call to the history."""
    tool_call_manager.add_tool_call_to_history(tool_name, tool_args, called_tools)

def get_tool_call_stats(called_tools: list) -> Dict[str, Any]:
    """Get statistics about tool call history."""
    return tool_call_manager.get_tool_call_stats(called_tools)
