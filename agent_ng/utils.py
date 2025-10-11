"""
Utility functions for the next-generation agent.

This module contains utility functions that were previously imported from agent_old
but are now part of the next-gen agent architecture.
"""

from typing import Any



def ensure_valid_answer(answer: Any) -> str:
    """
    Ensure the answer is a valid string. Only returns "No answer provided" 
    when there is literally no answer at all.
    
    Args:
        answer (Any): The answer to validate
        
    Returns:
        str: A valid string answer, preserving all error information
    """
    if answer is None:
        return "No answer provided"
    elif not isinstance(answer, str):
        return str(answer)
    elif answer.strip() == "":
        return "No answer provided"
    else:
        return answer


def safe_string(value: Any, default: str = "") -> str:
    """Convert value to string, handling None gracefully."""
    return default if value is None else str(value)


def get_tool_call_count(agent, session_id: str) -> int:
    """
    Get total tool call count from session-specific agent - shared implementation.
    
    This function uses the sophisticated logic from conversation_summary.py that actually works.
    Used across:
    - Sidebar status display
    - Stats tab display  
    
    Args:
        agent: The session-specific agent instance
        session_id: The session ID to get tool calls for (required, no default)
        
    Returns:
        int: Total number of tool calls made in this session
    """
    if not agent or not session_id:
        return 0
    
    try:
        # Use the sophisticated logic from conversation_summary.py
        stats = agent.get_stats()
        actual_tool_calls = {}

        # First try to get from stats_manager_stats if available
        tool_usage_data = stats.get("stats_manager_stats", {}).get("tool_calls", {})
        if tool_usage_data:
            actual_tool_calls.update(tool_usage_data)

        # Get tools from memory manager directly
        if hasattr(agent, "memory_manager") and agent.memory_manager:
            try:
                tool_calls = agent.memory_manager.get_tool_calls(session_id)
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "unknown_tool")
                    actual_tool_calls[tool_name] = (
                        actual_tool_calls.get(tool_name, 0) + 1
                    )
            except:
                pass

            # Also count ToolMessage objects in conversation history
            try:
                conversation_history = agent.memory_manager.get_conversation_history(session_id)
                for message in conversation_history:
                    if (
                        hasattr(message, "__class__")
                        and "Tool" in message.__class__.__name__
                    ):
                        tool_name = getattr(message, "name", "unknown_tool")
                        actual_tool_calls[tool_name] = (
                            actual_tool_calls.get(tool_name, 0) + 1
                        )
            except:
                pass

        # Sum up all tool calls
        if actual_tool_calls:
            total_tool_calls = 0
            for tool_name, calls in actual_tool_calls.items():
                call_count = int(calls)
                total_tool_calls += call_count
            return total_tool_calls
        
    except:
        pass
        
    return 0