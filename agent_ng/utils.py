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


def format_cost(cost: float | None, max_decimals: int = 3) -> str:
    """Format cost with limited decimal places and rounding.

    Formats cost values intelligently:
    - None → "—" (unknown)
    - 0.0 → "$0.0000" (free)
    - > 0.0 → Rounded to 2-3 decimal places (e.g., "$0.042", "$0.043", "$1.23")

    Examples:
        $0.04211483 → $0.042
        $0.0429 → $0.043
        $1.234567 → $1.235

    Args:
        cost: Cost value (None = unknown, 0.0 = free)
        max_decimals: Maximum decimal places to use (default: 3)

    Returns:
        Formatted cost string
    """
    if cost is None:
        return "—"
    if cost == 0.0:
        return "$0.0000"

    # Round to max_decimals places
    rounded = round(cost, max_decimals)

    # For very small values that round to zero, show original with limited precision
    if rounded == 0.0 and cost > 0.0:
        # Show up to 4 decimal places for very small values
        formatted = f"${cost:.4f}"
        return formatted.rstrip('0').rstrip('.')

    # Format with max_decimals places, then remove trailing zeros
    formatted = f"${rounded:.{max_decimals}f}"
    cleaned = formatted.rstrip('0').rstrip('.')

    # Ensure we show at least one decimal place for values < 1.0
    if cost < 1.0 and '.' not in cleaned:
        return f"${rounded:.{max_decimals}f}".rstrip('0').rstrip('.')

    return cleaned


def parse_env_bool(env_name: str) -> bool:
    """Parse a boolean feature flag from environment variables in a single place.

    Accepted truthy values (case-insensitive, trimmed):
    - "1", "true", "yes", "on"
    """
    import os

    env_val = os.getenv(env_name, "")
    if not env_val:
        return False
    return env_val.strip().lower() in ("1", "true", "yes", "on")


def get_tool_call_count(agent, session_id: str) -> int:
    """
    Get total tool call count from session-specific agent - shared implementation.

    This function uses direct access to memory manager to avoid circular dependencies.
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
        actual_tool_calls = {}

        # Get tools from memory manager directly (avoid calling get_stats to prevent recursion)
        if hasattr(agent, "memory_manager") and agent.memory_manager:
            try:
                tool_calls = agent.memory_manager.get_tool_calls(session_id)
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "unknown_tool")
                    actual_tool_calls[tool_name] = (
                        actual_tool_calls.get(tool_name, 0) + 1
                    )
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(
                    "Failed to count tool calls from memory manager: %s", e
                )

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
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(
                    "Failed to count tool calls from conversation history: %s", e
                )

        # Sum up all tool calls
        if actual_tool_calls:
            total_tool_calls = 0
            for tool_name, calls in actual_tool_calls.items():
                call_count = int(calls)
                total_tool_calls += call_count
            return total_tool_calls

    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(
            "Failed to count tool calls: %s", e
        )

    return 0