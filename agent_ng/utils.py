"""
Utility functions for the next-generation agent.

This module contains utility functions that were previously imported from agent_old
but are now part of the next-gen agent architecture.
"""

from typing import Any

# Import from centralized provider adapters
try:
    from .provider_adapters import convert_messages_for_mistral
except ImportError:
    # Fallback for backward compatibility
    convert_messages_for_mistral = None


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