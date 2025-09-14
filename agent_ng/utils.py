"""
Utility functions for the next-generation agent.

This module contains utility functions that were previously imported from agent_old
but are now part of the next-gen agent architecture.
"""

from typing import Any


def ensure_valid_answer(answer: Any) -> str:
    """
    Ensure the answer is a valid string, preserving meaningful error information.
    
    Args:
        answer (Any): The answer to validate
        
    Returns:
        str: A valid string answer, preserving error context when available
    """
    if answer is None:
        return "No answer provided"
    elif not isinstance(answer, str):
        return str(answer)
    elif answer.strip() == "":
        return "No answer provided"
    else:
        return answer


def ensure_meaningful_response(response: Any, context: str = "", tool_calls: list = None) -> str:
    """
    Ensure a meaningful response that preserves error information and context.
    
    Args:
        response (Any): The response to validate
        context (str): Additional context about what was being processed
        tool_calls (list): List of tool calls that were made
        
    Returns:
        str: A meaningful response that preserves error information
    """
    if response is None:
        if tool_calls:
            tool_names = [call.get('name', 'unknown') for call in tool_calls]
            return f"Tools executed successfully ({', '.join(tool_names)}) but no response generated. This may indicate a context length limit or processing error."
        return "No response generated"
    elif not isinstance(response, str):
        return str(response)
    elif response.strip() == "":
        if tool_calls:
            tool_names = [call.get('name', 'unknown') for call in tool_calls]
            return f"Tools executed successfully ({', '.join(tool_names)}) but response is empty. This may indicate a context length limit or processing error."
        return "Empty response generated"
    else:
        return response
