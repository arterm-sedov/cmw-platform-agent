"""
Conversation Summary Module
==========================

Provides session-agent aware conversation summary functionality.
"""

from datetime import datetime
import logging
from typing import Any

import gradio as gr

from .i18n_translations import format_translation


def generate_detailed_conversation_summary(
    session_manager: Any,  # SessionManager type
    request: gr.Request | None = None,
    session_id: str | None = None,
    language: str = "en",
) -> str:
    """
    Generate a detailed conversation summary for a specific session.

    Args:
        session_manager: The session manager instance
        request: Gradio request object (used to determine session ID if session_id not provided)
        session_id: Explicit session ID (takes precedence over request-derived ID)
        language: Language for localization (default: "en")

    Returns:
        str: Formatted detailed conversation summary or error message
    """
    logger = logging.getLogger(__name__)

    try:
        # Determine session ID
        if session_id is None and request:
            session_id = session_manager.get_session_id(request)
        elif session_id is None:
            return format_translation("logs_not_available", language) + "\n"

        # Get session-specific agent
        agent = session_manager.get_session_agent(session_id)
        if not agent:
            return (
                f"--- {format_translation('conversation_summary', language)} ---\n"
                + f"{format_translation('agent_not_available', language)}\n\n"
            )

        # Get agent statistics
        stats = agent.get_stats()
        conversation_stats = stats.get("conversation_stats", {})
        llm_info = stats.get("llm_info", {})

        # Check if there's meaningful conversation data
        message_count = conversation_stats.get("message_count", 0)
        if message_count <= 0:
            return (
                f"--- {format_translation('conversation_summary', language)} ---\n"
                + f"{format_translation('total_messages_label', language)}: 0\n\n"
            )

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract conversation metrics
        user_messages = conversation_stats.get("user_messages", 0)
        assistant_messages = conversation_stats.get("assistant_messages", 0)

        # Fix message count discrepancy - use sum of user+assistant if higher
        calculated_total = user_messages + assistant_messages
        if calculated_total > message_count:
            message_count = calculated_total

        # Get model usage info
        provider = llm_info.get("provider", "unknown")
        model = llm_info.get("model_name", "unknown")

        # Get tool usage from stats manager
        stats_manager_stats = stats.get("stats_manager_stats", {})

        # Check different possible locations for tool call data
        tool_calls_data = (
            stats_manager_stats.get("tool_calls", {})
            or stats.get("tool_calls", {})
            or conversation_stats.get("tool_calls", {})
        )

        # Check for model usage data
        model_usage_data = (
            stats_manager_stats.get("model_usage", {})
            or stats.get("model_usage", {})
            or conversation_stats.get("model_usage", {})
        )

        # Get additional stats
        core_stats = stats.get("core_agent_stats", {})
        memory_stats = stats.get("memory_manager_stats", {})

        # Build enhanced summary
        summary = f"--- {format_translation('conversation_summary', language)} ({timestamp}) ---\n"

        # We'll add the consolidated message/memory info later after gathering tool counts

        # Models used - check both stats and existing model_usage data
        actual_model_usage = {}

        # First try to get from stats_manager_stats if available
        model_usage_data = stats.get("stats_manager_stats", {}).get("model_usage", {})
        if model_usage_data:
            actual_model_usage.update(model_usage_data)

        # Also check conversation histories in stats manager
        if hasattr(agent, "stats_manager") and agent.stats_manager:
            try:
                session_conv_key = f"{session_id or 'default'}_"
                for (
                    conv_key,
                    conversations,
                ) in agent.stats_manager.conversation_histories.items():
                    if conv_key.startswith(session_conv_key):
                        for conv in conversations:
                            llm_used = conv.get("llm_used")
                            if llm_used:
                                actual_model_usage[llm_used] = (
                                    actual_model_usage.get(llm_used, 0) + 1
                                )
            except:
                pass

        # Fallback: use current model if no tracking data found
        if not actual_model_usage:
            current_model = f"{provider}/{model}"
            actual_model_usage[current_model] = 1

        # Use actual model usage if found, otherwise fall back to stats data or current model
        if actual_model_usage:
            model_summary = []
            unique_models_count = len(
                actual_model_usage
            )  # Count unique models, not total calls
            for model_key in actual_model_usage.keys():
                # Show each model as (1) since we only care about unique usage
                model_summary.append(f"{model_key}")

            model_list = ", ".join(model_summary)  # Show all models
            summary += f"{format_translation('model_label', language)} ({unique_models_count}): {model_list}\n"
        elif model_usage_data and any(
            isinstance(calls, (int, float)) and calls > 0
            for calls in model_usage_data.values()
        ):
            # Extract unique models from stats data
            model_summary = []
            unique_models_count = 0
            for model_key, calls in model_usage_data.items():
                if isinstance(calls, (int, float)) and calls > 0:
                    model_summary.append(
                        f"{model_key} (1)"
                    )  # Show as (1) for unique usage
                    unique_models_count += 1

            if model_summary:
                model_list = ", ".join(model_summary)  # Show all models
                summary += f"{format_translation('model_label', language)} ({unique_models_count}): {model_list}\n"
            else:
                summary += f"{format_translation('model_label', language)}: {provider}/{model}\n"
        else:
            # Show current model without usage count since tracking isn't available
            summary += (
                f"{format_translation('model_label', language)}: {provider}/{model}\n"
            )

        # Tools with call counts - check multiple sources
        actual_tool_calls = {}

        # First try to get from stats_manager_stats if available
        tool_usage_data = stats.get("stats_manager_stats", {}).get("tool_calls", {})
        if tool_usage_data:
            actual_tool_calls.update(tool_usage_data)

        # Get tools from memory manager directly
        if hasattr(agent, "memory_manager") and agent.memory_manager:
            try:
                tool_calls = agent.memory_manager.get_tool_calls(
                    session_id or "default"
                )
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "unknown_tool")
                    actual_tool_calls[tool_name] = (
                        actual_tool_calls.get(tool_name, 0) + 1
                    )
            except:
                pass

        # Also count ToolMessage objects in conversation history
        if hasattr(agent, "memory_manager") and agent.memory_manager:
            try:
                conversation_history = agent.memory_manager.get_conversation_history(
                    session_id or "default"
                )
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

        # If no tool usage found, skip tools section entirely instead of showing useless count
        if actual_tool_calls:
            tool_summary = []
            total_tool_calls = 0
            for tool_name, calls in actual_tool_calls.items():
                call_count = int(calls)
                tool_summary.append(f"{tool_name} ({call_count})")
                total_tool_calls += call_count

            tool_list = ", ".join(tool_summary)  # Show all tools
            summary += f"{format_translation('tools_label', language)} ({total_tool_calls}): {tool_list}\n"
        # Skip tools section entirely if no usage data (instead of showing useless "available" count)

        # Consolidated memory info with breakdown
        memory_count = memory_stats.get("total_memories", 0)

        # Try to get actual conversation message count from memory
        actual_memory_count = memory_count
        if hasattr(agent, "memory_manager") and agent.memory_manager:
            try:
                conversation_history = agent.memory_manager.get_conversation_history(
                    session_id or "default"
                )
                # Count all messages in conversation history
                actual_memory_count = (
                    len(conversation_history) if conversation_history else 0
                )
            except:
                pass  # Fall back to stats count

        # Calculate tool message count (use total_tool_calls if available from tools section)
        total_tool_messages = 0
        if "total_tool_calls" in locals():
            total_tool_messages = total_tool_calls
        elif actual_tool_calls:
            total_tool_messages = sum(actual_tool_calls.values())

        # Consolidated format: Memory entries: X (Y user, Z assistant, W tools)
        summary += f"{format_translation('memory_entries', language, count=actual_memory_count)} "
        summary += f"({user_messages} {format_translation('user_messages_label', language).lower()}, "
        summary += f"{assistant_messages} {format_translation('assistant_messages_label', language).lower()}"

        if total_tool_messages > 0:
            summary += f", {total_tool_messages} {format_translation('tools_label', language).lower()}"

        summary += ")\n\n"

        return summary

    except Exception as e:
        logger.exception("Error generating conversation summary: %s", e)
        return f"--- {format_translation('error_loading_stats', language)}: {e} ---\n"
