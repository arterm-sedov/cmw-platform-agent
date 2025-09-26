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
        provider = llm_info.get("provider", "unknown")
        model = llm_info.get("model_name", "unknown")

        # Get additional stats
        core_stats = stats.get("core_agent_stats", {})
        memory_stats = stats.get("memory_manager_stats", {})

        # Build summary using i18n translations
        summary = f"--- {format_translation('conversation_summary', language)} ({timestamp}) ---\n"
        summary += (
            f"{format_translation('total_messages_label', language)}: {message_count} "
        )
        summary += f"({user_messages} {format_translation('user_messages_label', language).lower()}, "
        summary += f"{assistant_messages} {format_translation('assistant_messages_label', language).lower()})\n"
        summary += f"{format_translation('provider_model_label', language)}: {provider} / {model}\n"
        summary += f"{format_translation('tools_section', language).replace('**', '').replace(':', '')}: {core_stats.get('tools_count', 0)}\n"

        # Add memory info if available
        memory_count = memory_stats.get("total_memories", 0)
        if language == "ru":
            summary += f"Память: {memory_count} записей\n\n"
        else:
            summary += f"Memory: {memory_count} entries\n\n"

        return summary

    except Exception as e:
        logger.exception("Error generating conversation summary: %s", e)
        return f"--- {format_translation('error_loading_stats', language)}: {e} ---\n"
