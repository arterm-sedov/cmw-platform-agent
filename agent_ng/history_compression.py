"""
History Compression Module
==========================

Provides semantic compression of conversation history to prevent context window overflow.
Compression replaces older messages with a concise summary while preserving key context.

Key Features:
- Semantic compression using LLM summarization
- System message exclusion
- Thread-safe compression stats tracking
- User-friendly notifications
- Comprehensive error handling
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .i18n_translations import get_translation_key
from .prompts import get_history_compression_prompt
from .token_budget import (
    HISTORY_COMPRESSION_MID_TURN_THRESHOLD,
    HISTORY_COMPRESSION_TARGET_TOKENS_PCT,
    TOKEN_STATUS_CRITICAL,
    TOKEN_STATUS_CRITICAL_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Thread-safe compression stats storage
_compression_stats: dict[str, dict[str, int]] = {}  # conversation_id -> {count: int, total_saved: int}
_stats_lock = Lock()


def filter_non_system_messages(history: list[BaseMessage]) -> list[BaseMessage]:
    """
    Filter out SystemMessage from history.

    Args:
        history: List of BaseMessage objects

    Returns:
        List of messages excluding SystemMessage
    """
    return [msg for msg in history if not isinstance(msg, SystemMessage)]


def format_messages_for_compression(messages: list[BaseMessage]) -> str:
    """
    Format BaseMessage objects to text for compression.

    Args:
        messages: List of BaseMessage objects to format

    Returns:
        Formatted string representation of messages
    """
    formatted_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            content = msg.content or ""
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_str = ", ".join(
                    [f"{tc.get('name', 'unknown')}()" for tc in msg.tool_calls]
                )
                formatted_parts.append(f"Assistant: {content} [Tool calls: {tool_calls_str}]")
            else:
                formatted_parts.append(f"Assistant: {content}")
        elif isinstance(msg, ToolMessage):
            tool_result = str(msg.content)[:500]  # Truncate long tool results
            formatted_parts.append(f"Tool ({msg.name or 'unknown'}): {tool_result}")
    return "\n\n".join(formatted_parts)


async def compress_history_to_tokens(
    history: list[BaseMessage],
    target_tokens: int,
    llm_instance: Any,
) -> str | None:
    """
    Compress conversation history to target token count using LLM summarization.

    Args:
        history: List of BaseMessage objects to compress
        target_tokens: Target token count for compressed summary
        llm_instance: LLMInstance object for compression

    Returns:
        Compressed summary string or None on failure
    """
    try:
        # Filter out SystemMessage
        history_to_compress = filter_non_system_messages(history)

        if not history_to_compress:
            logger.warning("No messages to compress after filtering SystemMessage")
            return None

        # Calculate target_words using Russian tokenization heuristic
        # Russian text uses ~2 tokens per word vs English ~1.33 tokens per word
        target_words = int(target_tokens * 0.5)  # Conservative estimate

        # Format messages for compression
        formatted_history = format_messages_for_compression(history_to_compress)

        # Get compression prompt with system context injection
        prompt = get_history_compression_prompt(
            target_tokens=target_tokens, target_words=target_words
        )

        # Create messages for LLM
        compression_messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Conversation history to compress:\n\n{formatted_history}"),
        ]

        # Call LLM for compression (no tools needed)
        if not llm_instance or not hasattr(llm_instance, "llm"):
            logger.error("Invalid LLM instance for compression")
            return None

        llm = llm_instance.llm
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(compression_messages)
        else:
            # Fallback to sync invoke if async not available
            response = llm.invoke(compression_messages)

        # Extract content from response
        if hasattr(response, "content"):
            compressed_summary = response.content
        elif isinstance(response, str):
            compressed_summary = response
        else:
            logger.error("Unexpected response type from compression LLM: %s", type(response))
            return None

        # Validate compressed result
        if not compressed_summary or not compressed_summary.strip():
            logger.warning("Compression returned empty string")
            return None

        return compressed_summary.strip()

    except Exception as e:
        logger.exception("Error during history compression: %s", e)
        return None


def _is_compression_enabled(agent: Any) -> bool:
    """Check if history compression is enabled for the given agent.

    Defaults to True if the flag is missing to preserve existing behavior.
    """
    try:
        return bool(agent.compression_enabled)
    except AttributeError:
        return True


def should_compress_on_completion(
    agent: Any, conversation_id: str, status: str | None
) -> bool:
    """
    Check if compression is needed after turn completion.

    Args:
        agent: Agent instance
        conversation_id: Conversation ID
        status: Token status string (e.g., "critical", "warning")

    Returns:
        True if compression should be triggered
    """
    try:
        # Allow per-agent toggle to disable compression
        if not _is_compression_enabled(agent):
            return False
        # Check if status is Critical
        if status != TOKEN_STATUS_CRITICAL:
            return False

        # Verify minimum history length
        if not hasattr(agent, "memory_manager") or not agent.memory_manager:
            return False

        history = agent.memory_manager.get_conversation_history(conversation_id)
        if not history:
            return False

        # Filter out SystemMessage for length check
        non_system_history = filter_non_system_messages(history)
        if len(non_system_history) < 2:  # Need at least 2 messages to compress
            return False

        return True

    except Exception as e:
        logger.exception("Error checking compression trigger on completion: %s", e)
        return False


def should_compress_mid_turn(
    agent: Any, conversation_id: str, messages_override: list[BaseMessage] | None = None
) -> bool:
    """
    Check if proactive compression is needed mid-turn.

    Args:
        agent: Agent instance
        conversation_id: Conversation ID
        messages_override: Optional messages to use for snapshot calculation

    Returns:
        True if compression should be triggered
    """
    try:
        # Allow per-agent toggle to disable compression
        if not _is_compression_enabled(agent):
            return False
        if not hasattr(agent, "token_tracker") or not agent.token_tracker:
            return False

        # Refresh budget snapshot with projected tokens
        snapshot = agent.token_tracker.refresh_budget_snapshot(
            agent=agent,
            conversation_id=conversation_id,
            messages_override=messages_override,
        )

        if not snapshot:
            return False

        percentage_used = snapshot.get("percentage_used", 0.0)
        threshold = HISTORY_COMPRESSION_MID_TURN_THRESHOLD

        if percentage_used >= threshold:
            logger.info(
                "Mid-turn compression triggered: %.1f%% >= %.1f%%",
                percentage_used,
                threshold,
            )
            return True

        return False

    except Exception as e:
        logger.exception("Error checking compression trigger mid-turn: %s", e)
        return False


def track_compression_stats(conversation_id: str, tokens_saved: int) -> tuple[int, int]:
    """
    Track compression statistics for a conversation.

    Args:
        conversation_id: Conversation ID
        tokens_saved: Number of tokens saved in this compression

    Returns:
        Tuple of (compression_count, total_tokens_saved)
    """
    with _stats_lock:
        if conversation_id not in _compression_stats:
            _compression_stats[conversation_id] = {"count": 0, "total_saved": 0}
        _compression_stats[conversation_id]["count"] += 1
        _compression_stats[conversation_id]["total_saved"] += tokens_saved
        return (
            _compression_stats[conversation_id]["count"],
            _compression_stats[conversation_id]["total_saved"],
        )


def get_compression_stats(conversation_id: str) -> tuple[int, int]:
    """
    Get compression statistics for a conversation.

    Args:
        conversation_id: Conversation ID

    Returns:
        Tuple of (compression_count, total_tokens_saved) or (0, 0) if no stats exist
    """
    with _stats_lock:
        stats = _compression_stats.get(conversation_id, {"count": 0, "total_saved": 0})
        return (stats["count"], stats["total_saved"])


def clear_compression_stats(conversation_id: str) -> None:
    """
    Clear compression statistics for a conversation.

    Args:
        conversation_id: Conversation ID
    """
    with _stats_lock:
        if conversation_id in _compression_stats:
            del _compression_stats[conversation_id]
            logger.debug("Cleared compression stats for conversation %s", conversation_id)


async def compress_conversation_history(
    agent: Any,
    conversation_id: str,
    keep_recent_turns: int,
    target_tokens: int,
    reason: str,
) -> tuple[bool, int]:
    """
    Compress conversation history, keeping recent turns uncompressed.

    Args:
        agent: Agent instance
        conversation_id: Conversation ID
        keep_recent_turns: Number of recent turns to keep uncompressed
        target_tokens: Target token count for compressed summary
        reason: Reason for compression (for logging)

    Returns:
        Tuple of (success: bool, tokens_saved: int)
    """
    try:
        # Get history from memory manager
        if not hasattr(agent, "memory_manager") or not agent.memory_manager:
            logger.warning("Memory manager unavailable for compression")
            return False, 0

        history = agent.memory_manager.get_conversation_history(conversation_id)
        if not history:
            logger.warning("No history found for compression")
            return False, 0

        # Filter out SystemMessage
        non_system_history = filter_non_system_messages(history)

        # Edge case: keep_recent_turns exceeds total history length
        if keep_recent_turns >= len(non_system_history):
            logger.info(
                "keep_recent_turns (%d) >= history length (%d), skipping compression",
                keep_recent_turns,
                len(non_system_history),
            )
            return False, 0

        # Calculate tokens before compression (for savings calculation)
        try:
            from .token_budget import compute_context_tokens

            conv_tokens_before, tool_tokens_before = compute_context_tokens(non_system_history)
            tokens_before = conv_tokens_before + tool_tokens_before
        except Exception as e:
            logger.warning("Failed to calculate tokens before compression: %s", e)
            tokens_before = 0

        # Identify messages to compress (all except last N turns)
        if keep_recent_turns > 0:
            messages_to_compress = non_system_history[:-keep_recent_turns]
            recent_turns = non_system_history[-keep_recent_turns:]
        else:
            messages_to_compress = non_system_history
            recent_turns = []

        if not messages_to_compress:
            logger.info("No messages to compress after keeping recent turns")
            return False, 0

        # Get LLM instance for compression
        if not hasattr(agent, "llm_instance") or not agent.llm_instance:
            # Try to get LLM from manager
            if hasattr(agent, "llm_manager") and agent.llm_manager:
                # Get current provider/model
                if hasattr(agent, "llm_provider") and agent.llm_provider:
                    llm_instance = agent.llm_manager.get_llm(
                        agent.llm_provider, use_tools=False
                    )
                else:
                    # Fallback to default
                    llm_instance = agent.llm_manager.get_llm("openrouter", use_tools=False)
            else:
                logger.error("No LLM instance available for compression")
                return False, 0
        else:
            llm_instance = agent.llm_instance

        # Compress messages
        compressed_summary = await compress_history_to_tokens(
            messages_to_compress, target_tokens, llm_instance
        )

        if not compressed_summary:
            logger.warning("Compression failed or returned empty, keeping normal history")
            return False, 0

        # Create single AIMessage with compressed summary
        compressed_message = AIMessage(
            content=f"[Compressed conversation history: {compressed_summary}]"
        )

        # Calculate tokens after compression
        try:
            compressed_messages = [compressed_message] + recent_turns
            conv_tokens_after, tool_tokens_after = compute_context_tokens(compressed_messages)
            tokens_after = conv_tokens_after + tool_tokens_after
            tokens_saved = max(0, tokens_before - tokens_after)
        except Exception as e:
            logger.warning("Failed to calculate tokens after compression: %s", e)
            tokens_saved = 0

        # Replace messages in memory manager
        # Get full history (including SystemMessage) to preserve it
        full_history = history
        system_messages = [msg for msg in full_history if isinstance(msg, SystemMessage)]

        # Build new history: system messages + compressed message + recent turns
        new_history = system_messages + [compressed_message] + recent_turns

        # Replace history in memory manager
        memory = agent.memory_manager.get_memory(conversation_id)
        if memory and hasattr(memory, "chat_memory"):
            memory.chat_memory.chat_memory = new_history
            logger.info(
                "Compressed %d messages to 1, keeping %d recent turns. Tokens saved: ~%d",
                len(messages_to_compress),
                len(recent_turns),
                tokens_saved,
            )
        else:
            logger.error("Failed to update memory manager with compressed history")
            return False, 0

        # Track compression stats
        track_compression_stats(conversation_id, tokens_saved)

        return True, tokens_saved

    except Exception as e:
        logger.exception("Error during conversation history compression: %s", e)
        return False, 0


def emit_compression_notification(
    tokens_saved: int,
    previous_pct: float,
    current_pct: float,
    reason: str,
    language: str = "en",
    compression_count: int | None = None,
    is_before: bool = False,
) -> None:
    """
    Emit Gradio popup notification for compression.

    Args:
        tokens_saved: Number of tokens saved (0 for before compression)
        previous_pct: Previous token usage percentage
        current_pct: Current token usage percentage
        reason: Reason for compression (translation key)
        language: Language for localization
        compression_count: Current compression count (optional)
        is_before: True if this is before compression, False if after
    """
    try:
        # Get reason translation
        reason_key_map = {
            "critical": "history_compression_reason_critical",
            "proactive": "history_compression_reason_proactive",
            "interrupted": "history_compression_reason_interrupted",
        }
        reason_key = reason_key_map.get(reason, "history_compression_reason_critical")
        reason_template = get_translation_key(reason_key, language)
        # Format with threshold if the template contains {threshold}
        if "{threshold}" in reason_template:
            reason_text = reason_template.format(threshold=int(TOKEN_STATUS_CRITICAL_THRESHOLD))
        else:
            reason_text = reason_template

        # Get title
        title = get_translation_key("history_compression_title", language)

        if is_before:
            # Before compression notification
            info_key = "history_compression_info_before"
            message_template = get_translation_key(info_key, language)
            message = message_template.format(
                previous_pct=previous_pct,
                reason=reason_text,
            )
        else:
            # After compression notification with results
            info_key = "history_compression_info"
            message_template = get_translation_key(info_key, language)
            count = compression_count or 0
            message = message_template.format(
                tokens_saved=tokens_saved,
                previous_pct=previous_pct,
                current_pct=current_pct,
                reason=reason_text,
                compression_count=count,
            )

        # Emit notification
        gr.Info(f"{title}\n\n{message}")

    except Exception as e:
        logger.exception("Error emitting compression notification: %s", e)


async def perform_compression_with_notifications(
    agent: Any,
    conversation_id: str,
    language: str,
    keep_recent_turns: int,
    reason: str,
    budget_snapshot: dict[str, Any] | None = None,
    rebuild_messages: bool = False,
) -> tuple[bool, list[BaseMessage] | None]:
    """
    Perform compression with before/after notifications.

    This is a reusable helper function that handles the complete compression flow:
    - Gets budget snapshot if not provided
    - Emits "before" compression notification
    - Calculates target tokens
    - Performs compression
    - Optionally rebuilds messages from memory
    - Refreshes token snapshot
    - Emits "after" compression notification with results

    Args:
        agent: The LangChain agent instance
        conversation_id: Conversation identifier
        language: Language for notifications
        keep_recent_turns: Number of recent turns to keep uncompressed
        reason: Compression reason ("proactive", "critical", "interrupted")
        budget_snapshot: Optional budget snapshot (fetched if not provided)
        rebuild_messages: If True, rebuild messages from memory after compression

    Returns:
        Tuple of (success: bool, updated_messages: list[BaseMessage] | None)
    """
    try:
        # Get budget snapshot if not provided
        if budget_snapshot is None:
            if hasattr(agent, "token_tracker") and agent.token_tracker:
                budget_snapshot = agent.token_tracker.get_budget_snapshot()
            else:
                budget_snapshot = {}
        previous_pct = budget_snapshot.get("percentage_used", 0.0) if budget_snapshot else 0.0

        # Emit compression popup before compression
        compression_stats = get_compression_stats(conversation_id)
        emit_compression_notification(
            tokens_saved=0,
            previous_pct=previous_pct,
            current_pct=previous_pct,
            reason=reason,
            language=language,
            compression_count=compression_stats[0],
            is_before=True,
        )

        # Calculate target tokens (10% of context window)
        context_window = 0
        if (
            hasattr(agent, "llm_instance")
            and agent.llm_instance
            and hasattr(agent.llm_instance, "config")
        ):
            context_window = agent.llm_instance.config.get("token_limit", 0)
        if context_window == 0 and budget_snapshot:
            context_window = budget_snapshot.get("context_window", 0)
        target_tokens = (
            int(context_window * HISTORY_COMPRESSION_TARGET_TOKENS_PCT)
            if context_window > 0
            else 1000
        )

        # Compress the context
        success, tokens_saved = await compress_conversation_history(
            agent,
            conversation_id,
            keep_recent_turns=keep_recent_turns,
            target_tokens=target_tokens,
            reason=reason,
        )

        # Handle post-compression actions
        updated_messages = None
        if success:
            # Rebuild messages from compressed memory if requested
            if rebuild_messages:
                updated_history = agent.memory_manager.get_conversation_history(
                    conversation_id
                )
                # Rebuild messages list: system message + compressed history
                updated_messages = []
                if hasattr(agent, "system_prompt") and agent.system_prompt:
                    updated_messages.append(SystemMessage(content=agent.system_prompt))
                # Filter out SystemMessage from history (it's added separately)
                non_system_history = [
                    msg
                    for msg in updated_history
                    if not isinstance(msg, SystemMessage)
                ]
                updated_messages.extend(non_system_history)

            # Refresh snapshot to reflect new token counts
            refresh_kwargs = {
                "agent": agent,
                "conversation_id": conversation_id,
            }
            if updated_messages is not None:
                refresh_kwargs["messages_override"] = updated_messages
            new_snapshot = agent.token_tracker.refresh_budget_snapshot(**refresh_kwargs)

            # Get updated compression stats
            updated_stats = get_compression_stats(conversation_id)
            # Emit results popup with actual compression results
            emit_compression_notification(
                tokens_saved=tokens_saved,
                previous_pct=previous_pct,
                current_pct=new_snapshot.get("percentage_used", 0.0)
                if new_snapshot
                else previous_pct,
                reason=reason,
                language=language,
                compression_count=updated_stats[0],
                is_before=False,
            )

        return success, updated_messages

    except Exception as exc:
        # Non-fatal: log and return failure
        logger.exception("Failed to perform compression with notifications: %s", exc)
        return False, None
