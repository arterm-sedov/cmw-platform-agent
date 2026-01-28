"""
Native LangChain Streaming Implementation
========================================

This module implements proper LangChain native streaming using astream()
and astream_events() for real-time token-by-token streaming.

Key Features:
- Native LangChain streaming (astream, astream_events)
- Real-time token-by-token streaming
- Proper tool calling with streaming
- No artificial delays
- Uses LangChain's built-in streaming capabilities
- LangSmith tracing at the LLM call level
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
import json
import logging
import os
import sys
from typing import Any

# LangChain imports
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from .debug_streamer import get_debug_streamer
from .history_compression import (
    compress_conversation_history,
    emit_compression_notification,
    get_compression_stats,
    perform_compression_with_notifications,
    should_compress_mid_turn,
    should_compress_on_completion,
)
from .i18n_translations import get_translation_key
from .streaming_config import get_streaming_config
from .tool_deduplicator import get_deduplicator
from .token_budget import (
    HISTORY_COMPRESSION_KEEP_RECENT_TURNS_MID_TURN,
    HISTORY_COMPRESSION_KEEP_RECENT_TURNS_SUCCESS,
    HISTORY_COMPRESSION_MID_TURN_THRESHOLD,
    HISTORY_COMPRESSION_TARGET_TOKENS_PCT,
    MAX_TOOL_RESULT_TOKENS_PCT,
    count_tokens,
)

# LangSmith tracing
try:
    from langsmith import traceable

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

    def traceable(func):
        return func


# Optional Langfuse integration
try:
    # Local, soft dependency wrapper
    from .langfuse_config import get_langfuse_callback_handler

    _LANGFUSE_AVAILABLE = True
except Exception:
    _LANGFUSE_AVAILABLE = False


@dataclass
class StreamingEvent:
    """Represents a streaming event"""

    event_type: str
    content: str
    metadata: dict[str, Any] = None


class NativeLangChainStreaming:
    """
    Native LangChain streaming implementation using astream() and astream_events().
    """

    # Get UI icons from i18n translations
    def _get_ui_icons(self, language: str = "en"):
        """Get UI icons from i18n translations"""

        return {
            "finish_icons": get_translation_key("finish_icons", language),
            "completion_icons": get_translation_key("completion_icons", language),
            "max_icons": get_translation_key("max_icons", language),
            "completion_final_icons": get_translation_key(
                "completion_final_icons", language
            ),
            "error_icons": get_translation_key("error_icons", language),
        }

    def __init__(self):
        # Module logger
        self._logger = logging.getLogger(__name__)
        # Get configuration from centralized config
        self.config = get_streaming_config()
        self.max_iterations = self.config.get_max_tool_call_iterations()

    def _get_response_completed_message(self, language: str = "en") -> str:
        """Get the localized response completed message with completion icon"""

        icons = self._get_ui_icons(language)
        completion_icon = icons["completion_final_icons"][0]
        message = get_translation_key("response_completed", language)
        return f"{completion_icon} {message}"

    def _get_processing_failed_message(self, language: str = "en") -> str:
        """Get the localized processing failed message with error icon"""

        icons = self._get_ui_icons(language)
        error_icon = icons["error_icons"][0]  # Always use first icon for error
        message = get_translation_key("processing_failed", language)
        return f"{error_icon} **{message}**"

    def _get_iteration_processing_message(
        self, iteration: int, max_iterations: int, language: str = "en"
    ) -> str:
        """Get the localized iteration processing message"""

        template = get_translation_key("iteration_processing", language)
        return template.format(iteration=iteration, max_iterations=max_iterations)

    def _get_iteration_finished_message(
        self, iteration: int, max_iterations: int, language: str = "en"
    ) -> str:
        """Get the localized iteration finished message"""

        template = get_translation_key("iteration_finished", language)
        return template.format(iteration=iteration, max_iterations=max_iterations)

    def _get_iteration_completed_message(
        self, iteration: int, language: str = "en"
    ) -> str:
        """Get the localized iteration completed message with completion icon"""

        icons = self._get_ui_icons(language)
        completion_icon = icons["completion_icons"][
            iteration % len(icons["completion_icons"])
        ]
        template = get_translation_key("iteration_completed", language)
        message = template.format(iteration=iteration)
        return f"{completion_icon} **{message}**"

    def _get_iteration_max_reached_message(
        self, iteration: int, max_iterations: int, language: str = "en"
    ) -> str:
        """Get the localized iteration max reached message"""

        template = get_translation_key("iteration_max_reached", language)
        return template.format(iteration=iteration, max_iterations=max_iterations)

    def _get_max_iterations_warning_message(
        self, max_iterations: int, language: str = "en"
    ) -> str:
        """Get the localized max iterations warning message"""

        template = get_translation_key("max_iterations_warning", language)
        return template.format(max_iterations=max_iterations)

    def _get_tool_called_message(self, tool_name: str, language: str = "en") -> str:
        """Get the localized tool called message"""

        # Safety check for None language
        if language is None:
            language = "en"
        template = get_translation_key("tool_called", language)
        return template.format(tool_name=tool_name)

    def _get_call_count_message(self, total_calls: int, language: str = "en") -> str:
        """Get the localized call count message"""

        # Safety check for None language
        if language is None:
            language = "en"
        template = get_translation_key("call_count", language)
        return template.format(total_calls=total_calls)

    def _get_result_message(self, tool_result: str, language: str = "en") -> str:
        """Get the localized result message with chat display truncation"""

        # Safety check for None language
        if language is None:
            language = "en"
        template = get_translation_key("result", language)

        # Lean truncation for chat display only (200 chars max)
        display_result = tool_result
        if len(tool_result) > 400:
            display_result = tool_result[:400] + "... [truncated]"

        return template.format(tool_result=display_result)

    def _deduplicate_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """
        Deduplicate tool calls and count duplicates BEFORE execution.

        Args:
            tool_calls: List of tool calls from LLM response

        Returns:
            Tuple of (deduplicated_tool_calls, duplicate_counts)
        """
        unique_tool_calls = []
        duplicate_counts = {}

        for tool_call in tool_calls:
            if not tool_call:
                continue
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("args", {})
            try:
                tool_key = f"{tool_name}:{hash(json.dumps(tool_args, sort_keys=True, default=str))}"
            except Exception:
                tool_key = f"{tool_name}:{hash(str(tool_args))}"

            if tool_key in duplicate_counts:
                # Increment count for duplicate
                duplicate_counts[tool_key] += 1
                print(
                    f"🔍 DEBUG: Found duplicate tool call {tool_name} (total count: {duplicate_counts[tool_key]})"
                )
            else:
                # First occurrence - add to unique list and initialize count
                unique_tool_calls.append(tool_call)
                duplicate_counts[tool_key] = 1
                self._logger.debug("Added unique tool call %s", tool_name)

        return unique_tool_calls, duplicate_counts

    def _get_tool_error_message(self, error: str, language: str = "en") -> str:
        """Get the localized tool error message"""

        template = get_translation_key("tool_error", language)
        return template.format(error=error)

    def _get_unknown_tool_message(self, tool_name: str, language: str = "en") -> str:
        """Get the localized unknown tool message"""

        template = get_translation_key("unknown_tool", language)
        return template.format(tool_name=tool_name)

    def _get_error_message(self, error: str, language: str = "en") -> str:
        """Get the localized error message with icon - let native error wording speak for itself"""
        return f"❌ {error}"

    def _get_agent_language(self, agent) -> str:
        """Get language from agent, defaulting to English"""
        if agent is None:
            return "en"
        return getattr(agent, "language", "en")

    def _select_fallback_model_for_agent(self, agent) -> tuple[str | None, int | None]:
        """Select a fallback model for the current agent provider.

        Preference order:
        1) Agent's explicit fallback_model_name (if valid and larger context)
        2) FALLBACK_MODEL_DEFAULT from environment (if valid and larger context)
        3) Largest context window model for the current provider.
        """
        if not hasattr(agent, "llm_manager") or not agent.llm_manager:
            return None, None
        if not hasattr(agent, "llm_instance") or not agent.llm_instance:
            return None, None

        provider_enum = agent.llm_instance.provider
        provider = provider_enum.value
        config = agent.llm_manager.get_provider_config(provider)
        if not config or not config.models:
            return None, None

        current_model = agent.llm_instance.model_name
        current_limit = 0
        for model_cfg in config.models:
            if model_cfg.get("model") == current_model:
                current_limit = int(model_cfg.get("token_limit", 0))
                break

        current_limit = max(current_limit, 0)

        candidates: list[tuple[str, int, int]] = []
        for idx, model_cfg in enumerate(config.models):
            model_name = model_cfg.get("model")
            token_limit = int(model_cfg.get("token_limit", 0))
            if not model_name:
                continue
            if token_limit > current_limit:
                candidates.append((model_name, token_limit, idx))

        # If no strictly larger models, fall back to the largest available model
        if not candidates:
            for idx, model_cfg in enumerate(config.models):
                model_name = model_cfg.get("model")
                token_limit = int(model_cfg.get("token_limit", 0))
                if not model_name:
                    continue
                candidates.append((model_name, token_limit, idx))

        if not candidates:
            return None, None

        # Sort by context window descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        available_names = {name for name, _limit, _idx in candidates}

        agent_pref = getattr(agent, "fallback_model_name", None)
        env_pref = os.getenv("FALLBACK_MODEL_DEFAULT", "").strip() or None

        def _resolve_preferred(name: str | None) -> tuple[str | None, int | None]:
            if not name or name not in available_names:
                return None, None
            for model_name, _limit, idx in candidates:
                if model_name == name:
                    return model_name, idx
            return None, None

        selected_name, selected_index = _resolve_preferred(agent_pref)
        if selected_name is None:
            selected_name, selected_index = _resolve_preferred(env_pref)
        if selected_name is None:
            selected_name, _limit, selected_index = candidates[0]

        # Avoid "fallback" to the same model
        if selected_name == current_model:
            return None, None

        return selected_name, selected_index

    def _try_switch_to_fallback_model(
        self,
        agent,
        conversation_id: str,
        trigger: str,
    ) -> bool:
        """Try to switch the agent to a larger-context fallback model.

        Returns True if a switch was performed.
        """
        try:
            if not getattr(agent, "use_fallback_model", False):
                return False

            selected_name, selected_index = self._select_fallback_model_for_agent(agent)
            if selected_name is None or selected_index is None:
                return False

            provider_enum = agent.llm_instance.provider
            provider = provider_enum.value

            new_instance = agent.llm_manager.create_new_llm_instance(
                provider, selected_index
            )
            if not new_instance:
                return False

            agent.llm_instance = new_instance

            # Reset token budget so future snapshots use the new context window
            if hasattr(agent, "token_tracker") and agent.token_tracker:
                try:
                    agent.token_tracker.reset_current_conversation_budget()
                except Exception as exc:
                    self._logger.debug(
                        "Failed to reset token budget after fallback switch: %s", exc
                    )

            # Persist chosen fallback model on the agent
            setattr(agent, "fallback_model_name", selected_name)

            self._logger.info(
                "Switched to fallback model %s/%s for conversation %s (trigger=%s)",
                provider,
                selected_name,
                conversation_id,
                trigger,
            )
            return True
        except Exception as exc:
            self._logger.debug("Failed to switch to fallback model: %s", exc)
            return False

    @traceable
    async def stream_agent_response(
        self, agent, message: str, conversation_id: str = "default"
    ) -> AsyncGenerator[StreamingEvent, None]:
        """
        Stream a complete QA turn using native LangChain streaming.

        This method handles the entire conversation turn including:
        - Memory retrieval
        - LLM streaming
        - Tool calling
        - Memory persistence

        Based on: https://python.langchain.com/docs/how_to/tool_streaming/

        Args:
            agent: The LangChain agent instance
            message: User message
            conversation_id: Conversation identifier

        Yields:
            StreamingEvent objects with real-time content
        """
        try:
            # Safety check for agent
            if not agent:
                raise ValueError("Agent is None")

            if not hasattr(agent, "llm_instance") or not agent.llm_instance:
                raise ValueError("Agent has no LLM instance")

            if not hasattr(agent, "memory_manager") or not agent.memory_manager:
                raise ValueError("Agent has no memory manager")

            # Get language from agent once at the beginning
            language = self._get_agent_language(agent)

            # Get conversation history
            chat_history = agent.memory_manager.get_conversation_history(
                conversation_id
            )
            print(
                f"🔍 DEBUG: Streaming manager - chat_history length: {len(chat_history) if chat_history else 0}"
            )

            # Create messages list for LLM context
            messages = []

            # Always add system message to LLM context (required for every call)
            system_message = SystemMessage(content=agent.system_prompt)
            messages.append(system_message)

            # Check if system message is already in memory, if not add it
            system_in_history = any(
                isinstance(msg, SystemMessage) for msg in chat_history
            )
            if not system_in_history:
                # Store system message in memory only once
                agent.memory_manager.add_message(conversation_id, system_message)
                self._logger.debug("Added system message to memory (first time)")
            else:
                self._logger.debug("System message already in memory, skipping storage")

            # Add conversation history (excluding system messages to avoid duplication)
            # Filter out orphaned tool messages to prevent message order issues
            non_system_history = []
            if chat_history:
                for i, msg in enumerate(chat_history):
                    # Safety check for None message
                    if msg is None:
                        self._logger.debug("Skipping None message at index %d", i)
                        continue

                    if isinstance(msg, SystemMessage):
                        continue
                    elif isinstance(msg, ToolMessage):
                        # Only include tool messages that have a corresponding AI message with tool calls
                        for j in range(i - 1, -1, -1):
                            # Safety check for None message in history
                            if (
                                j >= 0
                                and j < len(chat_history)
                                and chat_history[j] is None
                            ):
                                continue
                            if (
                                j >= 0
                                and j < len(chat_history)
                                and isinstance(chat_history[j], AIMessage)
                                and hasattr(chat_history[j], "tool_calls")
                                and chat_history[j].tool_calls
                            ):
                                # Safety check for None tool calls
                                tool_call_ids = {
                                    tc.get("id")
                                    for tc in chat_history[j].tool_calls
                                    if tc is not None and tc.get("id")
                                }
                                if (
                                    hasattr(msg, "tool_call_id")
                                    and msg.tool_call_id in tool_call_ids
                                ):
                                    non_system_history.append(msg)
                                    break
                    else:
                        non_system_history.append(msg)

            messages.extend(non_system_history)
            print(
                f"🔍 DEBUG: Added {len(non_system_history)} messages from history to LLM context"
            )

            # Create user message and save to memory
            user_message = HumanMessage(content=message)
            messages.append(user_message)
            agent.memory_manager.add_message(conversation_id, user_message)

            # Multi-turn conversation loop for proper tool calling
            iteration = 0

            # Begin per-QA-turn usage accumulation (iterations will add to this).
            try:
                if hasattr(agent, "token_tracker") and agent.token_tracker:
                    agent.token_tracker.begin_turn()
            except Exception as exc:
                # Log directly - if logging fails, let it fail visibly for debugging
                self._logger.debug("Failed to begin turn usage accumulation: %s", exc)

            # Check for overflow protection at the start of the turn (before first LLM call)
            # This prevents context overflow errors when history is already too large.
            try:
                if hasattr(agent, "token_tracker") and agent.token_tracker:
                    # Refresh budget snapshot with current messages
                    agent.token_tracker.refresh_budget_snapshot(
                        agent=agent,
                        conversation_id=conversation_id,
                        messages_override=messages,
                    )
                    snapshot = agent.token_tracker.get_budget_snapshot() or {}
                    percentage_used = snapshot.get("percentage_used", 0.0)
                    # Fallback first: switch to a larger model when enabled and threshold reached
                    if (
                        getattr(agent, "use_fallback_model", False)
                        and percentage_used >= HISTORY_COMPRESSION_MID_TURN_THRESHOLD
                    ):
                        switched = self._try_switch_to_fallback_model(
                            agent, conversation_id, trigger="start_of_turn"
                        )
                        if switched and hasattr(agent, "token_tracker") and agent.token_tracker:
                            agent.token_tracker.refresh_budget_snapshot(
                                agent=agent,
                                conversation_id=conversation_id,
                                messages_override=messages,
                            )
                    # After fallback (if any), decide whether compression is still needed
                    if should_compress_mid_turn(
                        agent, conversation_id, messages_override=messages
                    ):
                        budget_snapshot = agent.token_tracker.get_budget_snapshot()
                        success, updated_messages = await perform_compression_with_notifications(
                            agent=agent,
                            conversation_id=conversation_id,
                            language=language,
                            keep_recent_turns=HISTORY_COMPRESSION_KEEP_RECENT_TURNS_MID_TURN,
                            reason="proactive",
                            budget_snapshot=budget_snapshot,
                            rebuild_messages=True,
                        )
                        if success and updated_messages is not None:
                            messages = updated_messages
                            # Ensure user message is present after compression
                            # (it should be preserved as part of recent turns, but verify)
                            user_message_present = any(
                                isinstance(msg, HumanMessage)
                                and msg.content == user_message.content
                                for msg in messages
                            )
                            if not user_message_present:
                                messages.append(user_message)
                                self._logger.debug("Re-added user message after compression")
            except Exception as comp_exc:
                # Non-fatal: log and continue
                self._logger.debug(
                    "Failed to check/perform start-of-turn overflow protection: %s",
                    comp_exc,
                )

            while iteration < self.max_iterations:
                iteration += 1
                self._logger.debug("Starting iteration %d", iteration)

                # Budget moment: before each LLM call, compute an exact token budget snapshot
                # from the *actual* messages list used for the next model call.
                try:
                    if hasattr(agent, "token_tracker") and agent.token_tracker:
                        agent.token_tracker.refresh_budget_snapshot(
                            agent=agent,
                            conversation_id=conversation_id,
                            messages_override=messages,
                        )
                except Exception as exc:
                    # Keep this low-noise: budget snapshots are best-effort and must not
                    # impact the streaming loop.
                    # Log directly - if logging fails, let it fail visibly for debugging
                    self._logger.debug("Budget snapshot pre-iteration failed: %s", exc)

                # Mid-turn overflow protection: fallback first, then compression if still needed
                try:
                    snapshot = None
                    if hasattr(agent, "token_tracker") and agent.token_tracker:
                        snapshot = agent.token_tracker.get_budget_snapshot()
                    percentage_used = (
                        snapshot.get("percentage_used", 0.0) if snapshot else 0.0
                    )
                    if (
                        getattr(agent, "use_fallback_model", False)
                        and percentage_used >= HISTORY_COMPRESSION_MID_TURN_THRESHOLD
                    ):
                        switched = self._try_switch_to_fallback_model(
                            agent, conversation_id, trigger="mid_turn"
                        )
                        if switched and hasattr(agent, "token_tracker") and agent.token_tracker:
                            agent.token_tracker.refresh_budget_snapshot(
                                agent=agent,
                                conversation_id=conversation_id,
                                messages_override=messages,
                            )
                    if should_compress_mid_turn(
                        agent, conversation_id, messages_override=messages
                    ):
                        budget_snapshot = agent.token_tracker.get_budget_snapshot()
                        success, updated_messages = await perform_compression_with_notifications(
                            agent=agent,
                            conversation_id=conversation_id,
                            language=language,
                            keep_recent_turns=HISTORY_COMPRESSION_KEEP_RECENT_TURNS_MID_TURN,
                            reason="proactive",
                            budget_snapshot=budget_snapshot,
                            rebuild_messages=True,
                        )
                        if success and updated_messages is not None:
                            messages = updated_messages
                except Exception as comp_exc:
                    # Non-fatal: log and continue
                    self._logger.debug(
                        "Failed to check/perform mid-turn overflow protection: %s",
                        comp_exc,
                    )

                # Stream iteration progress as separate message (no icon - UI will add rotating clock)

                yield StreamingEvent(
                    event_type="iteration_progress",
                    content=self._get_iteration_processing_message(
                        iteration, self.max_iterations, language
                    ),
                    metadata={
                        "iteration": iteration,
                        "max_iterations": self.max_iterations,
                    },
                )

                # Use proper LangChain streaming with tool call handling
                accumulated_chunk = None
                last_chunk = None
                tool_calls_in_progress = {}
                processed_tools = {}  # Track processed tools to avoid duplicates in same response
                has_tool_calls = False
                self._logger.debug("Starting streaming loop for iteration %d", iteration)

                # Stream LLM response - tracing handled by @traceable on stream_agent_response
                # Attach optional Langfuse callback handler when available
                runnable_config = None
                if _LANGFUSE_AVAILABLE:
                    try:
                        handler = get_langfuse_callback_handler()
                        if handler is not None:
                            # Add Langfuse session id to metadata as per docs
                            # https://langfuse.com/docs/observability/features/sessions
                            session_id = getattr(agent, "session_id", None)
                            metadata = (
                                {"langfuse_session_id": session_id}
                                if session_id
                                else {}
                            )
                            runnable_config = {
                                "callbacks": [handler],
                                "metadata": metadata,
                            }
                    except Exception as exc:
                        self._logger.debug(
                            "Failed to get runnable config: %s", exc
                        )
                        runnable_config = None

                # Get LLM with tools fresh for this iteration to reflect any fallback switch
                llm_with_tools = agent.llm_instance.llm

                async for chunk in llm_with_tools.astream(
                    messages, config=runnable_config
                ):
                    # Essential safety check for None chunk
                    if chunk is None:
                        continue
                    last_chunk = chunk

                    # Accumulate chunks for proper tool call parsing
                    if accumulated_chunk is None:
                        accumulated_chunk = chunk
                    else:
                        accumulated_chunk = accumulated_chunk + chunk

                    # Early-finish hint (no break): if finish_reason signals end and no tool calls in this iteration,
                    # emit a completion hint so UI can react promptly, but keep reading to allow usage/native counts.
                    try:
                        response_meta = getattr(chunk, "response_metadata", None)
                        additional = getattr(chunk, "additional_kwargs", None)
                        normalized_finish = None
                        native_finish = None
                        if isinstance(response_meta, dict):
                            normalized_finish = response_meta.get("finish_reason")
                            native_finish = response_meta.get("native_finish_reason")
                        if normalized_finish is None and isinstance(additional, dict):
                            normalized_finish = additional.get("finish_reason")
                            native_finish = additional.get("native_finish_reason")

                        if (
                            normalized_finish
                            in {"stop", "length", "content_filter", "error"}
                            and not has_tool_calls
                        ):
                            yield StreamingEvent(
                                event_type="iteration_progress",
                                content=get_translation_key(
                                    "processing_complete", language
                                ),
                                metadata={
                                    "conversation_complete": True,
                                    "early_finish": True,
                                    "finish_reason": normalized_finish,
                                    "native_finish_reason": native_finish,
                                },
                            )
                    except Exception as exc:
                        # Log directly - if logging fails, let it fail visibly for debugging
                        self._logger.debug("Failed to emit early-finish hint: %s", exc)

                    # Stream content as it arrives
                    if hasattr(chunk, "content") and chunk.content:
                        yield StreamingEvent(
                            event_type="content",
                            content=chunk.content,
                            metadata={
                                "chunk_type": "llm_stream",
                                "provider": agent.llm_instance.provider.value,
                            },
                        )

                    # Process tool call chunks as they stream
                    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        for tool_call_chunk in chunk.tool_call_chunks:
                            # Essential safety check for None tool_call_chunk
                            if tool_call_chunk is None:
                                continue

                            tool_name = (
                                tool_call_chunk.get("name")
                                if isinstance(tool_call_chunk, dict)
                                else None
                            )
                            tool_call_id = (
                                tool_call_chunk.get("id")
                                if isinstance(tool_call_chunk, dict)
                                else None
                            )
                            tool_args = (
                                tool_call_chunk.get("args", "")
                                if isinstance(tool_call_chunk, dict)
                                else ""
                            )

                            if tool_name and tool_call_id not in tool_calls_in_progress:
                                # New tool call starting
                                tool_calls_in_progress[tool_call_id] = {
                                    "name": tool_name,
                                    "args": "",
                                    "id": tool_call_id,
                                }
                                # yield StreamingEvent(
                                #     event_type="tool_start",
                                #     content=f"\n\n2🔧 **Using tool: {tool_name}**",
                                #     metadata={
                                #         "tool_name": tool_name,
                                #         "tool_call_id": tool_call_id,
                                #         "title": f"1 🔧 Tool called: {tool_name}"
                                #     }
                                # )

                            if tool_call_id in tool_calls_in_progress:
                                # Accumulate tool arguments - safety check for None
                                safe_tool_args = (
                                    tool_args if tool_args is not None else ""
                                )
                                tool_calls_in_progress[tool_call_id]["args"] += (
                                    safe_tool_args
                                )

                # Accumulate provider usage for this iteration call (when available).
                # This is per-LLM-call usage and will be committed once per QA turn at finalization.
                try:
                    if (
                        (last_chunk is not None or accumulated_chunk is not None)
                        and hasattr(agent, "token_tracker")
                        and agent.token_tracker
                    ):
                        usage_source = last_chunk if last_chunk is not None else accumulated_chunk
                        ok = agent.token_tracker.update_turn_usage_from_api(usage_source)
                        if not ok and accumulated_chunk is not None and accumulated_chunk is not usage_source:
                            # Some providers attach usage only to the final aggregated chunk.
                            agent.token_tracker.update_turn_usage_from_api(accumulated_chunk)
                except Exception as exc:
                    # Log directly - if logging fails, let it fail visibly for debugging
                    self._logger.debug("Failed to accumulate iteration usage: %s", exc)

                # Process completed tool calls
                if (
                    hasattr(accumulated_chunk, "tool_calls")
                    and accumulated_chunk.tool_calls
                ):
                    has_tool_calls = True
                    print(
                        f"🔍 DEBUG: Found {len(accumulated_chunk.tool_calls)} tool calls"
                    )

                    # STEP 1: Filter and count duplicates BEFORE execution
                    deduplicated_tool_calls, duplicate_counts = (
                        self._deduplicate_tool_calls(accumulated_chunk.tool_calls)
                    )
                    print(
                        f"🔍 DEBUG: Original tool calls: {len(accumulated_chunk.tool_calls)}, Deduplicated: {len(deduplicated_tool_calls)}"
                    )

                    # STEP 2: Execute only deduplicated tool calls and create result mapping
                    tool_result_cache = {}  # Map tool_key -> result for duplicate handling

                    for tool_call in deduplicated_tool_calls:
                        # Essential safety check for None tool_call
                        if not tool_call or not isinstance(tool_call, dict):
                            continue

                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("args", {})
                        tool_call_id = tool_call.get("id")

                        if tool_name and tool_call_id in tool_calls_in_progress:
                            # Get duplicate count for this tool call - safety check for None tool_args
                            safe_tool_args = tool_args if tool_args is not None else {}
                            try:
                                tool_key = f"{tool_name}:{hash(json.dumps(safe_tool_args, sort_keys=True, default=str))}"
                            except Exception:
                                tool_key = f"{tool_name}:{hash(str(safe_tool_args))}"
                            duplicate_count = duplicate_counts.get(tool_key, 1)

                            # Find the tool in our tools list
                            tool_obj = None
                            for tool in agent.tools:
                                if tool.name == tool_name:
                                    tool_obj = tool
                                    break

                            try:
                                if tool_obj:
                                    # Execute tool once with agent context
                                    # Inject agent instance for file resolution (same as memory manager)
                                    tool_args_with_agent = {
                                        **safe_tool_args,
                                        "agent": agent,
                                    }
                                    # Ensure session-bound config is visible to backend tools
                                    try:
                                        if (
                                            hasattr(agent, "session_id")
                                            and agent.session_id
                                        ):
                                            # Lazy import to avoid circular dependency
                                            from .session_manager import set_current_session_id
                                            set_current_session_id(agent.session_id)
                                    except Exception as exc:
                                        # Log directly - if logging fails, let it fail visibly for debugging
                                        self._logger.debug(
                                            "Failed to set current session ID: %s", exc
                                        )
                                    tool_result = tool_obj.invoke(tool_args_with_agent)

                                    # Safety check for None tool_result
                                    if tool_result is None:
                                        print(
                                            f"⚠️ Tool {tool_name} returned None result"
                                        )
                                        tool_result = "Tool execution completed but returned no result"

                                    # Store the tool call result for future deduplication
                                    deduplicator = get_deduplicator()
                                    deduplicator.store_tool_call(
                                        tool_name,
                                        safe_tool_args,
                                        tool_result,
                                        conversation_id,
                                    )

                                    # Cache result for duplicate tool calls
                                    tool_result_cache[tool_key] = tool_result

                                    # CRITICAL: Stream tool completion only ONCE per unique tool
                                    # Show duplicate count in the message but don't stream multiple times

                                    # Safety check for None tool_result to prevent concatenation errors
                                    safe_tool_result = (
                                        tool_result
                                        if tool_result is not None
                                        else "Tool execution returned no result"
                                    )

                                    yield StreamingEvent(
                                        event_type="tool_end",
                                        content=f"\n{self._get_call_count_message(duplicate_count, language)}\n\n{self._get_result_message(str(safe_tool_result), language)}",
                                        metadata={
                                            "tool_name": tool_name,
                                            "tool_output": str(safe_tool_result),
                                            "duplicate": duplicate_count > 1,
                                            "duplicate_count": duplicate_count,
                                            "title": self._get_tool_called_message(
                                                tool_name, language
                                            ),
                                        },
                                    )

                                    # Remove from in-progress
                                    del tool_calls_in_progress[tool_call_id]
                                else:
                                    # Unknown tool - return as tool result instead of error
                                    unknown_tool_result = self._get_unknown_tool_message(
                                        tool_name, language
                                    )

                                    # Cache result for duplicate tool calls
                                    tool_result_cache[tool_key] = unknown_tool_result

                                    yield StreamingEvent(
                                        event_type="tool_end",
                                        content=f"\n{self._get_call_count_message(duplicate_count, language)}\n\n{self._get_result_message(unknown_tool_result, language)}",
                                        metadata={
                                            "tool_name": tool_name,
                                            "tool_output": unknown_tool_result,
                                            "duplicate": duplicate_count > 1,
                                            "duplicate_count": duplicate_count,
                                            "title": self._get_tool_called_message(
                                                tool_name, language
                                            ),
                                        },
                                    )

                                    # Stream unknown tool to debug logs
                                    try:
                                        debug_streamer = get_debug_streamer(conversation_id)
                                        debug_streamer.warning(
                                            f"Unknown tool: {tool_name}",
                                            category="tool_execution",
                                            metadata={"tool_name": tool_name},
                                        )
                                    except Exception as exc:
                                        # Log directly - if logging fails, let it fail visibly
                                        self._logger.debug(
                                            "Failed to stream tool start debug: %s", exc
                                        )

                                    # Remove from in-progress
                                    del tool_calls_in_progress[tool_call_id]
                                    continue

                            except Exception as e:
                                # Tool execution error - return as tool result instead of separate error
                                tool_error_result = self._get_tool_error_message(
                                    str(e), language
                                )

                                # Cache result for duplicate tool calls
                                tool_result_cache[tool_key] = tool_error_result

                                yield StreamingEvent(
                                    event_type="tool_end",
                                    content=f"\n{self._get_call_count_message(duplicate_count, language)}\n\n{self._get_result_message(tool_error_result, language)}",
                                    metadata={
                                        "tool_name": tool_name,
                                        "tool_output": tool_error_result,
                                        "duplicate": duplicate_count > 1,
                                        "duplicate_count": duplicate_count,
                                        "title": self._get_tool_called_message(
                                            tool_name, language
                                        ),
                                    },
                                )

                                # Stream tool error to debug logs
                                try:
                                    debug_streamer = get_debug_streamer(conversation_id)
                                    debug_streamer.error(
                                        f"Tool execution error: {str(e)}",
                                        category="tool_execution",
                                        metadata={"tool_name": tool_name, "error": str(e)},
                                    )
                                except Exception as exc:
                                    # Log directly - if logging fails, let it fail visibly
                                    self._logger.debug(
                                        "Failed to stream tool error debug: %s", exc
                                    )

                                # Remove from in-progress
                                tool_calls_in_progress.pop(tool_call_id, None)

                    # CRITICAL: Add AI message with tool_calls to working messages FIRST
                    if (
                        accumulated_chunk
                        and hasattr(accumulated_chunk, "tool_calls")
                        and accumulated_chunk.tool_calls
                    ):
                        # Add the AIMessage with tool_calls to the working messages
                        ai_message_with_tool_calls = AIMessage(
                            content=accumulated_chunk.content or "",
                            tool_calls=accumulated_chunk.tool_calls,
                        )
                        messages.append(ai_message_with_tool_calls)
                        print(
                            f"🔍 DEBUG: Added AIMessage with {len(accumulated_chunk.tool_calls)} tool calls to working messages"
                        )

                        # UI-only separation: emit a lean trailing newline after AI message that calls a tool
                        # This improves readability without affecting persisted memory
                        yield StreamingEvent(
                            event_type="content",
                            content="\n",
                            metadata={
                                "chunk_type": "ui_separator",
                                "provider": agent.llm_instance.provider.value,
                            },
                        )

                        # STEP 3: Create ToolMessage for EACH original tool call (including duplicates)
                        # This ensures proper tool_call_id mapping and message sequence
                        # CRITICAL: Add ToolMessages to working messages for proper sequence
                        tool_messages = []
                        for original_tool_call in accumulated_chunk.tool_calls:
                            # Essential safety check for None original_tool_call
                            if not original_tool_call or not isinstance(
                                original_tool_call, dict
                            ):
                                continue

                            tool_name = original_tool_call.get("name")
                            tool_args = original_tool_call.get("args", {})
                            tool_call_id = original_tool_call.get("id")

                            if tool_name and tool_call_id:
                                # Get the tool key for result lookup (exclude agent from key calculation)
                                cache_args = {k: v for k, v in tool_args.items() if k != "agent"}
                                try:
                                    tool_key = f"{tool_name}:{hash(json.dumps(cache_args, sort_keys=True, default=str))}"
                                except Exception:
                                    tool_key = f"{tool_name}:{hash(str(cache_args))}"

                                # Get cached result (same for all duplicates)
                                # This includes both successful results and error results
                                tool_result = tool_result_cache.get(
                                    tool_key, "Tool execution failed"
                                )

                                # Truncate tool result if it exceeds maximum size
                                # This prevents huge tool results (like 141MB PDFs) from overflowing context
                                tool_result_str = str(tool_result)
                                tool_result_tokens = count_tokens(tool_result_str)

                                # Get context window from agent's LLM instance
                                context_window = 0
                                if (
                                    hasattr(agent, "llm_instance")
                                    and agent.llm_instance
                                    and hasattr(agent.llm_instance, "config")
                                ):
                                    context_window = agent.llm_instance.config.get("token_limit", 0)

                                # Only truncate if context window is available (relational threshold)
                                if context_window > 0:
                                    max_tool_result_tokens = int(
                                        context_window * MAX_TOOL_RESULT_TOKENS_PCT
                                    )

                                    if tool_result_tokens > max_tool_result_tokens:
                                        # Truncate to approximately max_tool_result_tokens
                                        # Use character-based truncation as approximation (~3 chars per token)
                                        max_chars = max_tool_result_tokens * 3
                                        truncated_result = tool_result_str[:max_chars]
                                        tool_result_str = (
                                            f"{truncated_result}\n\n"
                                            f"[Tool result truncated: {tool_result_tokens:,} tokens → "
                                            f"{count_tokens(truncated_result):,} tokens "
                                            f"(max: {max_tool_result_tokens:,} tokens, "
                                            f"{MAX_TOOL_RESULT_TOKENS_PCT*100:.0f}% of "
                                            f"{context_window:,} token context window)]"
                                        )
                                        self._logger.warning(
                                            "Tool result for %s exceeded max size (%d tokens), "
                                            "truncated to %d tokens (max: %d tokens, %.0f%% of "
                                            "%d token context window)",
                                            tool_name,
                                            tool_result_tokens,
                                            count_tokens(tool_result_str),
                                            max_tool_result_tokens,
                                            MAX_TOOL_RESULT_TOKENS_PCT * 100,
                                            context_window,
                                        )
                                elif tool_result_tokens > 100000:
                                    # Safety check: warn if extremely large even without context window
                                    self._logger.warning(
                                        "Tool result for %s is extremely large (%d tokens) but "
                                        "context window unavailable, proceeding without truncation",
                                        tool_name,
                                        tool_result_tokens,
                                    )

                                # Create ToolMessage with original tool_call_id
                                tool_message = ToolMessage(
                                    content=tool_result_str,
                                    tool_call_id=tool_call_id,
                                    name=tool_name,
                                )
                                tool_messages.append(tool_message)
                                self._logger.debug(
                                    "Created ToolMessage for %s with ID %s", tool_name, tool_call_id
                                )

                        # CRITICAL: Add ToolMessages to working messages for next LLM call
                        # This ensures proper sequence: AIMessage(with tool_calls) → ToolMessages
                        messages.extend(tool_messages)
                        self._logger.debug(
                            "Added %d ToolMessages to working messages", len(tool_messages)
                        )

                        # Check for overflow protection AFTER tool results are added
                        # This prevents overflow when large tool results push context over limit.
                        try:
                            if hasattr(agent, "token_tracker") and agent.token_tracker:
                                # Refresh budget snapshot with messages including tool results
                                agent.token_tracker.refresh_budget_snapshot(
                                    agent=agent,
                                    conversation_id=conversation_id,
                                    messages_override=messages,
                                )
                                snapshot = agent.token_tracker.get_budget_snapshot() or {}
                                percentage_used = snapshot.get("percentage_used", 0.0)
                                # Fallback first when enabled and threshold reached
                                if (
                                    getattr(agent, "use_fallback_model", False)
                                    and percentage_used >= HISTORY_COMPRESSION_MID_TURN_THRESHOLD
                                ):
                                    switched = self._try_switch_to_fallback_model(
                                        agent, conversation_id, trigger="post_tool"
                                    )
                                    if (
                                        switched
                                        and hasattr(agent, "token_tracker")
                                        and agent.token_tracker
                                    ):
                                        agent.token_tracker.refresh_budget_snapshot(
                                            agent=agent,
                                            conversation_id=conversation_id,
                                            messages_override=messages,
                                        )
                                if should_compress_mid_turn(
                                    agent, conversation_id, messages_override=messages
                                ):
                                    budget_snapshot = agent.token_tracker.get_budget_snapshot()
                                    success, updated_messages = await perform_compression_with_notifications(
                                        agent=agent,
                                        conversation_id=conversation_id,
                                        language=language,
                                        keep_recent_turns=HISTORY_COMPRESSION_KEEP_RECENT_TURNS_MID_TURN,
                                        reason="proactive",
                                        budget_snapshot=budget_snapshot,
                                        rebuild_messages=True,
                                    )
                                    if success and updated_messages is not None:
                                        messages = updated_messages
                                        # Re-add tool messages after compression (they're recent)
                                        # Filter to only add tool messages that aren't already present
                                        existing_tool_call_ids = {
                                            getattr(msg, "tool_call_id", None)
                                            for msg in messages
                                            if isinstance(msg, ToolMessage)
                                        }
                                        for tool_msg in tool_messages:
                                            if (
                                                isinstance(tool_msg, ToolMessage)
                                                and getattr(tool_msg, "tool_call_id", None)
                                                not in existing_tool_call_ids
                                            ):
                                                messages.append(tool_msg)
                                                existing_tool_call_ids.add(
                                                    getattr(tool_msg, "tool_call_id", None)
                                                )
                        except Exception as comp_exc:
                            # Non-fatal: log and continue
                            self._logger.debug(
                                "Failed to check/perform post-tool overflow protection: %s",
                                comp_exc,
                            )

                    # CRITICAL: Continue to next iteration to get final response
                    # Don't break here - we need the final AI response after tool calls
                elif (
                    accumulated_chunk
                    and hasattr(accumulated_chunk, "content")
                    and accumulated_chunk.content
                ):
                    # Regular AI message without tool calls
                    ai_message = AIMessage(content=accumulated_chunk.content)
                    messages.append(ai_message)
                    self._logger.debug("Added regular AIMessage to working messages")

                # If no tool calls, we're done
                if not has_tool_calls:
                    self._logger.debug(
                        "No tool calls in iteration %d, conversation complete", iteration
                    )

                    # Stream conversation completion
                    icons = self._get_ui_icons(language)
                    finish_icon = icons["finish_icons"][
                        iteration % len(icons["finish_icons"])
                    ]

                    yield StreamingEvent(
                        event_type="iteration_progress",
                        content=f"{finish_icon} **{self._get_iteration_finished_message(iteration, self.max_iterations, language)}**",
                        metadata={
                            "iteration": iteration,
                            "max_iterations": self.max_iterations,
                            "conversation_complete": True,
                        },
                    )

                    # Add messages to memory for conversations without tool calls
                    self._logger.debug("Adding new messages to memory manager")

                    # Get current memory content for deduplication
                    current_memory = agent.memory_manager.get_conversation_history(
                        conversation_id
                    )
                    memory_content = {
                        (type(msg).__name__, msg.content)
                        for msg in current_memory
                        if hasattr(msg, "content")
                    }

                    new_messages_added = 0
                    for message in messages:
                        # Skip system messages - they're handled separately above
                        if isinstance(message, SystemMessage):
                            print(
                                "🔍 DEBUG: Skipped system message (handled separately)"
                            )
                            continue

                        # Skip empty AIMessages - they cause message order issues
                        if (
                            isinstance(message, AIMessage)
                            and not message.content
                            and not message.tool_calls
                        ):
                            print(
                                "🔍 DEBUG: Skipped empty AIMessage (no content or tool calls)"
                            )
                            continue

                        # Create a unique identifier for this message
                        message_key = (
                            type(message).__name__,
                            message.content
                            if hasattr(message, "content")
                            else str(message),
                        )

                        # Only add if not already in memory
                        if message_key not in memory_content:
                            agent.memory_manager.add_message(conversation_id, message)
                            new_messages_added += 1
                            print(
                                f"🔍 DEBUG: Added new {type(message).__name__} to memory"
                            )
                        else:
                            print(
                                f"🔍 DEBUG: Skipped duplicate {type(message).__name__}"
                            )

                    self._logger.debug(
                        "Added %s new messages to memory", new_messages_added
                    )

                    # Emit turn_complete event for conversations without tool calls
                    try:
                        ordered_messages_snapshot = []
                        for m in messages:
                            try:
                                if isinstance(m, SystemMessage):
                                    role = "system"
                                elif isinstance(m, HumanMessage):
                                    role = "user"
                                elif isinstance(m, ToolMessage):
                                    role = "tool"
                                elif isinstance(m, AIMessage):
                                    role = "assistant"
                                else:
                                    role = type(m).__name__
                                ordered_messages_snapshot.append(
                                    {
                                        "role": role,
                                        "content": getattr(m, "content", ""),
                                        "tool_call_id": getattr(
                                            m, "tool_call_id", None
                                        ),
                                        "tool_calls": getattr(m, "tool_calls", None),
                                        "name": getattr(m, "name", None),
                                    }
                                )
                            except Exception as exc:
                                # Log directly - if logging fails, let it fail visibly
                                self._logger.debug(
                                    "Failed to snapshot message for turn_complete: %s", exc
                                )
                                continue

                        yield StreamingEvent(
                            event_type="turn_complete",
                            content="",
                            metadata={
                                "ordered_messages": ordered_messages_snapshot,
                                "conversation_id": conversation_id,
                            },
                        )
                    except Exception as exc:
                        # Non-fatal if snapshot fails - log directly
                        self._logger.debug(
                            "Failed to create ordered messages snapshot: %s", exc
                        )

                    break

                self._logger.debug("Completed iteration %d, continuing...", iteration)

                # Stream iteration completion as separate message
                yield StreamingEvent(
                    event_type="iteration_progress",
                    content=self._get_iteration_completed_message(iteration, language),
                    metadata={
                        "iteration": iteration,
                        "completed": True,
                    },
                )

            # Check if we hit max iterations
            if iteration >= self.max_iterations:
                print(
                    f"🔍 DEBUG: Reached max iterations ({self.max_iterations}), stopping conversation"
                )

                # Stream max iterations completion
                icons = self._get_ui_icons(language)
                max_icon = icons["max_icons"][iteration % len(icons["max_icons"])]

                yield StreamingEvent(
                    event_type="iteration_progress",
                    content=f"{max_icon} **{self._get_iteration_max_reached_message(iteration, self.max_iterations, language)}**",
                    metadata={
                        "iteration": iteration,
                        "max_iterations": self.max_iterations,
                        "max_reached": True,
                    },
                )

                yield StreamingEvent(
                    event_type="warning",
                    content=self._get_max_iterations_warning_message(
                        self.max_iterations, language
                    ),
                    metadata={"max_iterations_reached": True},
                )

            # Use the last accumulated chunk for token tracking
            final_chunk = accumulated_chunk
            if "last_chunk" in locals() and last_chunk is not None:
                # Prefer last raw chunk for message content, but keep accumulated as fallback.
                final_chunk = last_chunk or accumulated_chunk

            # Finalize per-turn usage: commit accumulated per-iteration usage once per QA turn
            # (preserves avg-per-message semantics).
            if hasattr(agent, "token_tracker") and agent.token_tracker:
                try:
                    # IMPORTANT: OpenRouter cost/cache details often appear only on the final chunk
                    # under response_metadata.token_usage. Refresh once from final_chunk before finalize.
                    try:
                        agent.token_tracker.update_turn_usage_from_api(final_chunk)
                    except Exception as exc:
                        self._logger.debug(
                            "Failed to refresh final usage from final_chunk: %s", exc
                        )
                    agent.token_tracker.finalize_turn_usage(final_chunk, messages)
                except Exception as e:
                    self._logger.exception("Error finalizing turn usage: %s", e)

            # Budget moment: after finalization, store a final budget snapshot for UI.
            try:
                if hasattr(agent, "token_tracker") and agent.token_tracker:
                    agent.token_tracker.refresh_budget_snapshot(
                        agent=agent,
                        conversation_id=conversation_id,
                        messages_override=messages,
                    )
            except Exception as exc:
                # Log directly - if logging fails, let it fail visibly for debugging
                self._logger.debug("Failed to persist/stream error: %s", exc)

            # Check for compression after successful turn completion
            try:
                if hasattr(agent, "token_tracker") and agent.token_tracker:
                    budget_snapshot = agent.token_tracker.get_budget_snapshot()
                    if should_compress_on_completion(
                        agent, conversation_id, budget_snapshot.get("status")
                    ):
                        await perform_compression_with_notifications(
                            agent=agent,
                            conversation_id=conversation_id,
                            language=language,
                            keep_recent_turns=HISTORY_COMPRESSION_KEEP_RECENT_TURNS_SUCCESS,
                            reason="critical",
                            budget_snapshot=budget_snapshot,
                            rebuild_messages=False,
                        )
            except Exception as comp_exc:
                # Non-fatal: log and continue
                self._logger.debug("Failed to check/perform compression: %s", comp_exc)

            # Extract normalized/native finish_reason if present (provider-agnostic; OpenRouter supplies both)
            def _extract_finish_reason(chunk) -> tuple[str | None, str | None]:
                try:
                    if not chunk:
                        return None, None
                    # Prefer response_metadata.native fields if available
                    response_meta = getattr(chunk, "response_metadata", None)
                    if isinstance(response_meta, dict):
                        normalized = response_meta.get("finish_reason")
                        native = response_meta.get("native_finish_reason")
                        if normalized or native:
                            return normalized, native
                    # LangChain messages sometimes carry additional_kwargs
                    additional = getattr(chunk, "additional_kwargs", None)
                    if isinstance(additional, dict):
                        normalized = additional.get("finish_reason")
                        native = additional.get("native_finish_reason")
                        if normalized or native:
                            return normalized, native
                except Exception as exc:
                    self._logger.debug(
                        "Failed to extract finish reason: %s", exc
                    )
                    return None, None
                return None, None

            normalized_finish, native_finish = _extract_finish_reason(final_chunk)

            # Add all new messages to memory at the end (avoid duplication)
            self._logger.debug("Adding new messages to memory manager")

            # Get current memory content for deduplication
            current_memory = agent.memory_manager.get_conversation_history(
                conversation_id
            )
            memory_content = {
                (type(msg).__name__, msg.content)
                for msg in current_memory
                if hasattr(msg, "content")
            }

            new_messages_added = 0
            for message in messages:
                # Skip system messages - they're handled separately above
                if isinstance(message, SystemMessage):
                    self._logger.debug("Skipped system message (handled separately)")
                    continue

                # Skip empty AIMessages - they cause message order issues
                if (
                    isinstance(message, AIMessage)
                    and not message.content
                    and not message.tool_calls
                ):
                    print(
                        "🔍 DEBUG: Skipped empty AIMessage (no content or tool calls)"
                    )
                    continue

                # Create a unique identifier for this message
                message_key = (
                    type(message).__name__,
                    message.content if hasattr(message, "content") else str(message),
                )

                # Only add if not already in memory
                if message_key not in memory_content:
                    agent.memory_manager.add_message(conversation_id, message)
                    new_messages_added += 1
                    self._logger.debug("Added new %s to memory", type(message).__name__)
                else:
                    self._logger.debug("Skipped duplicate %s", type(message).__name__)

            self._logger.debug("Added %s new messages to memory", new_messages_added)

            # Emit final turn payload for persistence/observability consumers
            try:
                ordered_messages_snapshot = []
                for m in messages:
                    try:
                        if isinstance(m, SystemMessage):
                            role = "system"
                        elif isinstance(m, HumanMessage):
                            role = "user"
                        elif isinstance(m, ToolMessage):
                            role = "tool"
                        elif isinstance(m, AIMessage):
                            role = "assistant"
                        else:
                            role = type(m).__name__
                        ordered_messages_snapshot.append(
                            {
                                "role": role,
                                "content": getattr(m, "content", ""),
                                "tool_call_id": getattr(m, "tool_call_id", None),
                                "tool_calls": getattr(m, "tool_calls", None),
                                "name": getattr(m, "name", None),
                            }
                        )
                    except Exception as exc:
                        # Log directly - if logging fails, let it fail visibly for debugging
                        self._logger.debug("Failed to snapshot message metadata: %s", exc)
                        continue

                yield StreamingEvent(
                    event_type="turn_complete",
                    content="",
                    metadata={
                        "ordered_messages": ordered_messages_snapshot,
                        "conversation_id": conversation_id,
                        "finish_reason": normalized_finish,
                        "native_finish_reason": native_finish,
                    },
                )
            except Exception as exc:
                # Non-fatal if snapshot fails - log directly
                self._logger.debug("Failed to create turn_complete snapshot: %s", exc)

            # Final completion event

            yield StreamingEvent(
                event_type="completion",
                content=self._get_response_completed_message(language),
                metadata={
                    "final_response": True,
                    "finish_reason": normalized_finish,
                    "native_finish_reason": native_finish,
                },
            )

            # Final iteration progress completion

            yield StreamingEvent(
                event_type="iteration_progress",
                content=get_translation_key("processing_complete", language),
                metadata={"conversation_complete": True, "final": True},
            )

        except Exception as e:
            # Enhanced error logging for debugging
            self._logger.exception("ERROR in stream_agent_response: %s", e)

            # Persist partial turn: Human already stored; add completed ToolMessages and a truncated AIMessage if available
            try:
                current_memory = agent.memory_manager.get_conversation_history(
                    conversation_id
                )
                memory_content = {
                    (type(msg).__name__, msg.content)
                    for msg in current_memory
                    if hasattr(msg, "content")
                }

                # Collect any ToolMessages and AI messages already in working list
                partial_messages_to_add = []
                try:
                    for m in messages:
                        if isinstance(m, ToolMessage) or isinstance(m, AIMessage):
                            key = (type(m).__name__, getattr(m, "content", ""))
                            if key not in memory_content:
                                partial_messages_to_add.append(m)
                except Exception as exc:
                    # Log directly - if logging fails, let it fail visibly for debugging
                    self._logger.debug(
                        "Failed to collect partial messages for persistence: %s", exc
                    )

                # Add a lean truncated AIMessage if we have partial content
                try:
                    partial_text = ""
                    if (
                        "accumulated_chunk" in locals()
                        and hasattr(accumulated_chunk, "content")
                        and accumulated_chunk.content
                    ):
                        partial_text = accumulated_chunk.content
                    if partial_text:
                        truncated_msg = AIMessage(content=f"{partial_text} [truncated]")
                        key = ("AIMessage", truncated_msg.content)
                        if key not in memory_content:
                            partial_messages_to_add.append(truncated_msg)
                except Exception as exc:
                    # Log directly - if logging fails, let it fail visibly for debugging
                    self._logger.debug(
                        "Failed to create truncated message for persistence: %s", exc
                    )

                for m in partial_messages_to_add:
                    try:
                        agent.memory_manager.add_message(conversation_id, m)
                    except Exception as exc:
                        # Log directly - if logging fails, let it fail visibly for debugging
                        self._logger.debug("Failed to emit early-finish hint: %s", exc)
                        continue
            except Exception as persist_err:
                self._logger.debug("Failed partial-turn persistence: %s", persist_err)

            yield StreamingEvent(
                event_type="error",
                content=self._get_error_message(str(e), language),
                metadata={"error": str(e)},
            )

            # Persist top-level streamed error as AIMessage
            try:
                agent.memory_manager.add_message(
                    conversation_id,
                    AIMessage(content=self._get_error_message(str(e), language)),
                )
            except Exception as exc:
                # Log directly - if logging fails, let it fail visibly for debugging
                self._logger.debug("Failed to persist error message to memory: %s", exc)

            # Stream top-level error to debug logs
            try:
                debug_streamer = get_debug_streamer(conversation_id)
                debug_streamer.error(
                    f"Streaming error: {str(e)}",
                    category="streaming",
                    metadata={"error": str(e)},
                )
            except Exception as exc:
                # Log directly - if logging fails, let it fail visibly for debugging
                self._logger.debug("Failed to persist/stream error: %s", exc)

            # Finalize turn usage on error: mark as interrupted so estimates can be used
            # if no API counts were accumulated.
            if hasattr(agent, "token_tracker") and agent.token_tracker:
                try:
                    agent.token_tracker.finalize_turn_usage(None, messages)
                except Exception as finalize_exc:
                    # Log directly - if logging fails, let it fail visibly for debugging
                    self._logger.debug(
                        "Failed to finalize turn usage on error: %s", finalize_exc
                    )

            # Check for compression after interrupted turn (if critical status)
            try:
                if hasattr(agent, "token_tracker") and agent.token_tracker:
                    budget_snapshot = agent.token_tracker.get_budget_snapshot()
                    if (
                        budget_snapshot
                        and budget_snapshot.get("status") == "critical"
                        and should_compress_on_completion(
                            agent, conversation_id, budget_snapshot.get("status")
                        )
                    ):
                        await perform_compression_with_notifications(
                            agent=agent,
                            conversation_id=conversation_id,
                            language=language,
                            keep_recent_turns=HISTORY_COMPRESSION_KEEP_RECENT_TURNS_MID_TURN,
                            reason="interrupted",
                            budget_snapshot=budget_snapshot,
                            rebuild_messages=False,
                        )
            except Exception as comp_exc:
                # Non-fatal: log and continue
                self._logger.debug("Failed to check/perform compression on error: %s", comp_exc)

            # Error completion for progress display
            yield StreamingEvent(
                event_type="iteration_progress",
                content=self._get_processing_failed_message(language),
                metadata={"error": True, "final": True},
            )


# Global streaming instance
_native_streaming_instance = None


def get_native_streaming() -> NativeLangChainStreaming:
    """Get the global native streaming instance"""
    global _native_streaming_instance
    if _native_streaming_instance is None:
        _native_streaming_instance = NativeLangChainStreaming()
    return _native_streaming_instance
