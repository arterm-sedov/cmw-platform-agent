"""Token budgeting helpers (lean, pure, non-breaking).

This module borrows best practices from the reference `cmw-rag` rag_engine:
- Centralized exact token counting via tiktoken `cl100k_base`
- Unified context token accounting (conversation vs tool)
- Optional overhead token accounting (system prompt + tool schemas + safety margin)
- A single reusable snapshot function to compute budget figures at "budget moments"

Design constraints for this repo:
- Count *all* tool results as they appear (no RAG article deduplication).
- Do not mutate agent logic, memory, or tool execution flow; only read existing state.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import tiktoken

if TYPE_CHECKING:
    from collections.abc import Iterable

try:
    from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
except Exception:  # pragma: no cover
    BaseMessage = Any  # type: ignore[assignment]
    SystemMessage = Any  # type: ignore[assignment]
    ToolMessage = Any  # type: ignore[assignment]


_ENCODING = tiktoken.get_encoding("cl100k_base")

# Global average tool size (calculated once at first tool binding)
_GLOBAL_AVG_TOOL_SIZE: int | None = None

# Token budget constants
DEFAULT_TOOL_JSON_OVERHEAD_PCT = 0.0  # Reduced from 0.10 to avoid inflation
DEFAULT_SAFETY_MARGIN = 0  # Reduced from 2000 as it was causing token inflation

# Overhead adjustment factor: heuristic to match API-reported tokens
# Calculated from API data: inferred_actual_overhead / estimated_overhead ≈ 0.8
# Set to 1.0 to disable adjustment (use raw estimates)
OVERHEAD_ADJUSTMENT_FACTOR = 0.8

# Token budget status thresholds (percentage of context window used)
TOKEN_STATUS_CRITICAL_THRESHOLD = 90.0
TOKEN_STATUS_WARNING_THRESHOLD = 75.0
TOKEN_STATUS_MODERATE_THRESHOLD = 50.0

# Token budget status strings
TOKEN_STATUS_CRITICAL = "critical"  # noqa: S105
TOKEN_STATUS_WARNING = "warning"  # noqa: S105
TOKEN_STATUS_MODERATE = "moderate"  # noqa: S105
TOKEN_STATUS_GOOD = "good"  # noqa: S105
TOKEN_STATUS_UNKNOWN = "unknown"  # noqa: S105

# History compression configuration
HISTORY_COMPRESSION_TARGET_TOKENS_PCT = 0.10  # 10% of context window for compressed summary
HISTORY_COMPRESSION_KEEP_RECENT_TURNS_MID_TURN = 1  # Keep N recent turns when compressing mid-turn or after interrupted turn
HISTORY_COMPRESSION_KEEP_RECENT_TURNS_SUCCESS = 0  # Keep N recent turns when compressing after successful completion
HISTORY_COMPRESSION_MID_TURN_THRESHOLD = TOKEN_STATUS_CRITICAL_THRESHOLD  # Trigger proactive compression at this %


def count_tokens(content: str | Any) -> int:
    """Count tokens in a string using exact cl100k_base encoding."""
    if content is None:
        return 0
    s = str(content)
    if not s:
        return 0
    return len(_ENCODING.encode(s))


def _message_content(msg: Any) -> str:
    if msg is None:
        return ""
    if hasattr(msg, "content"):
        try:
            return str(msg.content or "")
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to extract message content: %s", exc
            )
            return ""
    if isinstance(msg, dict):
        return str(msg.get("content", "") or "")
    return str(msg)


def _is_tool_message(msg: Any) -> bool:
    if msg is None:
        return False
    if ToolMessage is not Any and isinstance(msg, ToolMessage):
        return True
    msg_type = getattr(msg, "type", None)
    if msg_type == "tool":
        return True
    if isinstance(msg, dict):
        # Some tool UIs use role="tool"
        return msg.get("role") == "tool" or msg.get("type") == "tool"
    return False


def compute_context_tokens(
    messages: Iterable[Any],
    *,
    add_json_overhead: bool = True,
    json_overhead_pct: float = DEFAULT_TOOL_JSON_OVERHEAD_PCT,
) -> tuple[int, int]:
    """Compute (conversation_tokens, tool_tokens) for provided messages.

    Counts *all* tool messages as they appear (no deduplication).
    """
    conversation_tokens = 0
    tool_tokens = 0

    for msg in messages or []:
        content = _message_content(msg)
        if not content:
            continue
        if _is_tool_message(msg):
            tool_tokens += count_tokens(content)
        else:
            conversation_tokens += count_tokens(content)

    if add_json_overhead and tool_tokens > 0:
        try:
            pct = float(json_overhead_pct)
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to parse json_overhead_pct, using default: %s", exc
            )
            pct = DEFAULT_TOOL_JSON_OVERHEAD_PCT
        tool_tokens = int(tool_tokens * (1.0 + max(0.0, pct)))

    return conversation_tokens, tool_tokens


def _calculate_avg_tool_size(tools: Iterable[Any] | None) -> int:
    """Calculate average tool size from bound tools (dicts).

    Pure function: only calculates and returns value, no side effects.
    Returns 600 as fallback if calculation fails.
    """
    tool_list = list(tools or [])
    if not tool_list:
        return 600

    serialized_tokens = []
    for tool in tool_list:
        if isinstance(tool, dict):
            try:
                serialized = json.dumps(tool, separators=(",", ":"))
                tokens = count_tokens(serialized)
                serialized_tokens.append(tokens)
            except Exception as exc:
                # Log failed serialization (non-critical for average calculation)
                tool_name = tool.get("function", {}).get("name", tool.get("name", "?"))
                logging.getLogger(__name__).warning(
                    "Failed to serialize tool %s for avg calculation: %s",
                    tool_name, exc
                )

    if serialized_tokens:
        avg = sum(serialized_tokens) // len(serialized_tokens)
        logging.getLogger(__name__).info(
            "Calculated tool average: %d tokens (from %d tools)",
            avg, len(serialized_tokens)
        )
        return avg

    return 600


def compute_overhead_tokens(
    *,
    tools: Iterable[Any] | None,
) -> int:
    """Compute overhead tokens from tool schemas only.

    Note: system_prompt and safety_margin are hardcoded to "" and 0 respectively
    since they're always constants in our usage.
    """
    tool_list = list(tools or [])
    total = 0

    logger = logging.getLogger(__name__)
    logger.debug(
        "compute_overhead_tokens: processing %d tools, types: %s",
        len(tool_list),
        [type(t).__name__ for t in tool_list[:5]]  # Log first 5 types
    )

    # Get global average tool size (calculated once at first tool binding)
    avg_tokens = _GLOBAL_AVG_TOOL_SIZE if _GLOBAL_AVG_TOOL_SIZE is not None else 600

    for tool in tool_list:
        # Primary path: tool is already a dict (OpenAI format from llm.kwargs["tools"])
        # This is the normal case after bind_tools() - serialize and count directly.
        if isinstance(tool, dict):
            try:
                serialized = json.dumps(tool, separators=(",", ":"))
                tokens = count_tokens(serialized)
                total += tokens
                tool_name = tool.get("function", {}).get("name", tool.get("name", "?"))
                logger.debug("Tool %s: %d tokens", tool_name, tokens)
            except Exception as exc:
                # Use cached average for failed serialization
                total += avg_tokens
                logger.debug(
                    "Tool dict serialization failed, using cached avg: %d tokens (%s)",
                    avg_tokens, exc
                )
        else:
            # Fallback: not a dict (e.g., raw BaseTool object)
            # Use cached average calculated at binding time
            total += avg_tokens
            tool_name = getattr(tool, "name", "?")
            logger.debug(
                "Tool %s is not a dict, using cached avg: %d tokens",
                tool_name, avg_tokens
            )

    # Safety margin is hardcoded to 0, so no addition needed

    # Apply heuristic adjustment factor to match API-reported tokens
    # This compensates for differences between tiktoken and provider tokenization
    adjusted_total = int(total * OVERHEAD_ADJUSTMENT_FACTOR)

    logger.debug(
        "compute_overhead_tokens: raw=%d, adjusted=%d (factor=%.3f) for %d tools",
        total, adjusted_total, OVERHEAD_ADJUSTMENT_FACTOR, len(tool_list)
    )

    return adjusted_total


def compute_token_budget_snapshot(
    *,
    agent: Any,
    conversation_id: str = "default",
    messages_override: list[Any] | None = None,
    include_overhead: bool = True,
    add_json_overhead: bool = True,
    json_overhead_pct: float = DEFAULT_TOOL_JSON_OVERHEAD_PCT,
) -> dict[str, Any]:
    """Return a budget snapshot dict for UI/stats at a budget moment.

    Prefers `messages_override` when provided (e.g., the exact messages passed to an LLM
    call). Otherwise reads from existing
    `agent.memory_manager.get_conversation_history(conversation_id)`.

    Note: This is an ESTIMATE. During iterations, it shows the estimated context size
    (conversation + tools + overhead). At final answer, API tokens (from LLM provider)
    are the ground truth and should be preferred for "Сообщение" display.

    The snapshot total may be higher than API tokens because:
    - It includes overhead (tool schemas + safety margin)
    - It includes JSON overhead for tool results
    - API tokens reflect actual provider counting/optimization
    """
    msgs: list[Any] = []
    has_system = False

    if messages_override is not None:
        msgs = list(messages_override)
    else:
        try:
            if hasattr(agent, "memory_manager") and agent.memory_manager:
                msgs = list(
                    agent.memory_manager.get_conversation_history(conversation_id)
                    or []
                )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to read conversation history for budget snapshot: %s", exc
            )
            msgs = []

        # Ensure system prompt is represented (memory may already contain it).
        try:
            has_system = any(
                (getattr(m, "type", None) == "system")
                or (SystemMessage is not Any and isinstance(m, SystemMessage))
                for m in msgs
            )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed system message detection for budget snapshot: %s", exc
            )
        if not has_system and getattr(agent, "system_prompt", None):
            msgs = [SystemMessage(content=str(agent.system_prompt)), *msgs]

    # Detect system message presence for both memory-based and override-based snapshots.
    try:
        has_system = any(
            (getattr(m, "type", None) == "system")
            or (SystemMessage is not Any and isinstance(m, SystemMessage))
            for m in msgs
        )
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Failed system message detection for budget snapshot: %s", exc
        )
        has_system = False

    conversation_tokens, tool_tokens = compute_context_tokens(
        msgs,
        add_json_overhead=add_json_overhead,
        json_overhead_pct=json_overhead_pct,
    )

    overhead_tokens = 0
    # Always include tool schemas since they're sent to the LLM and counted by API
    tool_schema_tokens = 0
    if include_overhead:
        # Extract bound tool schemas from LLM.
        # After bind_tools(), LangChain stores tools in RunnableBinding.kwargs["tools"]
        # as OpenAI-format dicts (already serializable).
        tools_payload = None
        try:
            llm_instance = getattr(agent, "llm_instance", None)
            llm = getattr(llm_instance, "llm", None) if llm_instance else None
            if llm is not None:
                # Primary: RunnableBinding stores bound tools in kwargs
                kwargs = getattr(llm, "kwargs", None)
                if isinstance(kwargs, dict):
                    tools_payload = kwargs.get("tools")
                # Fallback: some LLMs might store directly
                if tools_payload is None:
                    tools_payload = getattr(llm, "tools", None)
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to read bound tool payload from LLM: %s", exc
            )
            tools_payload = None

        # Log which tool source we're using
        if tools_payload is not None:
            logging.getLogger(__name__).debug(
                "Using bound tools from LLM.kwargs: %d tools, type=%s",
                len(tools_payload) if hasattr(tools_payload, "__len__") else "?",
                type(tools_payload).__name__
            )
        else:
            agent_tools = getattr(agent, "tools", None)
            logging.getLogger(__name__).debug(
                "Falling back to agent.tools: %s tools",
                len(agent_tools) if agent_tools else 0
            )

        # Count only tool schemas (since they're always sent to LLM)
        tool_schema_tokens = compute_overhead_tokens(
            tools=tools_payload if tools_payload is not None
                   else getattr(agent, "tools", None),
        )

    overhead_tokens = tool_schema_tokens

    total_tokens = conversation_tokens + tool_tokens + overhead_tokens

    # Debug logging to understand token breakdown
    logger = logging.getLogger(__name__)
    logger.debug(
        "Token breakdown - conv:%s, tool:%s, overhead:%s, total:%s",
        conversation_tokens, tool_tokens, overhead_tokens, total_tokens
    )

    context_window = 0
    try:
        llm_instance = getattr(agent, "llm_instance", None)
        config = getattr(llm_instance, "config", None) if llm_instance else None
        if isinstance(config, dict):
            context_window = int(config.get("token_limit", 0) or 0)
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Failed to read context window from LLM config: %s", exc
        )
        context_window = 0

    if context_window > 0:
        percentage_used = round((total_tokens / context_window) * 100.0, 1)
        remaining_tokens = max(0, context_window - total_tokens)
        if percentage_used >= TOKEN_STATUS_CRITICAL_THRESHOLD:
            status = TOKEN_STATUS_CRITICAL
        elif percentage_used >= TOKEN_STATUS_WARNING_THRESHOLD:
            status = TOKEN_STATUS_WARNING
        elif percentage_used >= TOKEN_STATUS_MODERATE_THRESHOLD:
            status = TOKEN_STATUS_MODERATE
        else:
            status = TOKEN_STATUS_GOOD
    else:
        percentage_used = 0.0
        remaining_tokens = 0
        status = TOKEN_STATUS_UNKNOWN

    return {
        "conversation_tokens": conversation_tokens,
        "tool_tokens": tool_tokens,
        "overhead_tokens": overhead_tokens,
        "total_tokens": total_tokens,
        "context_window": context_window,
        "percentage_used": percentage_used,
        "remaining_tokens": remaining_tokens,
        "status": status,
        "ts": time.time(),
        "source": "computed",
    }

