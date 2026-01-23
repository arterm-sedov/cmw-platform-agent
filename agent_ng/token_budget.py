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
    from langchain_core.utils.function_calling import convert_to_openai_tool
except Exception:  # pragma: no cover
    BaseMessage = Any  # type: ignore[assignment]
    SystemMessage = Any  # type: ignore[assignment]
    ToolMessage = Any  # type: ignore[assignment]
    convert_to_openai_tool = None  # type: ignore[assignment]


_ENCODING = tiktoken.get_encoding("cl100k_base")

# Very small cache keyed by system prompt + tool names + safety margin.
# Cleared cache to avoid using inflated values from failed serialization
_OVERHEAD_CACHE: dict[tuple[int, tuple[str, ...], int], int] = {}

# Token budget constants
DEFAULT_TOOL_JSON_OVERHEAD_PCT = 0.0  # Reduced from 0.10 to avoid inflation
DEFAULT_SAFETY_MARGIN = 0  # Reduced from 2000 as it was causing token inflation

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


def compute_overhead_tokens(
    *,
    tools: Iterable[Any] | None,
) -> int:
    """Compute overhead tokens from tool schemas only.

    Uses a tiny cache to avoid recounting stable overhead repeatedly.
    Note: system_prompt and safety_margin are hardcoded to "" and 0 respectively
    since they're always constants in our usage.
    """
    # Hardcoded constants since we never vary them
    prompt = ""  # Always empty to avoid double-counting system prompts
    safety_margin = 0  # Always 0 for accurate estimates

    tool_list = list(tools or [])
    tool_names = tuple(
        str(
            (t.get("name") if isinstance(t, dict) else None)
            or ((t.get("function") or {}).get("name") if isinstance(t, dict) else None)
            or getattr(t, "name", None)
            or getattr(t, "__name__", None)
            or "tool"
        )
        for t in tool_list
    )
    key = (hash(prompt), tool_names, int(safety_margin))
    cached = _OVERHEAD_CACHE.get(key)
    if cached is not None:
        return cached

    total = count_tokens(prompt)  # Always 0 since prompt is ""

    logger = logging.getLogger(__name__)
    logger.debug(
        "compute_overhead_tokens: processing %d tools, types: %s",
        len(tool_list),
        [type(t).__name__ for t in tool_list[:5]]  # Log first 5 types
    )

    for tool in tool_list:
        tool_counted = False
        tool_name = "?"

        # Step 1: If tool is already a dict (OpenAI format from llm.kwargs["tools"]),
        # serialize and count directly - this is the exact format sent to the LLM.
        if isinstance(tool, dict):
            try:
                serialized = json.dumps(tool, separators=(",", ":"))
                tokens = count_tokens(serialized)
                total += tokens
                tool_counted = True
                tool_name = tool.get("function", {}).get("name", tool.get("name", "?"))
                logger.debug(
                    "Tool %s (dict format): %d tokens", tool_name, tokens
                )
            except Exception as exc:
                logger.debug(
                    "Failed to serialize dict tool %s: %s", tool.get("name", "?"), exc
                )

        # Step 2: If tool is a LangChain BaseTool object, convert it to OpenAI format
        # using LangChain's built-in converter, then serialize and count.
        # This ensures we count the exact format that gets sent to the LLM.
        if not tool_counted and convert_to_openai_tool is not None:
            try:
                # Convert BaseTool to OpenAI format
                # (same as bind_tools() does internally)
                openai_tool = convert_to_openai_tool(tool)
                if isinstance(openai_tool, dict):
                    serialized = json.dumps(openai_tool, separators=(",", ":"))
                    tokens = count_tokens(serialized)
                    total += tokens
                    tool_counted = True
                    tool_name = openai_tool.get("function", {}).get("name", "?")
                    logger.debug(
                        "Tool %s (converted from BaseTool): %d tokens",
                        tool_name, tokens
                    )
            except Exception as exc:
                tool_name = getattr(tool, "name", "?")
                logger.debug(
                    "Failed to convert BaseTool %s to OpenAI format: %s", tool_name, exc
                )

        # Step 3: Last resort fallback - try to extract schema manually
        # (should rarely be needed if convert_to_openai_tool works)
        if not tool_counted:
            tool_name = getattr(tool, "name", "tool")
            try:
                # Try to get args_schema and serialize it
                args_schema = getattr(tool, "args_schema", None)
                if args_schema is not None:
                    schema_str = ""
                    if hasattr(args_schema, "model_json_schema"):
                        schema_str = json.dumps(
                            args_schema.model_json_schema(), separators=(",", ":")
                        )
                    elif hasattr(args_schema, "schema"):
                        schema_str = json.dumps(
                            args_schema.schema(), separators=(",", ":")
                        )
                    if schema_str:
                        # Count schema + description (minimal but better than estimate)
                        schema_tokens = count_tokens(schema_str)
                        desc = getattr(tool, "description", "")
                        desc_tokens = count_tokens(desc) if desc else 0
                        # Add name tokens and basic structure overhead
                        name_tokens = count_tokens(tool_name)
                        structure_overhead = 30  # Basic JSON structure
                        tokens = (
                            schema_tokens + desc_tokens + name_tokens
                            + structure_overhead
                        )
                        total += tokens
                        tool_counted = True
                        logger.debug(
                            "Tool %s (schema extraction): %d tokens "
                            "(schema:%d + desc:%d + name:%d + struct:%d)",
                            tool_name, tokens, schema_tokens, desc_tokens,
                            name_tokens, structure_overhead
                        )
            except Exception as exc:
                logger.debug(
                    "Failed to extract schema for tool %s: %s", tool_name, exc
                )

        # Step 4: Absolute last resort - estimation (should never happen in practice)
        if not tool_counted:
            tool_name = getattr(tool, "name", "tool")
            tool_desc = getattr(tool, "description", "")
            name_tokens = count_tokens(tool_name)
            desc_tokens = count_tokens(tool_desc)
            # Minimal estimate - this should rarely be used
            estimated_tokens = max(50, name_tokens + desc_tokens) + 80
            total += estimated_tokens
            logger.warning(
                "Tool %s: All serialization methods failed, using fallback estimate: "
                "name(%d) + desc(%d) + overhead(80) = %d tokens",
                tool_name, name_tokens, desc_tokens, estimated_tokens
            )

    # Safety margin is hardcoded to 0, so no addition needed

    logger.debug(
        "compute_overhead_tokens: total=%d for %d tools", total, len(tool_list)
    )

    _OVERHEAD_CACHE[key] = total
    return total


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

