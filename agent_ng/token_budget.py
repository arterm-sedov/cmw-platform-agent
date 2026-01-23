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

# Very small cache keyed by system prompt + tool names + safety margin.
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
    system_prompt: str | None,
    tools: Iterable[Any] | None,
    safety_margin: int = DEFAULT_SAFETY_MARGIN,
) -> int:
    """Compute overhead tokens from system prompt + tool schemas + safety margin.

    Uses a tiny cache to avoid recounting stable overhead repeatedly.
    """
    prompt = str(system_prompt or "")
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

    total = count_tokens(prompt)

    for tool in tool_list:
        # If the tool payload is already a dict (e.g., OpenAI/OpenRouter tool spec),
        # count the actual serialized payload. This is closer to what the provider
        # receives than a full pydantic JSON schema dump.
        if isinstance(tool, dict):
            try:
                total += count_tokens(json.dumps(tool, separators=(",", ":")))
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed to serialize bound tool payload for overhead: %s", exc
                )
            continue

        # If tool spec is a pydantic model-like object, try to serialize it.
        if hasattr(tool, "model_dump"):
            try:
                total += count_tokens(
                    json.dumps(tool.model_dump(), separators=(",", ":"))
                )
                continue
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed to serialize tool model_dump for overhead: %s", exc
                )
        if hasattr(tool, "dict"):
            try:
                total += count_tokens(json.dumps(tool.dict(), separators=(",", ":")))
                continue
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed to serialize tool dict() for overhead: %s", exc
                )

        # Count tool schema (prefer pydantic v2, fallback to v1)
        schema_str = ""
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            try:
                if hasattr(args_schema, "model_json_schema"):
                    schema_str = json.dumps(
                        args_schema.model_json_schema(), separators=(",", ":")
                    )
                elif hasattr(args_schema, "schema"):
                    schema_str = json.dumps(
                        args_schema.schema(), separators=(",", ":")
                    )
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed to serialize tool schema for overhead: %s", exc
                )
                schema_str = ""
        if schema_str:
            total += count_tokens(schema_str)

        # Count tool description when present
        desc = getattr(tool, "description", None)
        if desc:
            total += count_tokens(desc)

    try:
        total += int(safety_margin)
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Failed to add safety margin, using default %d: %s",
            DEFAULT_SAFETY_MARGIN,
            exc,
        )
        total += DEFAULT_SAFETY_MARGIN

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
    safety_margin: int = DEFAULT_SAFETY_MARGIN,
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
        # Prefer tool payload actually bound to the underlying LLM, fall back to
        # agent.tools.
        tools_payload = None
        try:
            llm_instance = getattr(agent, "llm_instance", None)
            llm = getattr(llm_instance, "llm", None) if llm_instance else None
            tools_payload = getattr(llm, "tools", None) if llm is not None else None
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to read bound tool payload from LLM: %s", exc
            )
            tools_payload = None

        # Count only tool schemas (since they're always sent to LLM)
        tool_schema_tokens = compute_overhead_tokens(
            system_prompt="",  # Don't count system prompt separately
            tools=tools_payload if tools_payload is not None else getattr(agent, "tools", None),
            safety_margin=0,  # No safety margin
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

