"""
Invoke LangChain tools from sync and async agent paths.

MCP tools from langchain-mcp-adapters are coroutine-only StructuredTool instances
(no sync ``func``); they must be called via ``ainvoke``, not ``invoke``.
"""

from __future__ import annotations

import asyncio
from typing import Any

# Native-only injectables (InjectedToolArg); omit from MCP JSON payloads.
_RUNTIME_INJECTABLE_KEYS = frozenset({"agent"})


def tool_requires_async_invocation(tool: Any) -> bool:
    """True when the tool is coroutine-only (typical MCP adapter tools)."""
    return (
        getattr(tool, "coroutine", None) is not None
        and getattr(tool, "func", None) is None
    )


def prepare_tool_input(tool: Any, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Drop native-only injectables before MCP/async-only tools are invoked."""
    if not tool_requires_async_invocation(tool):
        return tool_input
    return {k: v for k, v in tool_input.items() if k not in _RUNTIME_INJECTABLE_KEYS}


async def ainvoke_agent_tool(tool: Any, tool_input: dict[str, Any]) -> Any:
    """Run a tool in an async context (streaming and other async callers)."""
    prepared = prepare_tool_input(tool, tool_input)
    if hasattr(tool, "ainvoke"):
        return await tool.ainvoke(prepared)
    if callable(tool):
        result = tool(**prepared)
        if asyncio.iscoroutine(result):
            return await result
        return result
    msg = f"Tool {tool!r} is not invokable"
    raise TypeError(msg)


def invoke_agent_tool_blocking(tool: Any, tool_input: dict[str, Any]) -> Any:
    """Run a tool from a synchronous caller (non-streaming memory loop)."""
    prepared = prepare_tool_input(tool, tool_input)
    if tool_requires_async_invocation(tool) and hasattr(tool, "ainvoke"):
        return asyncio.run(tool.ainvoke(prepared))
    if hasattr(tool, "invoke"):
        return tool.invoke(prepared)
    if callable(tool):
        return tool(**prepared)
    msg = f"Tool {tool!r} is not invokable"
    raise TypeError(msg)
