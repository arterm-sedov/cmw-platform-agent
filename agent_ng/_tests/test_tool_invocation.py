"""Tests for coroutine-only (MCP) tool invocation helpers."""

from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, StructuredTool, tool
from pydantic import BaseModel, Field
import pytest


def _async_only_tool() -> StructuredTool:
    async def _coro(query: str) -> str:
        return f"kb:{query}"

    class _Schema(BaseModel):
        query: str = Field(description="Question")

    return StructuredTool(
        name="comindware_kb_ask_comindware",
        description="Ask the knowledge base",
        args_schema=_Schema,
        coroutine=_coro,
    )


@pytest.mark.asyncio
async def test_ainvoke_agent_tool_runs_coroutine_only_structured_tool() -> None:
    from agent_ng.tool_invocation import ainvoke_agent_tool

    mcp_tool = _async_only_tool()
    with pytest.raises(NotImplementedError, match="sync invocation"):
        mcp_tool.invoke({"query": "x"})

    result = await ainvoke_agent_tool(mcp_tool, {"query": "platform"})
    assert result == "kb:platform"


@pytest.mark.asyncio
async def test_ainvoke_agent_tool_runs_sync_decorated_tool() -> None:
    from agent_ng.tool_invocation import ainvoke_agent_tool

    @tool
    def add(a: int, b: int) -> int:
        """Add numbers."""
        return a + b

    assert await ainvoke_agent_tool(add, {"a": 2, "b": 3}) == 5


def test_invoke_agent_tool_blocking_coroutine_only() -> None:
    from agent_ng.tool_invocation import invoke_agent_tool_blocking

    mcp_tool = _async_only_tool()
    result = invoke_agent_tool_blocking(mcp_tool, {"query": "sync-path"})
    assert result == "kb:sync-path"


def test_tool_requires_async_invocation_detects_mcp_shape() -> None:
    from agent_ng.tool_invocation import tool_requires_async_invocation

    assert tool_requires_async_invocation(_async_only_tool()) is True

    @tool
    def native(x: str) -> str:
        """Native."""
        return x

    assert tool_requires_async_invocation(native) is False


@pytest.mark.asyncio
async def test_streaming_path_uses_ainvoke_helper() -> None:
    """Same helper the streaming loop must call for MCP tools."""
    from agent_ng.tool_invocation import ainvoke_agent_tool

    calls: list[dict[str, Any]] = []

    async def _coro(query: str) -> str:
        calls.append({"query": query})
        return "ok"

    class _Schema(BaseModel):
        query: str = Field()

    mcp_tool = StructuredTool(
        name="mcp_mock",
        description="mock",
        args_schema=_Schema,
        coroutine=_coro,
    )
    sentinel_agent = object()
    tool_args_with_agent = {"query": "test", "agent": sentinel_agent}
    await ainvoke_agent_tool(mcp_tool, tool_args_with_agent)
    assert calls == [{"query": "test"}]


def test_mcp_prepared_args_json_serializable_without_agent() -> None:
    """MCP adapters JSON-serialize tool_input; CmwAgent must not be included."""
    from agent_ng.tool_invocation import prepare_tool_input

    class CmwAgentStub:
        """Non-JSON type like agent_ng.langchain_agent.CmwAgent."""

    payload = {"query": "kb", "agent": CmwAgentStub()}
    prepared = prepare_tool_input(_async_only_tool(), payload)
    assert prepared == {"query": "kb"}
    json.dumps(prepared)


def test_native_tool_invoke_blocking_keeps_agent() -> None:
    from agent_ng.tool_invocation import invoke_agent_tool_blocking

    received: list[Any] = []

    @tool
    def native_with_agent(
        x: str,
        agent: Annotated[Any | None, InjectedToolArg] = None,
    ) -> str:
        """Native tool that accepts injected agent."""
        received.append(agent)
        return x

    sentinel = object()
    result = invoke_agent_tool_blocking(
        native_with_agent, {"x": "hi", "agent": sentinel}
    )
    assert result == "hi"
    assert received == [sentinel]


def test_mcp_invoke_args_built_without_agent_key() -> None:
    """Streaming/memory call sites must not inject agent into MCP tool args."""
    from agent_ng.tool_invocation import tool_requires_async_invocation

    mcp_tool = _async_only_tool()
    safe_tool_args = {"query": "kb"}
    agent = object()
    tool_args_for_invoke = dict(safe_tool_args)
    if not tool_requires_async_invocation(mcp_tool):
        tool_args_for_invoke["agent"] = agent
    assert tool_args_for_invoke == {"query": "kb"}
    assert "agent" not in tool_args_for_invoke


def test_prepare_tool_input_strips_only_for_mcp_shape() -> None:
    from agent_ng.tool_invocation import prepare_tool_input

    payload = {"query": "q", "agent": object()}
    assert prepare_tool_input(_async_only_tool(), payload) == {"query": "q"}

    @tool
    def sync_tool(x: str) -> str:
        """Sync."""
        return x

    assert prepare_tool_input(sync_tool, payload) == payload
