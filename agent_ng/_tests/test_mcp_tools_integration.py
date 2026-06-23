"""Integration tests for remote HTTP MCP (gated; no CI by default)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

pytestmark = pytest.mark.integration


def _integration_enabled() -> bool:
    return os.getenv("CMW_MCP_INTEGRATION_TESTS", "").strip() == "1"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _integration_enabled(),
    reason="Set CMW_MCP_INTEGRATION_TESTS=1 to run live MCP HTTP tests",
)
async def test_ennoia_mcp_lists_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_ng.mcp_tools import fetch_mcp_tools_async, reset_mcp_tools_cache

    reset_mcp_tools_cache()
    monkeypatch.setenv("CMW_MCP_ENABLED", "true")
    monkeypatch.setenv("CMW_MCP_TOOL_NAME_PREFIX", "true")
    monkeypatch.setenv("CMW_MCP_ALLOWED_SERVERS", "comindware_kb")
    monkeypatch.delenv("CMW_MCP_ALLOWED_HOSTS", raising=False)

    tools = await fetch_mcp_tools_async()
    assert len(tools) >= 1
    names = {getattr(t, "name", "") for t in tools}
    assert any("ask_comindware" in n or "comindware_kb" in n for n in names), names
