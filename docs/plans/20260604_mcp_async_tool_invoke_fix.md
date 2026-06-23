# MCP async tool invoke fix

**Date:** 2026-06-04  
**Branch:** `20260604_external_mcp_tools`

## Root cause

`langchain-mcp-adapters` builds `StructuredTool(..., coroutine=call_tool)` with no sync `func`. LangChain raises `NotImplementedError: StructuredTool does not support sync invocation.` when `invoke()` is used.

`NativeLangChainStreaming` (async `stream_agent_response`) called `tool_obj.invoke()` at line ~946.

## Fix

1. Add `agent_ng/tool_invocation.py`: detect coroutine-only tools; `ainvoke_agent_tool` for async paths; `invoke_agent_tool_blocking` for sync memory loop.
2. Streaming: `await ainvoke_agent_tool(tool_obj, args)`.
3. `LangChainConversationChain._execute_tool`: use blocking helper.

## Verification

```powershell
.\.venv64\Scripts\python.exe -m pytest agent_ng/_tests/test_tool_invocation.py agent_ng/_tests/test_mcp_tools.py -q
ruff check agent_ng/tool_invocation.py agent_ng/native_langchain_streaming.py agent_ng/langchain_memory.py
```

**App restart:** required after deploy (tool execution is in-process; no hot reload).
