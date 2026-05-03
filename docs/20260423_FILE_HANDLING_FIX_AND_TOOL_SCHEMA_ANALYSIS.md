# File Handling Fix & Tool Schema Analysis

**Date:** 2026-04-23
**Commits:** `5dc2ea4` (fix), `4e9d7bd` (logging cleanup)

## Root Cause: args_schema Silently Strips Injected Agent

### Problem

Uploaded files always returned "File not found" on `main` branch, while `20260414_agentic_skills` branch worked perfectly.

### Root Cause

The `read_text_based_file` tool on `main` used an explicit Pydantic schema:

```python
class ReadTextBasedFileSchema(BaseModel):
    file_reference: str = Field(description="...")
    read_html_as_markdown: bool = Field(default=True, description="...")

@tool(args_schema=ReadTextBasedFileSchema)
def read_text_based_file(file_reference: str, read_html_as_markdown: bool = True, agent=None) -> str:
```

The working branch had:

```python
@tool
def read_text_based_file(file_reference: str, agent=None) -> str:
```

When `native_langchain_streaming.py:872` injects `agent` into tool args:

```python
tool_args_with_agent = {**safe_tool_args, "agent": agent}
```

Pydantic validation filters the input through `ReadTextBasedFileSchema`, which only declares `file_reference` and `read_html_as_markdown`. The `agent` key is **silently dropped** as an unknown field. The function always receives `agent=None`, file registry lookup is skipped, "File not found" every time.

### Fix

Removed `ReadTextBasedFileSchema` class, replaced `@tool(args_schema=ReadTextBasedFileSchema)` with plain `@tool`. LangChain auto-generates the schema from the function signature, which includes `agent=None`, so injection works.

### Verification

Standalone test confirmed the behavior:

```python
# WITH args_schema: agent is silently stripped
tool_args = {"file_reference": "test.html", "agent": fake_agent}
result = tool_with_schema.invoke(tool_args)
# -> "agent=None" (BROKEN)

# WITHOUT args_schema: agent passes through
result = tool_plain.invoke(tool_args)
# -> "agent=FAKE_AGENT_OBJECT" (WORKS)
```

---

## Tool Declaration Patterns in This Codebase

### Two Patterns

| Pattern | Where used | Count | Has `agent=None`? |
|---------|-----------|-------|-------------------|
| `@tool(args_schema=PydanticModel)` | CMW platform tools, image gen, submit | ~42 | No |
| `@tool` (plain) | File tools, search, math, media analysis | ~19 | 8 of 19 do |

### Why Two Patterns

- **CMW platform tools** (`templates_tools/`, `attributes_tools/`, `applications_tools/`): Complex validated parameters (system names, enums, nested configs). `Field(description=...)` provides parameter-level docs to the LLM alongside function docstrings for tool-level docs.
- **File/media tools** (`tools.py`): Simple string/bool params. Need `agent=None` for file registry injection. Plain `@tool` lets LangChain auto-generate the schema from the function signature, preserving `agent`.

### LangChain Description Hierarchy

The LLM sees descriptions from these sources (priority order):

1. `@tool(description="...")` kwarg -- highest priority
2. Function docstring -- standard approach (used by all tools here)
3. Pydantic schema class docstring -- only if no function docstring

For parameter-level descriptions:

- With `args_schema`: `Field(description="...")` values
- With plain `@tool`: extracted from docstring (if `parse_docstring=True`) or auto-generated from type hints

### Constraint: args_schema + agent=None Are Incompatible

Pydantic's default behavior drops unknown fields during validation. If a schema declares fields A and B, passing field C (like `agent`) is silently discarded. This means:

- **Never use `args_schema` on tools that need `agent=None` injection** (unless `agent` is declared in the schema)
- The 8 file tools MUST use plain `@tool`
- The ~42 CMW platform tools can safely use `args_schema` (no agent injection needed)

---

## InjectedToolArg: Future Option

LangChain provides `InjectedToolArg` (available since langchain-core 0.2.x, we have 0.3.79):

```python
from typing import Annotated, Any
from langchain_core.tools import InjectedToolArg

@tool
def read_text_based_file(
    file_reference: str,
    read_html_as_markdown: bool = True,
    agent: Annotated[Any, InjectedToolArg] = None,
) -> str:
```

This annotation tells LangChain to **exclude the parameter from the schema sent to the LLM** while still accepting it at runtime. Benefits:

- Could safely combine `args_schema=` with `agent` injection
- Makes the "hidden from LLM" intent explicit
- Used by LangGraph's ToolNode for automatic injection

### Caveats

- Our custom tool execution loop (`native_langchain_streaming.py:872`) still does manual injection. `InjectedToolArg` controls schema visibility, not automatic injection in custom executors.
- Neither `cmw-platform-agent` nor `cmw-rag` currently use it.
- Only ~2 tools would meaningfully benefit from adding schemas (`understand_video`, `understand_audio` with 5-6 params each).

### Recommendation

Leave as-is. The two patterns are consistent within their groups and don't conflict. If we later need Pydantic schemas on file tools (e.g., for validators or complex params), adopt `InjectedToolArg` at that point.

---

## Tools With agent=None (Complete List)

| Tool | Params | Notes |
|------|--------|-------|
| `read_text_based_file` | `file_reference`, `read_html_as_markdown`, `agent` | The fixed tool |
| `execute_code_multilang` | `code_reference`, `language`, `agent` | |
| `extract_text_from_image` | `file_reference`, `agent` | |
| `analyze_csv_file` | `file_reference`, `query`, `agent` | |
| `analyze_excel_file` | `file_reference`, `query`, `agent` | |
| `analyze_image` | `file_reference`, `agent` | |
| `understand_video` | `file_reference`, `prompt`, `system_prompt`, `agent`, `start_time`, `end_time`, `fps` | Could benefit from schema |
| `understand_audio` | `file_reference`, `prompt`, `system_prompt`, `agent`, `start_time`, `end_time` | Could benefit from schema |

---

## Additional Cleanup (Same Commit Batch)

- `langchain_agent.py`: Replaced `print()` with `logger.debug()`/`logger.warning()` in `register_file()` and `get_file_path()`
- `chat_tab.py`: Replaced nested `hasattr` with `getattr(..., None)` pattern, added `try/except` around file registration, replaced summary `print()` with `logging.debug()`
