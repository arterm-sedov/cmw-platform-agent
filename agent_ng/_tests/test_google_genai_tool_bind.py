"""OpenAI tool JSON: no model-facing agent param (GenAI Schema)."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_google_genai._function_utils import (
    convert_to_genai_function_declarations,
)

from agent_ng.llm_manager import LLMManager


def _openai_props_for_tool(tool: BaseTool) -> dict:
    spec = convert_to_openai_tool(tool)
    func = spec.get("function") if isinstance(spec, dict) else None
    params = func.get("parameters") if isinstance(func, dict) else None
    props = params.get("properties") if isinstance(params, dict) else None
    return props if isinstance(props, dict) else {}


def test_manager_tools_openai_schema_has_no_agent_property() -> None:
    mgr = LLMManager()
    for tool in mgr.get_tools():
        if not isinstance(tool, BaseTool):
            continue
        props = _openai_props_for_tool(tool)
        assert "agent" not in props, (
            f"Tool {tool.name!r} exposes agent in model schema; "
            "use InjectedToolArg for runtime injection."
        )


def test_convert_to_genai_accepts_full_toolbelt_openai_dicts() -> None:
    """Regression: all app tools must survive GenAI declaration conversion."""
    mgr = LLMManager()
    openai_tools: list[dict] = []
    for t in mgr.get_tools():
        if isinstance(t, BaseTool):
            spec = convert_to_openai_tool(t)
            if isinstance(spec, dict):
                openai_tools.append(spec)
    out = convert_to_genai_function_declarations(openai_tools)
    assert out
    assert out[0].function_declarations
