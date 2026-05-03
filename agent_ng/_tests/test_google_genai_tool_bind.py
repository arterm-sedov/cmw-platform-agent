"""Tests for Gemini bind_tools schema adaptation (OpenAI dict → GenAI Schema)."""

from __future__ import annotations

from langchain_google_genai._function_utils import (
    convert_to_genai_function_declarations,
)

from agent_ng.llm_manager import tools_for_google_genai_bind
from tools.tools import execute_code_multilang, generate_ai_image


def test_tools_for_google_genai_bind_removes_agent_from_openai_parameters() -> None:
    for tool in (generate_ai_image, execute_code_multilang):
        adapted = tools_for_google_genai_bind([tool])[0]
        assert isinstance(adapted, dict)
        props = (adapted.get("function") or {}).get("parameters") or {}
        props = props.get("properties") or {}
        assert "agent" not in props


def test_convert_to_genai_function_declarations_accepts_adapted_tools() -> None:
    bound = tools_for_google_genai_bind([generate_ai_image, execute_code_multilang])
    out = convert_to_genai_function_declarations(bound)
    assert out
    assert out[0].function_declarations


def test_convert_to_genai_accepts_full_toolbelt() -> None:
    """Regression: all app tools must survive GenAI declaration conversion."""
    from agent_ng.llm_manager import LLMManager

    mgr = LLMManager()
    raw = mgr.get_tools()
    adapted = tools_for_google_genai_bind(raw)
    convert_to_genai_function_declarations(adapted)
