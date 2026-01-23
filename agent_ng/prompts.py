"""Prompts for history compression and other LLM operations."""

import json
import os
from typing import Any


def _load_system_prompt_json() -> dict[str, Any] | None:
    """Load system prompt JSON file"""
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.json")
        if not os.path.exists(prompt_path):
            return None
        with open(prompt_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_critical_system_prompt_context() -> str:
    """
    Extract critical context from system_prompt.json for compression.

    Returns:
        Formatted string with critical system prompt context
    """
    system_prompt_data = _load_system_prompt_json()
    if not system_prompt_data:
        return ""

    context_parts = []

    # Extract critical terminology
    if "cmw_platform_terminology" in system_prompt_data:
        terms = system_prompt_data["cmw_platform_terminology"]
        if "description" in terms and isinstance(terms["description"], list):
            context_parts.append("Terminology: " + "; ".join(terms["description"]))

    # Extract key synonyms
    if "cmw_synonyms" in system_prompt_data:
        synonyms = system_prompt_data["cmw_synonyms"]
        key_synonyms = []
        for key, value in synonyms.items():
            if isinstance(value, list):
                syn_str = f"{key}: {', '.join(value)}"
            else:
                syn_str = f"{key}: {value}"
            key_synonyms.append(syn_str)
        if key_synonyms:
            context_parts.append("Key synonyms: " + "; ".join(key_synonyms))

    return "\n".join(context_parts) if context_parts else ""


def get_history_compression_prompt(target_tokens: int, target_words: int) -> str:
    """
    Get the history compression prompt with system context injection.

    Args:
        target_tokens: Target token count for compressed summary
        target_words: Target word count estimate

    Returns:
        Formatted compression prompt with extracted system context
    """
    system_context = _extract_critical_system_prompt_context()

    # Format system context section
    system_context_section = (
        f"Context:\n{system_context}\n" if system_context else ""
    )

    # Build the prompt template
    prompt_template = """
You are a Comindware Platform Copilot conversation summarizer.
Your goal is to compress the conversation history between Copilot and User
into a concise summary that preserves the key information,
context, and flow of the discussion.

{system_context}

Summarize the conversation history, maintaining:
- Key topics discussed
- Important decisions or conclusions
- User preferences or requirements
- Assistant responses and actions taken
- Any critical context needed for continuation
- Platform-specific terminology and entity references
  (applications, templates, attributes)
- Tool usage patterns and results

Target: ~{target_tokens} tokens (approximately {target_words} words)

Guidelines:
- Preserve technical accuracy and key terminology
- Preserve general ideas from tool results
- Maintain conversation flow and context
- Focus on essential information
- Use clear, structured format
- Use LLM-oriented format without human-oriented formatting bloat
- Preserve platform entity references
  (system names, applications, templates, attributes)
- The most recent messages may be more relevant for current context,
  while older messages can be summarized more concisely but without
  dissolving the context catastrophically
"""

    return prompt_template.format(
        system_context=system_context_section,
        target_tokens=target_tokens,
        target_words=target_words,
    )

