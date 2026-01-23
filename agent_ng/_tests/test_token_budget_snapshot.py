"""
Test token budget snapshot helpers
=================================

Focus: non-breaking, pure budget snapshots computed at budget moments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent_ng.token_budget import compute_token_budget_snapshot
from agent_ng.token_counter import ConversationTokenTracker, TokenCount


@dataclass
class _DummyLLMInstance:
    config: dict[str, Any]


class _DummyMemoryManager:
    def __init__(self, messages):
        self._messages = list(messages)

    def get_conversation_history(self, conversation_id: str = "default"):
        return list(self._messages)


class _DummyToolSchema:
    @staticmethod
    def model_json_schema():
        return {"type": "object", "properties": {"x": {"type": "string"}}}


class _DummyTool:
    name = "dummy_tool"
    description = "Dummy tool for tests."
    args_schema = _DummyToolSchema


class _DummyAgent:
    def __init__(self, messages):
        self.system_prompt = "You are a helpful assistant."
        self.memory_manager = _DummyMemoryManager(messages)
        self.llm_instance = _DummyLLMInstance(config={"token_limit": 1000})
        self.tools = [_DummyTool()]


def test_budget_snapshot_counts_tool_messages_and_conversation_messages():
    agent = _DummyAgent(
        [
            SystemMessage(content="System"),
            HumanMessage(content="Hello"),
            ToolMessage(content='{"result":"A"}', tool_call_id="t1", name="tool"),
            ToolMessage(content='{"result":"A"}', tool_call_id="t2", name="tool"),
        ]
    )

    snap = compute_token_budget_snapshot(agent=agent, conversation_id="default")

    assert snap["context_window"] == 1000
    assert snap["conversation_tokens"] > 0
    assert snap["tool_tokens"] > 0
    # No deduplication: two tool messages should be counted (tool_tokens should be >= single-tool tokens)
    single_tool = compute_token_budget_snapshot(
        agent=_DummyAgent(
            [
                SystemMessage(content="System"),
                HumanMessage(content="Hello"),
                ToolMessage(content='{"result":"A"}', tool_call_id="t1", name="tool"),
            ]
        ),
        conversation_id="default",
    )["tool_tokens"]
    assert snap["tool_tokens"] >= single_tool


def test_budget_snapshot_includes_overhead_tokens():
    agent = _DummyAgent([HumanMessage(content="Hello")])
    snap = compute_token_budget_snapshot(agent=agent, conversation_id="default", include_overhead=True)
    assert snap["overhead_tokens"] > 0
    assert snap["total_tokens"] >= snap["overhead_tokens"]


def test_token_budget_info_prefers_provider_usage_when_available():
    tracker = ConversationTokenTracker()
    # Store a computed snapshot (would otherwise be used)
    tracker.set_budget_snapshot(
        {
            "total_tokens": 999,
            "ts": 0.0,
        }
    )

    # Simulate provider usage (ground truth)
    tracker._last_api_tokens = TokenCount(10, 5, 15, False, "api")  # type: ignore[attr-defined]

    info = tracker.get_token_budget_info(1000)
    assert info["used_tokens"] == 15


if __name__ == "__main__":
    pytest.main([__file__])

