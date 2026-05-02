# // pragma: allowlist secret
# pragma: allowlist secret
"""Tests for OpenRouter native SDK chat model (usage + billing passthrough)."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from agent_ng.openrouter_native_chat import (
    OpenRouterNativeChatModel,
    usage_dict_from_sdk,
)
from agent_ng.openrouter_usage_accounting import (
    normalize_openrouter_usage,
    summarize_usage_for_logging,
)


def test_usage_dict_from_sdk_extracts_cost() -> None:
    usage = MagicMock()
    usage.model_dump.return_value = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost": 0.001,
        "cost_details": {"upstream_inference_cost": 0.0009},
    }
    d = usage_dict_from_sdk(usage)
    assert d["cost"] == 0.001
    assert d["prompt_tokens"] == 10


def test_normalize_openrouter_usage_flatten() -> None:
    raw = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost": 0.002,
        "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 1},
        "completion_tokens_details": {"reasoning_tokens": 2},
        "cost_details": {"upstream_inference_cost": 0.001},
    }
    n = normalize_openrouter_usage(raw)
    assert n["cost"] == pytest.approx(0.002)
    assert n["upstream_cost"] == pytest.approx(0.001)
    assert n["cached_tokens"] == 3.0


def test_summarize_usage_for_logging_stable_keys() -> None:
    s = summarize_usage_for_logging({"prompt_tokens": 1, "cost": 0.5})
    assert "cost" in s
    assert "upstream_cost" in s


@pytest.fixture
def mock_completion_response() -> MagicMock:
    msg = MagicMock()
    msg.model_dump.return_value = {"content": "OK", "tool_calls": []}

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    resp = MagicMock()
    resp.id = "gen-test"
    resp.model = "test/model"
    resp.choices = [choice]
    resp.usage = MagicMock()
    resp.usage.model_dump.return_value = {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "cost": 0.00042,
    }
    return resp


def test_openrouter_native_invoke_maps_usage_and_cost(
    mock_completion_response: MagicMock,
) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = mock_completion_response

    llm = OpenRouterNativeChatModel(
        client=client,
        model_name="test/model",
        base_url="https://example.com/v1",
        api_key="sk-test",
        temperature=0,
        max_tokens=64,
    )

    out = llm.invoke([HumanMessage("Hi")])
    assert isinstance(out, AIMessage)
    assert out.usage_metadata is not None
    assert out.usage_metadata["total_tokens"] == 10
    rm = out.response_metadata or {}
    assert rm.get("cost") == 0.00042
    assert isinstance(rm.get("token_usage"), dict)
    assert rm["token_usage"]["cost"] == 0.00042


def test_usage_accounting_callback_accumulates(
    mock_completion_response: MagicMock,
) -> None:
    from agent_ng.openrouter_usage_accounting import OpenRouterUsageAccountingCallback

    client = MagicMock()
    client.chat.completions.create.return_value = mock_completion_response

    cb = OpenRouterUsageAccountingCallback()
    llm = OpenRouterNativeChatModel(
        client=client,
        model_name="test/model",
        base_url="https://example.com/v1",
        api_key="sk-test",
        temperature=0,
        max_tokens=64,
        callbacks=[cb],
    )

    _ = llm.invoke([HumanMessage("Hi")])

    snap = cb.flush_turn_summary()
    assert snap is not None
    assert snap["cost"] == pytest.approx(0.00042)


def test_usage_metadata_callback_tracks_ai_message_not_llm_result() -> None:
    """Regression: on_llm_end receives LLMResult; tracker needs AIMessage."""
    from langchain_core.outputs import ChatGeneration, ChatResult

    from agent_ng.token_counter import (
        ConversationTokenTracker,
        UsageMetadataCallbackHandler,
    )

    ai = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        response_metadata={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
                "cost": 0.01,
            }
        },
    )
    gen = ChatGeneration(message=ai)
    result = ChatResult(generations=[gen], llm_output={})

    tracker = ConversationTokenTracker()
    handler = UsageMetadataCallbackHandler(tracker)
    handler.on_llm_end(result)
    last = getattr(tracker, "_last_api_tokens", None)
    assert last is not None
    assert last.cost == pytest.approx(0.01)
