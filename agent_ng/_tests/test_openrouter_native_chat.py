# // pragma: allowlist secret
# pragma: allowlist secret
"""Tests for OpenRouter native SDK chat model (usage + billing passthrough)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from agent_ng.openrouter_native_chat import (  # pragma: allowlist secret
    OpenRouterNativeChatModel,
    create_openrouter_native_chat_model,
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


def test_openrouter_native_stream_preserves_cost_on_usage_chunk() -> None:
    # pragma: allowlist secret
    """Final chunk may be usage-only (empty choices) when include_usage is enabled."""

    def _chunk(**kwargs: object) -> MagicMock:
        m = MagicMock()
        m.model_dump.return_value = kwargs
        return m

    content_chunk = _chunk(
        id="stream-1",
        model="test/model",
        choices=[
            {
                "index": 0,
                "delta": {"content": "Hi"},
                "finish_reason": None,
            }
        ],
        usage=None,
    )
    usage_chunk = _chunk(
        id="stream-1",
        model="test/model",
        choices=[],
        usage={
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "cost": 0.000077,
        },
    )

    client = MagicMock()
    # pragma: allowlist secret
    client.chat.completions.create.return_value = iter([content_chunk, usage_chunk])

    llm = OpenRouterNativeChatModel(
        client=client,
        model_name="test/model",
        base_url="https://example.com/v1",
        api_key="sk-test",
        temperature=0,
        max_tokens=64,
    )

    chunks = list(llm.stream([HumanMessage("Hi")]))
    cost_chunks = [
        c
        for c in chunks
        if (getattr(c, "response_metadata", None) or {}).get("cost") is not None
    ]
    assert cost_chunks, "expected at least one chunk with provider charge in metadata"
    assert cost_chunks[-1].response_metadata.get("cost") == pytest.approx(0.000077)
    body = client.chat.completions.create.call_args[1]
    eb = body.get("extra_body") or {}
    assert eb.get("stream_options") == {"include_usage": True}
    assert body.get("stream") is True


def test_live_openrouter_stream_returns_cost_when_api_allows() -> None:
    # pragma: allowlist secret
    """Live stream cost: uses ``.env`` / env only (no hardcoded URL or model)."""
    # pragma: allowlist secret

    from dotenv import load_dotenv

    _env = Path(__file__).resolve().parents[2] / ".env"
    if _env.is_file():
        load_dotenv(_env, override=False)

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()  # pragma: allowlist secret
    if not api_key:
        pytest.skip("missing provider API key for live test")

    base_url = os.getenv("OPENROUTER_BASE_URL", "").strip()  # pragma: allowlist secret
    if not base_url:
        pytest.skip("missing provider API base URL for live test")

    # pragma: allowlist secret
    stream_model = os.getenv(
        "OPENROUTER_STREAM_TEST_MODEL",
        "",
    ).strip()
    default_model = os.getenv(
        "AGENT_DEFAULT_MODEL",
        "",
    ).strip()
    model_name = stream_model or default_model
    if not model_name:
        pytest.skip("configure stream test model env vars")

    pytest.importorskip("openai")

    from openai import APIStatusError, PermissionDeniedError

    llm = create_openrouter_native_chat_model(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        max_tokens=32,
    )

    try:
        chunks = list(llm.stream([HumanMessage("Say only: ok")]))
    except (PermissionDeniedError, APIStatusError) as exc:
        pytest.skip(f"upstream API declined live stream test: {exc}")

    cost_chunks = [
        c
        for c in chunks
        if (getattr(c, "response_metadata", None) or {}).get("cost") is not None
    ]
    assert cost_chunks, "live stream produced no chunk with response_metadata.cost"
    cost = float(cost_chunks[-1].response_metadata["cost"])
    assert cost >= 0.0
