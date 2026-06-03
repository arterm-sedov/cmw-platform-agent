"""Unit tests for OpenAI-shaped `usage` on agent completions."""

from __future__ import annotations

from agent_ng.openai_compat import (
    build_chat_completion_response,
    merge_openai_usage,
    usage_from_token_count,
)
from agent_ng.token_counter import TokenCount


def test_usage_from_token_count_maps_openai_fields() -> None:
    tc = TokenCount(10, 20, 30, cost=0.0012)
    u = usage_from_token_count(tc)
    assert u is not None
    assert u["prompt_tokens"] == 10
    assert u["completion_tokens"] == 20
    assert u["total_tokens"] == 30
    assert u["cost"] == 0.0012


def test_usage_from_token_count_omits_cost_when_unknown() -> None:
    tc = TokenCount(1, 2, 3, cost=None)
    u = usage_from_token_count(tc)
    assert u is not None
    assert "cost" not in u


def test_usage_from_token_none() -> None:
    assert usage_from_token_count(None) is None


def test_merge_openai_usage_sums_parts() -> None:
    a = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cost": 0.01,
    }
    b = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost": 0.02,
    }
    m = merge_openai_usage(a, b)
    assert m is not None
    assert m["prompt_tokens"] == 110
    assert m["completion_tokens"] == 55
    assert m["total_tokens"] == 165
    assert abs(m["cost"] - 0.03) < 1e-9


def test_merge_openai_usage_skips_nones() -> None:
    one = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert merge_openai_usage(None, one) == one
    assert merge_openai_usage(one, None) == one
    assert merge_openai_usage(None, None) is None


def test_build_chat_completion_response_includes_usage() -> None:
    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "cost": 0.0}
    r = build_chat_completion_response(
        request_model="openrouter/x",
        assistant_content="hi",
        usage=usage,
    )
    assert r["usage"] == usage
    assert r["choices"][0]["message"]["content"] == "hi"


def test_build_chat_completion_response_without_usage_key_when_absent() -> None:
    r = build_chat_completion_response(
        request_model="m", assistant_content="x", finish_reason="stop"
    )
    assert "usage" not in r
