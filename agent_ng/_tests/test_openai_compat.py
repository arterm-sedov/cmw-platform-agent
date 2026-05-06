"""Tests for OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_ng.openai_compat import (
    build_chat_completion_response,
    build_streaming_chat_completion_chunks,
    extract_cmw_credentials,
    handle_chat_completions_payload,
    register_openai_chat_completions_route,
    resolve_chat_model,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class _ProviderConfig:
    def __init__(self, models: list[dict[str, str]]) -> None:
        self.models = models


class _LlmManager:
    def __init__(self) -> None:
        self.configs = {
            "polza": _ProviderConfig(
                [{"model": "z-ai/glm-5.1"}, {"model": "openai/gpt-oss-120b"}]
            ),
            "openrouter": _ProviderConfig([{"model": "z-ai/glm-5.1"}]),
        }

    def get_provider_config(self, provider: str) -> _ProviderConfig | None:
        return self.configs.get(provider)


def test_resolve_chat_model_accepts_provider_prefixed_slug() -> None:
    resolved = resolve_chat_model(
        "polza/z-ai/glm-5.1",
        llm_manager=_LlmManager(),
        default_provider="openrouter",
    )

    assert resolved.provider == "polza"
    assert resolved.model == "z-ai/glm-5.1"
    assert resolved.model_index == 0
    assert resolved.request_model == "polza/z-ai/glm-5.1"


def test_resolve_chat_model_uses_default_provider_for_plain_slug() -> None:
    resolved = resolve_chat_model(
        "z-ai/glm-5.1",
        llm_manager=_LlmManager(),
        default_provider="openrouter",
    )

    assert resolved.provider == "openrouter"
    assert resolved.model == "z-ai/glm-5.1"
    assert resolved.model_index == 0


def test_extract_cmw_credentials_reads_snake_case_fields_from_extra_body() -> None:
    assert extract_cmw_credentials(
        {
            "extra_body": {
                "cmw_base_url": " https://example.test ",
                "cmw_login": " user ",
                "cmw_password": " secret ",
            }
        }
    ) == {
        "url": "https://example.test",
        "username": "user",
        "password": "secret",
    }


def test_build_chat_completion_response_matches_basic_openai_shape() -> None:
    response = build_chat_completion_response(
        request_model="polza/z-ai/glm-5.1",
        assistant_content="**Done**",
        finish_reason="stop",
    )

    assert response["object"] == "chat.completion"
    assert response["model"] == "polza/z-ai/glm-5.1"
    assert response["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "**Done**"},
            "finish_reason": "stop",
        }
    ]


def test_build_streaming_chat_completion_chunks_emit_sse_data_and_done() -> None:
    chunks = list(
        build_streaming_chat_completion_chunks(
            request_model="openrouter/z-ai/glm-5.1",
            content_parts=["Hello", " world"],
            finish_reason="stop",
        )
    )

    assert chunks[-1] == "data: [DONE]\n\n"
    first_payload = json.loads(chunks[0].removeprefix("data: ").strip())
    second_payload = json.loads(chunks[1].removeprefix("data: ").strip())
    final_payload = json.loads(chunks[2].removeprefix("data: ").strip())

    assert first_payload["object"] == "chat.completion.chunk"
    assert first_payload["model"] == "openrouter/z-ai/glm-5.1"
    assert first_payload["choices"][0]["delta"] == {
        "role": "assistant",
        "content": "Hello",
    }
    assert second_payload["choices"][0]["delta"] == {"content": " world"}
    assert final_payload["choices"][0]["finish_reason"] == "stop"


class _Demo:
    def __init__(self) -> None:
        self.app = FastAPI()


class _SessionManager:
    def __init__(self) -> None:
        self.updated: tuple[str, str, str] | None = None

    def update_llm_provider(self, session_id: str, provider: str, model: str) -> bool:
        self.updated = (session_id, provider, model)
        return True


class _Agent:
    async def stream_message(
        self, question: str, session_id: str
    ) -> AsyncGenerator[dict[str, object], None]:
        assert session_id
        yield {"type": "content", "content": f"answer to {question}", "metadata": {}}


class _App:
    def __init__(self) -> None:
        self.llm_manager = _LlmManager()
        self.session_manager = _SessionManager()
        self.context_session_id: str | None = None

    def set_session_context(self, session_id: str) -> None:
        self.context_session_id = session_id

    def get_user_agent(self, session_id: str) -> _Agent:
        assert session_id
        return _Agent()


def test_registered_chat_completions_route_returns_non_streaming_response() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        json={
            "model": "polza/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "ping"}],
            "extra_body": {
                "cmw_base_url": "https://example.test",
                "cmw_login": "user",
                "cmw_password": "secret",
                "session_id": "test-session",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "polza/z-ai/glm-5.1"
    assert payload["choices"][0]["message"]["content"] == "answer to ping"
    assert app.session_manager.updated == ("test-session", "polza", "z-ai/glm-5.1")


def test_registered_chat_completions_route_streams_sse_chunks() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert "answer to ping" in response.text


def test_gradio_queue_replaces_app_so_route_must_be_registered_after_queue() -> None:
    demo = _Demo()
    app = _App()

    register_openai_chat_completions_route(demo, app)
    assert "/v1/chat/completions" in {
        getattr(route, "path", None) for route in demo.app.routes
    }

    demo.app = FastAPI()

    assert "/v1/chat/completions" not in {
        getattr(route, "path", None) for route in demo.app.routes
    }

    register_openai_chat_completions_route(demo, app)

    assert "/v1/chat/completions" in {
        getattr(route, "path", None) for route in demo.app.routes
    }


def test_nextgen_app_registers_openai_route_after_queue_configuration() -> None:
    from agent_ng.app_ng_modular import NextGenApp

    source = inspect.getsource(NextGenApp.create_interface)
    queue_pos = source.rfind("self.queue_manager.configure_queue(demo)")
    route_pos = source.rfind("register_openai_chat_completions_route(demo, self)")

    assert queue_pos != -1
    assert route_pos != -1
    assert route_pos > queue_pos


def test_nextgen_app_registers_gradio_chat_completions_api_name() -> None:
    from agent_ng.app_ng_modular import NextGenApp

    source = inspect.getsource(NextGenApp.create_interface)
    assert 'api_name="chat_completions"' in source


def test_handle_chat_completions_payload_returns_openai_json() -> None:
    app = _App()
    payload = {
        "model": "openrouter/z-ai/glm-5.1",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "extra_body": {"session_id": "test-session-plain"},
    }
    response = asyncio.run(handle_chat_completions_payload(app, payload))
    assert response["object"] == "chat.completion"
    assert response["model"] == "openrouter/z-ai/glm-5.1"

