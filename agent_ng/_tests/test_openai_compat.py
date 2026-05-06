"""Tests for OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar

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
    history_messages: ClassVar[list[object]] = []

    def __init__(self) -> None:
        self.llm_instance = SimpleNamespace(llm=_FormatterLLM())

    async def stream_message(
        self, question: str, session_id: str
    ) -> AsyncGenerator[dict[str, object], None]:
        assert session_id
        yield {"type": "content", "content": f"answer to {question}", "metadata": {}}

    def get_conversation_history(self, conversation_id: str) -> list[object]:
        assert conversation_id
        return list(_Agent.history_messages)


class _SystemMessage:
    def __init__(self, content: object) -> None:
        self.content = content


class _HumanMessage:
    def __init__(self, content: object) -> None:
        self.content = content


class _AIMessage:
    def __init__(self, content: object) -> None:
        self.content = content


class _ToolMessage:
    def __init__(self, content: object) -> None:
        self.content = content


class _FormatterBoundLLM:
    last_prompt: str = ""
    next_args: object = {"objects": [{"id": "obj.1", "system_name": "TestObject"}]}

    async def ainvoke(self, messages: list[object]) -> object:
        assert messages
        _FormatterBoundLLM.last_prompt = str(messages)
        return SimpleNamespace(
            tool_calls=[
                {
                    "id": "tool-call-1",
                    "name": "emit_structured_output",
                    "args": _FormatterBoundLLM.next_args,
                }
            ]
        )


class _FormatterLLM:
    last_tool_description: str = ""

    def bind_tools(
        self,
        tools: list[dict[str, object]],
        tool_choice: object = None,
        strict: object = None,
    ) -> _FormatterBoundLLM:
        assert tools
        assert tool_choice is None
        assert strict is True
        _FormatterLLM.last_tool_description = str(
            tools[0]["function"].get("description", "")
        )
        return _FormatterBoundLLM()


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
        headers={"Authorization": "Bearer provider-key-polza"},
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
        headers={"Authorization": "Bearer provider-key-openrouter"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert "answer to ping" in response.text


def test_registered_chat_completions_route_requires_bearer_token() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)
    client = TestClient(demo.app)

    unauthorized = client.post(
        "/v1/chat/completions",
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["message"] == "Missing bearer token"


def test_registered_chat_completions_route_accepts_any_bearer_token_as_provider_key() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)
    client = TestClient(demo.app)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer any-client-supplied-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert response.status_code == 200


def test_registered_chat_completions_route_does_not_accept_x_api_key_header() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)
    client = TestClient(demo.app)

    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": "secret-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert response.status_code == 401


def test_registered_chat_completions_route_passes_bearer_as_selected_provider_key() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer provider-specific-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "ping"}],
            "extra_body": {"session_id": "provider-key-session"},
        },
    )

    assert response.status_code == 200
    from agent_ng.session_manager import get_session_config

    session_config = get_session_config("provider-key-session") or {}
    assert session_config.get("llm_provider_api_keys") == {
        "openrouter": "provider-specific-key"
    }


def test_registered_chat_completions_route_formats_structured_output_via_tool_call() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer provider-specific-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "list worked objects"}],
            "response_format": json.dumps(
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "worked_objects",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "objects": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "system_name": {"type": "string"},
                                        },
                                        "required": ["id", "system_name"],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["objects"],
                            "additionalProperties": False,
                        },
                    },
                }
            ),
        },
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert json.loads(content) == {
        "objects": [{"id": "obj.1", "system_name": "TestObject"}]
    }
    assert "emit_structured_output" in _FormatterBoundLLM.last_prompt


def test_structured_formatter_uses_compact_context_from_session_history() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)
    _Agent.history_messages = [
        _SystemMessage("Root system policy"),
        _HumanMessage("Old user question"),
        _AIMessage("Old assistant response"),
        _HumanMessage("Latest user question from history"),
        _AIMessage(""),
        _ToolMessage('{"id":"obj.2"}'),
        _ToolMessage('{"system_name":"Object2"}'),
        _AIMessage("Latest assistant response from history"),
    ]

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer provider-specific-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "fresh question"}],
            "extra_body": {"session_id": "history-session"},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "worked_objects",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"objects": {"type": "array"}},
                        "required": ["objects"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    assert response.status_code == 200
    prompt = _FormatterBoundLLM.last_prompt
    assert "Root system policy" in prompt
    assert "Latest user question from history" in prompt
    assert "Latest assistant response from history" in prompt
    assert '{"id":"obj.2"}' in prompt
    assert "Old user question" not in prompt
    assert "Old assistant response" not in prompt


def test_structured_output_uses_root_schema_description_when_present() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer provider-specific-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "list worked objects"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "worked_objects",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "description": "Use only facts from final answer and conversation",
                        "properties": {"objects": {"type": "array"}},
                        "required": ["objects"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    assert response.status_code == 200
    assert (
        _FormatterLLM.last_tool_description
        == "Use only facts from final answer and conversation"
    )


def test_structured_output_falls_back_to_default_tool_description() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer provider-specific-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "list worked objects"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "worked_objects",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"objects": {"type": "array"}},
                        "required": ["objects"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    assert response.status_code == 200
    assert "Do not invent facts." in _FormatterLLM.last_tool_description


def test_structured_output_repairs_common_primitive_types() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)
    _FormatterBoundLLM.next_args = {
        "count": "42",
        "is_valid": "TRUE",
        "score": "3.5",
        "name": "  Alice  ",
    }

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer provider-specific-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "format output"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "coerce_case",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "is_valid": {"type": "boolean"},
                            "score": {"type": "number"},
                            "name": {"type": "string"},
                        },
                        "required": ["count", "is_valid", "score", "name"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    assert response.status_code == 200
    assert json.loads(response.json()["choices"][0]["message"]["content"]) == {
        "count": 42,
        "is_valid": True,
        "score": 3.5,
        "name": "Alice",
    }


def test_structured_output_repairs_empty_string_to_null_only_for_null_type() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)
    _FormatterBoundLLM.next_args = {"note": ""}

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer provider-specific-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "format output"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "null_case",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"note": {"type": "null"}},
                        "required": ["note"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    assert response.status_code == 200
    assert json.loads(response.json()["choices"][0]["message"]["content"]) == {
        "note": None
    }


def test_structured_output_does_not_repair_empty_string_for_integer() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)
    _FormatterBoundLLM.next_args = {"count": ""}

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer provider-specific-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "format output"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "int_case",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    assert response.status_code == 422


def test_registered_chat_completions_route_rejects_invalid_response_format_json() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer provider-specific-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": "{not-json}",
        },
    )

    assert response.status_code == 400
    assert "response_format" in response.json()["error"]["message"]


def test_registered_chat_completions_route_rejects_stream_with_structured_output() -> None:
    demo = _Demo()
    app = _App()
    register_openai_chat_completions_route(demo, app)

    response = TestClient(demo.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer provider-specific-key"},
        json={
            "model": "openrouter/z-ai/glm-5.1",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "worked_objects",
                    "schema": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
        },
    )

    assert response.status_code == 400
    assert "stream" in response.json()["error"]["message"]


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

