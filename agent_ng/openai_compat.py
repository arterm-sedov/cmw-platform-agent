"""Minimal OpenAI-compatible Chat Completions adapter for the CMW agent."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import TYPE_CHECKING, Any
import uuid

from fastapi import FastAPI, Request  # noqa: TC002
from fastapi.responses import JSONResponse, StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable


KNOWN_FINISH_REASON = "stop"


@dataclass(frozen=True)
class ResolvedChatModel:
    """Resolved provider/model selection for a Chat Completions request."""

    provider: str
    model: str
    model_index: int
    request_model: str


def _response_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _now() -> int:
    return int(time.time())


def resolve_chat_model(
    request_model: str,
    *,
    llm_manager: Any,
    default_provider: str,
) -> ResolvedChatModel:
    """Resolve `<provider>/<model>` or plain `<model>` to a configured model."""

    raw_model = (request_model or "").strip()
    if not raw_model:
        message = "model is required"
        raise ValueError(message)

    provider = default_provider.strip().lower()
    model = raw_model

    if "/" in raw_model:
        maybe_provider, maybe_model = raw_model.split("/", 1)
        if llm_manager.get_provider_config(maybe_provider.strip().lower()):
            provider = maybe_provider.strip().lower()
            model = maybe_model.strip()

    config = llm_manager.get_provider_config(provider)
    if not config or not getattr(config, "models", None):
        message = f"Unknown or unavailable provider: {provider}"
        raise ValueError(message)

    for index, model_config in enumerate(config.models):
        if (model_config.get("model") or "").strip() == model:
            return ResolvedChatModel(
                provider=provider,
                model=model,
                model_index=index,
                request_model=raw_model,
            )

    message = f"Unknown model for provider {provider}: {model}"
    raise ValueError(message)


def extract_cmw_credentials(payload: dict[str, Any]) -> dict[str, str]:
    """Extract non-standard CMW credential fields into session config keys."""

    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, dict):
        return {}

    mapping = {
        "url": ("cmw_base_url",),
        "username": ("cmw_login",),
        "password": ("cmw_password",),
    }
    credentials: dict[str, str] = {}
    for target, keys in mapping.items():
        for key in keys:
            value = extra_body.get(key)
            if isinstance(value, str) and value.strip():
                credentials[target] = value.strip()
                break
    return credentials


def latest_user_message(messages: list[dict[str, Any]]) -> str:
    """Return the latest user-visible text from OpenAI-style messages."""

    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
    message = "messages must include at least one user message"
    raise ValueError(message)


def build_chat_completion_response(
    *,
    request_model: str,
    assistant_content: str,
    finish_reason: str = KNOWN_FINISH_REASON,
) -> dict[str, Any]:
    """Build a minimal OpenAI-compatible non-streaming response."""

    return {
        "id": _response_id(),
        "object": "chat.completion",
        "created": _now(),
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": assistant_content},
                "finish_reason": finish_reason,
            }
        ],
    }


def _chunk_payload(
    *,
    request_model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": _response_id(),
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_streaming_chat_completion_chunks(
    *,
    request_model: str,
    content_parts: Iterable[str],
    finish_reason: str = KNOWN_FINISH_REASON,
) -> Iterable[str]:
    """Build OpenAI-style SSE chunks from text deltas."""

    first = True
    for part in content_parts:
        delta = {"content": part}
        if first:
            delta["role"] = "assistant"
            first = False
        yield _sse(
            _chunk_payload(
                request_model=request_model,
                delta=delta,
                finish_reason=None,
            )
        )
    yield _sse(
        _chunk_payload(
            request_model=request_model,
            delta={},
            finish_reason=finish_reason,
        )
    )
    yield "data: [DONE]\n\n"


def _default_provider() -> str:
    try:
        from agent_ng.agent_config import get_llm_settings

        return str(get_llm_settings().get("default_provider", "openrouter"))
    except Exception:
        return "openrouter"


def _error_response(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


def _extract_api_key(request: Request) -> str:
    """Extract API key from `Authorization: Bearer <token>`."""
    auth_header = request.headers.get("Authorization")
    if isinstance(auth_header, str):
        value = auth_header.strip()
        bearer_prefix = "bearer "
        if value.lower().startswith(bearer_prefix):
            token = value[len(bearer_prefix) :].strip()
            if token:
                return token

    return ""


def _prepare_request(
    app: Any, payload: dict[str, Any], *, provider_api_key: str | None = None
) -> tuple[str, str, Any] | JSONResponse:
    from agent_ng.session_manager import set_current_session_id, set_session_config

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return _error_response("messages must be a list")

    try:
        question = latest_user_message(messages)
        resolved = resolve_chat_model(
            str(payload.get("model") or ""),
            llm_manager=app.llm_manager,
            default_provider=_default_provider(),
        )
    except ValueError as exc:
        return _error_response(str(exc))

    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}

    session_id = str(extra_body.get("session_id") or f"openai_{uuid.uuid4().hex}")
    set_current_session_id(session_id)
    app.set_session_context(session_id)

    if not app.session_manager.update_llm_provider(
        session_id, resolved.provider, resolved.model
    ):
        return _error_response(
            f"Failed to initialize model {resolved.provider}/{resolved.model}",
            status_code=500,
        )

    config_update: dict[str, Any] = {}
    credentials = extract_cmw_credentials(payload)
    if credentials:
        config_update.update(credentials)
    if provider_api_key:
        config_update["llm_provider_api_keys"] = {resolved.provider: provider_api_key}
    if config_update:
        set_session_config(session_id, config_update)

    return question, session_id, resolved


async def _collect_agent_response(agent: Any, question: str, session_id: str) -> str:
    content = ""
    async for event in agent.stream_message(question, session_id):
        if not event:
            continue
        event_type = event.get("type")
        if event_type == "content":
            piece = event.get("content", "")
            if piece:
                content += str(piece)
        elif event_type == "error":
            return str(event.get("content", "Error"))
    return content


async def _stream_agent_response(
    *,
    agent: Any,
    question: str,
    session_id: str,
    request_model: str,
) -> AsyncGenerator[str, None]:
    first = True
    async for event in agent.stream_message(question, session_id):
        if not event:
            continue
        event_type = event.get("type")
        if event_type not in {"content", "error"}:
            continue
        piece = str(event.get("content", ""))
        if not piece:
            continue
        delta = {"content": piece}
        if first:
            delta["role"] = "assistant"
            first = False
        yield _sse(
            _chunk_payload(
                request_model=request_model,
                delta=delta,
                finish_reason=None,
            )
        )
        if event_type == "error":
            break

    yield _sse(
        _chunk_payload(
            request_model=request_model,
            delta={},
            finish_reason=KNOWN_FINISH_REASON,
        )
    )
    yield "data: [DONE]\n\n"


def register_openai_chat_completions_on_fastapi(fastapi_app: FastAPI, app: Any) -> None:
    """Register `POST /v1/chat/completions` on a FastAPI app."""

    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        provided_api_key = _extract_api_key(request)
        if not provided_api_key:
            return _error_response("Missing bearer token", status_code=401)

        payload = await request.json()
        if not isinstance(payload, dict):
            return _error_response("request body must be a JSON object")

        prepared = _prepare_request(
            app, payload, provider_api_key=provided_api_key
        )
        if isinstance(prepared, JSONResponse):
            return prepared

        question, session_id, resolved = prepared
        agent = app.get_user_agent(session_id)
        if payload.get("stream") is True:
            return StreamingResponse(
                _stream_agent_response(
                    agent=agent,
                    question=question,
                    session_id=session_id,
                    request_model=resolved.request_model,
                ),
                media_type="text/event-stream",
            )

        content = await _collect_agent_response(agent, question, session_id)
        return JSONResponse(
            content=build_chat_completion_response(
                request_model=resolved.request_model,
                assistant_content=content,
                finish_reason=KNOWN_FINISH_REASON,
            )
        )

    route_paths = {getattr(route, "path", None) for route in fastapi_app.routes}
    if "/v1/chat/completions" not in route_paths:
        fastapi_app.add_api_route(
            "/v1/chat/completions",
            chat_completions,
            methods=["POST"],
            response_model=None,
        )


def register_openai_chat_completions_route(demo: Any, app: Any) -> None:
    """Register `POST /v1/chat/completions` on Gradio's underlying FastAPI app."""

    fastapi_app = getattr(demo, "app", None)
    if fastapi_app is None:
        message = "Gradio demo does not expose a FastAPI app"
        raise RuntimeError(message)
    register_openai_chat_completions_on_fastapi(fastapi_app, app)


async def handle_chat_completions_payload(
    app: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    """Handle Chat Completions payload and return OpenAI-shaped JSON.

    This helper is used by Gradio `gr.api` wiring where requests arrive as a
    plain Python dict argument rather than a FastAPI Request object.
    """
    if not isinstance(payload, dict):
        return json.loads(
            _error_response("request body must be a JSON object").body.decode("utf-8")
        )

    prepared = _prepare_request(app, payload)
    if isinstance(prepared, JSONResponse):
        return json.loads(prepared.body.decode("utf-8"))

    question, session_id, resolved = prepared
    agent = app.get_user_agent(session_id)

    if payload.get("stream") is True:
        chunks = [
            chunk
            async for chunk in _stream_agent_response(
                agent=agent,
                question=question,
                session_id=session_id,
                request_model=resolved.request_model,
            )
        ]
        return {
            "object": "chat.completion.stream",
            "model": resolved.request_model,
            "chunks": chunks,
        }

    content = await _collect_agent_response(agent, question, session_id)
    return build_chat_completion_response(
        request_model=resolved.request_model,
        assistant_content=content,
        finish_reason=KNOWN_FINISH_REASON,
    )
