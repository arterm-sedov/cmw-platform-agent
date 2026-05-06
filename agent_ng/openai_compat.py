"""OpenAI-shaped HTTP adapter for CMW agent completions (`/v1/agent_completions`)."""

from __future__ import annotations

import copy
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

# Root schema property name: verbatim assistant text before structured formatter.
# Declared as `type: string` in `response_format.json_schema.schema`; stripped from
# the formatter tool and injected into `message.content` JSON (not sent by the model).
CMW_ASSISTANT_LAST_MESSAGE_SCHEMA_KEY = "cmw_assistant_last_message"
# Backward-compatible alias (string value identical).
CMW_PROPRIETARY_ASSISTANT_LAST_MESSAGE = CMW_ASSISTANT_LAST_MESSAGE_SCHEMA_KEY

AGENT_COMPLETIONS_PATH = "/v1/agent_completions"


@dataclass(frozen=True)
class ResolvedChatModel:
    """Resolved provider/model selection for a Chat Completions request."""

    provider: str
    model: str
    model_index: int
    request_model: str


@dataclass(frozen=True)
class StructuredOutputSpec:
    name: str
    schema: dict[str, Any]
    strict: bool


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


def extract_cmw_credentials(extra_body: dict[str, Any] | None) -> dict[str, str]:
    """Extract CMW credential fields from agent extra dict (system message JSON)."""

    if not isinstance(extra_body, dict) or not extra_body:
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


def extract_agent_extra_body_from_messages(
    messages: list[Any],
) -> tuple[dict[str, Any], JSONResponse | None]:
    """Build agent extra fields from `role: system` message content (JSON object).

    Later system messages override earlier ones. Top-level request `extra_body`
    is not used. Non-empty system content that is not a JSON object is rejected.
    """

    extra: dict[str, Any] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        raw = msg.get("content")
        if raw is None:
            continue
        if isinstance(raw, dict):
            extra = dict(raw)
            continue
        if isinstance(raw, list):
            continue
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}, _error_response(
                "system message content must be valid JSON for agent extra fields",
                status_code=400,
            )
        if not isinstance(parsed, dict):
            return {}, _error_response(
                "system message JSON must be an object",
                status_code=400,
            )
        extra = parsed
    return extra, None


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

    message: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content,
    }

    return {
        "id": _response_id(),
        "object": "chat.completion",
        "created": _now(),
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "message": message,
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


def _parse_response_format(
    payload: dict[str, Any],
) -> StructuredOutputSpec | JSONResponse | None:
    raw = payload.get("response_format")
    if raw is None:
        return None

    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return _error_response(
                "response_format must be valid JSON", status_code=400
            )

    if not isinstance(parsed, dict):
        return _error_response("response_format must be an object", status_code=400)

    if parsed.get("type") != "json_schema":
        return _error_response(
            "response_format.type must be 'json_schema'",
            status_code=400,
        )

    json_schema = parsed.get("json_schema")
    if not isinstance(json_schema, dict):
        return _error_response(
            "response_format.json_schema must be an object",
            status_code=400,
        )

    name = str(json_schema.get("name") or "structured_output").strip()
    if not name:
        name = "structured_output"

    schema = json_schema.get("schema")
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            return _error_response(
                "response_format.json_schema.schema must be valid JSON",
                status_code=400,
            )

    if not isinstance(schema, dict):
        return _error_response(
            "response_format.json_schema.schema must be an object",
            status_code=400,
        )

    return StructuredOutputSpec(
        name=name,
        schema=schema,
        strict=bool(json_schema.get("strict", True)),
    )


def _formatter_schema_and_injection_flag(
    root_schema: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Schema for formatter tool + coerce/validate; True if body injection applies.

    When the client adds root property ``cmw_assistant_last_message`` with
    ``type: string`` (or omits ``type``), it is omitted from the tool parameters
    so the model is not prompted to fill it; the HTTP handler injects the raw
    assistant reply into the returned JSON object instead.
    """

    key = CMW_ASSISTANT_LAST_MESSAGE_SCHEMA_KEY
    out = copy.deepcopy(root_schema)
    if out.get("type") != "object":
        return out, False
    props = out.get("properties")
    if not isinstance(props, dict) or key not in props:
        return out, False
    slot = props[key]
    if not isinstance(slot, dict):
        return out, False
    slot_type = slot.get("type")
    if slot_type not in (None, "string"):
        return out, False
    del props[key]
    req = out.get("required")
    if isinstance(req, list) and key in req:
        out["required"] = [r for r in req if r != key]
    return out, True


def _validate_schema_output(
    value: Any, schema: dict[str, Any], path: str = "$"
) -> str | None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                return f"{path}.{key} is required"
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    return f"{path}.{key} is not allowed"
        for key, sub_schema in properties.items():
            if key in value and isinstance(sub_schema, dict):
                err = _validate_schema_output(value[key], sub_schema, f"{path}.{key}")
                if err:
                    return err
        return None

    if expected_type == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                err = _validate_schema_output(item, item_schema, f"{path}[{idx}]")
                if err:
                    return err
        return None

    type_checks: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "null": (type(None),),
    }
    if expected_type in type_checks and not isinstance(
        value, type_checks[expected_type]
    ):
        return f"{path} must be of type {expected_type}"
    return None


def _coerce_schema_output(value: Any, schema: dict[str, Any]) -> Any:
    expected_type = schema.get("type")

    if expected_type == "object":
        if not isinstance(value, dict):
            return value
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return value
        return {
            key: _coerce_schema_output(item, properties[key])
            if key in properties and isinstance(properties[key], dict)
            else item
            for key, item in value.items()
        }

    if expected_type == "array":
        if not isinstance(value, list):
            return value
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return value
        return [_coerce_schema_output(item, item_schema) for item in value]

    if expected_type == "string":
        if isinstance(value, str):
            return value.strip()
        if value is None:
            return value
        return str(value)

    if expected_type == "integer":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if text and text.lstrip("+-").isdigit():
                return int(text)
        return value

    if expected_type == "number":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return value
            try:
                return float(text)
            except ValueError:
                return value
        return value

    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "1", "yes"}:
                return True
            if text in {"false", "0", "no"}:
                return False
        return value

    if expected_type == "null":
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"", "null"}:
                return None
        return value

    return value


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip() if content is not None else ""


def _message_role(message: Any) -> str:
    class_name = message.__class__.__name__.lower()
    if "system" in class_name:
        return "system"
    if "human" in class_name:
        return "user"
    if "tool" in class_name:
        return "tool"
    if "ai" in class_name:
        return "assistant"
    return "unknown"


def _build_formatter_prompt_from_history(
    *,
    history: list[Any],
    user_question: str,
    assistant_content: str,
    tool_name: str,
) -> str:
    system_text = ""
    latest_user = ""
    latest_assistant = ""
    last_turn_tools: list[str] = []
    for msg in history:
        role = _message_role(msg)
        text = _message_text(msg)
        if not text:
            continue
        if role == "system" and not system_text:
            system_text = text
        elif role == "user":
            latest_user = text
        elif role == "assistant":
            latest_assistant = text

    # Collect tool outputs directly preceding the latest assistant response.
    last_assistant_index = -1
    for idx in range(len(history) - 1, -1, -1):
        if _message_role(history[idx]) == "assistant" and _message_text(history[idx]):
            last_assistant_index = idx
            break
    if last_assistant_index > 0:
        idx = last_assistant_index - 1
        while idx >= 0 and _message_role(history[idx]) == "tool":
            text = _message_text(history[idx])
            if text:
                last_turn_tools.append(text)
            idx -= 1
        last_turn_tools.reverse()

    selected_user = latest_user or user_question
    selected_assistant = latest_assistant or assistant_content
    tools_block = "\n".join(last_turn_tools).strip()

    sections = [
        f"Call `{tool_name}` and emit JSON strictly matching the provided schema."
    ]
    if system_text:
        sections.append(f"Root system message:\n{system_text}")
    sections.append(f"Last user message:\n{selected_user}")
    sections.append(f"Last assistant message:\n{selected_assistant}")
    if tools_block:
        sections.append(f"Last turn tool results:\n{tools_block}")
    return "\n\n".join(sections).strip() + "\n"


def _args_from_tool_call(call: Any) -> Any:
    if isinstance(call, dict):
        return call.get("args")
    return getattr(call, "args", None)


def _name_from_tool_call(call: Any) -> str | None:
    if isinstance(call, dict):
        name = call.get("name")
        return str(name) if name else None
    raw = getattr(call, "name", None)
    return str(raw) if raw else None


def _pick_structured_tool_args(tool_calls: list[Any], tool_name: str) -> Any:
    for tc in tool_calls:
        if _name_from_tool_call(tc) == tool_name:
            return _args_from_tool_call(tc)
    return _args_from_tool_call(tool_calls[0])


async def _format_structured_output(
    *,
    agent: Any,
    session_id: str,
    user_question: str,
    assistant_content: str,
    spec: StructuredOutputSpec,
) -> dict[str, Any] | JSONResponse:
    """Return structured dict for assistant ``content`` (caller serializes JSON)."""

    llm = getattr(getattr(agent, "llm_instance", None), "llm", None)
    if llm is None or not hasattr(llm, "bind_tools"):
        return _error_response(
            "Structured output is not available for this model",
            status_code=500,
        )

    stripped, inject_into_content = _formatter_schema_and_injection_flag(spec.schema)
    tool_name = "emit_structured_output"
    schema_description = str(spec.schema.get("description") or "").strip()
    fallback_description = (
        "Fill this schema using the latest user request, "
        "your final answer, "
        "and relevant conversation context. Do not invent facts."
    )
    tool_description = schema_description or fallback_description
    tool_def = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": stripped,
        },
    }
    llm_formatter = llm.bind_tools([tool_def], strict=spec.strict)
    history: list[Any] = []
    get_history = getattr(agent, "get_conversation_history", None)
    if callable(get_history):
        try:
            maybe_history = get_history(session_id)
            if isinstance(maybe_history, list):
                history = maybe_history
        except Exception:
            history = []
    prompt = _build_formatter_prompt_from_history(
        history=history,
        user_question=user_question,
        assistant_content=assistant_content,
        tool_name=tool_name,
    )
    prompt = (
        prompt.rstrip()
        + f"\n\nYou must call the `{tool_name}` tool once with JSON arguments "
        "that match the schema.\n"
    )
    response = await llm_formatter.ainvoke(prompt)
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return _error_response(
            "Structured output tool call was not produced", status_code=422
        )

    args = _pick_structured_tool_args(tool_calls, tool_name)
    if args is None:
        return _error_response(
            "Structured output tool call was not produced", status_code=422
        )

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return _error_response(
                "Structured output tool arguments are invalid JSON",
                status_code=422,
            )
    if not isinstance(args, dict):
        return _error_response(
            "Structured output tool arguments must be an object",
            status_code=422,
        )

    args = _coerce_schema_output(args, stripped)
    validation_error = _validate_schema_output(args, stripped)
    if validation_error:
        return _error_response(
            f"Structured output validation failed: {validation_error}",
            status_code=422,
        )
    if inject_into_content:
        args[CMW_ASSISTANT_LAST_MESSAGE_SCHEMA_KEY] = assistant_content
    return args


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

    extra_body, extra_err = extract_agent_extra_body_from_messages(messages)
    if extra_err is not None:
        return extra_err

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
    credentials = extract_cmw_credentials(extra_body)
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


def register_agent_completions_on_fastapi(fastapi_app: FastAPI, app: Any) -> None:
    """Register `POST /v1/agent_completions` on a FastAPI app."""

    async def agent_completions(request: Request) -> JSONResponse | StreamingResponse:
        provided_api_key = _extract_api_key(request)
        if not provided_api_key:
            return _error_response("Missing bearer token", status_code=401)

        payload = await request.json()
        if not isinstance(payload, dict):
            return _error_response("request body must be a JSON object")
        structured_spec = _parse_response_format(payload)
        if isinstance(structured_spec, JSONResponse):
            return structured_spec
        if structured_spec is not None and payload.get("stream") is True:
            return _error_response(
                "stream=true is not supported with response_format",
                status_code=400,
            )

        prepared = _prepare_request(app, payload, provider_api_key=provided_api_key)
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
        if structured_spec is not None:
            structured_result = await _format_structured_output(
                agent=agent,
                session_id=session_id,
                user_question=question,
                assistant_content=content,
                spec=structured_spec,
            )
            if isinstance(structured_result, JSONResponse):
                return structured_result
            content = json.dumps(structured_result, ensure_ascii=False)
        return JSONResponse(
            content=build_chat_completion_response(
                request_model=resolved.request_model,
                assistant_content=content,
                finish_reason=KNOWN_FINISH_REASON,
            )
        )

    route_paths = {getattr(route, "path", None) for route in fastapi_app.routes}
    if AGENT_COMPLETIONS_PATH not in route_paths:
        fastapi_app.add_api_route(
            AGENT_COMPLETIONS_PATH,
            agent_completions,
            methods=["POST"],
            response_model=None,
        )


def register_agent_completions_route(demo: Any, app: Any) -> None:
    """Register `POST /v1/agent_completions` on Gradio's underlying FastAPI app."""

    fastapi_app = getattr(demo, "app", None)
    if fastapi_app is None:
        message = "Gradio demo does not expose a FastAPI app"
        raise RuntimeError(message)
    register_agent_completions_on_fastapi(fastapi_app, app)


async def handle_agent_completions_payload(
    app: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    """Handle agent completions payload and return OpenAI-shaped JSON.

    Used by Gradio `gr.api` where requests arrive as a plain dict.
    """
    if not isinstance(payload, dict):
        return json.loads(
            _error_response("request body must be a JSON object").body.decode("utf-8")
        )

    structured_spec = _parse_response_format(payload)
    if isinstance(structured_spec, JSONResponse):
        return json.loads(structured_spec.body.decode("utf-8"))
    if structured_spec is not None and payload.get("stream") is True:
        response = _error_response(
            "stream=true is not supported with response_format",
            status_code=400,
        )
        return json.loads(response.body.decode("utf-8"))

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
    if structured_spec is not None:
        structured_result = await _format_structured_output(
            agent=agent,
            session_id=session_id,
            user_question=question,
            assistant_content=content,
            spec=structured_spec,
        )
        if isinstance(structured_result, JSONResponse):
            return json.loads(structured_result.body.decode("utf-8"))
        content = json.dumps(structured_result, ensure_ascii=False)
    return build_chat_completion_response(
        request_model=resolved.request_model,
        assistant_content=content,
        finish_reason=KNOWN_FINISH_REASON,
    )
