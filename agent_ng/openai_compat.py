"""OpenAI-shaped HTTP adapter for CMW agent completions (`/api/v1/chat/completions`)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any
import uuid

from fastapi import FastAPI, Request  # noqa: TC002
from fastapi.responses import JSONResponse, StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable

    from agent_ng.token_counter import TokenCount


KNOWN_FINISH_REASON = "stop"

# Root schema property name: verbatim assistant text before structured formatter.
# Declared as `type: string` in `response_format.json_schema.schema`; stripped from
# the formatter tool and injected into `message.content` JSON (not sent by the model).
CMW_ASSISTANT_LAST_MESSAGE_SCHEMA_KEY = "cmw_assistant_last_message"
# Backward-compatible alias (string value identical).
CMW_PROPRIETARY_ASSISTANT_LAST_MESSAGE = CMW_ASSISTANT_LAST_MESSAGE_SCHEMA_KEY
CMW_EXTRA_BODY_KEY = "cmw_extra_body"

AGENT_COMPLETIONS_PATH = "/api/v1/chat/completions"

from agent_ng.logging_config import setup_openapi_debug_log  # noqa: E402

_debug_handler = setup_openapi_debug_log()

_REDACT_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "token",
    "password",
    "cmw_password",
    "llm_provider_api_keys",
}


def _mask_secret(value: str) -> str:
    text = value.strip()
    if len(text) <= 10:
        return "***"
    return f"{text[:6]}...{text[-4:]}"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k).lower()
            if key in _REDACT_KEYS:
                if isinstance(v, dict):
                    out[k] = dict.fromkeys(v, "***")
                elif isinstance(v, str):
                    out[k] = _mask_secret(v)
                else:
                    out[k] = "***"
                continue
            out[k] = _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _debug_log_io(event: str, payload: dict[str, Any]) -> None:
    if _debug_handler is None:
        return
    try:
        record = {
            "ts": int(time.time()),
            "event": event,
            "path": AGENT_COMPLETIONS_PATH,
            "payload": _redact(payload),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        _debug_handler.emit(
            logging.LogRecord("_", logging.DEBUG, "", 0, line, (), None)
        )
    except Exception:
        return
    try:
        record = {
            "ts": int(time.time()),
            "event": event,
            "path": AGENT_COMPLETIONS_PATH,
            "payload": _redact(payload),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        _debug_handler.emit(
            logging.LogRecord("_", logging.DEBUG, "", 0, line, (), None)
        )
    except Exception:
        return


def _json_response_content(response: JSONResponse) -> dict[str, Any]:
    try:
        return json.loads(response.body.decode("utf-8"))
    except Exception:
        return {"raw_body": str(response.body)}


def usage_from_token_count(tc: TokenCount | None) -> dict[str, Any] | None:
    """Map ``TokenCount`` to OpenAI-style ``usage`` (optional ``cost``)."""

    if tc is None:
        return None
    out: dict[str, Any] = {
        "prompt_tokens": int(tc.input_tokens),
        "completion_tokens": int(tc.output_tokens),
        "total_tokens": int(tc.total_tokens),
    }
    if tc.cost is not None:
        out["cost"] = float(tc.cost)
    return out


def merge_openai_usage(
    *parts: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Sum token fields and costs (e.g. main agent turn + structured formatter call)."""

    acc_p = acc_c = acc_t = 0
    cost_sum = 0.0
    has_cost = False
    saw = False
    for p in parts:
        if not p:
            continue
        saw = True
        acc_p += int(p.get("prompt_tokens", 0) or 0)
        acc_c += int(p.get("completion_tokens", 0) or 0)
        acc_t += int(p.get("total_tokens", 0) or 0)
        raw = p.get("cost")
        if raw is not None:
            cost_sum += float(raw)
            has_cost = True
    if not saw:
        return None
    if acc_p == 0 and acc_c == 0 and acc_t == 0 and not has_cost:
        return None
    out: dict[str, Any] = {
        "prompt_tokens": acc_p,
        "completion_tokens": acc_c,
        "total_tokens": acc_t,
    }
    if has_cost:
        out["cost"] = cost_sum
    return out


def _completion_usage(
    agent: Any, *, pre_structured: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    getter = getattr(agent, "get_last_api_tokens", None)
    last_tc: TokenCount | None = getter() if callable(getter) else None
    return merge_openai_usage(pre_structured, usage_from_token_count(last_tc))


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
    """Build agent extra fields from `role: system` content via `cmw_extra_body`.

    Later system messages override earlier ones. Top-level request `extra_body`
    is not used.
    """

    extra: dict[str, Any] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        raw = msg.get("content")
        if raw is None:
            continue
        parsed_wrapper: dict[str, Any] | None = None
        parse_error: JSONResponse | None = None
        if isinstance(raw, dict):
            parsed_wrapper, parse_error = _extract_cmw_extra_body_from_obj(raw)
        elif isinstance(raw, list):
            continue
        elif isinstance(raw, str):
            parsed_wrapper, parse_error = _extract_cmw_extra_body_from_text(raw)
        else:
            continue
        if parse_error is not None:
            return {}, parse_error
        if parsed_wrapper is not None:
            extra = parsed_wrapper
    return extra, None


def _extract_cmw_extra_body_from_obj(
    parsed: dict[str, Any],
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    if CMW_EXTRA_BODY_KEY not in parsed:
        return None, None
    body = parsed.get(CMW_EXTRA_BODY_KEY)
    if not isinstance(body, dict):
        return None, _error_response(
            "system message cmw_extra_body must be a JSON object",
            status_code=400,
        )
    return dict(body), None


def _extract_json_object_candidates(text: str) -> list[str]:
    """Extract top-level JSON object candidates from free-form text."""
    candidates: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : idx + 1])
                start = None
    return candidates


def _extract_cmw_extra_body_from_text(
    text: str,
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    stripped = text.strip()
    if not stripped:
        return None, None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return _extract_cmw_extra_body_from_obj(parsed)
    for candidate in _extract_json_object_candidates(stripped):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        wrapped, err = _extract_cmw_extra_body_from_obj(obj)
        if err is not None:
            return None, err
        if wrapped is not None:
            return wrapped, None
    return None, None


def _is_standard_json_schema(schema: dict[str, Any]) -> bool:
    if not isinstance(schema, dict):
        return False
    if "type" in schema:
        return True
    return any(
        key in schema for key in ("properties", "items", "required", "allOf", "oneOf")
    )


def _extract_schema_from_injected_system_text(
    messages: list[Any],
) -> dict[str, Any] | None:
    marker = "Target schema (JSON Schema):"
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        raw = msg.get("content")
        if not isinstance(raw, str):
            continue
        idx = raw.find(marker)
        if idx < 0:
            continue
        tail = raw[idx + len(marker) :]
        for candidate in _extract_json_object_candidates(tail):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and _is_standard_json_schema(parsed):
                return parsed
    return None


def _convert_custom_schema_node(node: dict[str, Any]) -> dict[str, Any]:
    raw_type = str(node.get("Type") or "String").strip().lower()
    is_list = bool(node.get("IsList"))
    attrs = node.get("Attributes")
    desc = node.get("Description")
    type_map: dict[str, str] = {
        "string": "string",
        "text": "string",
        "boolean": "boolean",
        "number": "number",
        "decimal": "number",
        "integer": "integer",
        "datetime": "string",
        "timespan": "string",
        "dynamic": "object",
    }
    if raw_type == "complex":
        out: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        required: list[str] = []
        if isinstance(attrs, list):
            for attr in attrs:
                if not isinstance(attr, dict):
                    continue
                name = str(attr.get("Name") or "").strip()
                if not name:
                    continue
                child = _convert_custom_schema_node(attr)
                child_desc = attr.get("Description")
                if isinstance(child_desc, str) and child_desc.strip():
                    child["description"] = child_desc
                out["properties"][name] = child
                if bool(attr.get("Required")):
                    required.append(name)
        if required:
            out["required"] = required
    else:
        out = {"type": type_map.get(raw_type, "string")}
        if raw_type == "datetime":
            out["format"] = "date-time"
    if isinstance(desc, str) and desc.strip():
        out["description"] = desc
    if is_list:
        return {"type": "array", "items": out}
    return out


def _normalize_structured_schema(
    schema: dict[str, Any], messages: list[Any]
) -> tuple[dict[str, Any], JSONResponse | None]:
    if _is_standard_json_schema(schema):
        return schema, None
    injected = _extract_schema_from_injected_system_text(messages)
    if isinstance(injected, dict):
        return injected, None
    if "Type" in schema:
        converted = _convert_custom_schema_node(schema)
        if _is_standard_json_schema(converted):
            return converted, None
    return {}, _error_response(
        (
            "response_format.json_schema.schema must be standard JSON Schema "
            "or convertible"
        ),
        status_code=400,
    )


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
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal OpenAI-compatible non-streaming response."""

    message: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content,
    }

    out: dict[str, Any] = {
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
    if usage is not None:
        out["usage"] = usage
    return out


def _chunk_payload(
    *,
    request_model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
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
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_streaming_chat_completion_chunks(
    *,
    request_model: str,
    content_parts: Iterable[str],
    finish_reason: str = KNOWN_FINISH_REASON,
    usage: dict[str, Any] | None = None,
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
            usage=usage,
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

    messages = payload.get("messages")
    schema_messages = messages if isinstance(messages, list) else []
    normalized_schema, schema_err = _normalize_structured_schema(
        schema, schema_messages
    )
    if schema_err is not None:
        return schema_err

    return StructuredOutputSpec(
        name=name,
        schema=normalized_schema,
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
    if value is None:
        return None if expected_type in (None, "null") else f"{path} must not be null"
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


def _repair_schema_output(value: Any, schema: dict[str, Any]) -> Any:
    """Best-effort schema repair for structured formatter outputs."""
    return _repair_schema_output_impl(value, schema, required=True)


_DROP = object()


def _default_for_schema(schema: dict[str, Any]) -> Any:
    expected_type = schema.get("type")
    if expected_type == "object":
        return {}
    if expected_type == "array":
        return []
    if expected_type == "string":
        return ""
    if expected_type == "integer":
        return 0
    if expected_type == "number":
        return 0.0
    if expected_type == "boolean":
        return False
    if expected_type == "null":
        return None
    return None


def _repair_schema_output_impl(
    value: Any, schema: dict[str, Any], *, required: bool
) -> Any:
    """Repair schema output; return `_DROP` for optional unrecoverable values."""
    expected_type = schema.get("type")

    if expected_type == "object":
        source = value if isinstance(value, dict) else {}
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        required_keys = schema.get("required")
        if not isinstance(required_keys, list):
            required_keys = []
        repaired: dict[str, Any] = {}
        if schema.get("additionalProperties") is False:
            for key, sub_schema in properties.items():
                if key in source:
                    if isinstance(sub_schema, dict):
                        is_required = key in required_keys
                        child = _repair_schema_output_impl(
                            source[key], sub_schema, required=is_required
                        )
                        if child is not _DROP:
                            repaired[key] = child
                    else:
                        repaired[key] = source[key]
        else:
            for key, item in source.items():
                sub_schema = properties.get(key)
                if isinstance(sub_schema, dict):
                    is_required = key in required_keys
                    child = _repair_schema_output_impl(
                        item, sub_schema, required=is_required
                    )
                    if child is not _DROP:
                        repaired[key] = child
                else:
                    repaired[key] = item
        if required_keys:
            for key in required_keys:
                if isinstance(key, str) and key not in repaired:
                    sub_schema = properties.get(key)
                    repaired[key] = (
                        _default_for_schema(sub_schema)
                        if isinstance(sub_schema, dict)
                        else None
                    )
        return repaired

    if expected_type == "array":
        item_schema = schema.get("items")
        if not isinstance(value, list):
            return []
        if not isinstance(item_schema, dict):
            return value
        repaired_items: list[Any] = []
        for item in value:
            child = _repair_schema_output_impl(item, item_schema, required=False)
            if child is _DROP:
                continue
            repaired_items.append(child)
        return repaired_items

    coerced = _coerce_schema_output(value, schema)
    validation_error = _validate_schema_output(coerced, schema)
    if validation_error:
        if required:
            return _default_for_schema(schema)
        return _DROP
    return coerced


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
    _debug_log_io(
        "structured.formatter.request",
        {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_description": tool_description,
            "schema": stripped,
        },
    )
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
            candidates = _extract_json_object_candidates(args)
            recovered: dict[str, Any] | None = None
            for candidate in candidates:
                try:
                    parsed_candidate = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed_candidate, dict):
                    recovered = parsed_candidate
                    break
            if recovered is None:
                return _error_response(
                    "Structured output tool arguments are invalid JSON",
                    status_code=422,
                )
            args = recovered
    if not isinstance(args, dict):
        return _error_response(
            "Structured output tool arguments must be an object",
            status_code=422,
        )

    args = _repair_schema_output(args, stripped)
    validation_error = _validate_schema_output(args, stripped)
    if validation_error:
        _debug_log_io(
            "structured.formatter.validation_error",
            {"session_id": session_id, "error": validation_error, "args": args},
        )
    if inject_into_content:
        args[CMW_ASSISTANT_LAST_MESSAGE_SCHEMA_KEY] = assistant_content
    _debug_log_io(
        "structured.formatter.response",
        {
            "session_id": session_id,
            "args": args,
            "inject_into_content": inject_into_content,
        },
    )
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
    debug_event_prefix: str | None = None,
) -> AsyncGenerator[str, None]:
    first = True
    chunk_index = 0
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
        sse_chunk = _sse(
            _chunk_payload(
                request_model=request_model,
                delta=delta,
                finish_reason=None,
            )
        )
        if debug_event_prefix:
            _debug_log_io(
                f"{debug_event_prefix}.stream.chunk",
                {
                    "session_id": session_id,
                    "model": request_model,
                    "chunk_index": chunk_index,
                    "chunk": sse_chunk,
                },
            )
        yield sse_chunk
        chunk_index += 1
        if event_type == "error":
            break

    usage = _completion_usage(agent)
    final_sse = _sse(
        _chunk_payload(
            request_model=request_model,
            delta={},
            finish_reason=KNOWN_FINISH_REASON,
            usage=usage,
        )
    )
    if debug_event_prefix:
        _debug_log_io(
            f"{debug_event_prefix}.stream.final",
            {
                "session_id": session_id,
                "model": request_model,
                "chunk_index": chunk_index,
                "chunk": final_sse,
                "usage": usage,
            },
        )
    yield final_sse
    done_chunk = "data: [DONE]\n\n"
    if debug_event_prefix:
        _debug_log_io(
            f"{debug_event_prefix}.stream.done",
            {"session_id": session_id, "model": request_model, "chunk": done_chunk},
        )
    yield done_chunk


def register_agent_completions_on_fastapi(fastapi_app: FastAPI, app: Any) -> None:
    """Register `POST /api/v1/chat/completions` on a FastAPI app."""

    async def agent_completions(request: Request) -> JSONResponse | StreamingResponse:
        provided_api_key = _extract_api_key(request)
        if not provided_api_key:
            response = _error_response("Missing bearer token", status_code=401)
            _debug_log_io(
                "fastapi.outgoing.error",
                {"response": _json_response_content(response)},
            )
            return response

        payload = await request.json()
        if not isinstance(payload, dict):
            response = _error_response("request body must be a JSON object")
            _debug_log_io(
                "fastapi.outgoing.error",
                {"response": _json_response_content(response)},
            )
            return response
        _debug_log_io(
            "fastapi.incoming.request",
            {
                "authorization": _mask_secret(provided_api_key),
                "payload": payload,
            },
        )
        structured_spec = _parse_response_format(payload)
        if isinstance(structured_spec, JSONResponse):
            _debug_log_io(
                "fastapi.outgoing.error",
                {"response": _json_response_content(structured_spec)},
            )
            return structured_spec
        if structured_spec is not None and payload.get("stream") is True:
            response = _error_response(
                "stream=true is not supported with response_format",
                status_code=400,
            )
            _debug_log_io(
                "fastapi.outgoing.error",
                {"response": _json_response_content(response)},
            )
            return response

        prepared = _prepare_request(app, payload, provider_api_key=provided_api_key)
        if isinstance(prepared, JSONResponse):
            _debug_log_io(
                "fastapi.outgoing.error",
                {"response": _json_response_content(prepared)},
            )
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
                    debug_event_prefix="fastapi.outgoing",
                ),
                media_type="text/event-stream",
            )

        content = await _collect_agent_response(agent, question, session_id)
        usage_pre: dict[str, Any] | None = None
        if structured_spec is not None:
            _getter = getattr(agent, "get_last_api_tokens", None)
            usage_pre = usage_from_token_count(_getter() if callable(_getter) else None)
            structured_result = await _format_structured_output(
                agent=agent,
                session_id=session_id,
                user_question=question,
                assistant_content=content,
                spec=structured_spec,
            )
            if isinstance(structured_result, JSONResponse):
                _debug_log_io(
                    "fastapi.outgoing.error",
                    {"response": _json_response_content(structured_result)},
                )
                return structured_result
            content = json.dumps(structured_result, ensure_ascii=False)
        turn_usage = _completion_usage(
            agent, pre_structured=usage_pre if structured_spec is not None else None
        )
        response_payload = build_chat_completion_response(
            request_model=resolved.request_model,
            assistant_content=content,
            finish_reason=KNOWN_FINISH_REASON,
            usage=turn_usage,
        )
        _debug_log_io("fastapi.outgoing.response", {"response": response_payload})
        return JSONResponse(content=response_payload)

    route_paths = {getattr(route, "path", None) for route in fastapi_app.routes}
    if AGENT_COMPLETIONS_PATH not in route_paths:
        fastapi_app.add_api_route(
            AGENT_COMPLETIONS_PATH,
            agent_completions,
            methods=["POST"],
            response_model=None,
            openapi_extra={
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["messages"],
                                "properties": {
                                    "model": {
                                        "type": "string",
                                        "description": (
                                            "Provider/model slug,"
                                            " e.g. openai/gpt-4o"
                                        ),
                                        "example": "openai/gpt-4o",
                                    },
                                    "messages": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["role", "content"],
                                            "properties": {
                                                "role": {
                                                    "type": "string",
                                                    "enum": [
                                                        "system",
                                                        "user",
                                                        "assistant",
                                                    ],
                                                },
                                                "content": {"type": "string"},
                                            },
                                        },
                                        "example": [
                                            {
                                                "role": "user",
                                                "content": "Hello!",
                                            }
                                        ],
                                    },
                                    "stream": {
                                        "type": "boolean",
                                        "default": False,
                                        "description": "Enable SSE streaming",
                                    },
                                    "response_format": {
                                        "type": "object",
                                        "nullable": True,
                                        "description": "Structured output JSON schema",
                                    },
                                },
                            },
                            "example": {
                                "model": "openai/gpt-4o",
                                "messages": [
                                    {"role": "user", "content": "Hello!"}
                                ],
                                "stream": False,
                            },
                        }
                    }
                }
            },
        )


def register_agent_completions_route(demo: Any, app: Any) -> None:
    """Register `POST /api/v1/chat/completions` on Gradio's underlying FastAPI app."""

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
        response = json.loads(
            _error_response("request body must be a JSON object").body.decode("utf-8")
        )
        _debug_log_io("gradio.outgoing.error", {"response": response})
        return response
    _debug_log_io("gradio.incoming.request", {"payload": payload})

    structured_spec = _parse_response_format(payload)
    if isinstance(structured_spec, JSONResponse):
        response = json.loads(structured_spec.body.decode("utf-8"))
        _debug_log_io("gradio.outgoing.error", {"response": response})
        return response
    if structured_spec is not None and payload.get("stream") is True:
        response = _error_response(
            "stream=true is not supported with response_format",
            status_code=400,
        )
        out = json.loads(response.body.decode("utf-8"))
        _debug_log_io("gradio.outgoing.error", {"response": out})
        return out

    prepared = _prepare_request(app, payload)
    if isinstance(prepared, JSONResponse):
        response = json.loads(prepared.body.decode("utf-8"))
        _debug_log_io("gradio.outgoing.error", {"response": response})
        return response

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
                debug_event_prefix="gradio.outgoing",
            )
        ]
        out: dict[str, Any] = {
            "object": "chat.completion.stream",
            "model": resolved.request_model,
            "chunks": chunks,
        }
        stream_usage = _completion_usage(agent)
        if stream_usage is not None:
            out["usage"] = stream_usage
        _debug_log_io("gradio.outgoing.response", {"response": out})
        return out

    content = await _collect_agent_response(agent, question, session_id)
    usage_pre: dict[str, Any] | None = None
    if structured_spec is not None:
        _getter = getattr(agent, "get_last_api_tokens", None)
        usage_pre = usage_from_token_count(_getter() if callable(_getter) else None)
        structured_result = await _format_structured_output(
            agent=agent,
            session_id=session_id,
            user_question=question,
            assistant_content=content,
            spec=structured_spec,
        )
        if isinstance(structured_result, JSONResponse):
            response = json.loads(structured_result.body.decode("utf-8"))
            _debug_log_io("gradio.outgoing.error", {"response": response})
            return response
        content = json.dumps(structured_result, ensure_ascii=False)
    turn_usage = _completion_usage(
        agent, pre_structured=usage_pre if structured_spec is not None else None
    )
    response_payload = build_chat_completion_response(
        request_model=resolved.request_model,
        assistant_content=content,
        finish_reason=KNOWN_FINISH_REASON,
        usage=turn_usage,
    )
    _debug_log_io("gradio.outgoing.response", {"response": response_payload})
    return response_payload
