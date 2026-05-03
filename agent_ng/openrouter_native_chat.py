# // pragma: allowlist secret
# pragma: allowlist secret
"""OpenRouter chat model via OpenAI Python SDK.

Captures full ``usage`` metadata including ``cost``. Alternative to ``ChatOpenAI``
with OpenRouter ``base_url``: avoids LangChain dropping OpenRouter-specific usage
fields during streaming. Aligns with cmw-rag ``OpenRouterNativeFullChatModel``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

DEFAULT_OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "https://github.com/arterm-sedov/cmw-platform-agent",
    "X-Title": "CMW Platform Agent",
}


def usage_dict_from_sdk(usage: Any) -> dict[str, Any]:
    """Serialize SDK ``usage`` object or dict to plain dict (preserves ``cost``)."""
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass
    if isinstance(usage, dict):
        return usage
    data: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "completion_tokens_details",
        "prompt_tokens_details",
        "cost_details",
        "input_tokens",
        "output_tokens",
    ):
        if hasattr(usage, key):
            val = getattr(usage, key)
            data[key] = val.model_dump() if hasattr(val, "model_dump") else val
    return data or {"raw": repr(usage)}


def _create_usage_metadata(token_usage: dict[str, Any]) -> UsageMetadata:
    input_tokens = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
    output_tokens = int(
        token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
    )
    total_tokens = int(token_usage.get("total_tokens") or input_tokens + output_tokens)
    out: UsageMetadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    prompt_details = (
        token_usage.get("prompt_tokens_details") or token_usage.get("input_tokens_details") or {}
    )
    comp_details = (
        token_usage.get("completion_tokens_details")
        or token_usage.get("output_tokens_details")
        or {}
    )
    if isinstance(prompt_details, dict):
        cache_read = prompt_details.get("cached_tokens")
        cache_creation = prompt_details.get("cache_write_tokens")
        if cache_read is not None or cache_creation is not None:
            details_kwargs: dict[str, int] = {}
            if cache_read is not None:
                details_kwargs["cache_read"] = int(cache_read)
            if cache_creation is not None:
                details_kwargs["cache_creation"] = int(cache_creation)
            out["input_token_details"] = InputTokenDetails(**details_kwargs)
    if isinstance(comp_details, dict) and comp_details.get("reasoning_tokens") is not None:
        out["output_token_details"] = OutputTokenDetails(
            reasoning=int(comp_details["reasoning_tokens"])
        )
    return out


def _message_to_dict(msg: Any) -> dict[str, Any]:
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if isinstance(msg, dict):
        return msg
    return {"content": getattr(msg, "content", "") or ""}


def _convert_response_message_to_ai(
    msg_dict: dict[str, Any],
    usage_dict: dict[str, Any],
    response_id: str | None,
) -> AIMessage:
    content = msg_dict.get("content") or ""
    tool_calls_raw = msg_dict.get("tool_calls") or []
    tool_calls: list[dict[str, Any]] = []
    invalid_tool_calls: list[Any] = []
    for tc in tool_calls_raw:
        try:
            parsed = parse_tool_call(tc, return_id=True)
            tool_calls.append(parsed)
        except Exception as exc:
            invalid_tool_calls.append(make_invalid_tool_call(tc, str(exc)))

    additional_kwargs: dict[str, Any] = {}
    if reasoning := msg_dict.get("reasoning"):
        additional_kwargs["reasoning_content"] = reasoning
    if reasoning_details := msg_dict.get("reasoning_details"):
        additional_kwargs["reasoning_details"] = reasoning_details

    usage_metadata = _create_usage_metadata(usage_dict) if usage_dict else None
    response_metadata: dict[str, Any] = {"model_provider": "openrouter"}
    if response_id:
        response_metadata["id"] = response_id
    if usage_dict:
        response_metadata["token_usage"] = usage_dict
        if "cost" in usage_dict:
            response_metadata["cost"] = usage_dict["cost"]
        elif "cost_rub" in usage_dict:
            # Polza reports cost in RUB; convert to USD using env rate so that
            # token_counter.py sees a non-zero cost figure.
            from agent_ng.openrouter_usage_accounting import _get_polza_rate
            rub = float(usage_dict["cost_rub"] or 0.0)
            response_metadata["cost"] = rub / _get_polza_rate()
            response_metadata["cost_rub"] = rub
        if usage_dict.get("cost_details"):
            response_metadata["cost_details"] = usage_dict["cost_details"]

    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
        additional_kwargs=additional_kwargs,
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
    )


def _convert_delta_to_chunk(
    delta: dict[str, Any],
    chunk_usage: dict[str, Any] | None,
    default_class: type[BaseMessageChunk],
) -> BaseMessageChunk:
    from langchain_core.messages.tool import tool_call_chunk

    content = delta.get("content") or ""
    tool_call_chunks: list[Any] = []
    if raw_tc := delta.get("tool_calls"):
        for rtc in raw_tc or []:
            try:
                tool_call_chunks.append(
                    tool_call_chunk(
                        name=rtc.get("function", {}).get("name"),
                        args=rtc.get("function", {}).get("arguments"),
                        id=rtc.get("id"),
                        index=rtc.get("index", 0),
                    )
                )
            except (KeyError, TypeError):
                pass

    additional_kwargs: dict[str, Any] = {}
    if reasoning := delta.get("reasoning"):
        additional_kwargs["reasoning_content"] = reasoning
    if reasoning_details := delta.get("reasoning_details"):
        additional_kwargs["reasoning_details"] = reasoning_details

    usage_metadata = _create_usage_metadata(chunk_usage) if chunk_usage else None
    response_metadata: dict[str, Any] = {"model_provider": "openrouter"}
    if chunk_usage:
        if "cost" in chunk_usage:
            response_metadata["cost"] = chunk_usage["cost"]
        elif "cost_rub" in chunk_usage:
            from agent_ng.openrouter_usage_accounting import _get_polza_rate
            rub = float(chunk_usage["cost_rub"] or 0.0)
            response_metadata["cost"] = rub / _get_polza_rate()
            response_metadata["cost_rub"] = rub

    return AIMessageChunk(
        content=content,
        tool_call_chunks=tool_call_chunks,
        additional_kwargs=additional_kwargs,
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
    )


class OpenRouterNativeChatModel(BaseChatModel):
    """OpenRouter via ``openai.OpenAI`` — preserves ``usage.cost`` for invoke and stream."""

    # ``Any`` allows tests to inject ``MagicMock`` clients; runtime uses ``OpenAI`` / ``AsyncOpenAI``.
    client: Any = Field(default_factory=lambda: OpenAI())
    async_client: Any = None
    model: str = Field(alias="model_name")
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    extra_body: dict[str, Any] = Field(default_factory=dict)
    default_headers: dict[str, str] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "openrouter_native_sdk"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url}

    def _get_request_body(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        oai_messages = convert_to_openai_messages(messages, text_format="block", include_id=False)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")},
        }
        if stop:
            body["stop"] = stop
        if self.extra_body:
            body["extra_body"] = {**self.extra_body, **body.get("extra_body", {})}
        if tools := kwargs.get("tools"):
            body["tools"] = tools
        if tool_choice := kwargs.get("tool_choice"):
            body["tool_choice"] = tool_choice
        return body

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        body = self._get_request_body(messages, stop=stop, **kwargs)
        resp = self.client.chat.completions.create(**body)

        usage_dict = usage_dict_from_sdk(getattr(resp, "usage", None))
        msg_obj = resp.choices[0].message
        msg_dict = _message_to_dict(msg_obj)
        response_id = getattr(resp, "id", None)

        ai_msg = _convert_response_message_to_ai(msg_dict, usage_dict, response_id)
        gen_info: dict[str, Any] = {
            "finish_reason": resp.choices[0].finish_reason,
            "model_name": getattr(resp, "model", self.model),
            "model_provider": "openrouter",
            "token_usage": usage_dict,
        }
        if response_id:
            gen_info["id"] = response_id

        llm_output: dict[str, Any] = {
            "model_name": getattr(resp, "model", self.model),
            "token_usage": usage_dict,
            "model_provider": "openrouter",
        }
        if response_id:
            llm_output["id"] = response_id

        return ChatResult(
            generations=[ChatGeneration(message=ai_msg, generation_info=gen_info)],
            llm_output=llm_output,
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        body = self._get_request_body(messages, stop=stop, **kwargs)
        aclient = self.async_client or AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key, default_headers=self.default_headers
        )
        resp = await aclient.chat.completions.create(**body)

        usage_dict = usage_dict_from_sdk(getattr(resp, "usage", None))
        msg_obj = resp.choices[0].message
        msg_dict = _message_to_dict(msg_obj)
        response_id = getattr(resp, "id", None)

        ai_msg = _convert_response_message_to_ai(msg_dict, usage_dict, response_id)
        gen_info: dict[str, Any] = {
            "finish_reason": resp.choices[0].finish_reason,
            "model_name": getattr(resp, "model", self.model),
            "model_provider": "openrouter",
            "token_usage": usage_dict,
        }
        if response_id:
            gen_info["id"] = response_id

        llm_output: dict[str, Any] = {
            "model_name": getattr(resp, "model", self.model),
            "token_usage": usage_dict,
            "model_provider": "openrouter",
        }
        if response_id:
            llm_output["id"] = response_id

        return ChatResult(
            generations=[ChatGeneration(message=ai_msg, generation_info=gen_info)],
            llm_output=llm_output,
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        body = self._get_request_body(messages, stop=stop, **kwargs)
        body["stream"] = True
        body["extra_body"] = {
            **body.get("extra_body", {}),
            "stream_options": {"include_usage": True},
        }

        stream = self.client.chat.completions.create(**body)
        default_class: type[BaseMessageChunk] = AIMessageChunk

        for chunk in stream:
            chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            usage_dict = usage_dict_from_sdk(chunk_dict.get("usage"))
            choices = chunk_dict.get("choices") or []

            if not choices:
                if usage_dict:
                    rm: dict[str, Any] = {"model_provider": "openrouter"}
                    if "cost" in usage_dict:
                        rm["cost"] = usage_dict["cost"]
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            usage_metadata=_create_usage_metadata(usage_dict),
                            response_metadata=rm,
                        ),
                        generation_info={"token_usage": usage_dict},
                    )
                continue

            delta = choices[0].get("delta") or {}
            msg_chunk = _convert_delta_to_chunk(
                delta, usage_dict if usage_dict else None, default_class
            )
            gen_info: dict[str, Any] = {}
            if fr := choices[0].get("finish_reason"):
                gen_info["finish_reason"] = fr
                gen_info["model_name"] = chunk_dict.get("model", self.model)
                if usage_dict:
                    gen_info["token_usage"] = usage_dict
            if chunk_dict.get("id"):
                gen_info["id"] = chunk_dict["id"]

            yield ChatGenerationChunk(message=msg_chunk, generation_info=gen_info or None)
            if msg_chunk.content:
                default_class = type(msg_chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        body = self._get_request_body(messages, stop=stop, **kwargs)
        body["stream"] = True
        body["extra_body"] = {
            **body.get("extra_body", {}),
            "stream_options": {"include_usage": True},
        }

        aclient = self.async_client or AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key, default_headers=self.default_headers
        )
        stream = await aclient.chat.completions.create(**body)
        default_class: type[BaseMessageChunk] = AIMessageChunk

        async for chunk in stream:
            chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            usage_dict = usage_dict_from_sdk(chunk_dict.get("usage"))
            choices = chunk_dict.get("choices") or []

            if not choices:
                if usage_dict:
                    rm_async: dict[str, Any] = {"model_provider": "openrouter"}
                    if "cost" in usage_dict:
                        rm_async["cost"] = usage_dict["cost"]
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            usage_metadata=_create_usage_metadata(usage_dict),
                            response_metadata=rm_async,
                        ),
                        generation_info={"token_usage": usage_dict},
                    )
                continue

            delta = choices[0].get("delta") or {}
            msg_chunk = _convert_delta_to_chunk(
                delta, usage_dict if usage_dict else None, default_class
            )
            gen_info: dict[str, Any] = {}
            if fr := choices[0].get("finish_reason"):
                gen_info["finish_reason"] = fr
                gen_info["model_name"] = chunk_dict.get("model", self.model)
                if usage_dict:
                    gen_info["token_usage"] = usage_dict
            if chunk_dict.get("id"):
                gen_info["id"] = chunk_dict["id"]

            yield ChatGenerationChunk(message=msg_chunk, generation_info=gen_info or None)
            if msg_chunk.content:
                default_class = type(msg_chunk)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | type | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        formatted = [convert_to_openai_tool(t, strict=strict) for t in tools]
        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            if isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError("tool_choice can only be True when there is one tool.")
                tool_choice = {
                    "type": "function",
                    "function": {"name": formatted[0]["function"]["name"]},
                }
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted, **kwargs)


def create_openrouter_native_chat_model(
    *,
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict[str, Any] | None = None,
    default_headers: dict[str, str] | None = None,
    callbacks: list | None = None,
) -> OpenRouterNativeChatModel:
    """Factory: wired OpenAI clients against OpenRouter base URL."""
    hdrs = {**DEFAULT_OPENROUTER_HEADERS, **(default_headers or {})}
    return OpenRouterNativeChatModel(
        client=OpenAI(base_url=base_url, api_key=api_key, default_headers=hdrs),
        async_client=AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers=hdrs),
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=dict(extra_body or {}),
        default_headers=hdrs,
        callbacks=callbacks or [],
    )
