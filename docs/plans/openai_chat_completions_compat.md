# OpenAI Chat Completions Compatibility Plan

## Goal

Expose the existing CMW Platform Agent chat through a minimal OpenAI-compatible
Chat Completions API while preserving current Gradio UI behavior.

## Scope

- Add a basic `POST /v1/chat/completions` route.
- Support request fields: `model`, `messages`, `stream`.
- Accept non-standard fields in `extra_body`: `cmw_base_url`, `cmw_login`,
  `cmw_password`, and `session_id`.
- Resolve provider/model slugs such as `polza/z-ai/glm-5.1` and
  `openrouter/z-ai/glm-5.1`.
- Return OpenAI-shaped non-streaming JSON responses and streaming SSE chunks.
- Ignore unsupported OpenAI fields for this first version.
- Require `Authorization: Bearer <token>` and use that token as runtime
  provider API key for the selected provider.

## Research Notes

- OpenAI Chat Completions accepts a list of messages and returns either a full
  `chat.completion` object or streamed `chat.completion.chunk` events.
- Gradio custom API endpoints can be added through Gradio's API helpers, and
  the Blocks app runs on FastAPI under the hood, allowing custom routes when
  exact OpenAI path compatibility is needed.
- The existing app already has `_api_ask` and `_api_ask_stream` using
  `user_agent.stream_message(...)`, `set_session_config(...)`, and per-session
  isolation.

## Design

1. Add `agent_ng/openai_compat.py` as a small adapter module.
2. Keep validation and response formatting separate from Gradio UI code.
3. Add a model resolver that maps:
   - `<provider>/<model_slug>` to `(provider, model_slug)`
   - `<model_slug>` to the configured default provider.
4. Collect only `content` and `error` stream events for the OpenAI API surface.
5. Use the existing session manager to apply CMW credentials, LLM selection,
   and caller-provided `sessionId` for conversation continuity.
6. Register the exact route from `NextGenApp.create_interface()` after the
   Blocks app exists.

## TDD Checkpoints

1. RED: test model resolution for provider-prefixed and default-provider slugs.
2. RED: test non-streaming response shape.
3. RED: test streaming SSE chunk shape and `[DONE]`.
4. GREEN: implement adapter functions with minimal code.
5. GREEN: wire the route to the Gradio/FastAPI app.
6. REFACTOR: remove duplication while keeping tests green.

## Verification Commands

```bash
python -m pytest agent_ng/_tests/test_openai_compat.py
ruff check agent_ng/openai_compat.py agent_ng/app_ng_modular.py agent_ng/_tests/test_openai_compat.py
python lint.py
```
