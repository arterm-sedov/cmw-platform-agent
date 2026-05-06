# OpenAI Chat Completions Compatibility Plan

## Goal

Expose the existing CMW Platform Agent chat through a minimal OpenAI-**shaped**
HTTP API (`POST /v1/agent_completions`) while preserving current Gradio UI
behavior.

## Scope

- Add `POST /v1/agent_completions` (OpenAI-shaped contract, custom path).
- Support request fields: `model`, `messages`, `stream`.
- Carry session and CMW/runtime fields only in a **`messages` entry with
  `role: "system"`** whose `content` is a JSON **object** (or a string that
  parses to one). Supported keys include `cmw_base_url`, `cmw_login`,
  `cmw_password`, and `session_id`. Later system carrier messages override
  earlier ones. Request top-level `extra_body` is **ignored**.
- Resolve provider/model slugs such as `polza/z-ai/glm-5.1` and
  `openrouter/z-ai/glm-5.1`.
- Return OpenAI-shaped non-streaming JSON responses and streaming SSE chunks.
- Ignore unsupported OpenAI fields for this first version.
- Require `Authorization: Bearer <token>` and use that token as runtime
  provider API key for the selected provider.
- Support optional `response_format` (`json_schema`) for non-stream calls.
- When `response_format` is present, run a final formatter step: bind a single
  synthetic tool (prompt instructs the model to call it; no forced `tool_choice`
  for downstream compatibility) and return validated JSON in assistant content.
- Add safe repair/coercion for structured output before validation (string trim,
  conservative primitive coercion), while preserving schema guardrails.
- When structured output is enabled: if the client **does not** declare root
  property `cmw_assistant_last_message` (`type: string`) in
  `response_format.json_schema.schema`, add proprietary sibling
  `message.cmw_assistant_last_message`. If the client **does** declare that
  string slot, omit the sibling and **inject** the raw assistant text into
  `message.content` JSON under that key (the slot is stripped from the
  formatter tool schema so the model does not fill it).

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
   and `session_id` from the system JSON carrier for conversation continuity.
6. Register the exact route from `NextGenApp.create_interface()` after the
   Blocks app exists.
7. Add structured-output parser/validator for `response_format` object or JSON
   string payload.
8. Add synthetic final formatter with one bound tool (`strict` schema when
   supported); prompt requests the tool call; slice session history for context.
9. Add a schema-aware normalization pass:
   - Trim strings.
   - Coerce obvious primitive forms (`"1"` -> `1`, `"true"` -> `true`,
     `"3.14"` -> `3.14`) when schema expects those types.
   - Treat empty string as null only when schema explicitly expects `null`.
   - Never auto-add required fields and never bypass `additionalProperties`.

## TDD Checkpoints

1. RED: test model resolution for provider-prefixed and default-provider slugs.
2. RED: test non-streaming response shape.
3. RED: test streaming SSE chunk shape and `[DONE]`.
4. GREEN: implement adapter functions with minimal code.
5. GREEN: wire the route to the Gradio/FastAPI app.
6. REFACTOR: remove duplication while keeping tests green.
7. RED: add failing tests for structured-output coercion/repair rules.
8. GREEN: implement minimal coercion helper and keep strict validation.
9. REFACTOR: keep helper DRY and focused with no behavior drift.

## Verification Commands

```bash
python -m pytest agent_ng/_tests/test_openai_compat.py
ruff check agent_ng/openai_compat.py agent_ng/app_ng_modular.py agent_ng/_tests/test_openai_compat.py
python lint.py
```
